"""1D Transformer backbone for time series (vanilla PyTorch + StochasticDepth)."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision.ops import StochasticDepth

from .base import TimeSeriesBackbone


@dataclass
class TransformerConfig:
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 8
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.1
    # GPT-2 / Fixup-style depth-scaled init: divide std of each block's residual
    # out-projections (attn.out_proj, mlp[2]) by sqrt(2 * num_hidden_layers).
    # Default off to preserve existing encoder behavior; the decoder script
    # opts in by default.
    rescale_residual_init: bool = False
    # ── HOW A TOKEN LEARNS WHERE IT IS ────────────────────────────────────
    #
    # "learned" is one randomly-initialised vector per slot, trained from
    # scratch on one month of data. "sinusoidal" is the fixed Transformer
    # table: no parameters, no per-slot sample requirement, and every slot is
    # already at a known distance from every other before a single step.
    pos_embed: str = "learned"
    # WHERE THE CLS TOKEN SITS, for the position embedding.
    #
    # A prepended CLS gets slot 0, the FIRST position -- the opposite end of
    # the view from the one a readout usually wants. Nothing else tells it
    # where "now" is, so this is the only knob that does.
    #
    #   "own"   slot 0's own vector (what every checkpoint before this used)
    #   "last"  the LAST PATCH's position embedding, so the readout is
    #           positioned at the end of the view
    #   "none"  no position embedding, like the info token -- the readout has
    #           no time, so it is given none
    #
    # "own" and "none" are the SAME MODEL under a learned table: a learned CLS
    # token plus a learned CLS position vector is one learned vector. The
    # distinction only bites once the table is fixed (sinusoidal).
    cls_pos: str = "own"
    # INIT SCALE of the learned position table (ignored when sinusoidal).
    # 0.02 is the ViT default every checkpoint here was trained at. It sets how
    # much position information the model starts with relative to the patch
    # embedding, which is the knob that decides whether position is used at all
    # early in a ONE-MONTH run.
    pos_init_std: float = 0.02
    # FREEZE THE PATCH PROJECTION TO AN IDENTITY. Requires hidden_size ==
    # n_features * patch_size, i.e. the token IS the patch, flattened.
    #
    # A "square" projection (288 -> 288 at patch 32) is still a LEARNED square
    # matrix and tests nothing about whether the projection earns its place --
    # it is the same map as 288 -> 384 with a different output width. This
    # makes the map the identity, so a token carries the raw standardized
    # patch values and the transformer has to do all the work.
    #
    # Implemented as a FROZEN Conv1d holding the identity permutation rather
    # than as a reshape, for one reason: eval/checkpoints.py recovers the input
    # width from patch_embed.proj.weight, and a reshape would delete the only
    # record of it. The extra matmul is 5.3 MMACs against ~830, i.e. nothing.
    patch_embed_identity: bool = False



class PatchEmbedding1D(nn.Module):
    """Convert time series into patch embeddings using convolution.

    Uses Conv1d where kernel_size=stride=patch_size, following timm's PatchEmbed pattern.
    See: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py

    Args:
        n_features: Number of input features (channels).
        patch_size: Size of each patch (number of timesteps).
        d_model: Embedding dimension.
        norm_layer: Optional normalization layer.
        bias: Whether to use bias in conv layer.
    """

    def __init__(
        self,
        n_features: int,
        patch_size: int,
        d_model: int,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        identity: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        self.d_model = d_model
        self.identity = bool(identity)

        # Convolutional projection: kernel and stride are both patch_size
        self.proj = nn.Conv1d(
            in_channels=n_features,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        if self.identity:
            # Conv1d output o for patch p is sum_{c,k} W[o,c,k] x[c, p*P+k], so
            # the identity is W[c*P + k, c, k] = 1 -- the flattening that puts
            # a patch's channels in channel-major order. Frozen, and the bias
            # zeroed, so the token is EXACTLY the raw patch.
            if d_model != n_features * patch_size:
                raise ValueError(
                    f"patch_embed_identity needs hidden_size == n_features * "
                    f"patch_size ({n_features} * {patch_size} = "
                    f"{n_features * patch_size}), got {d_model}")
            with torch.no_grad():
                self.proj.weight.zero_()
                for c in range(n_features):
                    for k in range(patch_size):
                        self.proj.weight[c * patch_size + k, c, k] = 1.0
                if self.proj.bias is not None:
                    self.proj.bias.zero_()
            self.proj.weight.requires_grad_(False)
            if self.proj.bias is not None:
                self.proj.bias.requires_grad_(False)
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch embeddings.

        Args:
            x: Input of shape (batch, n_features, length).

        Returns:
            Patch embeddings of shape (batch, n_patches, d_model).
        """
        # Pad to make divisible by patch_size
        _, _, length = x.shape
        pad_len = (self.patch_size - length % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Conv1d: (batch, n_features, length) -> (batch, d_model, n_patches)
        x = self.proj(x)

        # Transpose to (batch, n_patches, d_model) for transformer
        x = x.transpose(1, 2)

        x = self.norm(x)
        return x


def _sinusoidal_table(n_pos: int, d_model: int) -> torch.Tensor:
    """The standard fixed sin/cos position table, shape (1, n_pos, d_model).

    Nothing here is learned, which is the point: a randomly-initialised table
    has to see every slot enough times to place it, and a one-month supervised
    run does not obviously provide that for 257 slots.
    """
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, d_model, 2, dtype=torch.float32)
    denom = torch.pow(10000.0, idx / d_model)
    pe = torch.zeros(1, n_pos, d_model)
    pe[0, :, 0::2] = torch.sin(pos / denom)
    pe[0, :, 1::2] = torch.cos(pos / denom)
    return pe


def _rope_cos_sin(pos: torch.Tensor, head_dim: int, device, dtype):
    """(1, 1, L, head_dim) cos/sin for rotary embeddings at positions ``pos``.

    GPT-NeoX layout: the angle for dimension pair i is ``pos * 10000^(-2i/d)``,
    tiled twice so it pairs with ``_rotate_half``.
    """
    half = head_dim // 2
    theta = torch.pow(10000.0, -torch.arange(half, device=device,
                                             dtype=torch.float32) / half)
    ang = pos.to(torch.float32).unsqueeze(1) * theta.unsqueeze(0)   # (L, half)
    emb = torch.cat([ang, ang], dim=-1)                             # (L, d)
    return (emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class RoPESelfAttention(nn.Module):
    """Self-attention with ROTARY position, and no absolute table at all.

    Why this exists here. Absolute position has been the weak part of this
    backbone: a 257-slot learned table cannot be estimated from one month, and
    swapping it for a fixed sinusoidal one was worth +0.05 vol / +0.046 spread
    to a mean-pooled LeJEPA. RoPE removes absolute position rather than
    fixing its estimation -- the attention logit becomes a function of the
    SEPARATION between two patches. Each head learns its own dependence on
    that separation, so one can settle on local attention, another on lagged,
    another on global.

    TOKENS WITHOUT A TIME ARE NOT ROTATED. The info token, the state token and
    the CLS get rotation 0 -- a fictional separation is worse than none.

    A CAVEAT FOR ANY FUTURE CLS ARM. RoPE cannot express "the last patch"
    absolutely, only separations. A PREPENDED CLS therefore sits ~256 steps
    from the recent patches, the range where rotations have decorrelated and
    RoPE is weakest; it belongs after the patches, not before. The arms this
    was built for (mean and last pooling) have no CLS and are unaffected.

    Projection names mirror nn.MultiheadAttention's so ``_init_weights`` and
    ``_rescale_residual_branches`` reach ``out_proj`` without a special case.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"{hidden_size} not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, key_padding_mask=None, attn_mask=None, rope=None):
        B, L, H = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if rope is not None:
            cos, sin = rope
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        # One float bias: SDPA takes a single mask, so a bool causal mask and
        # a float padding mask have to be folded into the same tensor.
        #
        # A BOOL MASK MUST BECOME -inf, NOT 1.0. `.to(dtype)` on a bool tensor
        # gives 1.0 where True, and an additive +1.0 does not block attention
        # -- it mildly ENCOURAGES it. That made causal masking a no-op on this
        # path: with recency the caller always folded the causal mask into a
        # float bias before it arrived here, so the bare bool only started
        # reaching this line when the prior was retired. Verified by
        # perturbing the final patch and reading an early token: 2.1e-03 of
        # drift before, exactly 0 after.
        bias = None
        if attn_mask is not None:
            m = attn_mask if attn_mask.dim() == 4 else attn_mask.view(1, 1, L, L)
            bias = (torch.zeros_like(m, dtype=q.dtype).masked_fill(m, float("-inf"))
                    if m.dtype == torch.bool else m.to(q.dtype))
        if key_padding_mask is not None:
            kpm = key_padding_mask
            if kpm.dtype == torch.bool:
                kpm = torch.zeros_like(kpm, dtype=q.dtype).masked_fill(
                    kpm, float("-inf"))
            kpm = kpm.view(B, 1, 1, L).to(q.dtype)
            bias = kpm if bias is None else bias + kpm
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(o.transpose(1, 2).reshape(B, L, H))


class ViTBlock(nn.Module):
    """Pre-norm transformer block with StochasticDepth (DropPath) regularization.

    Architecture: norm -> attn -> drop_path -> residual, norm -> mlp -> drop_path -> residual.
    Uses nn.MultiheadAttention which leverages SDPA automatically in PyTorch 2.0+.

    Args:
        hidden_size: Hidden dimension.
        num_attention_heads: Number of attention heads.
        intermediate_size: MLP intermediate dimension.
        layer_norm_eps: Layer norm epsilon.
        drop_path_rate: Drop path probability for this block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float = 1e-12,
        drop_path_rate: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.use_rope = use_rope
        self.attn = (
            RoPESelfAttention(hidden_size, num_attention_heads) if use_rope
            else nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                batch_first=True,
                dropout=0.0,
            )
        )
        self.drop_path1 = StochasticDepth(p=drop_path_rate, mode="row")

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.drop_path2 = StochasticDepth(p=drop_path_rate, mode="row")

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        rope: tuple | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.use_rope:
            attn_out = self.attn(x_norm, key_padding_mask=key_padding_mask,
                                 attn_mask=attn_mask, rope=rope)
        else:
            attn_out, _ = self.attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )
        x = x + self.drop_path1(attn_out)

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


def _rescale_residual_branches(blocks: nn.ModuleList) -> None:
    """In-place GPT-2 / Fixup-style depth-scaled init for ViTBlock residuals.

    Scales the std of each block's residual out-projections — ``attn.out_proj``
    and the second MLP linear (``mlp[2]``) — by ``1 / sqrt(2 * L)`` where ``L``
    is the number of blocks. Counteracts the activation-variance growth from
    summing 2L residual branches.
    """
    n_layers = len(blocks)
    if n_layers == 0:
        return
    scale = 1.0 / math.sqrt(2.0 * n_layers)
    for block in blocks:
        block.attn.out_proj.weight.data.mul_(scale)
        block.mlp[2].weight.data.mul_(scale)


class TransformerBackbone(TimeSeriesBackbone):
    """1D Transformer backbone for time series using vanilla PyTorch blocks.

    Processes time series by:
    1. Splitting into patches via Conv1d
    2. Prepending CLS token (if cls pooling)
    3. Adding learnable position embeddings
    4. Processing through transformer blocks with StochasticDepth
    5. Pooling to get final embedding

    Args:
        config: TransformerConfig specifying transformer architecture.
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
        pool: Pooling strategy ('cls', 'mean', 'max', 'last').
        max_seq_len: Maximum number of raw timesteps (for position embedding pre-allocation).
        gradient_checkpointing: If True, use gradient checkpointing on transformer
            blocks to reduce memory at the cost of extra compute.
    """

    def __init__(
        self,
        config: TransformerConfig,
        n_features: int,
        d_embedding: int = 512,
        pool: str | None = None,
        max_seq_len: int = 2048,
        gradient_checkpointing: bool = False,
        causal: bool = False,
        state_token: bool = False,
        diff_channels: bool = False,
        n_info_channels: int = 0,
    ):
        super().__init__(n_features, d_embedding)
        self.config = config
        # None = NOT SET, which is how the hydra schema spells "let the mode
        # or this default decide" (see ModeBackboneOverrides). A directly
        # constructed backbone -- tests, eval loaders -- still gets "cls".
        pool = "cls" if pool is None else pool
        self.pool = pool
        self.causal = causal
        # UNDER A CAUSAL MASK THE CLS TOKEN MOVES TO THE END. The mask is
        # triu(diagonal=1), so position i attends to j <= i; a CLS token
        # PREPENDED at index 0 would attend to nothing but itself and the
        # readout would be bit-for-bit identical for every input (checked --
        # two random batches differ by exactly 0.0). Appending it instead
        # leaves it attending to every patch, so it stays what it is meant to
        # be: a learned query that aggregates the whole view. The alternative,
        # forcing pool="last", would have made a causal run differ from the
        # control in TWO knobs at once and stopped being an ablation.
        #
        # Tied to `causal` rather than exposed as its own option on purpose:
        # the two placements produce IDENTICAL state dicts (same cls_token
        # shape, same n_pos), so a checkpoint trained under one and loaded
        # under the other would be silently wrong. Deriving the position from
        # a flag that IS recorded in train_meta.json removes that failure.
        self.cls_at_end = causal and pool == "cls"
        # A DEDICATED TOKEN FOR THE DECISION INSTANT. patch_size timesteps are
        # projected into one patch, so the value AT the anchor is mixed with
        # its 7 neighbours -- 48-88 s of them at the free scale. That mixing
        # is invertible in principle (72 numbers into 384 is over-complete),
        # but the encoder has to learn to invert it, and the single best
        # classical predictor of return_900 is an instantaneous function of
        # exactly that row. This projects the RAW final timestep into its own
        # token appended after the patches, so attending to the anchor costs
        # one attention head rather than an inversion.
        # EVERY CLASSICAL PREDICTOR THAT WORKS IS A DIFFERENCE, and the model
        # is handed levels. Ridge ARDL's return_900 coefficients are -ask(t),
        # -bid(t), +high(t), +low(t) -- a position within a range -- and AR/
        # ARMA read lagged returns; both are first differences of what the
        # encoder sees. A transformer can difference through attention, but it
        # has to learn to, so this concatenates the first difference of every
        # channel to the input and lets the patch projection start from it.
        # Computed HERE rather than in the dataset on purpose: it is a pure
        # function of the view, so there is nothing to plumb through
        # iter_panel, nothing to record in train_meta beyond the flag, and no
        # way for train and eval to disagree about it.
        # ONE INFORMATION TOKEN for the facts that belong to the WINDOW rather
        # than to any instant in it. The last ``n_info_channels`` input
        # channels carry a payload at the final valid timestep and are routed
        # here instead of through the patch embedding.
        #
        # They used to go through the patch embedding: view normalization
        # broadcasts the per-group (mu, sigma) along all 2048 timesteps and
        # concatenates, so 8 constants were re-read 256 times and the patch
        # projection's whole response to them was one vector added identically
        # to every patch. The compute was negligible (0.1% of the forward),
        # but it made n_features a function of a conditioning flag -- and
        # n_features is what the checkpoint's first projection is shaped by,
        # so turning a per-window fact on or off invalidated every checkpoint.
        # That coupling is what this separates.
        #
        # The token deliberately gets NO position embedding: it has no time,
        # so any position against it is a fiction. It is the natural home for
        # anything else that is
        # true of the whole view -- seconds per token, session fraction,
        # ticker or industry identity -- each of which costs one more input
        # column here and nothing anywhere else.
        self.n_info_channels = int(n_info_channels)
        n_patch_features = n_features - self.n_info_channels
        if n_patch_features < 1:
            raise ValueError(
                f"n_info_channels={n_info_channels} leaves no feature channels "
                f"for the patch embedding (n_features={n_features})")
        self.info_proj = (
            nn.Linear(self.n_info_channels, config.hidden_size)
            if self.n_info_channels else None
        )

        self.diff_channels = diff_channels
        in_ch = n_patch_features * (2 if diff_channels else 1)
        self.state_token = state_token
        self.state_proj = (
            nn.Linear(in_ch, config.hidden_size) if state_token else None
        )
        self.patch_size = config.patch_size
        self.gradient_checkpointing = gradient_checkpointing

        # Custom 1D patch embedding (reused)
        self.patch_embed = PatchEmbedding1D(
            n_features=in_ch,
            patch_size=config.patch_size,
            d_model=config.hidden_size,
            identity=bool(getattr(config, "patch_embed_identity", False)),
        )

        # CLS token (if cls pooling)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
            if pool == "cls"
            else None
        )

        # Learnable position embeddings
        # The info token is NOT counted: it is appended after the position
        # embeddings are added, so it never receives one.
        n_pos = max_seq_len + (1 if pool == "cls" else 0) + (1 if state_token else 0)
        # Same None-means-unset contract as pool above.
        self.pos_embed_kind = str(getattr(config, "pos_embed", None) or "learned")
        self.cls_pos = str(getattr(config, "cls_pos", None) or "own")
        if self.pos_embed_kind == "rope":
            # NO absolute position anywhere. A zero buffer keeps every code
            # path that adds `position_embeddings` (forward_patches, the CPC /
            # I-JEPA / MAE entry points) a no-op instead of an AttributeError.
            self.register_buffer("position_embeddings",
                                 torch.zeros(1, n_pos, config.hidden_size))
        elif self.pos_embed_kind == "sinusoidal":
            # Registered PERSISTENT so it travels in the state dict: a loader
            # that rebuilds this as "learned" then receives the same numbers
            # instead of a fresh random table, which turns a silent mis-score
            # into a correct one.
            self.register_buffer(
                "position_embeddings",
                _sinusoidal_table(n_pos, config.hidden_size))
            # cls_pos="own" needs a vector that is NOT one of the patch
            # positions: under sinusoidal the patches occupy 0..P-1 with no
            # slot to spare, so the CLS gets its own learned one rather than
            # stealing position 0 from the first patch.
            if pool == "cls" and self.cls_pos == "own":
                self.cls_pos_embed = nn.Parameter(
                    torch.randn(1, 1, config.hidden_size) * 0.02)
        elif self.pos_embed_kind == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(1, n_pos, config.hidden_size)
                * float(getattr(config, "pos_init_std", 0.02))
            )
        else:
            raise ValueError(f"unknown pos_embed: {self.pos_embed_kind}")


        # Transformer blocks with linearly scaled drop path rates
        dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                drop_path_rate=dpr[i],
                use_rope=(self.pos_embed_kind == "rope"),
            )
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm + projection
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, d_embedding)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Fused in_proj_weight is a raw Parameter, not nn.Linear
                if module.in_proj_weight is not None:
                    nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                # out_proj is nn.Linear, handled above
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.config.rescale_residual_init:
            _rescale_residual_branches(self.blocks)

    def _position_embedding(self, n_patch_pos: int, has_state: bool):
        """(1, L, H) positional term, assembled REGION BY REGION.

        THE PATCHES OWN POSITIONS 0..P-1. Slicing the table by sequence index
        instead -- which is what the learned path does -- shifts every patch by
        one whenever a CLS is prepended, so a 256-patch view occupies 1..256
        and "the last patch" resolves to position 256. The CLS is not a
        timestep and must not consume one.

        Order matches the sequence: [cls?] patches [state?] [cls if at end].
        """
        P = n_patch_pos
        patch = self.position_embeddings[:, :P]
        cls_vec = None
        if self.cls_token is not None:
            if self.cls_pos == "last":
                cls_vec = patch[:, -1:]
            elif self.cls_pos == "none":
                cls_vec = torch.zeros_like(patch[:, :1])
            else:
                cls_vec = self.cls_pos_embed
        parts = []
        if self.cls_token is not None and not self.cls_at_end:
            parts.append(cls_vec)
        parts.append(patch)
        if has_state:
            parts.append(self.position_embeddings[:, P:P + 1])
        if self.cls_token is not None and self.cls_at_end:
            parts.append(cls_vec)
        return torch.cat(parts, dim=1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process time series through transformer.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        batch_size = x.shape[0]

        # Split the per-window constants off the front of everything else.
        # The final valid timestep carries the metadata payload. Respect
        # ``lengths`` so right-padding cannot silently replace it with zeros.
        info = None
        if self.info_proj is not None:
            last = (
                lengths.to(device=x.device, dtype=torch.long) - 1
                if lengths is not None
                else torch.full((batch_size,), x.shape[-1] - 1, device=x.device)
            )
            payload = x[
                torch.arange(batch_size, device=x.device),
                -self.n_info_channels:,
                last,
            ]
            info = self.info_proj(payload).unsqueeze(1)
            x = x[:, :-self.n_info_channels]

        # Create patch embeddings
        # First differences alongside the levels. The leading column is zero
        # rather than dropped, so the sequence length -- and therefore the
        # patch grid and every position embedding -- is unchanged.
        if self.diff_channels:
            d = torch.zeros_like(x)
            d[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            x = torch.cat([x, d], dim=1)

        # The decision instant, BEFORE patching flattens it away. x is
        # (batch, channels, length) here, so the anchor is its last column.
        state = (self.state_proj(x[:, :, -1]).unsqueeze(1)
                 if self.state_proj is not None else None)

        x = self.patch_embed(x)  # (batch, n_patches, hidden_size)
        n_patch_pos = x.size(1)

        # Order is [cls?] patches [state?] [cls if causal]. The state token
        # goes AFTER the patches so that, under a causal mask, the trailing
        # CLS still attends to it.
        if state is not None:
            x = torch.cat([x, state], dim=1)

        # Add CLS token if using cls pooling; at the END under a causal mask,
        # where a prepended one could see nothing. See __init__.
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = (torch.cat([x, cls_tokens], dim=1) if self.cls_at_end
                 else torch.cat([cls_tokens, x], dim=1))

        # Add position embeddings (slice to actual seq_len)
        seq_len = x.size(1)
        if self.pos_embed_kind == "rope":
            pass                      # position enters through the rotation
        elif self.pos_embed_kind == "sinusoidal" or self.cls_pos != "own":
            # Region-wise, so the patches keep positions 0..P-1 (see
            # _position_embedding). Reached for ANY non-default cls_pos: the
            # learned table honours it too, or a learned+cls_pos=none run would
            # silently keep the CLS on slot 0 and train the arm it was not
            # asked for. cls_pos="own" on the learned table keeps the legacy
            # slice, because every existing checkpoint was trained under it.
            x = x + self._position_embedding(
                n_patch_pos, self.state_proj is not None)
        else:
            x = x + self.position_embeddings[:, :seq_len]

        # The info token goes on LAST, after the position embeddings, which is
        # how it gets none. A per-window fact has no place in the sequence, so
        # giving it a learned position would only teach the model to associate
        # the fact with a slot index.
        n_info_tok = 0
        if info is not None:
            x = torch.cat([x, info], dim=1)
            n_info_tok = 1
            seq_len = x.size(1)

        # Causal attn_mask, shape (L, L), True = mask out. Built here as well
        # as in forward_patches because the two entry points do not share a
        # body, and a flag honoured by only one of them is worse than no flag.
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            # ...except the info token, which sits last and would otherwise be
            # visible to nothing: under cls_at_end the CLS readout is at L-2
            # and triu would hide L-1 from it. Causality is a statement about
            # time and the info token has none.
            if n_info_tok:
                attn_mask[:, -n_info_tok:] = False

        # Compute attention masks from lengths
        key_padding_mask = None
        pool_mask = None
        if lengths is not None:
            # Built REGION BY REGION rather than by index arithmetic over the
            # whole sequence: with an optional CLS at either end and an
            # optional state token between, the off-by-one cases multiply and
            # every one of them is a silent correctness bug rather than a
            # crash. Each special token is always valid; only padded patch
            # slots are masked.
            n_patches = (lengths + self.patch_size - 1) // self.patch_size
            patch_idx = torch.arange(n_patch_pos, device=x.device).unsqueeze(0)
            patch_mask = patch_idx >= n_patches.unsqueeze(1)      # (B, P)
            valid = torch.zeros((batch_size, 1), dtype=torch.bool,
                                device=x.device)
            parts = []
            if self.cls_token is not None and not self.cls_at_end:
                parts.append(valid)
            parts.append(patch_mask)
            if self.state_proj is not None:
                parts.append(valid)
            if self.cls_token is not None and self.cls_at_end:
                parts.append(valid)
            if n_info_tok:
                parts.append(valid)
            key_padding_mask = torch.cat(parts, dim=1)
            # pool_mask: True = VALID (inverted from key_padding_mask)
            pool_mask = ~key_padding_mask

        kpm = key_padding_mask

        # Transformer blocks
        # Rotary positions, assembled the way _position_embedding assembles the
        # absolute ones: PATCHES get 0..P-1, and every token without a time
        # (CLS, state, info) gets 0 -- untimed tokens are given no position
        # rather than a fictional one.
        rope = None
        if self.pos_embed_kind == "rope":
            rp = torch.zeros(seq_len, device=x.device)
            lo = 1 if (self.cls_token is not None and not self.cls_at_end) else 0
            n = min(n_patch_pos, max(seq_len - lo, 0))
            if n > 0:
                rp[lo:lo + n] = torch.arange(n, device=x.device,
                                             dtype=rp.dtype)
            rope = _rope_cos_sin(rp, self.blocks[0].attn.head_dim,
                                 x.device, x.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, kpm, attn_mask, rope,
                                    use_reentrant=False)
            else:
                x = block(x, key_padding_mask=kpm, attn_mask=attn_mask,
                          rope=rope)

        # DROP THE INFO TOKEN BEFORE POOLING. It is an input to attention, not
        # a member of the sequence, and leaving it in breaks every pooling mode
        # in a different way: pool="cls" with cls_at_end reads x[:, -1] and
        # would return the info token instead of CLS, pool="last" would index
        # it as the final timestep, and mean/max would average it in. Removing
        # it here means the pooling switch below is bit-identical to what it
        # was before this token existed.
        if n_info_tok:
            x = x[:, :-n_info_tok]
            if pool_mask is not None:
                pool_mask = pool_mask[:, :-n_info_tok]

        # Pooling
        if self.pool == "cls":
            x = x[:, -1] if self.cls_at_end else x[:, 0]  # Take CLS token
        elif self.pool == "mean":
            if pool_mask is not None:
                mask_expanded = pool_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pool == "max":
            if pool_mask is not None:
                x = x.masked_fill(~pool_mask.unsqueeze(-1), float("-inf"))
            x = x.max(dim=1).values
        elif self.pool == "last":
            if pool_mask is not None:
                last_idx = pool_mask.sum(dim=1).clamp(min=1) - 1
            else:
                last_idx = torch.full(
                    (x.size(0),), x.size(1) - 1, device=x.device, dtype=torch.long
                )
            batch_idx = torch.arange(x.size(0), device=x.device)
            x = x[batch_idx, last_idx]
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        # Final projection
        x = self.layernorm(x)
        x = self.head(x)

        return x

    @property
    def feature_dim(self) -> int:
        """Width of what ``compute_features_multi`` returns.

        This is the TRUNK width, not ``d_embedding``: the sweep reads hidden
        states, and the head that maps hidden_size -> d_embedding is skipped.
        They happen to both be 384 in the sweep configs, which is exactly why
        this is stated rather than assumed.
        """
        return self.config.hidden_size

    @property
    def n_layers(self) -> int:
        """Depths available to ``compute_features_multi``: 0 .. n_layers."""
        return len(self.blocks)

    def compute_features_multi(
        self,
        view: torch.Tensor,
        lengths: torch.Tensor | None,
        layers: list[int],
        metadata: dict | None = None,
    ) -> dict[int, torch.Tensor]:
        """Pooled hidden state at several depths from ONE forward pass.

        Mirrors ``PretrainedTSFM.compute_features_multi`` so a layer sweep can
        treat a trained ViT and a frozen TSFM as the same kind of thing.

        Depth 0 is the patch+CLS+position embedding, i.e. the input to block 0;
        depth i is the output of block i-1. Every depth is read out with the
        model's OWN pooling, so a layer's feature is the same kind of quantity
        the trained model uses -- but WITHOUT the final layernorm+head, which
        are fitted for the last layer alone and would not mean the same thing
        applied to a shallower one.
        """
        bad = [L for L in layers if not (0 <= L <= self.n_layers)]
        if bad:
            raise ValueError(f"layers {bad} outside 0..{self.n_layers}")
        want = sorted(set(layers))
        deepest = max(want)

        # CHUNK, like PretrainedTSFM does. The caller hands over a whole panel
        # batch; a 2048-step view is 257 tokens, and holding every block's
        # activations for a few hundred of those at once is an OOM on a 44 GB
        # card. Sub-batching costs nothing -- the depths are captured per chunk
        # and concatenated.
        bs = int(getattr(self, "features_multi_batch_size", 128) or 128)
        if view.shape[0] > bs:
            parts: list[dict[int, torch.Tensor]] = []
            for i in range(0, view.shape[0], bs):
                sl = slice(i, i + bs)
                parts.append(self.compute_features_multi(
                    view[sl], None if lengths is None else lengths[sl],
                    want, metadata))
            return {L: torch.cat([p[L] for p in parts], dim=0) for L in want}

        batch_size = view.shape[0]
        x = self.patch_embed(view)
        n_pp_multi = x.size(1)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        seq_len = x.size(1)
        if self.pos_embed_kind == "sinusoidal":
            # Region-wise, so the patches keep positions 0..P-1 (see
            # _position_embedding). The legacy learned table is left exactly as
            # it was -- every existing checkpoint was trained under it.
            x = x + self._position_embedding(
                n_pp_multi, self.state_proj is not None)
        else:
            x = x + self.position_embeddings[:, :seq_len]

        key_padding_mask = pool_mask = None
        if lengths is not None:
            n_patches = (lengths + self.patch_size - 1) // self.patch_size
            seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            if self.cls_token is not None:
                key_padding_mask = seq_indices >= (n_patches + 1).unsqueeze(1)
            else:
                key_padding_mask = seq_indices >= n_patches.unsqueeze(1)
            pool_mask = ~key_padding_mask

        out: dict[int, torch.Tensor] = {}
        if 0 in want:
            out[0] = self._pool_hidden(x, pool_mask)
        # Stop at the deepest requested depth: the blocks past it cannot
        # change a shallower readout, so running them is pure cost.
        for i, block in enumerate(self.blocks[:deepest]):
            x = block(x, key_padding_mask=key_padding_mask)
            if (i + 1) in want:
                out[i + 1] = self._pool_hidden(x, pool_mask)
        return out

    def _pool_hidden(
        self, x: torch.Tensor, pool_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """The pooling half of ``forward``, applied to any depth's tokens."""
        if self.pool == "cls":
            return x[:, 0]
        if self.pool == "mean":
            if pool_mask is None:
                return x.mean(dim=1)
            m = pool_mask.unsqueeze(-1).float()
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if self.pool == "max":
            if pool_mask is not None:
                x = x.masked_fill(~pool_mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1).values
        if self.pool == "last":
            if pool_mask is not None:
                idx = pool_mask.sum(dim=1).clamp(min=1) - 1
            else:
                idx = torch.full((x.size(0),), x.size(1) - 1,
                                 device=x.device, dtype=torch.long)
            return x[torch.arange(x.size(0), device=x.device), idx]
        raise ValueError(f"Unknown pooling: {self.pool}")

    def patch_channels(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` reduced to exactly what ``patch_embed`` is shaped for.

        ANY caller that reaches ``self.patch_embed`` directly must come
        through here. The trailing ``n_info_channels`` columns are per-window
        constants routed to the information token, and ``diff_channels``
        doubles what remains, so the projection's input width is a function of
        two settings that a mode has no reason to know about. TimeMAE reached
        past both to tokenize its patches and died in conv1d on a 20-column
        view (2026-09-15); the helper exists so the next one cannot.

        Returns the channels only -- the info payload itself belongs to
        ``forward``/``forward_patches``, which append it as a token.
        """
        if self.n_info_channels:
            x = x[:, :-self.n_info_channels]
        if self.diff_channels:
            d = torch.zeros_like(x)
            d[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            x = torch.cat([x, d], dim=1)
        return x

    def forward_patches(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask_indices: list[torch.Tensor] | None = None,
        causal: bool = False,
        pos_offset: int = 0,
        zero_mask: torch.Tensor | None = None,
        relative_pos: bool = False,
    ) -> torch.Tensor:
        """Return per-patch representations (no pooling / head projection).

        Used by I-JEPA: the encoder processes only unmasked patches and
        returns their hidden representations.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).
            mask_indices: If provided, a list of tensors each of shape
                (num_kept_patches,) specifying which patch indices to keep.
                All tensors in the list must have the same length.
                The batch dimension is inferred: ``B = len(mask_indices)``.
                When given, patches are gathered before the transformer
                blocks (like the I-JEPA encoder).
            zero_mask: Optional (batch, n_kept_patches) bool tensor — True
                zeroes that patch's *content* embedding while keeping its
                position embedding (TS2Vec-style timestamp masking). Indexed
                in the post-gather patch order when ``mask_indices`` is given.
            relative_pos: With ``mask_indices``, index position embeddings by
                the patch's position *within the gathered crop* (0..K-1)
                instead of gathering the absolute codes at ``mask_indices``.
                Positions then encode order but not global timestamp — needed
                by contrastive objectives that align the same patch across
                two differently-offset crops (TS2Vec), where shared absolute
                codes hand every positive pair a position-matching shortcut.
                Leave False when cross-view positional alignment is the
                point (I-JEPA, CPC).

        Returns:
            Per-patch embeddings of shape (batch, n_patches, hidden_size),
            after layer-norm but *before* the final ``head`` linear.
        """
        batch_size = x.shape[0]

        # Split the per-window constants off, exactly as forward() does.
        # THIS ENTRY POINT WAS MISSED when the information token landed
        # (2026-09-13): forward() stripped the trailing columns and
        # forward_patches did not, so every mode that reaches the backbone
        # here -- cost, cpc, ijepa, mae, timemae, ts2vec -- handed a 20-column
        # view to a patch embedding built for 9 and died in conv1d. The six
        # that use forward() instead (byol, dino, tfc, lejepa, supervised)
        # were unaffected, which is why the wave failed at exactly 6 of 9 arms.
        info = None
        if self.info_proj is not None:
            last = (
                lengths.to(device=x.device, dtype=torch.long) - 1
                if lengths is not None
                else torch.full((batch_size,), x.shape[-1] - 1, device=x.device)
            )
            payload = x[
                torch.arange(batch_size, device=x.device),
                -self.n_info_channels:,
                last,
            ]
            info = self.info_proj(payload).unsqueeze(1)

        # The payload is read off the RAW view above, so this strips and
        # differences afterwards -- same helper every direct patch_embed
        # caller uses, so the width can only be got right in one place.
        x = self.patch_channels(x)

        # Patch embeddings
        x = self.patch_embed(x)  # (batch, n_patches, hidden_size)

        # Position embeddings (no CLS token for patch-level output).
        # ``pos_offset`` lets callers encode a slice of the full sequence with
        # its absolute position embeddings (e.g. CPC target encoder consuming
        # patches t_c..t_c+K).
        n_patches = x.size(1)
        pos_embed = self.position_embeddings[:, pos_offset:pos_offset + n_patches]

        if mask_indices is not None:
            # Apply masks: gather only the kept patches
            # mask_indices: list of B tensors, each (num_kept,)
            # Stack into (B, num_kept), expand to (B, num_kept, D)
            idx = torch.stack(mask_indices, dim=0).to(x.device)  # (B, K)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, dim=1, index=idx_expanded)
            if zero_mask is not None:
                x = x.masked_fill(zero_mask.unsqueeze(-1), 0.0)
            if relative_pos:
                x = x + self.position_embeddings[:, :idx.shape[1]]
            else:
                # Gather matching position embeddings
                pos_expand = pos_embed.expand(batch_size, -1, -1)
                pos = torch.gather(pos_expand, dim=1, index=idx_expanded)
                x = x + pos

            # Guard: clamp mask indices to valid (non-padded) patch range
            # so padded positions never leak into the context encoder.
            if lengths is not None:
                n_patch_lens = (lengths + self.patch_size - 1) // self.patch_size
                max_valid = n_patch_lens.unsqueeze(1)  # (B, 1)
                # True where a gathered index falls in the padded region
                key_padding_mask = idx >= max_valid  # (B, K)
        else:
            if zero_mask is not None:
                x = x.masked_fill(zero_mask.unsqueeze(-1), 0.0)
            x = x + pos_embed

        # Transformer blocks
        if not (mask_indices is not None and lengths is not None):
            key_padding_mask = None
        if mask_indices is None and lengths is not None:
            n_patch_lens = (lengths + self.patch_size - 1) // self.patch_size
            seq_len = x.size(1)
            seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            key_padding_mask = seq_indices >= n_patch_lens.unsqueeze(1)

        # The info token goes on AFTER every patch mask is built and comes off
        # again before the return. Both halves of that matter: the masks above
        # index the PATCH grid (key_padding_mask from lengths, mask_indices
        # from the caller), so a token appended earlier would be counted as a
        # padded patch; and this entry point's contract is one row per patch,
        # which I-JEPA's predictor, MAE's decoder and TS2Vec's alignment all
        # index positionally. So it rides through attention and is dropped.
        n_info_tok = 0
        if info is not None:
            x = torch.cat([x, info], dim=1)
            n_info_tok = 1
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask,
                     torch.zeros((x.size(0), 1), dtype=torch.bool,
                                 device=x.device)],
                    dim=1,
                )

        # Build causal attn_mask if requested. Shape (L, L), True = mask out.
        attn_mask = None
        if causal:
            seq_len = x.size(1)
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            # The info token sits last and has no time, so causality says
            # nothing about it; leave it visible to every position, as
            # forward() does.
            if n_info_tok:
                attn_mask[:, -n_info_tok:] = False

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    block, x, key_padding_mask, attn_mask, use_reentrant=False,
                )
            else:
                x = block(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        x = self.layernorm(x)
        if n_info_tok:
            x = x[:, :-n_info_tok]
        return x
