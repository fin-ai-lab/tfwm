"""I-JEPA model for time series self-supervised learning."""


from market_jepa.backbone_config import backbone_block
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_pytorch import EMA

from ..backbones.transformer import TransformerBackbone
from .base import TrainingModel, compute_collapse_metrics


def _sample_block_sizes(
    n_patches: int,
    n_targets: int = 4,
    target_scale: tuple[float, float] = (0.15, 0.2),
) -> list[int]:
    """Sample uniform block sizes for I-JEPA target blocks.

    Block sizes are sampled once and shared across all samples in the batch
    (required for tensor batching), but positions are sampled per-sample.

    Args:
        n_patches: Total number of patches in the sequence.
        n_targets: Number of target blocks to sample.
        target_scale: (min, max) fraction of total patches per target block.

    Returns:
        List of ``n_targets`` block lengths.
    """
    sizes = []
    for _ in range(n_targets):
        frac = torch.empty(1).uniform_(target_scale[0], target_scale[1]).item()
        block_len = max(1, round(frac * n_patches))
        block_len = min(block_len, n_patches)
        sizes.append(block_len)
    return sizes


def _sample_1d_block_masks(
    n_patches: int,
    n_targets: int = 4,
    target_scale: tuple[float, float] = (0.15, 0.2),
    context_crop_max: float = 0.15,
    block_sizes: list[int] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample 1D block masks for I-JEPA (single sample).

    Generates *target* blocks (contiguous patch ranges) and a *context* mask
    that excludes the target blocks, following the I-JEPA masking strategy
    adapted to 1D time series.

    Per the paper (Appendix A.1), mask **positions** are sampled independently
    per image while block **sizes** are uniform across the batch.  Pass
    ``block_sizes`` (from :func:`_sample_block_sizes`) to reuse sizes across
    samples; if ``None``, sizes are sampled here (single-sample convenience).

    Args:
        n_patches: Total number of patches in the sequence.
        n_targets: Number of target blocks to sample.
        target_scale: (min, max) fraction of total patches per target block.
        context_crop_max: Maximum fraction of patches to crop from one end
            of the context (after removing targets). Crop is applied to
            either the front or back, chosen randomly.
        block_sizes: Pre-sampled block lengths (one per target). When provided,
            ``n_targets`` and ``target_scale`` are ignored for sizing.

    Returns:
        context_indices: (N_ctx,) sorted indices of context patches.
        target_indices: List of ``n_targets`` tensors, each (N_t,) with sorted
            indices for one target block.
    """
    if block_sizes is None:
        block_sizes = _sample_block_sizes(n_patches, n_targets, target_scale)

    # Retry loop: if all patches are covered by target blocks the context
    # would be empty, which crashes downstream.  This is mathematically
    # possible for very small n_patches (<=4).  Resample positions on each
    # retry; block_sizes stay fixed per the paper.
    max_retries = 10
    for _attempt in range(max_retries):
        target_blocks: list[torch.Tensor] = []
        all_target_set: set[int] = set()

        for block_len in block_sizes:
            # Sample start position (independently per call = per sample)
            max_start = max(0, n_patches - block_len)
            start = torch.randint(0, max_start + 1, (1,)).item()
            indices = torch.arange(start, start + block_len)
            target_blocks.append(indices)
            all_target_set.update(indices.tolist())

        # Context: all patches not in any target block
        all_indices = set(range(n_patches))
        ctx_list = sorted(all_indices - all_target_set)

        if ctx_list:
            break
    else:
        raise RuntimeError(
            f"Could not produce a non-empty context mask after {max_retries} "
            f"retries (n_patches={n_patches}, block_sizes={block_sizes}). "
            f"Consider increasing n_patches or reducing target coverage."
        )

    # 1D adaptation of the paper's 2D context block sampling:
    # The paper samples a large 2D context block (85-100% scale with unit
    # aspect ratio) then removes target overlaps.  For 1D time series, we
    # instead crop 0-context_crop_max fraction from either the front or back
    # of the remaining context patches, which achieves the analogous effect
    # of limiting the context extent without a spatial aspect ratio.
    # Crop 0-context_crop_max fraction from either front or back
    if ctx_list and context_crop_max > 0:
        crop_frac = torch.empty(1).uniform_(0.0, context_crop_max).item()
        n_crop = round(crop_frac * n_patches)
        if n_crop > 0 and n_crop < len(ctx_list):
            if torch.rand(1).item() < 0.5:
                # Crop from front: remove patches with smallest indices
                ctx_list = ctx_list[n_crop:]
            else:
                # Crop from back: remove patches with largest indices
                ctx_list = ctx_list[:-n_crop]

    context_indices = torch.tensor(ctx_list, dtype=torch.long)
    return context_indices, target_blocks


class IJEPAPredictor(nn.Module):
    """Lightweight transformer predictor for I-JEPA.

    Takes context-encoder output for visible patches and predicts
    representations at masked (target) positions.

    Args:
        hidden_size: Encoder hidden dimension (output of ``forward_patches``).
        pred_emb_dim: Predictor's internal dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        max_patches: Maximum number of patches (for position embeddings).
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        hidden_size: int,
        pred_emb_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        max_patches: int = 256,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(hidden_size, pred_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_emb_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Learnable position embeddings
        self.predictor_pos_embed = nn.Parameter(
            torch.randn(1, max_patches, pred_emb_dim) * 0.02
        )

        from ..backbones.transformer import ViTBlock

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.predictor_blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=pred_emb_dim,
                num_attention_heads=num_heads,
                intermediate_size=pred_emb_dim * 4,
                drop_path_rate=dpr[i],
            )
            for i in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(pred_emb_dim)
        self.predictor_proj = nn.Linear(pred_emb_dim, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict representations at target positions.

        Args:
            context_tokens: (B, N_ctx, hidden_size) — encoder output for
                context patches.
            context_indices: (B, N_ctx) — original patch positions of context.
            target_indices: (B, N_tgt) — original patch positions of targets.
            context_padding_mask: (B, N_ctx) — boolean mask where True means
                the context position is padding and should be ignored.

        Returns:
            Predicted representations (B, N_tgt, hidden_size).
        """
        B = context_tokens.shape[0]

        # Project encoder dim -> predictor dim
        x = self.predictor_embed(context_tokens)  # (B, N_ctx, pred_dim)

        # Add positional embeddings for context positions
        ctx_pos = self._gather_pos(context_indices, B)  # (B, N_ctx, pred_dim)
        x = x + ctx_pos

        N_ctx = x.shape[1]

        # Create mask tokens for target positions
        N_tgt = target_indices.shape[1]
        tgt_pos = self._gather_pos(target_indices, B)  # (B, N_tgt, pred_dim)
        pred_tokens = self.mask_token.expand(B, N_tgt, -1) + tgt_pos

        # Concatenate: [context_tokens, mask_tokens]
        x = torch.cat([x, pred_tokens], dim=1)  # (B, N_ctx + N_tgt, pred_dim)

        # Build key_padding_mask for the concatenated sequence:
        # padded context positions are ignored, target positions are never padded.
        key_padding_mask = None
        if context_padding_mask is not None:
            tgt_mask = torch.zeros(B, N_tgt, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([context_padding_mask, tgt_mask], dim=1)

        # Transformer blocks
        for block in self.predictor_blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.predictor_norm(x)

        # Return only the target token predictions
        x = x[:, N_ctx:]  # (B, N_tgt, pred_dim)
        x = self.predictor_proj(x)  # (B, N_tgt, hidden_size)
        return x

    def _gather_pos(self, indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Gather position embeddings for given patch indices."""
        # indices: (B, K)
        D = self.predictor_pos_embed.shape[-1]
        pos = self.predictor_pos_embed.expand(batch_size, -1, -1)  # (B, max_patches, D)
        idx = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
        return torch.gather(pos, dim=1, index=idx)


class IJEPA(TrainingModel):
    """I-JEPA (Image-style JEPA) for time series self-supervised learning.

    Implements the I-JEPA architecture adapted for 1D time series:
    - Context encoder processes only visible (unmasked) patches.
    - Target encoder (EMA of context encoder) processes all patches.
    - Predictor predicts target representations from context.
    - Loss: smooth L1 (default) or MSE between predicted and target representations.

    Only supports TransformerBackbone.

    Args:
        backbone: A TransformerBackbone instance (used as context encoder).
        pred_depth: Number of transformer layers in the predictor.
        pred_emb_dim: Hidden dimension of the predictor.
        ema_start: Minimum EMA decay (clamped floor for ema-pytorch schedule).
        ema_end: Target EMA decay (beta for ema-pytorch).
        n_targets: Number of target blocks to sample.
        target_scale: (min, max) fraction of patches per target block.
        context_crop_max: Maximum fraction of patches to crop from one end
            of the context after removing targets (0 to 0.15 typical).
        loss_fn: Loss function for predicted vs target representations.
            ``"smooth_l1"`` (default) or ``"mse"``.
        gradient_checkpointing: If True, enable gradient checkpointing on
            context encoder and predictor. **Not supported on LeJEPA** — only
            on IJEPA.
    """

    def __init__(
        self,
        backbone: TransformerBackbone,
        pred_depth: int = 6,
        pred_emb_dim: int = 192,
        pred_num_heads: int | None = None,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        n_targets: int = 4,
        target_scale: list[float] | None = None,
        context_crop_max: float = 0.15,
        loss_fn: str = "smooth_l1",
        gradient_checkpointing: bool = False,
    ):
        if backbone.pool == "cls":
            raise ValueError(
                "IJEPA requires pool='mean' or pool='max' (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )

        super().__init__()

        if target_scale is None:
            target_scale = [0.15, 0.2]

        # Context encoder = the backbone itself
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding

        # Store EMA schedule endpoints for linear momentum schedule
        self._ema_start = ema_start
        self._ema_end = ema_end

        # Enable gradient checkpointing on the context encoder
        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        # Target encoder = EMA of context encoder (managed by ema-pytorch)
        # NOTE: We pass beta=ema_start (not ema_end) to avoid is_frozen=True
        # when ema_end=1.0. We bypass ema-pytorch's internal schedule anyway
        # by passing current_decay directly in post_training_step.
        self.ema = EMA(
            backbone,
            beta=ema_start,
            update_after_step=0,
            update_every=1,
            include_online_model=False,
        )
        self.ema.ema_model.gradient_checkpointing = False
        self.ema.ema_model.eval()
        for p in self.ema.ema_model.parameters():
            p.requires_grad = False

        # Predictor
        hidden_size = backbone.config.hidden_size
        max_patches = backbone.position_embeddings.shape[1] // backbone.patch_size
        # Use same number of heads as the context encoder unless overridden
        num_heads = pred_num_heads if pred_num_heads is not None else backbone.config.num_attention_heads

        self.predictor = IJEPAPredictor(
            hidden_size=hidden_size,
            pred_emb_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=num_heads,
            max_patches=max_patches,
        )

        # Loss function
        if loss_fn == "mse":
            self._loss_fn = F.mse_loss
        elif loss_fn == "smooth_l1":
            self._loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn!r}. Expected 'mse' or 'smooth_l1'.")

        # Masking config
        self.n_targets = n_targets
        self.target_scale = tuple(target_scale)
        self.context_crop_max = context_crop_max

        # Depth-scaled weight init (matches original I-JEPA repo)
        self._fix_init_weight()

    mode_label: str = "I-JEPA"
    mode_str: str = "I-JEPA"
    uses_multi_view: bool = False

    def train(self, mode: bool = True):
        """Override to keep EMA target encoder in eval mode.

        StochasticDepth (DropPath) is active during training; the target
        encoder must produce deterministic representations, as prescribed
        by the paper and all EMA-based self-supervised methods (BYOL, DINO,
        data2vec, I-JEPA reference code).
        """
        super().train(mode)
        # Always keep target encoder in eval mode so DropPath is disabled
        self.ema.ema_model.eval()
        return self

    def _fix_init_weight(self):
        """Rescale attn proj and MLP output weights by 1/sqrt(2*layer_id).

        Matches the depth-scaled init from the original I-JEPA repo
        (vision_transformer.py fix_init_weight) and ijepa_example.py.
        Stabilizes forward-pass variance in deep transformers.
        """
        def _rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        # Context encoder (backbone)
        for layer_id, block in enumerate(self.backbone.blocks):
            _rescale(block.attn.out_proj.weight.data, layer_id + 1)
            _rescale(block.mlp[2].weight.data, layer_id + 1)

        # Target encoder (EMA copy) — apply same rescaling
        for layer_id, block in enumerate(self.ema.ema_model.blocks):
            _rescale(block.attn.out_proj.weight.data, layer_id + 1)
            _rescale(block.mlp[2].weight.data, layer_id + 1)

        # Predictor
        for layer_id, block in enumerate(self.predictor.predictor_blocks):
            _rescale(block.attn.out_proj.weight.data, layer_id + 1)
            _rescale(block.mlp[2].weight.data, layer_id + 1)

    def encode(
        self,
        x: torch.Tensor | list[torch.Tensor],
        lengths: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Produce embeddings using the target encoder (EMA model).

        Per the paper (Appendix A.1, "Architectures"): "We use the
        target-encoder for evaluation and average pool its output to produce
        a global image representation."
        """
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            views = [x[:, v, :, :] for v in range(x.shape[1])]
            view_lengths = [lengths] * x.shape[1] if lengths is not None else [None] * x.shape[1]
        elif isinstance(x, list):
            views = x
            view_lengths = lengths if lengths is not None else [None] * len(views)
        else:
            views = [x]
            view_lengths = [lengths]

        embeddings = [self.ema.ema_model(view, vl) for view, vl in zip(views, view_lengths)]
        return {"embeddings": torch.stack(embeddings, dim=1)}

    def default_run_name(self, backbone_type, cfg):
        # Use mode.backbone if present (IJEPA owns its backbone config),
        # otherwise fall back to top-level cfg.backbone.
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=ijepa",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(self, backbone=self.backbone, predictor=self.predictor)
        n_target = sum(p.numel() for p in self.ema.ema_model.parameters())
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"I-JEPA model parameters:\n"
            f"  Context encoder: {param_counts['backbone']:,}\n"
            f"  Predictor: {param_counts['predictor']:,}\n"
            f"  Total (trainable): {param_counts['total']:,}\n"
            f"  Target encoder (frozen, EMA): {n_target:,}"
        )
        return param_counts, summary

    def post_training_step(self, completed_steps, max_train_steps):
        # Linear momentum schedule per paper: linearly increase from
        # ema_start to ema_end over training (Appendix A.1, "Optimization").
        frac = completed_steps / max(1, max_train_steps - 1)
        momentum = self._ema_start + frac * (self._ema_end - self._ema_start)
        momentum = min(momentum, 1.0)

        # Bypass ema-pytorch's internal schedule by passing current_decay directly
        self.ema.update_moving_average(self.ema.ema_model, self.ema.model, current_decay=momentum)
        return {"train/ema_momentum": momentum}

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """I-JEPA forward pass: mask → encode → predict → loss.

        Per the paper (Appendix A.1), mask positions are sampled independently
        per sample while block sizes are uniform across the batch for efficient
        tensor batching.
        """
        batch_size = x.shape[0]

        # Determine number of patches
        _, _, T = x.shape
        patch_size = self.backbone.patch_size
        n_patches = (T + patch_size - 1) // patch_size

        # --- Target encoder: process ALL patches (no grad) ---
        with torch.no_grad():
            target_repr = self.ema.ema_model.forward_patches(x, lengths)
            # (B, n_patches, hidden_size)
            # NOTE: forward_patches already applies the backbone's learned
            # LayerNorm; no additional normalization is needed here.
            # The paper does not prescribe extra normalization beyond the
            # standard ViT architecture, and the reference implementation
            # does not apply one.

        # --- Sample block sizes ONCE for the batch, positions PER SAMPLE ---
        block_sizes = _sample_block_sizes(
            n_patches=n_patches,
            n_targets=self.n_targets,
            target_scale=self.target_scale,
        )

        per_sample_masks = [
            _sample_1d_block_masks(
                n_patches=n_patches,
                context_crop_max=self.context_crop_max,
                block_sizes=block_sizes,
            )
            for _ in range(batch_size)
        ]
        # per_sample_masks[i] = (ctx_indices_i, [tgt_block_0, tgt_block_1, ...])

        # Context sizes may differ per sample due to overlap/cropping.
        # Truncate all to the minimum size (as in the original I-JEPA repo)
        # to avoid duplicate-padding that biases self-attention.
        ctx_masks = [m[0] for m in per_sample_masks]
        min_ctx = min(len(c) for c in ctx_masks)
        ctx_mask_padded = [c[:min_ctx] for c in ctx_masks]

        # --- Context encoder: process only context patches ---
        context_repr = self.backbone.forward_patches(
            x, lengths, mask_indices=ctx_mask_padded
        )  # (B, min_ctx, hidden_size)

        ctx_indices = torch.stack(ctx_mask_padded).to(x.device)  # (B, min_ctx)

        # --- Predictor: batched prediction across all target blocks ---
        # Stack all target blocks into one tensor: (B, n_targets, K_t) → (B, N_tgt)
        hidden_size = target_repr.shape[-1]
        all_tgt_indices = []
        for block_idx in range(len(block_sizes)):
            tgt_indices = torch.stack([
                per_sample_masks[i][1][block_idx].to(x.device)
                for i in range(batch_size)
            ])  # (B, K_t)
            all_tgt_indices.append(tgt_indices)

        # Concatenate all target blocks: (B, total_tgt_patches)
        all_tgt = torch.cat(all_tgt_indices, dim=1)  # (B, N_tgt)

        # Single predictor forward pass with all targets concatenated
        predicted = self.predictor(
            context_repr, ctx_indices, all_tgt
        )  # (B, N_tgt, hidden_size)

        # Gather all target representations at once
        tgt_expanded = all_tgt.unsqueeze(-1).expand(-1, -1, hidden_size)
        target_at_mask = torch.gather(
            target_repr, dim=1, index=tgt_expanded
        )  # (B, N_tgt, hidden_size)

        loss = self._loss_fn(predicted, target_at_mask)

        return {
            "ijepa_loss": loss,
            "_target_repr": target_repr,  # (B, n_patches, hidden) for collapse monitoring
            "_predicted": predicted,       # (B, N_tgt, hidden) for baseline ratio
            "_target_at_mask": target_at_mask,  # (B, N_tgt, hidden)
        }

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        all_target_reprs = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                output = self(x, lengths)
                loss = output["ijepa_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]

                # Collect for collapse monitoring (detached, no grad)
                all_target_reprs.append(output["_target_repr"].detach())

        if batch_n == 0:
            return None

        metrics = {"train/loss": batch_loss / batch_n}

        # Collapse monitoring on target encoder (EMA) representations
        with torch.no_grad():
            if all_target_reprs:
                # Pool target patches → per-sample embeddings, then compute metrics
                target_cat = torch.cat(all_target_reprs, dim=0)  # (N, n_patches, H)
                target_pooled = target_cat.mean(dim=1)  # (N, H)
                metrics.update(compute_collapse_metrics(target_pooled, prefix="train"))

        return {
            "loss": batch_loss / batch_n,
            "metrics": metrics,
        }

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses = []

        for batch in eval_batches:
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self(x, lengths)
                    loss = output["ijepa_loss"]

                if torch.isfinite(loss):
                    losses.append(loss.item())

        return {"eval/ijepa_loss": np.mean(losses) if losses else float("nan")}

