"""1D PatchTST backbone for time series.

Adapted from Nie et al. 2023 ("A Time Series is Worth 64 Words: Long-Term
Forecasting with Transformers"). The defining idea is **channel independence**:
each of the `n_features` input channels is processed by the *same* transformer,
with no cross-channel attention during the mixing stage — the backbone only
fuses channels at the very end via a flatten-and-project head.

Why this is a useful comparison point vs. our plain ViT backbone:
 - ViT's PatchEmbedding1D projects all channels jointly into each patch token,
   so attention mixes channels from layer 0.
 - PatchTST learns a per-channel temporal prior that is shared across channels,
   then does late-fusion at the head. On market data this often helps because
   different tickers / features have similar local temporal structure.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeSeriesBackbone
from .transformer import ViTBlock


@dataclass
class PatchTSTConfig:
    hidden_size: int = 128
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    intermediate_size: int = 512
    patch_size: int = 16
    patch_stride: int = 8
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.1


class PatchTSTBackbone(TimeSeriesBackbone):
    """Channel-independent patched transformer for time series.

    Per the PatchTST paper:
      1. Reshape (B, C, L) → (B*C, L) and apply a sliding-window patching with
         stride < patch_size (overlapping patches).
      2. Project each patch via shared Linear → (B*C, n_patches, d_model).
      3. Add per-patch positional embeddings.
      4. Run through L transformer blocks with shared weights across channels.
      5. Pool patches (mean) → (B*C, d_model), reshape to (B, C, d_model),
         flatten and project to d_embedding.

    Args:
        config: PatchTSTConfig.
        n_features: Number of input features (channels).
        d_embedding: Output embedding dimension.
        max_seq_len: Max raw sequence length (for position embedding buffer).
        pool: Patch pooling ('mean', 'last', 'max').
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 512,
        config: PatchTSTConfig | None = None,
        max_seq_len: int = 2048,
        pool: str = "mean",
    ):
        super().__init__(n_features, d_embedding)
        if config is None:
            config = PatchTSTConfig()
        self.config = config
        self.pool = pool
        self.patch_size = config.patch_size
        self.patch_stride = config.patch_stride

        # Per-channel patch embedding: Linear over a raw window of length
        # patch_size → d_model. Shared across channels (that's the "channel
        # independent" part).
        self.patch_proj = nn.Linear(config.patch_size, config.hidden_size)

        # Max number of patches at the longest seq_len; we slice per forward
        # pass to the actual number of patches for each input.
        max_patches = max(1, (max_seq_len - config.patch_size) // config.patch_stride + 1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_patches, config.hidden_size) * 0.02
        )

        dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                drop_path_rate=dpr[i],
            )
            for i in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Late fusion head: flatten all channels' pooled embeddings and project
        # to d_embedding. Using n_features * d_model as the flatten width keeps
        # the head parametric-free of raw L (patches are mean-pooled first).
        self.head = nn.Sequential(
            nn.LayerNorm(n_features * config.hidden_size),
            nn.Linear(n_features * config.hidden_size, d_embedding),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _sliding_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window patching along the time dim.

        x: (B*C, L) → (B*C, n_patches, patch_size) with stride = patch_stride
        and window = patch_size (overlapping when patch_stride < patch_size).
        Pads right so (L - patch_size) is divisible by patch_stride.
        """
        BC, L = x.shape
        pad = 0
        if L < self.patch_size:
            pad = self.patch_size - L
        else:
            remainder = (L - self.patch_size) % self.patch_stride
            if remainder != 0:
                pad = self.patch_stride - remainder
        if pad > 0:
            x = F.pad(x, (0, pad))
        # unfold emits (B*C, n_patches, patch_size).
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process time series through PatchTST.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        batch, C, L = x.shape
        assert C == self.n_features, f"Expected {self.n_features} channels, got {C}"

        # Flatten (B, C, L) → (B*C, L) so we process each channel independently
        # with shared weights.
        x = x.reshape(batch * C, L)
        x = self._sliding_patches(x)                       # (B*C, n_patches, patch_size)
        n_patches = x.size(1)

        x = self.patch_proj(x)                             # (B*C, n_patches, d_model)
        x = x + self.position_embeddings[:, :n_patches]

        # key_padding_mask from lengths: we derive a mask over patches from
        # the real length (ignoring the channel dim — valid mask is the same
        # for every channel of the same input).
        key_padding_mask = None
        pool_mask = None
        if lengths is not None:
            # A patch is "valid" if its start index < length.
            start_idx = torch.arange(n_patches, device=x.device) * self.patch_stride
            # (B, n_patches)
            valid = start_idx.unsqueeze(0) < lengths.unsqueeze(1)
            # Expand to per-channel batch dim: (B*C, n_patches).
            valid = valid.repeat_interleave(C, dim=0)
            key_padding_mask = ~valid  # True = ignore
            pool_mask = valid

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.layernorm(x)

        # Pool across patches (per channel).
        if self.pool == "mean":
            if pool_mask is not None:
                m = pool_mask.unsqueeze(-1).float()
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pool == "last":
            if pool_mask is not None:
                last_idx = pool_mask.sum(dim=1).clamp(min=1) - 1
            else:
                last_idx = torch.full(
                    (x.size(0),), x.size(1) - 1, device=x.device, dtype=torch.long
                )
            batch_idx = torch.arange(x.size(0), device=x.device)
            x = x[batch_idx, last_idx]
        elif self.pool == "max":
            if pool_mask is not None:
                x = x.masked_fill(~pool_mask.unsqueeze(-1), float("-inf"))
            x = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        # Reshape (B*C, d_model) → (B, C*d_model), then project to d_embedding.
        x = x.reshape(batch, C * self.config.hidden_size)
        return self.head(x)
