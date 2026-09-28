"""1D ConvNeXt backbone for time series.

Adapted from torchvision's ConvNeXt (Liu et al., 2022 — "A ConvNet for the 2020s")
to operate on 1D sequences. Keeps the modern-conv design choices that let pure
convs compete with ViT: depthwise 7-wide kernels, LayerNorm instead of BN,
inverted bottleneck (4x expansion) MLPs, GELU, LayerScale, and StochasticDepth.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from .base import TimeSeriesBackbone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LayerNorm1dChannelsFirst(nn.Module):
    """LayerNorm over channel dim for (B, C, L) tensors.

    ConvNeXt's CNBlock applies LN over channels, which torchvision does by
    permuting to channels-last, applying nn.LayerNorm, then permuting back.
    For the stem / downsample layers (which stay channels-first), this fused
    version avoids two transposes.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class CNBlock1D(nn.Module):
    """ConvNeXt block for 1D sequences.

    Structure: DWConv1d(7) → permute → LN → Linear(4x) → GELU → Linear(1x)
    → permute → LayerScale → StochasticDepth → residual.

    Args:
        dim: Channel dimension.
        layer_scale: Initial value for LayerScale (0 disables).
        drop_path_rate: StochasticDepth probability for this block.
        kernel_size: Depthwise kernel size (default 7, ConvNeXt's choice).
    """

    def __init__(
        self,
        dim: int,
        layer_scale: float,
        drop_path_rate: float,
        kernel_size: int = 7,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Depthwise conv (channels-first), then the pointwise MLP runs
        # channels-last so the LayerNorm + 2 Linears are well-behaved.
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=True)
        # Per-channel scale; broadcasts over (B, L, C).
        self.layer_scale = (
            nn.Parameter(torch.ones(dim) * layer_scale) if layer_scale > 0 else None
        )
        self.drop_path = StochasticDepth(p=drop_path_rate, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        identity = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, L, C) for the MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale
        x = x.transpose(1, 2)  # back to (B, C, L)
        x = self.drop_path(x)
        return identity + x


# ---------------------------------------------------------------------------
# Variant configs  (Table 9 of the ConvNeXt paper)
# ---------------------------------------------------------------------------


@dataclass
class _ConvNeXtVariantConfig:
    depths: list[int]
    dims: list[int]


_VARIANTS: dict[str, _ConvNeXtVariantConfig] = {
    # Tiny   — 28M params in 2D; depths × dims per the paper.
    "tiny":  _ConvNeXtVariantConfig(depths=[3, 3, 9, 3],  dims=[96, 192, 384, 768]),
    "small": _ConvNeXtVariantConfig(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base":  _ConvNeXtVariantConfig(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large": _ConvNeXtVariantConfig(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
}

VARIANT_TYPE = Literal["tiny", "small", "base", "large"]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ConvNeXtBackbone(TimeSeriesBackbone):
    """1D ConvNeXt backbone for time series.

    Supports Tiny / Small / Base / Large. Uses a stride-4 stem (matching the
    ConvNeXt paper's image patchify stem) and three stride-2 downsamplers
    between stages, so total stride is 32 — comparable to ResNet.

    Args:
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
        variant: ConvNeXt variant ('tiny', 'small', 'base', 'large').
        stochastic_depth_prob: Maximum stochastic depth probability.
        layer_scale: Initial value for per-block LayerScale (0 disables).
        pool: Pooling strategy ('mean', 'max', 'both').
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 512,
        variant: VARIANT_TYPE = "tiny",
        stochastic_depth_prob: float = 0.1,
        layer_scale: float = 1e-6,
        pool: str = "mean",
    ):
        super().__init__(n_features, d_embedding)
        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown ConvNeXt variant: {variant}. Choose from {list(_VARIANTS.keys())}"
            )
        self.pool = pool

        cfg = _VARIANTS[variant]
        depths, dims = cfg.depths, cfg.dims
        norm_first = partial(LayerNorm1dChannelsFirst, eps=1e-6)

        # ── Stem: Conv(kernel=4, stride=4) → LN, like the paper's patchify stem. ──
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, dims[0], kernel_size=4, stride=4),
            norm_first(dims[0]),
        )

        # ── Downsamplers between stages (3 of them, each stride-2). ──
        self.downsample_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    norm_first(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # ── Stages: each is a sequence of CNBlock1Ds at that stage's dim. ──
        total_blocks = sum(depths)
        dpr = torch.linspace(0, stochastic_depth_prob, total_blocks).tolist()

        self.stages = nn.ModuleList()
        block_idx = 0
        for stage_idx, n_blocks in enumerate(depths):
            blocks: list[nn.Module] = []
            for _ in range(n_blocks):
                blocks.append(
                    CNBlock1D(
                        dim=dims[stage_idx],
                        layer_scale=layer_scale,
                        drop_path_rate=dpr[block_idx],
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        # ── Final head: LN on pooled features → Linear to d_embedding. ──
        final_channels = dims[-1]
        pool_factor = 2 if pool == "both" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(final_channels * pool_factor),
            nn.Linear(final_channels * pool_factor, d_embedding),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process time series through ConvNeXt.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        x = self.stem(x)
        x = self.stages[0](x)
        for i in range(1, len(self.stages)):
            x = self.downsample_layers[i - 1](x)
            x = self.stages[i](x)

        # Global pooling. Total stride = 4 (stem) * 2^3 (three downsamples) = 32.
        if lengths is not None:
            reduced_lengths = (lengths + 31) // 32
            max_len = x.size(-1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < reduced_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()

            if self.pool == "mean":
                x = (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            elif self.pool == "max":
                x = x.masked_fill(~mask.bool(), float("-inf")).max(dim=-1).values
            elif self.pool == "both":
                mean_pool = (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                max_pool = x.masked_fill(~mask.bool(), float("-inf")).max(dim=-1).values
                x = torch.cat([mean_pool, max_pool], dim=-1)
            else:
                raise ValueError(f"Unknown pooling: {self.pool}")
        else:
            if self.pool == "mean":
                x = x.mean(dim=-1)
            elif self.pool == "max":
                x = x.max(dim=-1).values
            elif self.pool == "both":
                x = torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
            else:
                raise ValueError(f"Unknown pooling: {self.pool}")

        return self.head(x)
