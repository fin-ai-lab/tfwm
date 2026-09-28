"""1D Inception backbone for time series.

Adapted from torchvision's ``inception_v3`` (Szegedy et al., 2016 —
"Rethinking the Inception Architecture for Computer Vision") to operate on 1D
sequences. The identifying feature of Inception is **multi-scale parallel
branches**: each block runs several convolutions of different kernel sizes in
parallel and concatenates the outputs along the channel dim. This gives a
different inductive bias from ResNet/ConvNeXt (sequential single-kernel
residual blocks) or EfficientNet (single-kernel MBConv bottleneck).

Two 1D-specific simplifications from the 2D reference:

 1. **Asymmetric factored convs collapse.** In 2D, Inception factorizes large
    kernels as ``(1, k) → (k, 1)`` to save params while keeping a k×k receptive
    field. In 1D there's only one spatial axis (time), so the two factored
    convs become the same op — we replace each ``(1, k) + (k, 1)`` pair with a
    single ``kernel=k`` Conv1d of equal receptive field.
 2. **InceptionE's parallel (1,3)+(3,1) branches stay parallel.** Both become
    ``kernel=3`` Conv1d in 1D but keep distinct weights so the block preserves
    its original output width and expressive capacity.

Input is ``(B, n_features, L)``: ``n_features`` is treated as the Conv1d
input-channel dim with no spatial structure assumed (the raw column order is
meaningless — features are e.g. bid_price vs volume, not temporally ordered).
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeSeriesBackbone


# ---------------------------------------------------------------------------
# Basic block
# ---------------------------------------------------------------------------


class BasicConv1d(nn.Module):
    """Conv1d → BN → ReLU, matching the stddev=0.001 BN eps used in the paper."""

    def __init__(self, in_channels: int, out_channels: int, **conv_kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **conv_kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ---------------------------------------------------------------------------
# Inception blocks (1D)
# ---------------------------------------------------------------------------


class InceptionA1D(nn.Module):
    """Paper's first Inception block — 4 parallel branches, no downsampling."""

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        conv_block = conv_block or BasicConv1d
        self.branch1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b5 = self.branch5_2(self.branch5_1(x))
        b3 = self.branch3dbl_3(self.branch3dbl_2(self.branch3dbl_1(x)))
        bp = self.branch_pool(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b5, b3, bp], dim=1)


class InceptionB1D(nn.Module):
    """Downsampling block — stride-2 branches, no 1×1 branch."""

    def __init__(
        self, in_channels: int, conv_block: Callable[..., nn.Module] | None = None
    ):
        super().__init__()
        conv_block = conv_block or BasicConv1d
        self.branch3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        bd = self.branch3dbl_3(self.branch3dbl_2(self.branch3dbl_1(x)))
        bp = F.max_pool1d(x, kernel_size=3, stride=2)
        return torch.cat([b3, bd, bp], dim=1)


class InceptionC1D(nn.Module):
    """"7×7" block — the 2D factored (1,7)+(7,1) pairs collapse to single conv7s in 1D."""

    def __init__(
        self,
        in_channels: int,
        channels_7: int,
        conv_block: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        conv_block = conv_block or BasicConv1d
        self.branch1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7
        # 2D: (1,7) + (7,1) — collapses to a single kernel=7 Conv1d.
        self.branch7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7_2 = conv_block(c7, 192, kernel_size=7, padding=3)

        # 2D: conv1 → (7,1) → (1,7) → (7,1) → (1,7); four factored pairs
        # collapse to two conv7s (total RF still matches: 1 + 6 + 6 = 13).
        self.branch7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7dbl_2 = conv_block(c7, c7, kernel_size=7, padding=3)
        self.branch7dbl_3 = conv_block(c7, 192, kernel_size=7, padding=3)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b7 = self.branch7_2(self.branch7_1(x))
        bd = self.branch7dbl_3(self.branch7dbl_2(self.branch7dbl_1(x)))
        bp = self.branch_pool(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b7, bd, bp], dim=1)


class InceptionD1D(nn.Module):
    """Second downsampling block; 2D factored (1,7)+(7,1) collapses to one conv7."""

    def __init__(
        self, in_channels: int, conv_block: Callable[..., nn.Module] | None = None
    ):
        super().__init__()
        conv_block = conv_block or BasicConv1d
        self.branch3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        # 2D: conv1 → (1,7) → (7,1) → conv3 s2.  1D: conv1 → conv7 → conv3 s2.
        self.branch7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x3_2 = conv_block(192, 192, kernel_size=7, padding=3)
        self.branch7x3_3 = conv_block(192, 192, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3_2(self.branch3_1(x))
        b7 = self.branch7x3_3(self.branch7x3_2(self.branch7x3_1(x)))
        bp = F.max_pool1d(x, kernel_size=3, stride=2)
        return torch.cat([b3, b7, bp], dim=1)


class InceptionE1D(nn.Module):
    """Final "3×3" block; two parallel conv3s per sub-branch kept distinct for capacity.

    In 2D the sub-branches are ``(1, 3)`` and ``(3, 1)`` concatenated. In 1D
    both are ``kernel=3`` Conv1d — we keep them parallel so the block's
    output width (384 × 2 per inner branch) is preserved; the two convs
    learn different weights.
    """

    def __init__(
        self, in_channels: int, conv_block: Callable[..., nn.Module] | None = None
    ):
        super().__init__()
        conv_block = conv_block or BasicConv1d
        self.branch1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3_2a = conv_block(384, 384, kernel_size=3, padding=1)
        self.branch3_2b = conv_block(384, 384, kernel_size=3, padding=1)

        self.branch3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3dbl_3a = conv_block(384, 384, kernel_size=3, padding=1)
        self.branch3dbl_3b = conv_block(384, 384, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)

        b3 = self.branch3_1(x)
        b3 = torch.cat([self.branch3_2a(b3), self.branch3_2b(b3)], dim=1)

        bd = self.branch3dbl_2(self.branch3dbl_1(x))
        bd = torch.cat([self.branch3dbl_3a(bd), self.branch3dbl_3b(bd)], dim=1)

        bp = self.branch_pool(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b3, bd, bp], dim=1)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class InceptionBackbone(TimeSeriesBackbone):
    """1D Inception-v3 backbone for time series.

    Stem (7 convs + 2 max-pools, stride 8 total) → 3× InceptionA → InceptionB
    (downsample) → 4× InceptionC → InceptionD (downsample) → 2× InceptionE →
    pool → Linear to d_embedding. Auxiliary classifier omitted (training-only
    crutch for ImageNet supervised; irrelevant for a representation backbone).

    Args:
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
        variant: Only 'v3' supported today (kept for API symmetry with other
            backbones that have size variants).
        dropout: Dropout applied before the final linear head.
        pool: Pooling strategy ('mean', 'max', 'both').
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 512,
        variant: Literal["v3"] = "v3",
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__(n_features, d_embedding)
        if variant != "v3":
            raise ValueError(f"Only 'v3' is supported, got {variant}")
        self.pool = pool

        # Stem — total stride 8 after maxpool2 (stride 2 stem + 2 max-pools).
        # The paper's 2D stem has stride 16 after a further ×2 from the first
        # Inception downsample; we keep that cumulative geometry below.
        self.conv1a = BasicConv1d(n_features, 32, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv1d(32, 32, kernel_size=3, padding=1)
        self.conv2b = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3b = BasicConv1d(64, 80, kernel_size=1)
        self.conv4a = BasicConv1d(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Inception blocks (same channel widths as the paper).
        self.mixed_5b = InceptionA1D(192, pool_features=32)   # 192 → 256
        self.mixed_5c = InceptionA1D(256, pool_features=64)   # 256 → 288
        self.mixed_5d = InceptionA1D(288, pool_features=64)   # 288 → 288

        self.mixed_6a = InceptionB1D(288)                      # 288 → 768 (stride 2)

        self.mixed_6b = InceptionC1D(768, channels_7=128)      # 768 → 768
        self.mixed_6c = InceptionC1D(768, channels_7=160)
        self.mixed_6d = InceptionC1D(768, channels_7=160)
        self.mixed_6e = InceptionC1D(768, channels_7=192)

        self.mixed_7a = InceptionD1D(768)                      # 768 → 1280 (stride 2)

        self.mixed_7b = InceptionE1D(1280)                     # 1280 → 2048
        self.mixed_7c = InceptionE1D(2048)                     # 2048 → 2048

        final_channels = 2048
        pool_factor = 2 if pool == "both" else 1
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(final_channels * pool_factor),
            nn.Linear(final_channels * pool_factor, d_embedding),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.1, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process time series through Inception-v3.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        # Stem
        x = self.conv1a(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool1(x)
        x = self.conv3b(x)
        x = self.conv4a(x)
        x = self.maxpool2(x)

        # Inception stages
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.mixed_5d(x)
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)
        x = self.mixed_7a(x)
        x = self.mixed_7b(x)
        x = self.mixed_7c(x)

        # Global pooling. Cumulative stride from stem + two Inception
        # downsamples: 2 (stem) × 2 (maxpool1) × 2 (maxpool2) × 2 (6a) × 2 (7a) = 32.
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

        x = self.dropout(x)
        return self.head(x)
