"""1D ResNet backbone for time series."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from .base import TimeSeriesBackbone


class BasicBlock1D(nn.Module):
    """Basic residual block for 1D ResNet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for downsampling.
        drop_path_rate: Drop path probability for StochasticDepth.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop_path = StochasticDepth(p=drop_path_rate, mode="row")

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.shortcut(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = identity + self.drop_path(out)
        out = F.gelu(out)

        return out


class BottleneckBlock1D(nn.Module):
    """Bottleneck residual block for 1D ResNet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels (before expansion).
        stride: Stride for downsampling.
        drop_path_rate: Drop path probability for StochasticDepth.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # 1x1 reduce
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 3x1 conv
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 expand
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.drop_path = StochasticDepth(p=drop_path_rate, mode="row")

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.shortcut(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = identity + self.drop_path(out)
        out = F.gelu(out)

        return out


class ResNetBackbone(TimeSeriesBackbone):
    """1D ResNet backbone for time series.

    Supports ResNet-18, 34, 50, 101, 152 configurations.

    Args:
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
        variant: ResNet variant ('18', '34', '50', '101', '152').
        initial_channels: Channels after stem convolution.
        drop_path_rate: Drop path probability (linearly scaled across blocks).
        pool: Pooling strategy ('mean', 'max', 'both', 'last').
    """

    CONFIGS = {
        "18": (BasicBlock1D, [2, 2, 2, 2]),
        "34": (BasicBlock1D, [3, 4, 6, 3]),
        "50": (BottleneckBlock1D, [3, 4, 6, 3]),
        "101": (BottleneckBlock1D, [3, 4, 23, 3]),
        "152": (BottleneckBlock1D, [3, 8, 36, 3]),
    }

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 512,
        variant: Literal["18", "34", "50", "101", "152"] = "50",
        initial_channels: int = 64,
        drop_path_rate: float = 0.1,
        pool: str = "mean",
        zero_init_residual: bool = True,
    ):
        super().__init__(n_features, d_embedding)

        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown ResNet variant: {variant}. Choose from {list(self.CONFIGS.keys())}")

        block_class, n_blocks = self.CONFIGS[variant]
        self.pool = pool
        self.zero_init_residual = zero_init_residual

        # Stem: initial convolution
        in_channels = n_features
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(initial_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Linearly scaled drop path rates across all blocks
        total_blocks = sum(n_blocks)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        # Build residual layers
        self.in_channels = initial_channels
        self._block_idx = 0
        self._dpr = dpr
        channels = [initial_channels, initial_channels * 2, initial_channels * 4, initial_channels * 8]

        self.layer1 = self._make_layer(block_class, channels[0], n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_class, channels[1], n_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_class, channels[2], n_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_class, channels[3], n_blocks[3], stride=2)

        del self._block_idx, self._dpr

        # Final channels after all layers
        final_channels = channels[3] * block_class.expansion

        # Final projection
        pool_factor = 2 if pool == "both" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(final_channels * pool_factor),
            nn.Linear(final_channels * pool_factor, d_embedding),
        )

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        block_class: type,
        out_channels: int,
        n_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []

        # First block may downsample
        layers.append(
            block_class(
                self.in_channels,
                out_channels,
                stride=stride,
                drop_path_rate=self._dpr[self._block_idx],
            )
        )
        self.in_channels = out_channels * block_class.expansion
        self._block_idx += 1

        # Remaining blocks
        for _ in range(1, n_blocks):
            layers.append(
                block_class(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    drop_path_rate=self._dpr[self._block_idx],
                )
            )
            self._block_idx += 1

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init the last BN weight in each residual branch so each block
        # starts as identity (Goyal et al. 2017, https://arxiv.org/abs/1706.02677).
        # Matters more for deeper nets (ResNet-50+) where residual variance compounds.
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock1D):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock1D):
                    nn.init.zeros_(m.bn2.weight)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process time series through ResNet.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        # Stem
        x = self.stem(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        if lengths is not None:
            # Calculate approximate reduced length (rough estimate)
            # After stem: /4, after layers 2-4: /8 total
            reduced_lengths = (lengths + 31) // 32  # Conservative estimate
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
            elif self.pool == "last":
                # The ResNet analogue of the transformer's pool="last": read
                # the feature column over the DECISION INSTANT rather than an
                # average over the whole window. Without this the architecture
                # arm is not comparable -- mean pooling is exactly the thing
                # last pooling was worth +0.0179 for replacing.
                idx = (reduced_lengths.clamp(min=1) - 1).clamp(max=max_len - 1)
                x = x.gather(
                    -1, idx.view(-1, 1, 1).expand(-1, x.size(1), 1),
                ).squeeze(-1)
            else:
                raise ValueError(f"Unknown pooling: {self.pool}")
        else:
            if self.pool == "mean":
                x = x.mean(dim=-1)
            elif self.pool == "max":
                x = x.max(dim=-1).values
            elif self.pool == "both":
                x = torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
            elif self.pool == "last":
                x = x[..., -1]
            else:
                raise ValueError(f"Unknown pooling: {self.pool}")

        return self.head(x)
