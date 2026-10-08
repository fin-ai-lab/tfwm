"""1D EfficientNet backbone for time series.

Adapted from torchvision's EfficientNet (V1 & V2) to operate on 1D sequences.
Supports B0–B7 (compound-scaled MBConv) and V2-S/M/L (FusedMBConv + MBConv).
"""

import copy
import math
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

def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """Round *v* to the nearest multiple of *divisor* (≥ *min_value*)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv1dNormActivation(nn.Sequential):
    """Conv1d → Norm → (optional) Activation helper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm1d,
        activation_layer: Callable[..., nn.Module] | None = nn.SiLU,
    ):
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False,
            ),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps."""

    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(squeeze_channels, in_channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=-1, keepdim=True)  # (B, C, 1)
        scale = self.activation(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# ---------------------------------------------------------------------------
# Block configs
# ---------------------------------------------------------------------------

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: int | None = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    """EfficientNet V1 block config with compound scaling."""

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Callable[..., nn.Module] | None = None,
    ):
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = int(math.ceil(num_layers * depth_mult))
        if block is None:
            block = MBConv1D
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class FusedMBConvConfig(_MBConvConfig):
    """EfficientNet V2 fused block config."""

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Callable[..., nn.Module] | None = None,
    ):
        if block is None:
            block = FusedMBConv1D
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class MBConv1D(nn.Module):
    """Mobile inverted-bottleneck convolution block (1D)."""

    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        # expand
        if expanded_channels != cnf.input_channels:
            layers.append(Conv1dNormActivation(
                cnf.input_channels, expanded_channels,
                kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU,
            ))

        # depthwise
        layers.append(Conv1dNormActivation(
            expanded_channels, expanded_channels,
            kernel_size=cnf.kernel, stride=cnf.stride,
            groups=expanded_channels, norm_layer=norm_layer, activation_layer=nn.SiLU,
        ))

        # squeeze-and-excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(SqueezeExcitation1D(expanded_channels, squeeze_channels))

        # project
        layers.append(Conv1dNormActivation(
            expanded_channels, cnf.out_channels,
            kernel_size=1, norm_layer=norm_layer, activation_layer=None,
        ))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result


class FusedMBConv1D(nn.Module):
    """Fused mobile inverted-bottleneck convolution block (1D) for EfficientNet V2."""

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(Conv1dNormActivation(
                cnf.input_channels, expanded_channels,
                kernel_size=cnf.kernel, stride=cnf.stride,
                norm_layer=norm_layer, activation_layer=nn.SiLU,
            ))
            # project
            layers.append(Conv1dNormActivation(
                expanded_channels, cnf.out_channels,
                kernel_size=1, norm_layer=norm_layer, activation_layer=None,
            ))
        else:
            layers.append(Conv1dNormActivation(
                cnf.input_channels, cnf.out_channels,
                kernel_size=cnf.kernel, stride=cnf.stride,
                norm_layer=norm_layer, activation_layer=nn.SiLU,
            ))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result


# ---------------------------------------------------------------------------
# Variant configs  (Table 1 of EfficientNet paper / Table 4 of V2 paper)
# ---------------------------------------------------------------------------

# B0–B7 compound scaling coefficients: (width_mult, depth_mult)
_B_SCALING: dict[str, tuple[float, float]] = {
    "b0": (1.0, 1.0),
    "b1": (1.0, 1.1),
    "b2": (1.1, 1.2),
    "b3": (1.2, 1.4),
    "b4": (1.4, 1.8),
    "b5": (1.6, 2.2),
    "b6": (1.8, 2.6),
    "b7": (2.0, 3.1),
}

VARIANT_TYPE = Literal[
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
    "v2_s", "v2_m", "v2_l",
]


def _build_inverted_residual_setting(
    variant: VARIANT_TYPE,
) -> tuple[list[_MBConvConfig], int | None]:
    """Return (block_configs, last_channel) for *variant*."""
    if variant.startswith("b"):
        w, d = _B_SCALING[variant]
        bneck = partial(MBConvConfig, width_mult=w, depth_mult=d)
        return [
            bneck(1, 3, 1, 32, 16, 1),
            bneck(6, 3, 2, 16, 24, 2),
            bneck(6, 5, 2, 24, 40, 2),
            bneck(6, 3, 2, 40, 80, 3),
            bneck(6, 5, 1, 80, 112, 3),
            bneck(6, 5, 2, 112, 192, 4),
            bneck(6, 3, 1, 192, 320, 1),
        ], None

    if variant == "v2_s":
        return [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ], 1280

    if variant == "v2_m":
        return [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ], 1280

    if variant == "v2_l":
        return [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ], 1280

    raise ValueError(f"Unknown EfficientNet variant: {variant}")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class EfficientNetBackbone(TimeSeriesBackbone):
    """1D EfficientNet backbone for time series.

    Supports EfficientNet B0–B7 and V2-S/M/L configurations.

    Args:
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
        variant: EfficientNet variant.
        stochastic_depth_prob: Maximum stochastic depth probability.
        pool: Pooling strategy ('mean', 'max', 'both').
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 512,
        variant: VARIANT_TYPE = "b0",
        stochastic_depth_prob: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__(n_features, d_embedding)
        self.pool = pool

        inverted_residual_setting, last_channel = _build_inverted_residual_setting(variant)
        norm_layer = nn.BatchNorm1d

        layers: list[nn.Module] = []

        # Stem
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv1dNormActivation(
            n_features, firstconv_output_channels,
            kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU,
        ))

        # Inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                # After the first block in a stage, no more downsampling
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))

        # Head conv (expand channels before pooling)
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(Conv1dNormActivation(
            lastconv_input_channels, lastconv_output_channels,
            kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU,
        ))

        self.features = nn.Sequential(*layers)
        self._final_channels = lastconv_output_channels

        # Projection to d_embedding
        pool_factor = 2 if pool == "both" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(self._final_channels * pool_factor),
            nn.Linear(self._final_channels * pool_factor, d_embedding),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process time series through EfficientNet.

        Args:
            x: Input of shape (batch, n_features, length).
            lengths: Optional actual lengths of shape (batch,).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        x = self.features(x)

        # Global pooling
        if lengths is not None:
            # Estimate reduced length: stem stride 2 + each stride-2 stage
            # Conservative: just use ratio of output to input length
            max_len = x.size(-1)
            input_len = lengths.float()
            ratio = max_len / (input_len.max().clamp(min=1).item())
            reduced_lengths = (input_len * ratio).long().clamp(min=1)
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
