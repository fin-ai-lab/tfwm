"""Backbone architectures for time series models."""

from typing import Literal

from .base import TimeSeriesBackbone

from .transformer import (
    TransformerBackbone,
    TransformerConfig,
    PatchEmbedding1D,
    ViTBlock,
)
from .resnet import (
    ResNetBackbone,
    BasicBlock1D,
    BottleneckBlock1D,
)
from .efficientnet import (
    EfficientNetBackbone,
    MBConv1D,
    FusedMBConv1D,
)
from .convnext import (
    ConvNeXtBackbone,
    CNBlock1D,
)
from .inception import (
    InceptionBackbone,
    BasicConv1d,
    InceptionA1D,
    InceptionB1D,
    InceptionC1D,
    InceptionD1D,
    InceptionE1D,
)
from .patchtst import (
    PatchTSTBackbone,
    PatchTSTConfig,
)


# Extra kwargs the transformer branch forwards. Each changes what the model
# IS, so a checkpoint trained with one cannot be scored without it.
_TRANSFORMER_PASSTHROUGH = ("causal", "state_token", "diff_channels",
                            "n_info_channels")


def create_backbone(
    backbone_type: Literal[
        "transformer", "resnet", "efficientnet", "convnext", "inception", "patchtst"
    ],
    n_features: int,
    d_embedding: int = 512,
    config=None,
    pool: str = "cls",
    **backbone_kwargs,
) -> TimeSeriesBackbone:
    """Factory function to create a bare TimeSeriesBackbone.

    Used by both JEPA pre-training and supervised baselines so both always
    use the identical backbone.

    Args:
        backbone_type: Type of backbone
            ('transformer', 'resnet', 'efficientnet', 'convnext', 'inception', 'patchtst').
        n_features: Number of input features per timestep.
        d_embedding: Embedding dimension.
        config: TransformerConfig / PatchTSTConfig for transformer-style backbones.
        pool: Pooling strategy for transformer backbone ('cls', 'mean', 'max').
        **backbone_kwargs: Additional arguments passed to the concrete backbone.

    Returns:
        Configured TimeSeriesBackbone.
    """
    if backbone_type == "transformer":
        if config is None:
            config = TransformerConfig()

        # This branch DROPPED every extra kwarg until 2026-08-22: it listed
        # its arguments and silently discarded the rest, while every other
        # branch forwards **backbone_kwargs. Training never noticed, because
        # hydra instantiates the backbone from the config directly -- only the
        # EVAL path comes through here, so a knob set at train time was simply
        # absent at score time. state_token and diff_channels then failed
        # loudly (the parameter set differs); CAUSAL did not, because causal
        # masking adds no parameters, so the checkpoint loaded and was scored
        # as though it had been trained bidirectionally.
        #
        # A WHITELIST, not **backbone_kwargs. Forwarding everything also
        # un-silences max_seq_len, which callers have been passing and having
        # ignored for long enough that several save/load round trips depend on
        # both sides landing on the 2048 default. Fixing that is a separate
        # change with its own blast radius; this one only restores the knobs
        # that alter what the model IS.
        return TransformerBackbone(
            config=config,
            n_features=n_features,
            d_embedding=d_embedding,
            pool=pool,
            **{k: v for k, v in backbone_kwargs.items()
               if k in _TRANSFORMER_PASSTHROUGH},
        )
    elif backbone_type == "resnet":
        return ResNetBackbone(
            n_features=n_features,
            d_embedding=d_embedding,
            pool=pool,
            **backbone_kwargs,
        )
    elif backbone_type == "efficientnet":
        return EfficientNetBackbone(
            n_features=n_features,
            d_embedding=d_embedding,
            pool=pool if pool != "cls" else "mean",
            **backbone_kwargs,
        )
    elif backbone_type == "convnext":
        return ConvNeXtBackbone(
            n_features=n_features,
            d_embedding=d_embedding,
            pool=pool if pool != "cls" else "mean",
            **backbone_kwargs,
        )
    elif backbone_type == "inception":
        return InceptionBackbone(
            n_features=n_features,
            d_embedding=d_embedding,
            pool=pool if pool != "cls" else "mean",
            **backbone_kwargs,
        )
    elif backbone_type == "patchtst":
        return PatchTSTBackbone(
            n_features=n_features,
            d_embedding=d_embedding,
            config=config,
            pool=pool if pool != "cls" else "mean",
            **backbone_kwargs,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


__all__ = [
    "TimeSeriesBackbone",
    "create_backbone",
    # Transformer
    "TransformerBackbone",
    "TransformerConfig",
    "PatchEmbedding1D",
    "ViTBlock",
    # ResNet
    "ResNetBackbone",
    "BasicBlock1D",
    "BottleneckBlock1D",
    # EfficientNet
    "EfficientNetBackbone",
    "MBConv1D",
    "FusedMBConv1D",
    # ConvNeXt
    "ConvNeXtBackbone",
    "CNBlock1D",
    # Inception
    "InceptionBackbone",
    "BasicConv1d",
    "InceptionA1D",
    "InceptionB1D",
    "InceptionC1D",
    "InceptionD1D",
    "InceptionE1D",
    # PatchTST
    "PatchTSTBackbone",
    "PatchTSTConfig",
]


def backbone_kwargs_from_state_dict(sd: dict, prefix: str = "backbone.") -> dict:
    """Constructor knobs recovered from a saved backbone's own weights.

    THE CONFIG IS NOT A RELIABLE SOURCE FOR THESE. Every mode's
    ``save_pretrained`` records ``n_features`` and the inner TransformerConfig
    and stops there, so ``state_token``, ``diff_channels`` and
    ``n_info_channels`` -- all three of which change the PARAMETER SET -- were
    simply absent at load. A checkpoint trained with an information token
    rebuilt without one puts the per-window columns through the patch
    embedding and fails on a size mismatch, after training has completed.

    Reading them from the weights cannot drift and needs no config migration,
    so checkpoints written before any of these knobs existed load unchanged:
    a missing tensor means the knob was off.

    Returns only the keys it can establish, so callers can splat it over
    whatever the config already supplied.
    """
    # THE PREFIX IS A HINT, NOT A CONTRACT. A mode that nests its encoder --
    # BYOL and DINO keep theirs at "student.backbone." with a second copy under
    # "ema.ema_model.backbone." -- matched neither "backbone." nor the bare
    # name, so this returned {} and every caller splatted nothing: the rebuild
    # then had a 20-channel patch embedding against a 9-channel checkpoint and
    # died in load_state_dict. Finding the prefix in the keys costs one scan and
    # cannot be got wrong by a caller.
    if f"{prefix}patch_embed.proj.weight" not in sd \
            and "patch_embed.proj.weight" not in sd:
        found = sorted(k for k in sd if k.endswith("patch_embed.proj.weight"))
        if found:
            prefix = found[0][: -len("patch_embed.proj.weight")]

    def _get(name):
        return sd.get(f"{prefix}{name}", sd.get(name))

    w = _get("patch_embed.proj.weight")
    if w is None or w.dim() != 3:
        return {}                       # not a patched transformer; nothing to say
    info = _get("info_proj.weight")
    n_info = int(info.shape[1]) if info is not None else 0
    state = _get("state_proj.weight")
    out = {"n_features": int(w.shape[1]) + n_info}
    if n_info:
        out["n_info_channels"] = n_info
    if state is not None:
        out["state_token"] = True
        # diff_channels doubles the patch projection's fan-in, and the state
        # projection sees the same width, so the two agree unless it is on.
        if int(state.shape[1]) == 2 * int(w.shape[1]):
            out["diff_channels"] = True
    return out
