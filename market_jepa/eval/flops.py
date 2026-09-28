"""Analytic training FLOPs for the transformer backbone plus its head.

THE AXIS OF THE SCALING FIGURE (plots/scaling/supervised_scaling.py) is
training compute, and it has to be a number that means the same thing for
every rung of the ladder and for every month. Wall clock does not: it moves
with the loader, the node and the partition. So compute is COUNTED, from the
architecture and the number of views the optimizer saw.

WHAT IS COUNTED. Multiply-adds as two FLOPs, over every matrix product in a
forward: the patch projection, the information-token projection, QKV and the
attention output projection, the QK^T and AV products, the two MLP
projections, the backbone's output projection and the regression head. A
training step is taken as forward plus a backward of twice the forward, the
convention every scaling paper uses (Kaplan 2020, Hoffmann 2022), so
``training = 3 x forward``.

WHAT IS NOT: normalization, softmax, GELU, biases, residual adds, drop-path,
the optimizer update and the RankNet loss. All are O(tokens x width) or
smaller against the O(tokens x width^2) products above, and leaving them out
is the same convention.

VERIFIED AGAINST torch.utils.flop_counter (tests/test_supervised_scaling.py):
the counter reproduces every term here EXCEPT the attention products, which
it does not see through F.scaled_dot_product_attention on this torch build,
and the difference is that term to four figures. The 6N x tokens rule of
thumb under-counts these models by ~10% because it has no attention term and
carries the embedding parameters.

THE SHAPE COMES FROM THE CHECKPOINT, not from the config. The saved config
records ``n_info_channels: 0`` on every run because the dataset, not the
config, decides how many channels go to the information token (pretrain.py:
"the dataset is the authority"); the state dict carries the truth as the
shape of ``info_proj.weight``. ``shape_from_state_dicts`` reads every width
the count needs off the tensors so a run is billed for the model it actually
trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TransformerShape:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    patch_size: int
    n_patch_features: int      # channels through the patch embedding
    n_info_channels: int       # channels routed to the information token
    seq_len: int               # raw timesteps in a view
    d_embedding: int           # backbone output width
    head_hidden: int           # RegressionHead's hidden width (0 = no head)
    cls: bool = False
    state_token: bool = False

    @property
    def n_patches(self) -> int:
        return math.ceil(self.seq_len / self.patch_size)

    @property
    def n_tokens(self) -> int:
        return (self.n_patches + (1 if self.n_info_channels else 0)
                + int(self.cls) + int(self.state_token))


def forward_flops(s: TransformerShape) -> dict[str, int]:
    """One view's forward, by component, in FLOPs (2 per multiply-add)."""
    d, m, T = s.hidden_size, s.intermediate_size, s.n_tokens
    patch = 2 * s.n_patches * d * s.n_patch_features * s.patch_size
    info = 2 * d * s.n_info_channels if s.n_info_channels else 0
    qkv_out = 8 * T * d * d                # in_proj (3 d^2) + out_proj (d^2)
    attention = 4 * T * T * d              # QK^T and AV, all heads together
    mlp = 4 * T * d * m                    # d -> m -> d
    layers = s.num_hidden_layers * (qkv_out + attention + mlp)
    out_proj = 2 * d * s.d_embedding
    head = (2 * (s.d_embedding * s.head_hidden + s.head_hidden * 1)
            if s.head_hidden else 0)
    total = patch + info + layers + out_proj + head
    return {
        "patch_embed": patch, "info_proj": info,
        "qkv_out": s.num_hidden_layers * qkv_out,
        "attention": s.num_hidden_layers * attention,
        "mlp": s.num_hidden_layers * mlp,
        "out_proj": out_proj, "head": head, "total": total,
    }


def training_flops_per_view(s: TransformerShape) -> int:
    """Forward plus a 2x backward, the standard 3F accounting."""
    return 3 * forward_flops(s)["total"]


def training_flops(s: TransformerShape, views: int) -> int:
    return training_flops_per_view(s) * int(views)


def shape_from_state_dicts(backbone_sd: Mapping, head_sd: Mapping | None,
                           *, seq_len: int) -> TransformerShape:
    """Every width the count needs, read off the saved tensors.

    ``backbone_sd`` is what ``save_checkpoint`` writes to ``backbone.pt`` for
    a supervised run, ``head_sd`` its ``head.pt`` (None for a run with no
    head). ``seq_len`` is the view length, which no tensor records: it is
    ``dataset.augmentations[0].global_seq_len`` in the run's config.
    """
    shp = {k: tuple(v.shape) for k, v in backbone_sd.items()}
    d, n_patch_feats, patch = shp["patch_embed.proj.weight"]
    layers = len({k.split(".")[1] for k in shp if k.startswith("blocks.")})
    if layers == 0:
        raise ValueError("no transformer blocks in the backbone state dict")
    m = shp["blocks.0.mlp.0.weight"][0]
    d_emb = shp["head.weight"][0]
    n_info = shp["info_proj.weight"][1] if "info_proj.weight" in shp else 0
    head_hidden = 0
    if head_sd is not None:
        hshp = {k: tuple(v.shape) for k, v in head_sd.items()}
        if "mlp.0.weight" in hshp:
            head_hidden = hshp["mlp.0.weight"][0]
        else:
            raise ValueError(
                f"unrecognised head state dict keys {sorted(hshp)}: "
                "only the RegressionHead MLP is counted here")
    return TransformerShape(
        hidden_size=d, intermediate_size=m, num_hidden_layers=layers,
        patch_size=patch, n_patch_features=n_patch_feats,
        n_info_channels=n_info, seq_len=int(seq_len), d_embedding=d_emb,
        head_hidden=head_hidden,
        cls="cls_token" in shp, state_token="state_proj.weight" in shp)


def count_params(state_dict: Mapping, *, exclude: tuple[str, ...] = ()) -> int:
    """Elements in every tensor of ``state_dict`` except the named keys.

    A state dict does not distinguish a parameter from a buffer, and the
    recipe's sinusoidal position table is a fixed buffer stored under
    ``position_embeddings`` -- 2048 x width, a tenth of ViT-Tiny. Callers
    that know the table is fixed pass it in ``exclude``; a learned table is a
    parameter and stays in.
    """
    return int(sum(math.prod(v.shape) for k, v in state_dict.items()
                   if k not in exclude))


# ── Scales by name, and FLOPs targets as step counts ─────────────────────────


def shape_for_scale(scale: str, *, seq_len: int = 2048) -> TransformerShape:
    """The recipe's model at a VIT_SCALES rung, read the way the collector
    reads a trained checkpoint: build the modules and read their weights.

    Everything but the width comes from the supervised specialist recipe
    (SupervisedModeConfig): 9 patch channels plus the 8-channel info token,
    patch 8, last-pool, sinusoidal positions, a RegressionHead of the width.
    tests/test_supervised_scaling.py pins this to the per-view cost measured
    from the wave's own checkpoints.
    """
    from market_jepa.eval.heads import RegressionHead
    from market_jepa.modeling.backbones.transformer import (
        TransformerBackbone, TransformerConfig)
    from market_jepa.schemas import VIT_SCALES

    t = VIT_SCALES[scale]
    cfg = TransformerConfig(hidden_size=t["hidden_size"],
                            num_attention_heads=t["num_attention_heads"],
                            intermediate_size=t["intermediate_size"],
                            pos_embed="sinusoidal")
    bb = TransformerBackbone(cfg, n_features=17, d_embedding=t["hidden_size"],
                             pool="last", n_info_channels=8)
    head = RegressionHead(t["hidden_size"])
    return shape_from_state_dicts(bb.state_dict(), head.state_dict(), seq_len=seq_len)


def steps_for_flops(scale: str, targets: Iterable[float], *, views_per_step: int,
                    warmup_steps: int, anneal_frac: float) -> list[int]:
    """Total optimizer steps at which a run of ``scale`` has spent each FLOPs
    target, as checkpoint.anneal_steps: the nearest whole step, in order,
    WITHOUT the targets whose branch point (total minus its cooldown) would
    sit inside the warmup -- a cooldown from a model still ramping is not a
    finished model at that compute, so those dots are dropped for the
    larger scales rather than drawn (user, 2026-09-20).
    """
    per_step = training_flops_per_view(shape_for_scale(scale)) * int(views_per_step)
    out: list[int] = []
    for f in targets:
        T = max(1, int(round(float(f) / per_step)))
        n = max(1, int(round(anneal_frac * T)))
        if T - n < warmup_steps:
            continue
        if out and T <= out[-1]:
            continue
        out.append(T)
    return out


def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="checkpoint.anneal_steps for a scale at FLOPs targets")
    p.add_argument("--scale", required=True)
    p.add_argument("--targets", nargs="+", type=float, required=True)
    p.add_argument("--views-per-step", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, required=True)
    p.add_argument("--anneal-frac", type=float, default=0.1)
    p.add_argument("--per-view", action="store_true",
                   help="print the per-view training FLOPs instead")
    a = p.parse_args(argv)
    if a.per_view:
        print(training_flops_per_view(shape_for_scale(a.scale)))
        return 0
    print(" ".join(str(t) for t in steps_for_flops(
        a.scale, a.targets, views_per_step=a.views_per_step,
        warmup_steps=a.warmup_steps, anneal_frac=a.anneal_frac)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
