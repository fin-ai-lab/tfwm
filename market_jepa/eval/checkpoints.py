"""Load a saved checkpoint back into a model, and find its training config.

Extracted from the retired ``scripts/evaluate_checkpoint.py``: every offline eval needs to
turn a checkpoint directory into a frozen encoder, and that capability had no
business living inside one script — when the IC migration deleted
``eval.discretize`` out from under that script's imports, it took the loader
down with it and broke callers that never touched discretization.

Supports the three on-disk layouts the repo has produced:
  * JEPA / MAE / SSL modes: ``config.json`` + ``model.pt``
  * I-JEPA: ``model.pt`` holding backbone / ema / predictor keys
  * supervised: ``backbone.pt`` (+ ``head.pt``)
"""

from __future__ import annotations

import json
import os
from dataclasses import fields as dc_fields
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from ema_pytorch import EMA

from market_jepa.backbone_config import backbone_block as _shared_backbone_block
from market_jepa.modeling import IJEPA, MAE, LeJEPA, create_backbone
from market_jepa.modeling.backbones import backbone_kwargs_from_state_dict
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.modeling.modes.byol import BYOL
from market_jepa.modeling.modes.cost import CoST
from market_jepa.modeling.modes.cpc import CPC
from market_jepa.modeling.modes.dino import DINO
from market_jepa.modeling.modes.ijepa import IJEPAPredictor
from market_jepa.modeling.modes.pretrained_tsfm import PretrainedTSFM
from market_jepa.modeling.modes.tfc import TFC
from market_jepa.modeling.modes.timemae import TimeMAE
from market_jepa.modeling.modes.ts2vec import TS2Vec

logger = logging.getLogger(__name__)

# KEYED ON THE CLASS NAME, NOT THE DOTTED PATH. Every entry here used to be a
# full module path under market_jepa.modeling.* -- a layout that has not
# existed since the backbones moved to market_jepa.modeling.backbones.*. So
# every lookup missed, and the miss fell through to the "transformer" default:
# a ResNet checkpoint was rebuilt as a ViT and died on a state dict with no
# overlapping keys, AFTER a full training run. The transformer only survived
# by accident, being the default it was wrongly falling back to.
#
# The class name is the stable half of the target. Both the legacy spelling
# (which post_train_ic_eval._load_cfg still writes) and the real one end in
# the same class, so both resolve.
_CLASS_TO_TYPE = {
    "TransformerBackbone": "transformer",
    "CNNBackbone": "cnn",
    "ResNetBackbone": "resnet",
    "InceptionCNN": "inception",
    "InceptionBackbone": "inception",
    "EfficientNetBackbone": "efficientnet",
    "ConvNeXtBackbone": "convnext",
    "PatchTSTBackbone": "patchtst",
}


def _backbone_type_from_target(target: str) -> str:
    """Map a hydra _target_ to a create_backbone type name.

    An ABSENT target still defaults to "transformer" -- plenty of old wandb
    configs carry no target at all, and every one of them is a ViT. A target
    that is PRESENT but unrecognized raises instead of defaulting, because
    that silent fallback is exactly what made a ResNet come back as a ViT.
    """
    if not target:
        return "transformer"
    cls = str(target).rsplit(".", 1)[-1]
    try:
        return _CLASS_TO_TYPE[cls]
    except KeyError:
        raise ValueError(
            f"unknown backbone _target_ {target!r}; add {cls!r} to "
            f"_CLASS_TO_TYPE rather than letting it default to a ViT"
        ) from None

def _head_out_features(state_dict) -> int | None:
    """Width of a saved prediction head, read off its last linear layer.

    Both heads are an MLP ending in Linear(hidden, out), so the LAST 2-D
    weight's row count is the output width: 1 for RegressionHead, n_bins for
    ClassificationHead. Returns None when nothing 2-D is present.
    """
    sd = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict
    out = None
    for k in sd:
        v = sd[k]
        if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 2:
            out = int(v.shape[0])
    return out


def fetch_wandb_config(run_id: str) -> dict:
    """Fetch the training config for a run from wandb.

    Discovers the project automatically from the run's checkpoint_path summary
    field or by searching known projects.
    """
    api = wandb.Api()

    # The run's project is encoded in the wandb.project config field,
    # but we need to find it first.  Search the entity's projects for
    # runs matching this ID.
    run_path = None
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    for project in api.projects(entity):
        try:
            r = api.run(f"{entity}/{project.name}/{run_id}")
            run_path = f"{entity}/{project.name}/{run_id}"
            break
        except (wandb.errors.CommError, ValueError):
            continue

    if run_path is None:
        raise ValueError(
            f"Could not find wandb run {run_id} in any {entity} project"
        )

    run = api.run(run_path)
    logger.info("Found run %s in project %s (name: %s)", run_id, project.name, run.name)
    return run.config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------



def _n_features_from_state_dict(sd: dict, prefix: str = "") -> int | None:
    """The input width the checkpoint was actually trained at, read off its
    own first projection.

    THE CONFIG IS NOT A RELIABLE SOURCE FOR THIS. n_features is not a knob
    anyone sets; it is computed by the dataset from several that are --
    info_norm_stats adds 8, info_window adds 3, diff_channels doubles, and
    risk-factor tickers add a block -- and only the wandb config ever carried
    the resolved number.
    train_meta.json carries the hydra config, which does not, so rebuilding
    from it silently gives 9 and the load fails on a size mismatch after the
    whole training run has completed.

    Reading it from the weights cannot drift: whatever combination of knobs
    produced the width, the width is right there in
    ``patch_embed.proj.weight`` as (hidden, n_features, patch_size).
    """
    n_feat, _ = _widths_from_state_dict(sd, prefix)
    return n_feat


def _widths_from_state_dict(sd: dict, prefix: str = "") -> tuple[int | None, int]:
    """``(n_features, n_info_channels)`` as the checkpoint was actually built.

    The information token takes the per-window constants OUT of the patch
    embedding, so patch_embed is narrower than the input the dataset hands over
    by exactly info_proj's fan-in. Both widths are in the weights, so the split
    needs no config -- which matters, because getting it from a config is how
    the 9-vs-17 mismatch cost a whole sweep.
    """
    w = sd.get(f"{prefix}patch_embed.proj.weight")
    if w is None or w.dim() != 3:
        return None, 0
    info = sd.get(f"{prefix}info_proj.weight")
    n_info = int(info.shape[1]) if info is not None else 0
    return int(w.shape[1]) + n_info, n_info


def _transformer_config_from(inner: dict | None) -> TransformerConfig:
    """Rebuild a saved TransformerConfig WITHOUT enumerating its fields.

    This used to be written out field by field, in two places, and every knob
    the dataclass grew after that was silently dropped at load time while
    training honoured it. The (since retired) recency prior is what caught it:
    60 runs trained with the bias, saved a `recency_slope_raw` parameter, and
    then died in the post-training scorer with

        Unexpected key(s) in state_dict: "recency_slope_raw"

    which is the GOOD outcome -- the bias added a parameter, so strict loading
    failed loudly. A knob that adds no parameter (as `causal` once did) would
    have loaded clean and been scored as though it had never been set.

    So: take whatever the checkpoint saved, keep the keys the dataclass
    actually declares, and let the dataclass supply its own defaults for the
    rest. Unknown keys (a `_target_` from hydra, a field from a newer commit)
    are dropped rather than raising, because an old checkpoint must stay
    loadable by newer code.
    """
    valid = {f.name for f in dc_fields(TransformerConfig)}
    return TransformerConfig(**{k: v for k, v in (inner or {}).items()
                                if k in valid})


def _backbone_block(cfg: dict) -> dict:
    """The backbone config the mode ACTUALLY trained.

    Delegates to market_jepa.backbone_config, which is the single definition of
    the rule. Kept as a module-local name because three call sites here and the
    eval tests refer to it.
    """
    return _shared_backbone_block(cfg)


def load_model(checkpoint_dir: str, cfg: dict, device: torch.device) -> nn.Module:
    """Load a model from a checkpoint directory using wandb config.

    Supports:
      - LeJEPA:     config.json + model.pt
      - I-JEPA:     model.pt (state dict with backbone/ema/predictor keys)
      - Supervised:  backbone.pt (+ head.pt, or heads.pt for a multihead)
    """
    ckpt_path = Path(checkpoint_dir)
    mode_target = cfg.get("mode", {}).get("_target_", "")

    # This resolution used to be duplicated: one copy located the config and
    # built a backbone nothing read, the other picked the backbone TYPE but
    # indexed cfg["backbone"] raw. Each half was missing what the other had —
    # the first hardcoded "transformer", the second died on runs that logged
    # cfg.backbone as a bare parameter count — and a local re-import of
    # TransformerConfig between them made the name function-local, so the
    # loader raised UnboundLocalError before reaching either bug. Merged.
    # THE SAME ONE RULE THE TRAINER USES -- see _backbone_block. This was a
    # second copy of the `"IJEPA" in mode_target` test, and it broke the moment
    # modes started OWNING their backbone and sweeps stopped pinning one at the
    # top level: cfg["backbone"] still resolves, but to the COMPOSED DEFAULT
    # (pool="cls", learned positions) rather than to what the mode trained.
    #
    # It fails loudly, which is the only mercy here -- a supervised checkpoint
    # is last/rope, so the loader built a cls-pooled model with 2049 positions
    # and nn.MultiheadAttention's in_proj against a checkpoint carrying 2048
    # positions and fused qkv, and every post-train IC eval died on
    # load_state_dict. Silent would have been far worse: causal is read from
    # this same block and does NOT change the parameter set.
    backbone_cfg = _backbone_block(cfg)
    backbone_type = _backbone_type_from_target(backbone_cfg.get("_target_", ""))
    d_embedding = backbone_cfg.get("d_embedding", 384)
    pool = backbone_cfg.get("pool", "cls")
    # Causal masking does NOT change the parameter set -- a causal checkpoint
    # loads cleanly into a bidirectional backbone and scores as if it had been
    # trained bidirectionally. pool at least fails loudly (no cls_token, and
    # n_pos differs); this one is silent, so it has to be read, not assumed.
    causal = bool(backbone_cfg.get("causal", False))
    # Unlike causal, this one DOES change the parameter set (a state_proj and
    # one more position), so a mismatch fails loudly. Read anyway: a loud
    # failure at scoring time is still a failed run.
    state_token = bool(backbone_cfg.get("state_token", False))
    # Doubles the patch projection's input width, so a mismatch fails loudly.
    diff_channels = bool(backbone_cfg.get("diff_channels", False))
    # Moves channels from the patch embedding to the info token, so it changes
    # BOTH parameter shapes. Read from the checkpoint below where possible;
    # this is the fallback for configs that carry it and weights that do not.
    n_info_channels = int(backbone_cfg.get("n_info_channels", 0) or 0)

    # Build TransformerConfig if needed
    transformer_config = None
    if backbone_type == "transformer":
        transformer_config = _transformer_config_from(
            backbone_cfg.get("config", {}))

    # Case 1: JEPA / MAE checkpoint directory (config.json + model.pt).
    # Dispatch on the "class" field written by save_pretrained; fall back to
    # LeJEPA for older checkpoints that predate the field.
    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        with open(ckpt_path / "config.json") as f:
            ckpt_config = json.load(f)
        model_class = ckpt_config.get("class", "LeJEPA")
        logger.info("Loading %s checkpoint from %s", model_class, ckpt_path)
        registry = {
            "MAE": MAE,
            "DINO": DINO,
            "BYOL": BYOL,
            "CPC": CPC,
            "TS2Vec": TS2Vec,
            "CoST": CoST,
            "TFC": TFC,
            "TimeMAE": TimeMAE,
            "PretrainedTSFM": PretrainedTSFM,
        }
        model = registry.get(model_class, LeJEPA).from_pretrained(str(ckpt_path))
        model.to(device).eval()
        return model

    # Case 2: I-JEPA (model.pt with backbone/ema/predictor keys)
    if (ckpt_path / "model.pt").exists() and "IJEPA" in mode_target:
        logger.info("Loading I-JEPA checkpoint from %s", ckpt_path)
        mode_cfg = cfg["mode"]

        # NameError until 2026-08-19: this branch referenced a `backbone` that
        # was never built, so no I-JEPA checkpoint could be loaded at all. The
        # shape is the one resolved above (cfg["mode"]["backbone"] for I-JEPA),
        # constructed exactly as the supervised branch and
        # build_untrained_encoder construct theirs.
        ijepa_sd = torch.load(
            ckpt_path / "model.pt", map_location="cpu", weights_only=False,
        )
        n_feat_ck, n_info_ck = _widths_from_state_dict(ijepa_sd, "backbone.")
        backbone = create_backbone(
            backbone_type=backbone_type,
            n_features=(n_feat_ck or cfg.get("n_features", 9)),
            d_embedding=d_embedding,
            config=transformer_config,
            pool=pool,
            **({"causal": True} if causal else {}),
            **({"state_token": True} if state_token else {}),
            **({"diff_channels": True} if diff_channels else {}),
            **({"n_info_channels": (n_info_ck or n_info_channels)}
               if (n_info_ck or n_info_channels) else {}),
        )
        model = IJEPA(
            backbone=backbone,
            pred_depth=mode_cfg.get("pred_depth", 6),
            pred_emb_dim=mode_cfg.get("pred_emb_dim", 192),
            pred_num_heads=mode_cfg.get("pred_num_heads"),
            ema_start=mode_cfg.get("ema_start", 0.996),
            ema_end=mode_cfg.get("ema_end", 1.0),
            n_targets=mode_cfg.get("n_targets", 4),
            target_scale=mode_cfg.get("target_scale"),
            context_crop_max=mode_cfg.get("context_crop_max", 0.15),
            loss_fn=mode_cfg.get("loss_fn", "smooth_l1"),
        )

        model.load_state_dict(ijepa_sd)
        model.to(device).eval()
        return model

    # Case 3: Supervised (backbone.pt + head.pt)
    if (ckpt_path / "backbone.pt").exists():
        logger.info("Loading supervised checkpoint from %s", ckpt_path)
        from market_jepa.modeling.modes.supervised import (
            SupervisedModel,
        )

        backbone_sd = torch.load(
            ckpt_path / "backbone.pt", map_location="cpu", weights_only=False,
        )
        n_feat_ck, n_info_ck = _widths_from_state_dict(backbone_sd)
        backbone = create_backbone(
            backbone_type=backbone_type,
            n_features=(n_feat_ck or cfg.get("n_features", 9)),
            d_embedding=d_embedding,
            config=transformer_config,
            pool=pool,
            **({"causal": True} if causal else {}),
            **({"state_token": True} if state_token else {}),
            **({"diff_channels": True} if diff_channels else {}),
            **({"n_info_channels": (n_info_ck or n_info_channels)}
               if (n_info_ck or n_info_channels) else {}),
        )
        backbone.load_state_dict(backbone_sd)

        # EVERY HEAD IS SCALAR since the binned family was retired
        # (2026-09-07), so the width no longer has to be inferred: a k-way head
        # can only come from a checkpoint predating the retirement, and those
        # do not load anyway -- the recency prior went at the same time and
        # took their state dicts with it. What remains is read off head.pt so
        # a size mismatch fails loudly rather than scoring a stale checkpoint.
        mode_cfg_d = cfg.get("mode", {}) or {}

        # MULTIHEAD CHECKPOINTS SAVE ``heads.pt``, NOT ``head.pt`` (see
        # training/utils.py save), and their config carries ``tasks`` rather
        # than ``task``. Until 2026-08-29 this branch had neither fact: every
        # MultiTaskSupervisedModel checkpoint loaded as a SINGLE-task
        # SupervisedModel on the ``or "return_900"`` default below, found no
        # head.pt, and returned a RANDOMLY INITIALIZED head -- which the IC
        # scorer then dutifully scored and wrote to xs_ic.json as
        # ``head:return_900``. Every multihead head number produced before that
        # date is noise from an untrained head, on the one task the fallback
        # happened to name. Tasks are read off the state dict, not the config,
        # because the state dict is what the weights actually are.
        heads_path = ckpt_path / "heads.pt"
        if heads_path.exists() and not (ckpt_path / "head.pt").exists():
            from market_jepa.modeling.modes.supervised import (
                MultiTaskSupervisedModel,
            )

            heads_sd = torch.load(
                heads_path, map_location="cpu", weights_only=False)
            tasks = sorted({k.split(".", 1)[0] for k in heads_sd})
            cfg_tasks = list(mode_cfg_d.get("tasks") or [])
            if cfg_tasks and sorted(cfg_tasks) != tasks:
                raise ValueError(
                    f"{ckpt_path}: heads.pt holds {tasks} but the config says "
                    f"{sorted(cfg_tasks)} — refusing to guess which head "
                    "predicts what.")
            # Width off ONE head: the heads share a loss, so they share a
            # shape, and a per-task disagreement would fail the load below.
            first = {k.split(".", 1)[1]: v for k, v in heads_sd.items()
                     if k.startswith(tasks[0] + ".")}
            n_out = _head_out_features(first)
            if n_out is not None and n_out > 1:
                raise ValueError(
                    f"{ckpt_path}: heads.pt holds a {n_out}-way head. The "
                    "binned loss family was retired on 2026-09-07 and cannot "
                    "be rebuilt.")
            model = MultiTaskSupervisedModel(
                backbone, tasks=(cfg_tasks or tasks),
                loss_fn=mode_cfg_d.get("loss_fn", "mse"),
            )
            model.heads.load_state_dict(heads_sd)
            model.to(device).eval()
            return model

        task = mode_cfg_d.get("task") or "return_900"
        head_path = ckpt_path / "head.pt"
        head_sd = (torch.load(head_path, map_location="cpu", weights_only=False)
                   if head_path.exists() else None)
        n_out = _head_out_features(head_sd) if head_sd is not None else None
        if n_out is not None and n_out > 1:
            raise ValueError(
                f"{ckpt_path}: head.pt holds a {n_out}-way head. The binned "
                "loss family was retired on 2026-09-07 and cannot be rebuilt.")
        model = SupervisedModel(
            backbone, task=task, loss_fn=mode_cfg_d.get("loss_fn", "mse"),
        )

        # A FINETUNE SAVES A DIFFERENT HEAD. mode.init_head_from swaps the
        # plain MLP for a SkipRegressionHead, whose skip carries the ridge
        # probe; its state dict has two extra keys and would otherwise fail the
        # strict load below. The saved KEYS decide, not the config: this runs
        # against checkpoints whose mode block may be absent or stale, and a
        # head rebuilt as the wrong type is exactly the failure that shipped
        # untrained heads as results for months (see the branch after this).
        if head_sd is not None and any(k.startswith("skip.") for k in head_sd):
            from market_jepa.eval.heads import SkipRegressionHead
            model.head = SkipRegressionHead(backbone.d_embedding)

        # The TRAINED head, when it was saved. This branch used to return a
        # randomly initialized one and rely on callers only ever touching
        # encode() — but the supervised arm of the IC eval scores with the head
        # itself, and a random head silently reports noise as a result.
        if head_sd is not None:
            model.head.load_state_dict(head_sd)
        else:
            # AND SAY SO ON THE MODEL. A warning in a SLURM log is not a
            # guard: the head readout only ever checked that ``model.head``
            # was not None, so a random head scored and shipped as a result
            # for months. Consumers read this flag; see head_readout.
            model._head_is_random = True
            logger.warning(
                "%s has no head.pt — the head is randomly initialized and only "
                "encode() is meaningful.", ckpt_path,
            )
        model.to(device).eval()
        return model

    raise FileNotFoundError(
        f"No valid checkpoint found at {checkpoint_dir}. "
        f"Expected config.json+model.pt (LeJEPA), model.pt (I-JEPA), "
        f"or backbone.pt+head.pt (supervised)."
    )


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def dataset_flag(cfg, *names: str) -> bool:
    """A boolean off this checkpoint's ``dataset`` config, under any of ``names``.

    SEVERAL NAMES, because the key was renamed on 2026-08-25 and a checkpoint
    is a historical record: norm_stats_channels became info_norm_stats when the
    stats stopped being broadcast as channels, and time_info became info_window.
    A run trained before the rename stamped the old key and must still score
    the way it trained.

    DEFAULT FALSE, and it stays false even though the config defaults are now
    true. save_train_meta writes these keys only when they are ON, so absence
    is a real signal here -- reading the live default instead would tell every
    pre-2026-08-25 checkpoint that it had an information token it never saw.

    LIVES HERE, not in the scorer, because build_untrained_encoder derives the
    floor's width from it and market_jepa cannot import from scripts/.
    xs_ic_eval re-exports it so there is one reader, not two that drift.
    """
    if cfg is None:
        return False
    get = cfg.get if isinstance(cfg, dict) else (
        lambda k, d=None: getattr(cfg, k, d))
    ds = get("dataset", None) or {}
    dget = ds.get if isinstance(ds, dict) else (
        lambda k, d=None: getattr(ds, k, d))
    return any(bool(dget(n, False)) for n in names)


def architecture_signature(cfg: dict) -> str:
    """Stable id for the ENCODER architecture a config describes.

    The random-init baseline depends on the architecture and nothing else —
    not on the mode, not on the objective, not on which checkpoint happened to
    reference it. Two runs sharing this signature share a baseline, so it is
    embedded once instead of once per run.
    """
    backbone_cfg = _backbone_block(cfg)
    inner = backbone_cfg.get("config", {}) or {}
    parts = [
        _backbone_type_from_target(backbone_cfg.get("_target_", "")),
        str(backbone_cfg.get("d_embedding", 384)),
        str(backbone_cfg.get("pool", "cls")),
        str(cfg.get("n_features", 9)),
        str(inner.get("hidden_size", 384)),
        str(inner.get("num_hidden_layers", 12)),
        str(inner.get("num_attention_heads", 6)),
        str(inner.get("intermediate_size", 1536)),
        str(inner.get("patch_size", 8)),
        str(inner.get("pos_embed", "learned")),
        str(inner.get("cls_pos", "own")),
    ]
    return "-".join(parts)


def build_untrained_encoder(cfg: dict, device: torch.device, seed: int = 0,
                            pool: str | None = None,
                            n_info_channels: int | None = None):
    """Build the RANDOM-INIT baseline encoder for the architecture in ``cfg``.

    THE FLOOR GETS THE INFORMATION TOKEN, like every model it floors, and it
    is DERIVED rather than demanded. ``n_info_channels`` defaults to the width
    this checkpoint's own ``dataset`` block implies -- the same two flags, read
    through the same ``dataset_flag``, that decide the panel's width in
    ``xs_ic_eval.panel_kwargs_for``. Pass it only to override.

    It used to default to nothing and the caller had to remember: the backbone
    block cannot answer the question (pretrain.py takes the width from the
    DATASET, so train_meta.json records the unset config value 0 while the
    weights carry a real ``info_proj``), so an unset argument silently built
    the floor 0.8M parameters lighter than the models it floors and blind to
    the normalization stats and window metadata they see.
    ``architecture_signature`` reports the SAME string for both, so nothing
    downstream could notice. Deriving it makes the floor right by default and
    leaves the override for the caller that genuinely knows better -- the
    scorer, which takes the width from the PANEL it is about to embed.

    Historical checkpoints stay correct because ``dataset_flag`` defaults
    FALSE: ``save_train_meta`` writes these keys only when they are on, so a
    pre-2026-08-25 run derives width 0, which is what it trained with.

    This is the subtrahend of the paper's delta IC: every reported number is
    quoted against the IC that the SAME architecture, with the SAME probe, on
    the SAME synchronized panel, reaches with untrained weights. Without it a
    method can look successful while doing nothing a random projection does
    not already do — which on this data is not hypothetical (see the
    manipulation and firm-identity analyses, where the untrained floor matches
    trained encoders outright).

    Returned as a SupervisedModel wrapper purely so ``encode()`` exists with
    the signature ``embed_month`` expects; its head is never touched.

    ``seed`` selects the draw. Report several: a single random init is one
    sample of the floor, not the floor.

    THE READOUT COMES FROM ``cfg``, and a floor read at a different token than
    the model it is subtracted from is not that model's floor. The scorer
    therefore pins the config's pool before calling this
    (``xs_ic_eval.PREDICT_POOL`` = "last"), so both arms are floored at the
    last token: LeJEPA trains its invariance loss on the mean but its reported
    prediction comes from a last-token probe, and the supervised arm trains
    "last" outright.

    THEY DO NOT THEREBY SHARE A FLOOR, and an earlier version of this note
    claimed they would. ``architecture_signature`` keys on the position
    encoding as well as the pool, and the two arms differ there -- SSL is
    sinusoidal, the supervised family RoPE -- so pinning the pool aligns the
    TOKEN they are read at and nothing more. Each arm still computes its own
    floor, which is correct: a RoPE floor is not a sinusoidal model's floor.

    ``pool`` overrides it for the latent suite, which reads every encoder at
    the MEAN regardless of what it trained with -- see ``load_backbone``'s note
    and ``panel_lib.LATENT_POOL``. The rule is the same in both places: a floor
    is read the way the models it floors are read.
    """
    from market_jepa.modeling.modes.supervised import SupervisedModel

    from market_jepa.augmentations import info_channel_width

    backbone_cfg = _backbone_block(cfg)
    backbone_type = _backbone_type_from_target(backbone_cfg.get("_target_", ""))
    inner = backbone_cfg.get("config", {}) or {}

    transformer_config = None
    if backbone_type == "transformer":
        transformer_config = _transformer_config_from(inner)

    if n_info_channels is None:
        n_info_channels = info_channel_width(
            info_norm_stats=dataset_flag(
                cfg, "info_norm_stats", "norm_stats_channels"),
            info_window=dataset_flag(cfg, "info_window", "time_info"),
        )
    n_info_channels = int(n_info_channels or 0)

    # ONLY THE TRANSFORMER HAS AN INFORMATION TOKEN to route them into. The
    # other backbones forward **backbone_kwargs straight to a constructor that
    # does not take the argument, so this is passed by name rather than always.
    # n_features stays the TOTAL width either way: the eleven are in the panel
    # whoever consumes them, so a non-transformer floor reads them as eleven
    # more series, exactly as a non-transformer model would.
    info_kw = ({"n_info_channels": n_info_channels}
               if backbone_type == "transformer" and n_info_channels else {})

    # Seed immediately before construction: the draw IS the baseline, so it
    # must not depend on whatever consumed the global RNG earlier in the run.
    torch.manual_seed(seed)
    backbone = create_backbone(
        backbone_type=backbone_type,
        # n_features is the TOTAL width and n_info_channels a subset of it,
        # so the info token widens the input rather than reallocating it: a
        # trained checkpoint here is (9 real + 11 info) = 20, which is what
        # _widths_from_state_dict reads back off the weights.
        n_features=(cfg.get("n_features") or 9) + n_info_channels,
        **info_kw,
        d_embedding=backbone_cfg.get("d_embedding", 384),
        config=transformer_config,
        pool=pool if pool is not None else backbone_cfg.get("pool", "cls"),
    )
    model = SupervisedModel(backbone, task="return_900")
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Bare-backbone loading (rescued from the deleted mass_eval pipeline)
# ---------------------------------------------------------------------------
#
# ``load_model`` above rebuilds the whole TRAINING model, which is what the IC
# eval wants. The geometry and factor-structure analyses want the opposite: the
# bare encoder as an ``nn.Module`` you can call as ``backbone(x, lengths)``,
# with no mode wrapper, no head, and no wandb round-trip — they iterate over
# hundreds of (checkpoint, month) pairs and only ever read embeddings.
#
# That capability lived in the retired ``mass_eval_world_model.py``
# and was deleted with the delta-AUC pipeline (8b508a8). It was collateral
# damage, not a deliberate removal: nothing here computes an AUC or bins a
# target. Every consumer of it broke at IMPORT time, so the geometry and
# factor-structure analyses (then ``plots/embedding_geometry`` and
# ``plots/factor_structure``, since consolidated into ``plots/latent_eval``)
# stopped running entirely. Restored verbatim, minus the AUC machinery, and
# put where the rest of the checkpoint loading lives.

N_FEATURES = 9
GLOBAL_SEQ_LEN = 2048

_DATE_SUFFIX_RE = re.compile(r".+-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})$")

# Modes whose encoder is not a plain backbone: their embedding comes from the
# mode's own ``encode()`` (a projection head, a masked readout, a frequency
# branch), so a raw backbone forward would not be the representation the method
# actually proposes.
ENCODE_MODE_CLASSES = frozenset({"TS2Vec", "CoST", "TFC", "TimeMAE"})


def parse_project_dates(project_name: str) -> tuple[str, str]:
    """``...-2023-01-01-2023-01-31`` -> the two dates; a default when absent."""
    m = _DATE_SUFFIX_RE.match(project_name)
    if not m:
        return "2023-01-01", "2023-01-31"
    return m.group(1), m.group(2)


def _backbone_kind_from_project(project_name: str) -> str:
    p = project_name.lower()
    if p.startswith("supervised-resnet") or "resnet" in p:
        return "resnet"
    if p.startswith("supervised-effnet") or "effnet" in p or "efficientnet" in p:
        return "efficientnet"
    return "transformer"


def _extract_backbone_state(state: dict) -> dict:
    """Pull the bare backbone state-dict out of a saved model.pt.

    LeJEPA / IJEPA / MAE / supervised store the backbone at the top level
    (``backbone.*``); DINO and BYOL nest it under ``ema.ema_model.backbone.*``
    (the EMA teacher — the canonical eval encoder per ``encode()`` in
    market_jepa/modeling/modes/{dino,byol}.py). The student backbone at
    ``student.backbone.*`` is the last-resort fallback for ckpts saved
    before the EMA teacher was wired up.
    """
    for prefix in ("backbone.", "ema.ema_model.backbone.", "student.backbone."):
        bb_state = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        if bb_state:
            return bb_state
    return {}


def load_backbone(ckpt_dir, run_dir=None, project_name: str = "",
                  pool: str | None = None):
    """Build a bare backbone ``nn.Module`` (on CPU) and load ``ckpt_dir``'s weights.

    Tries config.json + model.pt (JEPA), then bare model.pt, then backbone.pt
    (supervised; no config.json shipped, so the shape is inferred from the
    project name).

    ``pool`` OVERRIDES THE READOUT THE CHECKPOINT TRAINED WITH, and the latent
    evaluations use it. What a training readout selects is a prediction head's
    vantage point, not the representation: the supervised arm trains
    pool="last", so reading its latent at that token would make every
    latent-structure number a statement about ONE patch of the day. The latent
    suite asks what is in the representation of the whole view, which is the
    mean over patch tokens.

    It also removes a confound from every supervised-vs-SSL latent comparison.
    LeJEPA is mean-pooled by construction, so without this the two arms would
    differ in the objective AND in where the embedding is read, and the tables
    could not separate them.

    SAFE BECAUSE POOLING IS A READOUT, not a parameter: it is applied after the
    last block, so no weight depends on it and nothing about the load changes.
    The one case to know is a cls-trained checkpoint, where the CLS token
    survives into a mean over 257 tokens rather than 256; nothing currently
    trains that way, and forward() drops the information token before pooling
    either way.
    """
    ckpt_dir = Path(ckpt_dir)
    run_dir = Path(run_dir) if run_dir is not None else ckpt_dir
    if pool is not None:
        built = load_backbone(ckpt_dir, run_dir, project_name)
        built.pool = pool
        return built
    model_pt = ckpt_dir / "model.pt"
    backbone_pt = ckpt_dir / "backbone.pt"
    config_json = ckpt_dir / "config.json"
    if not config_json.exists() and run_dir != ckpt_dir:
        config_json = run_dir / "config.json"

    if config_json.exists() and model_pt.exists():
        with open(config_json) as f:
            meta = json.load(f)
        bb_type = meta.get("backbone_type", "transformer")
        cfg_key = {
            "transformer": "backbone_config",
            "resnet": "resnet_config",
            "efficientnet": "efficientnet_config",
        }[bb_type]
        bb_cfg = meta.get(cfg_key) or meta.get("backbone_config") or {}
        kwargs = dict(
            backbone_type=bb_type,
            n_features=meta.get("n_features", N_FEATURES),
            d_embedding=meta.get("d_embedding", 384),
            pool=meta.get("pool", "mean"),
        )
        if bb_type == "transformer":
            kwargs["config"] = TransformerConfig(**bb_cfg)
        else:
            kwargs.update(bb_cfg)
        state = torch.load(model_pt, map_location="cpu", weights_only=False)
        bb_state = _extract_backbone_state(state)
        if not bb_state:
            raise RuntimeError(f"No backbone weights in {model_pt}")
        # THE WEIGHTS OUTRANK config.json ON WIDTH. An information-token run
        # writes n_features as the DATASET width (e.g. 20) while the patch
        # embedding only sees the non-constant channels (9) and info_proj
        # takes the rest -- so building from the config alone yields a
        # 20-channel patch_embed with no info_proj and dies on
        #     Unexpected key(s): "info_proj.weight"
        #     size mismatch patch_embed.proj.weight: [384,9,8] vs [384,20,8]
        # which is exactly what every LeJEPA pairing checkpoint did here.
        # backbone_kwargs_from_state_dict recovers the split (and state_token /
        # diff_channels) from the tensors, where it cannot drift; the mode
        # classes have always loaded this way, and this is the one loader that
        # did not. A checkpoint with no info token is unaffected: the helper
        # returns just the n_features it can already see.
        if bb_type == "transformer":
            kwargs.update(backbone_kwargs_from_state_dict(bb_state, prefix=""))
        backbone = create_backbone(**kwargs)
        backbone.load_state_dict(bb_state)
        return backbone

    if model_pt.exists():
        # NO config.json, SO THE ARCHITECTURE COMES FROM train_meta + THE
        # WEIGHTS -- exactly what the backbone.pt branch below already does.
        # Building from TransformerConfig() DEFAULTS instead is wrong for any
        # run that moved a backbone knob, and the defaults have moved away from
        # what the SSL family trains: pos_embed defaults to "learned" against
        # their "sinusoidal", which is a different PARAMETER SET, and an
        # information-token run needs the 9/11 channel split that only the
        # tensors carry.
        #
        # WHICH CHECKPOINTS LAND HERE: every mode writes a config.json in its
        # save_pretrained -- except IJEPA, which defines none at all, so all 27
        # of the 2026-09-15 six-month I-JEPA checkpoints arrive with model.pt
        # and train_meta.json and nothing else. Before this they died on
        #     Unexpected key(s) in state_dict: "info_proj.weight"
        # with the arm silently absent from the latent table.
        state = torch.load(model_pt, map_location="cpu", weights_only=False)
        bb_state = _extract_backbone_state(state)
        if not bb_state:
            raise RuntimeError(f"No backbone weights in {model_pt}")
        meta_f = ckpt_dir / "train_meta.json"
        if not meta_f.is_file() and run_dir != ckpt_dir:
            meta_f = run_dir / "train_meta.json"
        bb_meta = {}
        if meta_f.is_file():
            try:
                cfg_meta = json.loads(meta_f.read_text()).get("config") or {}
                # THE MODE'S OWN BLOCK. The top-level one is not absent -- it
                # carries pos_embed=None, which resolves to "learned".
                bb_meta = _shared_backbone_block(cfg_meta) or {}
            except (json.JSONDecodeError, AttributeError):
                bb_meta = {}
        kwargs = dict(
            backbone_type="transformer",
            n_features=N_FEATURES,
            d_embedding=bb_meta.get("d_embedding", 384),
            config=_transformer_config_from(bb_meta.get("config")),
            pool=bb_meta.get("pool") or "mean",
        )
        for k in ("n_info_channels", "state_token", "diff_channels",
                  "causal", "max_seq_len"):
            if bb_meta.get(k) is not None:
                kwargs[k] = bb_meta[k]
        # The weights outrank both: n_info_channels is recorded as 0 in the
        # meta of runs that trained WITH an information token.
        kwargs.update(backbone_kwargs_from_state_dict(bb_state, prefix=""))
        backbone = create_backbone(**kwargs)
        backbone.load_state_dict(bb_state)
        return backbone

    if backbone_pt.exists():
        state = torch.load(backbone_pt, map_location="cpu", weights_only=False)
        # SUPERVISED CHECKPOINTS SHIP NO config.json, and this branch used to
        # build from TransformerConfig() DEFAULTS + N_FEATURES, which is wrong
        # for any run that moved a backbone knob: the (retired) recency-window
        # sweep trained at W=8 against a default of 16, and every one of those
        # runs carries an information token, so the defaults produced a
        # 9-channel patch
        # embedding with no info_proj and died on
        #     Unexpected key(s): "info_proj.weight"
        # They DO ship the whole thing in train_meta.json under
        # config.backbone, so read it, then let the weights settle the widths.
        meta_f = ckpt_dir / "train_meta.json"
        if not meta_f.is_file() and run_dir != ckpt_dir:
            meta_f = run_dir / "train_meta.json"
        # THE MODE'S OWN BLOCK, via the one resolver. Reading the top-level
        # block here is not a no-op: it is not absent, it carries pool=None and
        # pos_embed=None, which the backbone resolves to "cls" and "learned"
        # (transformer.py). So a last/rope checkpoint got a cls-pooled
        # 2049-position model and load_state_dict died -- the exact failure
        # 5187622 fixed in load_model, in a sibling function it did not touch.
        bb_meta = {}
        if meta_f.is_file():
            try:
                cfg_meta = json.loads(meta_f.read_text()).get("config") or {}
                bb_meta = _shared_backbone_block(cfg_meta) or {}
            except (json.JSONDecodeError, AttributeError):
                bb_meta = {}
        if bb_meta:
            kwargs = dict(
                backbone_type="transformer",
                n_features=N_FEATURES,
                d_embedding=bb_meta.get("d_embedding", 384),
                config=_transformer_config_from(bb_meta.get("config")),
                # `or`, NOT a .get default: a block that carries an
                # explicit pool=None would otherwise pass None straight
                # through, and the backbone resolves that to "cls"
                # (transformer.py) -- the very cls/last mismatch this
                # branch was rewritten to stop. Every mode block sets
                # pool today, so this is the door, not the bug.
                pool=bb_meta.get("pool") or "cls",
            )
            for k in ("n_info_channels", "state_token", "diff_channels",
                      "causal", "max_seq_len"):
                if bb_meta.get(k) is not None:
                    kwargs[k] = bb_meta[k]
            kwargs.update(backbone_kwargs_from_state_dict(state, prefix=""))
            backbone = create_backbone(**kwargs)
            backbone.load_state_dict(state)
            return backbone

        kind = _backbone_kind_from_project(project_name)
        if kind == "resnet":
            backbone = create_backbone(
                backbone_type="resnet", n_features=N_FEATURES, d_embedding=512,
                variant="50", pool="mean",
            )
        elif kind == "efficientnet":
            backbone = create_backbone(
                backbone_type="efficientnet", n_features=N_FEATURES,
                d_embedding=512, variant="b4", pool="mean",
            )
        else:
            backbone = create_backbone(
                backbone_type="transformer", n_features=N_FEATURES,
                d_embedding=384, config=TransformerConfig(), pool="cls",
            )
        backbone.load_state_dict(state)
        return backbone

    raise FileNotFoundError(f"No recognizable checkpoint files in {ckpt_dir}")


class EncodeAdapter(nn.Module):
    """``forward(x, lengths) -> (B, d)`` over a TrainingModel's own encode()."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def n_features(self) -> int | None:
        """Input channels the wrapped encoder was BUILT for, or None.

        Callers narrow the grid window to this before a forward: the grid
        carries the information-token channels as trailing constants, and an
        encoder trained without them reads the identical data channels by
        dropping the tail (build_fullday_embs._narrow, and every other
        consumer that has to feed one grid to encoders of both generations).

        Without this the adapter simply had no such attribute, ``getattr``
        returned None, the narrow became a no-op and the four ENCODE_MODE
        classes -- whose encode() reaches the backbone's forward_patches with
        the raw tensor -- died on a conv1d channel mismatch (weight 9, input
        20). The bare-backbone path never hit it because a backbone carries
        ``n_features`` itself; only the wrapper hid it.
        """
        return getattr(getattr(self.model, "backbone", None),
                       "n_features", None)

    def forward(self, x, lengths=None):
        return self.model.encode(x, lengths)["embeddings"][:, 0, :]


def load_encoder(ckpt_dir, project_name: str = "",
                 pool: str | None = None) -> nn.Module:
    """Mode-aware encoder: a bare backbone where that IS the representation.

    The dispatch matters for the TS-SSL baselines, whose published embedding is
    their ``encode()`` and not the backbone's pooled output — scoring the
    backbone instead would measure a method nobody proposed.

    ``pool`` IS THE LATENT SUITE'S READOUT OVERRIDE and reaches the bare-backbone
    branch only -- see ``load_backbone`` and ``panel_lib.LATENT_POOL``. It has to
    be threaded through HERE, not just there: every encoder in
    ``industry_nn_sweep.MODEL_ORDER`` is manifest-resolved, and a manifest row
    is loaded through this function, so before this argument existed the whole
    reported roster was read at whatever readout it trained with while only the
    floor and the (retired) glob-resolved keys took the override. The supervised
    arm trains pool="last", so its every latent number was a statement about one
    patch of the day, against mean-pooled SSL.

    THE OTHER TWO BRANCHES DEFINE THEIR OWN READOUT AND IGNORE IT, correctly. A
    ``PretrainedTSFM`` pools inside ``_pool_time`` (``time_pool="mean"`` by
    default, which is what the latent suite wants anyway), and an encode()-mode
    class has no pooled backbone output to re-read -- its ``encode()`` is the
    representation being evaluated. Neither is a checkpoint whose readout is a
    free choice, which is why this is not an error.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        klass = json.loads(cfg_path.read_text()).get("class")
        if klass == "PretrainedTSFM":
            return PretrainedTSFM.from_pretrained(str(ckpt_dir)).eval()
        if klass in ENCODE_MODE_CLASSES:
            import market_jepa.modeling as modeling

            return EncodeAdapter(
                getattr(modeling, klass).from_pretrained(str(ckpt_dir)).eval()
            )
    return load_backbone(ckpt_dir, ckpt_dir, project_name, pool=pool)

# ── Full-history supervised specialists ──────────────────────────────────────
# One trained encoder per TASK per month (plots/full_data_supervised). Shared
# by the IC sweep and the latent sweep so the two cannot disagree about which
# checkpoint a month means.

SUP_PROJECTS: dict[str, str] = {
    "sup_return": "supervised-full-month-return",
    "sup_vol": "supervised-full-month-vol-change",
    "sup_spread": "supervised-full-month-spread-change",
}
SUP_LABEL = {"sup_return": "Supervised return",
             "sup_vol": "Supervised vol",
             "sup_spread": "Supervised spread"}

# WHICH GENERATION OF THE SWEEP. Each project holds two: the original
# mid-to-mid return target and the 2026-08-22 forward-VWAP rerun
# (docs/return_bad_calculation.md). They share a checkpoint tree and their run
# names share the month prefix, so the anchor-stat table is the only thing
# separating them. Stamped into train_meta by save_train_meta since
# 2026-08-23. The historical backfill is complete; anything still unstamped
# is pre-fwdvwap by construction.
SUP_XS_STATS = os.environ.get("MJ_SUP_XS_STATS", "xs_anchor_stats_fwdvwap60")


def meta_xs_stats(meta: dict) -> str:
    """The anchor-stat table basename a checkpoint was trained against."""
    return str(meta.get("xs_anchor_stats") or "xs_anchor_stats")


def sup_ckpt_root() -> Path:
    """Overridable: compute nodes do not mount lab/."""
    return Path(os.environ.get("MJ_SUP_CKPT_ROOT",
                               "lab/market-jepa-checkpoints"))


def sup_run_dir(family: str, train_month: str, root: Path | None = None) -> Path:
    """The checkpoint trained on ``train_month`` for one supervised task.

    Run names carry the arm as a suffix (``2009-01_k11_mse0.025_tau0.5``), so
    the month is a PREFIX of train_meta's run_name, never the whole string.

    Restricted to ``SUP_XS_STATS``: the month alone is ambiguous now that two
    target generations share the tree, and without the filter every rerun
    month raises "2 runs match".
    """
    proj = (root or sup_ckpt_root()) / SUP_PROJECTS[family]
    hits, wrong_target = [], 0
    for d in sorted(p for p in proj.iterdir() if p.is_dir()):
        meta_f = d / "train_meta.json"
        if not meta_f.is_file():
            continue
        try:
            meta = json.loads(meta_f.read_text())
        except json.JSONDecodeError:
            continue
        rn = meta.get("run_name", "")
        if not (rn.startswith(f"{train_month}_") or rn == train_month):
            continue
        if meta_xs_stats(meta) != SUP_XS_STATS:
            wrong_target += 1
            continue
        hits.append(d)
    if not hits:
        extra = (f" ({wrong_target} run(s) match the month but were trained on "
                 f"a different anchor-stat target than {SUP_XS_STATS})"
                 if wrong_target else "")
        raise FileNotFoundError(
            f"{family}: no run for {train_month} in {proj}{extra}")
    if len(hits) > 1:
        raise RuntimeError(f"{family} {train_month}: {len(hits)} runs match")
    return hits[0]


def load_supervised(family: str, train_month: str, device=None,
                    pool: str | None = None):
    """The trained ViT trunk, frozen, exposing ``compute_features_multi``.

    ``pool`` defaults to None = the readout the checkpoint trained with, which
    is what the IC path wants (``tsfm_layer_ic``: prediction is scored at the
    last token, and these train there). The LATENT suite passes
    ``panel_lib.LATENT_POOL`` instead -- same rule as ``load_encoder``.
    """
    d = sup_run_dir(family, train_month)
    bb = load_backbone(str(d), SUP_PROJECTS[family], pool=pool)
    bb = bb.to(device).eval() if device is not None else bb.eval()
    for prm in bb.parameters():
        prm.requires_grad_(False)
    return bb


def prev_month(ym: str) -> str:
    """'2009-02' -> '2009-01'. The eval month's model trained the month before."""
    y, m = (int(v) for v in ym.split("-")[:2])
    return f"{y - (m == 1):04d}:{12 if m == 1 else m - 1:02d}".replace(":", "-")
