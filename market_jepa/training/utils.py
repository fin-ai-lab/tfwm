"""Shared training utilities for JEPA pre-training and supervised baselines.

Includes:
- collate_bucketed: DataLoader collate function for bucketed views
- MetricsAccumulator: aggregates per-step metrics before logging
- append_view_info: the information-token wire format
- setup_reproducibility: seed all RNGs and configure cudnn
- apply_mega_epoch: resize StreamingDataset to cover all training steps
- build_lr_scheduler: warmup + cosine LR schedule
- targets_config_from_task: derive dataset targets config from a TaskSpec
- Checkpoint save, probe eval dispatch
"""

import logging
import copy
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR, LinearLR,
                                      SequentialLR)
# RE-EXPORTS, not uses: nothing in this module calls these any more, but a
# dozen call sites across training, eval and tests import them from here and
# stable-finance is the single definition behind that name. Do not "clean up"
# as unused imports.
from stable_finance.dataset.transforms import (  # noqa: F401
    EPS,
    build_norm_groups,
    ffill_vwap,
    normalize as normalize_numpy,
    prior_vwap,
)
from stable_finance.dataset.mds import configure_epoch_size as apply_mega_epoch

logger = logging.getLogger(__name__)


# =============================================================================
# DataLoader collate
# =============================================================================


def collate_bucketed(batch: list[dict]) -> dict:
    """Collate batch into buckets grouped by augmentation config.

    Each bucket is padded independently so short-config pairs are not
    inflated to the max length of long-config pairs.

    Returns:
        {"buckets": [{"views": [Tensor, ...], "lengths": [Tensor, ...]}, ...]}
        Each bucket has views padded to its own max length.
    """
    # Flatten: each item may be a list of pair-dicts (n_pairs_per_obs > 1)
    flat_batch: list[dict] = []
    for item in batch:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)

    # Group by bucket_key
    groups: dict[int, list[dict]] = defaultdict(list)
    for sample in flat_batch:
        groups[sample.get("bucket_key", -1)].append(sample)

    buckets = []
    for _key, samples in sorted(groups.items()):
        if _key == -1:
            continue  # Skip sentinel zero-padded samples
        n_views = len(samples[0]["views"])

        views_by_idx: list[list[torch.Tensor]] = [[] for _ in range(n_views)]
        lengths_by_idx: list[list[int]] = [[] for _ in range(n_views)]

        for sample in samples:
            for v, view in enumerate(sample["views"]):
                views_by_idx[v].append(view)
                lengths_by_idx[v].append(sample["lengths"][v].item())

        padded_views = []
        view_lengths = []
        for v in range(n_views):
            max_len = max(lengths_by_idx[v])
            padded = []
            for view, length in zip(views_by_idx[v], lengths_by_idx[v]):
                if view.shape[-1] < max_len:
                    pad_size = max_len - view.shape[-1]
                    view = torch.nn.functional.pad(view, (0, pad_size))
                padded.append(view)
            padded_views.append(torch.stack(padded))
            view_lengths.append(torch.tensor(lengths_by_idx[v]))

        bucket_dict = {"views": padded_views, "lengths": view_lengths}
        if "targets" in samples[0]:
            bucket_dict["targets"] = torch.stack([s["targets"] for s in samples])
        if "target_metadata" in samples[0]:
            bucket_dict["target_metadata"] = {
                name: torch.stack([
                    sample["target_metadata"][name] for sample in samples
                ])
                for name in samples[0]["target_metadata"]
            }
        if "n_global_views" in samples[0]:
            bucket_dict["n_global_views"] = samples[0]["n_global_views"]
        if "pair_weights" in samples[0]:
            bucket_dict["pair_weights"] = torch.stack(
                [s["pair_weights"] for s in samples]
            )
        # THE CROSS-SECTION ID, and it must reach the loss. The dataset emits
        # xs_cell -- one id per (date, wall-clock anchor) -- precisely so
        # SupervisedModel.compute_loss can rank WITHIN a cross-section, which
        # is the quantity the reported rank IC measures. It was not collated,
        # so bucket.get("xs_cell") was always None, compute_loss took its
        # cells-is-None branch, and _within_cell_loss never executed: every
        # supervised head trained on the FLAT surrogate that ranks across
        # unrelated cross-sections (~99.9% of its pairs compare stocks at
        # different instants -- see _within_cell_loss's own docstring).
        #
        # Three pieces implemented the intended behaviour and this one missing
        # line disconnected them, silently, for the whole campaign. Stacked as
        # int64 (B,) so the loss can build its same-cell mask directly.
        if "xs_cell" in samples[0]:
            bucket_dict["xs_cell"] = torch.stack(
                [s["xs_cell"] for s in samples]
            ).to(torch.int64)
        if "ticker" in samples[0]:
            bucket_dict["tickers"] = [s.get("ticker", "") for s in samples]
        if "date" in samples[0]:
            bucket_dict["dates"] = [s.get("date", "") for s in samples]
        if "agg_factor" in samples[0]:
            bucket_dict["agg_factors"] = torch.tensor(
                [int(s.get("agg_factor", 1)) for s in samples], dtype=torch.long,
            )
        # Per-sample probe-eval labels keyed off view[0]'s start time. Stacked
        # as int64 tensors so downstream eval (the offline IC evals and
        # the random-init baseline script) can index them like targets.
        if "tod_bucket" in samples[0]:
            bucket_dict["tod_buckets"] = torch.tensor(
                [int(s.get("tod_bucket", -1)) for s in samples], dtype=torch.long,
            )
        if "tod_sec" in samples[0]:
            bucket_dict["tod_secs"] = torch.tensor(
                [int(s.get("tod_sec", -1)) for s in samples], dtype=torch.long,
            )
        if "weekday" in samples[0]:
            bucket_dict["weekdays"] = torch.tensor(
                [int(s.get("weekday", -1)) for s in samples], dtype=torch.long,
            )
        buckets.append(bucket_dict)

    return {"buckets": buckets}



# =============================================================================
# Metrics accumulator
# =============================================================================


class MetricsAccumulator:
    """Accumulates per-step training metrics and logs aggregated stats every N steps."""

    # Each mode emits a single "train/loss" for its primary objective;
    # auxiliaries (sigreg/inv/mae_valid_frac/cpc_acc, multi-task per-head
    # losses, collapse-monitor stats) are tracked as plain averages.
    FULL_STATS = {"train/loss", "train/grad_norm"}  # avg + min + max
    AVG_ONLY = {
        "train/loss_units_per_step", "train/rows_per_step",
        "train/rows_with_grad_frac", "train/cells_per_step",
        "train/sigreg_loss", "train/inv_loss", "train/mae_valid_frac",
        "train/cpc_acc",
        "train/cost_acc", "train/timemae_token_acc",
        "train/repr_std", "train/repr_std_min", "train/repr_cos_sim", "train/repr_eff_rank",
    }   # avg only
    # multi-task supervised: train/loss_<task> and the per-task gradient
    # balancing diagnostics emitted by MultiTaskSupervisedModel
    AVG_ONLY_PREFIXES = (
        "train/loss_",
        "train/task_gradient_norm_",
        "train/task_gradient_norm_ema_",
        "train/task_gradient_scale_",
        "train/task_gradient_contribution_norm_",
        "train/task_head_output_gradient_norm_",
        "train/task_weight_",
    )
    LAST_ONLY = {"train/lr", "train/aug_epoch", "train/virtual_epoch", "train/ema_momentum", "train/weight_decay", "obs_seen", "train/backbone_drift"}  # last value

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self._values: dict[str, list[float]] = defaultdict(list)
        self._steps_since_log = 0
        self._warned_keys: set[str] = set()

    def record(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._values[k].append(v)
        self._steps_since_log += 1

    def should_log(self) -> bool:
        return self._steps_since_log >= self.log_every

    def flush(self) -> dict[str, float]:
        out = {}
        for k, vals in self._values.items():
            if k in self.FULL_STATS:
                out[f"{k}_avg"] = sum(vals) / len(vals)
                out[f"{k}_min"] = min(vals)
                out[f"{k}_max"] = max(vals)
            elif k in self.AVG_ONLY or k.startswith(self.AVG_ONLY_PREFIXES):
                out[f"{k}_avg"] = sum(vals) / len(vals)
            elif k in self.LAST_ONLY:
                out[k] = vals[-1]
            else:
                if k not in self._warned_keys:
                    logger.warning(
                        "MetricsAccumulator: unrecognized key %r, defaulting to average",
                        k,
                    )
                    self._warned_keys.add(k)
                out[f"{k}_avg"] = sum(vals) / len(vals)
        self._values.clear()
        self._steps_since_log = 0
        return out


# =============================================================================
# Information-token wire format
# =============================================================================
#
# The VALUES in that payload are encoded by augmentations.encode_view_metadata,
# which is the only implementation: this module used to carry a second copy
# (norm_stat_channels / window_info / normalize_view) that the training path
# stopped calling when view preparation moved to stable-finance, leaving two
# definitions of the same arithmetic and one of them untested against reality.


def append_view_info(view: np.ndarray, info: np.ndarray) -> np.ndarray:
    """Attach per-window metadata in the final row of reserved columns.

    This is the compatibility wire format between data loaders and existing
    model modes. Values are no longer repeated at every timestep; the
    transformer projects the final-row payload directly into its information
    token and removes these reserved columns before patch tokenization.
    """
    metadata = np.zeros((len(view), info.size), dtype=view.dtype)
    metadata[-1] = info
    return np.concatenate([view, metadata], axis=1)


# =============================================================================
# Training setup helpers
# =============================================================================


def setup_reproducibility(args) -> None:
    """Seed all RNGs and configure cudnn."""
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Unconditional: TrainingConfig.reproducible has defaulted to True
        # since it was introduced and no launcher, sweep or config has ever
        # set it False, so the benchmark-mode branch that used to sit here
        # was dead. Reinstate the knob if a run ever wants to trade
        # determinism for autotuned kernels -- do not assume it still exists.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_save_steps(save_steps, save_fractions, max_train_steps: int | None) -> set[int]:
    """Absolute steps at which to write an intermediate checkpoint.

    ``save_fractions`` are fractions of THIS RUN's length. A sweep whose axis
    is the label budget has a different ``max_train_steps`` in every arm --
    steps_per_epoch is ``n_train_avail // (batch x accum)``, which moves with
    ``train_data_fraction`` and with the month's cross-section -- so a literal
    step list would sit at a different point on each arm's training curve.
    That is exactly what a progress plot must not do.

    Rounds to nearest, clamps into ``[1, max_train_steps]`` and returns a SET:
    a short arm rounds several fractions onto one step (12 steps puts both 1%
    and 10% on step 1), and saving it twice writes the same weights under two
    names, which reads downstream as two points that are really one.

    Step 0 is never returned because ``completed_steps`` starts at 1, so such
    an entry could never fire.
    """
    out = {int(x) for x in save_steps}
    if not save_fractions:
        return out
    if max_train_steps is None:
        raise ValueError(
            "checkpoint.save_fractions needs a known run length; this run has "
            "neither training.num_epochs nor training.max_train_steps."
        )
    for f in save_fractions:
        if not 0.0 < float(f) <= 1.0:
            raise ValueError(
                f"checkpoint.save_fractions must lie in (0, 1], got {f}")
        out.add(min(max(1, round(float(f) * max_train_steps)), max_train_steps))
    return out


def build_lr_scheduler(optimizer, total_steps: int, warmup_frac: float, min_lr: float,
                       warmup_start_frac: float = 0.01, schedule: str = "cosine",
                       decay_frac: float = 0.2, warmup_steps: int | None = None):
    """Build the LR schedule. ``cosine`` (default) or ``wsd``.

    Warmup is linear from ``warmup_start_frac`` of peak LR over
    ``warmup_frac * total_steps`` steps, in both -- or over exactly
    ``warmup_steps`` when that is given, which is what a checkpoint ladder in
    ABSOLUTE steps needs (see OptimizerConfig.warmup_steps).

    ``cosine``  anneals from peak to ``min_lr`` over every remaining step, so
                the LR is different at every point of the run.
    ``wsd``     warmup-stable-decay: holds peak LR CONSTANT through the stable
                phase, then decays linearly to ``min_lr`` over the last
                ``decay_frac`` of the run.

    WHY WSD EXISTS HERE. A label-budget curve wants "the model after N
    labels" at several N. Under cosine, the only way to get that is one run
    per N, because a checkpoint taken part-way through a cosine run is at a
    high LR that its own schedule never intended as an endpoint -- so the
    points are not comparable to each other and none is a finished model. A
    constant stable phase removes that: every checkpoint in it was taken under
    the SAME optimizer state, so one run's checkpoint ladder is a budget
    ladder, and the run count drops by the size of that ladder.

    THE CAVEAT THAT REMAINS. A stable-phase checkpoint still has not annealed.
    The ladder is internally consistent -- the shape of the curve is real --
    but each point sits slightly below what a decayed model at that budget
    would reach, and the decay is only applied at the END of the run. Points
    read against a converged reference (a ridge probe fit to convergence, say)
    are therefore a lower bound, not a like-for-like number.

    Returns:
        (scheduler, num_warmup_steps)
    """
    # BEFORE the warmup scheduler is constructed. LinearLR.__init__ applies
    # start_factor to the optimizer immediately, so reading the peak LR after
    # this line returns the WARMED-DOWN value (peak * warmup_start_frac) --
    # which made the WSD decay's end_factor 1.0 and the decay a silent no-op.
    peak_lr = max((g["lr"] for g in optimizer.param_groups), default=0.0) or 1.0

    if warmup_steps is None:
        num_warmup_steps = max(1, int(warmup_frac * total_steps))
    else:
        num_warmup_steps = int(warmup_steps)
        if not 1 <= num_warmup_steps < total_steps:
            raise ValueError(
                f"optimizer.warmup_steps={num_warmup_steps} must lie in "
                f"[1, {total_steps}) for a {total_steps}-step run")
    warmup_scheduler = LinearLR(
        optimizer, start_factor=warmup_start_frac, total_iters=num_warmup_steps)
    rest = max(1, total_steps - num_warmup_steps)

    if schedule == "cosine":
        tail = CosineAnnealingLR(optimizer, T_max=rest, eta_min=min_lr)
        return SequentialLR(
            optimizer, schedulers=[warmup_scheduler, tail],
            milestones=[num_warmup_steps],
        ), num_warmup_steps

    if schedule == "stable":
        # Warmup, then the peak rate to the end. The decays are branches
        # (checkpoint.anneal_steps, run by pretrain.py at each branch point);
        # the run's own trajectory never anneals, which is what makes every
        # branch leave from the same schedule.
        tail = ConstantLR(optimizer, factor=1.0, total_iters=rest)
        return SequentialLR(
            optimizer, schedulers=[warmup_scheduler, tail],
            milestones=[num_warmup_steps],
        ), num_warmup_steps

    if schedule != "wsd":
        raise ValueError(
            f"lr_schedule must be 'cosine', 'wsd' or 'stable', got {schedule!r}")
    if not 0.0 < decay_frac < 1.0:
        raise ValueError(f"decay_frac must lie in (0, 1), got {decay_frac}")

    num_decay_steps = max(1, int(decay_frac * total_steps))
    # At least one stable step, even on a run so short that warmup and decay
    # would otherwise cover all of it.
    num_stable_steps = max(1, rest - num_decay_steps)
    stable = ConstantLR(optimizer, factor=1.0, total_iters=num_stable_steps)
    # end_factor is RELATIVE to the peak LR; min_lr is absolute, so convert.
    decay = LinearLR(
        optimizer, start_factor=1.0,
        end_factor=min(1.0, max(0.0, min_lr / peak_lr)),
        total_iters=num_decay_steps,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, stable, decay],
        milestones=[num_warmup_steps, num_warmup_steps + num_stable_steps],
    )
    return scheduler, num_warmup_steps


# =============================================================================
# Model utilities
# =============================================================================


def count_parameters(model: nn.Module, **named_submodules: nn.Module) -> dict[str, int]:
    """Count total and named sub-module parameters.

    Example:
        count_parameters(model, backbone=model.backbone, projection=model.proj)
        # → {"total": ..., "backbone": ..., "projection": ...}
    """
    result = {"total": sum(p.numel() for p in model.parameters())}
    for name, submod in named_submodules.items():
        result[name] = sum(p.numel() for p in submod.parameters())
    return result


def maybe_compile(model: nn.Module, args) -> nn.Module:
    """Optionally torch.compile the model based on args.compile / args.compile_mode."""
    if getattr(args, "compile", False):
        mode = getattr(args, "compile_mode", "default")
        print(f"Compiling model with mode={mode}...")
        model = torch.compile(model, mode=mode)
    return model


def scale_lr(blr: float, effective_batch_size: int, default_batch_size: int | None) -> float:
    """Scale base LR by batch size ratio.

    Returns blr unchanged when default_batch_size is None (no scaling).
    """
    if default_batch_size is None:
        return blr
    return blr * effective_batch_size / default_batch_size


def cache_batches(dataloader, max_batches: int) -> list[dict]:
    """Consume up to max_batches from a DataLoader and return as a list."""
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batches.append(batch)
    return batches


def supervised_cell_view(one_global: dict, n_stocks: int) -> dict:
    """The TRAIN view of a supervised specialist: one cross_stock CELL.

    ``one_global`` is the single random-resized-crop global the eval/probe
    side keeps; the cell borrows its geometry (sequence length, scale band,
    resolution band) so a stock's view is drawn from the same distribution
    it is later probed on, and adds K stocks sharing that window. No locals
    and no industry table: the cell is a random subset of the cross-section
    the reported IC ranks over.
    """
    if int(n_stocks) < 2:
        raise ValueError(f"a cell needs at least 2 stocks, got n_stocks={n_stocks}")
    return {
        "name": "cross_stock",
        "n_stocks": int(n_stocks),
        "n_global_views": 1,
        "n_local_views": 0,
        "cross_stock_local_views": 0,
        "industry_table": None,
        # Anchors drawn as the panel draws them (uniform over the band, then
        # a resolution that fits), so the head trains where it is scored. The
        # window-first draw put 58% of cells in the last two hours against
        # the panel's 35% and cost the spread head half its panel IC.
        "anchor_uniform": True,
        "global_scale_range": list(one_global["global_scale_range"]),
        "global_seq_len": one_global["global_seq_len"],
        "global_agg_range": one_global.get("global_agg_range"),
    }


class ParamDrift:
    """``||w - w0|| / ||w0||`` over a module's parameters, from a snapshot.

    Cheap enough to call at every log flush (one pass over the parameters)
    and the single most direct answer to "did this train at all". A run whose
    backbone reports ~1e-3 here after thousands of steps has not.
    """

    def __init__(self, module: nn.Module):
        self._params = [p for p in module.parameters() if p.dtype.is_floating_point]
        with torch.no_grad():
            self._w0 = [p.detach().clone() for p in self._params]
            self._norm0 = torch.sqrt(
                sum((w.float() ** 2).sum() for w in self._w0)
            ).clamp_min(1e-12)

    @torch.no_grad()
    def __call__(self) -> float:
        d = sum(((p.detach() - w0).float() ** 2).sum()
                for p, w0 in zip(self._params, self._w0))
        return float((torch.sqrt(d) / self._norm0).item())


def targets_config_from_task(task_spec) -> dict:
    """Derive StreamingMarketDataset targets config from a single TaskSpec.

    The one-task case of :func:`targets_config_from_tasks`, delegating rather
    than restating it: both are live (``mode=supervised`` and
    ``mode=multi_supervised`` are launched side by side), and two independent
    spellings of "which target columns does the dataset emit" is exactly the
    kind of pair that drifts.
    """
    return targets_config_from_tasks([task_spec])


def targets_config_from_tasks(task_specs: list) -> dict:
    """Derive a unioned StreamingMarketDataset targets config from many TaskSpecs.

    Used by multi-task supervised mode so the dataset emits all required
    target columns in a single pass.
    """
    horizons: list[int] = []
    types: list[str] = []
    for spec in task_specs:
        if spec.horizon not in horizons:
            horizons.append(spec.horizon)
        if spec.target_type not in types:
            types.append(spec.target_type)
    return {"horizons": horizons, "types": types}


# =============================================================================
# Branch cooldowns (checkpoint.anneal_steps)
# =============================================================================


def resolve_anneal_plan(anneal_steps, anneal_frac: float, max_train_steps: int,
                        num_warmup_steps: int, schedule: str) -> dict[int, tuple[int, int]]:
    """{branch_step: (total_steps, cooldown_steps)} for checkpoint.anneal_steps.

    A branch at ``branch_step`` of the stable run cools for ``cooldown_steps``
    and lands at ``total_steps`` = branch_step + cooldown_steps, which names
    the checkpoint directory. Empty input is an empty plan.

    Refuses what would make a branch not a finished model at its count: a
    schedule other than "stable" (the run would decay under the branches),
    a branch point inside the warmup (the model has not reached its rate),
    a total past the run, and a last entry that is not the run's end (the
    root would then be an un-annealed model beside annealed step dirs).
    Duplicate totals, and two totals whose branch points coincide, are
    refused rather than silently merged.
    """
    totals = sorted({int(t) for t in (anneal_steps or [])})
    if not totals:
        return {}
    if schedule != "stable":
        raise ValueError(
            f"checkpoint.anneal_steps needs optimizer.lr_schedule='stable', "
            f"got {schedule!r}: a run that decays on its own hands the "
            f"branches a moving rate")
    if len(totals) != len(list(anneal_steps)):
        raise ValueError(f"checkpoint.anneal_steps has duplicates: {list(anneal_steps)}")
    if not 0.0 < anneal_frac < 1.0:
        raise ValueError(f"optimizer.anneal_frac must lie in (0, 1), got {anneal_frac}")
    if totals[-1] != int(max_train_steps):
        raise ValueError(
            f"the last of checkpoint.anneal_steps ({totals[-1]}) must equal "
            f"training.max_train_steps ({max_train_steps}) so the run root "
            f"is an annealed model")
    plan: dict[int, tuple[int, int]] = {}
    for T in totals:
        n = max(1, int(round(anneal_frac * T)))
        b = T - n
        if b < num_warmup_steps:
            raise ValueError(
                f"anneal step {T} branches at {b}, inside the {num_warmup_steps}-step "
                f"warmup; a cooldown from a model still ramping is not a finished "
                f"model at {T} steps -- drop it or shorten the warmup")
        if b in plan:
            raise ValueError(f"anneal steps {plan[b][0]} and {T} both branch at step {b}")
        plan[b] = (T, n)
    return plan


def cooldown_factor(k: int, n: int, min_ratio: float) -> float:
    """Peak-relative LR factor for cooldown step k of n (k in 0..n-1):
    linear from just below 1 to exactly ``min_ratio`` on the last step, the
    same line build_lr_scheduler's WSD decay walks."""
    if not 0 <= k < n:
        raise ValueError(f"cooldown step {k} outside 0..{n - 1}")
    return 1.0 + (min_ratio - 1.0) * (k + 1) / n


def take_snapshot(model, optimizer) -> dict:
    """Clone of the model weights and optimizer state, on the same device,
    for restore_snapshot. The copy is what a branch costs in memory: one
    more set of parameters and moments (about 3x the parameter bytes for
    AdamW), which is ~1 GB for ViT-Base."""
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"model": weights, "optimizer": copy.deepcopy(optimizer.state_dict())}


def restore_snapshot(model, optimizer, snap: dict) -> None:
    """Put the model and optimizer back where take_snapshot found them."""
    model.load_state_dict(snap["model"])
    optimizer.load_state_dict(snap["optimizer"])


# =============================================================================
# Checkpoint save / load
# =============================================================================


def save_checkpoint(model, path):
    """Save model weights for eval.  Handles LeJEPA, I-JEPA, and Supervised models."""
    from market_jepa.modeling.modes.supervised import (
        SupervisedModel,
        MultiTaskSupervisedModel,
    )
    from market_jepa.modeling.modes.ijepa import IJEPA

    os.makedirs(path, exist_ok=True)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if isinstance(m, MultiTaskSupervisedModel):
        torch.save(m.backbone.state_dict(), os.path.join(path, "backbone.pt"))
        torch.save(m.heads.state_dict(), os.path.join(path, "heads.pt"))
    elif isinstance(m, SupervisedModel):
        torch.save(m.backbone.state_dict(), os.path.join(path, "backbone.pt"))
        torch.save(m.head.state_dict(), os.path.join(path, "head.pt"))
    elif isinstance(m, IJEPA):
        torch.save(m.state_dict(), os.path.join(path, "model.pt"))
    else:
        m.save_pretrained(path)
    print(f"Saved checkpoint to {path}")


def save_train_meta(cfg, path: str | Path, *, run_name: str | None = None,
                    wandb_run_id: str | None = None,
                    wandb_project: str | None = None,
                    task_name: str | None = None,
                    augmentations: list | None = None,
                    backbone_drift: float | None = None,
                    max_train_steps: int | None = None,
                    completed_steps: int | None = None,
                    obs_seen: int | None = None) -> None:
    """Write train_meta.json next to the checkpoint so eval/plot code can
    populate metric rows offline (no W&B at eval time).

    Persists the training-time hyperparams that aren't recoverable from the
    model's own config.json: lambda, view counts (data-augmentation knobs),
    plus the W&B run name/id/project as breadcrumbs.

    Mirrors the schema produced by scripts/backfill_train_meta.py so existing
    checkpoints (backfilled) and future ones look identical to consumers.

    ``augmentations`` is the RESOLVED list the datasets were actually built
    from. Pass it whenever it is at hand: ``mode.dataset_overrides`` rewrites
    n_global_views / n_local_views / name after the config is composed (the
    supervised modes drop to one global view and no locals). Since 2026-09-06
    pretrain.py writes the resolved views back onto ``cfg`` before saving, so
    the fallback below now agrees with this argument for any new run -- but it
    does NOT for the checkpoints written before that, whose configs still
    record what was asked for rather than what ran. Falls back to the config
    when not given, which is right for the backfill script -- it has no run to
    ask.
    """
    import json

    mode_cfg = cfg.get("mode", {}) if isinstance(cfg, dict) else getattr(cfg, "mode", {})
    ds_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else getattr(cfg, "dataset", {})

    def _get(d, key, default=None):
        if isinstance(d, dict):
            return d.get(key, default)
        return getattr(d, key, default)

    lamb = _get(mode_cfg, "lamb")

    # `isinstance(augs, dict)` was the test here until 2026-08-22 and it is
    # FALSE for an OmegaConf DictConfig, which is what a hydra run actually
    # passes. So the mapping was never converted, iterating it yielded its
    # STRING KEYS, and `_get("0", "n_global_views", 0)` fell through to the
    # default -- every checkpoint written through this path recorded
    # n_global_views: 0 and n_local_views: 0. Silently, for a long time.
    # Duck-type the mapping instead: dict and DictConfig both have .values(),
    # list and ListConfig do not.
    if augmentations is not None:
        augs = list(augmentations)
    else:
        augs = _get(ds_cfg, "augmentations", {}) or {}
        augs = list(augs.values()) if hasattr(augs, "values") else list(augs)
    n_global = sum(_get(a, "n_global_views", 0) or 0 for a in augs)
    n_local = sum(_get(a, "n_local_views", 0) or 0 for a in augs)

    meta = {
        "lamb": float(lamb) if lamb is not None else None,
        "n_local_views": int(n_local),
        "n_global_views": int(n_global),
        "num_views": int(n_local + n_global),
        "run_name": run_name,
        "wandb_run_id": wandb_run_id,
        "wandb_project": wandb_project,
        # HOW FAR THROUGH TRAINING THIS CHECKPOINT IS. Written so an
        # intermediate checkpoint (checkpoint.save_fractions) is
        # self-describing: its directory is named by step, but the fraction
        # that step represents is only knowable next to the run's own length,
        # and the length moves with train_data_fraction. Both are None for
        # anything that does not pass them, and consumers ignore unknown keys.
        "max_train_steps": int(max_train_steps) if max_train_steps is not None else None,
        "completed_steps": int(completed_steps) if completed_steps is not None else None,
        # THE BUDGET AXIS, counted rather than derived. Under WSD the label
        # budget of a checkpoint is how many observations it has consumed, and
        # the trainer already accumulates exactly that. Reconstructing it from
        # steps x batch x accum would re-derive three knobs that can each move.
        "obs_seen": int(obs_seen) if obs_seen is not None else None,
    }

    # Record which task the head was trained on so downstream scorers can pick
    # the right column, and WHAT SHAPE the head is. The shape matters because
    # this file is the only config the in-job IC eval gets: it runs on a
    # compute node with no W&B access, so a missing loss_fn made every binned
    # checkpoint look scalar. load_model now reads the width off head.pt
    # instead of trusting this, but recording it keeps the two agreeing and
    # makes a checkpoint self-describing.
    if task_name is not None:
        meta["task"] = task_name
    # ||w - w0|| / ||w0|| of the encoder at the end of training. Written next
    # to the weights because it is the cheapest proof that they are not the
    # init; see ParamDrift.
    if backbone_drift is not None:
        meta["backbone_drift"] = float(backbone_drift)
    loss_fn = _get(mode_cfg, "loss_fn")
    if loss_fn is not None:
        meta["loss_fn"] = str(loss_fn)
        n_bins = _get(mode_cfg, "n_bins")
        if n_bins is not None:
            meta["n_bins"] = int(n_bins)

    # Risk factors change the INPUT CHANNEL COUNT, so a scorer that does not
    # replay them builds views of the wrong width — a mismatch that either
    # crashes or, worse, loads and reports noise. Recorded here because
    # post_train_ic_eval runs on a compute node with no W&B access and the
    # checkpoint is the only thing it can read.
    rf_tickers = list(_get(ds_cfg, "risk_factor_tickers", []) or [])
    if rf_tickers:
        meta["risk_factor_tickers"] = rf_tickers
        meta["risk_factor_columns"] = _get(ds_cfg, "risk_factor_columns")
        meta["risk_factor_targets"] = bool(
            _get(ds_cfg, "risk_factor_targets", True))
    xs_target = _get(ds_cfg, "xs_target", "uniform")
    if xs_target and xs_target != "uniform":
        meta["xs_target"] = str(xs_target)
    xs_eval_target = _get(ds_cfg, "xs_eval_target", "uniform")
    if xs_eval_target and xs_eval_target != "uniform":
        meta["xs_eval_target"] = str(xs_eval_target)

    # THE TWO KNOBS THAT CHANGE THE VIEW ITSELF, recorded for exactly the
    # reason the risk-factor block above is: this file is the only config the
    # in-job IC eval gets, and a scorer that does not replay them builds views
    # the model never trained on. Neither failure raises -- an unnormalized
    # encoder scored on normalized views, or a single-resolution encoder
    # scored across the 6-11 band, simply returns a worse number that looks
    # like a result. Both are read back by xs_ic_eval.panel_kwargs_for.
    norm_mode = _get(ds_cfg, "norm_mode", "per_view")
    if norm_mode and norm_mode != "per_view":
        meta["norm_mode"] = str(norm_mode)
    # THE INFORMATION TOKEN's two contributions. These widen the input, so
    # they are not merely panel knobs: a scorer that misses one builds a
    # narrower backbone and the state dict refuses to load. That is the good
    # failure -- the bad one would be scoring on views the encoder never saw.
    #
    # Written only when TRUE, and read back with a default of False, which is
    # what lets a checkpoint from before the knob existed still score. That
    # asymmetry survives the defaults flipping to True on 2026-08-25: a new run
    # always stamps them, so absence still means off.
    #
    # OLD KEYS, still accepted by every reader: norm_stats_channels (the stats
    # were broadcast as channels through the patch embedding until 2026-08-25)
    # and time_info.
    if _get(ds_cfg, "info_norm_stats", _get(ds_cfg, "norm_stats_channels", False)):
        meta["info_norm_stats"] = True
    if _get(ds_cfg, "info_window", _get(ds_cfg, "time_info", False)):
        meta["info_window"] = True
    agg_band = _get(augs[0], "global_agg_range", None) if augs else None
    if agg_band:
        meta["global_agg_range"] = [int(x) for x in agg_band]
    # THE OTHER WAY TO SHORTEN THE CONTEXT. global_agg_range holds the token
    # count and shrinks each token; this holds the token size and feeds fewer
    # of them. A scorer that misses it builds 2048-token views for a model
    # trained on 256 -- eight times the context it ever saw, and no error.
    seq_len = _get(augs[0], "global_seq_len", None) if augs else None
    if seq_len and int(seq_len) != 2048:
        meta["global_seq_len"] = int(seq_len)

    # POOLING CHANGES THE PARAMETER SET, not just the readout. The backbone
    # allocates a cls_token only under pool="cls", and n_pos is max_seq_len + 1
    # there against max_seq_len otherwise -- so a checkpoint trained under one
    # pooling cannot even be LOADED under the other. post_train_ic_eval's
    # _load_cfg hardcoded "cls" because nothing else had ever been trained;
    # recording it is what lets that stop being true.
    bb_cfg = cfg.get("backbone", {}) if isinstance(cfg, dict) else getattr(
        cfg, "backbone", {})
    pool = _get(bb_cfg, "pool", "cls")
    if pool and pool != "cls":
        meta["pool"] = str(pool)
    # Worse than pool, and recorded for that reason: causal masking adds no
    # parameters, so a causal checkpoint loads into a bidirectional backbone
    # without complaint and is scored as something it is not.
    if _get(bb_cfg, "causal", False):
        meta["causal"] = True
    if _get(bb_cfg, "state_token", False):
        meta["state_token"] = True
    inner = _get(bb_cfg, "config", None)
    # POSITION KNOBS. Recorded for the same reason as causal: they live in the
    # INNER TransformerConfig, which post_train_ic_eval._load_cfg synthesizes
    # as {} -- so a run trained with one was rebuilt without it at score time.
    # pos_embed swaps a Parameter for a buffer, so a mismatch is at least
    # visible; cls_pos adds NOTHING to the state dict and would load clean
    # while scoring a differently-positioned readout -- the exact failure mode
    # `causal` once had.
    if inner is not None:
        pe = _get(inner, "pos_embed", "learned")
        if pe and pe != "learned":
            meta["pos_embed"] = str(pe)
        cp = _get(inner, "cls_pos", "own")
        if cp and cp != "own":
            meta["cls_pos"] = str(cp)
        pis = _get(inner, "pos_init_std", 0.02)
        if pis is not None and float(pis) != 0.02:
            meta["pos_init_std"] = float(pis)
    if _get(bb_cfg, "diff_channels", False):
        meta["diff_channels"] = True

    # WHICH TARGET DEFINITION. The anchor-stat tables ARE the target: the same
    # month scored against xs_anchor_stats vs xs_anchor_stats_fwdvwap60 is a
    # different quantity (docs/return_bad_calculation.md), and nothing else in
    # this file tells them apart. Both generations of the full-history sweep
    # share one wandb project AND one checkpoint tree, keyed by a run_name
    # whose month prefix is identical, so any consumer that keys on the month
    # would silently average two different targets into one curve. Basename
    # only: a compute node sees the staged node-local copy, a local run sees
    # lab/market-jepa-mosaic.
    xs_dir = _get(ds_cfg, "xs_anchor_stats_dir", None)
    if xs_dir:
        meta["xs_anchor_stats"] = str(xs_dir).rstrip("/").rsplit("/", 1)[-1]

    # WHICH BACKBONE, not just how it was configured. post_train_ic_eval
    # synthesizes a ViT-384 config because that is all this project had ever
    # trained; a resnet checkpoint would be rebuilt as a transformer and fail
    # to load. Recorded whenever it is not the ViT default, so existing
    # checkpoints keep resolving the way they always have.
    tgt = _get(bb_cfg, "_target_", None)
    if tgt and "transformer.TransformerBackbone" not in str(tgt):
        meta["backbone_target"] = str(tgt)
        for k in ("d_embedding", "variant", "pool"):
            v = _get(bb_cfg, k, None)
            if v is not None:
                meta[f"backbone_{k}"] = v

    # THE WHOLE CONFIG, not just the summary above.
    #
    # Every key in this function is a hand-maintained mirror of one knob, and
    # post_train_ic_eval._load_cfg keeps a SECOND hand-maintained list that
    # turns them back into a config. Two lists that must track the schema by
    # hand, and they have failed repeatedly and quietly: n_global_views was 0
    # in every checkpoint until 0c5fb8f; `causal` was dropped at load, so
    # causal runs were scored as though trained bidirectionally, with no error;
    # and global_seq_len and xs_anchor_stats had to be added in 2026-08.
    #
    # _load_cfg already short-circuits on a real config --
    #     if "backbone" in cfg and "mode" in cfg: return cfg
    # -- and never had one to short-circuit on. Writing it means a NEW knob
    # needs no entry here and none there: it is simply in the config, the way
    # it was at training time.
    #
    # The summary keys stay. Plot and table code reads them by name, and a
    # checkpoint written before this still has only those, so both shapes have
    # to remain readable.
    try:
        from omegaconf import DictConfig, OmegaConf
        full = (OmegaConf.to_container(cfg, resolve=True)
                if isinstance(cfg, DictConfig) else
                json.loads(json.dumps(cfg, default=str)))
    except Exception:                      # unresolvable interpolation, etc.
        full = None
    # Only when the backbone block is actually USABLE. A config can carry an
    # empty `backbone: {}` (a test fixture, a mode that builds its own), and
    # writing that would make _load_cfg short-circuit onto a block with no
    # _target_ and no pool -- strictly worse than the summary it replaced.
    bb_full = (full or {}).get("backbone") if isinstance(full, dict) else None
    if (isinstance(full, dict) and "mode" in full
            and isinstance(bb_full, dict) and bb_full.get("_target_")):
        meta["config"] = full

    out_path = Path(path) / "train_meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))


# =============================================================================
# Probe evaluation
# =============================================================================


def collect_probe_data(
    model: nn.Module,
    eval_batches: list[dict],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode eval batches through frozen backbone and extract embeddings + targets.

    Args:
        model: Model with an ``encode()`` method.
        eval_batches: List of batch dicts with ``"buckets"``.
        device: Target device.

    Returns:
        (X, y) where X is (N, d) and y is (N, n_targets), both float32.
        Only rows whose targets are ALL NaN are dropped; per-column NaNs ride
        through for the consumer to mask.

    Dropping a row for any NaN would be catastrophic here. NaN now carries the
    no-clamp rule, and which rows it hits depends on the horizon: at h=7200
    only ~10% of anchors leave a two-hour forward window. An all-columns filter
    would keep just the intersection — the handful of early-day anchors valid
    at every horizon — and then fit even the h=300 probe on that biased sliver.
    ``probe_eval_worker`` masks each column independently instead.
    """
    model.eval()
    all_X = []
    all_y = []

    with torch.no_grad():
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                n_global = bucket.get("n_global_views")
                if n_global is None or n_global < 1:
                    continue
                views = [v.to(device) for v in bucket["views"]]
                lengths = [l.to(device) for l in bucket["lengths"]]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # Opt-in metadata channel. Learned encoders see only the
                    # tensors; a model that needs per-sample context the view
                    # cannot carry (the finance baselines need `dates` to pick
                    # that month's fitted parameters and `agg_factors` to turn a
                    # horizon in seconds into a number of steps) sets
                    # `wants_metadata = True` and receives the bucket dict.
                    if getattr(model, "wants_metadata", False):
                        enc = model.encode(views, lengths, metadata=bucket)
                    else:
                        enc = model.encode(views, lengths)

                embeddings = enc["embeddings"][:, 0, :].float().cpu().numpy()
                all_X.append(embeddings)

                if "targets" in bucket:
                    all_y.append(bucket["targets"].numpy())

    if not all_X or not all_y:
        return np.empty((0, 0), dtype=np.float32), np.empty(
            (0, 0), dtype=np.float32
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    valid = ~np.isnan(y).all(axis=1)
    return X[valid], y[valid]


def save_probe_embeddings(
    model: nn.Module,
    eval_batches: list[dict],
    device: torch.device,
    *,
    obs_seen: int,
    target_names: list[str],
    probe_train_batches: list[dict] | None = None,
) -> Path | None:
    """Run a single forward pass and save embeddings to an npz file.

    Returns the path to the saved npz, or None if insufficient data.
    """
    temp_dir = Path(
        os.environ.get("MARKET_JEPA_TEMP_PROBE_EVAL_DIR", "./temp_probe_eval")
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    # The two shapes the worker understands, and the ONLY difference between
    # them: a pre-split train/val pair when a separate probe-train window was
    # configured, or one array the worker splits temporally itself. Everything
    # downstream of here -- the model.train() restore, the emptiness check, the
    # uid'd filename -- was written out twice and is written once.
    if probe_train_batches is not None:
        X_train, y_train = collect_probe_data(model, probe_train_batches, device)
        X_val, y_val = collect_probe_data(model, eval_batches, device)
        arrays = {"X_train": X_train, "y_train": y_train,
                  "X_val": X_val, "y_val": y_val}
        empty = len(X_train) == 0 or len(X_val) == 0
        shortfall = "train=%d, val=%d" % (len(X_train), len(X_val))
    else:
        X, y = collect_probe_data(model, eval_batches, device)
        arrays = {"X": X, "y": y}
        empty = len(X) == 0
        shortfall = "n=%d" % len(X)

    model.train()

    if empty:
        logger.warning(
            "Insufficient probe data at obs_seen=%d (%s)", obs_seen, shortfall,
        )
        return None

    uid = uuid.uuid4().hex[:8]
    npz_path = temp_dir / f"probe_{obs_seen}_{uid}.npz"
    np.savez(npz_path, target_names=np.array(target_names), **arrays)
    return npz_path


def spawn_probe_worker(
    npz_path: Path,
    *,
    obs_seen: int,
    wandb_run_id: str,
    wandb_project: str,
    wandb_entity: str | None,
    temporal_train_frac: float = 0.5,
    num_threads: int = 4,
) -> tuple[subprocess.Popen, Path]:
    """Spawn a single probe eval worker subprocess."""
    temp_dir = npz_path.parent
    worker_script = (
        Path(__file__).resolve().parent.parent / "eval" / "probe_eval_worker.py"
    )

    log_uid = uuid.uuid4().hex[:8]
    log_path = temp_dir / f"probe_{obs_seen}_{log_uid}.log"

    cmd = [
        sys.executable,
        str(worker_script),
        "--npz_path", str(npz_path),
        "--obs_seen", str(obs_seen),
        "--wandb_run_id", wandb_run_id,
        "--wandb_project", wandb_project,
        "--temporal_train_frac", str(temporal_train_frac),
    ]
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])

    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB")}
    t = str(num_threads)
    env.update(
        {"OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t}
    )

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    return proc, log_path


def dispatch_probe_eval(
    model: nn.Module,
    eval_batches: list[dict],
    device: torch.device,
    *,
    obs_seen: int,
    wandb_run_id: str,
    wandb_project: str,
    wandb_entity: str | None,
    target_names: list[str],
    temporal_train_frac: float = 0.5,
    probe_train_batches: list[dict] | None = None,
    num_threads: int = 4,
) -> list[tuple[subprocess.Popen, Path]]:
    """Extract embeddings and dispatch a single probe eval worker.

    Used during training to log probe metrics to the parent W&B run.

    Returns list of (process, log_path) tuples.
    The worker cleans up its own npz file after finishing.
    """
    npz_path = save_probe_embeddings(
        model, eval_batches, device,
        obs_seen=obs_seen,
        target_names=target_names,
        probe_train_batches=probe_train_batches,
    )
    if npz_path is None:
        return []

    proc, log_path = spawn_probe_worker(
        npz_path,
        obs_seen=obs_seen,
        wandb_run_id=wandb_run_id,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        temporal_train_frac=temporal_train_frac,
        num_threads=num_threads,
    )
    logger.info("Dispatched probe worker: obs_seen=%d", obs_seen)
    return [(proc, log_path)]


def calibrate_discretizers(
    dataloader,
    target_col_idx: dict,
    n_bins: int,
    n_batches: int,
) -> dict:
    """Fit ONE set of bin edges per task, in a single pass over the loader.

    Every task gets its own :class:`Discretizer` because bin edges are
    equal-count quantiles of a specific target's empirical distribution -- a
    return's quantiles say nothing about a spread change's, and sharing one set
    would put most of one target's mass in a single bin.

    The edges are quantiles of the target's empirical distribution over
    ``n_batches`` batches of the TRAINING dataloader -- the bins that month's
    cross-section actually produced, not a fixed grid carried across regimes.
    One scheme serves every target type; values that tie across a bin edge are
    spread by overlap at apply time, so the atom at zero needs no special bin.

    Deliberately ONE pass collecting every task's column, not one pass per
    task: the loader is the expensive part. It also guarantees all
    discretizers see the SAME calibration sample, so a difference between
    heads is a difference between targets.

    Raises rather than returning None: a supervised run whose calibration data
    was too sparse to bin has nothing to train against, and should stop here
    instead of at the first batch.

    NOTE on multi-month training spans: the dataloader is shuffled over the
    whole span, so the edges are fitted over every training month rather than
    only the last one. For the single-month runs that make up the reported
    campaign these are the same thing.
    """
    from market_jepa.eval.discretize import fit

    tasks = list(target_col_idx)
    collected: dict[str, list] = {t: [] for t in tasks}
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        for bucket in batch["buckets"]:
            if "targets" not in bucket:
                continue
            tgt = bucket["targets"]
            for t in tasks:
                # (B, K, n_targets) from the grouped sampler; the cell axis is
                # flattened because bin edges are a marginal quantity.
                collected[t].append(
                    tgt[..., target_col_idx[t]].reshape(-1).numpy()
                )

    out: dict = {}
    for t in tasks:
        if not collected[t]:
            raise RuntimeError(
                f"No targets collected for task {t!r} during calibration — "
                "check the dataset targets config"
            )
        y_cal = np.concatenate(collected[t])
        y_cal = y_cal[~np.isnan(y_cal)]
        logger.info(
            "Calibration[%s]: %d valid samples -> fitting %d-bin discretizer",
            t, len(y_cal), n_bins,
        )
        disc = fit(y_cal, n_bins)
        if disc is None:
            raise RuntimeError(
                f"Calibration produced no Discretizer for task '{t}' "
                f"({n_bins}-bin) — calibration data may be too sparse."
            )
        logger.info(
            "Calibration[%s]: edges=%s  spread ties=%d %s",
            t, np.array2string(disc.edges, precision=6),
            len(disc.atom_vals),
            np.array2string(disc.atom_vals, precision=6),
        )
        out[t] = disc
    return out


def calibrate_discretizer(
    dataloader,
    target_col_idx: int,
    task_spec,
    n_bins: int,
    n_batches: int,
):
    """Fit a single supervised head's bin edges. See :func:`calibrate_discretizers`.

    The one-task case, delegating rather than restating it. Both are live --
    ``mode=supervised`` trains one head, ``mode=multi_supervised`` trains
    several -- and the two used to be sixty near-identical lines whose only
    real difference was the shape of ``target_col_idx``.
    """
    return calibrate_discretizers(
        dataloader,
        {task_spec.name: target_col_idx},
        n_bins=n_bins,
        n_batches=n_batches,
    )[task_spec.name]
