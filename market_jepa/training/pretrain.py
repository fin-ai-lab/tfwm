"""Pre-training script for Time Series JEPA models.

Supports three modes via Hydra config groups:

  - **LeJEPA** (mode=lejepa): Self-supervised pre-training with
    SIGReg + invariance loss.
  - **I-JEPA** (mode=ijepa): Self-supervised pre-training with
    masked patch prediction and EMA target encoder.
  - **Supervised** (mode=supervised): End-to-end training with a task-
    specific prediction head (MSE / CrossEntropy).  Evaluation strips
    the head and runs probe-based evaluation identical to JEPA.

Usage:
    uv run train.py                          # LeJEPA mode (default)
    uv run train.py mode=ijepa               # I-JEPA mode
    uv run train.py mode=supervised          # Supervised mode
    uv run train.py backbone=resnet          # Different backbone
"""

import json
import logging
import math
import os
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf

from market_jepa.schemas import (
    Config,
    LAMBDA_BY_PAIRING,
    LeJEPAModeConfig,
    IJEPAModeConfig,
    MAEModeConfig,
    SupervisedModeConfig,
    MultiTaskSupervisedModeConfig,
    lejepa_pairing_key,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

logger = logging.getLogger(__name__)

from hydra.utils import instantiate

from market_jepa.backbone_config import (
    assert_no_ignored_backbone_overrides,
    backbone_block,
    owns_backbone,
)

from market_jepa.modeling.modes.supervised import SupervisedModel, MultiTaskSupervisedModel
from market_jepa.modeling.modes.ijepa import IJEPA
from market_jepa.eval.tasks import TASK_REGISTRY
from ..augmentations import canonicalize_augmentations
from .utils import (
    ParamDrift,
    supervised_cell_view,
    collate_bucketed,
    MetricsAccumulator,
    setup_reproducibility,
    apply_mega_epoch,
    build_lr_scheduler,
    resolve_anneal_plan,
    resolve_save_steps,
    restore_snapshot,
    take_snapshot,
    cooldown_factor,
    targets_config_from_task,
    targets_config_from_tasks,
    maybe_compile,
    scale_lr,
    cache_batches,
    save_checkpoint,
    save_train_meta,
    dispatch_probe_eval,
    calibrate_discretizer,
    calibrate_discretizers,
)

# Re-exports for backwards compatibility (moved to utils.py)
from .utils import collect_probe_data  # noqa: F401


def run_jepa_eval(model, eval_batches, device):
    """Backwards-compat wrapper — delegates to model.eval_step()."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    return m.eval_step(eval_batches, device)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------


def record_resolved_views(cfg, augmentations: list, eval_augmentations: list) -> None:
    """Write the resolved view lists back onto ``cfg``, so the saved config
    records what TRAINED rather than what was asked for.

    Everything upstream rewrites the PYTHON lists: the supervised/ijepa/mae
    branch replaces the pairing with a single random_resized_crop view, and
    ``mode.dataset_overrides`` then rewrites name and the view counts. None of
    that reached ``cfg``, so the config hydra saves recorded the request -- a
    supervised checkpoint claimed cross_stock with 2 global and 6 local views
    while having trained on one crop. ``save_train_meta`` is handed the
    resolved list and so has always been right; the saved config was the half
    that lied, and it is what anything reading a checkpoint directly sees.

    ``dataset.augmentations`` is ``dict[str, Any]``, so the resolved entries go
    back wholesale. ``dataset.eval_augmentations`` is typed
    ``dict[str, AugmentationConfig]`` and rejects plain dicts, so its entries
    are updated field by field -- which is enough, because only
    ``dataset_overrides``' three scalars ever change them in place.
    """
    OmegaConf.update(cfg, "dataset.augmentations",
                     {str(i): a for i, a in enumerate(augmentations)},
                     merge=False)
    for i, aug in enumerate(eval_augmentations):
        if str(i) not in cfg.dataset.eval_augmentations:
            continue
        for field_name in ("n_global_views", "n_local_views", "name"):
            if field_name in aug:
                OmegaConf.update(
                    cfg, f"dataset.eval_augmentations.{i}.{field_name}",
                    aug[field_name], merge=False)


def resolve_lejepa_lambda(cfg, augmentations: list) -> float:
    """Resolve ``mode.lamb`` against the pairing the views resolved TO, and
    write it back onto ``cfg``. Returns the resolved value.

    Precedence is **explicit pin > this pairing's swept optimum > hard error**.
    There is no numeric default to fall through to, deliberately: the 0.01 that
    used to sit on the field was the k2ind-era value and the optimum for no arm
    in ``LAMBDA_BY_PAIRING``, so every sweep that did not pin lambda trained a
    value nothing had been tuned at and said nothing about it.

    An unmapped pairing -- any cross_stock K != 2, or a same-stock aug outside
    the swept three -- has no tuned value at all, so it raises rather than
    borrowing another arm's number, which would be the same failure with a
    different constant.

    Takes the RESOLVED augmentation list, not ``cfg.dataset.augmentations``,
    because ``mode.dataset_overrides`` can have rewritten the pairing out from
    under the composed config.
    """
    lamb = cfg.mode.lamb
    if lamb is None:
        pairing = lejepa_pairing_key(augmentations)
        lamb = LAMBDA_BY_PAIRING.get(pairing) if pairing else None
        if lamb is None:
            raise ValueError(
                "mode.lamb is not set and cannot be resolved: lambda's "
                f"optimum is a property of the PAIRING, and {pairing!r} has "
                "no swept value (known: "
                f"{', '.join(sorted(LAMBDA_BY_PAIRING))}). Pin it explicitly, "
                "e.g. mode.lamb=0.01. See LAMBDA_BY_PAIRING in "
                "market_jepa/schemas.py for the table and its caveats."
            )
        logger.info(
            "mode.lamb not pinned; resolved to %g from the %s row of "
            "LAMBDA_BY_PAIRING.", lamb, pairing)
    # Write it back so the model, the run name and train_meta all report the
    # lambda that actually trained.
    OmegaConf.update(cfg, "mode.lamb", float(lamb), merge=False)
    return float(lamb)


def train(cfg: Config) -> None:
    # FIRST, BEFORE ANYTHING COSTS ANYTHING. A top-level backbone FIELD
    # override on a mode that owns its backbone is always a mistake, and it has
    # to be caught here rather than at model-build time: the run would
    # otherwise spend dataset discovery, mosaic staging and shard
    # materialization before saying so, and on the cluster that is minutes of
    # an allocated GPU to report a typo.
    assert_no_ignored_backbone_overrides(cfg)

    # -------------------------------------------------------------------------
    # Detect mode: supervised vs LeJEPA vs I-JEPA
    # -------------------------------------------------------------------------
    is_multi_supervised = cfg.mode._target_.endswith(".MultiTaskSupervisedModel")
    is_supervised = (
        cfg.mode._target_.endswith(".SupervisedModel") or is_multi_supervised
    )
    is_ijepa = "IJEPA" in cfg.mode._target_ and "LeJEPA" not in cfg.mode._target_
    is_mae = cfg.mode._target_.endswith(".MAE")
    is_lejepa = cfg.mode._target_.endswith(".LeJEPA")

    # Narrow mode config type for IDE hints
    if is_ijepa:
        mode_cfg: IJEPAModeConfig = cfg.mode
    elif is_mae:
        mode_cfg: MAEModeConfig = cfg.mode
    elif is_multi_supervised:
        mode_cfg: MultiTaskSupervisedModeConfig = cfg.mode
    elif is_supervised:
        mode_cfg: SupervisedModeConfig = cfg.mode
    else:
        mode_cfg: LeJEPAModeConfig = cfg.mode

    if is_multi_supervised:
        task_names = list(mode_cfg.tasks)
        task_specs = {t: TASK_REGISTRY[t] for t in task_names}
    else:
        task_names = [mode_cfg.task] if is_supervised else []
        task_specs = (
            {mode_cfg.task: TASK_REGISTRY[mode_cfg.task]} if is_supervised else {}
        )
    # Single-task convenience reference (None for multi/non-supervised paths).
    task_spec = task_specs[task_names[0]] if (is_supervised and not is_multi_supervised) else None
    target_col_idx = None

    if is_supervised:
        if is_multi_supervised:
            logger.info(
                "Multi-task supervised mode: tasks=%s",
                ",".join(task_names),
            )
            for t, spec in task_specs.items():
                logger.info("  task=%s | target=%s h=%ds",
                            spec.name, spec.target_type, spec.horizon)
        else:
            logger.info(
                "Supervised mode: task=%s | target=%s h=%ds",
                task_spec.name, task_spec.target_type, task_spec.horizon,
            )

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    setup_reproducibility(cfg.training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    from .streaming_dataset import StreamingMarketDataset, discover_streams
    from stable_finance.dataset import MarketSchedule

    mosaic_dir = cfg.machine.mosaic_dir

    train_date_start = cfg.dataset.train_date_start
    train_date_end = cfg.dataset.train_date_end
    eval_date_start = cfg.dataset.eval_date_start
    eval_date_end = cfg.dataset.eval_date_end

    probe_train_date_start = cfg.dataset.eval_train_date_start
    probe_train_date_end = cfg.dataset.eval_train_date_end
    use_separate_probe_train = bool(probe_train_date_start and probe_train_date_end)

    live_eval = bool(OmegaConf.select(cfg, "training.live_eval", default=True))

    # Gate the TRAIN discovery on the backend, for the same reason the probe
    # discovery is gated below: dataset.backend=days builds the train dataset
    # from the day store (DayStoreCellDataset), and train_streams is read only
    # on the mds path -- so discovering it asks a days run for a mosaic it has
    # no reason to stage, and raises on any node that does not happen to hold
    # the training span in BOTH layouts. A ten-year days run is the case that
    # surfaced it: the mosaic is the eval dataset's, and the eval window is
    # one month.
    train_streams = None
    if str(OmegaConf.select(cfg, "dataset.backend", default="mds")) != "days":
        train_streams = discover_streams(mosaic_dir, train_date_start, train_date_end)
    eval_streams = discover_streams(mosaic_dir, eval_date_start, eval_date_end)
    # Gate the DISCOVERY, not use_separate_probe_train — that flag also guards
    # the temporal_train_frac validation below, which must still fire.
    # eval_train_date_* defaults to a fixed month (2023-01) that a live_eval=false
    # job has no reason to stage, so discovering it raises on any node that
    # doesn't happen to have that month lying around from an earlier sweep.
    # Every pythia hopper node did; the l40s nodes did not, which is the only
    # reason this surfaced.
    probe_train_streams = None
    if live_eval and use_separate_probe_train:
        probe_train_streams = discover_streams(mosaic_dir, probe_train_date_start, probe_train_date_end)

    n_pairs_per_obs = cfg.dataset.n_pairs_per_obs

    rf_kwargs = {
        "risk_factor_dir": cfg.machine.risk_factor_dir,
        "risk_factor_tickers": list(cfg.dataset.risk_factor_tickers),
        "risk_factor_columns": cfg.dataset.risk_factor_columns,
        # Not a risk-factor knob, but rides in the shared kwargs so the
        # ablation reaches ALL datasets (train/eval/probe) identically —
        # train and eval must always see the same zeroed channels.
        "zero_feature_columns": list(
            OmegaConf.select(cfg, "dataset.zero_feature_columns", default=None) or []
        ),
        # Input-channel-only risk factors: see DatasetConfig.risk_factor_targets.
        # Rides in the shared kwargs so train, eval and probe datasets agree.
        "risk_factor_targets": bool(
            OmegaConf.select(cfg, "dataset.risk_factor_targets", default=True)
        ),
        # Also not a risk-factor knob. Here for the same reason as
        # zero_feature_columns: a normalization the eval dataset does not
        # share with the train dataset is not an ablation, it is a bug.
        "norm_mode": str(
            OmegaConf.select(cfg, "dataset.norm_mode", default="per_view")
        ),
        # And the same again for the two halves of the INFORMATION TOKEN.
        # Both change n_features, so a train dataset that carried one while the
        # eval dataset did not would build a wide backbone and feed it narrow
        # eval views.
        "info_norm_stats": bool(
            OmegaConf.select(cfg, "dataset.info_norm_stats", default=True)
        ),
        "info_window": bool(
            OmegaConf.select(cfg, "dataset.info_window", default=True)
        ),
    }

    holiday_csv = cfg.machine.holiday_csv
    schedule = MarketSchedule(holiday_csv) if holiday_csv else None

    # Targets config
    if is_multi_supervised:
        targets_config = targets_config_from_tasks(
            list(task_specs.values())
        )
    elif is_supervised:
        targets_config = targets_config_from_task(task_spec)
    else:
        targets_config = OmegaConf.to_container(cfg.dataset.targets, resolve=True)

    epoch_dependent_seed = cfg.dataset.epoch_dependent_seed
    predownload = cfg.dataset.predownload
    cache_limit = OmegaConf.select(cfg, "dataset.cache_limit", default=None)

    # --- Augmentations (dict[str, AugmentationConfig] → list[dict] for StreamingMarketDataset) ---
    augmentations = list(OmegaConf.to_container(cfg.dataset.augmentations, resolve=True).values())
    eval_augmentations = list(OmegaConf.to_container(cfg.dataset.eval_augmentations, resolve=True).values())

    # Get global defaults from first augmentation config
    _first_aug = augmentations[0] if augmentations else {}
    _global_seq_len = _first_aug.get("global_seq_len", 2048)
    _global_scale_range = _first_aug.get("global_scale_range", [0.5, 1.0])
    _global_agg_range = _first_aug.get("global_agg_range")

    # Extended-hours grid (04:00 ET to four hours past the close). The session
    # length then varies per ticker-day, so a fractional scale range no longer
    # pins view resolution — require the absolute band instead of silently
    # training on a resolution the position embeddings never saw.
    extended_hours = bool(OmegaConf.select(cfg, "dataset.extended_hours", default=False))
    xs_anchor_stats_dir = OmegaConf.select(
        cfg, "dataset.xs_anchor_stats_dir", default=None,
    )
    # uniform | rank | zscore | raw — target transform on identical geometry.
    # Rank IC is invariant to all four within a cell.
    #
    # The TRAIN dataset and the EVAL/PROBE datasets take this separately so an
    # experiment can vary them deliberately. The default is uniform for both:
    #
    #   raw     bins are quantiles of the POOLED return distribution, so a
    #           label mixes "this stock moved" with "the whole market moved".
    #   zscore  bins are quantiles of the per-cell standardized target, so a
    #           market-wide shift is removed.
    #   uniform label is the exact rank in ITS OWN cross-section — the thing
    #           rank IC actually measures. `rank` maps that percentile to a
    #           Gaussian score instead.
    #
    # This used to be forced to `raw`, because the binning carved out a zero
    # bin at +/-5bp and that threshold is meaningless on a standardized score.
    # The bins are now equal-count quantiles with ties spread by overlap
    # (market_jepa.eval.discretize), which is well defined on any monotone
    # image of the target, so the constraint is gone.
    xs_target = str(OmegaConf.select(cfg, "dataset.xs_target", default="uniform"))
    # Anchor the label at the view's FIRST row (the whole-session decoder).
    # Applies to train and eval alike: a head trained on a start-anchored
    # label has to be SCORED on one, or the eval reads a different variable.
    label_at_view_start = bool(
        OmegaConf.select(cfg, "dataset.label_at_view_start", default=False)
    )
    xs_eval_target = str(
        OmegaConf.select(cfg, "dataset.xs_eval_target", default="uniform")
    )
    if xs_anchor_stats_dir and extended_hours:
        # The anchor grid runs 09:30–16:00; an extended-hours view can end in
        # the pre/post session where no cross-section is tabulated.
        raise ValueError(
            "dataset.xs_anchor_stats_dir is not supported with "
            "dataset.extended_hours=true — the anchor grid covers the regular "
            "session only."
        )
    if extended_hours:
        for label, aug_list in (("augmentations", augmentations),
                                ("eval_augmentations", eval_augmentations)):
            for i, aug in enumerate(aug_list):
                if aug.get("name") != "random_resized_crop":
                    continue
                if aug.get("global_agg_range") is None:
                    raise ValueError(
                        f"dataset.extended_hours=true requires "
                        f"dataset.{label}.{i}.global_agg_range (e.g. [6,11]). "
                        "Without it, resolution is a fraction of the session "
                        "length, which varies ~1.5x across the extended session "
                        "and drifts outside the band the encoder was trained on."
                    )
                if aug.get("n_local_views", 0) > 0 and aug.get("local_agg_range") is None:
                    raise ValueError(
                        f"dataset.{label}.{i} has n_local_views>0 under "
                        "extended_hours but no local_agg_range (e.g. [2,23])."
                    )

    # --- Probe eval config ---
    eval_targets_config = {
        "horizons": list(cfg.probe_eval.targets.horizons),
        "types": list(cfg.probe_eval.targets.types),
    }

    # Supervised eval reads the task's target column from the eval dataset
    # (eval_step looks it up by name), so make sure the (type, horizon) of
    # the supervised task is included even when the user-provided
    # probe_eval.targets covers a different set (e.g. the default of
    # types=[return] hard-coded in slurm_train_bundle.sh).
    if is_supervised:
        for spec in task_specs.values():
            if spec.target_type not in eval_targets_config["types"]:
                eval_targets_config["types"].append(spec.target_type)
            if spec.horizon not in eval_targets_config["horizons"]:
                eval_targets_config["horizons"].append(spec.horizon)
    probe_num_threads = cfg.probe_eval.num_threads
    temporal_train_frac = cfg.probe_eval.temporal_train_frac

    if temporal_train_frac is not None and use_separate_probe_train:
        # When using separate probe train dates, temporal_train_frac is unused
        temporal_train_frac = 0.5  # placeholder, not used in this path

    if not use_separate_probe_train and temporal_train_frac is None:
        raise ValueError(
            "Must provide either eval_train_date_start/eval_train_date_end "
            "or set probe_eval.temporal_train_frac to split the eval set."
        )
    if temporal_train_frac is None:
        temporal_train_frac = 0.5

    ds_overrides = getattr(mode_cfg, "dataset_overrides", None)
    if is_supervised or is_ijepa or is_mae:
        train_targets_config = targets_config
        n_pairs_per_obs = 1
        one_global = {
            "name": "random_resized_crop",
            "n_global_views": 1,
            "n_local_views": 0,
            "global_scale_range": list(_global_scale_range),
            "global_seq_len": _global_seq_len,
            "global_agg_range": _global_agg_range,
        }
        augmentations = [dict(one_global)]
        # A mode that asks for CELLS gets them on the TRAIN side only. The
        # supervised specialist trains on cross_stock cells -- K labelled
        # stocks at one anchor, see SupervisedModeConfig.dataset_overrides --
        # while the eval/probe datasets keep one rrc global per row: the probe
        # scores one stock per row and eval_step reads views[0].
        if getattr(ds_overrides, "name", None) == "cross_stock":
            augmentations = [supervised_cell_view(
                one_global, int(ds_overrides.n_stocks or 2))]
        if not eval_augmentations:
            eval_augmentations = [dict(one_global)]
    else:
        train_targets_config = targets_config

    # Mode-level dataset overrides (e.g. CPC's single-view requirement —
    # see CPCModeConfig.dataset_overrides). Applied to every augmentation
    # entry in both the train and eval lists -- except that a cross_stock
    # request never reaches the eval list, whose rows must stay one stock
    # each (see above).
    if ds_overrides is not None:
        for field_name in ("n_global_views", "n_local_views", "name", "n_stocks"):
            value = getattr(ds_overrides, field_name, None)
            if value is None:
                continue
            for aug in augmentations:
                aug[field_name] = value
            if field_name == "n_stocks" or value == "cross_stock":
                continue
            for aug in eval_augmentations:
                aug[field_name] = value

    record_resolved_views(cfg, augmentations, eval_augmentations)

    if is_lejepa:
        resolve_lejepa_lambda(cfg, augmentations)

    seed = cfg.training.seed
    # Mode training overrides (batch size, LR scaling, WD schedule, etc.)
    overrides = mode_cfg.training_overrides

    # EXPLICIT PIN > MODE OVERRIDE > fallback, the same order blr resolves in.
    # This used to run the other way and a mode override beat the sweep, so
    # batch_size_stability swept a batch it never trained at. None now means
    # "not set", which is the only way an explicit 128 is distinguishable from
    # the old default of 128.
    batch_size = cfg.training.per_device_train_batch_size
    if batch_size is None:
        batch_size = overrides.per_device_train_batch_size
    if batch_size is None:
        batch_size = cfg.training.fallback_train_batch_size
    # Write it back so run names and logs report the batch that actually
    # trained -- the mode files build "bs=..." from the raw config.
    OmegaConf.update(cfg, "training.per_device_train_batch_size", batch_size,
                     merge=False)

    grad_accum_steps = 1
    if overrides.effective_batch_size is not None:
        grad_accum_steps = overrides.effective_batch_size // batch_size

    train_data_fraction = OmegaConf.select(cfg, "dataset.train_data_fraction", default=1.0)

    backend = str(OmegaConf.select(cfg, "dataset.backend", default="mds"))
    if backend not in ("mds", "days"):
        raise ValueError(f"dataset.backend must be 'mds' or 'days', got {backend!r}")
    if backend == "days":
        # Supervised cells from the day-major store: the writer precomputed
        # the targets, so no anchor tables, no risk factors, no grid cache.
        from .cell_dataset import DayStoreCellDataset

        daystore_dir = getattr(cfg.machine, "daystore_dir", None)
        if not daystore_dir:
            raise ValueError("dataset.backend=days needs machine.daystore_dir")
        if not (is_supervised or is_multi_supervised) or not (
                augmentations and augmentations[0].get("name") == "cross_stock"):
            raise ValueError(
                "dataset.backend=days serves supervised cross_stock cells only "
                "(mode=supervised|multi_supervised with dataset_overrides.name=cross_stock)")
        if train_data_fraction != 1.0 or extended_hours or label_at_view_start \
                or rf_kwargs["risk_factor_tickers"]:
            raise ValueError(
                "dataset.backend=days does not support train_data_fraction, "
                "extended_hours, label_at_view_start or risk_factor_tickers")
        train_dataset = DayStoreCellDataset(
            daystore_dir=daystore_dir,
            date_start=train_date_start,
            date_end=train_date_end,
            cell=canonicalize_augmentations(augmentations)[0],
            targets=train_targets_config,
            xs_target=xs_target,
            seed=seed,
            zero_feature_columns=rf_kwargs["zero_feature_columns"],
            norm_mode=rf_kwargs["norm_mode"],
            info_norm_stats=rf_kwargs["info_norm_stats"],
            info_window=rf_kwargs["info_window"],
            schedule=schedule,
        )
        print(f"dataset.backend=days: {train_dataset.n_days} days, "
              f"{len(train_dataset)} cells/epoch from {daystore_dir}")
    else:
        train_dataset = StreamingMarketDataset(
            augmentations=augmentations,
            date_start=train_date_start,
            date_end=train_date_end,
            seed=seed,
            n_pairs_per_obs=n_pairs_per_obs,
            targets=train_targets_config,
            epoch_dependent_seed=epoch_dependent_seed,
            schedule=schedule,
            streams=train_streams,
            shuffle=True,
            # Tie batch composition/order to the training seed. Without this,
            # StreamingDataset's shuffle_seed default (9176) pins the data order
            # across runs, so varying training.seed changes only init + augs.
            shuffle_seed=seed,
            # Only the TRAIN dataset: eval/probe run shuffle=False, where the
            # algorithm is not consulted.
            **({} if OmegaConf.select(cfg, "dataset.shuffle_algo", default=None) is None
               else {"shuffle_algo": cfg.dataset.shuffle_algo}),
            **({} if OmegaConf.select(cfg, "dataset.shuffle_block_size", default=None) is None
               else {"shuffle_block_size": int(cfg.dataset.shuffle_block_size)}),
            batch_size=batch_size,
            allow_unsafe_types=True,
            extended_hours=extended_hours,
            xs_anchor_stats_dir=xs_anchor_stats_dir,
            xs_target=xs_target,
            label_at_view_start=label_at_view_start,
            data_fraction=train_data_fraction,
            # Train only: this is where a sample is revisited every epoch. The
            # eval/probe datasets get one pass per eval and would just multiply
            # the per-worker RSS.
            grid_cache_gb=float(
                OmegaConf.select(cfg, "dataset.grid_cache_gb", default=0.0)
            ),
            **({} if predownload is None else {"predownload": predownload}),
            **({} if cache_limit is None else {"cache_limit": cache_limit}),
            **rf_kwargs,
        )

    eval_batch_size = cfg.training.per_device_eval_batch_size

    eval_dataset = StreamingMarketDataset(
        augmentations=eval_augmentations,
        date_start=eval_date_start,
        date_end=eval_date_end,
        seed=seed,
        n_pairs_per_obs=1,
        targets=eval_targets_config,
        schedule=schedule,
        streams=eval_streams,
        shuffle=False,
        batch_size=eval_batch_size,
        allow_unsafe_types=True,
        extended_hours=extended_hours,
        xs_anchor_stats_dir=xs_anchor_stats_dir,
        xs_target=xs_eval_target,
        label_at_view_start=label_at_view_start,
        **({} if predownload is None else {"predownload": predownload}),
        **({} if cache_limit is None else {"cache_limit": cache_limit}),
        **rf_kwargs,
    )

    probe_train_dataset = None
    # Only the live-eval probe consumes this. Constructing it regardless is not
    # free: every StreamingDataset claims a block of POSIX shm segments, and the
    # prefix scan that finds them is what blew the fd limit on 2026-08-15 (see
    # prepare_shm_limits in scripts/pythia/lib/common.sh) — the traceback landed
    # on exactly this call, the third of three datasets.
    if live_eval and use_separate_probe_train:
        probe_train_dataset = StreamingMarketDataset(
            augmentations=eval_augmentations,
            date_start=probe_train_date_start,
            date_end=probe_train_date_end,
            seed=seed,
            n_pairs_per_obs=1,
            targets=eval_targets_config,
            schedule=schedule,
            streams=probe_train_streams,
            shuffle=True,
            batch_size=eval_batch_size,
            allow_unsafe_types=True,
            extended_hours=extended_hours,
            xs_anchor_stats_dir=xs_anchor_stats_dir,
            xs_target=xs_eval_target,
            label_at_view_start=label_at_view_start,
            **({} if predownload is None else {"predownload": predownload}),
            **({} if cache_limit is None else {"cache_limit": cache_limit}),
            **rf_kwargs,
        )

    full_dataset = train_dataset
    # Probe eval target_names always come from the eval dataset (which uses
    # probe_eval.targets from config).  Training targets may be narrower
    # (supervised) or absent (JEPA), so we can't rely on train_dataset.target_names.
    target_names = eval_dataset.target_names

    def _resolve_target_col(names: list[str], spec, which: str) -> int:
        try:
            return names.index(spec.target_col)
        except ValueError:
            raise ValueError(
                f"Target column '{spec.target_col}' not found in the {which} "
                f"dataset's target_names."
            ) from None

    if is_multi_supervised:
        target_col_idx = {
            t: _resolve_target_col(train_dataset.target_names, spec, "train")
            for t, spec in task_specs.items()
        }
        eval_target_col_idx = {
            t: _resolve_target_col(target_names, spec, "eval")
            for t, spec in task_specs.items()
        }
    elif is_supervised:
        target_col_idx = _resolve_target_col(
            train_dataset.target_names, task_spec, "train"
        )
        eval_target_col_idx = _resolve_target_col(target_names, task_spec, "eval")

    n_train = len(train_dataset)
    n_eval = len(eval_dataset)

    eval_split_n = cfg.training.eval_split_n
    if eval_split_n is not None:
        max_eval_batches = max(1, math.ceil(eval_split_n / eval_batch_size))
    else:
        eval_split_frac = cfg.training.eval_frac
        max_eval_batches = max(1, math.ceil(n_eval * eval_split_frac / eval_batch_size))

    probe_train_n = cfg.training.probe_train_n
    if probe_train_n is not None:
        max_probe_train_batches = max(1, math.ceil(probe_train_n / eval_batch_size))
    else:
        max_probe_train_batches = max_eval_batches

    # -------------------------------------------------------------------------
    # Validate and compute max_train_steps
    # -------------------------------------------------------------------------
    # Same precedence as batch size. The override applies only when the config
    # said nothing at all: a sweep passing `num_epochs=null` beside
    # max_train_steps is choosing a STEP budget, and resurrecting epochs there
    # used to kill it on the mutual-exclusion check below.
    num_epochs = cfg.training.num_epochs
    max_train_steps = cfg.training.max_train_steps
    if num_epochs is None and max_train_steps is None:
        num_epochs = overrides.num_epochs
    if num_epochs is None and max_train_steps is None:
        num_epochs = cfg.training.fallback_num_epochs
    OmegaConf.update(cfg, "training.num_epochs", num_epochs, merge=False)

    if num_epochs is not None and max_train_steps is not None:
        raise ValueError("Only one of `num_epochs` or `max_train_steps` can be specified, not both.")
    if num_epochs is None and max_train_steps is None:
        raise ValueError("Either `num_epochs` or `max_train_steps` must be specified.")

    if is_supervised:
        n_pairs_per_obs = 1
    # Label-efficiency runs: an "epoch" covers only the available data subset,
    # so num_epochs-based training does proportionally less work (1% of the
    # data → 1% of the steps).
    n_train_avail = n_train
    if train_data_fraction < 1.0:
        n_train_avail = max(1, int(round(n_train * train_data_fraction)))
        print(
            f"train_data_fraction={train_data_fraction}: "
            f"{n_train_avail}/{n_train} observations available"
        )
    steps_per_epoch = max(1, n_train_avail // (batch_size * grad_accum_steps))

    if num_epochs is not None:
        max_train_steps = int(num_epochs * steps_per_epoch)
        print(f"Training for {num_epochs} epochs = {max_train_steps} steps ({steps_per_epoch} steps/epoch)")

    # ---- Intermediate checkpoints ----
    # Resolved HERE because this is the first line at which max_train_steps is
    # final for this run: it came from num_epochs x steps_per_epoch just above,
    # and steps_per_epoch already carries train_data_fraction. See
    # CheckpointConfig.save_fractions for why a label-budget sweep cannot use
    # literal steps.
    save_step_set = resolve_save_steps(
        cfg.checkpoint.save_steps, cfg.checkpoint.save_fractions, max_train_steps)
    if cfg.checkpoint.save_fractions:
        print(
            f"checkpoint.save_fractions={list(cfg.checkpoint.save_fractions)} "
            f"-> steps {sorted(save_step_set)} of {max_train_steps}"
        )

    # ---- Mega-epoch ----
    if cfg.dataset.mega_epoch and backend == "mds":
        # The daystore loader is an endless iterator over span-wide epochs;
        # the step count stops it, and there is nothing to resize.
        # Total dataloader iterations = steps * grad_accum (each step consumes
        # grad_accum_steps batches from the dataloader).
        total_dl_iters = max_train_steps * grad_accum_steps
        apply_mega_epoch(train_dataset, total_dl_iters, batch_size)

    num_workers = cfg.training.num_workers
    if getattr(cfg.machine, "num_workers", None) is not None:
        num_workers = cfg.machine.num_workers
    pin_memory = cfg.training.pin_memory

    if any(a.get("name") == "cross_stock" for a in augmentations):
        # A cross-stock batch holds K stacked view tensors (256 at the
        # sweep's constant view budget); with the default file_descriptor
        # sharing strategy every in-flight tensor costs the worker an open
        # fd, and prefetch_factor × K exhausts the per-process fd limit on
        # cluster nodes (killed pythia job 176422 with EMFILE). file_system
        # shares tensors by /dev/shm path instead.
        torch.multiprocessing.set_sharing_strategy("file_system")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_bucketed,
        persistent_workers=num_workers > 0,
        prefetch_factor=16 if num_workers > 0 else None,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_bucketed,
        persistent_workers=num_workers > 0,
        prefetch_factor=64 if num_workers > 0 else None,
    )

    if live_eval:
        print(f"Caching eval batches (max {max_eval_batches})...")
        cached_eval_batches = cache_batches(eval_dataloader, max_eval_batches)
        print(f"Cached {len(cached_eval_batches)} eval batches")
    else:
        # Not just "skip scoring": caching costs a full dataloader pass before
        # the first training step, on the CPUs this flag exists to free.
        cached_eval_batches = []
        print("training.live_eval=false — skipping eval batch caching")
    del eval_dataloader, eval_dataset

    cached_probe_train_batches = None
    if live_eval and probe_train_dataset is not None:
        probe_train_dataloader = DataLoader(
            probe_train_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_bucketed,
            persistent_workers=num_workers > 0,
            prefetch_factor=64 if num_workers > 0 else None,
        )
        print(f"Caching probe train batches (max {max_probe_train_batches})...")
        cached_probe_train_batches = cache_batches(probe_train_dataloader, max_probe_train_batches)
        print(f"Cached {len(cached_probe_train_batches)} probe train batches")
        del probe_train_dataloader, probe_train_dataset

    # -------------------------------------------------------------------------
    # Model (via Hydra instantiate)
    # -------------------------------------------------------------------------
    n_features = full_dataset.n_features
    # HOW MANY TRAILING COLUMNS ARE PER-WINDOW VALUES rather than a time
    # series. Taken from the dataset, not from the backbone config, because
    # only the dataset knows: it is 2 per normalization group for
    # dataset.info_norm_stats plus 3 for dataset.info_window, and a hand-set
    # backbone.n_info_channels that disagreed would silently route real feature
    # columns into the info token (or leave constants in the patch embedding)
    # with no shape error at all.
    # ASKED FOR, NOT DEFAULTED. This read was getattr(..., 0) until 2026-09-13,
    # and DayStoreCellDataset spelled the attribute `_n_info` -- so every
    # dataset.backend=days run silently built n_info_channels=0 against a
    # 20-column view, patch-embedding the 11 info columns as a time series with
    # no info_proj and no shape error. Exactly the failure the note above
    # describes. A dataset that cannot say how wide its info block is has no
    # safe default, so this raises instead of guessing.
    if not hasattr(full_dataset, "_n_info_features"):
        raise AttributeError(
            f"{type(full_dataset).__name__} does not expose _n_info_features; "
            "the backbone's n_info_channels comes from the dataset and has no "
            "safe default (see the note above)."
        )
    n_info_channels = int(full_dataset._n_info_features)

    # IJEPA owns its backbone config (always transformer + mean pool);
    # other modes use the top-level backbone config group.
    def _backbone_ctor_kwargs(bb_cfg):
        # `blr` is a config-time hint for LR scaling, not a constructor argument.
        container = OmegaConf.to_container(bb_cfg, resolve=True)
        container.pop("blr", None)
        # The dataset is the authority (see n_info_channels above). A config
        # that sets it is overridden rather than merged: two sources for one
        # number is how they end up different. Transformer only -- no other
        # backbone has an information token to route them into, and passing it
        # to one would be a TypeError at construction.
        if "TransformerBackbone" in str(container.get("_target_", "")):
            if "n_info_channels" in container or n_info_channels:
                container["n_info_channels"] = n_info_channels
        elif n_info_channels:
            raise ValueError(
                f"{n_info_channels} per-window columns are in the input but "
                f"{container.get('_target_')} has no information token to read "
                f"them; they would be fed to it as if they were a time series.")
        return container

    def _record_backbone_readout(container, key):
        """Fill in any readout knob the config left unset, and RECORD it.

        EVERY MODE OWNS ITS BACKBONE (2026-09-10), so pool and pos_embed are
        stated on the mode's own ``backbone`` field and arrive here already
        set: LeJEPA and the other SSL modes at mean_sin, the supervised arm at
        last_rope. This used to resolve a separate ``ModeBackboneOverrides``
        against the top-level backbone under EXPLICIT PIN > MODE OVERRIDE >
        schema, which meant two places could describe one knob and a reader
        had to know which mode used which. There is one place now.

        What remains is the WRITE-BACK. A knob still unset (cls_pos, or an
        explicit ``backbone=resnet`` that carries no readout) is filled from
        the backbone's own dataclass defaults and written onto cfg, so
        save_train_meta, architecture_signature and the saved config record
        what actually trained rather than a None a later reader would have to
        guess at. An explicit pin is never touched.
        """
        from market_jepa.modeling.backbones.transformer import TransformerConfig

        container = OmegaConf.select(cfg, key)
        if "TransformerBackbone" not in str(
                OmegaConf.select(cfg, f"{key}._target_") or ""):
            return container            # only the transformer has these knobs
        fallback = {"pool": "cls",
                    "pos_embed": TransformerConfig.pos_embed,
                    "cls_pos": TransformerConfig.cls_pos}
        for name in ("pool", "pos_embed", "cls_pos"):
            path = (f"{key}.config.{name}" if name in ("pos_embed", "cls_pos")
                    else f"{key}.{name}")
            if OmegaConf.select(cfg, path) is not None:
                continue                      # already stated; leave it alone
            OmegaConf.update(cfg, path, fallback[name], merge=False)
        return OmegaConf.select(cfg, key)

    # EVERY MODE OWNS A BACKBONE NOW, so this is the normal path and the
    # top-level cfg.backbone is a leftover of the "backbone" group selection.
    # It is NOT warned about any more: it used to fire on every run of the
    # seven modes that already owned one, which made it noise rather than a
    # signal. Selecting a non-default group (backbone=resnet) with a mode that
    # owns its backbone is still ignored, and still worth saying out loud.
    # Checked at the top of train() -- a field override here would be ignored.
    if owns_backbone(cfg):
        selected = str(OmegaConf.select(cfg, "backbone._target_") or "")
        owned = str(OmegaConf.select(cfg, "mode.backbone._target_") or "")
        if selected and owned and selected != owned:
            logger.warning(
                "Mode '%s' owns its backbone (%s); the selected top-level "
                "backbone (%s) is being ignored. Configure it at "
                "mode.backbone.* instead.",
                mode_cfg._target_.split(".")[-1],
                owned.rsplit(".", 1)[-1], selected.rsplit(".", 1)[-1],
            )
        bb_cfg = _record_backbone_readout(mode_cfg.backbone, "mode.backbone")
        backbone = instantiate(_backbone_ctor_kwargs(bb_cfg), n_features=n_features)
    else:
        bb_cfg = _record_backbone_readout(cfg.backbone, "backbone")
        backbone = instantiate(_backbone_ctor_kwargs(bb_cfg), n_features=n_features)
    mode_init_cfg = {
        k: v
        for k, v in OmegaConf.to_container(cfg.mode, resolve=True).items()
        if k not in {"backbone", "training_overrides", "dataset_overrides",
                     "backbone_overrides"}
    }
    model = instantiate(mode_init_cfg, backbone=backbone)
    model = model.to(device)

    # Validate model implements the required training interface
    _required_methods = ["training_step", "eval_step", "describe_parameters", "post_training_step"]
    for method_name in _required_methods:
        if not callable(getattr(model, method_name, None)):
            raise TypeError(
                f"Model {type(model).__name__} must implement {method_name}(). "
                f"See LeJEPA, IJEPA, or SupervisedModel for reference."
            )
    model._validate_instance_attrs()

    param_counts, param_summary = model.describe_parameters()
    print(param_summary)

    # -------------------------------------------------------------------------
    # Auto-generate run name
    # -------------------------------------------------------------------------
    run_name = cfg.wandb.run_name
    if run_name is None:
        bb_cfg = backbone_block(cfg)
        backbone_type = bb_cfg._target_.split(".")[-1].replace("Backbone", "").lower()
        run_name = model.default_run_name(backbone_type, cfg)

    model = maybe_compile(model, cfg.training)

    # -------------------------------------------------------------------------
    # Optimizer / Scheduler
    # -------------------------------------------------------------------------
    effective_batch = batch_size * grad_accum_steps * (1 if not model.uses_multi_view else n_pairs_per_obs)
    bb_cfg = backbone_block(cfg)
    backbone_blr = OmegaConf.select(bb_cfg, "blr", default=None)
    if cfg.optimizer.blr is not None:
        blr = cfg.optimizer.blr
    elif backbone_blr is not None:
        blr = backbone_blr
    else:
        blr = overrides.blr
    lr = scale_lr(
        blr,
        effective_batch,
        overrides.default_batch_size,
    )
    if overrides.default_batch_size is not None:
        print(f"Auto-scaled learning rate: {lr:.2e} (effective batch {effective_batch})")
    else:
        print(f"Using base learning rate without scaling: {lr:.2e}")

    weight_decay = (
        overrides.weight_decay_start if overrides.weight_decay_start is not None else cfg.optimizer.weight_decay
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_frac = cfg.optimizer.warmup_frac
    min_lr = 0.1 * lr
    warmup_start_frac = 0.01
    if is_ijepa:
        warmup_start_frac = 0.1
        min_lr = 0.001 * lr
    scheduler, num_warmup_steps = build_lr_scheduler(
        optimizer, max_train_steps, warmup_frac, min_lr, warmup_start_frac,
        schedule=OmegaConf.select(cfg, "optimizer.lr_schedule", default="cosine"),
        decay_frac=OmegaConf.select(cfg, "optimizer.decay_frac", default=0.2),
        warmup_steps=OmegaConf.select(cfg, "optimizer.warmup_steps", default=None),
    )
    # Written back RESOLVED, like blr and the batch: train_meta then records
    # where the ramp ended whether it was pinned or came from warmup_frac, so
    # a collector placing absolute-step checkpoints can tell which sit inside
    # it without re-deriving the trainer's rounding.
    OmegaConf.update(cfg, "optimizer.warmup_steps", int(num_warmup_steps), merge=False)

    # ---- Branch cooldowns (checkpoint.anneal_steps) ----
    # {branch_step: (total_steps, cooldown_steps)}; see CheckpointConfig.
    # Resolved here because it needs the warmup length the scheduler settled.
    anneal_plan = resolve_anneal_plan(
        OmegaConf.select(cfg, "checkpoint.anneal_steps", default=None),
        float(OmegaConf.select(cfg, "optimizer.anneal_frac", default=0.1)),
        max_train_steps, num_warmup_steps,
        OmegaConf.select(cfg, "optimizer.lr_schedule", default="cosine"))
    if anneal_plan:
        print("checkpoint.anneal_steps -> "
              + ", ".join(f"{T} (leave {b}, cool {n})" for b, (T, n) in sorted(anneal_plan.items())))
    anneal_min_ratio = min_lr / lr

    eval_frac = cfg.training.eval_frac
    eval_steps = max(1, int(max_train_steps * eval_frac))

    completed_steps = 0
    # Steps skipped in a row because the batch held nothing trainable. A
    # supervised cell run whose K exceeds the month's cross-section produces
    # NOTHING but such steps -- every draw fails, every loss is None -- and
    # without this it would run to max_train_steps, save an untrained
    # backbone and score it as a result (seen 2026-09-11 with K=256 on
    # 2012-12, a 247-stock month).
    consecutive_skipped = 0
    MAX_CONSECUTIVE_SKIPPED = 50

    # -------------------------------------------------------------------------
    # Bind the dataset's target column order onto the model, and — for a
    # binned loss only — fit the head's bin edges from the training data.
    # A scalar loss needs no calibration: it regresses onto the
    # cross-sectional z-score the dataset already emits.
    # -------------------------------------------------------------------------
    if is_supervised or is_multi_supervised:
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        expected = MultiTaskSupervisedModel if is_multi_supervised else SupervisedModel
        if not isinstance(m, expected):
            raise TypeError(
                f"Expected {expected.__name__} for this mode, got {type(m).__name__}"
            )
        m.target_col_idx = target_col_idx
        m.eval_target_col_idx = eval_target_col_idx
        if getattr(m, "binned", False):
            if is_multi_supervised:
                # target_col_idx is a {task: column} map here, and each task
                # needs bin edges fitted on ITS OWN target distribution.
                m.discretizer = calibrate_discretizers(
                    train_dataloader,
                    target_col_idx,
                    n_bins=m.n_bins,
                    n_batches=m.calibration_batches,
                )
            else:
                m.discretizer = calibrate_discretizer(
                    train_dataloader,
                    target_col_idx,
                    task_spec,
                    n_bins=m.n_bins,
                    n_batches=m.calibration_batches,
                )

    # -------------------------------------------------------------------------
    # W&B
    # -------------------------------------------------------------------------
    from wandb.sdk.wandb_settings import Settings as WandbSettings

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config.update(param_counts)
    wandb_config["n_features"] = n_features
    if is_supervised:
        from dataclasses import asdict

        if is_multi_supervised:
            wandb_config["task_specs"] = {t: asdict(s) for t, s in task_specs.items()}
        else:
            wandb_config["task_spec"] = asdict(task_spec)

    wandb_project = cfg.wandb.project
    # An explicit settings.mode overrides WANDB_MODE, so read the env var here:
    # otherwise a box without an API key cannot train at all. "shared" stays the
    # default, so cluster runs are unaffected.
    wandb_mode = os.environ.get("WANDB_MODE") or "shared"
    # NOTHING BEFORE THE FIRST TRAINING STEP MAY COST A JOB. wandb.init reaches
    # api.wandb.ai, and its 90 s default is not generous enough for a sweep
    # whose jobs all start together: on 2026-08-29 a CommError out of this call
    # killed the k2-lambda sweep's 2021-08 k2ind run outright -- after staging,
    # before a single step, with the queue slot burned and the month left as a
    # hole in the panel that nothing on disk explained.
    #
    # Third instance of one root cause. pythia's compute nodes have flaky
    # outbound connectivity, and every external call a job makes before
    # training must survive it: `uv sync` reaching github.com (retry loop in
    # scripts/pythia/slurm_train.sh) and the skip_if_done probe reaching
    # api.wandb.ai (broad except in train.py) were the first two.
    #
    # AND IF EVERY RETRY FAILS, TRAIN OFFLINE RATHER THAN DIE. The deliverable
    # is the checkpoint and the xs_ic.json the in-job scorer writes beside it;
    # those are what every plot reads, and the W&B row is a convenience. Losing
    # hours of GPU time because a log sink was unreachable is the worse trade.
    # The cost is that an offline run is invisible to skip_if_done, so a later
    # resubmission repeats it -- redoing work beats losing it.
    _WANDB_INIT_TRIES = 4
    for _attempt in range(1, _WANDB_INIT_TRIES + 1):
        try:
            wandb.init(
                settings=WandbSettings(
                    mode=wandb_mode, x_primary=True, init_timeout=180
                ),
                project=wandb_project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                name=run_name,
                config=wandb_config,
            )
            break
        except Exception as exc:
            print(
                f"wandb.init failed (attempt {_attempt}/{_WANDB_INIT_TRIES}): "
                f"{type(exc).__name__}: {exc}"
            )
            if _attempt == _WANDB_INIT_TRIES:
                print("wandb.init: giving up on W&B; training OFFLINE.")
                wandb.init(
                    settings=WandbSettings(mode="offline", x_primary=True),
                    project=wandb_project,
                    entity=cfg.wandb.entity,
                    group=cfg.wandb.group,
                    name=run_name,
                    config=wandb_config,
                )
                break
            time.sleep(30 * _attempt)
    wandb.define_metric("*", step_metric="obs_seen")

    max_grad_norm = cfg.optimizer.max_grad_norm

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    model_unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    mode_label = model_unwrapped.mode_label
    mode_str = model_unwrapped.mode_str
    print(f"\nStarting training ({mode_str}):")
    print(f"  Total steps: {max_train_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Learning rate: {lr:.2e}")
    print(f"  Batch size: {batch_size} x {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective")
    print(f"  Train samples: {n_train}, Eval samples: {n_eval}")
    if cfg.training.log_every_frac is not None:
        log_every_n_steps = max(1, int(max_train_steps * cfg.training.log_every_frac))
    elif cfg.training.log_every_n_steps is not None:
        log_every_n_steps = int(cfg.training.log_every_n_steps)
    else:
        raise ValueError("Either `log_every_frac` or `log_every_n_steps` must be specified.")
    print(f"  Log train metrics every: {log_every_n_steps} steps\n")

    obs_seen = completed_steps * batch_size * grad_accum_steps

    last_probe_procs: list[tuple[subprocess.Popen, Path]] = []
    probe_log_paths: list[Path] = []

    if live_eval and cfg.training.initial_eval and completed_steps == 0:
        print("Running initial evaluation...")
        eval_metrics = model_unwrapped.eval_step(cached_eval_batches, device)
        wandb.log({**eval_metrics, "obs_seen": obs_seen})
        first_key = next(iter(eval_metrics))
        print(f"  Initial eval loss: {eval_metrics[first_key]:.4f}")

        # Initial probe eval
        last_probe_procs = dispatch_probe_eval(
            model,
            cached_eval_batches,
            device,
            obs_seen=obs_seen,
            wandb_run_id=wandb.run.id,
            wandb_project=wandb_project,
            wandb_entity=cfg.wandb.entity,
            target_names=target_names,
            temporal_train_frac=temporal_train_frac,
            probe_train_batches=cached_probe_train_batches,
            num_threads=probe_num_threads,
        )
        probe_log_paths.extend(lp for _, lp in last_probe_procs)

    accumulator = MetricsAccumulator(log_every=log_every_n_steps)
    progress_bar = tqdm(total=max_train_steps, initial=completed_steps, desc=mode_label)
    model.train()

    # HOW FAR THE ENCODER HAS MOVED FROM ITS INIT, logged with every flush.
    # The 2026-09-11 audit found a whole supervised campaign whose backbones
    # ended 0.15% from where they started: the loss (pinned near log 2 for
    # any weak ranking signal) and the ridge probe (which reads random
    # features perfectly well) both looked normal. This number is the one
    # that would have said so on the first log line.
    backbone_drift = (ParamDrift(model_unwrapped.backbone)
                      if getattr(model_unwrapped, "backbone", None) is not None
                      else None)

    data_iter = iter(train_dataloader)

    accum_step = 0
    optimizer.zero_grad()

    while completed_steps < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if cfg.dataset.mega_epoch and backend == "mds":
                raise RuntimeError(
                    f"DataLoader exhausted at step {completed_steps}/{max_train_steps}. "
                    f"This should not happen with mega-epoch sizing. "
                    f"epoch_size={train_dataset.epoch_size}, "
                    f"batch_size={batch_size}"
                )
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        virtual_epoch = completed_steps // steps_per_epoch

        # ----- Unified training step -----
        result = model_unwrapped.training_step(batch, device, grad_accum_steps=grad_accum_steps)

        if result is None:
            # No valid data in this batch — reset accumulated gradients
            # to prevent stale grads from contaminating the next window.
            if accum_step > 0:
                optimizer.zero_grad()
                accum_step = 0
            logger.warning(
                "Non-finite loss at step %d, skipping optimizer step",
                completed_steps,
            )
            consecutive_skipped += 1
            if consecutive_skipped >= MAX_CONSECUTIVE_SKIPPED:
                raise RuntimeError(
                    f"{consecutive_skipped} consecutive steps had no trainable "
                    "batch (None from training_step). The loader is producing "
                    "nothing usable -- for supervised cells, check that "
                    "n_stocks does not exceed the month's cross-section.")
            # Still advance schedule so training time stays predictable
            # and schedule-dependent hyperparams stay in sync.
            scheduler.step()
            if overrides.weight_decay_end is not None:
                wd_start = (
                    overrides.weight_decay_start
                    if overrides.weight_decay_start is not None
                    else cfg.optimizer.weight_decay
                )
                wd_end = overrides.weight_decay_end
                frac = completed_steps / max(1, max_train_steps - 1)
                wd = wd_start + frac * (wd_end - wd_start)
                for param_group in optimizer.param_groups:
                    param_group["weight_decay"] = wd
            extra_metrics = model_unwrapped.post_training_step(completed_steps, max_train_steps)
            completed_steps += 1
            obs_seen += batch_size * grad_accum_steps
            progress_bar.update(1)
            continue

        accum_step += 1
        consecutive_skipped = 0

        if accum_step < grad_accum_steps:
            continue

        # ----- Optimizer step (after grad_accum_steps micro-batches) -----
        accum_step = 0
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Linear weight decay schedule (e.g. I-JEPA: 0.04 → 0.4)
        if overrides.weight_decay_end is not None:
            wd_start = (
                overrides.weight_decay_start if overrides.weight_decay_start is not None else cfg.optimizer.weight_decay
            )
            wd_end = overrides.weight_decay_end
            frac = completed_steps / max(1, max_train_steps - 1)
            wd = wd_start + frac * (wd_end - wd_start)
            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = wd

        # Post-step hooks (e.g. EMA for I-JEPA — noop for others)
        extra_metrics = model_unwrapped.post_training_step(completed_steps, max_train_steps)

        completed_steps += 1
        obs_seen += batch_size * grad_accum_steps
        progress_bar.set_description(f"{mode_label} [epoch {virtual_epoch}]")
        progress_bar.update(1)

        if cfg.checkpoint.save_model and completed_steps in save_step_set:
            wandb_id = wandb.run.id if wandb.run else "local"
            step_ckpt_path = os.path.join(
                cfg.checkpoint.chkpt_dir, wandb_project, wandb_id, str(completed_steps)
            )
            save_checkpoint(model, step_ckpt_path)
            save_train_meta(
                cfg, step_ckpt_path,
                run_name=run_name, wandb_run_id=wandb_id,
                wandb_project=wandb_project,
                augmentations=augmentations,
                max_train_steps=max_train_steps,
                completed_steps=completed_steps,
                obs_seen=obs_seen,
            )

        # ----- Branch cooldown (checkpoint.anneal_steps) -----
        # Leave the stable run here, decay to min_lr over n steps on the
        # batches the loader yields next, save the cooled model as the
        # checkpoint at T = completed_steps + n total steps, and put the run
        # back. The last branch is not restored: the loop's own end is the
        # cooled model, so the run root IS <run>/<T_last>/.
        if completed_steps in anneal_plan:
            anneal_total, anneal_n = anneal_plan[completed_steps]
            is_last_branch = anneal_total == max_train_steps
            stable_lrs = [g["lr"] for g in optimizer.param_groups]
            snap = None if is_last_branch else take_snapshot(model, optimizer)
            print(f"==> branch at step {completed_steps}: cooling {anneal_n} steps "
                  f"to {anneal_total} (lr {stable_lrs[0]:.2e} -> "
                  f"{stable_lrs[0] * anneal_min_ratio:.2e})"
                  + ("" if is_last_branch else ", then back"))
            cool_losses = []
            k = 0
            branch_skipped = 0
            while k < anneal_n:
                f = cooldown_factor(k, anneal_n, anneal_min_ratio)
                for g, lr0 in zip(optimizer.param_groups, stable_lrs):
                    g["lr"] = lr0 * f
                # One optimizer step of grad_accum_steps micro-batches, the
                # main loop's step without its bookkeeping.
                got = 0
                while got < grad_accum_steps:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        if cfg.dataset.mega_epoch and backend == "mds":
                            raise RuntimeError(
                                f"DataLoader exhausted inside the cooldown to {anneal_total}")
                        data_iter = iter(train_dataloader)
                        batch = next(data_iter)
                    r = model_unwrapped.training_step(batch, device, grad_accum_steps=grad_accum_steps)
                    if r is None:
                        optimizer.zero_grad()
                        got = 0
                        branch_skipped += 1
                        if branch_skipped >= MAX_CONSECUTIVE_SKIPPED:
                            raise RuntimeError(
                                f"{branch_skipped} untrainable batches in a row inside "
                                f"the cooldown to {anneal_total}")
                        continue
                    got += 1
                    branch_skipped = 0
                    cool_losses.append(float(r["metrics"].get("train/loss", float("nan"))))
                nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                model_unwrapped.post_training_step(completed_steps + k, max_train_steps)
                k += 1
            wandb_id = wandb.run.id if wandb.run else "local"
            branch_path = os.path.join(
                cfg.checkpoint.chkpt_dir, wandb_project, wandb_id, str(anneal_total))
            save_checkpoint(model, branch_path)
            save_train_meta(
                cfg, branch_path,
                run_name=run_name, wandb_run_id=wandb_id,
                wandb_project=wandb_project,
                augmentations=augmentations,
                max_train_steps=max_train_steps,
                # The total this model trained: the stable steps plus its
                # cooldown, which is what its directory is named after and
                # what its compute is billed on.
                completed_steps=anneal_total,
                obs_seen=obs_seen + anneal_n * batch_size * grad_accum_steps,
            )
            n_fin = [x for x in cool_losses if x == x]
            cool_mean = (sum(n_fin) / len(n_fin)) if n_fin else float("nan")
            print(f"==> branch to {anneal_total}: mean cooldown loss {cool_mean:.4f}, "
                  f"lr ended at {optimizer.param_groups[0]['lr']:.2e}")
            if wandb.run:
                wandb.log({f"anneal/{anneal_total}/loss": cool_mean,
                           f"anneal/{anneal_total}/lr_end": optimizer.param_groups[0]["lr"],
                           "obs_seen": obs_seen})
            if is_last_branch:
                # The run ends on the cooled model.
                completed_steps = max_train_steps
                obs_seen += anneal_n * batch_size * grad_accum_steps
                progress_bar.update(anneal_n)
            else:
                restore_snapshot(model, optimizer, snap)
                del snap
                for g, lr0 in zip(optimizer.param_groups, stable_lrs):
                    g["lr"] = lr0
                print(f"==> branch to {anneal_total} saved; stable run resumes at step {completed_steps}")

        metrics = {
            "obs_seen": obs_seen,
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm.item(),
            "train/virtual_epoch": virtual_epoch,
        }
        if overrides.weight_decay_end is not None:
            metrics["train/weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        metrics.update(result["metrics"])
        metrics.update(extra_metrics)
        accumulator.record(metrics)

        # ----- Eval -----
        if live_eval and completed_steps % eval_steps == 0:
            eval_metrics = model_unwrapped.eval_step(cached_eval_batches, device)
            model.train()
            wandb.log({**eval_metrics, "obs_seen": obs_seen})

            # Probe eval (both modes)
            if target_names:
                # Wait for previous probe workers before spawning new ones
                for prev_proc, prev_log in last_probe_procs:
                    prev_proc.wait()
                    if prev_proc.returncode != 0:
                        log_tail = ""
                        if prev_log and prev_log.exists():
                            log_tail = prev_log.read_text()[-500:]
                        logger.warning(
                            "Probe worker exited with code %d. Log tail:\n%s",
                            prev_proc.returncode,
                            log_tail,
                        )
                last_probe_procs = dispatch_probe_eval(
                    model,
                    cached_eval_batches,
                    device,
                    obs_seen=obs_seen,
                    wandb_run_id=wandb.run.id,
                    wandb_project=wandb_project,
                    wandb_entity=cfg.wandb.entity,
                    target_names=target_names,
                    temporal_train_frac=temporal_train_frac,
                            probe_train_batches=cached_probe_train_batches,
                    num_threads=probe_num_threads,
                )
                probe_log_paths.extend(lp for _, lp in last_probe_procs)

        if accumulator.should_log():
            flushed = accumulator.flush()
            if backbone_drift is not None:
                flushed["train/backbone_drift"] = backbone_drift()
            wandb.log(flushed)

    progress_bar.close()

    if accumulator._steps_since_log > 0:
        flushed = accumulator.flush()
        if backbone_drift is not None:
            flushed["train/backbone_drift"] = backbone_drift()
        wandb.log(flushed)
    elif "flushed" not in locals():
        flushed = {}
    final_drift = backbone_drift() if backbone_drift is not None else None
    if final_drift is not None:
        print(f"Backbone drift from init (relative L2): {final_drift:.5f}")
        wandb.run.summary["train/backbone_drift"] = final_drift
    # The last logged window, in the job log: W&B's own end-of-run table
    # truncates, and an offline run has no other readable record of whether
    # the batch was doing work (rows_with_grad_frac, loss_units_per_step).
    print("final train metrics: "
          + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in sorted(flushed.items())}))

    # Final probe eval after dataloader finishes
    if live_eval and target_names:
        for prev_proc, prev_log in last_probe_procs:
            prev_proc.wait()
            if prev_proc.returncode != 0:
                log_tail = ""
                if prev_log and prev_log.exists():
                    log_tail = prev_log.read_text()[-500:]
                logger.warning(
                    "Probe worker exited with code %d. Log tail:\n%s",
                    prev_proc.returncode,
                    log_tail,
                )
        last_probe_procs = dispatch_probe_eval(
            model,
            cached_eval_batches,
            device,
            obs_seen=obs_seen,
            wandb_run_id=wandb.run.id,
            wandb_project=wandb_project,
            wandb_entity=cfg.wandb.entity,
            target_names=target_names,
            temporal_train_frac=temporal_train_frac,
            probe_train_batches=cached_probe_train_batches,
            num_threads=probe_num_threads,
        )
        probe_log_paths.extend(lp for _, lp in last_probe_procs)

    if cfg.checkpoint.save_model:
        wandb_id = wandb.run.id if wandb.run else "local"
        ckpt_path = os.path.join(cfg.checkpoint.chkpt_dir, wandb_project, wandb_id)
        save_checkpoint(model, ckpt_path)
        save_train_meta(
            cfg, ckpt_path,
            run_name=run_name, wandb_run_id=wandb_id,
            wandb_project=wandb_project,
            task_name=task_spec.name if is_supervised and not is_multi_supervised else None,
            # The list the datasets were BUILT from, after
            # mode.dataset_overrides rewrote the view counts.
            augmentations=augmentations,
            backbone_drift=final_drift,
            # The endpoint: completed == max, so a progress plot can place the
            # run root at fraction 1.0 without inferring it from the step
            # directories (which stop at the largest fraction asked for).
            max_train_steps=max_train_steps,
            completed_steps=completed_steps,
            obs_seen=obs_seen,
        )
        if cfg.checkpoint.remote_dir and wandb.run:
            remote_path = os.path.join(cfg.checkpoint.remote_dir, wandb_project, wandb_id)
            wandb.run.summary["checkpoint_path"] = f"bll01:{remote_path}"

    # Wait for all final probe workers
    if last_probe_procs:
        print(f"Waiting for {len(last_probe_procs)} final probe eval worker(s) to finish...")
        for proc, lp in last_probe_procs:
            try:
                proc.wait(timeout=1000)
            except subprocess.TimeoutExpired:
                logger.warning("Final probe worker timed out after 1000s, killing it")
                proc.kill()
            if proc.returncode and proc.returncode != 0:
                log_tail = ""
                if lp and lp.exists():
                    log_tail = lp.read_text()[-500:]
                logger.warning(
                    "Final probe worker exited with code %d. Log tail:\n%s",
                    proc.returncode,
                    log_tail,
                )

    failed_probes = False
    for log_path in probe_log_paths:
        try:
            if log_path.exists() and log_path.stat().st_size == 0:
                failed_probes = True
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
    if failed_probes:
        logger.warning("Some probe workers produced empty logs — likely crashed silently")

    wandb_run_id = wandb.run.id if wandb.run else None
    wandb.finish()

    print("\nTraining complete!")
