"""Hydra structured config dataclasses for market-jepa."""

from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Any

from omegaconf import MISSING

# ABSOLUTE, because hydra chdirs into its output directory before anything
# reads this. A relative "data/market_holidays.csv" resolves against the repo
# root only for a process that never moved -- which is every local script and
# no training job. The migration moved the file from scripts/data/ to data/
# and kept the path relative, so every job died in MarketSchedule with
# FileNotFoundError after the GPU was already allocated.
_REPO_ROOT = _Path(__file__).resolve().parents[1]
HOLIDAY_CSV = str(_REPO_ROOT / "data" / "market_holidays.csv")


# ── Machine ──────────────────────────────────────────────────────────────────


@dataclass
class MachineConfig:
    mosaic_dir: str = MISSING
    risk_factor_dir: str = MISSING
    metadata_path: str = MISSING
    holiday_csv: str = MISSING
    num_workers: int | None = None
    # The day-major store (stable_finance.dataset.daystore): one record per
    # trading day with the whole cross-section and its targets built in.
    # Read by dataset.backend=days; None where it has not been staged.
    daystore_dir: str | None = None


def machine_from_env(base: type[MachineConfig]) -> MachineConfig:
    """``base()`` with its data paths overridden from the environment.

    Analysis scripts under plots/ hardcode ``BLL01MachineConfig()`` because
    that is where they are normally run. A cluster job stages the mosaic to a
    node-local directory instead, so those scripts cannot run there at all --
    on 2026-08-20 the latent sweep died reading
    /data/lab/market-jepa-mosaic/... on a node that does not mount /data/lab.

    This is a FUNCTION, deliberately, not env-aware defaults on the dataclasses:
    those are Hydra structured configs and non-literal defaults would change how
    training's ConfigStore registration behaves. With none of the variables set,
    every field is exactly the base config's own default.
    """
    import os

    cfg = base()
    for var, attr in (("MJ_MOSAIC_DIR", "mosaic_dir"),
                      ("MJ_RISK_FACTOR_DIR", "risk_factor_dir"),
                      ("MJ_METADATA_PATH", "metadata_path")):
        val = os.environ.get(var)
        if val:
            setattr(cfg, attr, val)
    return cfg


HUB_DENSE = "hf://datasets/fin-ai-lab/Market-1T-1Hz-2019H2-2020-dense"
HUB_DAYSTORE = "hf://datasets/fin-ai-lab/Market-1T-1Hz-2019H2-2020-daystore"


@dataclass
class HubMachineConfig(MachineConfig):
    """The released Market-1T data, read straight from the Hugging Face Hub.

    stable-finance resolves ``hf://`` roots itself, downloading only the months
    a run asks for (into ``$HF_HOME``), so any machine can train on the
    released window (2019-07 -> 2020-12) with no staging step. There is no
    risk-factor store or metadata table on the Hub; neither is read unless
    ``dataset.risk_factor_tickers`` is set.
    """

    mosaic_dir: str = f"{HUB_DENSE}/1Hz_mosaic_mnth"
    risk_factor_dir: str = ""
    metadata_path: str = ""
    holiday_csv: str = HOLIDAY_CSV
    daystore_dir: str | None = f"{HUB_DAYSTORE}/1Hz_daystore"


@dataclass
class RacoonMachineConfig(MachineConfig):
    mosaic_dir: str = "/mnt/spinning/ts_jepa/polygon/snapshots/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/mnt/spinning/ts_jepa/polygon/snapshots/1Hz_risk_factors"
    metadata_path: str = "/mnt/spinning/ts_jepa/polygon/snapshots/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV


@dataclass
class BLL01MachineConfig(MachineConfig):
    mosaic_dir: str = "/data/lab/market-jepa-mosaic/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/data/lab/market-jepa-mosaic/1Hz_risk_factors"
    metadata_path: str = "/data/polygon/snapshots/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV
    daystore_dir: str | None = "/data/lab/market-jepa-mosaic/1Hz_daystore"


@dataclass
class A100Cluster40GBMachineConfig(MachineConfig):
    """8xA100 40GB cluster (124 vCPUs ⇒ 15 CPUs/job, 8 concurrent).

    Paths are overridden at submit time by
    ``scripts/generic/slurm_train_bundle.sh`` from ``$CLUSTER_DATA``;
    defaults here just match the standard ``/home/ubuntu`` layout.

    num_workers 6 → 13 on 2026-08-27. Six workers against a 15-CPU
    reservation is the exact starvation the H100 config's docstring warns
    about -- the GPUs waited while the cores idled. This box has the FEWEST
    cores of the three (124 vs 208/240) and the same 1771 GiB of RAM, so it
    is the one that must not waste a core; the surplus memory goes into a
    12 GiB/worker grid cache (.env.40GB), enough to hold a whole month.

    40 GB OF VRAM IS A REAL CONSTRAINT, unlike the 80GB boxes: the
    big-batch single-view modes (MAE, I-JEPA at per-device 1024) peaked at
    71.6 GB measured on the H100 and DO NOT FIT here. Send them to
    a100_80gb, or halve per_device_train_batch_size and double the
    accumulation (effective batch is preserved, and this pipeline is CPU-
    bound so the smaller batch costs almost nothing).
    """

    mosaic_dir: str = "/home/ubuntu/market-jepa-data/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/home/ubuntu/market-jepa-data/1Hz_risk_factors"
    metadata_path: str = "/home/ubuntu/market-jepa-data/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV
    num_workers: int | None = 12  # 15 GiB cache x 12 + main =~ 209G of the 215G job


@dataclass
class A100Cluster80GBMachineConfig(MachineConfig):
    """8xA100 80GB cluster (240 vCPUs ⇒ 29 CPUs/job, 8 concurrent).

    num_workers 8 → 26 on 2026-08-27, same reasoning as the 40GB box: eight
    workers against a 29-CPU reservation starved the GPU. This is the
    core-RICHEST box (240), so it carries the heavy modes.

    RAM is the same 1771 GiB as the other two but split 26 ways, so its
    grid cache is 6 GiB/worker against the 40GB box's 12 (.env.80GB). That
    is deliberate: a miss costs 5.2 ms/sample against 1.8 ms warm, but
    worker count beats cache depth while the CPU is the binding constraint.
    """

    mosaic_dir: str = "/home/ubuntu/market-jepa-data/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/home/ubuntu/market-jepa-data/1Hz_risk_factors"
    metadata_path: str = "/home/ubuntu/market-jepa-data/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV
    num_workers: int | None = 12  # 15 GiB cache x 12 + main =~ 209G of the 215G job


@dataclass
class H100ClusterMachineConfig(MachineConfig):
    """8xH100 80GB cluster (208 vCPUs ⇒ 24 CPUs/job, 8 concurrent).

    num_workers is sized to the 24-CPU budget, not the old 7-CPU pythia
    ratio. 2026-08-27: 20 → 14 — the grid caches are PER WORKER, and at 20
    workers the RAM budget capped them at 5 GiB each, a measured 62% hit
    rate against ~12 GiB for a full month (5736 of ~13.6k sessions
    resident). 14 workers x 12 GiB (.env.h100 GRID_CACHE_GB) hold the
    whole month within the 215G job budget (14 x ~13.2G RSS + main ≈
    195G), and the freed cores go to the now multi-process post-train IC
    eval. The A100 boxes reserved 26 CPUs but ran only 8 workers, so the
    dataloader starved the GPU while the cores idled — don't repeat that
    here either.
    """

    mosaic_dir: str = "/home/ubuntu/market-jepa-data/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/home/ubuntu/market-jepa-data/1Hz_risk_factors"
    metadata_path: str = "/home/ubuntu/market-jepa-data/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV
    num_workers: int | None = 12  # 15 GiB cache x 12 + main =~ 209G of the 215G job


@dataclass
class PythiaMachineConfig(MachineConfig):
    mosaic_dir: str = "/hpc_temp/bll/polygon/snapshots/1Hz_mosaic_mnth"
    risk_factor_dir: str = "/hpc_temp/bll/polygon/snapshots/1Hz_risk_factors"
    metadata_path: str = "/hpc_temp/bll/polygon/snapshots/metadata.parquet"
    holiday_csv: str = HOLIDAY_CSV
    daystore_dir: str | None = "/hpc_temp/bll/polygon/snapshots/1Hz_daystore"
    num_workers: int | None = 6  # 7 CPUs/GPU on Pythia; 6 workers + main.
    # Was 5 to leave a core for the probe subprocess; production sweeps run
    # training.live_eval=false, so that core goes back to the dataloader.


# Single source of truth for all machine configs. To add a new machine,
# define a `<Name>MachineConfig(MachineConfig)` dataclass above and add
# one entry here: `"<name>": <Name>MachineConfig`. Both the Hydra
# ConfigStore registration in `train.py` and the CLI lookup in
# the offline evals under `scripts/eval/` consume this mapping directly.
MACHINE_CONFIGS: dict[str, type[MachineConfig]] = {
    "hub": HubMachineConfig,
    "racoon": RacoonMachineConfig,
    "bll01": BLL01MachineConfig,
    "a100_40gb": A100Cluster40GBMachineConfig,
    "a100_80gb": A100Cluster80GBMachineConfig,
    "h100_cluster": H100ClusterMachineConfig,
    "pythia": PythiaMachineConfig,
}


# ── Targets ──────────────────────────────────────────────────────────────────


@dataclass
class TargetsConfig:
    horizons: list[int] = field(default_factory=lambda: [300, 600, 900])
    types: list[str] = field(default_factory=lambda: ["return", "spread_change", "volatility_change"])


# The FF49 industry map behind the standard k2ind pairing. Relative to the
# repo root, which is where hydra runs; rebuilt from Compustat sich and
# spanning 2007-01..2024-12 (stable_finance.dataset.industry, `sf-industry-map`).
STANDARD_INDUSTRY_TABLE = "data/industry_map.parquet"


# ── Augmentations ────────────────────────────────────────────────────────────


@dataclass
class AugmentationConfig:
    name: str = "random_resized_crop"
    # Relative sampling weight when several augmentations are configured:
    # each drawn pair picks augmentation i with probability
    # weight_i / sum(weights). Ignored (uniform) when all weights are equal.
    weight: float = 1.0
    n_global_views: int = 2
    n_local_views: int = 6
    global_seq_len: int = 2048
    global_scale_range: list[float] = field(default_factory=lambda: [0.5, 1.0])
    local_scale_range: list[float] = field(default_factory=lambda: [0.05, 0.5])
    local_seq_len: int = 512
    # Absolute resolution band, inclusive, in seconds per output token. When
    # set, it replaces the corresponding *_scale_range: resolution is sampled
    # directly instead of being derived from the session length. Required for
    # extended-hours training, where the session length varies by 1.5x and a
    # fixed fraction of it no longer pins the resolution. None = fractional.
    global_agg_range: list[int] | None = None
    local_agg_range: list[int] | None = None
    # random_resized_crop only. Snap each GLOBAL view's last row to a
    # wall-clock lattice of this many seconds past the 09:30 open, so the
    # label anchor lands where the cross-sectional (mu, sigma) tables are
    # tabulated. Left None here and forced to AnchorSpec.step_seconds by the
    # dataset whenever dataset.xs_anchor_stats_dir is set — the two are
    # meaningless apart, so they are not independently settable.
    end_grid_sec: int | None = None
    # random_resized_crop only. Rows that must remain after the view end.
    # Setting it to the training horizon keeps every drawn sample inside the
    # no-clamp region instead of throwing away the ~31% (at h=7200) whose
    # forward window would run past the close. Purely a training-efficiency
    # knob: the eval cross-sections apply the same rule as a row filter.
    end_min_slack_sec: int = 0
    # random_resized_crop only, and the MIRROR of the two fields above: snap
    # each GLOBAL view's FIRST row to the anchor lattice and require the label
    # horizon to fit after it. Set by the dataset (never by hand) when
    # dataset.label_at_view_start is on; end_grid_sec is left alone then,
    # because a view has exactly one label anchor.
    start_grid_sec: int | None = None
    start_min_slack_sec: int = 0
    # cross_stock only: number of distinct same-date tickers whose identical
    # wall-clock window forms one positive group (the group's "views").
    n_stocks: int = 2
    # cross_stock only: path to a parquet with columns month (YYYY-MM),
    # ticker, ff49. When set, partner selection is restricted to tickers in
    # the same FF49 industry as the focal stock. Falls back to an
    # unrestricted draw when the focal is unmapped or no same-industry
    # partner aligns (so thin industries degrade to plain cross_stock
    # rather than dropping samples).
    industry_table: str | None = None
    # cross_stock only: number of matched local views. Deliberately a
    # SEPARATE knob from n_local_views — that field's rrc-oriented default
    # (6) would otherwise leak into cross_stock configs that never asked
    # for locals (this silently turned the first plain-K2 seed-replication
    # jobs into locals runs). When > 0, local views are sub-crops of the
    # group's shared wall-clock window drawn at MATCHED positions — each
    # local slot picks one sub-window and cuts it from every group stock
    # (must be divisible by n_stocks; slots = value / n_stocks).
    # local_scale_range is interpreted as a fraction of the shared window.
    cross_stock_local_views: int = 0
    # cross_stock only. False: all views pulled to the global mean (flat
    # multi-crop). True: pairwise edge loss instead — global<->global,
    # matched local<->local (cross-stock, same sub-window), and local<->own
    # stock's global.
    structured_matching: bool = False
    # cross_stock only, and only with end_grid_sec (i.e. under anchor tables):
    # draw the label anchor uniformly over the feasible lattice band FIRST and
    # the resolution among what fits second -- the eval panel's own draw --
    # instead of drawing the window and then an anchor it fits before, which
    # skews cells toward the close. On by the supervised specialist
    # (training.utils.supervised_cell_view); off for the SSL pairings.
    anchor_uniform: bool = False
    # time_warp / gaussian_noise only: same-stock two-global-view
    # augmentations. Both mirror cross_stock's window draw — ONE shared
    # wall-clock window (global_scale_range x global_seq_len) — but emit
    # n_global_views copies of the SAME stock, so the transformation is the
    # only thing that differs between views.
    # time_warp: each view re-aggregates the window on a randomly warped
    # bucket grid — warp_knots piecewise-linear segments whose interior
    # boundaries shift by up to warp_strength x the knot spacing, locally
    # slowing down / speeding up the series (volume/n totals preserved).
    warp_knots: int = 8
    # 0.75, NOT the 0.25 this defaulted to until 2026-08-27. Every time_warp
    # run this project has ever scored used 0.75 -- the sweep scripts pinned it
    # -- so 0.25 was a default nobody ran, and an unpinned run inherited a
    # strength no result was measured at.
    #
    # IT IS NOT A VALIDATED OPTIMUM. 0.75 is the sweeps/tw_noise_hpo.sh winner
    # from Jan 2023, ranked by return_900_k5 AUC on the mid-to-mid target -- a
    # metric retired and a target replaced. Strength has NEVER been swept under
    # rank IC. It is set here to match what runs, not because it was chosen.
    #
    # It is also strong: the docstring on _warped_aggregate_numpy calls >= 0.5
    # "increasingly fold-prone", so at 0.75 the monotonicity clamp is actively
    # truncating knot displacements and the effective warp is less than nominal.
    # Strength and lejepa lambda both set invariance pressure (see
    # LeJEPAModeConfig.lamb), so they need sweeping together, not separately.
    warp_strength: float = 0.75
    # gaussian_noise: iid N(0, noise_sigma^2) added to every SERIES channel
    # of each view AFTER per-view normalization (units = post-norm std, which
    # is why every series channel measures a post-norm SD of 1.00 and takes
    # the same relative hit).
    #
    # THE INFORMATION CHANNELS ARE EXEMPT, since 2026-08-29. They used to be
    # noised too -- the draw was taken at the full view shape, which includes
    # the per-group (mu, sigma) and the three window descriptors -- and
    # "units = post-norm std" was never true of them: they are log-scale and
    # fraction-scale constants that no normalization touches. Every
    # gaussian_noise number reported before that date is on the old behaviour.
    # See the comment at the application site in streaming_dataset.py.
    #
    # 0.75, NOT the 0.1 this defaulted to until 2026-08-27 -- the same gap
    # warp_strength had, and larger: every scored gaussian_noise run pinned
    # 0.75, so the default was 7.5x below anything measured. Units are post-norm
    # standard deviations, so 0.75 puts the noise variance at ~56% of the
    # signal's. Like warp_strength it is the stale Jan-2023 HPO winner under a
    # retired metric, never swept under rank IC, and set here to match what runs.
    noise_sigma: float = 0.75
    # volume_noise: iid Exponential additions to the volume and n channels
    # (pre-normalization), mean = vol_noise_frac x the window's mean bucket
    # value — fake trades at the prevailing price, prices untouched.
    vol_noise_frac: float = 0.5
    # price_jitter: per-bucket level noise added identically to all five
    # price channels, sigma = price_jitter_frac x that bucket's half-spread
    # (pre-normalization). The ladder moves as one, so the spread is
    # preserved exactly and the book can never cross.
    price_jitter_frac: float = 1.0
    # channel_drop: each view independently zeroes each of the 9 feature
    # channels with this probability AFTER normalization (>= 1 channel
    # always kept; post-norm zeros = flatten to the window mean).
    channel_drop_p: float = 0.2


# ── Dataset ──────────────────────────────────────────────────────────────────


@dataclass
class DatasetConfig:
    # WHERE THE TRAIN VIEWS COME FROM.
    #   "mds"   the ticker-day mosaic through StreamingMarketDataset: every
    #           mode, every augmentation.
    #   "days"  the day-major store (machine.daystore_dir) through
    #           DayStoreCellDataset: supervised CELLS only, with the targets
    #           the writer precomputed. The eval/probe datasets stay on the
    #           mosaic either way -- they score one stock per row.
    # Chosen per run, not per machine, so a mosaic run and a daystore run of
    # the same recipe can be compared on the same node.
    backend: str = "mds"
    train_date_start: str = "2023-01-01"
    train_date_end: str = "2023-01-31"
    # TRAINING SPAN, in months, ending at train_date_end -- i.e. the N months
    # BEFORE the eval month. The launchers resolve train_date_start from it
    # (run_span_bundle.sh's MAX_SPAN, the supervised_days_* sweeps'
    # SPAN_MONTHS), so the span is written here and nowhere else.
    #
    # SIX, settled 2026-09-13 across all three tasks. New ticker-days are the
    # lever for return (single-month arms sit at the floor on 2020-05 and
    # 2022-02; six months lifts both), and six is where the span stopped
    # paying: twelve months at 7 passes and six at 12 are the same head IC,
    # and six is half the staging and half the wall clock.
    train_span_months: int = 6
    # Fraction of distinct training observations available (label-efficiency
    # experiments). A deterministic, seed-stable subset of the train window's
    # samples is used, and an "epoch" covers only that subset — with
    # num_epochs-based training, 1% of the data means 1% of the optimizer
    # steps. (max_train_steps-based training is unaffected.) 1.0 = all data.
    train_data_fraction: float = 1.0
    eval_date_start: str = "2023-02-01"
    eval_date_end: str = "2023-02-28"
    eval_train_date_start: str = "2023-01-01"
    eval_train_date_end: str = "2023-01-31"
    risk_factor_tickers: list[str] = field(default_factory=list)
    risk_factor_columns: str | None = None
    # Risk-factor PRICE columns normally also generate risk-adjusted TARGET
    # columns, which the cross-sectional anchor tables do not tabulate — so
    # xs_anchor_stats_dir refuses to run alongside them. Setting this False
    # keeps the risk factor as an INPUT channel only (_merge_risk_factors is
    # gated on the risk factors existing, not on target generation), which is
    # what "give the model a market proxy" actually needs. The two are
    # independent in the dataset; only the target side conflicts.
    risk_factor_targets: bool = True
    # What the model is FIT to, on the anchor-snapped geometry that
    # xs_anchor_stats_dir imposes either way:
    #   "zscore" — (y - mu_t) / sigma_t, the cross-sectional z
    #   "uniform"— rankdata(y_cell) / (n_cell + 1), average ties
    #   "rank"   — Phi^-1(F_t(y)), the Gaussian rank within the cross-section
    #   "raw"    — y untouched
    # Rank IC is invariant to all four (each is monotone within a cell), so
    # this changes the regression geometry but not within-cell rank IC.
    # "uniform" needs exact order statistics and "rank" needs quantiles in the
    # anchor tables. Target and geometry are deliberately separate:
    # they used to be coupled, so changing the target also unsnapped the
    # views and moved two things at once.
    #
    # "uniform" is the standard target for both supervised heads and probes.
    # It is a DATASET default rather than a supervised-mode one because
    # ModeDatasetOverrides patches augmentation entries, not top-level dataset
    # keys -- but it costs the SSL modes nothing: they never read the target in
    # their loss.
    #
    # THE BINNED SUPERVISED LOSSES STILL NEED "raw" and must say so. Bins are
    # equal-count quantiles, and binning a uniform rank is a different label
    # from binning the return.
    xs_target: str = "uniform"
    # What the EVAL and PROBE datasets emit. This stays separate so experiments
    # can deliberately compare target transforms, but the standard recipe uses
    # the same empirical-uniform target for head training, ridge fitting, and
    # eval.
    xs_eval_target: str = "uniform"
    # Feature-ablation knob: names of FEATURE_COLUMNS zeroed in the dense
    # 1 Hz grid before any augmentation/normalization, in every dataset
    # (train, eval, probe) so train and eval always match. Channel count is
    # unchanged — the columns just carry zeros (e.g. ["bid_size", "ask_size"]
    # removes the order-book size information the volume-confound analysis
    # flagged as an identity shortcut).
    zero_feature_columns: list[str] = field(default_factory=list)
    # How each view is standardized before the model sees it.
    #   "per_view"  every feature GROUP (price / book size / volume / n) is
    #               z-scored by THAT VIEW'S OWN mean and std, with log1p on
    #               the three size groups first. The default, and what every
    #               model trained before 2026-08-22 used.
    #   "none"      no centring, no scaling, no log1p. Raw grid units reach
    #               the model: dollar prices, share counts in the millions.
    #               An ablation, not a recipe -- it exists to measure what
    #               per-view standardization costs, and it removes the
    #               absolute-level information a cross-sectional rank metric
    #               rewards along with the conditioning the optimizer needs.
    # Rides in the shared dataset kwargs so train, eval and probe agree; a
    # model trained under one mode must be SCORED under it too (see
    # xs_ic_eval.iter_panel's norm_groups).
    norm_mode: str = "per_view"
    # ── THE INFORMATION TOKEN ──────────────────────────────────────────────
    #
    # Both flags below contribute numbers to ONE token: no position embedding,
    # dropped before pooling. They travel to the
    # backbone in reserved columns at the final valid timestep; the values are
    # not repeated along the series. TransformerBackbone splits them off before
    # the patch embedding (see backbone.n_info_channels, which
    # pretrain derives from the dataset so the two cannot disagree).
    #
    # WHY THEY ARE NOT CHANNELS, which is what the old name said. Until
    # 2026-08-25 the (mu, sigma) really were broadcast along all 2048 timesteps
    # and fed through the patch embedding, moving n_features 9 -> 17: the same
    # eight numbers repeated 2048 times, widening every patch for facts that
    # have no location in the window. One token says it once.
    #
    # ON BY DEFAULT since 2026-08-25. Off, a model reads the standardized view
    # and nothing else, and standardization is exactly what deletes the price
    # level, the spread's absolute width and the activity level -- the
    # quantities a cross-sectional metric ranks stocks on, and the ones every
    # classical baseline that beats the encoder gets to keep.

    # Per-view (mu, sigma) that norm_mode="per_view" divides out: 2 per
    # normalization group, 8 in total. The substantive half -- they are per
    # stock, so they vary within a scoring cell and can rank.
    info_norm_stats: bool = True
    # Three numbers ABOUT the window: start fraction, end fraction, and
    # log(seconds per token). Constant within a scoring cell (cell_agg is
    # seeded from (date, anchor) alone), so they rank nothing directly -- but
    # the resolution is the one with a mechanism: a random resized crop samples
    # 5.7-11.4 s/token, so a fixed number of patches is a 2x-smeared wall-clock
    # window unless the model is told the scale.
    info_window: bool = True
    # The pre-2026-08-25 spellings of the two fields above -- norm_stats_channels
    # and time_info -- were carried here as None-defaulted aliases so that jobs
    # queued before the rename would not die on a hydra struct-mode error
    # (run_sweep.sh rsyncs the repo on every submit, so a pending job runs
    # today's code against its own frozen overrides). Deleted 2026-09-06 with
    # both cluster queues empty. THE READERS STILL ACCEPT BOTH SPELLINGS and
    # must keep doing so -- 128 supervised checkpoints record the old names in
    # their train_meta, and that is a historical record, not a live config.
    # See save_train_meta, xs_ic_eval._dataset_flag, post_train_ic_eval.
    # Build the 1 Hz grid over the extended session (04:00 ET to four hours
    # past the close) instead of regular hours only. Pair with
    # augmentations.*.global_agg_range — see AugmentationConfig.
    extended_hours: bool = False
    # Directory of monthly cross-sectional anchor tables (YYYY-MM.npz, built
    # by stable_finance.dataset.build_targets, `sf-build-targets`). When set, every target
    # column is standardized against its own cross-section,
    # z = (y_i - mu_t) / sigma_t, which IS the training objective and the
    # thing rank IC is computed on. Also forces global view ends onto the
    # anchor lattice, since off-lattice ends have no (mu, sigma) to divide by.
    xs_anchor_stats_dir: str | None = None
    # HOW MDS ORDERS SAMPLES, and it interacts with how the mosaic is stored.
    # None keeps StreamingDataset's own default ("py1e"), which shuffles ACROSS
    # shards -- correct for an i.i.d. sampler and destructive for a
    # cross-sectional one, because it scatters a date's tickers over the epoch.
    #
    # "py1s" shuffles the SHARD ORDER and keeps a shard's samples together. On
    # a date-ordered mosaic (stable_finance.dataset.reshard) a shard is one
    # date, so a batch stays date-local and _within_cell_loss finds pairs:
    # measured ~48 usable pairs per 256-row batch on the shuffled mosaic,
    # which is the whole of the head-vs-probe gap. On the SHUFFLED mosaic
    # "py1s" buys nothing -- a shard there already holds ~19 dates.
    #
    # Left at None: the reported checkpoints were trained this way, and the
    # date-ordered layout is not yet the default mosaic.
    shuffle_algo: str | None = None
    # SHUFFLE GRANULARITY, and on a date-ordered mosaic it is the dial between
    # decorrelation and pair supply. Measured on 2011-03 (batch 256, by-date
    # mosaic, py1br), against the shuffled mosaic's 23.0 dates / 73.7 pairs:
    #
    #     block   dates/batch   within-cell pairs/batch
    #      4096      20.7             106
    #      1024      10.0             196
    #       256       7.3             363
    #       128       5.7             427   <- knee
    #        64       5.7             457
    #     (shuffle=False)  1.3       1261   <- ceiling, no epoch randomization
    #
    # Smaller blocks mix less, so a batch holds fewer distinct dates and more
    # comparable pairs. 128 takes ~5.8x the pairs while still spreading a batch
    # over ~6 days, which is what keeps the gradient from being one regime.
    #
    # None keeps StreamingDataset's default (1 << 18), which on any layout is
    # far larger than a month and shuffles globally.
    shuffle_block_size: int | None = None

    # Take the label at the view's FIRST row instead of its last.
    #
    # Every other target in this project is measured forward from where the
    # view ENDS: the model sees the past and predicts the future. The
    # event world model's DECODER is the one exception — it reads a day's
    # realized cross-sectional return off that day's latent, so the view is
    # the day and the label is measured forward from the day's open, which is
    # the view's beginning. Turning this on snaps the global view's start
    # (not its end) to the anchor lattice and pins the slack to the horizon
    # measured from there.
    #
    # Only meaningful with a single-horizon target set and
    # random_resized_crop globals; see market_jepa/eval/tasks.py::DAY_HORIZON.
    label_at_view_start: bool = False
    n_pairs_per_obs: int = 1
    epoch_dependent_seed: bool = True
    predownload: int = 1024
    # Cap on the LOCAL shard cache, as streaming states it ("500gb", or
    # bytes); None leaves it unbounded, which is the default and was fine
    # while a run read one month. It is NOT fine over a decade: MDS
    # decompresses every zstd shard it touches into the dataset directory
    # and never evicts, so a ten-year mosaic inflates ~5x and fills the
    # disk mid-run. With a limit, streaming evicts the coldest shards
    # instead of growing, at the cost of re-decompressing them later.
    cache_limit: str | None = None
    mega_epoch: bool = True
    # Per-WORKER cache of the dense 1 Hz grid, in GiB (0 disables). The grid
    # for a (ticker, date) is deterministic and gets rebuilt once per epoch —
    # 100-200x over a run — and building it is ~half of a dataloader worker's
    # CPU, which is what holds the GPU at ~38% duty cycle. Cached entries are
    # float32 (bitwise exact, see tests/test_float32_grid_identity.py) at
    # ~936 KB each. 20 GiB covers the LARGEST month in the archive whole
    # (2022-03, 21791 sessions = 20.4 GB; median month is 12.0 GB), so
    # after one epoch a worker rebuilds nothing. Applies to the TRAIN
    # dataset only; the job pays num_workers x this in RSS, which is why
    # slurm_train_bundle.sh asks for 240G.
    # OFF SINCE THE DENSE MOSAIC. The cache existed to amortize the
    # sparse->dense grid rebuild across repeat visits to a ticker-day, and
    # that rebuild was 71% of a worker's per-sample CPU (2.89 ms of 4.08 ms).
    # The dataset now stores the reconstructed session, so there is nothing
    # left to amortize and this only takes memory from the reader --
    # num_workers x this in RSS, which is what made a job ask for 240G and
    # what stranded ~29 GiB per job when a scancel left DataLoader children
    # behind. Set it above 0 only when reading a SPARSE dataset.
    grid_cache_gb: float = 0.0
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    # Element type is Any (not AugmentationConfig) so extra entries can be
    # appended from the CLI (+dataset.augmentations.1.name=cross_stock ...)
    # for multi-augmentation mixing — OmegaConf rejects plain-dict values
    # under a dataclass-typed dict. Entry "0" is still a full
    # AugmentationConfig; appended entries are open dicts whose missing
    # knobs fall back to the per-name defaults in streaming_dataset's
    # parser, and whose name is validated against AUGMENTATION_REGISTRY.
    #
    # THE DEFAULT IS THE STANDARD MODEL: k2ind — cross_stock K=2 restricted to
    # the focal's FF49 industry. The joint-embedding methods (LeJEPA, DINO,
    # BYOL — the three with uses_multi_view) form their positive pair from TWO
    # DIFFERENT STOCKS over the same wall-clock window, not from two crops of
    # one ticker-day plus six locals. That older scheme is retired: it is the
    # one that let an encoder solve the pretext task by recognizing the firm,
    # which is exactly the shortcut the metric could not see past.
    #
    # Both schemes still sample their window with a random resized crop —
    # that never went away and applies to every mode. What changed is only
    # what counts as a POSITIVE PAIR, which is what this field names.
    #
    # Single-view modes (MAE, I-JEPA, CPC, TS2Vec, CoST, TFC, TimeMAE, the
    # supervised heads, the frozen TSFM and finance baselines) never form a
    # pair, so each pins itself back to random_resized_crop through
    # mode.dataset_overrides.name and is unaffected by this default.
    #
    # Still fully overridable per-sweep — the crop-family ablations set
    # dataset.augmentations.0.name=random_resized_crop explicitly, and the k2
    # control sets industry_table=null.
    augmentations: dict[str, Any] = field(default_factory=lambda: {"0": AugmentationConfig(
        name="cross_stock", n_stocks=2, industry_table=STANDARD_INDUSTRY_TABLE)})
    # Eval crops differ from train crops only in dropping the local views —
    # global_scale_range inherits the training default [0.5, 1.0]. (It used to
    # widen to [0.3, 1.0] at eval; that mismatch is gone, so any component
    # calibrated against the eval crop distribution shares the training one.)
    # Eval NEVER pairs: it embeds one stock's window at a time, so it stays on
    # the plain crop sampler whatever the training augmentation is. Named
    # explicitly rather than inherited, because the training default is now
    # cross_stock and an inherited name would silently make the probe read
    # partner tickers.
    eval_augmentations: dict[str, AugmentationConfig] = field(
        default_factory=lambda: {
            "0": AugmentationConfig(name="random_resized_crop", n_local_views=0)
        }
    )


# ── Training ─────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    seed: int = 42
    reproducible: bool = True
    # ── None MEANS "NOT SET", AND THAT IS LOAD-BEARING ─────────────────────
    #
    # These two used to default to 100 and 128, which made an explicit pin
    # indistinguishable from an unset field. A mode override then had to win to
    # be useful, so `if overrides.X is not None: X = overrides.X` in pretrain.py
    # SHADOWED the sweep -- batch_size_stability swept a batch it never trained
    # at, variance_decomp asked for 100 epochs at 128 and got 200 at 256, and
    # every sweep passing `num_epochs=null` beside max_train_steps died on
    # "Only one of num_epochs or max_train_steps can be specified".
    #
    # With None as the default the precedence can be the obvious one, and it now
    # matches how blr already resolved: EXPLICIT PIN > MODE OVERRIDE > the
    # fallbacks below. pretrain.py writes the resolved values back onto the
    # config so run names and logs report what actually trained.
    num_epochs: int | None = None
    max_train_steps: int | None = None
    per_device_train_batch_size: int | None = None
    # Used only when neither the config nor the mode override says anything.
    #
    # TWELVE PASSES OVER THE SPAN, for every method (2026-09-13). The budget
    # is a property of the COMPARISON, not of a method: each method gets the
    # same six months of ticker-days and the same number of passes over them,
    # so a difference in IC is a difference in representation rather than in
    # how much data the arm happened to see. It was 100, which paired with
    # the old single-month window; on a six-month span that is 600
    # month-passes and well past where every task saturates.
    fallback_num_epochs: int = 12
    fallback_train_batch_size: int = 128
    per_device_eval_batch_size: int = 1024
    eval_frac: float = 0.05
    eval_split_n: int = 4096
    probe_train_n: int = 4096
    initial_eval: bool = True
    # Run the in-training eval and probe at all. Off for production sweeps:
    # the probe is a CPU-bound subprocess competing with the dataloader on a
    # node with 7 CPUs to one H100, and neither of its numbers is what gets
    # reported -- the in-training probe fits on 4096 rows and under-reports IC
    # by ~2.4x, while the pooled eval IC carries a 0.018 sampling SD. The
    # synchronized cross-section pass after training supersedes both.
    #
    # OFF BY DEFAULT since 2026-09-09. It defaulted True, and every production
    # sweep then carried training.live_eval=false -- a line repeated in each
    # file whose absence silently costs cores rather than erroring. Production
    # is the common case and the rare interactive run can ask for the curve
    # with training.live_eval=true.
    #
    # THIS IS THE SWITCH pretrain.py:234 READS. LiveEvalConfig.enabled is a
    # dead dataclass nothing consults; changing it does nothing.
    #
    # IT DOES NOT TOUCH POST-TRAINING SCORING. The reported number comes from
    # scripts/generic/post_train_ic_eval.py, which the SLURM bundle runs after
    # training on the same node under POST_TRAIN_IC_EVAL (default 1).
    live_eval: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    log_every_n_steps: int | None = None
    log_every_frac: float | None = 0.001
    compile: bool = True
    compile_mode: str = "default"


# ── Optimizer ────────────────────────────────────────────────────────────────


@dataclass
class OptimizerConfig:
    blr: float | None = None
    weight_decay: float = 5e-2
    warmup_frac: float = 0.05
    max_grad_norm: float = 1.0
    # "cosine" (the reported recipe) or "wsd" (warmup-stable-decay).
    #
    # WSD IS FOR BUDGET LADDERS. Under cosine the LR differs at every step, so
    # a mid-run checkpoint is not a model anyone intended to stop at, and a
    # label-budget curve needs one RUN per budget. A constant stable phase
    # makes the checkpoints within a run comparable to each other, so one run's
    # checkpoint ladder is the budget ladder -- which is the difference
    # between 558 runs and 93. See training/utils.py:build_lr_scheduler for
    # the caveat that survives (stable-phase checkpoints are un-annealed).
    #
    # "stable" is warmup then the peak rate to the last step, with NO decay
    # in the run itself: the decays are BRANCHES (checkpoint.anneal_steps).
    lr_schedule: str = "cosine"
    # Fraction of the run spent decaying, WSD only.
    decay_frac: float = 0.2
    # BRANCH COOLDOWN LENGTH, as a fraction of the branch's total step count
    # (checkpoint.anneal_steps). A checkpoint at T total steps is produced by
    # leaving the stable run at T - round(anneal_frac * T), decaying linearly
    # to min_lr over the remaining steps, saving, and putting the stable run
    # back where it was. 0.1 is the WSD decay this codebase already used at
    # the end of a run (decay_frac), applied per branch.
    anneal_frac: float = 0.1
    # Warmup as an ABSOLUTE step count; beats warmup_frac when set.
    #
    # A checkpoint ladder in absolute steps (checkpoint.save_steps, the
    # scaling sweep) is only comparable across months if the LR at step s is
    # the same in every month -- and under warmup_frac it is not: 5% of a
    # 2,373-step month ends at step 119, 5% of a 5,358-step month at 268, so
    # a rung at 181 is past the ramp in one month and halfway up it in
    # another. Pinning the count makes the schedule identical up to the
    # decay in every run of the sweep. Fractions of the run (save_fractions,
    # the finetune ladder) do not need this and leave it None.
    warmup_steps: int | None = None


# ── Checkpoint ───────────────────────────────────────────────────────────────


@dataclass
class CheckpointConfig:
    chkpt_dir: str = "checkpoints"
    remote_dir: str | None = None
    save_model: bool = True
    # Absolute optimizer steps at which to also save. Each lands in
    # <chkpt_dir>/<project>/<run_id>/<step>/ — a SUBDIRECTORY of the run, so
    # the final checkpoint still sits at the run root and anything that
    # rsyncs the run dir carries the intermediates with it.
    save_steps: list[int] = field(default_factory=list)
    # The same thing as FRACTIONS of the run's own length, resolved once
    # max_train_steps is known.
    #
    # WHY NOT JUST save_steps. A sweep whose axis is the LABEL BUDGET has a
    # different total step count in every arm — steps_per_epoch is
    # n_train_avail // (batch x accum), so it moves with the fraction AND with
    # the month's cross-section. A literal step list would mean a different
    # point on the training curve in each arm, which is precisely what a
    # progress plot must not do. Fractions are resolved against each run's own
    # max_train_steps, so 0.1 is a tenth of training everywhere.
    #
    # Resolution rounds to the nearest step, clamps into [1, max_train_steps]
    # and de-duplicates: a short arm can round several fractions onto the same
    # step, and saving it twice would write the same weights to two names.
    #
    # NOTE ON WHAT AN INTERMEDIATE CHECKPOINT IS. The schedule is warmup +
    # cosine (see training/utils.py:build_lr_scheduler), so a checkpoint taken
    # part-way through has NOT annealed — it is a snapshot of a run still at
    # high LR, not a converged model trained on that much data. It answers
    # "is the encoder moving, and how fast", which is a diagnostic. The
    # converged number for a given budget is the ENDPOINT of the arm trained
    # at that budget. Reading a mid-run checkpoint as a budget result is the
    # mistake this comment exists to prevent; it needs a stable schedule and
    # a decay branch, which is anneal_steps below.
    save_fractions: list[float] = field(default_factory=list)
    # ANNEALED CHECKPOINTS AT ABSOLUTE STEP COUNTS, the WSD scaling method.
    # Every entry T is the TOTAL number of optimizer steps the saved model has
    # trained: the stable run is left at T - n (n = round(anneal_frac * T)),
    # the learning rate decays linearly to min_lr over n steps on the batches
    # the loader yields next, the model lands in <run>/<T>/, and the stable
    # run resumes from the weights and optimizer state it left -- so every
    # entry is a finished model at its own compute, from ONE run. The last
    # entry must equal training.max_train_steps and is not restored: the run
    # root is that model, so the root duplicates <run>/<T_last>/.
    #
    # Requires lr_schedule="stable" (a run that decays on its own would hand
    # the branches a moving rate) and every branch point past the warmup.
    # The cooldown's batches are consumed from the same loader: the stable
    # run does not see them again, which is the same as skipping n steps of
    # data order, and no more.
    anneal_steps: list[int] = field(default_factory=list)


# ── W&B ──────────────────────────────────────────────────────────────────────


@dataclass
class WandbConfig:
    project: str = "lejepa-v1"
    entity: str = "boothai"
    run_name: str | None = None
    group: str | None = None


# ── Probe Eval ───────────────────────────────────────────────────────────────


@dataclass
class ProbeEvalTargetsConfig:
    horizons: list[int] = field(default_factory=lambda: [300, 600, 900])
    types: list[str] = field(default_factory=lambda: ["return", "spread_change", "volatility_change"])
    risk_factor_tickers: list[str] = field(default_factory=list)


@dataclass
class ProbeEvalConfig:
    num_threads: int = 10
    temporal_train_frac: float = 0.5
    targets: ProbeEvalTargetsConfig = field(default_factory=ProbeEvalTargetsConfig)


@dataclass
class LiveEvalConfig:
    """In-training eval and probe — monitoring only, off for production sweeps.

    Neither number is reported. The in-training probe fits on
    ``training.probe_train_n`` rows (4096), which under-reports IC by ~2.4x
    against a full-month fit, and the in-training eval scores ~3.3k rows, whose
    Spearman sampling s.e. is 0.016 — larger than most differences it is asked
    to resolve. The synchronized cross-section pass
    (``scripts/eval/xs_ic_eval.py``) refits on the whole month and
    reports per-cell IC with an s.e. near 0.005, so it supersedes both.

    Turning them off matters because these nodes run 8 CPUs to one GPU: the
    probe's ridge fits and the eval dataloader compete with training for the
    same scarce cores. Checkpoints are still written, which is all the offline
    pass needs.
    """

    # DEAD CONFIG, kept only because it is referenced in prose. NOTHING READS
    # THIS CLASS -- not pretrain.py, not any yaml. The switch that actually
    # gates the in-training probe is TrainingConfig.live_eval; set that one.
    enabled: bool = True


# ── Backbone Config Groups ──────────────────────────────────────────────────


@dataclass
class TransformerInnerConfig:
    _target_: str = "market_jepa.modeling.backbones.transformer.TransformerConfig"
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 8
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.1
    rescale_residual_init: bool = False
    # Fixed sin/cos table instead of one randomly-initialised vector per slot.
    #
    # THIS DEFAULT AND TransformerConfig's MUST AGREE. Hydra instantiates the
    # backbone through THIS dataclass, so it is the one a run actually reads;
    # the other is what a directly-constructed backbone (tests, eval loaders)
    # gets. A knob that moved in only one of them is the failure 396939b is
    # named after.
    pos_embed: str | None = None
    # Where the CLS token sits FOR THE POSITION EMBEDDING: "own" (slot 0, what
    # every prior checkpoint used), "last" (the final patch's vector, so the
    # readout is positioned at the end of the view), or "none".
    cls_pos: str | None = None
    # Init scale of the LEARNED position table (ignored when sinusoidal).
    # THIS DEFAULT AND TransformerConfig's MUST AGREE.
    pos_init_std: float = 0.02
    # Freeze the patch projection to an identity, so a token IS the flattened
    # patch. Requires hidden_size == n_features * patch_size. A merely SQUARE
    # projection is still a learned matrix and tests nothing about whether the
    # projection earns its place; this removes it.
    patch_embed_identity: bool = False


@dataclass
class TransformerBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.transformer.TransformerBackbone"
    d_embedding: int = 384
    # BACK TO "cls" on 2026-08-22, the same day it was changed to "last".
    #
    # The evidence for "last" was a paired +0.0179 over the cls control on
    # return_900 (t=+3.7, positive in all five months) -- the only arm in the
    # whole view-ablation sweep that moved return. That number was measuring a
    # bug in the TARGET, not the readout.
    #
    # The return was mid-to-mid, mid = (bid+ask)/2 at one instant, so a quote
    # sitting at an extreme of its recent range reverted inside the spread and
    # the reversion read as a real return. "last" reads the final patch
    # directly instead of routing it through CLS, which is a SHARPER VIEW OF
    # THAT INSTANT -- exactly what the artifact rewarded. On the forward-VWAP
    # target the same paired comparison is -0.0026 +/- 0.0021 (t=-1.2): the
    # advantage is gone and if anything reversed. A swing of -0.0205.
    #
    # See docs/return_bad_calculation.md. "cls" is also what every archived
    # checkpoint in this project uses, so this restores reproducibility.
    pool: str | None = None
    # Mask each patch to the patches at or before it. Only meaningful with
    # pool="last": the CLS token is prepended at index 0, so under a causal
    # mask it attends to nothing but itself and the readout is a constant --
    # the backbone raises on that combination rather than training it.
    causal: bool = False
    # Project the RAW final timestep into its own token appended after the
    # patches. patch_size timesteps share a patch, so the value at the anchor
    # is mixed with its neighbours; this gives it a path of its own.
    state_token: bool = False
    # Concatenate the first difference of every channel to the input. Every
    # classical predictor that works on this panel is a difference or a
    # position within a range; the encoder is handed levels and has to learn
    # to difference them.
    diff_channels: bool = False
    # How many TRAILING input channels are per-window constants, routed to a
    # single information token instead of through the patch embedding. Set it
    # to whatever dataset.info_norm_stats contributes (8) to move the
    # per-view (mu, sigma) out of the time series; more columns can be
    # appended by the dataset later -- seconds per token, session fraction,
    # ticker or industry identity -- and only this number changes.
    #
    # The token carries no position embedding, because a fact about the whole
    # window has no location in it.
    n_info_channels: int = 0
    max_seq_len: int = 2048
    config: TransformerInnerConfig = field(default_factory=TransformerInnerConfig)


# ── The ViT scale ladder (scripts/sweeps/supervised_scaling.sh) ────────────
#
# WIDTH, HEADS AND MLP WIDTH ONLY. Depth (12), patch (8), the sequence (2048),
# the readout (last + sinusoidal), drop-path and the whole training recipe
# are the supervised arm's and do not move with the scale, so a difference
# between two rungs is a difference in parameters and nothing else. The
# output projection stays square (d_embedding = hidden_size), as it is in the
# reported model.
#
# "small" IS THE REPORTED MODEL: TransformerInnerConfig's own defaults, so the
# middle rung of the ladder is the arm every other figure reports and the
# scaling curve passes through the paper's own number (tests/
# test_supervised_scaling.py holds the two equal). The shapes are the
# canonical ViT-Tiny / ViT-Small / ViT-Base widths at head_dim 64.
#
# WHAT DOES NOT SCALE WITH THEM, deliberately: the learning rate. blr 2e-4 was
# measured on Small (see SupervisedModeConfig) and every rung inherits it.
# A per-scale LR sweep is the first thing to run if Base trails Small.
VIT_SCALES: dict[str, dict[str, int]] = {
    "tiny":  {"hidden_size": 192, "num_attention_heads": 3,  "intermediate_size": 768},
    "small": {"hidden_size": 384, "num_attention_heads": 6,  "intermediate_size": 1536},
    "base":  {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
}

# THE LEARNING RATE PER SCALE, measured on holdout set 2 by
# scripts/sweeps/supervised_scaling_lr.sh. A scale ABSENT from this table
# trains the recipe's own LR (SupervisedModeConfig.training_overrides.blr,
# which is Small's -- the rate was chosen on Small). A scale present with
# None has a grid pending, and supervised_scaling.sh refuses to run it:
# there is no numeric fallback, on purpose, because the only number to hand
# was measured on a different width (compare LAMBDA_BY_PAIRING). Fill in the
# winner from scripts/eval/summarize_supervised_scaling_lr.py, with the
# date and the paired margin over 2e-4, and the ladder can run.
VIT_SCALE_BLR: dict[str, float | None] = {
    # Holdout set 2, 2026-09-20 (scripts/sweeps/supervised_scaling_lr.sh, head
    # IC, month-mean over the five months, read before the grids finished on
    # the user's call). Every adjacent pair differed by a few thousandths of
    # IC, inside the month-to-month spread, so the pick is the rate that was
    # best-or-tied on most tasks, not a measured optimum.
    #   tiny: 2e-4 (the recipe's) won return and vol, 4e-4 tied it, 8e-4 fell
    #         to a third of it. 1e-4 was a hair behind on all three.
    #   base: 5e-5 and 1e-4 tied on all three tasks (4 months); 1e-4 is one
    #         2x step cooler than Small, the expected direction for 4x the
    #         width. 2e-4 had one month, 4e-4 none.
    "tiny": 2e-4,
    "base": 1e-4,
}


@dataclass
class ResNetBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.resnet.ResNetBackbone"
    d_embedding: int = 512
    variant: str = "50"
    initial_channels: int = 64
    drop_path_rate: float = 0.1
    pool: str = "mean"
    zero_init_residual: bool = True
    # ResNet's optimal LR for LeJEPA @ λ=0.01 is 1e-3 (measured; resnet-blr-low
    # sweep across {5e-5..1e-1} × {0.01, 0.1}). Lives on the backbone (not
    # mode.training_overrides) so it beats the LeJEPA mode default of 5e-4
    # via pretrain.py's precedence chain (optimizer.blr > backbone.blr >
    # mode.training_overrides.blr). Supervised sweeps always pass
    # optimizer.blr explicitly so they're unaffected.
    blr: float | None = 1e-3


@dataclass
class EfficientNetBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.efficientnet.EfficientNetBackbone"
    d_embedding: int = 512
    variant: str = "b0"
    stochastic_depth_prob: float = 0.2
    pool: str = "mean"


@dataclass
class ConvNeXtBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.convnext.ConvNeXtBackbone"
    d_embedding: int = 512
    variant: str = "tiny"
    stochastic_depth_prob: float = 0.1
    layer_scale: float = 1e-6
    pool: str = "mean"


@dataclass
class InceptionBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.inception.InceptionBackbone"
    d_embedding: int = 512
    variant: str = "v3"
    dropout: float = 0.0
    pool: str = "mean"


@dataclass
class PatchTSTInnerConfig:
    _target_: str = "market_jepa.modeling.backbones.patchtst.PatchTSTConfig"
    hidden_size: int = 128
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    intermediate_size: int = 512
    patch_size: int = 16
    patch_stride: int = 8
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.1


@dataclass
class PatchTSTBackboneConfig:
    _target_: str = "market_jepa.modeling.backbones.patchtst.PatchTSTBackbone"
    d_embedding: int = 512
    pool: str = "mean"
    max_seq_len: int = 2048
    config: PatchTSTInnerConfig = field(default_factory=PatchTSTInnerConfig)


# ── Mode Training Overrides ─────────────────────────────────────────────────


@dataclass
class ModeTrainingOverrides:
    """Training-loop overrides that a mode can set.

    Fields left as None fall back to the corresponding top-level config
    (optimizer, training, wandb).
    """

    num_epochs: int | None = None
    blr: float | None = None
    default_batch_size: int | None = None
    effective_batch_size: int | None = None
    per_device_train_batch_size: int | None = None
    weight_decay_start: float | None = None
    weight_decay_end: float | None = None


@dataclass
class ModeDatasetOverrides:
    """Dataset-config overrides a mode can set.

    Each non-None field replaces the corresponding key in every entry of
    cfg.dataset.augmentations and cfg.dataset.eval_augmentations. Applied
    in pretrain.py after the hardcoded single-view branch (so it can
    express CPC's single-view requirement without an external yaml).
    """

    n_global_views: int | None = None
    n_local_views: int | None = None
    # Positive-pair construction. Set to "random_resized_crop" by every
    # SINGLE-VIEW mode, because the dataset default is now the cross-stock
    # pairing (see DatasetConfig.augmentations) and a mode that never forms a
    # positive pair has no use for a partner ticker — it would just draw K
    # stocks and discard K-1 of them.
    #
    # THE ONE EXCEPTION IS THE SUPERVISED SPECIALIST, which sets
    # name="cross_stock" with n_stocks=K for a different reason than the
    # joint-embedding modes: not to form a positive pair but to hand the
    # ranking loss a whole CELL -- K stocks at one wall-clock anchor, every
    # one of them labelled -- so it can rank inside a cross-section the way
    # the reported IC does. See SupervisedModeConfig.dataset_overrides.
    name: str | None = None
    # cross_stock only: stocks per cell. None leaves the augmentation's own
    # n_stocks alone.
    n_stocks: int | None = None


# ── Mode Config Groups ──────────────────────────────────────────────────────


# LAMBDA IS PER-PAIRING. The 2026-08-27 sweep (5 arms x 7-10 lambdas x 5
# holdout-2 months, probe dIC vs randinit-fwd3) found lambda's optimum is a
# property of the PAIRING, not of the objective:
#
#   arm                  lambda   return      vol       spread
#   k2                   0.2      +0.0089   +0.0464*  +0.0182
#   k2ind                0.1      +0.0098   +0.0481   +0.0218*
#   random_resized_crop  0.05     +0.0015   +0.0420   +0.0283
#   time_warp            0.001    +0.0134   +0.0603   +0.0484
#   gaussian_noise       0.3      +0.0086   +0.0454   +0.0154
#
#   (* that task's own argmax sits elsewhere in the row; the lambda column is
#    the one arm-wide value taken to the 32 reported months.)
#
# Everything cross-stock or noise-based wants lambda >= 0.1; time_warp wants
# the smallest value on the grid. No single scalar is right for every
# augmentation, so LeJEPAModeConfig.lamb defaults to None and pretrain.py
# resolves it HERE -- explicit pin > this table > hard error. Until 2026-09-06
# this table was a comment and the field defaulted to 0.01, which is the
# optimum for no arm at all, so any run that did not pin trained a value
# nothing had been tuned at.
#
# TWO CAVEATS THAT ARE NOT FOOTNOTES:
#   - time_warp 0.001 IS A GRID EDGE, not a located optimum. It is the
#     smallest lambda run and the curve is still descending into it
#     (+0.0019/+0.0020/+0.0024 per halving, t = +1.27/+1.33/+0.55).
#   - warp_strength 0.75 and noise_sigma 0.75 were NEVER SWEPT. They are
#     stale Jan-2023 HPO winners, chosen under return_900_k5 AUC on the
#     retired mid-to-mid target, and they differ from the AugmentationConfig
#     defaults (0.25 and 0.1). Lambda and strength both set invariance
#     pressure, so time_warp's pull toward tiny lambda may be reading
#     warp_strength=0.75 rather than the pairing.
LAMBDA_BY_PAIRING: dict[str, float] = {
    "k2": 0.2,
    "k2ind": 0.1,
    "random_resized_crop": 0.05,
    "time_warp": 0.001,
    "gaussian_noise": 0.3,
}


def lejepa_pairing_key(augmentations) -> str | None:
    """Which row of ``LAMBDA_BY_PAIRING`` this augmentation list is, or None.

    Takes the RESOLVED list pretrain.py builds -- after the single-view modes
    have replaced it and ``mode.dataset_overrides`` has rewritten it -- because
    that is the pairing the model actually trains on.

    ``cross_stock`` splits on two knobs the run name does not always carry:
    K=2 unrestricted is "k2", K=2 restricted to the focal's FF49 industry is
    "k2ind", and every other K is deliberately unmapped. Returning None is not
    a failure, it means "the sweep has to say" -- lambda was never tuned for
    that pairing, so there is no value to fall back to.
    """
    if not augmentations:
        return None
    first = augmentations[0]
    get = first.get if hasattr(first, "get") else lambda k, d=None: getattr(first, k, d)
    name = str(get("name", "") or "")
    if name != "cross_stock":
        return name if name in LAMBDA_BY_PAIRING else None
    if int(get("n_stocks", 2) or 2) != 2:
        return None
    return "k2ind" if get("industry_table", None) else "k2"


@dataclass
class LeJEPAModeConfig:
    _target_: str = "market_jepa.modeling.modes.lejepa.LeJEPA"
    proj_dim: int | None = 64
    proj_hidden: list[int] | None = None
    n_projections: int = 256
    # NOT PINNED. See LAMBDA_BY_PAIRING above for the table and the caveats.
    #
    # None means "resolve me": pretrain.py takes an explicit pin first, falls
    # back to this pairing's swept optimum, and RAISES if the pairing has no
    # entry. There is deliberately no numeric default -- the 0.01 that stood
    # here until 2026-09-06 was the k2ind-era value and the optimum for no arm,
    # so an unpinned run silently trained something nothing had been tuned at.
    lamb: float | None = None
    # THE SHARED RECIPE, mirroring SupervisedModeConfig so the two arms differ
    # in the objective and nothing else -- which is the only way a LeJEPA-vs-
    # supervised number measures the objective rather than the budget.
    #
    # blr was 5e-4 (the historical LeJEPA+ViT value) and epochs/batch were unset
    # until 2026-08-27, so a LeJEPA run at defaults trained 100 epochs at batch
    # 128 on an 8x learning rate while every scored run pinned 200/256/6e-5.
    #
    # CONSEQUENCE, deliberately accepted: twelve older lejepa sweeps do not pin
    # blr (lejepa_lamb, k2ind_lamb, k2ind_scaling, cross_stock{,_industry},
    # industry_augs, samestock_augs_14mo, lejepa_views, tw_noise_hpo, rf_iwm_mid,
    # determinism_control, grid_cache_ab). Re-running any of them now trains the
    # current recipe, NOT the one their published results were produced under.
    #
    # default_batch_size stays None, so scale_lr returns blr unchanged rather
    # than rescaling 6e-5 by 256/128 -- the sweeps quote blr at the batch they
    # run, not at a reference batch.
    #
    # All three are safe to set since the 2026-08-27 precedence fix: every one
    # of them resolves EXPLICIT PIN > THIS > fallback, so a sweep that pins a
    # different budget still wins and a sweep passing `num_epochs=null` beside
    # max_train_steps keeps its step budget. Before that fix num_epochs and
    # batch resolved the other way and would have shadowed both.
    #
    # ResNet sets blr=1e-3 on its backbone config, which still beats this via
    # the precedence chain in pretrain.py.
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            blr=6e-5, per_device_train_batch_size=256, num_epochs=12))
    # mean_sin: the winning cell of sweeps/readout_grid_h2.sh. The invariance
    # loss is computed on the MEAN over patch tokens, so it shapes a whole-day
    # representation instead of a CLS anchored at one end, and position comes
    # from a FIXED sin/cos table because a 257-slot learned one cannot be
    # estimated from a month (worth +0.0115 / +0.0511 / +0.0459 to mean
    # pooling). Under pool="mean" no CLS token is built at all, so cls_pos is
    # inert here and left unset.
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))


@dataclass
class IJEPAModeConfig:
    """I-JEPA mode configuration.

    IJEPA owns its backbone config -- the top-level ``cfg.backbone`` is ignored
    when this mode is selected.  The training loop (pretrain.py) detects the
    nested ``backbone`` field here and instantiates it directly, logging a
    warning if a top-level backbone was also supplied.

    To configure the backbone for IJEPA, set ``mode.backbone.*`` (not the
    top-level ``backbone``).

    IJEPA requires ``pool="mean"`` (the default below).  Using ``pool="cls"``
    will raise at init because ``forward_patches`` operates on raw patch
    positions without a CLS token.
    """

    _target_: str = "market_jepa.modeling.modes.ijepa.IJEPA"
    backbone: TransformerBackboneConfig = field(default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))
    pred_depth: int = 6
    pred_emb_dim: int = 192
    pred_num_heads: int | None = None  # defaults to backbone's num_attention_heads
    ema_start: float = 0.996
    ema_end: float = 1.0
    n_targets: int = 4
    target_scale: list[float] = field(default_factory=lambda: [0.15, 0.2])
    context_crop_max: float = 0.15
    loss_fn: str = "smooth_l1"
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            num_epochs=12,
            effective_batch_size=2048,
            per_device_train_batch_size=1024,
            default_batch_size=2048,
            blr=1e-4,
            weight_decay_start=0.04,
            weight_decay_end=0.4,
        )
    )

    # Single-view mode: pinned to the plain crop sampler so the cross-stock
    # pairing that DatasetConfig.augmentations now defaults to never reaches
    # it.
    #
    # n_global_views=1 / n_local_views=0 SINCE 2026-08-27. Until then only
    # `name` was set and these two were left None, so the mode inherited the
    # dataset defaults (2 globals + 6 locals) and the loader built EIGHT views
    # per sample -- while training_step reads bucket["views"][0] and discards
    # the other seven. The model never saw them, which is why the old comment
    # here could truthfully say "this mode's behaviour is unchanged"; the cost
    # was entirely in the dataloader, and it was 3.41x measured (3.66 vs 1.07
    # ms/sample warm-cache; 7.5s vs 2.2s of CPU per 2048-sample step). Crop and
    # aggregate plus normalize are ~74% of per-sample cost and both are PER
    # VIEW, so unused views are pure waste on a pipeline whose GPUs sat at
    # ~20% duty waiting on exactly this work.
    #
    # The other five single-view modes (CPC, TS2Vec, CoST, TF-C, TimeMAE) have
    # always pinned 1/0 here. These two were the outliers. FinanceBaseline and
    # PretrainedTSFM still carry the old block deliberately -- neither trains,
    # so their loader cost is a one-off, not a per-step tax.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            name="random_resized_crop", n_global_views=1, n_local_views=0)
    )

@dataclass
class MAEModeConfig:
    """Masked Autoencoder mode configuration.

    MAE owns its backbone config (like IJEPA); the top-level ``cfg.backbone``
    is ignored when this mode is selected. Requires a TransformerBackbone with
    ``pool != "cls"``.

    The ``mask_ratio`` field is the primary sweep knob: run multiple MAE
    pretrainings with different mask ratios via ``mode.mask_ratio=0.5`` etc.
    """

    _target_: str = "market_jepa.modeling.modes.mae.MAE"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            # WAS pool="last". Moved to mean for the universal SSL
            # convention; MAE only requires pool != "cls".
            pool="mean",
            config=TransformerInnerConfig(
                rescale_residual_init=True, pos_embed="sinusoidal"),
        )
    )
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            num_epochs=12,
            effective_batch_size=2048,
            per_device_train_batch_size=1024,
            default_batch_size=2048,
            blr=5e-4,
            weight_decay_start=0.05,
            weight_decay_end=0.05,
        )
    )
    # Single-view mode: pinned to the plain crop sampler so the cross-stock
    # pairing that DatasetConfig.augmentations now defaults to never reaches
    # it.
    #
    # n_global_views=1 / n_local_views=0 SINCE 2026-08-27. Until then only
    # `name` was set and these two were left None, so the mode inherited the
    # dataset defaults (2 globals + 6 locals) and the loader built EIGHT views
    # per sample -- while training_step reads bucket["views"][0] and discards
    # the other seven. The model never saw them, which is why the old comment
    # here could truthfully say "this mode's behaviour is unchanged"; the cost
    # was entirely in the dataloader, and it was 3.41x measured (3.66 vs 1.07
    # ms/sample warm-cache; 7.5s vs 2.2s of CPU per 2048-sample step). Crop and
    # aggregate plus normalize are ~74% of per-sample cost and both are PER
    # VIEW, so unused views are pure waste on a pipeline whose GPUs sat at
    # ~20% duty waiting on exactly this work.
    #
    # The other five single-view modes (CPC, TS2Vec, CoST, TF-C, TimeMAE) have
    # always pinned 1/0 here. These two were the outliers. FinanceBaseline and
    # PretrainedTSFM still carry the old block deliberately -- neither trains,
    # so their loader cost is a one-off, not a per-step tax.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            name="random_resized_crop", n_global_views=1, n_local_views=0)
    )

@dataclass
class CPCModeConfig:
    """Contrastive Predictive Coding mode configuration.

    CPC owns its backbone config (like IJEPA / MAE); the top-level
    ``cfg.backbone`` is ignored when this mode is selected. Requires a
    TransformerBackbone with ``pool != "cls"``.

    Primary sweep knobs: ``n_predictions`` (K), ``gru_hidden_size``, and
    ``temperature``.
    """

    _target_: str = "market_jepa.modeling.modes.cpc.CPC"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal"))
    )
    gru_hidden_size: int = 256
    gru_num_layers: int = 1
    n_predictions: int = 12
    min_context_frac: float = 0.25
    temperature: float = 0.1
    # The three market adaptations of Oord et al., each gated so the IC HPO
    # can measure whether it earns its keep (sweeps/ssl_ic/cpc_r1.sh). The
    # paper-faithful arm is negative_scope=all, cosine_logits=false,
    # target_encoder=shared_causal; see the CPC class docstring for what each
    # deviation guards against.
    negative_scope: str = "xticker_xday"
    cosine_logits: bool = True
    target_encoder: str = "stopgrad_bidir"
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            num_epochs=12,
            effective_batch_size=2048,
            per_device_train_batch_size=1024,
            default_batch_size=2048,
            blr=1.5e-4,
            weight_decay_start=0.05,
            weight_decay_end=0.05,
        )
    )
    # CPC encoder consumes one global crop per sample — multi-view dataset
    # defaults (n_global_views=2, n_local_views=6) don't apply. The
    # global_scale_range comes from DatasetConfig: [0.5, 1.0] for both train
    # and eval. CPC relies on those values matching what the old
    # conf/experiment/cpc_init.yaml used to pin explicitly — if you change
    # the dataset defaults, port the prior CPC scale ranges here.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="random_resized_crop")
    )


@dataclass
class DINOModeConfig:
    """DINO mode configuration.

    Two time-warped views of one window (see dataset_overrides) with an
    EMA teacher network. OWNS its backbone, like every other mode; the teacher is a
    deep copy of it.

    Primary sweep knobs: ``optimizer.blr`` and ``mode.ema_start`` (the
    initial teacher EMA decay; ``ema_end`` is held fixed at 1.0 so the
    teacher's parameters approach a frozen point near training end).
    """

    _target_: str = "market_jepa.modeling.modes.dino.DINO"
    proj_dim: int = 4096
    proj_hidden: int = 2048
    proj_bottleneck: int = 256
    proj_n_layers: int = 3
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    ema_start: float = 0.996
    ema_end: float = 1.0
    gradient_checkpointing: bool = False
    # THE POSITIVE PAIR IS TWO WARPS OF ONE WINDOW (2026-09-14). Declaring
    # nothing here was DELIBERATE before this date, not an oversight: every
    # uses_multi_view mode inherited DatasetConfig.augmentations' default
    # INSTANCE -- cross_stock(n_stocks=2, industry_table=STANDARD) -- so these
    # two trained on an industry-matched partner ticker, and
    # tests/test_standard_augmentation.py pinned that rule.
    #
    # THE RULE CHANGED FOR COMPARABILITY. lejepa-6mo-warp draws its pair as two
    # warps of one window; with DINO and BYOL pinned the same way, the three
    # joint-embedding objectives differ only in the objective. LeJEPA itself
    # still inherits k2ind -- its arms name their pairing on the command line.
    #
    # THIS DOES NOT RESTORE MULTI-CROP, despite what the class docstrings used
    # to claim. cross_stock reads cross_stock_local_views (0) and the
    # corruption family emits no n_local_views key at all, so the local branch
    # of both losses is inert under either pairing; only random_resized_crop
    # would feed it.
    #
    # NOT COMPATIBLE WITH RISK FACTORS: the warped bucket grid breaks the
    # wall-clock alignment _merge_risk_factors assumes, and
    # StreamingMarketDataset.__init__ raises at construction when
    # risk_factor_tickers is non-empty. DatasetConfig.risk_factor_tickers is
    # empty by default, so this only bites a run that opts in.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(name="time_warp")
    )
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=5e-4)
    )
    # SSL READS AT THE MEAN, UNIVERSALLY (2026-09-10). Every self-supervised
    # mode trains a whole-view representation over a FIXED sin/cos table, the
    # mean_sin cell of sweeps/readout_grid_h2.sh, matching LeJEPA. The
    # supervised arm is the other half of the pair: pool="last" under ROPE,
    # because a prediction head reads one vantage point. DINO and BYOL DECLARE
    # this like every other mode now -- they used to read the top-level block
    # as their student and fell through to the cls/learned fallback, which is
    # what every DINO/BYOL checkpoint on disk before 2026-09-10 records.
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))


@dataclass
class BYOLModeConfig:
    """BYOL mode configuration.

    Two time-warped views of one window (see dataset_overrides), with an
    EMA teacher network. OWNS its backbone, like every other mode; the
    teacher is a deep copy of (backbone, projector). Only the student
    carries a predictor head — that's BYOL's asymmetry.
    """

    _target_: str = "market_jepa.modeling.modes.byol.BYOL"
    proj_dim: int = 256
    proj_hidden: int = 4096
    pred_hidden: int = 4096
    ema_start: float = 0.996
    ema_end: float = 1.0
    gradient_checkpointing: bool = False
    # THE POSITIVE PAIR IS TWO WARPS OF ONE WINDOW (2026-09-14). Declaring
    # nothing here was DELIBERATE before this date, not an oversight: every
    # uses_multi_view mode inherited DatasetConfig.augmentations' default
    # INSTANCE -- cross_stock(n_stocks=2, industry_table=STANDARD) -- so these
    # two trained on an industry-matched partner ticker, and
    # tests/test_standard_augmentation.py pinned that rule.
    #
    # THE RULE CHANGED FOR COMPARABILITY. lejepa-6mo-warp draws its pair as two
    # warps of one window; with DINO and BYOL pinned the same way, the three
    # joint-embedding objectives differ only in the objective. LeJEPA itself
    # still inherits k2ind -- its arms name their pairing on the command line.
    #
    # THIS DOES NOT RESTORE MULTI-CROP, despite what the class docstrings used
    # to claim. cross_stock reads cross_stock_local_views (0) and the
    # corruption family emits no n_local_views key at all, so the local branch
    # of both losses is inert under either pairing; only random_resized_crop
    # would feed it.
    #
    # NOT COMPATIBLE WITH RISK FACTORS: the warped bucket grid breaks the
    # wall-clock alignment _merge_risk_factors assumes, and
    # StreamingMarketDataset.__init__ raises at construction when
    # risk_factor_tickers is non-empty. DatasetConfig.risk_factor_tickers is
    # empty by default, so this only bites a run that opts in.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(name="time_warp")
    )
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=5e-4)
    )
    # SSL READS AT THE MEAN, UNIVERSALLY (2026-09-10). Every self-supervised
    # mode trains a whole-view representation over a FIXED sin/cos table, the
    # mean_sin cell of sweeps/readout_grid_h2.sh, matching LeJEPA. The
    # supervised arm is the other half of the pair: pool="last" under ROPE,
    # because a prediction head reads one vantage point. DINO and BYOL DECLARE
    # this like every other mode now -- they used to read the top-level block
    # as their student and fell through to the cls/learned fallback, which is
    # what every DINO/BYOL checkpoint on disk before 2026-09-10 records.
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))


@dataclass
class TS2VecModeConfig:
    """TS2Vec mode configuration (Yue et al., AAAI 2022).

    Owns its backbone config (like CPC / MAE). ``pool="max"`` mirrors the
    paper's full-series max-pooling readout; ``encode()`` evaluates the
    SWA-averaged encoder (the original's ``AveragedModel`` protocol).
    Single-view: the two overlapping crops are sampled inside the mode.

    Primary sweep knob: ``optimizer.blr`` (original lr: 1e-3).
    """

    _target_: str = "market_jepa.modeling.modes.ts2vec.TS2Vec"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            # WAS pool="max", the TS2Vec paper's full-series readout.
            # Moved to mean for the universal SSL convention; inert
            # either way, since TS2Vec is an ENCODE_MODE_CLASS whose
            # downstream embedding comes from encode(), not from a
            # pooled backbone forward, and its loss is per-timestep.
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal"))
    )
    alpha: float = 0.5
    temporal_unit: int = 0
    mask_p: float = 0.5
    swa: bool = True
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=1e-3)
    )
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="random_resized_crop")
    )


@dataclass
class CoSTModeConfig:
    """CoST mode configuration (Woo et al., ICLR 2022).

    Owns its backbone config. Downstream embeddings are the concatenated
    trend + season components at the last patch (``encode()`` override), so
    the backbone ``pool`` only labels the checkpoint. Single-view: the two
    augmented views (scale/shift/jitter) are created inside the mode.

    Primary sweep knob: ``optimizer.blr`` (original lr: 1e-3 SGD).
    """

    _target_: str = "market_jepa.modeling.modes.cost.CoST"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            # WAS pool="last". Moved to mean for the universal SSL
            # convention.
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal"))
    )
    kernels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    alpha: float = 0.0005
    queue_size: int = 256
    ema_momentum: float = 0.999
    temperature: float = 0.07
    sigma: float = 0.5
    aug_p: float = 0.5
    fourier_length: int = 256
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=1e-3)
    )
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="random_resized_crop")
    )


@dataclass
class TFCModeConfig:
    """TF-C mode configuration (Zhang et al., NeurIPS 2022).

    Owns its backbone config; the frequency encoder is an independent copy
    built inside the mode. Probe embeddings are ``[z_t, z_f]`` (256 dims).
    Single-view: jitter / frequency perturbations are created inside the mode.

    Primary sweep knob: ``optimizer.blr`` (original lr: 3e-4 Adam).
    """

    _target_: str = "market_jepa.modeling.modes.tfc.TFC"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal"))
    )
    proj_dim: int = 128
    proj_hidden: int = 256
    temperature: float = 0.2
    lam: float = 0.2
    jitter_sigma: float = 0.1
    freq_perturb_ratio: float = 0.1
    use_poly_loss: bool = True
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=3e-4)
    )
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="random_resized_crop")
    )


@dataclass
class TimeMAEModeConfig:
    """TimeMAE mode configuration (Cheng et al., 2023).

    Owns its backbone config; ``pool="mean"`` mirrors the official
    mean-over-patches readout. Single-view masked modeling.

    Primary sweep knob: ``optimizer.blr`` (original lr: 1e-3 AdamW).
    """

    _target_: str = "market_jepa.modeling.modes.timemae.TimeMAE"
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="mean",
            config=TransformerInnerConfig(pos_embed="sinusoidal"))
    )
    vocab_size: int = 192
    mask_ratio: float = 0.6
    ema_momentum: float = 0.99
    reg_layers: int = 4
    align_weight: float = 5.0
    reconstruct_weight: float = 1.0
    gradient_checkpointing: bool = False
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(blr=1e-3)
    )
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="random_resized_crop")
    )


@dataclass
class FinanceBaselineModeConfig:
    """Classical finance baselines: AR(1) returns, GARCH(1,1) vol, AR(1) spread.

    Not trained: it emits each model's forecast of the probe targets via
    ``encode()``, and the probe-eval pipeline scores them at
    ``probe/ridge_ic_*`` — the same metric used for learned representations.
    ``num_epochs=0`` skips the training loop; the headline numbers come from the
    initial/final probe eval. The top-level ``cfg.backbone`` is instantiated (to
    satisfy the harness) but unused.

    ``params_path`` should point at the JSON written by
    ``scripts/fit_finance_baselines.py``, which fits ``(alpha, beta, phi_return,
    phi_spread)`` pooled across assets and refit per calendar month. Leaving it
    unset falls back to uncalibrated textbook constants — fine for a smoke test,
    not for a reported number.

    ``horizons`` must match ``targets.horizons`` for the forecasts to line up
    with the columns the probe scores.
    """

    _target_: str = "market_jepa.modeling.modes.finance_baselines.FinanceBaseline"
    features: list[str] = field(
        default_factory=lambda: ["return", "volatility", "spread"]
    )
    horizons: list[int] = field(
        default_factory=lambda: [300, 600, 900, 1800, 3600, 7200]
    )
    params_path: str | None = None
    # Cap parameter resolution at this month ("YYYY-MM", inclusive). Set to the
    # TRAIN month whenever params_path covers the eval month; otherwise eval
    # samples resolve to cells fitted on their own month's future returns
    # (in-sample forecasts the learned-encoder side never gets).
    params_as_of: str | None = None
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(num_epochs=0)
    )

    # Single-view mode: pinned to the plain crop sampler so the cross-stock
    # pairing that DatasetConfig.augmentations now defaults to never reaches
    # it. Only `name` is set — n_global_views / n_local_views stay exactly
    # where they were, so this mode's behaviour is unchanged.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(name="random_resized_crop")
    )

@dataclass
class PretrainedTSFMModeConfig:
    """Frozen pretrained time-series foundation models (TimesFM 2.5, TimesFM
    3.0, Sundial, Chronos-2, Kronos) as probe-scored feature extractors.

    Not trained: ``encode()`` pools hidden states of the frozen TSFM and the
    probe-eval pipeline scores them at ``probe/ridge_ic_*`` — the same
    metric used for learned representations. ``num_epochs=0`` skips the
    training loop; the headline numbers come from the initial/final probe
    eval. The top-level ``cfg.backbone`` is instantiated (to satisfy the
    harness) but unused.

    Two of the five are multivariate — chronos2 through group attention,
    timesfm3 through variate attention — so their nine channels are one
    group rather than nine independent series; the readout concatenates the
    per-channel states either way.

    The sweep knob is ``layer``: which hidden state feeds the probe (0 =
    patch embedding, k = output of block k; timesfm and timesfm3 have 20
    blocks, sundial, chronos2 and kronos have 12; -1 = last).
    ``use_timestamps`` (chronos2 only)
    appends time-of-day sin/cos covariates to the group — the honest
    comparison leaves it False.
    """

    _target_: str = "market_jepa.modeling.modes.pretrained_tsfm.PretrainedTSFM"
    # "timesfm" | "timesfm3" | "sundial" | "chronos2" | "kronos"
    model: str = MISSING
    model_id: str | None = None
    layer: int = -1
    channels: list[int] | None = None
    channel_pool: str = "concat"
    time_pool: str = "mean"
    use_timestamps: bool = False
    max_context: int | None = None
    tsfm_batch_size: int = 64
    # kronos only: fine-token re-aggregation per bar (default 4 -> the full
    # 2048-token view fills the 512-bar context) and the BSQ tokenizer repo.
    bar_agg: int | None = None
    tokenizer_id: str | None = None
    # blr is inert (num_epochs=0 means the optimizer never steps) but the
    # harness formats and scales it during setup, so it must be a number.
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(num_epochs=0, blr=1e-4)
    )

    # Single-view mode: pinned to the plain crop sampler so the cross-stock
    # pairing that DatasetConfig.augmentations now defaults to never reaches
    # it. Only `name` is set — n_global_views / n_local_views stay exactly
    # where they were, so this mode's behaviour is unchanged.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(name="random_resized_crop")
    )

@dataclass
class SupervisedModeConfig:
    _target_: str = "market_jepa.modeling.modes.supervised.SupervisedModel"
    task: str = MISSING
    # THE supervised ablation axis (see modes/supervised.py:LOSS_FNS).
    # Every loss is SCALAR, on the cross-sectional target:
    #   "mse" | "smooth_l1"  pointwise; both are minimized at the conditional
    #                        mean, which is ~0 for a standardized target, so
    #                        they invite collapse onto a constant
    #   "corr" | "pairwise"  ranking; cannot be minimized by a constant
    #
    # PAIRWISE (LTR) IS THE DEFAULT AND THE ONE IN USE. The binned family --
    # cross_entropy, expected_bin_mse, expected_bin_mae -- was retired on
    # 2026-09-07 along with everything it needed: n_bins, calibration_batches,
    # the soft-label temperature and its anneal, and the auxiliary
    # expected-bin penalty. Four knobs and a calibration pass over the
    # training dataloader, none of it reachable once the objective is LTR.
    loss_fn: str = "pairwise"
    smooth_l1_beta: float = 1.0
    # SSL finetuning: initialize the backbone from a pretrained checkpoint
    # directory (LeJEPA save_pretrained format: config.json + model.pt).
    # Finetuning is always END-TO-END. `freeze_backbone` -- the head-only arm --
    # was retired 2026-09-16 together with the multihead's finetune path: what
    # a frozen encoder gives you is a probe, and the probe is measured properly
    # as a ridge on the full fit pool (plots/core/probe_fit_breadth.py).
    # Training a head to rediscover it by SGD answered the same question with
    # more compute and a worse estimator.
    init_backbone_from: str | None = None
    # THE HEAD STARTS AT THE PROBE, not at random. A directory written by
    # scripts/eval/fit_ridge_head_init.py, holding one folded ridge per task;
    # setting it swaps the plain MLP head for a SkipRegressionHead whose skip
    # IS that ridge and whose MLP branch starts at zero, so step 0 reproduces
    # the probe exactly. Requires init_backbone_from -- a probe fit on a
    # pretrained encoder's features is meaningless on a random one.
    #
    # WHY IT EXISTS. A head that starts at random spends the early part of any
    # label budget learning to read an embedding it is already handed. That is
    # the frozen-probe curve, which is measured elsewhere and more cheaply, so
    # paying for it again out of the finetune's budget only obscures what the
    # finetune is for: adapting the ENCODER.
    init_head_from: str | None = None
    # "unit" (default) rescales the ridge to unit output spread on its fit
    # pool; "raw" keeps its IC-sized spread. See heads.load_ridge_init -- under
    # a rank loss only the direction carries information, and the raw spread
    # (~0.02 against a target std of 0.288) starts training at softplus'
    # collapse point.
    head_init_scale: str = "unit"
    # THE RECIPE: blr 1e-5 at batch 256, 100 epochs.
    #
    # 1e-5 SINCE 2026-09-07, MEASURED. sweeps/supervised_lr_h2.sh ran nine
    # learning rates over five holdout-2 months against the current recipe
    # (pairwise, last_rope, uniform target, 100 epochs) and found an interior
    # optimum, flat between 7.5e-6 and 1.5e-5 and falling away on both sides;
    # 1e-5 is the round number inside it. Paired month-matched, 1.5e-5 beat
    # the previous 6e-5 by +0.0046 +- 0.0009 on return (t +5.2, 5/5 months)
    # and +0.0111 on spread change, with volatility flat across the whole
    # grid -- so the learning rate is a return and spread knob only.
    #
    # The 6e-5 it replaces was never measured against this recipe: it came
    # from a recency-window x LR grid whose other axis no longer exists.
    #
    # THIS IS THE LR ITSELF, not a bs128 quote. Sweep files differ on that --
    # supervised_loss_ablation.sh takes LR-at-128 and scales by BS/128, so the
    # same number means something else there. Set HERE and not on TrainingConfig
    # because nine modes -- LeJEPA, DINO, BYOL and the SSL baselines -- carry
    # default_batch_size=None, meaning NO LR auto-scaling: moving the global
    # batch 128 -> 256 would halve their LR-per-sample silently, which is a
    # different experiment rather than a default change.
    #
    # 100 EPOCHS SINCE 2026-09-07, down from 200. LeJEPAModeConfig moves with
    # it: the two mode recipes are deliberately identical so that a
    # LeJEPA-vs-supervised number measures the OBJECTIVE and not the budget,
    # and changing one alone would quietly make that comparison a budget
    # comparison instead.
    #
    # 1e-5 RE-CONFIRMED under the real loss (supervised_lr_h2_r3, round 3:
    # 5 months x {5e-6, 1e-5, 2e-5} x 3 tasks x 2 position encodings, 89/90
    # arms). Rounds 1-2 chose it against the flat surrogate that predates
    # 00fbb35, so it was inherited rather than measured; it survives the
    # re-measurement unchanged. Month-matched head IC:
    #
    #   volatility   1e-5 - 5e-6  +0.0041 +- 0.0011 (t 3.7, 5/5)  rope
    #                             +0.0013 +- 0.0001 (t 12.8, 5/5) sinusoidal
    #                2e-5 - 1e-5  -0.0004 and -0.0006 (both negative)
    #   return       every pair t <= 0.9 -- indifferent across the grid
    #   spread       mildly prefers HIGHER (2e-5 best on rope, +0.0076 over
    #                1e-5) but t 0.9 and 3/5, and 2e-5 costs volatility
    #
    # So ONE LR serves all three: volatility is the only task with a
    # significant preference, it is interior and unanimous on both arms, and
    # nothing else contradicts it. The multihead carries the same value --
    # the average of three identical choices.
    #
    # RETURN CANNOT DECIDE THIS, which is worth stating because it is the
    # task the project cares about most. Its sigma_month / sigma_seed is 4x,
    # so a +0.0007 LR effect is invisible at five months. That is not
    # evidence the LR does not matter for return; it is evidence this panel
    # cannot see it.
    #
    # ALL OF THE ABOVE WAS MEASURED ON A MODEL THAT DID NOT TRAIN (audit of
    # 2026-09-11, see the memory note supervised-campaign-backbone-never-
    # trained). Under the recipe those comments describe -- one random crop
    # per row, batch 256, blr 1e-5, "100 epochs" -- a month is 4,608
    # ticker-days, so 100 epochs is 1,800 optimizer steps; the within-cell
    # loss found ~74 pairs per 256-row batch because rows only share a cell
    # by chance; the loss sat at log 2 for the whole run; and the saved
    # backbones ended 0.15% (relative L2) from their init, with the three
    # task specialists of one month 0.2% from EACH OTHER. The LR grids were
    # monotone decreasing in probe IC and fell below the random-init floor at
    # 2.4e-4, which is what a starved objective looks like: a larger LR only
    # perturbs the random features the ridge was reading. "1e-5 is optimal"
    # meant "1e-5 trains least".
    #
    # THE RECIPE NOW. Every row arrives inside a CELL (dataset_overrides
    # below: cross_stock, n_stocks=16), so the loss ranks K labelled stocks at
    # one anchor and every row is in K-1 pairs -- ~1,900 pairs per step at
    # 16 cells x 16 stocks instead of ~74, with no row left out.
    # per_device_train_batch_size is therefore CELLS per step: 16 cells x 16
    # stocks = 256 views, the same forward cost as before. An epoch is still
    # one pass over the month's ticker-days as focals, so steps per epoch
    # rise 16x with the batch of cells.
    #
    # THE SETTLED RECIPE (2026-09-13, ten holdout months, every number quoted
    # against the rebuilt random-init floor; plots/metrics/holdout_sweeps.py):
    #
    #   256 cells per optimizer step (effective_batch_size, accumulated over
    #   16-cell micro-batches), blr 2e-4, SIX MONTHS of training data at 12
    #   passes, and the day-major store (dataset.backend=days). Against the
    #   floor: return +0.013 +- 0.004 head / +0.007 probe, vol +0.024 +-
    #   0.004 (6/6 months), spread +0.075 on the three months it had reached
    #   when the recipe was locked.
    #
    #   PASSES AND LR TOGETHER, not separately. Over 4/7/12/20 passes at
    #   5e-5..5e-4 on the six floored months, return rises 4 -> 12 passes at
    #   every LR (+0.004 -> +0.013 at 2e-4), and 20 passes then adds nothing
    #   at any LR (+0.014 at 2e-4, three months up and three down, paired
    #   +0.001 +- 0.002) while doubling weight drift to ~0.12. LR orders
    #   every row 5e-5 < 1e-4 < 2e-4, and 5e-4 is past the peak: +0.009 at 20
    #   passes with the PROBE back AT the floor -- the head keeps its score
    #   on features it is erasing. Vol agrees (1e-4 +0.023, 2e-4 +0.024,
    #   5e-4 +0.021) with a flatter curve. Twelve at 2e-4 equals the best
    #   20-pass arm and the old 12-month/7-pass recipe at 60% of the cost.
    #
    #   SPREAD INHERITS 2e-4 by decision, not measurement: its 2e-4 arms were
    #   still queued when the recipe was locked and the sweep was cancelled.
    #   Its 1e-4 arms beat the previous recipe on 3/3 months (+0.003). Note
    #   the untrained spread readout falls monotonically as LR rises on every
    #   return and vol arm (~0.10 at 1e-4, 0.081 at 2e-4, 0.069 at 5e-4 on
    #   one month), so spread features are the first thing a hot LR
    #   overwrites -- re-measure here first if spread regresses.
    #
    #   What did NOT move return, each tried on the same months: 64 cells
    #   (-0.002), 512 cells at equal steps (-0.003), 100 passes over one
    #   month (-0.006, the model memorizes the month by ~40), 1,024 cells per
    #   step at equal views (-0.006 vs 256: a quarter of the steps), and
    #   stocks per cell K=64/128/200 (noise).
    #
    #   The loader is not in these numbers. Day-major and MDS at the matched
    #   arm (6 months x 7 passes x 1e-4) differ by -0.002 +- 0.001 over nine
    #   months with no consistent sign: the views are equal element by
    #   element (tests/test_cell_dataset.py) and only the cell SAMPLING
    #   differs. It was adopted for throughput -- 95% GPU util at ~3,150
    #   views/s against ~60% and ~1,900.
    #
    # num_epochs=12 is passes over the SPAN (DatasetConfig.train_span_months,
    # six months), so it is 72 month-passes. A single-month caller wants ~40
    # (the pilots' budget) and every single-month sweep pins its own; the
    # campaign launcher trains spans (scripts/pythia/specific/run_span_bundle.sh).
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            blr=2e-4, per_device_train_batch_size=16, effective_batch_size=256,
            num_epochs=12)
    )
    # last + SINUSOIDAL: no CLS at all -- the final patch IS the readout --
    # with the same absolute position encoding every other mode uses.
    #
    # ROPE'S JUSTIFICATION WAS MEASURED ON A LOSS WE DO NOT TRAIN. It read
    # "+0.0026, t 1.3" for the supervised head, and every run behind that
    # number predates 00fbb35, when collate_bucketed dropped xs_cell and
    # _within_cell_loss had never executed -- so it ranked the flat cross-cell
    # surrogate, not the within-cross-section objective the reported IC
    # measures. Re-measured under the real loss (supervised_lr_h2_r3 vs
    # _sin, 5 months x 3 LRs x 3 tasks, month- and LR-matched). At the chosen
    # 1e-5 the HEAD is a wash favouring sinusoidal on all three -- return
    # +0.0016 (t 1.0), volatility +0.0011 (t 1.0), spread +0.0004 (t 0.4):
    # none significant, none negative.
    #
    # THE PROBE IS NOT A WASH. On spread_change the representation is
    # decisively better under sinusoidal:
    #
    #   blr 5e-6  +0.0327 +- 0.0016  t +19.9  5/5 months
    #   blr 1e-5  +0.0223 +- 0.0054  t  +4.1  5/5 months
    #   blr 2e-5  +0.0148 +- 0.0071  t  +2.1  3/4 months
    #
    # against a spread probe IC of order +0.10, so roughly a fifth of the
    # signal, unanimous across months at both lower LRs. Read it with the
    # caveat that spread_change carries a known mechanical leak -- a scalar
    # available at t scores +0.13..+0.22 on it -- so some of what the better
    # representation buys may be access to that leak rather than to
    # forecastable structure.
    #
    # The head alone would not have settled this, and the original argument
    # was that it need not: at parity the tie-break is uniformity. With sinusoidal everywhere, a supervised
    # checkpoint and an SSL checkpoint scored at the prediction readout share
    # an architecture_signature -- transformer-384-last-...-sinusoidal-own --
    # and therefore ONE random-init floor. That makes "IC minus random"
    # comparable ACROSS arms rather than merely self-consistent within each,
    # and halves the floor compute. It also removes the last config axis on
    # which the supervised family differed from everything else, which is the
    # axis that produced a cls-pooled 2049-position loader bug and a wrongly
    # matched floor.
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="last",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))
    # CELLS, NOT CROPS. cross_stock with n_stocks=K draws K same-date tickers
    # over ONE shared wall-clock window ending on the anchor lattice, and the
    # dataset labels every one of them at that anchor, so a sample is a
    # (K, n_targets) cell and training_step takes its compute_group_loss
    # path. That is the quantity the reported rank IC is computed over. The
    # single-crop path it replaces (random_resized_crop, one global view)
    # could only pair rows that happened to land in the same (date, anchor):
    # ~74 pairs per 256 rows, most rows in none.
    #
    # NO industry_table: a cell here is a random subset of the cross-section
    # the eval ranks over, not a same-industry pair. Locals stay off -- the
    # head reads one global per stock.
    #
    # The eval/probe datasets never inherit this (pretrain.py pins them to
    # one rrc global): the probe embeds one stock per row.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="cross_stock", n_stocks=16)
    )


@dataclass
class MultiTaskSupervisedModeConfig:
    """Multi-head supervised mode: one backbone, one head per task, with
    independently normalized gradients over the shared backbone parameters.

    ``task_weights`` controls the relative contribution of each task to the
    shared-backbone gradient (missing entries default to 1.0, and the weights
    are renormalized to sum to 1). Task-head gradients are not normalized
    because the heads do not share parameters.
    """

    _target_: str = "market_jepa.modeling.modes.supervised.MultiTaskSupervisedModel"
    tasks: list[str] = MISSING
    # Same objective axis as SupervisedModeConfig, and the binned arm means the
    # SAME thing here: a k-way head over quantile bins of the RAW target, one
    # head and one Discretizer per task. See that config for what each knob
    # does; the defaults below are deliberately identical to it, so a multihead
    # run and the three single-task specialists differ in the shared trunk and
    # in NOTHING ELSE. A run that also changed objective could not be read as
    # "what does sharing a backbone cost".
    #
    # PAIRWISE, matching SupervisedModeConfig. It was "mse" until 2026-09-08,
    # on the reasoning that non-sweep callers (ssl_finetune_multihead,
    # architectures) should not have their objective move underneath them and
    # that "sweeps pin it, as they pin everything else".
    #
    # That reasoning inverted the cost. Every sweep did pin pairwise, so the
    # default was never what ran -- it was only what ran when someone FORGOT,
    # and a multihead that quietly trains MSE while the specialists train LTR
    # is not comparable to them at all. It is the same failure that left
    # pool/pos_embed at cls/learned for the whole full-history wave. The
    # default now IS the recipe, so forgetting is harmless and the sweep files
    # have nothing to restate. See scripts/sweeps/README.md.
    loss_fn: str = "pairwise"
    smooth_l1_beta: float = 1.0
    # NO FINETUNE PATH. init_backbone_from / freeze_backbone were removed
    # 2026-09-16: the multihead end-to-end finetune arm is not being run, and
    # the head-only arm is not being run in either mode. The finetune lives on
    # SupervisedModeConfig, one task at a time, which is what the three
    # reported targets are read as anyway.
    # Gradient balancing across tasks. Each task's objective is divided by an
    # EMA of its backbone-gradient norm (floored at gradient_norm_min, with the
    # resulting scale capped at gradient_norm_max_scale to survive a task whose
    # gradient briefly collapses).
    task_weights: dict[str, float] = field(default_factory=dict)
    gradient_norm_ema_decay: float = 0.99
    gradient_norm_min: float = 1.0e-4
    gradient_norm_max_scale: float = 10.0
    # Same default as SupervisedModeConfig (see comment there). Normalization
    # fixes the trunk gradient magnitude at sum(task_weights)=1, so the
    # single-task blr remains the right starting point.
    # Same measured budget as SupervisedModeConfig -- see the comment there. A
    # multihead differing from the specialists in its training budget as well
    # as its shared trunk could not be read as "what does sharing a trunk
    # cost".
    #
    # CELLS, like the specialists (2026-09-11): the batch is 16 cells of 16
    # stocks and MultiTaskSupervisedModel.training_step flattens each cell
    # to K rows sharing one xs_cell id, so every head ranks within a
    # cross-section with K-1 partners per row. The old 256-row recipe left
    # the trunk 0.15% from its init; see SupervisedModeConfig.
    #
    # 256 cells a step, 12 passes over the six-month span and blr 2e-4, the
    # specialists' settled recipe (2026-09-13) -- see SupervisedModeConfig for
    # the measurements. Single-month callers pin their own epochs.
    training_overrides: ModeTrainingOverrides = field(
        default_factory=lambda: ModeTrainingOverrides(
            blr=2e-4, per_device_train_batch_size=16, effective_batch_size=256,
            num_epochs=12)
    )
    # SAME BACKBONE AS THE SPECIALISTS, and it has to be spelled out here.
    #
    # Omitting this block does NOT inherit SupervisedModeConfig's choice -- the
    # two mode configs are siblings, so with no override the transformer falls
    # through to its own schema defaults, which are pool=None and pos_embed=None
    # and resolve in the backbone to "cls" and "learned"
    # (backbones/transformer.py: `pool = "cls" if pool is None else pool`, and
    # `pos_embed_kind = str(... or "learned")`). That is what every multihead
    # month trained under until 2026-09-08.
    #
    # WHY IT MATTERS. The multihead exists to answer what SHARING A TRUNK
    # costs, and this file's own header claims it "differs from them in the
    # trunk and in NOTHING ELSE". While these were unset that claim was false
    # in two further ways at once -- cls vs last pooling and learned vs rotary
    # position -- so the reported multihead-minus-specialist delta was three
    # changes bundled into one number rather than the trunk on its own. The
    # 138 runs trained that way were deleted rather than kept, because a month
    # that looks like every other month in the project but was trained on a
    # different encoder is exactly the kind of thing that gets pooled by a
    # later reader.
    backbone: TransformerBackboneConfig = field(
        default_factory=lambda: TransformerBackboneConfig(
            pool="last",
            config=TransformerInnerConfig(pos_embed="sinusoidal")))
    # CELLS, like the specialists -- see SupervisedModeConfig.dataset_overrides.
    # training_step flattens a (B, K) cell to K rows with a shared xs_cell id.
    dataset_overrides: ModeDatasetOverrides = field(
        default_factory=lambda: ModeDatasetOverrides(
            n_global_views=1, n_local_views=0, name="cross_stock", n_stocks=16)
    )


# ── Top-Level Config ─────────────────────────────────────────────────────────


@dataclass
class Config:
    # Default group selections (overridable via CLI: backbone=resnet, mode=cpc, …).
    # `_self_` first means: my MISSING fields are placeholders; group
    # selections fill them in. Mirrors what conf/config.yaml used to encode.
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"backbone": "transformer"},
            {"mode": "lejepa"},
            {"machine": "hub"},
        ]
    )
    backbone: Any = MISSING
    mode: Any = MISSING
    machine: MachineConfig = MISSING
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    probe_eval: ProbeEvalConfig = field(default_factory=ProbeEvalConfig)
    skip_if_done: bool = True
