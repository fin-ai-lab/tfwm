"""Streaming dataset for JEPA training using MosaicML Streaming (MDS) format.

Each MDS sample is one date-ticker observation with variable-length ndarray
columns.  StreamingMarketDataset subclasses StreamingDataset and adds
preprocessing (canonical 1 Hz timeline, forward-fill, zero-fill) plus
augmentation / normalization / tensorization in __getitem__.
"""

import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from stable_finance.dataset.mds import (
    StreamingMarketDataset as MDSMarketDataset,
    discover_streams,
    infer_period_frequency as _infer_freq,
    validate_period_alignment as _validate_date_alignment,
)
from stable_finance.dataset.schema import MARKET_SCHEMA

from market_jepa.augmentations import (
    AUGMENTATION_REGISTRY,
    _add_volume_noise_numpy,
    _aggregate_numpy_jittered,
    _draw_channel_mask,
    _price_jitter_numpy,
    _random_resized_crop_numpy,
    _warped_aggregate_numpy,
    canonicalize_augmentations,
    encode_view_metadata,
    prepare_augmented_view,
    N_WINDOW_INFO,
)
from stable_finance.dataset import SessionPreprocessor, standard_open_est
from market_jepa.training.utils import append_view_info, build_norm_groups
from stable_finance.dataset import AnchorTargetStats
from stable_finance.dataset.anchors import ANCHOR_STEP, RETURN_VWAP_WINDOW, SESSION_LEN

FEATURE_COLUMNS = list(MARKET_SCHEMA.columns)

# Compatibility names for analysis scripts. Stable-finance is the authority;
# these are projections of its schema, not a second set of fill rules.
_FFILL_COLS = [
    column for column in FEATURE_COLUMNS if column in MARKET_SCHEMA.forward_fill
]
_ZERO_FILL_COLS = [
    column for column in FEATURE_COLUMNS if column in MARKET_SCHEMA.zero_fill
]

# Risk factor column presets.
# "mid_price" is a synthetic column computed at load time as (bid + ask) / 2 —
# see _merge_risk_factors. It is not one of FEATURE_COLUMNS.
_RF_PRESETS = {
    "all": FEATURE_COLUMNS,
    "vwap": ["vwap_all"],
    "vwap_volume": ["vwap_all", "volume"],
    "vwap_volume_orderbook": [
        "vwap_all", "volume", "bid_price", "ask_price", "bid_size", "ask_size",
    ],
    "mid": ["mid_price"],
    "mid_volume": ["mid_price", "volume"],
    "mid_volume_orderbook": [
        "mid_price", "volume", "bid_size", "ask_size",
    ],
}

# Per-column aggregation rules (used by _aggregate_numpy_subset)
_AGG_RULES: dict[str, str] = {
    **MARKET_SCHEMA.aggregation_rules,
    "mid_price": "last",
}

logger = logging.getLogger(__name__)


_TICKER_DATE_SIDECAR = "ticker_date_map.json"


def _month_dir_shards(month_dir: Path) -> tuple[list[dict], int]:
    """Return (shard metadata list, total sample count) from a month's index.json."""
    with open(month_dir / "index.json") as f:
        index = json.load(f)
    shards = index["shards"]
    return shards, sum(s["samples"] for s in shards)


def ensure_ticker_date_sidecar(month_dir: str | Path) -> dict:
    """Return ``{"tickers": [...], "dates": [...]}`` aligned with MDS sample order.

    The map is required for cross-stock partner lookup because samples are
    shuffled on disk. Built by scanning the month's shards once and cached as
    a sidecar JSON inside the month directory, so it rides along when the
    month is rsynced to a cluster. The write is atomic (tmp + rename), which
    makes concurrent builders from independent SLURM jobs safe — they all
    produce identical content. A read-only month directory just skips the
    caching.
    """
    month_dir = Path(month_dir)
    shards, n_total = _month_dir_shards(month_dir)
    sidecar = month_dir / _TICKER_DATE_SIDECAR
    if sidecar.is_file():
        try:
            with open(sidecar) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = None
        if meta and len(meta.get("tickers", [])) == n_total:
            return meta
        logger.warning("Stale/corrupt %s (rebuilding): %s", _TICKER_DATE_SIDECAR, sidecar)

    # open_shard, not reader_from_json: older months keep a third of their
    # shards only as .zstd on bll01, and a bare reader dies on the first one
    # with a FileNotFoundError for a raw .mds nobody ever wrote. Expanding into
    # process-owned scratch reads them without touching the canonical mosaic.
    import tempfile

    from stable_finance.dataset.mds import open_shard

    logger.info("Scanning %s (%d samples) to build %s", month_dir, n_total, _TICKER_DATE_SIDECAR)
    tickers: list[str] = []
    dates: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ticker_date_") as tmpdir:
        for shard_meta in shards:
            reader = open_shard(month_dir, shard_meta, tmpdir)
            for i in range(shard_meta["samples"]):
                sample = reader.get_item(i)
                tickers.append(str(sample["ticker"]))
                dates.append(str(sample["date"]))
    meta = {"tickers": tickers, "dates": dates}

    tmp = month_dir / f"{_TICKER_DATE_SIDECAR}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, sidecar)
    except OSError as e:
        logger.warning("Could not cache %s (%s) — proceeding uncached", sidecar, e)
        tmp.unlink(missing_ok=True)
    return meta


def _load_anchor_stats(stats_dir, date_start, date_end):
    """Load and concatenate the ``YYYY-MM.npz`` tables covering a date window.

    Every month the window touches must be present: a missing table would
    silently NaN out that month's labels rather than fail, and a training run
    that quietly drops a third of its targets is far worse than one that
    refuses to start.
    """
    stats_dir = Path(stats_dir)
    months, y, m = [], date_start.year, date_start.month
    while (y, m) <= (date_end.year, date_end.month):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    paths = [stats_dir / f"{ym}.npz" for ym in months]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing anchor-stat tables in {stats_dir}: {missing}. "
            "Build them with `sf-build-targets` (stable_finance.dataset.build_targets)."
        )
    return AnchorTargetStats.load_months(paths)


def _draw_anchor_first(rng, cfg: dict, n_rows: int, seq_len: int, tod_offset: int,
                       grid: int, slack: int):
    """Draw (anchor_tod, agg, window, start) the way the EVAL PANEL does.

    PANEL SEMANTICS, so the head is trained on the distribution it is scored
    on: the anchor is uniform over the lattice band that admits a full view
    at the FINEST resolution, and the resolution is then uniform over the
    choices that fit before that anchor (stable_finance.dataset.panels:
    evenly_spaced_anchors + choose_cell_aggregation).

    WHY NOT THE OTHER ORDER. Drawing the window first and then an anchor
    where it fits -- what random_resized_crop and the original cross_stock
    draw do -- makes late anchors far more likely (every window fits before
    the close; only short ones fit at 13:00). Measured on 2012-12 with a
    spread head: 58% of training cells ended in the last two hours against
    the panel's 35%, and the head scored +0.156 per training cell but +0.072
    on the panel of the same month. The features were fine; the anchors
    were not where the metric looks.

    Returns None when no resolution fits, so the caller retries.
    """
    agg_range = cfg.get("global_agg_range")
    if agg_range is not None:
        a_lo, a_hi = int(agg_range[0]), int(agg_range[1])
    else:
        lo, hi = cfg["global_scale_range"]
        a_lo = max(1, round(lo * n_rows / seq_len))
        a_hi = max(a_lo, round(hi * n_rows / seq_len))
    # The band, in wall-clock seconds past the standard open: the finest view
    # must fit before the anchor (grid >= a_lo * seq_len - 1, as the panel
    # filters its grid) and the anchor must leave `slack` before the close
    # and lie inside this ticker's own session.
    first = -(-(a_lo * seq_len - 1) // grid) * grid
    last = min(tod_offset + n_rows - 1, SESSION_LEN - 1 - slack)
    if first > last:
        return None
    anchor = first + grid * int(rng.randint(0, (last - first) // grid + 1))
    feasible = [g for g in range(a_lo, a_hi + 1)
                if g * seq_len <= anchor + 1
                and anchor - tod_offset - g * seq_len + 1 >= 0]
    if not feasible:
        return None
    agg = feasible[int(rng.randint(0, len(feasible)))]
    window = agg * seq_len
    return anchor, agg, window, anchor - tod_offset - window + 1


def _xs_cell_id(date_str: str, anchor_tod: int) -> int:
    """Stable integer id for one cross-section, i.e. one (date, anchor).

    Samples that share this id were cropped to the same wall-clock instant and
    are therefore directly comparable — the unit the reported rank IC is
    computed over. Packed rather than hashed so it stays reproducible across
    processes (Python randomizes str hashing per process, which would hand a
    different grouping to every dataloader worker).
    """
    y, m, d = date_str.split("-")
    return ((int(y) * 10000 + int(m) * 100 + int(d)) * 100_000) + int(anchor_tod)


class StreamingMarketDataset(MDSMarketDataset):
    """Streaming dataset that reads MDS shards and produces augmented views.

    MDS schema (per sample):
        * ``ticker``       – ``str``
        * ``date``         – ``str``  (``YYYY-MM-DD``)
        * ``ts_interval``  – ``ndarray:int32``  (timestamps in seconds)
        * ``features``     – ``ndarray:float32`` (N × 9 feature matrix)

    ``__getitem__`` reconstructs a canonical 1 Hz DataFrame, picks a random
    augmentation, normalises the two views, and returns tensors.
    """

    def __init__(
        self,
        *,
        augmentations,
        date_start,
        date_end,
        feature_columns=None,
        zero_feature_columns=None,
        norm_mode="per_view",
        # THE INFORMATION TOKEN IS ON BY DEFAULT, as it is in DatasetConfig and
        # in DayStoreCellDataset. These two read False until 2026-09-13, which
        # made the default a property of WHICH DATASET CLASS you constructed:
        # pretrain.py passes both explicitly so training was never affected,
        # but every direct construction got a silently narrower view than the
        # schema describes. Callers that want the 9-channel view -- the frozen
        # TSFM arms -- pass False and say so.
        info_norm_stats=True,
        info_window=True,
        seed=42,
        n_pairs_per_obs=1,
        risk_factor_dir=None,
        risk_factor_tickers=None,
        risk_factor_columns=None,
        targets=None,
        epoch_dependent_seed=False,
        schedule=None,
        data_fraction=1.0,
        extended_hours=False,
        xs_anchor_stats_dir=None,
        label_at_view_start=False,
        risk_factor_targets=True,
        xs_target="uniform",
        grid_cache_gb=0.0,
        **kwargs,
    ):
        """
        Args:
            augmentations: List of augmentation configs (str or dict with ``name``).
            date_start: Inclusive start date for sample-level filtering.
            date_end: Inclusive end date for sample-level filtering.
            feature_columns: Override default feature column list.
            seed: Base random seed.
            n_pairs_per_obs: Number of augmented pairs to generate per
                disk observation.  Each pair uses a distinct deterministic
                RNG so results are reproducible.  Default 1 (current
                behaviour).
            risk_factor_dir: Path to risk factor directory (e.g. ``…/1Hz_risk_factors``).
            risk_factor_tickers: List of risk factor tickers to load (e.g. ``["IWM"]``).
            risk_factor_columns: Feature subset — a preset string (``"all"``,
                ``"vwap"``, ``"vwap_volume"``, ``"vwap_volume_orderbook"``) or
                an explicit list of column names.
            epoch_dependent_seed: If True, mix the current epoch into the
                augmentation seed so each epoch produces different random
                crops for the same observation.  Default False (fully
                deterministic across epochs).
            data_fraction: Fraction of distinct disk observations to expose
                (label-efficiency experiments).  A deterministic subset of
                size ``round(num_samples * data_fraction)`` is drawn with a
                seed-stable permutation; every requested index is remapped
                into that subset.  The dataset's own length is unchanged —
                the training loop (pretrain.py) shrinks steps_per_epoch to
                the subset size so num_epochs-based runs do proportionally
                less work.  Default 1.0 (all).
            extended_hours: Build the 1 Hz grid over the extended session
                (04:00 ET to four hours past the close) rather than regular
                hours only.  The grid length then varies per ticker-day, so
                pair this with the augmentations' ``global_agg_range`` /
                ``local_agg_range`` to keep view resolution pinned.
            **kwargs: Forwarded to ``StreamingDataset`` (``streams``, ``shuffle``,
                      ``batch_size``, ``allow_unsafe_types``, etc.).
        """
        super().__init__(**kwargs)

        self._schedule = schedule
        self._extended_hours = bool(extended_hours)
        self.n_pairs_per_obs = n_pairs_per_obs
        self._epoch_dependent_seed = epoch_dependent_seed
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self._base_seed = seed
        self._getitem_counter = 0

        if not (0.0 < data_fraction <= 1.0):
            raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
        self._data_fraction = float(data_fraction)
        # Built lazily on first access (num_samples is known after
        # super().__init__, but workers rebuild identically from the seed).
        self._fraction_subset: np.ndarray | None = None

        # Sample-level date filter. Always on; for month-aligned windows it's
        # a no-op since every sample's date is already inside the bounds. For
        # callers that pass a non-month-aligned window (via
        # ``discover_streams(..., allow_unaligned_dates=True)``), the streams
        # cover full bracketing months and observations outside the window
        # are dropped to a zero-sample in ``__getitem__``.
        self._date_start = datetime.date.fromisoformat(str(date_start))
        self._date_end = datetime.date.fromisoformat(str(date_end))

        # Parse augmentation configs (same pattern as MarketDataset)
        self._augmentation_configs: list[dict] = []
        for aug in augmentations:
            if isinstance(aug, str):
                config = {"name": aug}
            elif isinstance(aug, dict):
                if "name" not in aug:
                    raise ValueError(f"Augmentation dict must have 'name' key: {aug}")
                config = aug.copy()
            else:
                raise ValueError(
                    f"Augmentation must be str or dict, got {type(aug)}: {aug}"
                )
            name = config["name"]
            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(
                    f"Unknown augmentation '{name}'. "
                    f"Available: {sorted(AUGMENTATION_REGISTRY)}"
                )
            self._augmentation_configs.append(config)

        # Feature ablation: columns zeroed in the dense grid before any
        # augmentation/normalization (see DatasetConfig.zero_feature_columns).
        zero_cols = list(zero_feature_columns or [])
        unknown = [c for c in zero_cols if c not in self.feature_columns]
        if unknown:
            raise ValueError(
                f"zero_feature_columns not in feature_columns: {unknown}"
            )
        # Stable-finance owns reconstruction and the bounded LRU. The policy
        # remains a training concern: this object is copied into every loader
        # worker, so ``grid_cache_gb`` is still a per-worker cap.
        self._session_preprocessor = SessionPreprocessor(
            schedule=self._schedule,
            extended_hours=self._extended_hours,
            zero_features=tuple(zero_cols),
            cache_bytes=int(float(grid_cache_gb) * (1 << 30)),
        )

        self._aug_configs = canonicalize_augmentations(self._augmentation_configs)

        # Relative sampling weights across the configured augmentations
        # (AugmentationConfig.weight): each drawn pair picks config i with
        # probability weight_i / sum(weights). The all-equal case keeps the
        # legacy randint draw so existing configs preserve their exact
        # per-pair RNG stream.
        _aug_w = np.asarray(
            [float(c.get("weight", 1.0)) for c in self._augmentation_configs],
            dtype=np.float64,
        )
        if _aug_w.size and (_aug_w <= 0).any():
            raise ValueError(
                f"augmentation weights must be > 0, got {_aug_w.tolist()}"
            )
        self._aug_uniform = _aug_w.size == 0 or bool(np.all(_aug_w == _aug_w[0]))
        self._aug_cum_probs = (
            np.cumsum(_aug_w / _aug_w.sum()) if _aug_w.size else None
        )
        # NORM_MODE "none" IS EXPRESSED AS AN EMPTY GROUP LIST, deliberately.
        # normalize_numpy loops over the groups, so no groups means it is a
        # no-op wherever it is called -- all eight call sites, including the
        # local views and the probe path -- and the ablation cannot be half
        # applied because one branch was missed. See DatasetConfig.norm_mode.
        if norm_mode not in ("per_view", "none"):
            raise ValueError(f"unknown norm_mode: {norm_mode!r}")
        self.norm_mode = norm_mode
        self._norm_groups = (
            [] if norm_mode == "none" else build_norm_groups(self.feature_columns)
        )
        # THE INFORMATION TOKEN. Stable-finance represents these facts as
        # ViewMetadata; market-jepa encodes them into a final-row payload in
        # reserved columns, which the backbone projects into one token before
        # patch embedding. They are never repeated along the time series.
        #
        # Give the per-view (mu, sigma) back instead of dropping it. Under
        # norm_mode=none there is nothing to give back and the flag would
        # silently widen the view by zero -- a model built for 17 columns fed 9
        # -- so the combination is rejected rather than quietly ignored.
        if info_norm_stats and norm_mode == "none":
            raise ValueError(
                "info_norm_stats needs a normalization to report; it is "
                "meaningless with norm_mode=none"
            )
        self.info_norm_stats = bool(info_norm_stats)
        # Three numbers about the window -- start, end, log(seconds per token).
        # Encoded by augmentations.encode_view_metadata.
        self.info_window = bool(info_window)
        self._n_info_features = (
            (2 * len(self._norm_groups) if self.info_norm_stats else 0)
            + (N_WINDOW_INFO if self.info_window else 0)
        )

        # ---- Risk factor setup ----
        self._risk_factors: dict[str, dict] = {}
        self._n_rf_features = 0

        if (
            risk_factor_dir
            and risk_factor_tickers
            and any(c["name"] == "time_warp" for c in self._aug_configs)
        ):
            # time_warp re-aggregates onto a non-uniform bucket grid, which
            # breaks the wall-clock alignment _merge_risk_factors assumes.
            # Checked here rather than at draw time so the run dies at
            # construction instead of inside a dataloader worker.
            raise ValueError(
                "time_warp with risk factors is not supported — the warped "
                "bucket grid breaks the wall-clock alignment "
                "_merge_risk_factors assumes. Clear risk_factor_tickers or "
                "pick a different augmentation."
            )

        if risk_factor_dir and risk_factor_tickers:
            # Resolve column subset
            if isinstance(risk_factor_columns, str):
                rf_cols = list(_RF_PRESETS[risk_factor_columns])
            else:
                rf_cols = list(risk_factor_columns or FEATURE_COLUMNS)
            self._rf_col_names = rf_cols

            # Build extraction plan: which raw FEATURE_COLUMNS indices to load
            # from disk, and how to derive each output column from them. Most
            # cols are passthrough; "mid_price" synthesizes (bid + ask) / 2.
            self._rf_raw_indices: list[int] = []
            self._rf_post_specs: list[tuple] = []
            _raw_idx_pos: dict[int, int] = {}

            def _ensure_raw(raw_idx: int) -> int:
                pos = _raw_idx_pos.get(raw_idx)
                if pos is None:
                    pos = len(self._rf_raw_indices)
                    self._rf_raw_indices.append(raw_idx)
                    _raw_idx_pos[raw_idx] = pos
                return pos

            for c in rf_cols:
                if c == "mid_price":
                    b = _ensure_raw(FEATURE_COLUMNS.index("bid_price"))
                    a = _ensure_raw(FEATURE_COLUMNS.index("ask_price"))
                    self._rf_post_specs.append(("mid", b, a))
                else:
                    p = _ensure_raw(FEATURE_COLUMNS.index(c))
                    self._rf_post_specs.append(("col", p))

            self._n_rf_features = len(rf_cols) * len(risk_factor_tickers)

            # Build aggregation rule list for the RF column subset
            self._rf_agg_rules = [_AGG_RULES[c] for c in rf_cols]
            # Build normalization groups for the RF column subset
            self._rf_norm_groups = build_norm_groups(rf_cols)

            # Find the vwap index within the RF subset (for volume-weighted avg)
            self._rf_vwap_local_idx = (
                rf_cols.index("vwap_all") if "vwap_all" in rf_cols else None
            )
            self._rf_volume_local_idx = (
                rf_cols.index("volume") if "volume" in rf_cols else None
            )

            from .risk_factors import RiskFactorMerger

            self._rf_merger = RiskFactorMerger(
                risk_factor_dir, list(risk_factor_tickers), rf_cols,
                FEATURE_COLUMNS, _AGG_RULES,
            )
            self._risk_factors = self._rf_merger.series

        # ---- Target computation setup ----
        self._targets_config = targets
        self._target_names: list[str] | None = None
        self._n_targets = 0
        if targets is not None:
            from stable_finance.dataset import get_target_names

            self._target_horizons = targets.get("horizons", [300, 600, 900])
            self._target_types = targets.get("types", ["return", "spread_change", "volatility_change"])

            # Determine RF price source (based on risk_factor_columns config)
            self._rf_price_mode = None  # None = skip RF targets
            rf_tickers_for_targets = []
            if self._risk_factors and risk_factor_targets:
                rf_col_names = getattr(self, "_rf_col_names", [])
                if "mid_price" in rf_col_names or (
                    "bid_price" in rf_col_names and "ask_price" in rf_col_names
                ):
                    self._rf_price_mode = "mid"
                elif "vwap_all" in rf_col_names:
                    self._rf_price_mode = "vwap"
                if self._rf_price_mode:
                    rf_tickers_for_targets = list(self._risk_factors.keys())

            self._target_names = get_target_names(
                self._target_horizons, self._target_types, rf_tickers_for_targets,
            )
            self._n_targets = len(self._target_names)

        # ---- Cross-sectional z-score setup ----
        # Every emitted target becomes z = (y_i - mu_t) / sigma_t against the
        # whole cross-section at its anchor. The tables are the only place
        # that cross-section exists (a streaming sample sees one stock), so
        # they are loaded once per worker and looked up per sample.
        self._xs_target = xs_target
        # Label instant = the view's FIRST row rather than its last. See
        # DatasetConfig.label_at_view_start.
        self._label_at_view_start = bool(label_at_view_start)
        self._xs_stats = None
        if xs_anchor_stats_dir:
            if targets is None:
                raise ValueError(
                    "xs_anchor_stats_dir is set but no targets are configured "
                    "— there is nothing to standardize."
                )
            if rf_tickers_for_targets:
                # get_target_names would append raw risk-adjusted columns the
                # anchor tables do not tabulate, silently mixing z-scored and
                # unscaled columns in one target vector.
                raise ValueError(
                    "xs_anchor_stats_dir is incompatible with risk-factor "
                    "target columns; clear risk_factor_tickers."
                )
            # xs_target="raw" keeps everything below — the anchor lattice, the
            # slack pin, the feasibility rules — and only skips the z
            # transform itself, so a raw-target arm differs from the z arm in
            # exactly one respect.
            self._xs_stats = _load_anchor_stats(
                xs_anchor_stats_dir, self._date_start, self._date_end,
            ) if xs_target != "raw" else None
            _stats_for_checks = self._xs_stats or _load_anchor_stats(
                xs_anchor_stats_dir, self._date_start, self._date_end,
            )
            missing_h = [
                h for h in self._target_horizons
                if int(h) not in set(_stats_for_checks.horizons.tolist())
            ]
            missing_t = [t for t in self._target_types
                         if t not in _stats_for_checks.types]
            if missing_h or missing_t:
                raise ValueError(
                    f"Anchor tables in {xs_anchor_stats_dir} lack "
                    f"horizons={missing_h} types={missing_t}; rebuild them."
                )
            # Off-lattice view ends have no (mu, sigma) at all, so snapping is
            # not optional once the tables are in play.
            #
            # A single-horizon target set (supervised training) also gets its
            # slack pinned to that horizon: every draw then lands in the
            # no-clamp region instead of ~31% (at h=7200) coming back NaN, and
            # the anchor distribution matches the eval cross-sections, which
            # apply slack >= h as a row filter. Multi-horizon sets (the probe's
            # 18 columns) can't share one bound, so they keep 0 and rely on the
            # per-column NaN mask.
            #
            # THE SLACK MUST COVER THE RETURN'S FORWARD VWAP WINDOW TOO. The
            # target is measured between vwap[t, t+w) and vwap[t+h, t+h+w), so
            # a draw with only h rows left has no forward window and its
            # target comes back NaN. Costs nothing in practice: anchors sit on
            # a 300 s lattice, and h+60 lands on the same last usable anchor as
            # h at every horizon we train.
            uniq_h = {int(h) for h in self._target_horizons}
            # min(), NOT pop(). Under the len == 1 guard the two return the
            # same number, but pop() EMPTIES the set, and the start-anchored
            # branch below reads max(uniq_h) -- which then raises on an empty
            # sequence in exactly the single-horizon case the day decoder runs.
            slack = (min(uniq_h) + RETURN_VWAP_WINDOW) if len(uniq_h) == 1 else 0

            if self._label_at_view_start:
                # MULTIPLE HORIZONS ARE FINE HERE, unlike the end-anchored
                # case. Every horizon is measured forward from the SAME
                # instant — the view's first row — so one slack serves them
                # all: reserve enough session for the longest and the shorter
                # ones fit inside it by construction. (End-anchored views
                # cannot do this: there each horizon wants its own last usable
                # anchor, which is why that path pins slack only when the
                # target set is a single horizon.)
                #
                # This is what lets one full-day view carry the whole event
                # ladder — 15 min, 2 h, and the close — off a single pass.
                slack = max(uniq_h) + RETURN_VWAP_WINDOW
                for c in self._aug_configs:
                    if c["name"] != "random_resized_crop":
                        raise ValueError(
                            "label_at_view_start is only implemented for "
                            f"random_resized_crop globals; got '{c['name']}'."
                        )
                    if not c["start_grid_sec"]:
                        c["start_grid_sec"] = ANCHOR_STEP
                    elif int(c["start_grid_sec"]) % ANCHOR_STEP != 0:
                        raise ValueError(
                            f"start_grid_sec={c['start_grid_sec']} must be a "
                            f"multiple of ANCHOR_STEP={ANCHOR_STEP}."
                        )
                    if not c["start_min_slack_sec"]:
                        c["start_min_slack_sec"] = slack
                    # A view has ONE label anchor. Leaving these set as well
                    # would over-determine a window whose length the scale
                    # draw already fixed, and _random_resized_crop_numpy
                    # rejects the pair outright.
                    c["end_grid_sec"] = None
                    c["end_min_slack_sec"] = 0

            # cross_stock is included because a supervised run over it needs
            # the SAME anchor geometry: its K stocks only form a cross-section
            # if their shared window ends on a tabulated anchor.
            #
            # Skipped entirely when the label is at the view's START: that
            # branch has already set the start-anchored pair and cleared the
            # end-anchored one, and a view has only one label anchor.
            for c in ([] if self._label_at_view_start else self._aug_configs):
                if c["name"] in ("random_resized_crop", "cross_stock"):
                    # An explicit end_grid_sec is a deliberate COARSENING of
                    # the anchor lattice: fewer distinct cells means samples
                    # collide into shared cross-sections more often, which is
                    # what lets a batch be ranked within cells without
                    # fetching partners. It must stay a multiple of
                    # ANCHOR_STEP or the crops land between tabulated anchors
                    # and every target comes back NaN.
                    g = c.get("end_grid_sec")
                    if not g:
                        c["end_grid_sec"] = ANCHOR_STEP
                    elif int(g) % ANCHOR_STEP != 0:
                        raise ValueError(
                            f"end_grid_sec={g} must be a multiple of "
                            f"ANCHOR_STEP={ANCHOR_STEP}; otherwise crops end "
                            f"between tabulated anchors and every "
                            f"cross-sectional target is NaN."
                        )
                    if not c["end_min_slack_sec"]:
                        c["end_min_slack_sec"] = slack

        # ---- Cross-stock partner lookup setup ----
        # Samples are shuffled on disk, so "another ticker, same date" needs
        # an explicit map. Sidecars are built once per month dir and cached.
        self._cs_tickers: np.ndarray | None = None
        self._cs_date_to_indices: dict[str, np.ndarray] | None = None
        if any(c["name"] == "cross_stock" for c in self._aug_configs):
            streams_arg = kwargs.get("streams")
            if not streams_arg:
                raise ValueError("cross_stock augmentation requires streams=[...]")
            all_tickers: list[str] = []
            all_dates: list[str] = []
            for stream in streams_arg:
                meta = ensure_ticker_date_sidecar(stream.local)
                all_tickers.extend(meta["tickers"])
                all_dates.extend(meta["dates"])
            if len(all_tickers) != self.num_samples:
                raise ValueError(
                    f"cross_stock sidecar total ({len(all_tickers)}) != dataset "
                    f"num_samples ({self.num_samples}); stale sidecar?"
                )
            self._cs_tickers = np.asarray(all_tickers)
            dates_arr = np.asarray(all_dates)

            # Optional month-level ticker -> ff49 map for hard same-industry
            # pairing. {month "YYYY-MM" -> {ticker -> ff49}}
            self._cs_ind_by_month: dict[str, dict[str, int]] | None = None
            ind_paths = {
                c["industry_table"] for c in self._aug_configs
                if c["name"] == "cross_stock" and c.get("industry_table")
            }
            if ind_paths:
                if len(ind_paths) > 1:
                    raise ValueError(f"Multiple industry_table paths: {ind_paths}")
                import pandas as pd

                ind_path = Path(next(iter(ind_paths)))
                ind_df = pd.read_parquet(ind_path)
                by_month: dict[str, dict[str, int]] = {}
                for m, t, ff in zip(
                    ind_df["month"], ind_df["ticker"], ind_df["ff49"]
                ):
                    by_month.setdefault(str(m), {})[str(t)] = int(ff)
                self._cs_ind_by_month = by_month
                sample_months = {d[:7] for d in all_dates}
                missing = sample_months - by_month.keys()
                if missing:
                    raise ValueError(
                        f"industry_table {ind_path} lacks dataset months: "
                        f"{sorted(missing)}"
                    )
                mapped = sum(
                    1 for t, d in zip(all_tickers, all_dates)
                    if t in by_month[d[:7]]
                )
                logger.info(
                    "Loaded industry_table %s: %d months, sample coverage %.1f%%",
                    ind_path, len(by_month), 100.0 * mapped / max(1, len(all_tickers)),
                )

            date_to_indices: dict[str, np.ndarray] = {}
            for d in np.unique(dates_arr):
                d_str = str(d)
                d_date = datetime.date.fromisoformat(d_str)
                if not (self._date_start <= d_date <= self._date_end):
                    continue
                date_to_indices[d_str] = np.flatnonzero(dates_arr == d)
            self._cs_date_to_indices = date_to_indices

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Total input width the model is handed.

        NOT all of it is a time series. ``_n_info_features`` counts the
        per-window values -- the norm stats and the three window descriptors --
        which occupy a final-row payload and are stripped by the
        backbone into one information token. The patch embedding only ever
        sees len(feature_columns) plus risk factors.
        """
        return (len(self.feature_columns) + self._n_info_features
                + self._n_rf_features)

    @property
    def target_names(self) -> list[str] | None:
        """Ordered target names matching the tensor layout, or None if disabled."""
        return self._target_names if self._targets_config else None

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def _remap_fraction_idx(self, idx: int) -> int:
        """Map a requested sample id into the data_fraction subset.

        Identity when data_fraction == 1.0. Otherwise a seed-stable random
        subset of sample ids is chosen once, and any requested id folds into
        it via modulo — uniform draws over [0, num_samples) become (nearly)
        uniform draws over the subset.
        """
        if self._data_fraction >= 1.0:
            return idx
        if self._fraction_subset is None:
            n = self.num_samples
            keep = max(1, int(round(n * self._data_fraction)))
            rng = np.random.default_rng(self._base_seed)
            self._fraction_subset = np.sort(rng.permutation(n)[:keep])
            logger.info(
                "data_fraction=%.4g: using %d of %d observations",
                self._data_fraction, keep, n,
            )
        return int(self._fraction_subset[idx % len(self._fraction_subset)])

    def _pick_aug_idx(self, rng) -> int:
        """Draw an augmentation-config index for one pair.

        Weighted draw (AugmentationConfig.weight) when weights differ; the
        all-equal case keeps the legacy randint call so existing configs
        preserve their exact per-pair RNG stream.
        """
        if self._aug_uniform:
            return rng.randint(0, len(self._aug_configs))
        return min(
            int(np.searchsorted(
                self._aug_cum_probs, rng.random_sample(), side="right",
            )),
            len(self._aug_configs) - 1,
        )

    def __getitem__(self, idx: int) -> list[dict]:
        # Note: the *original* idx feeds _getitem_numpy's augmentation seed,
        # so repeated visits to the same underlying observation still get
        # fresh crops.
        sample = self.raw_sample(self._remap_fraction_idx(idx))
        return self._getitem_numpy(idx, sample)

    def _getitem_numpy(self, idx: int, sample: dict) -> list[dict]:
        """Numpy-only __getitem__ — no DataFrames, no deepcopy, no groupby."""
        # Increment counter unconditionally (before any early return) so that
        # each __getitem__ call gets a unique seed.  With persistent_workers
        # and mega-epoch, the counter monotonically increases per worker for
        # the entire training run, guaranteeing unique augmentations even when
        # the same sample idx appears multiple times (due to resampling).
        self._getitem_counter += 1
        if self._epoch_dependent_seed:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            base_seed = int(
                (self._base_seed + idx + self._getitem_counter * 2_654_435_761 + worker_id * 48271)
                % (2**32)
            )
        else:
            base_seed = int((self._base_seed + idx) % (2**32))

        if self._schedule is not None and self._schedule.is_closed(sample.get("date", "")):
            return [self._zero_sample()]

        # Sample-level date-window filter. No-op for month-aligned
        # discover_streams callers; only drops samples when allow_unaligned_dates
        # streams pull in months that overlap but extend outside
        # [self._date_start, self._date_end].
        sample_date_str = sample.get("date", "")
        try:
            sample_date = datetime.date.fromisoformat(sample_date_str)
        except (TypeError, ValueError):
            sample_date = None
        if sample_date is not None and (
            sample_date < self._date_start or sample_date > self._date_end
        ):
            return [self._zero_sample()]

        result = self._preprocess_to_numpy(sample)
        if result is None or len(result[1]) < 10:
            n = 0 if result is None else len(result[1])
            logger.warning(
                "Zero-sample: preprocessing failed for idx=%d ticker=%s date=%s (rows=%d)",
                idx, sample.get("ticker", "?"), sample.get("date", "?"), n,
            )
            return [self._zero_sample()]
        canonical_sec, features = result

        # Pre-compute time-of-day offset (canonical_sec[0] relative to standard
        # 09:30 open). Used both for RF lookup (RF arrays are always indexed
        # from 09:30 — on late-open days like 2002-09-11 with 11:00 open the
        # offset must be relative to 09:30, not the actual open) and for the
        # meta.tod_sec / meta.tod_bucket labels consumed by the
        # uncorrelated_metrics probe.
        date_str = sample["date"]
        ts_std_open = standard_open_est(date_str)
        tod_offset_base = int(canonical_sec[0]) - ts_std_open
        rf_offset_base = tod_offset_base if self._risk_factors else None

        ticker_str = sample.get("ticker", "")
        try:
            weekday = datetime.date.fromisoformat(date_str).weekday()
        except (TypeError, ValueError):
            weekday = -1

        def _meta_fields(view_start_idx: int, agg_factor: int = 1) -> dict:
            """Per-sample probe-eval labels keyed off the start time of view[0]:
            ticker / date / weekday (constant per sample) and tod_sec / tod_bucket
            (depend on the chosen view start). Spread flat into the pair dict so
            ``collate_bucketed`` collects each field as a parallel list.

            ``agg_factor`` is view[0]'s aggregation scale in **seconds per output
            token**. It varies per sample (a random-resized crop picks its own
            scale), so a consumer that needs to reason in wall-clock seconds —
            e.g. the finance baselines converting a 300 s horizon into a number
            of steps — cannot recover it from the tensor shape alone.
            """
            tod_sec = tod_offset_base + int(view_start_idx)
            tod_bucket = min(12, max(0, tod_sec // 1800))
            return {
                "ticker": ticker_str,
                "date": date_str,
                "tod_sec": int(tod_sec),
                "tod_bucket": int(tod_bucket),
                "weekday": int(weekday),
                "agg_factor": int(agg_factor),
            }

        pairs = []
        for p in range(self.n_pairs_per_obs):
            pair_seed = int((base_seed + p * 6_364_136_223_846_793_005) % (2**32))
            rng = np.random.RandomState(pair_seed)

            # Pick augmentation config
            cfg_idx = self._pick_aug_idx(rng)
            cfg = self._aug_configs[cfg_idx]
            aug_name = cfg["name"]

            # === Random resized crop (multi-view, fractional scale) ===
            if aug_name == "random_resized_crop":
                n_global = cfg["n_global_views"]
                n_local = cfg["n_local_views"]
                global_scale = cfg["global_scale_range"]
                local_scale = cfg["local_scale_range"]
                global_seq_len = cfg["global_seq_len"]
                local_seq_len = cfg["local_seq_len"]
                global_agg = cfg.get("global_agg_range")
                local_agg = cfg.get("local_agg_range")
                date_str = sample["date"]

                all_views = []
                all_lengths = []
                ok = True
                g0_start = None
                g0_window = None
                g0_agg = 1

                # Only globals carry the label anchor, so only globals snap to
                # the lattice; locals keep their free positions.
                end_grid = cfg.get("end_grid_sec")
                end_slack = cfg.get("end_min_slack_sec", 0)
                start_grid = cfg.get("start_grid_sec")
                start_slack = cfg.get("start_min_slack_sec") or 0

                for view_type, n_views, scale_range, seq_len, agg_range in [
                    ("global", n_global, global_scale, global_seq_len, global_agg),
                    ("local", n_local, local_scale, local_seq_len, local_agg),
                ]:
                    is_global = view_type == "global"
                    collected = []
                    # A slack deadline rejects any (scale, anchor) draw whose
                    # window will not fit before it, and at h=7200 only the two
                    # finest resolutions ever fit — 2 attempts per view would
                    # drop most samples outright. Retries cost a few RNG draws.
                    max_attempts = (
                        12 * n_views
                        if (is_global and ((end_grid and end_slack)
                                           or (start_grid and start_slack)))
                        else 2 * n_views
                    )
                    for _ in range(max_attempts):
                        if len(collected) >= n_views:
                            break
                        view, v_start, v_agg, v_window = _random_resized_crop_numpy(
                            features, scale_range, seq_len, rng, agg_range=agg_range,
                            end_grid_sec=end_grid if is_global else None,
                            end_offset=tod_offset_base,
                            end_min_slack=end_slack if is_global else 0,
                            start_grid_sec=start_grid if is_global else None,
                            start_min_slack=start_slack if is_global else 0,
                        )
                        if view is None:
                            continue
                        collected.append((view, v_start, v_agg, v_window))

                    if len(collected) < n_views:
                        ok = False
                        break

                    for vi, (view, v_start, v_agg, v_window) in enumerate(collected):
                        # Track first global view for target computation
                        if view_type == "global" and vi == 0:
                            g0_start = v_start
                            g0_window = v_window
                            g0_agg = v_agg

                        prior_vwap = self._prior_vwap_numpy(features, v_start)
                        self._ffill_vwap_numpy(view, prior_vwap)
                        view = self._normalize_view(
                            view, tod_start_sec=tod_offset_base + v_start,
                            agg=v_agg)

                        if self._risk_factors:
                            rf_offset = rf_offset_base + v_start
                            view = self._merge_risk_factors(view, date_str, rf_offset, v_window, v_agg)

                        if np.isnan(view).any():
                            nan_cols = np.where(np.isnan(view).any(axis=0))[0].tolist()
                            logger.warning(
                                "NaN in %s view for idx=%d ticker=%s date=%s "
                                "start=%d agg=%d window=%d nan_cols=%s",
                                view_type, idx, sample.get("ticker", "?"),
                                date_str, v_start, v_agg, v_window, nan_cols,
                            )
                            np.nan_to_num(view, copy=False, nan=0.0)

                        t = torch.from_numpy(view.T.astype(np.float32))
                        all_views.append(t)
                        all_lengths.append(t.shape[-1])

                if not ok:
                    continue

                pair_dict = {
                    "views": all_views,
                    "lengths": torch.tensor(all_lengths, dtype=torch.long),
                    "bucket_key": cfg_idx,
                    "n_global_views": n_global,
                    **_meta_fields(g0_start if g0_start is not None else 0, g0_agg),
                }
                if self._targets_config is not None and g0_start is not None:
                    from stable_finance.dataset import compute_pair_targets

                    # A start-anchored view measures its target FORWARD
                    # from its first row (the day's open for the whole-session
                    # decoder), so that row — not the last — is the anchor.
                    t_idx = (g0_start if self._label_at_view_start
                             else g0_start + g0_window - 1)
                    # Cell identity: (date, wall-clock anchor). Samples drawn
                    # independently still LAND in shared cells — the anchor grid
                    # is finite — and a batch can be ranked within those cells
                    # without fetching any partners. How often they collide is
                    # set by how coarse the grid is; see _xs_cell_id.
                    pair_dict["xs_cell"] = torch.tensor(
                        _xs_cell_id(date_str, tod_offset_base + t_idx),
                        dtype=torch.int64,
                    )
                    rf_t = (rf_offset_base + t_idx) if rf_offset_base is not None else None
                    targets_np = compute_pair_targets(
                        focal_features=features,
                        t_idx=t_idx,
                        horizons=self._target_horizons,
                        types=self._target_types,
                        rf_data=self._risk_factors if getattr(self, "_rf_price_mode", None) else None,
                        rf_price_mode=getattr(self, "_rf_price_mode", None),
                        date_str=date_str,
                        rf_t_idx=rf_t,
                    )
                    target_meta = self._target_metadata(
                        targets_np, date_str, tod_offset_base + t_idx,
                    )
                    pair_dict["target_metadata"] = {
                        name: torch.from_numpy(value)
                        for name, value in target_meta.items()
                    }
                    pair_dict["targets"] = pair_dict["target_metadata"][self._xs_target]
                elif self._targets_config is not None:
                    pair_dict["targets"] = torch.full((self._n_targets,), float("nan"), dtype=torch.float32)
                pairs.append(pair_dict)
                continue

            # === Cross-stock: same wall-clock window across K tickers ===
            if aug_name == "cross_stock":
                n_stocks = cfg["n_stocks"]
                seq_len = cfg["global_seq_len"]
                candidates = (
                    self._cs_date_to_indices.get(date_str)
                    if self._cs_date_to_indices is not None else None
                )
                if candidates is None or len(candidates) < n_stocks:
                    # A day with fewer tickers than K can never form a cell.
                    # Say so once per date rather than failing every draw in
                    # silence: with K above the month's cross-section this
                    # branch is hit for EVERY sample and the run trains on
                    # nothing.
                    if not hasattr(self, "_cs_small_dates"):
                        self._cs_small_dates = set()
                    if date_str not in self._cs_small_dates:
                        self._cs_small_dates.add(date_str)
                        logger.warning(
                            "cross_stock: %s has %d tickers, fewer than n_stocks=%d; "
                            "no cell can be drawn on this date",
                            date_str, 0 if candidates is None else len(candidates), n_stocks)
                    continue

                N = len(features)
                cs_grid = cfg.get("end_grid_sec")
                cs_slack = cfg.get("end_min_slack_sec", 0) or 0
                anchor_first = bool(cs_grid and cfg.get("anchor_uniform"))
                if anchor_first:
                    # ANCHOR FIRST, resolution second -- the panel's draw. See
                    # _draw_anchor_first for why the order matters.
                    drawn = _draw_anchor_first(
                        rng, cfg, N, seq_len, tod_offset_base, cs_grid, cs_slack)
                    if drawn is None:
                        continue
                    anchor_tod, agg, window, start = drawn
                    if start < 0 or start + window > N:
                        continue
                else:
                    agg_range = cfg["global_agg_range"]
                    if agg_range is not None:
                        agg = int(rng.randint(agg_range[0], agg_range[1] + 1))
                    else:
                        scale = rng.uniform(*cfg["global_scale_range"])
                        agg = max(1, round(scale * N / seq_len))
                    window = agg * seq_len
                    if window > N:
                        agg = max(1, N // seq_len)
                        window = agg * seq_len
                    if window > N:
                        continue
                # SNAP THE WINDOW END TO THE ANCHOR LATTICE when targets are
                # in play. Without this the K stocks share a wall-clock window
                # but that window ends between tabulated anchors, so every
                # z/rank lookup misses and the batch is entirely NaN. The whole
                # point of a cross_stock supervised batch is that its K stocks
                # occupy ONE cell, which requires a real anchor.
                if anchor_first:
                    pass                      # anchor, agg, window, start drawn above
                elif cs_grid:
                    # Anchor in wall-clock seconds from the standard open. It
                    # must admit the full window behind it in the FOCAL's grid
                    # and leave `slack` before the close.
                    lo = tod_offset_base + window - 1
                    hi = min(tod_offset_base + N - 1, SESSION_LEN - 1 - cs_slack)
                    first = -(-lo // cs_grid) * cs_grid          # ceil to grid
                    if first > hi:
                        continue
                    n_opts = (hi - first) // cs_grid + 1
                    anchor_tod = first + cs_grid * int(rng.randint(0, n_opts))
                    start = anchor_tod - tod_offset_base - window + 1
                    if start < 0 or start + window > N:
                        continue
                else:
                    start = rng.randint(0, N - window + 1)
                # Window start in seconds from the standard 09:30 open — the
                # coordinate shared by every ticker's dense grid.
                focal_tod_start = tod_offset_base + start
                rf_offset = (rf_offset_base + start) if rf_offset_base is not None else 0

                focal_view = self._build_aligned_view(
                    features, start, window, agg, date_str, rf_offset,
                    tod_start_sec=tod_offset_base + start,
                )
                if focal_view is None:
                    continue

                n_local = cfg["n_local_views"]
                n_slots = n_local // n_stocks
                views = [focal_view]
                # Grids retained only when local views are needed (holding K
                # dense grids alive is wasteful for the large-K global-only
                # sweep configs).
                # Grids are also retained when targets are configured: a
                # supervised cross_stock batch needs EVERY stock's own target
                # at the shared anchor, not just the focal's, or there is
                # nothing to rank within the cell.
                grids = ([(features, start)]
                         if (n_slots or self._targets_config is not None)
                         else None)
                used_tickers = {ticker_str}
                # Same-industry restriction: partners must share the focal's
                # FF49 code. The unrestricted second pass keeps every sample
                # trainable — a focal with no aligned same-industry peer
                # degrades to a plain cross_stock draw instead of dropping.
                month_map: dict[str, int] = {}
                focal_ff = -1
                if cfg.get("industry_table") and self._cs_ind_by_month is not None:
                    month_map = self._cs_ind_by_month.get(date_str[:7], {})
                    focal_ff = month_map.get(ticker_str, -1)
                passes = [True, False] if focal_ff >= 0 else [False]

                # Scan candidates in random order until K-1 partners cover
                # the window; tickers whose first quote is after the window
                # start (or that fail preprocessing) are skipped. The break
                # below keeps the typical cost at ~K-1 fetches.
                order = rng.permutation(len(candidates))
                for restrict in passes:
                    if len(views) >= n_stocks:
                        break
                    for ci in order:
                        if len(views) >= n_stocks:
                            break
                        gidx = int(candidates[ci])
                        p_ticker = str(self._cs_tickers[gidx])
                        if p_ticker in used_tickers:
                            continue
                        if restrict and month_map.get(p_ticker, -1) != focal_ff:
                            continue
                        partner = self.raw_sample(gidx)
                        p_result = self._preprocess_to_numpy(partner)
                        if p_result is None:
                            continue
                        p_sec, p_feat = p_result
                        # Map the focal's wall-clock window into the partner's
                        # grid, whose row 0 is the partner's own first quote.
                        p_start = focal_tod_start - (int(p_sec[0]) - ts_std_open)
                        if p_start < 0 or p_start + window > len(p_feat):
                            continue
                        p_view = self._build_aligned_view(
                            p_feat, p_start, window, agg, date_str, rf_offset,
                            tod_start_sec=tod_offset_base + p_start,
                        )
                        if p_view is None:
                            continue
                        views.append(p_view)
                        if grids is not None:
                            grids.append((p_feat, p_start))
                        used_tickers.add(p_ticker)

                if len(views) < n_stocks:
                    logger.warning(
                        "cross_stock: only %d/%d aligned views for idx=%d date=%s start=%d agg=%d",
                        len(views), n_stocks, idx, date_str, start, agg,
                    )
                    continue

                # Matched local views: each slot draws ONE sub-window of the
                # shared wall-clock window and cuts it from every group
                # stock, so slot j's views are cross-stock positives at the
                # sub-moment scale. Slot lengths are exact (l_window =
                # l_agg * local_seq_len), so per-slot shapes always match.
                if n_slots:
                    l_seq = cfg["local_seq_len"]
                    l_scale = cfg["local_scale_range"]
                    local_ok = True
                    for _ in range(n_slots):
                        scale = rng.uniform(*l_scale)
                        l_agg = max(1, round(scale * window / l_seq))
                        l_window = l_agg * l_seq
                        if l_window > window:
                            l_agg = max(1, window // l_seq)
                            l_window = l_agg * l_seq
                        delta = rng.randint(0, window - l_window + 1)
                        for g_feat, g_start in grids:
                            lv = self._build_aligned_view(
                                g_feat, g_start + delta, l_window, l_agg,
                                date_str, rf_offset + delta,
                                tod_start_sec=tod_offset_base + g_start + delta,
                            )
                            if lv is None:
                                local_ok = False
                                break
                            views.append(lv)
                        if not local_ok:
                            break
                    if not local_ok:
                        continue

                pair_dict = {
                    "views": views,
                    "lengths": torch.tensor(
                        [v.shape[-1] for v in views], dtype=torch.long,
                    ),
                    "bucket_key": cfg_idx,
                    "n_global_views": n_stocks,
                    **_meta_fields(start, agg),
                }
                if cfg["structured_matching"]:
                    # Edge graph over the group's views instead of the flat
                    # pull-to-global-mean: global<->global, matched
                    # local<->local within each slot (cross-stock, same
                    # sub-window), and local<->its own stock's global.
                    n_views_total = len(views)
                    w = np.zeros((n_views_total, n_views_total), dtype=np.float32)
                    w[:n_stocks, :n_stocks] = 1.0
                    for j in range(n_slots):
                        base = n_stocks + j * n_stocks
                        w[base : base + n_stocks, base : base + n_stocks] = 1.0
                        for i in range(n_stocks):
                            w[base + i, i] = w[i, base + i] = 1.0
                    np.fill_diagonal(w, 0.0)
                    pair_dict["pair_weights"] = torch.from_numpy(w)
                if self._targets_config is not None:
                    from stable_finance.dataset import compute_pair_targets

                    t_idx = start + window - 1
                    # ONE anchor for the whole group: all K stocks share this
                    # wall-clock instant, which is what makes them a genuine
                    # cross-section and lets a ranking loss compare them
                    # against each other rather than across unrelated cells.
                    anchor_tod = tod_offset_base + t_idx
                    rf_t = (rf_offset_base + t_idx) if rf_offset_base is not None else None
                    metadata_rows = []
                    for g_feat, g_start in grids:
                        tn = compute_pair_targets(
                            focal_features=g_feat,
                            # Each stock's OWN grid index for the shared
                            # wall-clock instant: grids start at each ticker's
                            # first quote, so the index differs per stock even
                            # though the instant does not.
                            t_idx=g_start + window - 1,
                            horizons=self._target_horizons,
                            types=self._target_types,
                            rf_data=self._risk_factors if getattr(self, "_rf_price_mode", None) else None,
                            rf_price_mode=getattr(self, "_rf_price_mode", None),
                            date_str=date_str,
                            rf_t_idx=rf_t,
                        )
                        metadata_rows.append(
                            self._target_metadata(tn, date_str, anchor_tod)
                        )
                    pair_dict["target_metadata"] = {
                        name: torch.from_numpy(np.stack([
                            metadata[name] for metadata in metadata_rows
                        ]))
                        for name in metadata_rows[0]
                    }
                    # (K, n_targets): one labelled row per stock in the cell,
                    # instead of the focal's single row.
                    pair_dict["targets"] = pair_dict["target_metadata"][self._xs_target]
                    pair_dict["xs_group"] = n_stocks
                pairs.append(pair_dict)
                continue

            # === Same-stock invariance views ===
            # One shared wall-clock window (identical draw to cross_stock);
            # n_global_views copies of the SAME stock, differentiated only by
            # the transformation: a warp of the bucket grid (time_warp),
            # post-norm Gaussian noise (gaussian_noise), spurious trading
            # activity (volume_noise), within-spread ladder noise
            # (price_jitter), or post-norm channel masking (channel_drop).
            if aug_name in (
                "time_warp", "gaussian_noise", "volume_noise",
                "price_jitter", "channel_drop",
            ):
                n_views = cfg["n_global_views"]
                seq_len = cfg["global_seq_len"]
                N = len(features)
                agg_range = cfg["global_agg_range"]
                if agg_range is not None:
                    agg = int(rng.randint(agg_range[0], agg_range[1] + 1))
                else:
                    scale = rng.uniform(*cfg["global_scale_range"])
                    agg = max(1, round(scale * N / seq_len))
                window = agg * seq_len
                if window > N:
                    agg = max(1, N // seq_len)
                    window = agg * seq_len
                if window > N:
                    continue
                start = rng.randint(0, N - window + 1)
                rf_offset = (rf_offset_base + start) if rf_offset_base is not None else 0

                views = []
                if aug_name in ("gaussian_noise", "channel_drop"):
                    base = self._build_aligned_view(
                        features, start, window, agg, date_str, rf_offset,
                        tod_start_sec=tod_offset_base + start,
                    )
                    if base is None:
                        continue
                    if aug_name == "gaussian_noise":
                        sigma = cfg["noise_sigma"]
                        # SERIES CHANNELS ONLY, since 2026-08-29. This used to
                        # draw at base.shape, and base is NOT 9 channels: by
                        # this point _normalize_view has appended the per-group
                        # (mu, sigma) and the three window descriptors as
                        # reserved columns, so the 11-value information token
                        # was being noised too. Those are a per-WINDOW fact,
                        # not a per-token observation, and they are never
                        # standardized -- 0.75 is 0.7x price_sigma's natural
                        # spread and 8x tod_end's. It pushed bounded
                        # quantities out of range (tod_start, a fraction of the
                        # session, read 2.717) and, because the count groups
                        # are log1p'd before their mu is taken and log1p'd
                        # again by encode_view_metadata, moved implied volume
                        # from 8.5e3 to 6.4e8 shares a bucket.
                        #
                        # EVERY gaussian_noise RESULT REPORTED BEFORE THIS DATE
                        # IS ON THE OLD BEHAVIOUR -- the arm needs re-running
                        # before the numbers are final.
                        #
                        # It does hand the pair a bit-identical information
                        # token (plots/augmentations/info_token_shortcut.py
                        # measures how far that identifies a pair), which in
                        # principle lets an encoder cut the invariance term
                        # without reading the series. That is NOT a reason to
                        # keep the noise. time_warp leaves the three window
                        # descriptors exactly equal and is the strongest arm of
                        # the five, so availability plainly does not cause
                        # collapse; and SIGReg forbids the degenerate limit
                        # anyway -- the projection is 64-d, so a pure function
                        # of 11 constants lies on a <=11-dim manifold and fails
                        # the isotropy test in the other ~53 directions.
                        #
                        # Sliced by position, not as "all but the last k":
                        # _merge_risk_factors concatenates AFTER the info
                        # block, so the info channels are not trailing when
                        # risk factors are on.
                        i0 = len(self.feature_columns)
                        noise_rows = torch.ones(base.shape[0], dtype=torch.bool)
                        noise_rows[i0 : i0 + self._n_info_features] = False
                        n_rows = int(noise_rows.sum())
                        for _ in range(n_views):
                            v = base.clone()
                            noise = rng.standard_normal(size=(n_rows, base.shape[1]))
                            v[noise_rows] += sigma * torch.from_numpy(
                                noise.astype(np.float32))
                            views.append(v)
                    else:  # channel_drop — market channels only; rf rows ride
                        for _ in range(n_views):
                            v = base.clone()
                            v[_draw_channel_mask(rng, 9, cfg["channel_drop_p"]).tolist()] = 0.0
                            views.append(v)
                elif aug_name in ("volume_noise", "price_jitter"):
                    # One uniform aggregation of the shared window; per-view
                    # corruption in raw units, then the standard pipeline.
                    agg_view = _aggregate_numpy_jittered(
                        features[start : start + window], agg,
                    )
                    if agg_view is None:
                        continue
                    prior_vwap = self._prior_vwap_numpy(features, start)
                    for _ in range(n_views):
                        v = agg_view.copy()
                        if aug_name == "volume_noise":
                            _add_volume_noise_numpy(v, rng, cfg["vol_noise_frac"])
                        else:
                            _price_jitter_numpy(v, rng, cfg["price_jitter_frac"])
                        self._ffill_vwap_numpy(v, prior_vwap)
                        v = self._normalize_view(
                            v, tod_start_sec=tod_offset_base + start, agg=agg)
                        if self._risk_factors:
                            v = self._merge_risk_factors(v, date_str, rf_offset, window, agg)
                        if np.isnan(v).any():
                            np.nan_to_num(v, copy=False, nan=0.0)
                        views.append(torch.from_numpy(v.T.astype(np.float32)))
                else:  # time_warp — risk factors rejected in __init__
                    win_feats = features[start : start + window]
                    prior_vwap = self._prior_vwap_numpy(features, start)
                    warp_ok = True
                    for _ in range(n_views):
                        view = _warped_aggregate_numpy(
                            win_feats, seq_len, rng,
                            cfg["warp_knots"], cfg["warp_strength"],
                        )
                        if view is None:
                            warp_ok = False
                            break
                        self._ffill_vwap_numpy(view, prior_vwap)
                        # The warp resamples time non-uniformly, so there is no
                        # single seconds-per-token; report the nominal one the
                        # crop was drawn at, which is what the span still is.
                        view = self._normalize_view(
                            view, tod_start_sec=tod_offset_base + start,
                            agg=window / seq_len)
                        if np.isnan(view).any():
                            np.nan_to_num(view, copy=False, nan=0.0)
                        views.append(torch.from_numpy(view.T.astype(np.float32)))
                    if not warp_ok:
                        continue

                pair_dict = {
                    "views": views,
                    "lengths": torch.tensor(
                        [v.shape[-1] for v in views], dtype=torch.long,
                    ),
                    "bucket_key": cfg_idx,
                    "n_global_views": n_views,
                    **_meta_fields(start, agg),
                }
                if self._targets_config is not None:
                    from stable_finance.dataset import compute_pair_targets

                    t_idx = start + window - 1
                    rf_t = (rf_offset_base + t_idx) if rf_offset_base is not None else None
                    targets_np = compute_pair_targets(
                        focal_features=features,
                        t_idx=t_idx,
                        horizons=self._target_horizons,
                        types=self._target_types,
                        rf_data=self._risk_factors if getattr(self, "_rf_price_mode", None) else None,
                        rf_price_mode=getattr(self, "_rf_price_mode", None),
                        date_str=date_str,
                        rf_t_idx=rf_t,
                    )
                    target_meta = self._target_metadata(
                        targets_np, date_str, tod_offset_base + t_idx,
                    )
                    pair_dict["target_metadata"] = {
                        name: torch.from_numpy(value)
                        for name, value in target_meta.items()
                    }
                    pair_dict["targets"] = pair_dict["target_metadata"][self._xs_target]
                pairs.append(pair_dict)
                continue

            # === Fixed-length single-view crop (uncorrelated_metrics probe) ===
            if aug_name == "fixed_window":
                window_size_sec = cfg["window_size_sec"]
                if len(features) < window_size_sec + 60:
                    continue
                max_start = len(features) - window_size_sec
                start = rng.randint(0, max_start + 1)
                view = features[start : start + window_size_sec].copy()

                prior_vwap = self._prior_vwap_numpy(features, start)
                self._ffill_vwap_numpy(view, prior_vwap)
                view = self._normalize_view(
                    view, tod_start_sec=tod_offset_base + start, agg=1)

                if self._risk_factors:
                    rf_offset = rf_offset_base + start
                    view = self._merge_risk_factors(
                        view, date_str, rf_offset, window_size_sec, 1,
                    )

                if np.isnan(view).any():
                    np.nan_to_num(view, copy=False, nan=0.0)

                v1 = torch.from_numpy(view.T.astype(np.float32))
                pair_dict = {
                    "views": [v1],
                    "lengths": torch.tensor([v1.shape[-1]], dtype=torch.long),
                    "bucket_key": cfg_idx,
                    "n_global_views": 1,
                    # Raw 1 Hz slice — no aggregation, so 1 second per token.
                    **_meta_fields(start, 1),
                }
                if self._targets_config is not None:
                    pair_dict["targets"] = torch.full(
                        (self._n_targets,), float("nan"), dtype=torch.float32,
                    )
                pairs.append(pair_dict)
                continue

            window_size = cfg["window_size_sec"]

            # Window extraction — pure index slicing (1Hz = 1 row/sec)
            start_idx = rng.randint(0, max(1, len(features)))
            end_idx = min(start_idx + window_size, len(features))
            window = features[start_idx:end_idx]
            if len(window) < 10:
                continue

            prior_vwap = self._prior_vwap_numpy(features, start_idx)
            prior_vwap_v2 = prior_vwap  # overridden by contiguous below

            # === Dispatch: produce view1, view2 + RF params ===

            if aug_name == "multi_scale":
                scale_factors = cfg["scale_factors"]
                s1, s2 = sorted(rng.choice(scale_factors, 2, replace=False), reverse=True)
                view1 = _aggregate_numpy_jittered(window, s1)
                view2 = _aggregate_numpy_jittered(window, s2)
                if view1 is None or view2 is None:
                    continue
                # RF params: both views start at window begin, full window, own scale
                rf_offset_v1 = rf_offset_base + start_idx
                rf_offset_v2 = rf_offset_v1
                rf_window_v1 = end_idx - start_idx
                rf_window_v2 = rf_window_v1
                rf_scale_v1, rf_scale_v2 = s1, s2

            elif aug_name == "fast_timestamp_jittering":
                sf = cfg["scale_factor"]
                f = cfg["max_jitter_fraction"]
                min_delta = int(f * sf)
                max_delta = int((1 - f) * sf)
                if min_delta > max_delta:
                    min_delta, max_delta = max_delta, min_delta
                delta = min_delta if min_delta == max_delta else rng.randint(min_delta, max_delta + 1)

                view1 = _aggregate_numpy_jittered(window, sf, offset=0)
                view2 = _aggregate_numpy_jittered(window, sf, offset=delta)
                if view1 is None or view2 is None:
                    continue
                m = min(len(view1), len(view2))
                if m < 2:
                    continue
                view1, view2 = view1[:m], view2[:m]
                # RF params: view2 starts delta rows later, shorter window
                actual_window = end_idx - start_idx
                rf_offset_v1 = rf_offset_base + start_idx
                rf_offset_v2 = rf_offset_v1 + delta
                rf_window_v1 = actual_window
                rf_window_v2 = actual_window - delta
                rf_scale_v1, rf_scale_v2 = sf, sf

            elif aug_name == "contiguous":
                L1 = cfg["view1_length_sec"]
                L2 = cfg["view2_length_sec"]
                s1 = cfg["view1_scale"]
                s2 = cfg["view2_scale"]

                # 5-minute buffer to end of day
                if start_idx + L1 + L2 + 300 > len(features):
                    continue

                window1 = features[start_idx : start_idx + L1]
                window2 = features[start_idx + L1 : start_idx + L1 + L2]

                if len(window1) < 10 or len(window2) < 10:
                    continue

                view1 = _aggregate_numpy_jittered(window1, s1)
                view2 = _aggregate_numpy_jittered(window2, s2)
                if view1 is None or view2 is None:
                    continue

                # View2 prior VWAP: last valid vwap from view1's 1Hz region
                prior_vwap_v2 = self._prior_vwap_numpy(features, start_idx + L1)

                # RF params: each view covers its own time period
                rf_offset_v1 = rf_offset_base + start_idx
                rf_offset_v2 = rf_offset_base + start_idx + L1
                rf_window_v1 = L1
                rf_window_v2 = L2
                rf_scale_v1, rf_scale_v2 = s1, s2

            else:
                continue  # Unknown augmentation type

            # Shared post-processing
            self._ffill_vwap_numpy(view1, prior_vwap)
            self._ffill_vwap_numpy(view2, prior_vwap_v2)
            view1 = self._normalize_view(
                view1, tod_start_sec=tod_offset_base + rf_offset_v1 - rf_offset_base,
                agg=rf_window_v1 / len(view1))
            view2 = self._normalize_view(
                view2, tod_start_sec=tod_offset_base + rf_offset_v2 - rf_offset_base,
                agg=rf_window_v2 / len(view2))

            if self._risk_factors:
                view1 = self._merge_risk_factors(view1, sample["date"], rf_offset_v1, rf_window_v1, rf_scale_v1)
                view2 = self._merge_risk_factors(view2, sample["date"], rf_offset_v2, rf_window_v2, rf_scale_v2)

            # Compute prediction targets (before normalization data is lost)
            targets_np = None
            if self._targets_config is not None:
                from stable_finance.dataset import compute_pair_targets

                rf_t = (rf_offset_base + end_idx - 1) if rf_offset_base is not None else None
                targets_np = compute_pair_targets(
                    focal_features=features,
                    t_idx=end_idx - 1,
                    horizons=self._target_horizons,
                    types=self._target_types,
                    rf_data=self._risk_factors if getattr(self, "_rf_price_mode", None) else None,
                    rf_price_mode=getattr(self, "_rf_price_mode", None),
                    date_str=sample["date"],
                    rf_t_idx=rf_t,
                )
                target_meta = self._target_metadata(
                    targets_np, sample["date"], tod_offset_base + end_idx - 1,
                )

            v1 = torch.from_numpy(view1.T.astype(np.float32))
            v2 = torch.from_numpy(view2.T.astype(np.float32))
            lengths = torch.tensor([v1.shape[-1], v2.shape[-1]], dtype=torch.long)
            pair_dict = {
                "views": [v1, v2],
                "lengths": lengths,
                "bucket_key": cfg_idx,
                **_meta_fields(start_idx, rf_scale_v1),
            }
            if targets_np is not None:
                pair_dict["target_metadata"] = {
                    name: torch.from_numpy(value)
                    for name, value in target_meta.items()
                }
                pair_dict["targets"] = pair_dict["target_metadata"][self._xs_target]
            pairs.append(pair_dict)

        if not pairs:
            logger.warning(
                "Zero-sample: all %d augmentation attempts failed for idx=%d ticker=%s date=%s",
                self.n_pairs_per_obs, idx, sample.get("ticker", "?"), sample.get("date", "?"),
            )
            return [self._zero_sample()]
        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_sample(self) -> dict:
        """Minimal zero-padded sample for filtered / failed observations."""
        n_feat = self.n_features
        view = torch.zeros(n_feat, 1, dtype=torch.float32)
        result = {
            "views": [view, view.clone()],
            "lengths": torch.tensor([1, 1], dtype=torch.long),
            "bucket_key": -1,
        }
        if self._targets_config is not None:
            result["targets"] = torch.full((self._n_targets,), float("nan"), dtype=torch.float32)
        return result

    def _target_metadata(
        self, targets_np: np.ndarray, date_str: str, tod_anchor: int,
    ) -> dict[str, np.ndarray]:
        """Return every stable-finance target representation as metadata.

        ``tod_anchor`` is the label instant in seconds past the standard 09:30
        open — the key the anchor tables are built on. A view whose end is not
        on the lattice, or whose cell had too thin a cross-section, comes back
        all-NaN and is dropped by the NaN-masked loss.

        When no tables are configured only ``raw`` is available. Configured
        training pipelines normally provide tables and therefore expose all
        four representations to the supervised model.
        """
        if self._xs_stats is None:
            raw = np.asarray(targets_np, dtype=np.float32)
            missing = np.full(raw.shape, np.nan, dtype=np.float32)
            return {
                "raw": raw,
                "zscore": missing.copy(),
                "uniform": missing.copy(),
                "rank": missing.copy(),
            }
        bundle = self._xs_stats.transform(
            targets_np, date_str, tod_anchor,
            self._target_types, self._target_horizons,
        )
        return {
            name: value.astype(np.float32)
            for name, value in bundle.as_dict().items()
        }

    def _maybe_zscore(
        self, targets_np: np.ndarray, date_str: str, tod_anchor: int,
    ) -> np.ndarray:
        """Compatibility selector; transforms are owned by stable-finance."""
        metadata = self._target_metadata(targets_np, date_str, tod_anchor)
        try:
            return metadata[self._xs_target]
        except KeyError as exc:
            raise ValueError(
                f"unknown xs_target {self._xs_target!r}; expected one of "
                "'uniform', 'rank', 'zscore', or 'raw'"
            ) from exc

    def _maybe_log_grid_cache(self) -> None:
        """Report cache health every 20k reads.

        Counted on EVERY read, hit or miss. Testing the interval only on the
        miss path — which is what this did at first — makes the log go quiet
        exactly when the cache starts working: once hits dominate, misses stop
        landing on a multiple of the interval and the last line printed sticks
        at the warm-up hit rate forever.
        """
        info = self._session_preprocessor.cache_info()
        n = info.hits + info.misses
        if n == 0 or n % 20_000:
            return
        logger.info(
            "Grid cache: %.1f%% hit rate over %d reads, %d sessions resident, "
            "%.1f GiB", 100.0 * info.hits / n, n,
            info.entries, info.bytes / (1 << 30),
        )

    def _preprocess_to_numpy(self, sample: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """Preprocess MDS sample to raw numpy arrays (no DataFrame).

        Returns:
            (timestamps_sec: int32 1D, features: float64 2D (N, F)) or None.

        CACHED, because this is the single most expensive thing a dataloader
        worker does (~half its CPU) and it is recomputed identically once per
        epoch per sample — 100-200x over a run. The result depends only on
        (ticker, date) plus dataset-constant config, so (ticker, date) is the
        key. The MDS read that produces ``sample`` is ~1% of the cost and is
        NOT what is being avoided here; the grid construction is.

        Stored float32 and returned upcast to float64. ``features`` is float32
        on disk and this function only moves values, so the round trip is
        bitwise exact (tests/test_float32_grid_identity.py) while halving what
        the cache costs. The upcast also hands every caller a FRESH array, so
        in-place writes downstream can never corrupt a cached entry.
        """
        session = self._session_preprocessor.transform(sample)
        self._maybe_log_grid_cache()
        if session is None:
            return None
        if getattr(session, "bar_seconds", 1) != 1:
            # THE ANCHORING KNOBS ARE NAMED IN SECONDS AND SPENT AS ROWS.
            # end_min_slack_sec / start_min_slack_sec are horizons (seconds)
            # compared against len(features) (rows) inside
            # _random_resized_crop_numpy, and end_grid_sec / start_grid_sec
            # index the anchor lattice the same way. At 1 Hz the two units
            # coincide and every path is correct; at any other resolution they
            # silently differ by bar_seconds and a view lands off the lattice
            # with no z-score at all. stable-finance now records native
            # resolution and ships resample_session(session, 60), so this is
            # reachable rather than hypothetical -- fail loudly here until the
            # crop takes a resolution.
            raise ValueError(
                f"the view sampler assumes 1 Hz rows; this session is "
                f"{session.bar_seconds} s/bar. See the anchoring note in "
                f"market_jepa/augmentations.py."
            )
        return session.timestamps, session.features

    def _merge_risk_factors(
        self,
        view: np.ndarray,
        date_str: str,
        rf_offset: int,
        window_size: int,
        scale: int,
    ) -> np.ndarray:
        """Concatenate risk-factor input channels onto an aggregated view.

        Delegates to RiskFactorMerger so the synchronized-panel eval can build
        byte-identical views (see market_jepa/training/risk_factors.py). A
        second copy of this logic would silently break the one guarantee the
        reported metric depends on.
        """
        return self._rf_merger.merge(view, date_str, rf_offset, window_size, scale)

    def _aggregate_numpy_subset(
        self, features: np.ndarray, scale_factor: int,
    ) -> np.ndarray | None:
        """Aggregate (N, K) risk factor subset array at given scale.

        Unlike _aggregate_numpy which uses hardcoded column indices,
        this uses self._rf_agg_rules to apply the correct rule per column.
        """
        if scale_factor == 1:
            return features.copy()
        n = len(features)
        n_full = n // scale_factor
        remainder = n % scale_factor
        n_buckets = n_full + (1 if remainder > 0 else 0)
        if n_buckets < 2:
            return None

        k = features.shape[1]
        out = np.empty((n_buckets, k), dtype=np.float64)

        if n_full > 0:
            reshaped = features[:n_full * scale_factor].reshape(n_full, scale_factor, k)
            for ci, rule in enumerate(self._rf_agg_rules):
                if rule == "last":
                    out[:n_full, ci] = reshaped[:, -1, ci]
                elif rule == "max":
                    out[:n_full, ci] = reshaped[:, :, ci].max(axis=1)
                elif rule == "min":
                    out[:n_full, ci] = reshaped[:, :, ci].min(axis=1)
                elif rule == "sum":
                    out[:n_full, ci] = reshaped[:, :, ci].sum(axis=1)
                elif rule == "vwap":
                    # Volume-weighted average price
                    if self._rf_volume_local_idx is not None:
                        vol = reshaped[:, :, self._rf_volume_local_idx]
                        vwap_vals = reshaped[:, :, ci]
                        vol_sum = vol.sum(axis=1)
                        with np.errstate(invalid="ignore"):
                            out[:n_full, ci] = np.where(
                                vol_sum > 0,
                                (vwap_vals * vol).sum(axis=1) / vol_sum,
                                np.nan,
                            )
                    else:
                        # No volume column available — fall back to simple mean
                        out[:n_full, ci] = reshaped[:, :, ci].mean(axis=1)

        if remainder > 0:
            p = features[n_full * scale_factor:]
            for ci, rule in enumerate(self._rf_agg_rules):
                if rule == "last":
                    out[n_full, ci] = p[-1, ci]
                elif rule == "max":
                    out[n_full, ci] = p[:, ci].max()
                elif rule == "min":
                    out[n_full, ci] = p[:, ci].min()
                elif rule == "sum":
                    out[n_full, ci] = p[:, ci].sum()
                elif rule == "vwap":
                    if self._rf_volume_local_idx is not None:
                        vs = p[:, self._rf_volume_local_idx].sum()
                        out[n_full, ci] = (
                            (p[:, ci] * p[:, self._rf_volume_local_idx]).sum() / vs
                            if vs > 0 else np.nan
                        )
                    else:
                        out[n_full, ci] = p[:, ci].mean()

        return out

    @staticmethod
    def _ffill_column(arr: np.ndarray, col: int) -> None:
        """Forward-fill NaN in a single column, in-place. Zero backstop."""
        vals = arr[:, col]
        mask = np.isnan(vals)
        if not mask.any():
            return
        idx = np.arange(len(vals))
        idx[mask] = 0
        np.maximum.accumulate(idx, out=idx)
        vals[:] = vals[idx]
        # Backstop: zero for any leading NaN that couldn't be forward-filled
        still_nan = np.isnan(vals)
        if still_nan.any():
            vals[still_nan] = 0.0

    def _ffill_vwap_numpy(self, view: np.ndarray, prior: float | None) -> None:
        """Delegates to training.utils.ffill_vwap (shared with the eval path)."""
        from .utils import ffill_vwap

        ffill_vwap(view, prior)

    def _build_aligned_view(
        self,
        features: np.ndarray,
        start_idx: int,
        window: int,
        agg: int,
        date_str: str,
        rf_offset: int,
        tod_start_sec: float | None = None,
    ) -> torch.Tensor | None:
        """Aggregate + normalize one wall-clock-aligned crop (cross_stock helper).

        Returns a (n_features, window // agg) float32 tensor, or None when
        aggregation fails. ``window`` must be a multiple of ``agg`` so every
        view in a group has identical length.
        """
        view = _aggregate_numpy_jittered(features[start_idx : start_idx + window], agg)
        if view is None:
            return None
        prior_vwap = self._prior_vwap_numpy(features, start_idx)
        self._ffill_vwap_numpy(view, prior_vwap)
        view = self._normalize_view(
            view, tod_start_sec=tod_start_sec, agg=agg)
        if self._risk_factors:
            view = self._merge_risk_factors(view, date_str, rf_offset, window, agg)
        if np.isnan(view).any():
            np.nan_to_num(view, copy=False, nan=0.0)
        return torch.from_numpy(view.T.astype(np.float32))

    def _normalize_view(self, view: np.ndarray, *,
                        tod_start_sec: float | None = None,
                        agg: float | None = None) -> np.ndarray:
        """Normalize one view, widening it for the information token.

        RETURNS THE ARRAY TO USE -- it is not always the one passed in. Every
        normalize call in this class goes through here so the channel count
        cannot disagree between the supervised crop loop, the two-view
        augmentations and the cross-stock helper; a mismatch there is a load
        failure at eval, three days after the run.
        """
        view, metadata = prepare_augmented_view(
            view,
            self._norm_groups,
            start_seconds=0.0 if tod_start_sec is None else tod_start_sec,
            aggregation_seconds=1.0 if agg is None else agg,
        )
        if self.info_norm_stats or self.info_window:
            # KEYWORD-ONLY AND UNCONDITIONALLY CHECKED. There are seven view
            # builders in this class and one more in the eval panel; a site
            # that forgot to pass these would emit a view the backbone strips
            # anyway, so the model would train on garbage in three of its info
            # dimensions and nothing would fail. Raise instead.
            if self.info_window and (tod_start_sec is None or agg is None):
                raise ValueError(
                    "dataset.info_window is on but this view builder passed no "
                    "tod_start_sec/agg; every caller of _normalize_view must "
                    "supply them or the info token is fed zeros silently.")
            view = append_view_info(
                view,
                encode_view_metadata(
                    metadata,
                    include_normalization=self.info_norm_stats,
                    include_window=self.info_window,
                ),
            )
        return view

    def _prior_vwap_numpy(self, features: np.ndarray, window_start_idx: int) -> float | None:
        """Delegates to training.utils.prior_vwap (shared with the eval path)."""
        from .utils import prior_vwap

        return prior_vwap(features, window_start_idx)
