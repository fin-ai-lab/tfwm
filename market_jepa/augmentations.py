"""
Domain-agnostic time series augmentations for JEPA.

Optimized for high throughput using NumPy arrays and batch processing.

This module holds the numpy kernels; the per-augmentation control flow that
calls them lives in ``StreamingMarketDataset._getitem_numpy``.
"""

import numpy as np
from stable_finance.dataset import ViewMetadata, prepare_view
from stable_finance.dataset.transforms import aggregate


_EPS = 1e-8

# A standard session, 09:30-16:00, in seconds. Used only to put the two
# wall-clock numbers of the information token on a unit scale; a short session
# simply reads past 1.0, which is information rather than a problem.
SESSION_SECONDS = 6.5 * 3600

# How many values ``encode_view_metadata`` contributes under ``include_window``:
# start, end, log(seconds per token).
N_WINDOW_INFO = 3

# Normalization groups ``build_norm_groups`` produces for the standard nine
# feature columns. ``encode_view_metadata`` contributes 2 per group under
# ``include_normalization`` -- a (mu, sigma) each -- so 8.
N_NORM_GROUPS = 4


def info_channel_width(*, info_norm_stats: bool, info_window: bool,
                       n_norm_groups: int = N_NORM_GROUPS) -> int:
    """How many information-token channels these two flags append.

    THE ONE DEFINITION OF THE WIDTH. This arithmetic was written out
    independently in StreamingMarketDataset, DayStoreCellDataset, xs_ic_eval
    and build_untrained_encoder, and the copies are what let the floor be
    constructed at a different width from the models it floors -- and, in the
    day-store case, let a backbone be told 0 while its view carried 11.

    It stays a function of the two FLAGS rather than a constant, so the
    ablations that turn one off get the right width for free.
    """
    return ((2 * int(n_norm_groups) if info_norm_stats else 0)
            + (N_WINDOW_INFO if info_window else 0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Valid ``AugmentationConfig.name`` values. Every augmentation is implemented
# inline in StreamingMarketDataset._getitem_numpy, so this is purely the set
# of names config parsing accepts. ``timestamp_jittering`` is an alias the
# dataset normalizes to ``fast_timestamp_jittering``.
AUGMENTATION_REGISTRY: frozenset[str] = frozenset({
    "timestamp_jittering",
    "fast_timestamp_jittering",
    "multi_scale",
    "contiguous",
    "random_resized_crop",
    "fixed_window",
    "cross_stock",
    "time_warp",
    "gaussian_noise",
    "volume_noise",
    "price_jitter",
    "channel_drop",
})


def parse_aggregation_range(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("aggregation range must contain exactly two values")
    low, high = int(value[0]), int(value[1])
    if not 1 <= low <= high:
        raise ValueError("aggregation range must satisfy 1 <= low <= high")
    return low, high


def canonicalize_augmentations(configs: list[dict]) -> list[dict]:
    """Validate and expand augmentation recipes into executable parameters.

    Dataset IO should select a recipe; the recipe's geometry and corruption
    defaults belong here beside the kernels that implement them.
    """
    output: list[dict] = []
    for original in configs:
        cfg = dict(original)
        name = cfg["name"]
        if name == "timestamp_jittering":
            name = "fast_timestamp_jittering"
        if name == "multi_scale":
            output.append({"name": name, "scale_factors": cfg.get("scale_factors", [1, 16]),
                           "window_size_sec": cfg.get("window_size_sec", 60)})
        elif name == "fast_timestamp_jittering":
            output.append({"name": name, "scale_factor": cfg.get("scale_factor", 5),
                           "window_size_sec": cfg.get("window_size_sec", 1200),
                           "max_jitter_fraction": cfg.get("max_jitter_fraction", 0.25)})
        elif name == "contiguous":
            v1 = cfg.get("view1_length_sec", 1200)
            v2 = cfg.get("view2_length_sec", 1200)
            output.append({"name": name, "view1_length_sec": v1,
                           "view2_length_sec": v2,
                           "view1_scale": cfg.get("view1_scale", 1),
                           "view2_scale": cfg.get("view2_scale", 1),
                           "window_size_sec": v1 + v2})
        elif name == "random_resized_crop":
            output.append({
                "name": name, "n_global_views": cfg.get("n_global_views", 2),
                "n_local_views": cfg.get("n_local_views", 6),
                "global_scale_range": tuple(cfg.get("global_scale_range", [0.5, 1.0])),
                "local_scale_range": tuple(cfg.get("local_scale_range", [0.05, 0.3])),
                "global_seq_len": cfg.get("global_seq_len", 2048),
                "local_seq_len": cfg.get("local_seq_len", 512),
                "global_agg_range": parse_aggregation_range(cfg.get("global_agg_range")),
                "local_agg_range": parse_aggregation_range(cfg.get("local_agg_range")),
                # Anchor snapping applies to GLOBAL views only: the label
                # anchor is the first global view's last row, and locals never
                # contribute a target.
                "end_grid_sec": cfg.get("end_grid_sec"),
                "end_min_slack_sec": max(0, int(cfg.get("end_min_slack_sec", 0) or 0)),
                # The start-anchored mirror, filled in by the dataset when
                # dataset.label_at_view_start is on.
                "start_grid_sec": cfg.get("start_grid_sec"),
                "start_min_slack_sec": max(0, int(cfg.get("start_min_slack_sec", 0) or 0)),
            })
        elif name == "fixed_window":
            output.append({"name": name, "window_size_sec": cfg.get("window_size_sec", 2048)})
        elif name == "cross_stock":
            n_stocks = int(cfg.get("n_stocks", 2))
            n_local = int(cfg.get("cross_stock_local_views", 0))
            if n_local % max(1, n_stocks) != 0:
                raise ValueError(
                    f"cross_stock n_local_views ({n_local}) must be divisible by "
                    f"n_stocks ({n_stocks})"
                )
            output.append({
                "name": name, "n_stocks": n_stocks,
                "global_seq_len": cfg.get("global_seq_len", 2048),
                "global_scale_range": tuple(cfg.get("global_scale_range", [0.5, 1.0])),
                "global_agg_range": parse_aggregation_range(cfg.get("global_agg_range")),
                "industry_table": cfg.get("industry_table"), "n_local_views": n_local,
                "local_seq_len": cfg.get("local_seq_len", 512),
                "local_scale_range": tuple(cfg.get("local_scale_range", [0.1, 0.5])),
                "structured_matching": bool(cfg.get("structured_matching", False)),
                "end_grid_sec": cfg.get("end_grid_sec"),
                "end_min_slack_sec": max(0, int(cfg.get("end_min_slack_sec", 0) or 0)),
                # Draw the anchor uniformly over the lattice band FIRST and the
                # resolution second, as the eval panel does; only meaningful
                # with end_grid_sec. See streaming_dataset._draw_anchor_first.
                "anchor_uniform": bool(cfg.get("anchor_uniform", False)),
            })
        elif name in {"time_warp", "gaussian_noise", "volume_noise", "price_jitter", "channel_drop"}:
            drop_p = float(cfg.get("channel_drop_p", 0.2))
            if not 0.0 <= drop_p < 1.0:
                raise ValueError(f"channel_drop_p must be in [0, 1), got {drop_p}")
            output.append({
                "name": name, "n_global_views": int(cfg.get("n_global_views", 2)),
                "global_seq_len": cfg.get("global_seq_len", 2048),
                "global_scale_range": tuple(cfg.get("global_scale_range", [0.5, 1.0])),
                "global_agg_range": parse_aggregation_range(cfg.get("global_agg_range")),
                "warp_knots": int(cfg.get("warp_knots", 8)),
                "warp_strength": float(cfg.get("warp_strength", 0.25)),
                "noise_sigma": float(cfg.get("noise_sigma", 0.1)),
                "vol_noise_frac": float(cfg.get("vol_noise_frac", 0.5)),
                "price_jitter_frac": float(cfg.get("price_jitter_frac", 1.0)),
                "channel_drop_p": drop_p,
            })
        else:
            output.append(cfg)
    return output


def prepare_augmented_view(
    view: np.ndarray,
    norm_groups: list[tuple[list[int], bool]],
    *,
    start_seconds: float,
    aggregation_seconds: float,
) -> tuple[np.ndarray, ViewMetadata]:
    """Compatibility wrapper around stable-finance's view preparation."""
    return prepare_view(
        view, norm_groups,
        start_seconds=start_seconds,
        aggregation_seconds=aggregation_seconds,
    )


def encode_view_metadata(
    metadata: ViewMetadata,
    *,
    include_normalization: bool,
    include_window: bool,
) -> np.ndarray:
    """Encode stable-finance metadata for market-jepa's information token."""
    values: list[float] = []
    if include_normalization:
        for mean, scale in zip(
            metadata.normalization_means, metadata.normalization_scales
        ):
            values.extend((
                float(np.sign(mean) * np.log1p(abs(mean))),
                float(np.log(scale + _EPS)),
            ))
    if include_window:
        values.extend((
            metadata.start_seconds / SESSION_SECONDS,
            metadata.end_seconds / SESSION_SECONDS,
            float(np.log(max(metadata.aggregation_seconds, _EPS))),
        ))
    return np.asarray(values, dtype=np.float64)


# ── VIEW ANCHORING, AND WHERE IT IS SUPPOSED TO LIVE ────────────────────────
#
# The four *_grid_sec / *_min_slack arguments below are view GEOMETRY, and
# stable-finance's README says geometry is configuration rather than a hidden
# property of the dataset. ``ViewSpec`` there already declares half of this
# vocabulary:
#
#     ViewSpec.end_grid_seconds     <-> end_grid_sec
#     ViewSpec.min_future_seconds   <-> end_min_slack_sec
#
# NEITHER FIELD IS READ ANYWHERE IN STABLE-FINANCE. They are declared and
# validated and nothing consumes them, because ``build_session_panel`` solves
# the opposite problem: it is HANDED anchors and materialises the view ending
# at each one (``start = row + 1 - window``), so it never has to snap or
# reject. This function is the training path and runs it backwards -- it
# SAMPLES a window and then has to make that random window land on a legal
# anchor. There is no upstream function to call.
#
# So the two sides share a concept, and have already drifted three ways.
# Anyone touching this should know all three:
#
#   NAMES     `_seconds` upstream, `_sec` here, and the start-anchored pair
#             below has no upstream counterpart at all.
#   OFF-BY-ONE The end path is one row conservative (`t <= N - 1 - slack`);
#             the start path is exact (`start <= N - slack`), because at the
#             day horizon the only feasible start is 0 and one row of slop
#             rejects every draw. Whichever convention stable-finance picks,
#             one of these two paths will disagree with it.
#   UNITS     Named in seconds, spent as ROWS -- `N` is len(features). At 1 Hz
#             the units coincide; stable-finance now records bar_seconds and
#             ships resample_session(session, 60), so they need not.
#             StreamingMarketDataset raises on any non-1 Hz session.
#
# The SAMPLER belongs here -- it is an augmentation, which the package
# boundary assigns to market-jepa. Only the POLICY belongs upstream, and it
# wants one field rather than two mutually-exclusive pairs:
# ``label_anchor: "start" | "end"`` plus a grid and a required-remainder, so
# the exclusion this function enforces at runtime becomes structural.
def _random_resized_crop_numpy(
    features, scale_range, target_seq_len, rng, agg_range=None,
    end_grid_sec=None, end_offset=0, end_min_slack=0,
    start_grid_sec=None, start_min_slack=0,
):
    """Crop a random fraction of the observation and aggregate to target_seq_len.

    Analogous to image RandomResizedCrop: pick a random scale (fraction of
    the observation), then aggregate to a fixed output resolution.

    Args:
        features: (N, 9) float64 array of 1 Hz feature rows.
        scale_range: (lo, hi) tuple — fraction of N to crop.
        target_seq_len: Desired output length (number of aggregated rows).
        rng: numpy RandomState for reproducibility.
        agg_range: Optional (lo, hi) inclusive integer range of seconds per
            output token.  When given, resolution is sampled directly and
            *scale_range* is ignored.  The fractional path ties resolution to
            N, which is fine when N is effectively constant (a 23,400 s
            regular-hours session) but not when it varies — an extended-hours
            session runs 37.8k–57.6k rows depending on how early the ticker
            starts quoting, and a fixed fraction of that spans a 1.5x range of
            resolutions.  Pinning the range in seconds keeps every view inside
            the band the position embeddings were trained on.
        end_grid_sec: When set, the crop's LAST row — the label anchor
            ``t_idx`` — is drawn uniformly from wall-clock multiples of this
            many seconds, instead of the start being uniform. The
            cross-sectional (mu, sigma) tables exist only on that grid, so a
            view that ends off it has no z-score at all.
        end_offset: Index-to-wall-clock alignment for *end_grid_sec*: the crop
            end satisfies ``(end_offset + t_idx) % end_grid_sec == 0``. Pass
            the view's time-of-day offset relative to the standard 09:30 open,
            since a ticker that starts quoting late has a trimmed grid whose
            row 0 is not the open.
        end_min_slack: Require at least this many rows after the crop end
            (``t_idx <= N - 1 - end_min_slack``), which is how a forward window
            of that length is guaranteed to exist rather than being dropped as
            NaN downstream. A draw whose window cannot fit before that deadline
            is REJECTED, not shrunk: at h = 7200 a 2048-token view at 11 s/token
            spans 96% of the session and simply cannot end two hours early, so
            clamping its resolution down would pile probability mass onto the
            coarsest feasible scale. Rejection instead reproduces exactly the
            distribution an unconstrained draw would have after the downstream
            NaN filter — same samples, without paying for the ~90% (at h = 7200)
            that the filter would discard.
        start_grid_sec: The mirror of *end_grid_sec* for a view whose label
            instant is its FIRST row rather than its last. The crop start is
            drawn uniformly from wall-clock multiples of this many seconds
            (same ``end_offset`` alignment), and the window is required to fit
            entirely after it. Used by the contemporaneous day-return decoder,
            whose target is measured forward from the open over the whole
            session: the view IS the day, so its anchor is where it begins.
            Mutually exclusive with *end_grid_sec*.
        start_min_slack: Require at least this many rows after the crop START
            (``start <= N - start_min_slack``), the start-anchored analogue of
            *end_min_slack*: a horizon measured from the start needs ``h + w``
            rows of session after it or its target is NaN. The bound is exact
            rather than one-conservative — see the comment at its use.

    Returns:
        (view, start_idx, agg_factor, adjusted_window) tuple, or
        (None, None, None, None) on failure.
        - view: (target_seq_len, 9) float64 array
        - start_idx: int, index into features where the crop begins
        - agg_factor: int, aggregation scale used
        - adjusted_window: int, number of 1Hz rows in the crop
    """
    N = len(features)
    if agg_range is not None:
        agg_factor = int(rng.randint(int(agg_range[0]), int(agg_range[1]) + 1))
    else:
        scale = rng.uniform(scale_range[0], scale_range[1])
        window_size = max(1, round(scale * N))
        agg_factor = max(1, round(window_size / target_seq_len))
    adjusted_window = agg_factor * target_seq_len

    if adjusted_window > N:
        agg_factor = max(1, N // target_seq_len)
        adjusted_window = agg_factor * target_seq_len

    if end_grid_sec and start_grid_sec:
        raise ValueError(
            "end_grid_sec and start_grid_sec are mutually exclusive — a view "
            "has one label anchor, and snapping both ends would over-determine "
            "a window whose length is already fixed by the scale draw."
        )

    if start_grid_sec:
        # Sample the START on the anchor lattice: the label instant is the
        # view's first row. s_hi is the tighter of "the window still fits"
        # and "the horizon still fits", so a draw is rejected here rather
        # than returning a target the downstream NaN filter would drop.
        # N - start_min_slack, NOT N - 1 - start_min_slack. anchor_targets
        # accepts t exactly when t + h + w <= N, so with start_min_slack =
        # h + w the boundary start is feasible. end_min_slack keeps the extra
        # -1 (it has always been one row conservative, and every horizon it is
        # used at leaves hundreds of rows of headroom) — here the day horizon
        # spans the WHOLE session, the only feasible start is 0, and one row
        # of slop rejects every draw.
        s_hi = min(N - adjusted_window, N - start_min_slack)
        if s_hi < 0 or adjusted_window < 2:
            return None, None, None, None
        k_lo = -(-end_offset // start_grid_sec)          # ceil-div: start >= 0
        k_hi = (s_hi + end_offset) // start_grid_sec
        if k_hi < k_lo:
            return None, None, None, None
        k = int(rng.randint(k_lo, k_hi + 1))
        start = k * start_grid_sec - end_offset
        window = features[start : start + adjusted_window]
        result = _aggregate_numpy_jittered(window, agg_factor, offset=0)
        return result, start, agg_factor, adjusted_window

    if end_grid_sec:
        # Sample the END on the anchor lattice rather than the start.
        t_hi = N - 1 - end_min_slack
        if adjusted_window > t_hi + 1:
            return None, None, None, None
        t_lo = adjusted_window - 1
        # k indexes the lattice in wall-clock space; ceil-div for the lower
        # bound so the window never runs off the front of the grid.
        k_lo = -(-(t_lo + end_offset) // end_grid_sec)
        k_hi = (t_hi + end_offset) // end_grid_sec
        if k_hi < k_lo:
            return None, None, None, None
        k = int(rng.randint(k_lo, k_hi + 1))
        start = (k * end_grid_sec - end_offset) + 1 - adjusted_window
        window = features[start : start + adjusted_window]
        result = _aggregate_numpy_jittered(window, agg_factor, offset=0)
        return result, start, agg_factor, adjusted_window

    if adjusted_window > N or adjusted_window < 2:
        return None, None, None, None

    max_start = N - adjusted_window
    start = rng.randint(0, max_start + 1) if max_start > 0 else 0
    window = features[start : start + adjusted_window]
    result = _aggregate_numpy_jittered(window, agg_factor, offset=0)
    return result, start, agg_factor, adjusted_window


def _aggregate_numpy_jittered(features, scale_factor, offset=0):
    """Aggregate (N, 9) numpy array at given scale via reshape, starting from offset.

    For 1 Hz data with integer scale_factor, skipping `offset` rows then reshaping
    is equivalent to shifting aggregation bucket boundaries by `offset` seconds.

    Column layout (hardcoded, matches FEATURE_COLUMNS in streaming_dataset.py):
        0: bid_price (last), 1: vwap_all (volume-weighted), 2: high (max),
        3: low (min), 4: ask_price (last), 5: bid_size (last),
        6: ask_size (last), 7: volume (sum), 8: n (sum)

    Args:
        features: (N, 9) float64 array of 1 Hz feature rows.
        scale_factor: Number of 1 Hz rows per output bucket.
        offset: Number of leading rows to skip before bucketing.

    Returns:
        (n_buckets, 9) float64 array, or None if fewer than 2 buckets.
    """
    return aggregate(features, scale_factor, offset=offset)


def _warped_aggregate_numpy(window_feats, seq_len, rng, n_knots, strength):
    """Aggregate a 1 Hz window onto a randomly time-warped bucket grid.

    The time_warp augmentation: instead of ``seq_len`` equal-width buckets,
    bucket boundaries follow a smooth random monotone reparameterization of
    the window — a piecewise-linear map through ``n_knots + 1`` knots whose
    interior positions are jittered by up to ``strength`` x the inter-knot
    spacing. Regions where the map runs steep are compressed (sped up);
    where it runs shallow, stretched (slowed down). Per-bucket semantics
    match _aggregate_numpy_jittered (last/max/min/sum, volume-weighted
    vwap), so totals of volume and n over the window are preserved.

    Args:
        window_feats: (W, 9) float64 array, the full 1 Hz window.
        seq_len: Number of output buckets. Requires W >= seq_len.
        rng: numpy RandomState.
        n_knots: Number of piecewise-linear segments (>= 2 for any warp).
        strength: Max knot displacement as a fraction of the knot spacing.
            0 reproduces uniform aggregation exactly; values >= 0.5 are
            legal (monotonicity is enforced) but increasingly fold-prone.

    Returns:
        (seq_len, 9) float64 array, or None when W < seq_len or seq_len < 2.
    """
    f = window_feats
    W = len(f)
    if W < seq_len or seq_len < 2:
        return None

    knots_x = np.linspace(0.0, W, n_knots + 1)
    knots_y = knots_x.copy()
    if n_knots >= 2:
        knots_y[1:-1] += rng.uniform(-strength, strength, size=n_knots - 1) * (W / n_knots)
        np.maximum.accumulate(knots_y, out=knots_y)
    b = np.round(np.interp(np.linspace(0.0, W, seq_len + 1), knots_x, knots_y)).astype(np.int64)
    # Strictly increasing boundaries with >= 1 row per bucket: clip each
    # boundary into its feasible band, then a running max keeps unit gaps.
    idx = np.arange(seq_len + 1)
    b = np.clip(b, idx, W - seq_len + idx)
    b = np.maximum.accumulate(b - idx) + idx

    starts = b[:-1]
    last = b[1:] - 1
    out = np.empty((seq_len, 9), dtype=np.float64)
    out[:, 0] = f[last, 0]  # bid_price: last
    out[:, 2] = np.maximum.reduceat(f[:, 2], starts)  # high: max
    out[:, 3] = np.minimum.reduceat(f[:, 3], starts)  # low: min
    out[:, 4] = f[last, 4]  # ask_price: last
    out[:, 5] = f[last, 5]  # bid_size: last
    out[:, 6] = f[last, 6]  # ask_size: last
    out[:, 7] = np.add.reduceat(f[:, 7], starts, dtype=np.float64)  # volume: sum
    out[:, 8] = np.add.reduceat(f[:, 8], starts, dtype=np.float64)  # n: sum
    # vwap: NaN rides on zero-volume rows, so zeroing the products matches
    # the np.nansum semantics of the uniform aggregator.
    prod = np.multiply(f[:, 1], f[:, 7], dtype=np.float64)
    np.nan_to_num(prod, copy=False, nan=0.0)
    vol_sum = out[:, 7]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 1] = np.where(
            vol_sum > 0, np.add.reduceat(prod, starts, dtype=np.float64) / vol_sum, np.nan
        )
    return out


def _add_volume_noise_numpy(view, rng, frac):
    """Add spurious trading activity to an aggregated view, in place.

    The volume_noise augmentation: iid Exponential additions to the volume
    and n columns with mean ``frac`` x the view's own mean bucket value, so
    the corruption is scale-free across tickers. The fake trades execute at
    the prevailing price — vwap and every other channel stay untouched.
    Raw (pre-normalization) units.
    """
    T = len(view)
    mean_vol = float(np.nanmean(view[:, 7]))
    if mean_vol > 0:
        view[:, 7] += rng.exponential(frac * mean_vol, size=T)
    mean_n = float(np.nanmean(view[:, 8]))
    if mean_n > 0:
        view[:, 8] += rng.exponential(frac * mean_n, size=T)


def _price_jitter_numpy(view, rng, frac):
    """Jitter the price ladder without ever crossing the book, in place.

    The price_jitter augmentation: per-bucket level noise eta_t ~
    N(0, (frac x half-spread_t)^2) added to ALL five price columns
    (bid, vwap, high, low, ask) of that bucket. Moving the whole ladder
    together preserves the spread exactly — bid' <= ask' by construction,
    high >= low survives, and a locked book (zero spread) gets zero
    jitter. Raw (pre-normalization) units; NaN vwap rides through for the
    downstream forward-fill.
    """
    half = (view[:, 4] - view[:, 0]) / 2.0
    np.clip(half, 0.0, None, out=half)
    eta = rng.standard_normal(len(view)) * (frac * half)
    view[:, :5] += eta[:, None]


def _draw_channel_mask(rng, n_channels, drop_p):
    """Bernoulli(drop_p) per-channel drop mask with >= 1 channel kept.

    Returns the integer indices of the channels to zero. Resamples the
    all-dropped outcome so a view never goes fully blank — which requires
    ``drop_p < 1``, since ``drop_p == 1`` makes every draw all-dropped and
    the resample loop never terminates. The dataset's config parser enforces
    the bound (see StreamingMarketDataset's channel_drop_p validation).
    """
    while True:
        drop = rng.uniform(size=n_channels) < drop_p
        if not drop.all():
            return np.flatnonzero(drop)
