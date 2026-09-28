"""Per-month predictor tables built on the SYNCHRONIZED ANCHOR PANEL.

This replaces the retired ``raw_tables.py``, and the difference is not
cosmetic. The old tables sampled scattered ``(ticker, day, t)`` points off the
raw 1 Hz book and were scored by a logistic probe's macro-OvR AUC. The IC-era
protocol scores a **cross-section**: at one ``(date, anchor)`` cell every
ticker is measured at the same instant, the target is that cell's z-score, and
the metric is a within-cell rank correlation averaged over cells. A table whose
rows are not cells cannot be scored that way at all.

So the rows here ARE the encoder's rows. They come out of
``xs_ic_eval.iter_panel`` — the single generator every learned model is fed
from — which means a baseline and an encoder are graded on the same tickers, at
the same instants, against the same z-scored labels. Nothing is resampled and
nothing has to be argued about.

THE PREDICTORS ARE READ OFF THE VIEW, NOT THE BOOK.
Chosen deliberately (2026-08-21). ``iter_panel`` hands out the normalized
``(2048, 9)`` tensor the encoder actually receives: aggregated to ``agg``
seconds per token, vwap forward-filled, then ``normalize_numpy`` standardizing
the price group by EACH VIEW'S OWN mean and std. Every number below is a
function of that tensor and of nothing else, so a classical model here has
exactly the encoder's information — no more, no less. Three consequences, all
intended:

  * Absolute levels ARE AVAILABLE AGAIN, as of 2026-08-25, because the encoder
    now gets them: the per-view (mu, sigma) that standardization divides out
    ride to it in an information token, so withholding them here would make the
    classical arms read 9 channels against the encoder's 20. They are the
    ``norm_mu_g*`` / ``norm_sigma_g*`` predictors and they are PER STOCK, so
    they vary within a cell and can rank. Before this, two stocks with
    identically shaped views but different dollar spreads produced identical
    predictors: ``-spread(t)`` reached rank IC ~0.9 on the raw book and ~0.6 on
    the view, and that gap is what these restore. The three window descriptors
    (``view_start_frac``, ``view_end_frac``, ``log_agg``) come with them.
  * Realized vol is measured at ``agg`` seconds (6-11 s), not 1 s. Coarser,
    and coarser in exactly the way the encoder's input is coarser.
  * Returns and spread changes are DIFFERENCES, not ratios. A normalized mid
    is not positive, so ``mid[b]/mid[a] - 1`` is meaningless here. Within a
    cell a difference is the return times a per-stock constant; that constant
    is the thing the normalization already destroyed, so nothing further is
    lost by the choice.

``dspr_prevday_h{h}`` is GONE with the prior-day baseline that read it.
Yesterday's spread at the same clock time is not inside a 2048-token window,
so under a view-only rule it is not available to a baseline any more than it is
to the encoder.

WHAT IS FITTED, AND ON WHAT. ``Z`` is the cross-sectional z-score — the very
quantity the reported IC is computed against — and it is what the models
regress on. Fitting on the raw target instead would mix per-stock predictor
units with dollar-unit labels in one pooled OLS; fitting on Z keeps both sides
dimensionless. The fit month uses 36 anchors/day and the eval month 8, matching
``xs_ic_series.FIT_ANCHORS`` / ``EVAL_ANCHORS`` exactly.

Usage::

    uv run plots/finance_baselines/panel_tables.py --months 2013-01 --anchors 8
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO), str(_REPO / "scripts" / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stable_finance.dataset import MarketSchedule, next_month  # noqa: E402
from market_jepa.eval.tasks import HORIZONS, TARGET_TYPES  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import cell_agg, day_anchors, iter_panel  # noqa: E402

# Column positions inside the stable-finance market schema.
BID, ASK = 0, 4
N_FEATURES = 9

# HAR-RV components: short / medium / long trailing realized-vol windows.
HAR_WINDOWS: tuple[int, ...] = (300, 1800, 7200)

# GARCH(1,1)'s conditional variance IS an EWMA of squared returns with decay
# beta, so precomputing the EWMA at a spread of decays lets the GARCH baseline
# be fitted from the table. These are PER-SECOND decays (half-lives ~69 s to
# ~19 h); on an agg-second grid the filter runs at ``beta ** agg`` per token so
# the time constant is preserved and GARCH11._fwd_vol, which takes h in
# seconds, stays consistent with the column it reads.
EWMA_DECAYS: tuple[float, ...] = (0.99, 0.997, 0.999, 0.9997, 0.9999)

# Tokens at the START of the view used to seed the EWMA recursion. Causal by
# construction: the view begins strictly before its own anchor.
EWMA_SEED_TOKENS: int = 8

# Lags kept per horizon. 5 supports AR(p<=3) plus the constructed lagged
# residuals an ARMA(p,q<=2) needs — a Hannan-Rissanen residual at lag k is
# built from lags k+1, k+2, so each MA term costs two extra lags.
N_LAGS: int = 5

# Fixed lags (seconds) at which the full 9-channel vector is stored, for the
# Ridge ARDL distributed lag.
FEATURE_LAGS: tuple[int, ...] = (0, 300, 900, 3600)

# 2 stats x 4 normalization groups, then the three window descriptors, in the
# order training.utils puts them: info_norm_stats then info_window.
INFO_NAMES: list[str] = (
    [f"norm_{s}_g{g}" for g in range(4) for s in ("mu", "sigma")]
    + ["view_start_frac", "view_end_frac", "log_agg"]
)
N_INFO: int = len(INFO_NAMES)


# The learners in ``view_models`` regress straight off the input tensor rather
# than off hand-built predictors: the WHOLE 2048x9 view, no crop and no pool,
# plus the information token's eleven ONCE. At 506k rows a fit month's view
# tensor is 37 GB in float32, so it is never cached and never fully
# materialized: see ``iter_view_blocks``.
#
# NOT 2048 x 20. The eleven are constant along time, so streaming them per step
# would be 22,517 exactly-collinear duplicate columns, a 2.2x Gram, and an
# effective ridge penalty on those directions 2048x weaker than on the rest.
# The encoder does not read them 2048 times either -- they are one token.
VIEW_FEATURES: int = 2048 * N_FEATURES + N_INFO

# Tokens the view learners actually keep, counted back from the anchor. None
# is the whole 2048.
#
# WHY A TAIL IS A DEFENSIBLE VIEW AND A CROP IN GENERAL IS NOT. The reductions
# retired on 2026-08-22 were summaries -- a pooled or subsampled view, which
# answers "what can a learner do with a digest of the input" rather than the
# question asked. A tail is not a digest: it is the input at FULL resolution
# over the window the readout sits in. The encoder pools at ``last``
# (xs_ic_eval.PREDICT_POOL), so its readout token IS the final patch of
# ``patch_size`` = 8 steps, and a tail of 8/24/64 is that patch and its
# immediate neighbourhood.
#
# WHAT IT IS NOT. The encoder's last token ATTENDS over all 2048 steps, so a
# tail learner is not the encoder's information set -- it is a subset of it.
# That makes these lines a LOWER BOUND on a full-view learner: beating the
# encoder with a tail is a strictly stronger result, losing to it with one
# proves nothing about the full view. Say "tail" in the label, never "view".
VIEW_TAIL_TOKENS: int | None = 24


def view_features(tail_tokens: int | None = None) -> int:
    """Width of a block from :func:`iter_view_blocks` at this crop."""
    steps = 2048 if tail_tokens is None else min(int(tail_tokens), 2048)
    return steps * N_FEATURES + N_INFO

# 2 (2026-08-25): X gained the eleven information-token predictors, so a v1
# table is 135 columns wide where this code builds 146. The filename carries
# the version so a stale table is rebuilt rather than served; the guard in
# build_month_table is the backstop for a table that somehow has the wrong
# names at the right version.
PANEL_VERSION: int = 2

TABLE_DIR = Path(
    "/data/lab/market-jepa-checkpoints/finance_baselines/panel_tables"
)

TARGET_NAMES: list[str] = [f"{t}_{h}" for t in TARGET_TYPES for h in HORIZONS]


# ---------------------------------------------------------------------------
# Predictor layout
# ---------------------------------------------------------------------------


def predictor_names() -> list[str]:
    """Column order of the predictor matrix. Stable — the tables store it."""
    names: list[str] = []
    for h in HORIZONS:
        for lag in range(1, N_LAGS + 1):
            names.append(f"ret_h{h}_lag{lag}")
        for lag in range(1, N_LAGS + 1):
            names.append(f"dspr_h{h}_lag{lag}")
        for lag in range(1, N_LAGS + 1):
            names.append(f"rv_h{h}_lag{lag}")
    for w in HAR_WINDOWS:
        names.append(f"har_rv_{w}")
    for b in EWMA_DECAYS:
        names.append(f"ewma_var_{b}")
    names.append("spread_level")
    for lag in FEATURE_LAGS:
        names += [f"feat{i}_lag{lag}" for i in range(N_FEATURES)]
    names += INFO_NAMES
    return names


PREDICTOR_NAMES: list[str] = predictor_names()

# The eleven numbers the encoder gets through its INFORMATION TOKEN and the
# baselines did not. Withholding them was not a modelling choice, it was an
# oversight: panel_tables called iter_panel with neither half of the info
# token, so the classical arms read 9 channels while the encoder read 20.
#
# THE EIGHT NORM STATS ARE THE SUBSTANTIVE HALF. They are the per-view
# (mu, sigma) that standardization divides out -- price level, spread width,
# activity level -- and they are PER STOCK, so they vary within a cell and can
# genuinely rank. This is the header's "absolute levels are gone" caveat being
# retired: on the raw book -spread(t) reaches rank IC ~0.9 and on the view ~0.6,
# and the gap is exactly what these restore.
#
# The three window descriptors are constant within a cell (cell_agg is seeded
# from (date, anchor) alone, so every stock at a cell shares resolution, start
# and end). A constant cannot rank stocks directly, but it is NOT inert: the fit
# is pooled across cells, so including it changes the estimated coefficients on
# everything else, and log(agg) in particular is the UNIT OF THE LAG AXIS --
# lag 1 is 6 s in one cell and 11 s in another, and no model could tell.


# ---------------------------------------------------------------------------
# Featurization — vectorized over a batch of views sharing one resolution
# ---------------------------------------------------------------------------


def _split_info(views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(B, T, 9+N_INFO)`` -> the time series and the per-window constants.

    The trailing columns are constant along time by construction (see
    training.utils.append_view_info), so the last row carries them exactly --
    the same thing TransformerBackbone's information token reads. Splitting
    them off here is what keeps them OUT of the lag/vol/spread featurizers,
    where a constant column would produce a zero difference and a zero variance
    and quietly waste a regressor.
    """
    if views.shape[2] <= N_FEATURES:
        # An old panel with no info columns: report NaN so usable_cols drops
        # them rather than a zero that reads as a real measurement.
        return views, np.full((len(views), N_INFO), np.nan, dtype=np.float64)
    return views[:, :, :N_FEATURES], views[:, -1, N_FEATURES:].astype(np.float64)


def view_predictors(views: np.ndarray, agg: int) -> np.ndarray:
    """``(B, T, C)`` views at ONE resolution -> ``(B, len(PREDICTOR_NAMES))``.

    Every window is expressed in tokens (``round(seconds / agg)``) and anchored
    at the view's LAST row, which is the decision instant ``t``. A window that
    would reach before the view's first row yields NaN rather than a truncated
    statistic: the fit then drops that row for that column instead of being
    handed a shorter, differently-scaled measurement.
    """
    views, info = _split_info(views)
    B, T, _ = views.shape
    t = T - 1
    v = views.astype(np.float64)
    mid = (v[:, :, BID] + v[:, :, ASK]) * 0.5            # (B, T)
    spread = v[:, :, ASK] - v[:, :, BID]                 # (B, T)
    d = np.diff(mid, axis=1)                             # (B, T-1)

    # Prefix sums so any windowed std is O(1). d has T-1 entries; entry i
    # covers [i, i+1), so the std over mid-rows [a, b) reads d[a:b-1].
    z = np.zeros((B, 1))
    c1 = np.concatenate([z, np.cumsum(d, axis=1)], axis=1)          # (B, T)
    c2 = np.concatenate([z, np.cumsum(d * d, axis=1)], axis=1)      # (B, T)

    def tok(sec: int) -> int:
        return int(round(sec / agg))

    def rv(a: int, b: int) -> np.ndarray:
        """Population std of ``d`` over mid-rows ``[a, b)``; NaN if degenerate."""
        if a < 0 or b - a < 2:
            return np.full(B, np.nan)
        n = float(b - a)
        s1 = c1[:, b] - c1[:, a]
        s2 = c2[:, b] - c2[:, a]
        with np.errstate(invalid="ignore"):
            var = s2 / n - (s1 / n) ** 2
        return np.sqrt(np.maximum(var, 0.0))

    out = np.empty((B, len(PREDICTOR_NAMES)), dtype=np.float64)
    nan = np.full(B, np.nan)
    i = 0
    for h in HORIZONS:
        n = tok(h)
        # Lagged h-returns as DIFFERENCES of the normalized mid.
        for lag in range(1, N_LAGS + 1):
            a, b = t - lag * n, t - (lag - 1) * n
            out[:, i] = nan if (a < 0 or a >= b) else mid[:, b] - mid[:, a]
            i += 1
        # Lagged h-spread-changes.
        for lag in range(1, N_LAGS + 1):
            a, b = t - lag * n, t - (lag - 1) * n
            out[:, i] = nan if (a < 0 or a >= b) else spread[:, b] - spread[:, a]
            i += 1
        # Trailing realized vol, N_LAGS windows back. lag 1 is the target's own
        # backward leg, so it must equal mechanical_baseline's np.std(d[-n:]).
        for lag in range(1, N_LAGS + 1):
            out[:, i] = rv(t - lag * n, t - (lag - 1) * n)
            i += 1
    for w in HAR_WINDOWS:
        out[:, i] = rv(t - tok(w), t)
        i += 1

    # EWMA variance of the differenced mid, one column per per-second decay.
    # v_k = b_tok * v_{k-1} + (1 - b_tok) * d_{k-1}^2, seeded from the view's
    # own opening tokens. scipy's lfilter runs the recursion in C.
    from scipy.signal import lfilter
    d2 = d * d
    seed = np.nanmean(d2[:, :EWMA_SEED_TOKENS], axis=1)
    seed = np.where(np.isfinite(seed), seed, 0.0)
    u = np.concatenate([z, d2], axis=1)[:, :T]           # u[:, k] = d2[:, k-1]
    for b in EWMA_DECAYS:
        bt = b ** agg
        st = lfilter([1.0 - bt], [1.0, -bt], u, axis=1, zi=seed[:, None])[0]
        out[:, i] = st[:, t]
        i += 1

    out[:, i] = spread[:, t]
    i += 1
    for lag in FEATURE_LAGS:
        r = t - tok(lag)
        out[:, i:i + N_FEATURES] = v[:, r, :N_FEATURES] if r >= 0 else np.nan
        i += N_FEATURES

    # The information token's eleven, verbatim -- no transform, because the
    # encoder gets them untransformed too.
    out[:, i:i + N_INFO] = info
    i += N_INFO
    assert i == len(PREDICTOR_NAMES), (i, len(PREDICTOR_NAMES))
    return out




def iter_view_blocks(
    ym: str,
    *,
    anchors_per_day: int,
    mosaic_dir: str | Path | None = None,
    xs_anchor_stats_dir: str | Path = "/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60",
    batch_size: int = 512,
    shard_stride: int = 1,
    tail_tokens: int | None = None,
):
    """Re-stream ``ym``'s views as ``(B, view_features(tail_tokens))`` blocks.

    That is the 9 channels flattened over 2048 steps, then the information
    token's eleven appended once -- 18,443 columns, not 40,960.

    IN THE SAME ROW ORDER AS :func:`build_month_table`. That is the entire
    contract: the caller holds the month's ``Y`` from the cached table and
    matches it to these blocks by a running offset, so the grouping below must
    stay a copy of the one in ``build_month_table`` rather than merely a
    similar loop. Both walk ``iter_panel`` with the same batch size and shard
    count, split each batch by ``cell_agg`` in first-appearance order, and emit
    groups in that order.

    Nothing is cached and nothing is concatenated. A 36-anchor fit month is
    506k rows, which is 37 GB of float32 view tensor; the full-view learners
    consume it as a stream and keep only a 2.7 GB Gram.

    float32 is the SOURCE dtype -- ``_panel_for_ticker_day`` casts every view
    to float32 before it leaves the panel -- so emitting float32 here loses
    nothing. What must not happen is a float32 *solve*; the ridge accumulates
    its Gram in float64 for exactly that reason.
    """
    mosaic_dir = Path(mosaic_dir or BLL01MachineConfig().mosaic_dir)
    sched = MarketSchedule(BLL01MachineConfig().holiday_csv)
    stats = AnchorStats(Path(xs_anchor_stats_dir) / f"{ym}.npz")
    anchors = day_anchors(anchors_per_day)
    mdir = mosaic_dir / ym[:4] / ym[5:]

    # THE INFORMATION TOKEN ON: without it the baselines read 9 channels while
    # the encoder reads 20, which is not a modelling choice but an oversight.
    for views, metas in iter_panel(mdir, stats, sched, anchors, batch_size,
                                   num_shards=max(shard_stride, 1),
                                   info_norm_stats=True, info_window=True):
        by_agg: dict[int, list[int]] = {}
        for j, (_target, date, anchor, _tk, _raw, _quote) in enumerate(metas):
            a = cell_agg(date, int(anchor), int(anchor))
            if a is None:
                continue
            by_agg.setdefault(int(a), []).append(j)
        for _a, idx in by_agg.items():
            sel = np.asarray(idx)
            # Same split as the table's: the 9 channels flattened over time,
            # then the eleven appended once. See VIEW_FEATURES.
            v, info = _split_info(views[sel])
            # The tail is taken HERE rather than by slicing the assembled
            # block, so the 2048-step flatten never happens: at 506k rows the
            # full month is 37 GB and a 24-token tail is 440 MB. The crop is
            # the LAST tokens because the anchor is the view's last row -- see
            # VIEW_TAIL_TOKENS.
            if tail_tokens is not None:
                v = v[:, -int(tail_tokens):, :]
            block = np.concatenate(
                [v.reshape(len(sel), -1), info.astype(np.float32)], axis=1)
            yield block.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Month table
# ---------------------------------------------------------------------------


def table_path(ym: str, anchors_per_day: int,
               cache_dir: Path = TABLE_DIR) -> Path:
    """Where this month's predictor table is written."""
    return Path(cache_dir) / f"{ym}_a{anchors_per_day}_x_v{PANEL_VERSION}.npz"


def existing_table(ym: str, anchors_per_day: int,
                   cache_dir: Path = TABLE_DIR) -> Path | None:
    """An already-built table for ``ym``, whichever generation wrote it.

    The 2026-08-21 pass ran with ``--with-views`` and wrote ``_views_`` files
    carrying the crop/pool reductions alongside ``X``. Those reductions are
    gone (``view_models`` reads the whole view now), but the ``X`` inside those
    65 GB of files is bit-identical to what ``_x_`` would hold, and rebuilding
    32 months to rename them would be a day of decode for nothing. So a
    ``_views_`` file is still a valid source; the extra arrays are dropped on
    load and the file is only rewritten if it is rebuilt for another reason.
    """
    for kind in ("x", "views"):
        q = Path(cache_dir) / f"{ym}_a{anchors_per_day}_{kind}_v{PANEL_VERSION}.npz"
        if q.is_file():
            return q
    return None


def build_month_table(
    ym: str,
    *,
    anchors_per_day: int,
    mosaic_dir: str | Path | None = None,
    xs_anchor_stats_dir: str | Path = "/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60",
    batch_size: int = 512,
    cache_dir: Path = TABLE_DIR,
    force: bool = False,
    shard_stride: int = 1,
    verbose: bool = True,
) -> dict:
    """Build (or load) the panel table for ``ym``.

    The table holds the hand-built predictors only. The view tensors the
    ``view_models`` learners read are NOT stored — at 18,432 float32 per row a
    fit month is 37 GB — and are re-streamed by :func:`iter_view_blocks`, which
    reproduces this table's row order exactly.

    ``shard_stride > 1`` subsamples MDS shards. A diagnostic only — the result
    is a partial panel and ``build_month_table`` refuses to cache it.
    """
    cache_dir = Path(cache_dir)
    path = table_path(ym, anchors_per_day, cache_dir)
    found = None if force or shard_stride != 1 else existing_table(
        ym, anchors_per_day, cache_dir)
    if found is not None:
        z = np.load(found, allow_pickle=False)
        # A table written before the stamp existed carries the ORIGINAL
        # midpoint target, which is what the un-stamped default names.
        stamped = (str(z["xs_anchor_stats_dir"])
                   if "xs_anchor_stats_dir" in z.files
                   else "/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
        if stamped != str(xs_anchor_stats_dir):
            raise SystemExit(
                f"{found.name} was built against {stamped} but you asked for "
                f"{xs_anchor_stats_dir}. Z is a function of the RETURN "
                f"DEFINITION, so this cache cannot answer your question. "
                f"Point --cache-dir at a different directory (X is unchanged, "
                f"but rebuilding is the only way to get the new Z)."
            )
        # X CHANGED TOO, and the stamp above only guards Z. A table built
        # before the information columns existed has 11 fewer predictors, and
        # serving it would silently score the classical arms on strictly less
        # information than the encoder -- the exact asymmetry this fixes.
        cached_names = [str(x) for x in z["predictor_names"]]
        if cached_names != PREDICTOR_NAMES:
            missing = [n for n in PREDICTOR_NAMES if n not in cached_names]
            raise SystemExit(
                f"{found.name} holds {len(cached_names)} predictors, this code "
                f"builds {len(PREDICTOR_NAMES)}"
                + (f" (missing e.g. {missing[:3]})" if missing else "")
                + ". X is stale; rebuild with --force or a fresh --cache-dir."
            )
        out = {k: z[k] for k in z.files
               if k not in ("view_crop", "view_pool", "xs_anchor_stats_dir")}
        out["predictor_names"] = [str(x) for x in out["predictor_names"]]
        out["target_names"] = [str(x) for x in out["target_names"]]
        return _alias(out, ym, anchors_per_day)

    mosaic_dir = Path(mosaic_dir or BLL01MachineConfig().mosaic_dir)
    sched = MarketSchedule(BLL01MachineConfig().holiday_csv)
    stats = AnchorStats(Path(xs_anchor_stats_dir) / f"{ym}.npz")
    anchors = day_anchors(anchors_per_day)
    mdir = mosaic_dir / ym[:4] / ym[5:]

    X, Z, cells, tickers, dates, ancs, aggs = [], [], [], [], [], [], []
    # THE INFORMATION TOKEN ON: without it the baselines read 9 channels while
    # the encoder reads 20, which is not a modelling choice but an oversight.
    for views, metas in iter_panel(mdir, stats, sched, anchors, batch_size,
                                   num_shards=max(shard_stride, 1),
                                   info_norm_stats=True, info_window=True):
        # cell_agg is a function of (date, anchor) alone, so a batch splits into
        # at most a handful of resolution groups and each is featurized in one
        # vectorized pass.
        by_agg: dict[int, list[int]] = {}
        for j, (_target, date, anchor, _tk, _raw, _quote) in enumerate(metas):
            a = cell_agg(date, int(anchor), int(anchor))
            if a is None:
                continue
            by_agg.setdefault(int(a), []).append(j)
        for a, idx in by_agg.items():
            sel = np.asarray(idx)
            X.append(view_predictors(views[sel], a))
            for j in idx:
                zz, date, anchor, tk, _raw, _quote = metas[j]
                Z.append(zz)
                cells.append(f"{date}@{int(anchor)}")
                tickers.append(tk)
                dates.append(date)
                ancs.append(int(anchor))
                aggs.append(a)

    if not X:
        raise SystemExit(f"{ym}: empty panel — check the mosaic month exists")

    out = {
        "X": np.concatenate(X).astype(np.float32),
        "Z": np.asarray(Z, dtype=np.float32),
        "cell": np.asarray(cells),
        "ticker": np.asarray(tickers),
        "date": np.asarray(dates),
        "anchor": np.asarray(ancs, dtype=np.int32),
        "agg": np.asarray(aggs, dtype=np.int16),
        "predictor_names": np.asarray(PREDICTOR_NAMES),
        "target_names": np.asarray(TARGET_NAMES),
        # WHICH TARGET DEFINITION Z IS. X is a function of the view alone and
        # never changes; Z is a function of the RETURN, which became a ratio of
        # two forward VWAP windows on 2026-08-22. A cached table keyed only on
        # (ym, anchors) would be served happily to a caller asking for the new
        # target and would return the old one -- silently, as a result.
        "xs_anchor_stats_dir": np.asarray(str(xs_anchor_stats_dir)),
    }
    if verbose:
        mb = sum(v.nbytes for v in out.values() if isinstance(v, np.ndarray)) / 1e6
        print(f"  {ym} a{anchors_per_day}: {len(out['X'])} rows, "
              f"{len(set(cells))} cells, {mb:.0f} MB")

    if shard_stride == 1:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The temp name MUST end in .npz: np.savez silently appends the
        # extension when it does not, and the rename then looks for a file
        # that was never written.
        tmp = path.with_name(path.name[:-4] + ".tmp.npz")
        np.savez(tmp, **out)
        tmp.rename(path)
    out["predictor_names"] = PREDICTOR_NAMES
    out["target_names"] = TARGET_NAMES
    return _alias(out, ym, anchors_per_day)


def _alias(table: dict, ym: str, anchors_per_day: int) -> dict:
    """Expose ``Z`` under the name the models fit against, and stamp identity.

    ``Z`` says what the array IS — a cross-sectional z-score — and ``Y`` says
    what it is FOR. One array, two names, so the cache file stays honest and
    ``models.Baseline`` needs no knowledge of the panel at all.

    ``ym``/``anchors_per_day`` make the table a HANDLE as well as an array: the
    full-view learners cannot be handed 37 GB of tensors in the dict, so they
    reopen the month themselves through :func:`iter_view_blocks`. Stamping the
    identity here is what keeps that reopen pointing at the same rows.
    """
    table["Y"] = table["Z"]
    table["ym"] = ym
    table["anchors_per_day"] = anchors_per_day
    return table


def prev_month(ym: str) -> str:
    """``"2008-01"`` -> ``"2007-12"``. The inverse of ``next_month``."""
    y, m = (int(x) for x in ym.split("-"))
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"


def month_on_disk(ym: str, mosaic_dir: str | Path | None = None) -> bool:
    """Is there a mosaic month to decode? The panel starts at 2008-01."""
    mosaic_dir = Path(mosaic_dir or BLL01MachineConfig().mosaic_dir)
    return (mosaic_dir / ym[:4] / ym[5:]).is_dir()


def fit_pool(last_month: str, span: int,
             mosaic_dir: str | Path | None = None) -> list[str]:
    """The ``span`` months ending at ``last_month``, oldest first.

    UP TO ``span``, not exactly ``span``. The mosaic begins at 2008-01, so an
    early eval month cannot have six months behind it; a short pool is
    reported (``n_fit_months`` on every record) rather than silently padded or
    silently dropped. A month missing from the middle of the range would be a
    different problem and this does not hide it either -- it just is not one
    the archive has.
    """
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    months = []
    ym = last_month
    for _ in range(span):
        months.append(ym)
        ym = prev_month(ym)
    months = [m for m in reversed(months) if month_on_disk(m, mosaic_dir)]
    if not months:
        raise SystemExit(f"no mosaic months at or before {last_month}")
    return months


# Row-wise arrays a pooled table concatenates. ``predictor_names`` /
# ``target_names`` are per-panel and must MATCH rather than concatenate; the
# build guards them against the code's own layout, so equal-across-months
# follows.
_ROW_KEYS = ("X", "Z", "cell", "ticker", "date", "anchor", "agg")


def pool_tables(tables: list[dict]) -> dict:
    """One table over several months' rows, for a multi-month fit.

    EVERY MODEL HERE IS FITTED ROW-WISE, which is what makes this legitimate
    and is worth saying once: ``models.Baseline`` subclasses read ``X``, ``Y``
    and ``cell`` and nothing else, and none of them carries state along a time
    index -- an ARMA's MA terms are RECONSTRUCTED per row by Hannan-Rissanen
    (see ``ARMApq``), never carried across rows. So pooling months is exactly
    pooling their rows, and no model can tell a 6-month pool from a month with
    six times the days.

    ``cell`` stays unique across the pool because a cell id is
    ``"{date}@{anchor}"`` and a date names its month. GARCH's rank-IC
    selection therefore still ranks WITHIN a cross-section, never across two
    months' cross-sections.

    The result carries ``months`` -- ``[(ym, n_rows), ...]`` in row order --
    which is the handle the view learners re-stream by; see
    ``view_models.stream_month``.
    """
    if not tables:
        raise ValueError("pool_tables: nothing to pool")
    first = tables[0]
    for t in tables[1:]:
        if list(t["predictor_names"]) != list(first["predictor_names"]):
            raise SystemExit(
                f"{t['ym']} and {first['ym']} disagree on the predictor "
                "layout; one of them is a stale cache")
        if list(t["target_names"]) != list(first["target_names"]):
            raise SystemExit(
                f"{t['ym']} and {first['ym']} disagree on the target layout")
        if t["anchors_per_day"] != first["anchors_per_day"]:
            raise SystemExit("pooling tables built on different anchor grids")
    out = {k: np.concatenate([t[k] for t in tables]) for k in _ROW_KEYS}
    out["predictor_names"] = list(first["predictor_names"])
    out["target_names"] = list(first["target_names"])
    out["months"] = [(t["ym"], len(t["Z"])) for t in tables]
    out["stream_kw"] = first.get("stream_kw", {})
    return _alias(out, "+".join(t["ym"] for t in tables),
                  first["anchors_per_day"])


def build_pooled_table(months: list[str], *, anchors_per_day: int,
                       **kw) -> dict:
    """Load or build each month, then :func:`pool_tables` them."""
    if len(months) == 1:
        return build_month_table(months[0], anchors_per_day=anchors_per_day,
                                 **kw)
    return pool_tables([build_month_table(m, anchors_per_day=anchors_per_day,
                                          **kw) for m in months])


def restrict_targets(table: dict, targets: list[str]) -> dict:
    """A view of ``table`` carrying only ``targets``, in the order given.

    Scoring one horizon does not need the other fifteen columns, and dropping
    them is not just a saving: the view learners partition their rows by
    CENSORING PATTERN, and a pattern is a pattern over the kept targets. At
    three targets there are at most eight, which is inside ``MAX_SHELLS``, so
    the partition is exact and ``_patterns`` never has to project. Carrying
    all eighteen would put 69 patterns through a lossy projection to answer a
    question about three of them.
    """
    keep = [table["target_names"].index(t) for t in targets]
    out = dict(table)
    out["Z"] = np.ascontiguousarray(table["Z"][:, keep])
    out["target_names"] = list(targets)
    out["Y"] = out["Z"]
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--anchors", type=int, default=8,
                   help="8 = the eval grid, 36 = the fit grid")
    p.add_argument("--force", action="store_true")
    p.add_argument("--shard-stride", type=int, default=1)
    p.add_argument("--mosaic-dir", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    for ym in a.months:
        build_month_table(ym, anchors_per_day=a.anchors, force=a.force,
                          mosaic_dir=a.mosaic_dir,
                          batch_size=a.batch_size, shard_stride=a.shard_stride)


if __name__ == "__main__":
    main()
