"""The (stock, day) view grid the factor-structure stages are built on.

build_fullday_embs.py (stage 2) and build_return_panel.py (stage 3) read their
views and their per-observation metadata through these helpers, so this is
what the latent table's F1/F2 columns rest on.

Views follow the eval protocol's crop distribution: a random scale in
[0.5, 1.0] of the session at a random position, aggregated to SEQ_LEN tokens,
seed CROP_SEED. Two variants:

  * ``sync_daily=True`` -- the crop's (scale, position) is drawn ONCE PER
    DAY (``_day_crop_params``) and shared by every stock, so all of a day's
    views end at the same wall-clock instant.
  * ``sync_daily=False`` -- one crop per observation, replicating
    StreamingMarketDataset._getitem_numpy's RNG stream exactly, so the views
    are bit-identical to the eval protocol's crops.

MOVED 2026-10-06 from backtesting/build_cache.py, whose backtest is gone; only
the grid builder was still in use. The constants below were backtesting/
common.py's and are unchanged, so a grid built here is byte-identical to one
built there (checked on 2020-08, both variants, with and without the
information-token channels).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))
sys.path.insert(0, str(_REPO_ROOT / "plots"))

from streaming import StreamingDataset  # noqa: E402

from market_jepa.augmentations import (  # noqa: E402
    _aggregate_numpy_jittered, _random_resized_crop_numpy,
)
from market_jepa.training.streaming_dataset import (  # noqa: E402
    StreamingMarketDataset, discover_streams,
)
from stable_finance.dataset import (  # noqa: E402
    compute_pair_targets, standard_open_est,
)

SEQ_LEN = 2048            # tokens per encoder input
SCALE_RANGE = (0.5, 1.0)  # eval-protocol crop scales
CROP_SEED = 42            # eval-protocol dataset seed: crops are
                          # bit-identical to the paper's eval crops
HORIZONS = [300, 600, 900, 1800, 3600, 7200]
TYPES = ["return", "volatility_change", "spread_change"]

_BID, _VWAP, _ASK, _VOL = 0, 1, 4, 7


def _make_dataset(date_start: str, date_end: str, machine, schedule,
                  info: bool = False):
    """Dataset instance used only for MDS access + preprocessing internals
    (we never call its __getitem__; the augmentation config is a dummy).

    ``info`` appends the information-token channels (per-norm-group mean/std
    plus the three window descriptors) as trailing constant columns, so the
    grid can also feed encoders trained WITH the token. It does not touch the
    data channels or the crop draws, so a grid built with it is byte-identical
    to one built without it on the first ``len(feature_columns)`` channels --
    which is why one grid can serve both, and why grid_meta (metadata only,
    no views) still verifies.
    """
    streams = discover_streams(machine.mosaic_dir, date_start, date_end)
    return StreamingMarketDataset(
        augmentations=[{"name": "fixed_window", "window_size_sec": SEQ_LEN}],
        date_start=date_start, date_end=date_end, seed=CROP_SEED,
        n_pairs_per_obs=1, targets=None, schedule=schedule, streams=streams,
        shuffle=False, batch_size=64, allow_unsafe_types=True, predownload=64,
        risk_factor_dir=machine.risk_factor_dir,
        risk_factor_tickers=[], risk_factor_columns=None,
        info_norm_stats=info, info_window=info,
    )


def _day_crop_params(date_str: str, schedule) -> tuple[int, int]:
    """One crop draw per trading day, shared by every stock that day.

    Any cross-sectional quantity (a covariance across names, a factor
    estimate) needs the day's views to end at the same wall-clock instant, so
    the crop's (scale, position) is drawn once per date -- from the same
    distribution the eval protocol draws per observation (scale ~ U[0.5, 1.0]
    of the session, position uniform) -- and applied to all tickers.

    Returns (start_tod, window) in seconds relative to the standard open.
    """
    from stable_finance.dataset import timeline_bounds_est

    ts_open, ts_close = timeline_bounds_est(date_str, schedule=schedule)
    n_day = ts_close - ts_open
    rng = np.random.RandomState(
        int((CROP_SEED + int(date_str.replace("-", ""))) % (2 ** 32)))
    scale = rng.uniform(*SCALE_RANGE)
    agg = max(1, round(scale * n_day / SEQ_LEN))
    window = agg * SEQ_LEN
    if window > n_day:
        agg = max(1, n_day // SEQ_LEN)
        window = agg * SEQ_LEN
    max_start = n_day - window
    start_tod = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
    return start_tod, window


def build_grid(ds, obs_indices, schedule, sync_daily: bool) -> dict | None:
    """Build views + targets + meta for a set of (ticker, day) observations.

    ``sync_daily=False``: one eval-protocol crop per observation. The RNG
    replicates StreamingMarketDataset._getitem_numpy exactly (base_seed =
    seed + idx; one randint for the aug-config pick; then the crop's own
    draws), so with CROP_SEED=42 the views match the paper's eval crops
    bit-for-bit.

    ``sync_daily=True``: the crop is drawn once per date via
    ``_day_crop_params`` and shared by every stock, so the day's views are
    simultaneous. Tickers whose grid doesn't cover the day's window (late
    start / short data) drop out for that day.
    """
    views, tickers, dates, aggs_out, tods, slacks = [], [], [], [], [], []
    mids, halfspreads, dvs, r_closes, targets = [], [], [], [], []
    day_params: dict[str, tuple[int, int]] = {}
    n_obs_used = 0
    for idx in obs_indices:
        sample = StreamingDataset.__getitem__(ds, int(idx))
        date_str = sample.get("date", "")
        if schedule.is_closed(date_str):
            continue
        pre = ds._preprocess_to_numpy(sample)
        if pre is None or len(pre[1]) < 10:
            continue
        canonical_sec, features = pre
        n = len(features)
        tod_base = int(canonical_sec[0]) - standard_open_est(date_str)

        if sync_daily:
            if date_str not in day_params:
                day_params[date_str] = _day_crop_params(date_str, schedule)
            start_tod, window = day_params[date_str]
            start = start_tod - tod_base
            agg = window // SEQ_LEN
            if start < 0 or start + window > n:
                continue
            raw = features[start: start + window]
            view = _aggregate_numpy_jittered(raw, agg)
            if view is None or len(view) != SEQ_LEN:
                continue
        else:
            rng = np.random.RandomState(int((CROP_SEED + int(idx)) % (2 ** 32)))
            rng.randint(0, 1)   # the aug-config pick in _getitem_numpy
            view, start, agg, window = _random_resized_crop_numpy(
                features, SCALE_RANGE, SEQ_LEN, rng,
            )
            if view is None or len(view) != SEQ_LEN:
                continue
        t_idx = start + window - 1
        slack = (n - 1) - t_idx
        bid, ask = features[t_idx, _BID], features[t_idx, _ASK]
        if not (bid > 0 and ask > 0) or np.isnan(bid) or np.isnan(ask):
            continue
        bid_c, ask_c = features[n - 1, _BID], features[n - 1, _ASK]
        mid_close = (bid_c + ask_c) / 2.0 if (bid_c > 0 and ask_c > 0) else np.nan

        prior_vwap = ds._prior_vwap_numpy(features, start)
        ds._ffill_vwap_numpy(view, prior_vwap)
        # THE DATASET'S OWN normalize, not a local normalize_numpy call: it is
        # the single place that widens a view for the information token, and it
        # raises if the window descriptors are missing rather than silently
        # feeding the token zeros. With info=False it is exactly
        # normalize_numpy and the views are unchanged.
        view = ds._normalize_view(view, tod_start_sec=tod_base + start, agg=agg)
        if np.isnan(view).any():
            np.nan_to_num(view, copy=False, nan=0.0)

        mid_t = (bid + ask) / 2.0
        raw_window = features[start: t_idx + 1]
        with np.errstate(invalid="ignore"):
            dv = float(np.nansum(raw_window[:, _VWAP] * raw_window[:, _VOL]))
        tgt = compute_pair_targets(
            focal_features=features, t_idx=t_idx, horizons=HORIZONS,
            types=TYPES, rf_data=None, rf_price_mode=None,
            date_str=None, rf_t_idx=None,
        )
        views.append(view.T.astype(np.float16))   # (9, 2048), (20, 2048) w/ info
        tickers.append(sample.get("ticker", ""))
        dates.append(date_str)
        aggs_out.append(agg)
        tods.append(tod_base + t_idx)
        slacks.append(slack)
        mids.append(mid_t)
        halfspreads.append((ask - bid) / 2.0)
        dvs.append(dv)
        r_closes.append(mid_close / mid_t - 1.0 if mid_t > 0 else np.nan)
        targets.append(tgt)
        n_obs_used += 1
    if not views:
        return None
    return {
        "views": np.stack(views),
        "ticker": np.asarray(tickers, dtype="U12"),
        "date": np.asarray(dates, dtype="U10"),
        "agg": np.asarray(aggs_out, dtype=np.int32),
        "tod": np.asarray(tods, dtype=np.int32),
        "slack": np.asarray(slacks, dtype=np.int32),
        "mid": np.asarray(mids, dtype=np.float32),
        "halfspread": np.asarray(halfspreads, dtype=np.float32),
        "dv": np.asarray(dvs, dtype=np.float32),
        "r_close": np.asarray(r_closes, dtype=np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "n_obs": n_obs_used,
    }


META_KEYS = ("ticker", "date", "agg", "tod", "slack", "mid", "halfspread",
             "dv", "r_close", "targets")
