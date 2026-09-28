"""5-minute mid-quote panels for the latent-factor experiment.

For each of the 32 backtest months (the ff_fullday_cache set), read every
(stock, day) observation from mosaic and sample the quote mid at 5-minute
marks on the canonical session grid (09:30 .. 16:00 -> 79 marks, 78
intraday increments; overnight returns never enter). Only full-length
sessions are kept (half days dropped, logged). Leading missing marks are
backfilled with the first observed mid (Pelger's start-of-day rule);
interior gaps are previous-tick by construction of the canonical grid.
A ticker enters the month's panel only if it has a usable series on every
kept day (the paper's intersection rule).

Output: /data/lab/market-jepa-checkpoints/factor_structure_cache/panel_<ym>.npz
    mids     f64 (n_tickers, n_days, 79)
    tickers  U12
    dates    U10

Usage:  uv run python plots/latent_eval/factors/build_return_panel.py [2020-08 ...]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

# Derived, not hardcoded: this was the absolute bll01 checkout path
# until 2026-08-20, so the script could only ever run on bll01 -- on
# a cluster node it fails at `import build_cache` with the repo
# sitting somewhere else entirely.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backtesting"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "plots"))

import build_cache as bc  # noqa: E402
from streaming import StreamingDataset  # noqa: E402

from stable_finance.dataset import (  # noqa: E402
    MarketSchedule, standard_open_est, timeline_bounds_est,
)
from market_jepa.schemas import BLL01MachineConfig, machine_from_env  # noqa: E402

# Overridable so the sweep can run one month per SLURM job on a node
# that does not mount /data/lab; unset, the path is exactly what it
# always was, so local runs are unchanged. Added 2026-08-20.
CACHE = Path(os.environ.get(
    "FF_FULLDAY_CACHE",
    "/data/lab/market-jepa-checkpoints/ff_fullday_cache"))
# Separate constant from CACHE above, and it was missed when CACHE was made
# overridable -- this is the one phase 3 WRITES to.
OUT = Path(os.environ.get(
    "FACTOR_STRUCTURE_CACHE",
    "/data/lab/market-jepa-checkpoints/factor_structure_cache"))
STEP = 300
FULL_DAY = 23400                       # 09:30 - 16:00
MARKS = FULL_DAY // STEP + 1           # 79
_BID, _ASK = 0, 4


def month_endpoints(ym: str) -> tuple[str, str]:
    import calendar
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{ym}-01", f"{ym}-{calendar.monthrange(y, m)[1]:02d}"


def day_mids(features: np.ndarray, tod_base: int) -> np.ndarray | None:
    """79 mid quotes on the 5-min grid, start-of-day backfilled."""
    n = len(features)
    mids = np.full(MARKS, np.nan)
    for k in range(MARKS):
        pos = k * STEP - tod_base
        if pos < 0:
            continue
        if pos >= n:
            pos = n - 1                # previous tick past the data end
        bid, ask = features[pos, _BID], features[pos, _ASK]
        if bid > 0 and ask > 0 and np.isfinite(bid) and np.isfinite(ask):
            mids[k] = (bid + ask) / 2.0
    valid = np.flatnonzero(np.isfinite(mids))
    if len(valid) == 0 or len(valid) < MARKS - valid[0]:
        return None                    # interior hole — canonical grid broken
    mids[: valid[0]] = mids[valid[0]]  # backfill the late open
    return mids


def build_month(ym: str, machine, schedule) -> None:
    out_path = OUT / f"panel_{ym}.npz"
    if out_path.exists():
        print(f"[{ym}] exists — skip", flush=True)
        return
    t0 = time.time()
    es, ee = month_endpoints(ym)
    ds = bc._make_dataset(es, ee, machine, schedule)

    per_day: dict[str, dict[str, np.ndarray]] = {}
    n_half = n_broken = 0
    for idx in range(ds.num_samples):
        sample = StreamingDataset.__getitem__(ds, int(idx))
        date_str = sample.get("date", "")
        if schedule.is_closed(date_str):
            continue
        ts_open, ts_close = timeline_bounds_est(date_str, schedule=schedule)
        if ts_close - ts_open != FULL_DAY:
            n_half += 1
            continue
        pre = ds._preprocess_to_numpy(sample)
        if pre is None or len(pre[1]) < 10:
            continue
        canonical_sec, features = pre
        tod_base = int(canonical_sec[0]) - standard_open_est(date_str)
        mids = day_mids(features, tod_base)
        if mids is None:
            n_broken += 1
            continue
        per_day.setdefault(date_str, {})[sample.get("ticker", "")] = mids
    del ds

    dates = sorted(per_day)
    if not dates:
        raise RuntimeError(f"{ym}: no full-length days")
    common = set.intersection(*(set(per_day[d]) for d in dates))
    tickers = sorted(common)
    mids = np.stack([
        np.stack([per_day[d][t] for d in dates]) for t in tickers])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mids=mids,
                        tickers=np.asarray(tickers, dtype="U12"),
                        dates=np.asarray(dates, dtype="U10"))
    n_union = len(set().union(*(per_day[d] for d in dates)))
    print(f"[{ym}] {len(tickers)}/{n_union} tickers x {len(dates)} days "
          f"(half-day obs skipped {n_half}, broken {n_broken}) "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    months = sys.argv[1:] or sorted(
        p.name for p in CACHE.iterdir()
        if p.is_dir() and len(p.name) == 7 and p.name[4] == "-")
    machine = machine_from_env(BLL01MachineConfig)
    schedule = MarketSchedule(machine.holiday_csv)
    for ym in months:
        build_month(ym, machine, schedule)


if __name__ == "__main__":
    main()
