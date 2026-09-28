"""Extract the closing-auction return for every (date, anchor, ticker).

WHY THIS IS A BRIDGE AND NOT THE PIPELINE. stable_finance now measures to the
auction directly: ``anchor_targets(..., [TO_CLOSE], spec=CLOSING_AUCTION_SPEC)``
reads it from a session that retains one measurement window past the close.
But every cached artifact in this project -- dense mosaic, day store, target
tables, panel cache -- was built with the default spec, whose last row is
15:59:59, so the auction is simply not in them. Rebuilding that chain is the
right fix and this is not it. This reads the auction window out of the raw
1 Hz parquet, which has always had it, and emits the one array the analysis is
missing, so the strategy can be measured today rather than after a re-densify.

WHAT IT COMPUTES. Volume-weighted price over ``[a, a+60)`` for each evaluation
anchor, and over ``[close, close+60)`` for the auction, then

    to_close(a) = auction_vwap / base_vwap(a) - 1

which is the same estimator the fixed-horizon targets use -- deliberately, so
that a to-close return is comparable to r900 rather than measured on a
different price basis. A 60-second window at the close is the auction: it
carries on the order of a thousand times a median second's volume.

The auction print does not land on one second. At 16:00:00 exactly only ~57%
of names have printed, ~80% by +2s, plateauing near 86% by +15s, so a window
is required rather than a bar; names that never print stay NaN and are simply
not held.

Usage:
    uv run plots/ic_vs_sharpe/build_close_returns.py --months 2014-01
    uv run plots/ic_vs_sharpe/build_close_returns.py --all --workers 12
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_finance.dataset.anchors import RETURN_VWAP_WINDOW  # noqa: E402
from stable_finance.dataset.calendar import (  # noqa: E402
    MarketSchedule,
    timeline_bounds_est,
)

RAW = "/data/polygon/snapshots/1Hz"
PANELS = "/data/lab/ic_sharpe_configs/panels"
OUT = "/data/lab/ic_sharpe_configs/close_returns"
HOLIDAYS = ROOT / "data" / "market_holidays.csv"
#: The eight evaluation anchors, session-relative seconds (the a8 grid).
EVAL_ANCHORS = (12300, 13800, 15300, 16800, 18300, 19800, 21300, 22800)


def window_vwap(second, price, volume, lo, hi):
    """Volume-weighted price over ``[lo, hi)``, NaN where nothing traded."""
    inside = (second >= lo) & (second < hi) & np.isfinite(price) \
        & np.isfinite(volume) & (volume > 0)
    if not inside.any():
        return np.nan
    return float(np.average(price[inside], weights=volume[inside]))


def one_day(date: str) -> dict | None:
    """Anchor and auction VWAPs for every ticker quoting on ``date``."""
    day = Path(RAW) / date[:4] / date[5:7] / f"{date}.parquet"
    if not day.is_dir():
        return None
    schedule = MarketSchedule(str(HOLIDAYS)) if HOLIDAYS.is_file() else None
    try:
        ts_open, ts_close = timeline_bounds_est(date, schedule=schedule)
    except ValueError:
        return None                       # market closed

    frames = []
    for part in sorted(glob.glob(str(day / "partition=*" / "*.parquet"))):
        frame = pd.read_parquet(
            part, columns=["ticker", "ts_interval", "vwap_all", "volume"])
        if len(frame):
            frames.append(frame)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    # ts_interval is the bar's epoch nanosecond; the dataset's own grid origin
    # carries a +1 second offset, which timeline_bounds_est already applies, so
    # subtracting ts_open puts a bar at its session-relative row.
    second = (frame["ts_interval"].to_numpy() // 10 ** 9).astype(np.int64) - ts_open
    price = frame["vwap_all"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    close_row = ts_close - ts_open        # first row after the session

    tickers, base, auction = [], [], []
    for ticker, index in frame.groupby("ticker").indices.items():
        sec, pri, vol = second[index], price[index], volume[index]
        if not np.isfinite(pri).any():
            continue
        tickers.append(str(ticker))
        base.append([window_vwap(sec, pri, vol, a, a + RETURN_VWAP_WINDOW)
                     for a in EVAL_ANCHORS])
        auction.append(window_vwap(sec, pri, vol, close_row,
                                   close_row + RETURN_VWAP_WINDOW))
    if not tickers:
        return None
    return {"date": date, "tickers": np.array(tickers),
            "base": np.array(base, dtype=np.float64),
            "auction": np.array(auction, dtype=np.float64),
            "close_row": int(close_row)}


def dates_of(month: str) -> list[str]:
    pattern = f"{RAW}/{month[:4]}/{month[5:7]}/{month}-*.parquet"
    return sorted(Path(p).name.split(".")[0] for p in glob.glob(pattern))


def build_month(month: str, workers: int) -> None:
    target = Path(OUT) / f"{month}-close.npz"
    if target.exists():
        print(f"    skip {month} (cached)", flush=True)
        return
    dates = dates_of(month)
    if not dates:
        print(f"    !! {month}: no raw days", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        days = [d for d in pool.map(one_day, dates) if d is not None]
    if not days:
        print(f"    !! {month}: nothing extracted", flush=True)
        return

    # One ragged day-list flattened to (date, ticker) rows, which is how the
    # panel joins it: the universe changes between days and padding a fixed
    # ticker axis would invent quotes for names that were not trading.
    dates_out = np.concatenate([[d["date"]] * len(d["tickers"]) for d in days])
    tickers = np.concatenate([d["tickers"] for d in days])
    base = np.concatenate([d["base"] for d in days], axis=0)
    auction = np.concatenate([d["auction"] for d in days])
    with np.errstate(invalid="ignore", divide="ignore"):
        to_close = np.where(np.isfinite(base) & (base > 0),
                            auction[:, None] / base - 1.0, np.nan)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, month=month, dates=dates_out, tickers=tickers,
        anchors=np.array(EVAL_ANCHORS), base=base.astype(np.float32),
        auction=auction.astype(np.float32),
        to_close=to_close.astype(np.float32))
    covered = np.isfinite(to_close).mean(axis=0)
    print(f"    ok   {month}: {len(days)} days, {len(tickers)} (date, ticker) "
          f"rows, coverage by anchor "
          f"{' '.join(f'{c:.0%}' for c in covered)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--all", action="store_true",
                        help="every month present in the panel cache")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    months = args.months or []
    if args.all:
        months = sorted({Path(p).name.rsplit("-", 3)[1] + "-"
                         + Path(p).name.rsplit("-", 3)[2]
                         for p in glob.glob(f"{PANELS}/*-panel.npz")})
    if not months:
        parser.error("pass --months or --all")
    print(f"==> {len(months)} month(s), {args.workers} workers", flush=True)
    for month in months:
        build_month(month, args.workers)
    print("==> done", flush=True)


if __name__ == "__main__":
    main()
