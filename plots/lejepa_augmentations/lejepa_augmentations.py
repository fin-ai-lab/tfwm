"""The five LeJEPA positive pairings: one base view and each arm's second view.

A 2x3 grid of candles. The first panel is a clean crop of MCD on 2019-07-10.
The other five panels each show the second view one LeJEPA arm of the paper
would pair with it, drawn over the base view's VWAP as a grey reference line
(unlabelled in the figure; the caption carries it, as it does the colors).
The base and local panels name their wall-clock span, New York time:

    Same Stock, Diff. View   an independent local crop of the same stock-day
    Time Warping             the same window through a randomly warped clock
    Gaussian Noising         the same window, noised after normalization
    Cross Stock              ADP (FF49 34) over the same wall-clock window
    C-S, Same Industry       SBUX (FF49 44, MCD's industry), same window

FAITHFULNESS. Views come from the training code's own kernels and normalization
(``market_jepa.augmentations``, ``training.utils``), with the recipe read off
the tfwm-lejepa-* checkpoints' train_meta.json: 50-100% of the session per
global crop, 5-50% per local crop, warp_strength 0.75 with 8 knots,
noise_sigma 0.75. ONE THING IS CHANGED FOR LEGIBILITY: 64 tokens per global
view rather than 2048 (16 rather than 512 per local), so each candle aggregates
32x more seconds. The crop geometry is untouched. In training both views of a
pair are augmented; here the base stays clean so every panel is read against
the same reference.

SAME DRAWS AS THE TUTORIAL. The draws follow the random-number order of
``augmentation_gallery`` in fin-ai-lab/wm-booth-2026-finance-tutorial, so the
notebook's section 3 renders the same views from the same seed -- except the
local crop, which this figure takes from the session open (see ``gallery``).

NORMALIZED TIME. Every panel's x-axis runs 0 to 1 over its own view, one
equal-width slot per token, as the encoder sees it (equally spaced positions,
nothing in its input carrying a duration). So the 16-token local view fills
the panel with candles 4x wider than the 64-token ones, the time-warped view's
warp shows as its candles running ahead of or behind the grey base line, and
the grey line itself is the base view on the same 0-1 scale -- in the local
panel it covers a different stretch of the day than the candles do.

CANDLES. The stores hold no per-bucket open, so a body runs from the previous
token's VWAP to this one's; the wick is the token's own high and low. A body is
72% of its bucket, as in the earlier renderer.

The samples are all in the released Market-1T data. By default they are read
from the local day store; pass
``--daystore hf://datasets/fin-ai-lab/Market-1T-1Hz-2019H2-2020-daystore/1Hz_daystore``
to read the published copy instead.

Run:
    uv run plots/lejepa_augmentations/lejepa_augmentations.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
from market_jepa.augmentations import (  # noqa: E402
    _aggregate_numpy_jittered,
    _warped_aggregate_numpy,
    prepare_augmented_view,
)
from market_jepa.schemas import LocalMachineConfig  # noqa: E402
from market_jepa.training.streaming_dataset import FEATURE_COLUMNS  # noqa: E402
from market_jepa.training.utils import build_norm_groups, ffill_vwap, prior_vwap  # noqa: E402
from stable_finance.dataset.daystore import DayRecord, discover_days  # noqa: E402
from style import COMPACT_RC_PARAMS, WIDTH_FULL, apply_style, save_figure  # noqa: E402

OUT = Path(__file__).resolve().parent / "lejepa_augmentations"
DATE = "2019-07-10"
BASE, CROSS, INDUSTRY = "MCD", "ADP", "SBUX"
SEED = 7

# The trained recipe, display resolution aside (see the module docstring).
SEQ_LEN, LOCAL_SEQ_LEN = 64, 16
GLOBAL_SCALE, LOCAL_SCALE = (0.5, 1.0), (0.05, 0.5)
WARP_KNOTS, WARP_STRENGTH, NOISE_SIGMA = 8, 0.75, 0.75

ARMS = {
    "same_stock": "Same Stock, Diff. View",
    "time_warp": "Time Warping",
    "gaussian_noise": "Gaussian Noising",
    "cross_stock": "Cross Stock",
    "cross_stock_industry": "C-S, Same Industry",
}
# Each arm's SIGReg weight. Read off fin-ai-lab/tfwm-lejepa-*/<month>/train_meta.json
# (``lamb``); constant across all five published months, and config.json agrees.
LAMBDAS = {
    "same_stock": 0.05,
    "time_warp": 0.001,
    "gaussian_noise": 0.3,
    "cross_stock": 0.2,
    "cross_stock_industry": 0.1,
}
# tab10, per the project palette rule; grey is the base-view reference.
UP, DOWN, REFERENCE = "#2ca02c", "#d62728", "#7f7f7f"
VWAP, HIGH, LOW = (FEATURE_COLUMNS.index(c) for c in ("vwap_all", "high", "low"))
NORM_GROUPS = build_norm_groups(FEATURE_COLUMNS)
OPEN_TOD = 9.5 * 3600


def load_day(daystore: str) -> dict[str, dict]:
    """The three ticker-days, as dense regular-session grids."""
    [day] = discover_days(daystore, DATE, DATE)
    record = DayRecord(day)
    return {t: record.record(record.ticker_index(t)) for t in (BASE, CROSS, INDUSTRY)}


def crop(n_rows: int, rng: np.random.RandomState, length: int,
         scale_range: tuple[float, float]) -> tuple[int, int, int]:
    """(start, window, seconds per token): the training draw, scale then start."""
    aggregation = max(1, round(float(rng.uniform(*scale_range)) * n_rows / length))
    window = aggregation * length
    return int(rng.randint(0, n_rows - window + 1)), window, aggregation


def finish(raw: np.ndarray, features: np.ndarray, start: int, aggregation: int) -> np.ndarray:
    """Training's VWAP fill and per-view normalization, (tokens, 9)."""
    ffill_vwap(raw, prior_vwap(features, start))
    view, _ = prepare_augmented_view(
        raw, NORM_GROUPS, start_seconds=float(start), aggregation_seconds=float(aggregation))
    return np.nan_to_num(view, nan=0.0)


def uniform(record: dict, start: int, length: int, aggregation: int) -> dict:
    features = record["features"].astype(np.float64)
    raw = _aggregate_numpy_jittered(features[start:start + length * aggregation], aggregation)
    first = int(record["grid_start"]) + start
    return {"view": finish(raw, features, start, aggregation),
            "edges": first + aggregation * np.arange(length + 1),
            "seconds_per_token": aggregation, "ticker": record["ticker"]}


def gallery(records: dict[str, dict], seed: int = SEED) -> dict[str, dict]:
    base = records[BASE]
    features = base["features"].astype(np.float64)
    rng = np.random.RandomState(seed)
    start, window, aggregation = crop(len(features), rng, SEQ_LEN, GLOBAL_SCALE)
    views = {"base": {**uniform(base, start, SEQ_LEN, aggregation), "label": "Base View"}}

    # The local crop keeps its drawn scale but is taken from the SESSION OPEN,
    # a display choice; its panel names the span. The draw itself still
    # happens, so every later panel's randomness is unchanged.
    _, _, local_aggregation = crop(len(features), rng, LOCAL_SEQ_LEN, LOCAL_SCALE)
    views["same_stock"] = uniform(base, 0, LOCAL_SEQ_LEN, local_aggregation)

    warped = _warped_aggregate_numpy(
        features[start:start + window], SEQ_LEN, rng, WARP_KNOTS, WARP_STRENGTH)
    views["time_warp"] = {
        "view": finish(warped, features, start, aggregation),
        "edges": views["base"]["edges"],
        "seconds_per_token": aggregation, "ticker": BASE}

    # Channel-major draw, as the notebook adds noise to its (channels, tokens) tensor.
    noise = rng.standard_normal((len(FEATURE_COLUMNS), SEQ_LEN)).astype(np.float32).T
    views["gaussian_noise"] = {**views["base"], "view": views["base"]["view"] + NOISE_SIGMA * noise}

    base_first = int(base["grid_start"]) + start
    for arm, ticker in (("cross_stock", CROSS), ("cross_stock_industry", INDUSTRY)):
        partner = records[ticker]
        partner_start = base_first - int(partner["grid_start"])
        if partner_start < 0 or partner_start + window > len(partner["features"]):
            raise ValueError(f"{ticker} does not cover the base view's window")
        views[arm] = uniform(partner, partner_start, SEQ_LEN, aggregation)

    for arm, label in ARMS.items():
        views[arm]["label"] = label
    return views


def candles(ax, edges: np.ndarray, view: np.ndarray) -> None:
    """One candle per token, centred on and 72% as wide as its edges."""
    x = (edges[:-1] + edges[1:]) / 2
    half = 0.72 * np.diff(edges) / 2
    close = view[:, VWAP]
    opening = np.concatenate([close[:1], close[:-1]])
    colors = np.where(close >= opening, UP, DOWN)
    ax.add_collection(LineCollection(
        [((xi, lo), (xi, hi)) for xi, lo, hi in zip(x, view[:, LOW], view[:, HIGH])],
        colors=colors, linewidths=0.5))
    floor = 0.004 * float(np.ptp(view[:, [HIGH, LOW]]) or 1.0)
    bottom = np.minimum(opening, close)
    top = np.maximum(np.maximum(opening, close), bottom + floor)
    ax.add_collection(PolyCollection(
        [((xi - h, b), (xi + h, b), (xi + h, t), (xi - h, t))
         for xi, h, b, t in zip(x, half, bottom, top)],
        facecolors=colors, edgecolors=colors, linewidths=0.3))
    ax.autoscale_view()


def span(unix_seconds: np.ndarray) -> str:
    """A view's first and last edge as New York time, e.g. "9:30 AM-10:12 AM"."""
    ends = pd.to_datetime(unix_seconds[[0, -1]], unit="s", utc=True).tz_convert("America/New_York")
    return "\u2013".join(t.strftime("%I:%M %p").lstrip("0") for t in ends)


def plot(views: dict[str, dict]):
    apply_style(extra=COMPACT_RC_PARAMS)
    # Full width; 4/3 the height of plots/core/probe_fit_breadth.png (2/3 of double).
    # One y scale for all six panels (every view is in the same normalized
    # units), so tick labels appear only down the left column.
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_FULL, (2 / 3) * 2 * WIDTH_FULL * 0.36), sharey=True)
    base = views["base"]
    base_mid = (np.arange(SEQ_LEN) + 0.5) / SEQ_LEN
    for ax, arm in zip(axes.flat, ["base", *ARMS]):
        entry = views[arm]
        if arm != "base":
            ax.plot(base_mid, base["view"][:, VWAP], color=REFERENCE, lw=0.8, alpha=0.8)
        candles(ax, np.linspace(0.0, 1.0, len(entry["view"]) + 1), entry["view"])
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_title(entry["label"])
        if arm in LAMBDAS:
            ax.text(0.97, 0.96, rf"$\lambda$={LAMBDAS[arm]:g}", transform=ax.transAxes,
                    ha="right", va="top", fontsize="x-small")
        stock = {"base": f"{entry['ticker']}, {DATE}, {span(entry['edges'])}",
                 "same_stock": f"{entry['ticker']}, {span(entry['edges'])} (Local View)",
                 }.get(arm, entry["ticker"])
        ax.text(0.97, 0.04, stock, transform=ax.transAxes, ha="right", va="bottom",
                fontsize="x-small")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized Price")
    for ax in axes[-1]:
        ax.set_xlabel("Normalized Time")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--daystore", default=LocalMachineConfig.daystore_dir,
                   help="day-store root, local or hf:// (default: the data host's local store)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    fig = plot(gallery(load_day(args.daystore)))
    for path in save_figure(fig, args.out):
        print("wrote", path)


if __name__ == "__main__":
    main()
