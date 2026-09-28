"""Seed spread per month, as boxes: is the IC a property of the month or the seed?

One panel per task, and within a panel one BOX PER TRAINING MONTH, each box the
distribution over seeds of that month's head IC. It answers the question by eye
and without an estimator: if the boxes sit far apart and are individually short,
the month moves the number and the seed does not.

A box is drawn only when a month has at least ``--min-seeds`` scored seeds. The
per-month seed counts are PRINTED rather than drawn: a "n=10" under every tick
is nine-tenths redundant ink on a full sweep, and the one case it guards
against -- a half-filled month reading as a tight one -- is caught by the
stdout line, which names every month short of the fullest one on the panel.

THE HEAD, NOT THE PROBE. These runs score head-only (POST_TRAIN_PROBE=0), so
there is no ``xs_ic/<task>`` key to read; ``--readout probe`` still works for an
older tree that has one.

THE CHECKPOINT TREE IS THE GENERATION. Its suffix is the commit the sweep was
submitted at, so generations cannot be mixed by accident, and there is no
default that spans them -- see variance_decomp.py. This defaults to the NEWEST
variance-decomp tree on disk rather than a pinned one, because a figure that
names a fixed tree is how a figure goes stale.

Usage::

    uv run plots/variance_decomp/variance_decomp_boxes.py
    uv run plots/variance_decomp/variance_decomp_boxes.py --ckpt-root <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from variance_decomp import collect  # noqa: E402
from style import (  # noqa: E402
    COMPACT_RC_PARAMS,
    SERIES_STYLES,
    WIDTH_FULL,
    apply_style,
    save_figure,
    set_two_decimal_yticks,
)

CKPT_PARENT = Path("/data/lab/market-jepa-checkpoints")
# (checkpoint series, xs_ic task, style key). The style key is what ties this
# figure to the rest of the paper: the panel's colour and its title both come
# from SERIES_STYLES, so a task keeps the hue it has in
# plots/full_data_multihead -- return purple, vol green, spread orange.
TASKS = [
    ("vd_supervised_return", "return_900", "supervised_return"),
    ("vd_supervised_vol_change", "volatility_change_900", "supervised_vol"),
    ("vd_supervised_spread_change", "spread_change_900", "supervised_spread"),
]

# Panel titles come from SERIES_STYLES, except where a panel here is wide
# enough for the name spelled out. The abbreviation exists for the stacked
# rows of plots/full_data_multihead, which are a fifth of the height.
TITLES = {"supervised_vol": "Volatility Change"}


def _mix(color: str, other: str, w: float):
    """``color`` blended ``w`` of the way toward ``other`` in RGB."""
    import matplotlib.colors as mcolors
    a = np.asarray(mcolors.to_rgb(color))
    b = np.asarray(mcolors.to_rgb(other))
    return tuple((1.0 - w) * a + w * b)


def newest_tree(parent: Path) -> Path:
    """The most recently modified variance-decomp checkpoint tree."""
    trees = [p for p in parent.glob("variance-decomp-*") if p.is_dir()]
    if not trees:
        raise SystemExit(f"no variance-decomp-* tree under {parent}")
    return max(trees, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", default=None,
                    help="default: the newest variance-decomp-* tree on disk")
    ap.add_argument("--readout", choices=("head", "probe"), default="head")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--out", default=str(_HERE / "variance_decomp_boxes.png"))
    a = ap.parse_args()

    root = Path(a.ckpt_root) if a.ckpt_root else newest_tree(CKPT_PARENT)
    # collect() keys its metric as xs_<metric>/<task>; the head lives under
    # xs_ic/head:<task>, so the task name carries the prefix.
    data = collect(root, metric="ic")
    print(f"reading {root}")

    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.36))
    any_box = False
    short: list[str] = []
    for ax, (series, task, style_key) in zip(axes, TASKS):
        style = SERIES_STYLES[style_key]
        color = style["color"]
        # One hue per panel, three weights of it: a pale fill so the box reads
        # as an interval rather than a bar, the hue itself on the outline and
        # the seed dots, and a darkened hue on the median so it stays legible
        # against the fill.
        fill = _mix(color, "white", 0.72)
        dark = _mix(color, "black", 0.35)
        key = f"head:{task}" if a.readout == "head" else task
        by_month = data.get((series, key)) or {}
        months = sorted(m for m, seeds in by_month.items()
                        if len(seeds) >= a.min_seeds)
        title = TITLES.get(style_key, style["label"])
        if not months:
            ax.set_title(f"{title}\nno months with >= "
                         f"{a.min_seeds} seeds yet")
            ax.set_xticks([])
            continue
        any_box = True
        vals = [[by_month[m][s] for s in sorted(by_month[m])] for m in months]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6,
                        medianprops=dict(color=dark, lw=1.2),
                        whiskerprops=dict(color=color, lw=0.9),
                        capprops=dict(color=color, lw=0.9),
                        flierprops=dict(marker="o", ms=2.2, mfc=color,
                                        mec="none", alpha=0.6))
        for patch in bp["boxes"]:
            patch.set(facecolor=fill, edgecolor=color, lw=0.9)
        # Every seed as a dot: a box over ten points hides how they stack.
        for i, v in enumerate(vals, start=1):
            jitter = (np.random.RandomState(i).rand(len(v)) - 0.5) * 0.22
            ax.plot(i + jitter, v, "o", ms=2.2, color=color,
                    alpha=0.5, mec="none", zorder=3)
        ax.set_xticks(range(1, len(months) + 1))
        # MM-YY, and vertical: nine of them share a panel a little over two
        # inches wide, so anything rotated less than 90 degrees either overlaps
        # its neighbour or eats the height the boxes need.
        ax.set_xticklabels([f"{m[5:7]}-{m[2:4]}" for m in months],
                           rotation=90, fontsize=6.5)
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
        ax.set_title(title, pad=4)
        # nbins=5 is a CEILING on intervals and this ladder has no step
        # between 0.02 and 0.05, so 4 would round the spread panel up to a
        # 0.1 step -- two ticks, and the only panel labelled to one decimal.
        set_two_decimal_yticks(ax, nbins=5)
        full = max(len(by_month[m]) for m in months)
        short += [f"{style_key} {m} n={len(by_month[m])}"
                  for m in months if len(by_month[m]) < full]
    # No x axis label: the ticks are months and read as months, and the panel
    # is too short to spend a line of height saying so. --readout is not in the
    # y label either -- it is a switch, not a fact about the figure, and the
    # default is the only readout these runs have.
    axes[0].set_ylabel("Rank IC")
    fig.tight_layout()
    out = save_figure(fig, a.out)
    if short:
        print("  partial months (fewer seeds than the fullest on the panel): "
              + "; ".join(short))
    n = sum(len(data.get((s, f"head:{t}" if a.readout == "head" else t)) or {})
            for s, t, _ in TASKS)
    print(f"wrote {', '.join(str(o) for o in out)} "
          f"({'no boxes yet' if not any_box else str(n) + ' month-series'})")


if __name__ == "__main__":
    main()
