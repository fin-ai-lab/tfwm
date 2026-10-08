"""Rank IC against training compute for the DECADE multihead runs.

THE FIGURE: three panels, one per target, x = training FLOPs (log), y = rank
IC on 2018-01, one curve per ViT scale. A SEPARATE figure from
supervised_scaling.png on purpose -- that one is 31 six-month specialists per
point, this one is a single ten-year run per curve, and overlaying them would
put two different experiments on one axis.

WHAT A POINT IS. One multihead trunk (return + volatility change + spread
change together) trained on the day store over 2008-01..2017-12 under a WSD
schedule, branching at each half-decade FLOPs target to cool down, save and
return. So every point is an annealed model at its own compute, and the whole
curve comes from ONE run per scale rather than one run per point.

NO ERROR BAND, AND THAT IS NOT AN OVERSIGHT. The top-level figure's band is
the s.e. ACROSS the 31 evaluation months. Here there is one run and one
evaluation month, so there is no such distribution to take an s.e. of. The
per-cell s.e. that post_train_ic_eval reports is a different quantity -- the
spread within a month's cross-sections -- and drawing it here would invite the
reader to compare it against the other figure's band as though they meant the
same thing.

THE SHADE IS THE SCALE, THE COLOR IS THE TARGET, exactly as in
supervised_scaling.py, so the two figures read as one family.

    uv run scripts/eval/collect_multihead_decade.py
    uv run plots/scaling/supervised_scaling_decade.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_PLOTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLOTS))
from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, add_bottom_legend,
    apply_style, save_figure, set_two_decimal_yticks)

METRICS_JSON = _PLOTS / "metrics/supervised_scaling_decade.json"
REFERENCE_JSON = _PLOTS / "metrics/supervised_decade_reference.json"
OUT = Path(__file__).resolve().parent / "supervised_scaling_decade.png"

PANELS = (
    ("return_900", "Return", "supervised_return"),
    ("volatility_change_900", "Volatility Change", "supervised_vol"),
    ("spread_change_900", "Spread Change", "supervised_spread"),
)
# Same keys, markers and lightness shifts as supervised_scaling.py.
SCALES = (
    ("tiny",  "ViT-Tiny",  "v", +0.45),
    ("small", "ViT-Small", "o",  0.0),
    ("base",  "ViT-Base",  "^", -0.45),
)
LEGEND_GREY = "#555555"
PARAMS_M = {"tiny": 5.4, "small": 21.0, "base": 86.0}


def shade(color: str, shift: float) -> tuple:
    r, g, b = mcolors.to_rgb(color)
    if shift >= 0:
        return tuple(c + (1.0 - c) * shift for c in (r, g, b))
    return tuple(c * (1.0 + shift) for c in (r, g, b))


def reference(path: Path) -> dict | None:
    """The 6-month multihead on this same eval month, or None.

    THE STAR IS A REAL MEASUREMENT, not an interpolation. Unlike the star on
    supervised_scaling.png -- which sits ON the curve because the reported
    model's score is averaged over different months and is not comparable
    point-for-point -- this one is the SAME recipe scored on the SAME month
    (2018-01) with the same head readout, so both its compute and its IC are
    directly comparable and it is drawn where it actually landed.

    Its x carries the same x3 multihead factor as the curves, so "the
    multihead costs three times a single-task model" is already in both.
    """
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if d.get("flops") and d.get("ic_head") else None


def series(rows: list[dict], task: str, scale: str):
    """(xs, ys) for one curve, ordered by compute, or None."""
    pts = sorted(
        ((float(r["flops"]), float(r["ic_head"])) for r in rows
         if r["task"] == task and r["scale"] == scale),
        key=lambda p: p[0])
    if not pts:
        return None
    return [p[0] for p in pts], [p[1] for p in pts]


def draw(rows: list[dict], out: Path, ref: dict | None = None) -> bool:
    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.36))
    drew: set[str] = set()
    drew_ref = False
    for ax, (task, title, style_key) in zip(axes, PANELS):
        color = SERIES_STYLES[style_key]["color"]
        for key, label, marker, shift in SCALES:
            got = series(rows, task, key)
            if got is None:
                continue
            xs, ys = got
            drew.add(key)
            ax.plot(xs, ys, marker=marker, ls="-", color=shade(color, shift),
                    ms=2.6, lw=1.1, zorder=3)
        if ref is not None and ref["ic_head"].get(task) is not None:
            drew_ref = True
            ax.plot([ref["flops"]], [ref["ic_head"][task]], marker="*", ms=10,
                    color=color, mec="white", mew=0.7, ls="none", zorder=5)
        ax.set_title(title, pad=4)
        ax.set_xscale("log")
        ax.grid(False)
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
        set_two_decimal_yticks(ax, nbins=5)
    if not drew:
        plt.close(fig)
        return False
    axes[0].set_ylabel("Rank IC")
    axes[1].set_xlabel("Training FLOPs")

    handles, labels = [], []
    for key, label, marker, shift in SCALES:
        if key not in drew:
            continue
        handles.append(Line2D([], [], ls="-", lw=1.1, marker=marker, ms=2.6,
                              color=shade(LEGEND_GREY, shift)))
        n = PARAMS_M.get(key)
        labels.append(f"{label} ({n:.0f}M)" if n and n >= 10
                      else f"{label} ({n:.1f}M)" if n else label)
    if drew_ref:
        handles.append(Line2D([], [], marker="*", ms=10, ls="none",
                              color=LEGEND_GREY, mec="white", mew=0.7))
        labels.append(ref.get("label", "6-Month Multihead"))
    fig.tight_layout()
    add_bottom_legend(fig, handles, labels, ncol=len(labels),
                      columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, out)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, default=METRICS_JSON)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--reference", type=Path, default=REFERENCE_JSON,
                   help="the 6-month multihead on this eval month; the star")
    p.add_argument("--no-reference", action="store_true", help="leave the star off")
    a = p.parse_args()
    rows = json.loads(a.metrics.read_text())
    if not rows:
        raise SystemExit(f"{a.metrics} is empty")
    ref = None if a.no_reference else reference(a.reference)
    if not draw(rows, a.out, ref):
        raise SystemExit("nothing to draw")
    for scale in sorted({r["scale"] for r in rows}):
        n = len({r["step"] for r in rows if r["scale"] == scale})
        print(f"  {scale:6s} {n} rung(s)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
