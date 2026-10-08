"""The supervised family's rank IC on all three targets, from the probe sweep.

WHY THIS STOPPED BEING A HORIZON FIGURE. The old version drew each specialist
as a line across six forward horizons, read from ``xs_ic/<target>_<h>`` in the
checkpoints' own ``xs_ic.json``. The wave those series now point at
(``SUP_SPAN_COMMIT = 581eb2``, the six-month-span recipe) ran under
``POST_TRAIN_PROBE=0``: its checkpoints carry ``xs_ic/head:return_900`` and
NOTHING ELSE, so the probe lines resolved 31 runs over 0 eval months and the
figure rendered as three lone head stars. The horizon axis is not recoverable
from the probe-breadth sweep either -- ``probe_fit_size.py`` pins
``TASKS = [return_900, volatility_change_900, spread_change_900]``, so every
number in that sweep is the 15-minute horizon by construction.

SO THE X AXIS IS THE FIT POOL, not the horizon. That is the one axis this data
varies along, and it is the same axis probe_fit_breadth.py draws. Restoring
the horizon version means re-scoring the 93 pinned checkpoints (31 months x 3
specialists) with the ridge probe enabled; until then a horizon axis would be
five-sixths empty.

WHAT IT SHOWS THAT probe_fit_breadth DOES NOT. That figure asks whose FEATURES
carry the signal and puts one supervised arm against LeJEPA. This one puts ALL
FOUR supervised arms on ALL THREE panels, so the off-diagonal is visible: what
a trunk trained on spread does for volatility, and whether the multihead pays
for its generality. Those are exactly the off-diagonal cells of
Table~\\ref{tab:probe_fit}, drawn as curves.

NO DRAWN-ON VALUE LABELS. The old figure printed "+0.0274 +/- 0.0033" inside
each panel because a single star had no other way to say what it was. With
four curves per panel the numbers belong in the table, not on the axes.

Furniture follows all_metrics_finance_ic: bands at +/- 1 SE across eval
months, one bottom legend, "Rank IC" on the left panel, one x label under the
middle, no suptitle.

    uv run plots/metrics/all_metrics_supervised_ic.py [--alpha 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "plots/core"))
from style import (COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL,  # noqa: E402
                   add_bottom_legend, apply_style, save_figure,
                   set_two_decimal_yticks)
# THE LOADER IS THE BREADTH FIGURE'S, imported rather than copied: "same data
# as probe_fit_breadth" has to mean the same rows, the same alpha filter and
# the same every-month-reached-this-rung rule, or the two figures can disagree
# about a number the table also prints.
from probe_fit_breadth import _series, load_curves, load_head  # noqa: E402

PANELS = [
    ("return_900", "Return"),
    ("volatility_change_900", "Volatility Change"),
    ("spread_change_900", "Spread Change"),
]

# The supervised family, in the table's order. The style key carries the
# paper's color for that arm, so panel, curve and head star cannot drift.
ARMS = [
    ("sup_return_w8", "Supervised (return)", "supervised_return"),
    ("sup_vol_w8", "Supervised (vol)", "supervised_vol"),
    ("sup_spread_w8", "Supervised (spread)", "supervised_spread"),
    ("sup_multi_w8", "Supervised (multihead)", "multihead"),
]

# Which arm trained a head on which target. A specialist has one; the
# multihead has all three off one trunk.
HEAD_ON = {
    "sup_return_w8": {"return_900"},
    "sup_vol_w8": {"volatility_change_900"},
    "sup_spread_w8": {"spread_change_900"},
    "sup_multi_w8": {"return_900", "volatility_change_900",
                     "spread_change_900"},
}

STAR_GRAY = "#555555"   # the legend swatch only; panels use the arm's color


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                        / "all_metrics_supervised_ic.png"))
    a = p.parse_args()

    curves = load_curves(a.alpha)
    head = load_head()

    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.36))

    handles: list = []
    labels: list[str] = []
    any_data = False
    months_seen: set[str] = set()

    for ax, (task, title) in zip(axes, PANELS):
        for key, label, style in ARMS:
            color = SERIES_STYLES[style]["color"]
            got = _series(curves, key, task)
            if got is None:
                continue
            any_data = True
            ns, mu, se, months, tops = got
            months_seen.update(months)

            (ln,) = ax.plot(ns, mu, "-o", color=color, ms=2.4, lw=1.1,
                            zorder=3)
            ax.fill_between(ns, mu - se, mu + se, color=color, alpha=0.16,
                            lw=0)
            if label not in labels:
                handles.append(ln)
                labels.append(label)

            # THE FULL POOL IS ITS OWN POINT, hollow, at the MEAN of the
            # per-month maxima: no single n is common to every month up there,
            # so joining it to the ladder with a solid segment would claim a
            # rung that was never fit. The dotted stub says "same curve,
            # x is an average".
            if tops:
                xs = float(np.mean([t[0] for t in tops]))
                ys = float(np.mean([t[1] for t in tops]))
                ax.plot([ns[-1], xs], [mu[-1], ys], ":", color=color, lw=0.9,
                        zorder=2)
                ax.plot([xs], [ys], "o", mfc="none", mec=color, ms=4.2,
                        mew=1.0, zorder=4)

                # The trained head, where this arm has one on this target.
                # Drawn at the full-pool x so it sits beside the probe value
                # it is the ceiling for.
                if task in HEAD_ON.get(key, ()):
                    hv = head.get((key, task), {})
                    hv = [v for m, v in hv.items() if m in set(months)]
                    if hv:
                        ax.plot([xs], [float(np.mean(hv))], "*", color=color,
                                ms=7.0, mec="black", mew=0.5, zorder=5)

        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, lw=0.4)
        set_two_decimal_yticks(ax)

    if not any_data:
        print("no probe-breadth results yet", file=sys.stderr)
        return 1

    axes[0].set_ylabel("Rank IC")
    axes[1].set_xlabel("Rows the Ridge Was Fit On")

    # The star and the hollow full-pool key are neutral in the legend: one
    # swatch cannot be purple, green, orange and red at once, and the panel
    # supplies the color.
    handles.append(plt.Line2D([], [], ls="none", marker="*", color=STAR_GRAY,
                              mec="black", mew=0.5, ms=7.0))
    labels.append("Trained Head")
    handles.append(plt.Line2D([], [], ls="none", marker="o", mfc="none",
                              mec=STAR_GRAY, mew=1.0, ms=4.2))
    labels.append("Full Pool (Mean N)")

    # tight_layout FIRST -- add_bottom_legend reserves the bottom band with
    # subplots_adjust and tight_layout would overwrite it. ncol=3 over 6
    # entries gives the two-row legend all_metrics_finance_ic uses, and
    # add_bottom_legend infers the larger 2-row reserve from that ratio.
    fig.tight_layout()
    add_bottom_legend(fig, handles, labels, ncol=3, columnspacing=1.2,
                      handletextpad=0.5)
    save_figure(fig, Path(a.out))
    print(f"wrote {a.out} ({len(ARMS)} arm(s) over "
          f"{len(months_seen)} month(s), alpha {a.alpha:g})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
