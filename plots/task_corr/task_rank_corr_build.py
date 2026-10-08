"""Progressive-reveal frames of both task_rank_corr figures, for beamer \\only.

Each published figure lands all at once; a talk wants to build it up. This
writes each of them several times with a growing subset of the content, on a
canvas that never moves, so consecutive \\only overlays read as a reveal.

THE MATRIX -- ``task_rank_corr_build{1..4}``. The lower triangle of the 9x9,
revealed one block-pair at a time. All the axis furniture (row/column labels,
family brackets, colorbar) is drawn in full on every frame; only the CELLS
appear, so the slide can say what the audience is about to see before it is
there:

    _build1   Prediction x Prediction         (the forecasting tasks agree)
    _build2   + Organization x Prediction
    _build3   + Organization x Organization
    _build4   everything (identical to task_rank_corr)

THE SCATTER -- ``task_rank_corr_split_org_build{1..4}``, the ORGANIZATION
variant of ``plot_split``, revealed method by method:

    _build1   Multihead only
    _build2   + I-JEPA
    _build3   + Time Warp, BYOL
    _build4   everything (identical to task_rank_corr_split_org)

EVERY FRAME IS THE SAME CANVAS. For the scatter that takes some care: axes
limits, ticks, legend and -- the part that actually matters -- the label
OFFSETS are computed ONCE from the full all-points layout and reused
verbatim, so a point that appears in frame 2 sits in exactly the same place
in frames 3 and 4 and nothing slides around under the overlay. The legend is
drawn in full on every frame for the same reason. Consequently an
intermediate frame is not the figure the placement algorithm would produce
for its own subset, and is not meant to be: only the last frame of each
series stands alone, and each is byte-identical to the published figure.

Reads the saved table only (task_rank_corr.json); no scores are loaded, so
regenerate that first if the numbers moved.

Run:
    uv run plots/task_corr/task_rank_corr_build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# task_rank_corr sits beside this file (plots/task_corr). `style` resolves off
# that module's own sys.path insert, which runs on the import below.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_rank_corr import (  # noqa: E402
    FAM_COLOR, GROUP_COLOR, HERE, METHODS, PRED_GROUP, crit_rho,
    point_label,
)
from style import apply_style, save_figure  # noqa: E402

# THE SLIDE CANVAS. task_rank_corr's paper figures are now sized to sit beside
# probe_fit_breadth; these frames keep the slide layout their margins and font
# sizes below were tuned for.
FIG_SIZE = (6.6, 4.7)

# Matrix reveal order, as (row group, column group) pairs. Cumulative, and
# the last stage is the catch-all: whatever is left in the lower triangle.
ORG, FAC = "Organization", "Factor Structure"
MATRIX_STAGES: list[tuple[tuple[str, str], ...] | None] = [
    ((PRED_GROUP, PRED_GROUP),),
    ((ORG, PRED_GROUP),),
    ((ORG, ORG),),
    None,                                    # the rest
]

# The reveal order, by METHODS index (ROSTER order, Random ViT last).
# Cumulative: each stage adds to the last.
STAGES: list[tuple[int, ...]] = [
    (8,),           # Supervised (multi)  -> "Multihead"
    (13,),          # I-JEPA
    (1, 9),         # LeJEPA +time warp -> "Time Warp", and BYOL
    (),             # the rest
]
Y_GROUPS = ("Organization",)
YLABEL = f"Organization Rank  (1 = best of {len(METHODS)})"

# Label slots and placement weights, copied from plot_split so the frames
# inherit the published layout exactly. Keep in sync if that one is retuned.
CANDIDATES = [(0, 10), (0, -12), (14, 3), (-14, 3), (14, -7), (-14, -7),
              (0, 21), (0, -23), (28, 0), (-28, 0), (24, 12), (-24, 12)]
PAD = 8.0


def matrix_frame(d: dict, out, shown: set[tuple[int, int]]) -> None:
    """``task_rank_corr.plot`` with only the cells in `shown` drawn.

    A copy of that function, not a call into it: every line below except the
    two marked ones is verbatim, and the frames are checked against the
    published PNG (see main) so the copy cannot drift unnoticed. Factoring a
    `shown` argument into the paper figure itself was the alternative and was
    rejected -- that function's whole contract is that there is exactly one
    matrix to quote and no flag that produces a different one.

    Hidden cells are simply absent (masked -> transparent, no text), the same
    way the upper triangle is absent in the published figure, so nothing on
    the frame reads as a cell whose value happens to be blank.
    """
    import matplotlib.pyplot as plt

    apply_style()
    names, groups = d["tasks"], d["groups"]
    rho = np.asarray(d["rho"])
    n = len(names)
    crit = crit_rho(d["n_methods"])

    mask = np.triu(np.ones((n - 1, n - 1), bool), k=1)
    for i in range(n - 1):                          # +++ hide unrevealed
        for j in range(i + 1):
            if (i, j) not in shown:
                mask[i, j] = True
    M = np.ma.masked_array(rho[1:, :-1], mask=mask)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.335, right=0.871, bottom=0.202, top=0.988)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)

    for i in range(n - 1):
        for j in range(i + 1):
            if (i, j) not in shown:                 # +++ hide unrevealed
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.75 else "#111111",
                    fontweight="bold" if abs(v) >= crit else "normal")

    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(n - 1))
    ax.set_xticklabels(names[:-1], rotation=38, ha="right", fontsize=9.5)
    ax.set_yticklabels(names[1:], fontsize=9.5)
    for tick, g in zip(ax.get_xticklabels(), groups[:-1]):
        tick.set_color(GROUP_COLOR[g])
    for tick, g in zip(ax.get_yticklabels(), groups[1:]):
        tick.set_color(GROUP_COLOR[g])

    for e in [i for i in range(1, n) if groups[i] != groups[i - 1]]:
        ax.plot([e - 0.5, e - 0.5], [e - 1.5, n - 1.5], color="white", lw=3.5)
        ax.plot([-0.5, e - 0.5], [e - 1.5, e - 1.5], color="white", lw=3.5)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(rend)
    lab_x0 = min(t.get_window_extent(rend).x0 for t in ax.get_yticklabels())
    edge = (lab_x0 - ax_bb.x0) / ax_bb.width
    tr = ax.get_yaxis_transform()
    rows = groups[1:]
    starts = [i for i in range(len(rows)) if i == 0 or rows[i] != rows[i - 1]]
    for a in starts:
        b = next((i for i in range(a + 1, len(rows) + 1)
                  if i == len(rows) or rows[i] != rows[a]), len(rows))
        c = GROUP_COLOR[rows[a]]
        ax.plot([edge - 0.025, edge - 0.025], [a - 0.42, b - 1 + 0.42],
                transform=tr, color=c, lw=2.2, clip_on=False,
                solid_capstyle="butt")
        ax.text(edge - 0.055, (a + b - 1) / 2, rows[a], transform=tr, color=c,
                fontsize=10, rotation=90, ha="center", va="center")

    ax.set_xticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    cax = fig.add_axes([0.892, 0.36, 0.024, 0.46])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(r"Rank Correlation $\rho$", fontsize=9.5)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_visible(False)

    paths = save_figure(fig, out, bbox_inches=None)
    plt.close(fig)
    print("    wrote " + ", ".join(str(p) for p in paths))


def build_matrix(d: dict) -> None:
    """Write the matrix frames. Cell (i, j) is the pair (task i+1, task j)."""
    groups = d["groups"]
    n = len(groups)
    lower = [(i, j) for i in range(n - 1) for j in range(i + 1)]
    shown: set[tuple[int, int]] = set()
    stem = HERE / "task_rank_corr"
    for k, pairs in enumerate(MATRIX_STAGES, start=1):
        if pairs is None:
            shown = set(lower)
        else:
            shown |= {(i, j) for i, j in lower
                      if (groups[i + 1], groups[j]) in pairs}
        print(f"matrix build{k}: {len(shown):2d} cells")
        matrix_frame(d, stem.with_name(f"{stem.name}_build{k}"), shown)


def _figure(x, y, families, show: list[int], offsets: dict | None):
    """Draw one frame; return (fig, ax, texts, legend) with `show` plotted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.115, right=0.962, bottom=0.160, top=0.964)
    for fam, c in FAM_COLOR.items():
        m = [i for i in show if families[i] == fam]
        # Empty families still call scatter: the legend must carry all three
        # entries on every frame, including the first one where two of them
        # have no point yet.
        ax.scatter(x[m], y[m], s=80, c=c, edgecolor="white", linewidth=0.9,
                   label=fam, zorder=3)

    texts = {i: ax.annotate(point_label(METHODS[i]), (x[i], y[i]),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", va="bottom", fontsize=9.5, zorder=4,
                            color=FAM_COLOR[families[i]])
             for i in show}

    n = len(METHODS)
    lo, hi = 0.2, n + 0.8
    ax.set_xlim(hi, lo)          # inverted: rank 1 on the RIGHT
    ax.set_ylim(hi, lo)          # inverted: rank 1 at the TOP
    ax.set_xlabel(f"Forecasting Rank  (1 = best of {n})", fontsize=12)
    ax.set_ylabel(YLABEL, fontsize=12)
    ax.set_xticks([1, 5, 10, 15, n])
    ax.set_yticks([1, 5, 10, 15, n])
    ax.tick_params(labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(fontsize=10, frameon=True, loc="lower left",
                    bbox_to_anchor=(0.015, 0.015), handletextpad=0.4,
                    borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(5)

    if offsets is not None:
        for i, (dx, dy) in offsets.items():
            if i in texts:
                _apply(texts[i], dx, dy)
    return fig, ax, texts, leg


def _apply(t, dx, dy):
    t.set_position((dx, dy))
    t.set_ha("center" if dx == 0 else ("left" if dx > 0 else "right"))
    t.set_va("bottom" if dy >= 0 else "top")


def solve_offsets(x, y, families) -> dict[int, tuple[int, int]]:
    """Run plot_split's slot search on the FULL roster; return its offsets."""
    from matplotlib.transforms import Bbox
    import matplotlib.pyplot as plt

    n = len(METHODS)
    fig, ax, texts, leg = _figure(x, y, families, list(range(n)), None)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([x, y]))
    marks = [Bbox([[px - PAD, py - PAD], [px + PAD, py + PAD]])
             for px, py in pts]
    obstacles = marks + [leg.get_window_extent(rend)]
    ax_bb = ax.get_window_extent(rend)

    def cost(box, placed, dx, dy):
        c = 3.0 * sum(box.overlaps(o) for o in obstacles)
        c += 3.0 * sum(box.overlaps(b) for b in placed)
        c += 2.0 * (box.x0 < ax_bb.x0 or box.x1 > ax_bb.x1
                    or box.y0 < ax_bb.y0 or box.y1 > ax_bb.y1)
        c += 0.35 * ((dx ** 2 + dy ** 2) ** 0.5) / 10.0
        return c

    placed: list = []
    order = sorted(range(n), key=lambda i: -sum(
        1 for j in range(n)
        if j != i and abs(x[i] - x[j]) < 2.5 and abs(y[i] - y[j]) < 2.5))
    out: dict[int, tuple[int, int]] = {}
    for i in order:
        best, best_c = CANDIDATES[0], None
        for dx, dy in CANDIDATES:
            _apply(texts[i], dx, dy)
            c = cost(texts[i].get_window_extent(rend), placed, dx, dy)
            if best_c is None or c < best_c:
                best, best_c = (dx, dy), c
        _apply(texts[i], *best)
        placed.append(texts[i].get_window_extent(rend))
        out[i] = best
    plt.close(fig)
    return out


def main() -> int:
    import matplotlib.pyplot as plt

    apply_style()
    d = json.loads((HERE / "task_rank_corr.json").read_text())

    build_matrix(d)

    R = np.asarray(d["ranks"])
    groups, families = d["groups"], d["families"]
    pred = [i for i, g in enumerate(groups) if g == PRED_GROUP]
    other = [i for i, g in enumerate(groups) if g in Y_GROUPS]
    if not other:
        raise SystemExit(f"no task in groups {Y_GROUPS}")
    x, y = R[:, pred].mean(1), R[:, other].mean(1)

    offsets = solve_offsets(x, y, families)

    stem = HERE / "task_rank_corr_split_org"
    show: list[int] = []
    rest = [i for i in range(len(METHODS))
            if i not in {i for s in STAGES for i in s}]
    for k, add in enumerate(STAGES, start=1):
        show += list(add) if add else rest
        fig, *_ = _figure(x, y, families, show, offsets)
        paths = save_figure(fig, stem.with_name(f"{stem.name}_build{k}"),
                            bbox_inches=None)
        plt.close(fig)
        print(f"build{k}: {len(show):2d} pts "
              f"({', '.join(point_label(METHODS[i]) for i in show)})\n"
              "    wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
