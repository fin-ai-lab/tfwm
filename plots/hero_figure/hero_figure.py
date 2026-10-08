"""The hero figure: augmentation -> learned latent space -> task performance.

One row per LeJEPA arm (Same Stock, Diff. View; Time Warping; Cross-Stock
Same Industry), three columns joined by arrows, each drawn by its source
script's own code onto this figure's axes (nothing is pasted in as an image):

  1. The arm's second view over the base view's VWAP (grey):
     plots/lejepa_augmentations/lejepa_augmentations.py (``gallery``,
     ``candles``), same day, seed and recipe as that figure.
  2. One fixed panel of the latent table, embedded by the arm and t-SNE'd
     (plots/hero_figure/tsne_panel.py): firm by colour, day by marker
     shape; each day's 2-D centroid is a grey marker in that day's shape,
     drawn on top.
     The panel and 5-day block were picked as the ones whose maps separate
     day centroids most cleanly.
  3. Where the arm ranks among the self-supervised methods the paper's
     tables report (LeJEPA + SSL; rank 1 = longest bar), task by task:
     Day Clustering and Firm Clustering are latent tasks T2 and T3 of
     plots/core/fixed_panel_table.tex (mean-rank percentile, lower is
     better); Return, Volatility Change and Spread Change are the full-pool
     probe ICs of plots/core/probe_fit_table.tex. Both read through the
     table scripts' own loaders, so the ranks and the tables cannot disagree.

A second, separate figure (hero_return_scaling) is the supervised
specialist's Return panel from plots/scaling/supervised_scaling.py, sized to
be wrapped into a corner of the page.

Column 2 reads the embedding cache tsne_panel.py writes (one GPU forward per
arm and month on a miss) and the panel cache under panel_lib.CACHE_DIR.

Run (from the repo root):
    uv run python plots/hero_figure/hero_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE.parent
for p in (HERE, PLOTS / "lejepa_augmentations", PLOTS / "scaling",
          PLOTS / "core"):
    sys.path.insert(0, str(p))

import lejepa_augmentations as aug  # noqa: E402
import supervised_scaling as sc  # noqa: E402
import tsne_panel as tp  # noqa: E402
from market_jepa.schemas import LocalMachineConfig  # noqa: E402
from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, apply_style, save_figure)

OUT = HERE / "hero_figure"
OUT_SCALING = HERE / "hero_return_scaling"

# One row per arm: (augmentations-figure key, latent/probe series, title).
ROWS = [("same_stock", "pair_rrc_6mo", "Same Stock, Diff. View"),
        ("time_warp", "pair_warp_6mo", "Time Warping"),
        ("cross_stock_industry", "pair_k2ind_6mo",
         "Cross-Stock Same Industry")]
# Column 2: picked for cleanly separated day centroids, every map readable.
MONTH, PANEL, N_DAYS, WINDOW = "2010-01", 2, 5, 1
# As tsne_panel.py; run_tsne caps it at (n-1)/3, ~10 for 30 points.
PERPLEXITY = 30.0
# One title size for every panel, and one size for the words standing in
# for tick labels ("Norm. Price", "Rank IC", "Training FLOPs").
TITLE_SIZE = 8
AXIS_WORD_SIZE = 8
# Column 3's bars, top to bottom: (label, kind, key, colour key). Latent
# tasks are named inside their bars (the caption maps them: Day Clustering =
# T2, Firm Clustering = T3), as are the predictive ones.
TASKS = [("Time Org.", "latent", "metric2"),
         ("Firm Org.", "latent", "metric3"),
         ("Return", "probe", "return_900"),
         ("Volatility Change", "probe", "volatility_change_900"),
         ("Spread Change", "probe", "spread_change_900")]
# Latent tasks get their own colours, clear of the three task colours and of
# LeJEPA's tab:blue elsewhere in the paper.
# What a bar says inside itself; a bar as short as #14 still holds the name
# because the name starts at the axis, not in the bar's middle.
SHORT_TASK = {"Time Org.": "Day Clustering",
              "Firm Org.": "Firm Clustering", "Return": "Return",
              "Volatility Change": "Volatility Change",
              "Spread Change": "Spread Change"}
LATENT_COLORS = {"metric2": "#17becf", "metric3": "#8c564b"}
PROBE_COLORS = {"return_900": SERIES_STYLES["supervised_return"]["color"],
                "volatility_change_900": SERIES_STYLES["supervised_vol"]["color"],
                "spread_change_900": SERIES_STYLES["supervised_spread"]["color"]}
# FF49 long names are too wide for a third of a page.
SHORT_INDUSTRY = {2: "Food", 5: "Tobacco", 9: "Household", 10: "Apparel",
                  12: "Medical Equipment", 13: "Pharma", 14: "Chemicals",
                  19: "Steel", 23: "Autos", 24: "Aerospace", 28: "Mining",
                  30: "Oil", 31: "Utilities", 32: "Media",
                  34: "Business Services", 35: "Computers", 36: "Software",
                  41: "Transport", 43: "Retail", 44: "Restaurants",
                  7: "Entertainment", 45: "Banks", 48: "Finance",
                  27: "Precious Metals"}


# ── tick-word helpers ────────────────────────────────────────────────────────

def y_word(ax, ticks, at, word):
    """Stand ``word`` in for the ``at`` y tick label, rotated, in place of a
    y-axis label."""
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "" if abs(v - at) < 1e-9
        else f"{v:g}".replace("-", "−")))
    ax.text(-0.05, at, word, rotation=90, ha="right", va="center",
            fontsize=AXIS_WORD_SIZE, transform=mtransforms.
            blended_transform_factory(ax.transAxes, ax.transData))


def rank_ic_label(ax, ticks):
    """Label only the end ticks and stand "Rank IC" between them, rotated,
    in place of a y-axis label."""
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}" if v in (ticks[0], ticks[-1]) else ""))
    ax.text(-0.05, (ticks[0] + ticks[-1]) / 2, "Rank IC", rotation=90,
            ha="right", va="center", fontsize=AXIS_WORD_SIZE,
            transform=mtransforms.blended_transform_factory(ax.transAxes,
                                                            ax.transData))


def axis_word(ax, at, word, log=True, size=None):
    """Stand ``word`` in for the ``at`` x tick label, in place of an x-axis
    label (the mirror of rank_ic_label)."""
    from matplotlib.ticker import LogFormatterSciNotation
    sci = LogFormatterSciNotation()
    sci.set_axis(ax.xaxis)
    same = ((lambda v: abs(np.log10(v) - np.log10(at)) < 1e-6) if log
            else (lambda v: abs(v - at) < 1e-9))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: "" if same(v) else (sci(v, pos) if log else f"{v:g}")))
    ax.text(at, -0.045, word, ha="center", va="top",
            fontsize=size or AXIS_WORD_SIZE,
            transform=mtransforms.blended_transform_factory(ax.transData,
                                                            ax.transAxes))


def small_legend(ax, handles, labels, loc="lower right"):
    ax.legend(handles, labels, loc=loc, fontsize=5.2, frameon=False,
              handlelength=1.6, handletextpad=0.4, borderaxespad=0.25,
              labelspacing=0.25)


# ── column 1: the views ──────────────────────────────────────────────────────

def draw_views(axes):
    """Each arm's second view over the base view's VWAP."""
    views = aug.gallery(aug.load_day(LocalMachineConfig.daystore_dir))
    base = views["base"]
    base_mid = (np.arange(aug.SEQ_LEN) + 0.5) / aug.SEQ_LEN
    for ax, (arm, _, title) in zip(axes, ROWS):
        entry = views[arm]
        ax.plot(base_mid, base["view"][:, aug.VWAP], color=aug.REFERENCE,
                lw=0.8, alpha=0.8)
        aug.candles(ax, np.linspace(0.0, 1.0, len(entry["view"]) + 1),
                    entry["view"])
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        note = (f"{base['ticker']} (grey) vs. {entry['ticker']}"
                if entry["ticker"] != base["ticker"] else
                f"{entry['ticker']}, local view" if arm == "same_stock"
                else entry["ticker"])
        # The augmentation's name, then what the view is, bottom right.
        back = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6)
        t = ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, bbox=back)
        # Stacked on the description's own box, not at a fixed axes
        # fraction, so the two never touch however short the panel gets.
        ax.annotate(title, xy=(1, 1), xycoords=t, xytext=(0, 1.5),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.5, bbox=back)
        # Ticks without numbers: the scale is per-view normalised, and at
        # this height the numbers collide with the rotated word.
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelleft=False)
        ax.text(-0.03, 0.5, "Norm. Price", rotation=90, ha="right",
                va="center", fontsize=AXIS_WORD_SIZE, transform=ax.transAxes)
    for ax in axes[1:]:
        ax.sharey(axes[0])
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axis_word(axes[-1], 0.5, "Normalized Time", log=False)


# ── column 2: the latent space ───────────────────────────────────────────────

def draw_tsne(axes, month, panel_idx, window, perplexity, industry=False):
    """The fixed panel under each arm, one N_DAYS-day block."""
    import torch
    tp.eg.apply_variant("mixed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = [key for _, key, _ in ROWS]
    embs = {key: tp.embed_month(month, key, device) for key in keys}
    bi, _, panel, _ = tp.score_panels(month, embs)[panel_idx]
    days = sorted(set(tp.panel_points(embs[keys[0]], bi, panel)[2]))
    keep = days[window * N_DAYS:(window + 1) * N_DAYS]
    colors, _ = tp.firm_colors(panel)
    for ax, key in zip(axes, keys):
        X, tks, dts = tp.panel_points(embs[key], bi, panel)
        m = np.isin(dts, keep)
        Z = tp.eg.run_tsne(X[m], perplexity=perplexity, seed=42)
        tp.draw(ax, Z, tks[m], dts[m], colors, "shapes", panel, lines=False,
                industry=industry)
    return panel, colors, keep


def firm_key(ax, panel, colors, loc="upper left", anchor=(0.0, 1.0),
             industry=False):
    """One entry per industry, its two firms' shades side by side, the
    industry on a second line so the key is narrow enough to sit inside the
    panel. Returned and re-added, so the day key can share the axes."""
    from matplotlib.legend_handler import HandlerTuple
    handles, labels = [], []
    for ff in sorted(set(panel.values())):
        pair = sorted(t for t, f in panel.items() if f == ff)
        handles.append(tuple(
            Line2D([], [], ls="", marker="o", ms=4.5, mfc=colors[t],
                   mec="white", mew=0.3) for t in pair))
        labels.append(f"{', '.join(pair)}\n({SHORT_INDUSTRY.get(ff, ff)})")
    if industry:
        handles.append(tuple(
            Line2D([], [], ls="", marker="*", ms=6, mew=0.4, mec="white",
                   mfc=colors[sorted(t for t, f in panel.items()
                                     if f == ff)[0]])
            for ff in sorted(set(panel.values()))))
        labels.append("Industry Centroid")
    leg = ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor,
                    fontsize=5.2, frameon=False, handletextpad=0.3,
                    borderaxespad=0.3, labelspacing=0.4,
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
                    handlelength=1.6)
    ax.add_artist(leg)
    return leg


# ── column 3: task performance ───────────────────────────────────────────────

def ssl_methods():
    """{probe-table series: latent key} for the self-supervised methods the
    two tables report (LeJEPA + SSL families; not supervised, not the TSFMs,
    not the floor). The latent key is the six-month name the fixed-panel
    table carries."""
    import probe_fit_table as pft
    out = {}
    for series, _, fam, _, _ in pft.roster():
        if fam in ("LeJEPA", "SSL"):
            six = [k for k in pft.series_aliases(series) if k.endswith("_6mo")]
            out[series] = six[0] if six else series
    return out


def latent_scores(keys):
    """{latent key: {metricK: mean-rank pctile}}, the reported fixed-panel
    table's own merge (fixed_panel_table.load_geo over DEFAULT_TAGS)."""
    import fixed_panel_table as fpt
    geo, _ = fpt.load_geo(fpt.DEFAULT_TAGS, required=False)
    return {k: {m: geo[k][m]["mean_pctile"] for m in fpt.METRICS}
            for k in keys if k in geo}


def probe_scores():
    """{series: {task: IC}} as probe_fit_table renders it by default
    (floor-covered months, alpha 10)."""
    import probe_fit_table as pft
    ics, ns, _, _ = pft.load_results(10.0)
    restrict = pft.comparable_months(ics, pft.TASKS)
    rows = pft.build_rows(ics, ns, pft.load_head(), min_months=1, fair=False,
                          restrict=restrict)
    return {r[0]: {t: c[0] for t, c in r[4].items()} for r in rows}


def ranks():
    """{latent key: [(label, kind, key, rank)]} and the number of methods.

    Rank 1 = best of the self-supervised methods on that task: lowest
    mean-rank percentile for a latent task, highest probe IC for a
    predictive one. Only methods scored on BOTH tables are ranked, so every
    bar is out of the same field.
    """
    methods = ssl_methods()
    lat = latent_scores(methods.values())
    ic = probe_scores()
    field = {s: k for s, k in methods.items() if k in lat and s in ic}
    out = {}
    for _, key, _ in ROWS:
        series = next(s for s, k in field.items() if k == key)
        bars = []
        for label, kind, k in TASKS:
            if kind == "latent":
                vals = {s: lat[field[s]][k] for s in field}
                r = 1 + sum(v < vals[series] for v in vals.values())
            else:
                vals = {s: ic[s][k] for s in field}
                r = 1 + sum(v > vals[series] for v in vals.values())
            bars.append((label, kind, k, r))
        out[key] = bars
    n = len(field)
    for key, bars in out.items():
        print(f"  {key:<15} " + "  ".join(f"{b[0]} #{b[3]}" for b in bars)
              + f"  (of {n})")
    return out, n


def task_color(kind, k):
    return LATENT_COLORS[k] if kind == "latent" else PROBE_COLORS[k]


def draw_bars(ax, bars, n):
    """Bar length n - rank + 1, so #1 is the longest; each bar tagged #k."""
    y = np.arange(len(bars))[::-1].astype(float)
    n_lat = sum(b[1] == "latent" for b in bars)
    y[:n_lat] += 0.5  # a half-row gap between latent and predictive
    # Limits first: the rank's position is measured in data units.
    ax.set_yticks([])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlim(0, n + 3.0)
    ax.set_xticks([])
    renderer = ax.figure.canvas.get_renderer()
    for yi, (label, kind, k, r) in zip(y, bars):
        length = n - r + 1
        ax.barh(yi, length, height=0.8, color=task_color(kind, k))
        # Name inside the bar, rank past its end -- unless the name would
        # overrun the bar, in which case the rank goes right after the bar
        # and the name after the rank.
        t = ax.text(0.3, yi, SHORT_TASK[label], ha="left", va="center",
                    fontsize=4.6, color="black")
        end = ax.transData.inverted().transform(
            t.get_window_extent(renderer).get_points())[1, 0]
        rank = ax.text(length + 0.25, yi, f"#{r}", ha="left", va="center",
                       fontsize=5.0, color="#333333")
        if end > length - 0.2:
            t.remove()
            ax.annotate(SHORT_TASK[label], xy=(1, 0.5), xycoords=rank,
                        xytext=(2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=4.6, color="black")


# ── the separate Return-scaling figure ───────────────────────────────────────

def draw_scaling(ax, panel=0, ticks=(0.01, 0.02, 0.03), title=None,
                 legend=True, xword=True, name=False):
    """One of supervised_scaling's panels (Return by default), legend
    inside, upper left (the curves rise left to right and leave it empty)."""
    rows = sc.load(sc.METRICS_JSON)
    floor = sc.floor_by_month()
    star = sc.default_model(sc.DEFAULT_MODEL_JSON)
    task, _, style_key = sc.PANELS[panel]
    color = SERIES_STYLES[style_key]["color"]
    handles, labels = [], []
    for key, label, marker, shift in sc.SCALES:
        got = sc.series(rows, task, key, "head", 0, floor)
        if got is None or not len(got[0]):
            continue
        xs, mu, se, _, _ = got
        c = sc.shade(color, shift)
        ax.plot(xs, mu, marker=marker, ls="-", color=c, ms=2.6, lw=1.1,
                zorder=3)
        ax.fill_between(xs, mu - 2 * se, mu + 2 * se, color=c, alpha=0.16,
                        lw=0)
        handles.append(Line2D([], [], ls="-", lw=1.1, marker=marker, ms=2.6,
                              color=sc.shade(sc.LEGEND_GREY, shift)))
        labels.append(label + sc.params_label(rows, key))
        if star is not None and star["scale"] == key:
            y = sc.on_curve(xs, mu, float(star["flops"]))
            if y is not None:
                ax.plot([star["flops"]], [y], marker="*", ms=9, color=c,
                        mec="white", mew=0.6, ls="none", zorder=5)
    if star is not None:
        handles.append(Line2D([], [], marker="*", ms=7, ls="none",
                              color=sc.LEGEND_GREY, mec="white", mew=0.6))
        labels.append(star.get("label", "Reported Model"))
    task_title = title
    panel_title = sc.PANELS[panel][1]
    if task_title is None:
        task_title = (f"{panel_title} Prediction" if panel_title == "Return"
                      else panel_title)
    if task_title:
        ax.set_title(task_title, pad=3, fontsize=TITLE_SIZE)
    if name:  # the task named inside, bottom right, as column 1 names arms
        ax.text(0.97, 0.04, panel_title, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5)
    ax.set_xscale("log")
    if xword:
        axis_word(ax, 1e17, "Training FLOPs")
    else:
        ax.tick_params(labelbottom=False)
    ax.grid(False)
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
    rank_ic_label(ax, list(ticks))
    if legend:
        small_legend(ax, handles, labels, loc="upper left")


def arrow(fig, ax_from, ax_to):
    """A figure-level arrow from one axes' right edge to the next's left --
    to the left of its tick labels, so the bars' task names stay clear."""
    a, b = ax_from.get_position(), ax_to.get_position()
    fig.canvas.draw()
    x1 = ax_to.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).x0
    y = (a.y0 + a.y1) / 2
    fig.patches.append(FancyArrowPatch(
        (a.x1 + 0.006, y), (min(b.x0, x1) - 0.006, y),
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#333333"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", default=MONTH)
    ap.add_argument("--panel", type=int, default=PANEL)
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="N_DAYS block of trading days (0 = days 1-5)")
    ap.add_argument("--perplexity", type=float, default=PERPLEXITY,
                    help="t-SNE perplexity; run_tsne caps it at (n-1)/3")
    ap.add_argument("--industry-centroids", action="store_true",
                    help="add each industry's t-SNE centroid as a star")
    ap.add_argument("--no-latent", action="store_true",
                    help="drop the Latent Space column (augmentation -> "
                         "rank); writes hero_figure_no_latent unless --out")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    apply_style(extra=COMPACT_RC_PARAMS)

    height = 3.4
    fig = plt.figure(figsize=(WIDTH_FULL, height))
    # Column 1 full width; columns 2 and 3 half of it, joined by arrows; a
    # heavy rule; then column 4, the supervised scaling, which is a separate
    # result and must not read as a fourth step of the rows.
    # Margins and gaps fixed in INCHES, so a height change only changes the
    # panels. The keys for columns 2-3 sit in the bottom margin.
    left, right = 0.05, 0.99
    top, bottom = 1 - 0.225 / height, 0.36 / height
    w1, w2 = 0.285, 0.1425
    w3 = w2
    gap = 0.035  # the arrows' gaps
    if args.no_latent:
        # The latent column's width and gap go mostly to the rank column:
        # with no t-SNE beside it, it is the result the row leads to.
        w1, w3 = 0.37, 0.285 + 2 * w2 + gap - 0.37
    x2 = left + w1 + gap
    x3 = x2 + w2 + gap if not args.no_latent else x2
    x_rule = x3 + w3 + 0.022
    x4 = x_rule + 0.06  # room for column 4's y labels
    w4 = right - x4
    hgap = 0.09 / height
    h = (top - bottom - 2 * hgap) / 3
    ys = [bottom + 2 * (h + hgap), bottom + h + hgap, bottom]
    col1 = [fig.add_axes([left, y, w1, h]) for y in ys]
    col2 = ([] if args.no_latent
            else [fig.add_axes([x2, y, w2, h]) for y in ys])
    col3 = [fig.add_axes([x3, y, w3, h]) for y in ys]
    hgap4 = 0.06 / height
    h4 = (top - bottom - hgap4) / 2
    col4 = [fig.add_axes([x4, bottom + h4 + hgap4, w4, h4]),
            fig.add_axes([x4, bottom, w4, h4])]

    draw_views(col1)
    col1[0].set_title("Augmentation", pad=3, fontsize=TITLE_SIZE)
    if not args.no_latent:
        panel, colors, keep = draw_tsne(col2, args.month, args.panel, args.window,
                                        args.perplexity, args.industry_centroids)
        col2[0].set_title("Latent Space", pad=3, fontsize=TITLE_SIZE)
        # One key only, under the column: the grey, black-edged marker is a day
        # centroid. Colour = firm and shape = day are left to the caption.
        col2[-1].legend(
            [Line2D([], [], ls="", marker="o", ms=4.5, mfc=tp.CENTROID_GREY,
                    mec="black", mew=0.6)], ["Day Centroid"],
            loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=6.5,
            frameon=False, handletextpad=0.3, borderaxespad=0.0)
    scores, n = ranks()
    for i, (ax, (_, key, _)) in enumerate(zip(col3, ROWS)):
        draw_bars(ax, scores[key], n)
    # The full version's rank column is half as wide, so it keeps the short
    # title.
    col3[0].set_title(f"Rank of {n} SSL" + (" Methods" if args.no_latent
                                             else ""),
                      pad=3, fontsize=TITLE_SIZE)
    if args.no_latent:
        for a, c in zip(col1, col3):
            arrow(fig, a, c)
    else:
        for a, b, c in zip(col1, col2, col3):
            arrow(fig, a, b)
            arrow(fig, b, c)

    draw_scaling(col4[0], panel=0, ticks=(0.01, 0.02, 0.03),
                 title="Supervised Scaling", xword=False, name=True)
    draw_scaling(col4[1], panel=1, ticks=(0.06, 0.08, 0.10), title="",
                 legend=False, name=True, xword=False)
    col4[1].tick_params(labelbottom=True)
    col4[1].sharex(col4[0])
    # A plain x label: in this narrower panel the tick-word collides with
    # the 10^16 / 10^18 labels.

    from matplotlib.ticker import FixedLocator, NullLocator
    for ax in col4:
        ax.xaxis.set_major_locator(FixedLocator([1e16, 1e17, 1e18]))
        ax.xaxis.set_minor_locator(NullLocator())
    # "Training FLOPs" stands in for the 10^17 label, a touch smaller than
    # the other tick words so it fits between 10^16 and 10^18.
    axis_word(col4[1], 1e17, "Training FLOPs", size=AXIS_WORD_SIZE - 1)
    # The rule spans the panels and their tick labels, no further: running
    # it to the figure edge kept bbox="tight" from trimming the margin.
    fig.add_artist(Line2D([x_rule, x_rule],
                          [bottom - 0.22 / height, 1 - 0.03 / height],
                          lw=1.6, color="#888888", solid_capstyle="round",
                          transform=fig.transFigure))
    out = args.out or (HERE / "hero_figure_no_latent" if args.no_latent
                       else OUT)
    for pth in save_figure(fig, out):
        print(f"Saved {pth}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.3, 1.9))
    draw_scaling(ax)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.17)
    for pth in save_figure(fig, OUT_SCALING):
        print(f"Saved {pth}")
    plt.close(fig)


if __name__ == "__main__":
    main()
