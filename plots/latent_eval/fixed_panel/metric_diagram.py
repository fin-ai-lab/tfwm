"""Explainer strip for the four fixed-panel metrics.

One real panel (3 industries x 2 firms, one view per firm-day, eval
month) embedded and t-SNE'd; four side-by-side panels annotate what
each metric asks:

  1  is a view's nearest OTHER-FIRM view its partner's view from the
     same day?
  2  is the nearest day-centroid the view's own day? (the focal
     firm's views are excluded from every day centroid, so all
     candidates are unbiased 5-view centroids)
  3  is the nearest firm-centroid the view's own firm? (the mirror of
     2: the focal DAY's views are excluded from every firm centroid,
     so all 6 candidates are ~20-day centroids over the same days)
  4  is a firm's month centroid nearest to its partner firm's centroid
     (among the other 5 firm centroids)?

Colors: one hue per industry, dark/light = the two partner firms (same
convention as the mixed cross-stock grid). Annotation geometry (arrows,
centroids) is computed IN THE 2-D PROJECTION so the picture is
self-consistent; the reported metrics are computed in embedding space.

Run:
    uv run plots/latent_eval/fixed_panel/metric_diagram.py
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import panel_lib as eg
from stable_finance.dataset import MarketSchedule
from market_jepa.schemas import LocalMachineConfig
from style import save_figure

# Inlined from cross_stock_geometry.py when that script was deleted
# (2026-08-26 latent_eval consolidation): dark/light shade per industry pair.
SECTOR_SHADES = [
    ("#1f5f9e", "#9ecae1"),
    ("#c62828", "#f4a6a6"),
    ("#2e7d32", "#a5d6a7"),
]

from fixed_panel_metrics import load_panel_batches  # noqa: E402
from industry_nn_sweep import (
    MODEL_SPECS, month_dates_suffix, resolve_run,
)

P, S = 3, 2
FADE = 0.30
INK = "#333333"

TITLES = [
    "1 · partner's same-day view", "2 · own-day centroid",
    "3 · own-firm centroid", "4 · partner firm centroid",
]
# --clean: same geometry, no captions and no suptitle — slide/figure
# version where the titles carry the whole explanation.
CLEAN_TITLES = [
    "Partner Matched View", "Own-Day Centroid",
    "Own-Firm Centroid", "Partner Firm Centroid",
]


def build_panel_points(ym, model_key, panel_idx, device, machine, schedule):
    """(Z, tks, dts, ind_of_firm) for one panel, t-SNE'd."""
    ev_month, ev_start, ev_end = eg.eval_window_t_plus_n(ym, 1)
    batch_sets = load_panel_batches(
        ev_month, ev_start, ev_end, P, S, machine, schedule,
    )
    panels = [p for ps, _ in batch_sets for p in ps]
    panel = panels[panel_idx]
    batches = batch_sets[0][1] if panel_idx < len(batch_sets[0][0]) else batch_sets[1][1]

    spec = MODEL_SPECS[model_key]
    run_dir = resolve_run(
        spec["project_glob"].format(dates=month_dates_suffix(ym)),
        spec["run_name"],
    )
    backbone = eg._load_backbone(run_dir, run_dir, run_dir.parent.name,
                                 pool=eg.LATENT_POOL).to(device).eval()
    try:
        fwd = eg.forward_cached(backbone, batches, device, cap=10**9)
    finally:
        backbone.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tk = np.asarray([str(t) for t in fwd["tickers"]], dtype=object)
    dt = np.asarray([str(d) for d in fwd["dates"]], dtype=object)
    in_panel = np.isin(tk, sorted(panel))
    keep = np.zeros(len(tk), dtype=bool)
    for d in sorted(set(dt)):
        sel = in_panel & (dt == d)
        if sel.sum() == P * S:
            keep |= sel
    X, tks, dts = fwd["X"][keep], tk[keep], dt[keep]
    Z = eg.run_tsne(X, perplexity=30.0, seed=42)
    return Z, tks, dts, panel


def firm_colors(panel):
    """{firm: color}: hue per industry (fixed code order), dark/light pair."""
    by_ind = {}
    for t, ff in sorted(panel.items()):
        by_ind.setdefault(ff, []).append(t)
    colors = {}
    for hue_i, ff in enumerate(sorted(by_ind)):
        dark, light = SECTOR_SHADES[hue_i % len(SECTOR_SHADES)]
        pair = sorted(by_ind[ff])
        colors[pair[0]] = dark
        colors[pair[1]] = light
    return colors


def base_scatter(ax, Z, tks, colors, alpha=FADE, size=14):
    for f in sorted(set(tks)):
        m = tks == f
        ax.scatter(
            Z[m, 0], Z[m, 1], s=size, marker="o",
            facecolor=colors[f], edgecolor="none", alpha=alpha,
        )
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def arrow(ax, src, dst, color=INK):
    ax.annotate(
        "", xy=dst, xytext=src,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        shrinkA=4, shrinkB=4),
        zorder=5,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", default="2023-03")
    p.add_argument("--model", default="k2ind")
    p.add_argument("--panel", type=int, default=-1,
                   help="panel index; -1 = first panel where all three "
                        "metrics have a hit in the 2-D projection")
    p.add_argument("--device", default="auto")
    p.add_argument("--clean", action="store_true",
                   help="titles only: drop the captions and the suptitle, "
                        "write fixed_panel_metric_diagram_clean")
    args = p.parse_args()

    eg.apply_variant("mixed")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    machine = LocalMachineConfig()
    schedule = MarketSchedule(machine.holiday_csv)

    def layout_for_panel(panel_idx):
        Z, tks, dts, panel = build_panel_points(
            args.month, args.model, panel_idx, device, machine, schedule,
        )
        n = len(Z)
        same_firm = tks[:, None] == tks[None, :]
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)

        # M1 focal: nearest other-firm view is the partner's same-day view
        d2f = d2.copy()
        d2f[same_firm] = np.inf
        nn1 = d2f.argmin(1)
        partner_of = {
            f: next(g for g, ff in panel.items() if g != f and ff == panel[f])
            for f in panel
        }
        m1_hits = [
            i for i in range(n)
            if tks[nn1[i]] == partner_of[tks[i]] and dts[nn1[i]] == dts[i]
        ]

        # M2: day centroids — the focal FIRM's views are excluded from
        # EVERY centroid (matching day_centroid_hits), so all ~21
        # candidates are 5-view centroids and none is biased toward the
        # focal point.
        days = sorted(set(dts))
        m2_hits = []
        for i in range(n):
            not_own_firm = tks != tks[i]
            cents = {
                d: Z[not_own_firm & (dts == d)].mean(0) for d in days
            }
            dd = {d: ((c - Z[i]) ** 2).sum() for d, c in cents.items()}
            if min(dd, key=dd.get) == dts[i]:
                m2_hits.append((i, cents))

        # M3: firm centroids with the focal point's DAY excluded from
        # every one (the mirror of M2) — nearest is the own firm?
        firms = sorted(set(tks))
        m3_hits = []
        for i in range(n):
            other_day = dts != dts[i]
            cents = {f: Z[other_day & (tks == f)].mean(0) for f in firms}
            dd = {f: ((c - Z[i]) ** 2).sum() for f, c in cents.items()}
            if min(dd, key=dd.get) == tks[i]:
                m3_hits.append((i, cents))

        # M4: firm centroids (full month) — nearest is the partner?
        F = {f: Z[tks == f].mean(0) for f in firms}
        m4_hits = []
        for f in firms:
            dd = {g: ((F[g] - F[f]) ** 2).sum() for g in firms if g != f}
            if min(dd, key=dd.get) == partner_of[f]:
                m4_hits.append(f)

        print(f"{args.month} panel {panel_idx} [{args.model}]: "
              f"m1 hits {len(m1_hits)}, m2 hits {len(m2_hits)}, "
              f"m3 hits {len(m3_hits)}, m4 hits {len(m4_hits)}")
        return (Z, tks, dts, panel, partner_of, nn1,
                m1_hits, m2_hits, m3_hits, m4_hits, firms, F)

    candidates = [args.panel] if args.panel >= 0 else list(range(10))
    chosen = None
    for panel_idx in candidates:
        chosen = layout_for_panel(panel_idx)
        if all(chosen[i] for i in (6, 7, 8, 9)):
            break
    (Z, tks, dts, panel, partner_of, nn1,
     m1_hits, m2_hits, m3_hits, m4_hits, firms, F) = chosen
    colors = firm_colors(panel)
    if not all((m1_hits, m2_hits, m3_hits, m4_hits)):
        print("WARNING: no panel had hits on all four — drawing best effort")

    def caption(ax, text):
        if args.clean:
            return
        ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=5.8,
                color=INK, ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.5))

    titles = CLEAN_TITLES if args.clean else TITLES

    def title(ax, k):
        ax.set_title(titles[k], fontsize=9 if args.clean else 7.5, pad=4)

    fig, axes = plt.subplots(
        1, 4, figsize=(eg.WIDTH_FULL, 1.75 if args.clean else 2.4),
    )
    fig.subplots_adjust(wspace=0.06, top=0.88 if args.clean else 0.73,
                        bottom=0.03, left=0.01, right=0.99)

    # ---- panel 1: pick the hit with the most visible arrow ----
    ax = axes[0]
    base_scatter(ax, Z, tks, colors)
    i = max(m1_hits, key=lambda k: ((Z[k] - Z[nn1[k]]) ** 2).sum())
    j = nn1[i]
    ax.scatter(*Z[i], s=110, marker="*", facecolor=colors[tks[i]],
               edgecolor="black", linewidth=0.9, zorder=6)
    ax.scatter(*Z[j], s=52, marker="o", facecolor=colors[tks[j]],
               edgecolor="black", linewidth=1.2, zorder=6)
    arrow(ax, Z[i], Z[j])
    caption(ax, "star: one view (firm A, day t).\n"
                "where does the partner's day-t\n"
                "view rank among all other-firm\n"
                "views?")
    title(ax, 0)

    # ---- panel 2 ----
    ax = axes[1]
    base_scatter(ax, Z, tks, colors)
    i, cents = max(
        m2_hits, key=lambda ic: ((Z[ic[0]] - ic[1][dts[ic[0]]]) ** 2).sum(),
    )
    for d, c in cents.items():
        own = d == dts[i]
        ax.scatter(*c, s=48 if own else 26, marker="X",
                   facecolor=INK if own else "#999999",
                   edgecolor="white", linewidth=0.5, zorder=5 if own else 4)
    ax.scatter(*Z[i], s=110, marker="*", facecolor=colors[tks[i]],
               edgecolor="black", linewidth=0.9, zorder=6)
    arrow(ax, Z[i], cents[dts[i]])
    caption(ax, "x: the ~21 day centroids, each\n"
                "built from the other 5 firms'\n"
                "views (star's firm excluded\n"
                "everywhere). own day rank?")
    title(ax, 1)

    # ---- panel 3: mirror of 2 — firm centroids, focal DAY dropped ----
    ax = axes[2]
    base_scatter(ax, Z, tks, colors, alpha=0.18)
    i, cents = max(
        m3_hits, key=lambda ic: ((Z[ic[0]] - ic[1][tks[ic[0]]]) ** 2).sum(),
    )
    for g in firms:
        own = g == tks[i]
        ax.scatter(*cents[g], s=170 if own else 120, marker="o",
                   facecolor=colors[g], edgecolor="black" if own else "white",
                   linewidth=1.3 if own else 0.9, zorder=5)
    ax.scatter(*Z[i], s=110, marker="*", facecolor=colors[tks[i]],
               edgecolor="black", linewidth=0.9, zorder=6)
    arrow(ax, Z[i], cents[tks[i]])
    caption(ax, "big dots: the 6 firm centroids,\n"
                "each built WITHOUT the star's\n"
                "day t (dropped everywhere).\n"
                "where does the OWN firm rank?")
    title(ax, 2)

    # ---- panel 4 ----
    ax = axes[3]
    base_scatter(ax, Z, tks, colors, alpha=0.12)
    f = max(m4_hits, key=lambda g: ((F[g] - F[partner_of[g]]) ** 2).sum())
    for ff_code in sorted({panel[g] for g in firms}):
        pair = sorted(g for g in firms if panel[g] == ff_code)
        ax.plot(*zip(F[pair[0]], F[pair[1]]), color="#bbbbbb", lw=0.9,
                ls=":", zorder=4)
    for g in firms:
        big = g in (f, partner_of[f])
        ax.scatter(*F[g], s=190 if big else 130, marker="o",
                   facecolor=colors[g],
                   edgecolor="black" if big else "white",
                   linewidth=1.3 if big else 0.9, zorder=6)
    arrow(ax, F[f], F[partner_of[f]])
    caption(ax, "big dots: the 6 firm centroids\n"
                "(~21 views averaged); dotted =\n"
                "industry pairs. where does the\n"
                "partner's rank among the other 5?")
    title(ax, 3)

    if not args.clean:
        fig.suptitle(
            "One fixed panel: 3 industries x 2 firms, one view per firm-day "
            "(hue = industry, dark/light = the two partner firms)\n"
            "each metric is scored two ways: top-1 hit and the target's "
            "mean rank",
            fontsize=8.0, y=0.98,
        )
    stem = "fixed_panel_metric_diagram" + ("_clean" if args.clean else "")
    written = save_figure(fig, eg.OUT_DIR / stem)
    plt.close(fig)
    for pth in written:
        print(f"Saved {pth}")


if __name__ == "__main__":
    main()
