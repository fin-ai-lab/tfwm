"""Explainer strip for the Pelger factor-structure analyses.

One real month. Panels 1 -> 2 are a pipeline: the panel's 5-min returns
define a correlation matrix whose PCA gives the month's latent factors
(perturbed-ER count K_hat) and every stock becomes a point in loading
space. Panels 3 and 4 then FORK off panel 2 — two independent questions
about that same loading space, drawn as a fork and not as a chain:

  3  decode (decode_loadings.py) — can a ridge on the WHOLE month-mean
     embedding predict ONE loading, lambda_1, out of fold (folds split by
     gvkey firm)? Is the information recoverable at all?
  4  subspace alignment (subspace_alignment.py) — how many of the K_hat
     loading directions do the embedding's top-10 PCs actually reach?
     sum_k rho_k^2 is the effective number of shared dimensions, so
     rhobar = sum(rho^2)/K_hat is drawn as a filled fraction of a K_hat-wide
     box, with 200 row permutations as the chance row. Is the factor
     structure PROMINENT in the embedding, not merely present?

The two are easy to conflate and were: panel 4 used to plot per-direction
rho^2 bars, whose first bar has sqrt = 0.86 against panel 3's r = 0.87, so
the panel read as a restatement of its neighbour. It is not — panel 4's
leading direction resembles lambda_3, and lambda_1 is the axis the top-10
PCs capture WORST (0.51 against the full embedding's 0.87). Reporting
panel 4 as a COUNT OF DIRECTIONS removes the false comparison; the
nearest-loading table this script prints backs the claim.

Every number in the figure is the analysis scripts' own construction on
this month — the harness pieces are imported from decode_loadings.py /
pelger.py, not re-derived. (Fold draws use a fresh rng(0) for the single
month, so the annotated r can differ from the multi-month JSON in the
third decimal.)

All inputs come from factor_structure_cache + ff_fullday_cache — no GPU.

Run:
    uv run plots/latent_eval/factors/factor_diagram.py            # annotated
    uv run plots/latent_eval/factors/factor_diagram.py --clean    # paper

--clean drops the captions and the suptitle and writes
factor_diagram_clean.{png,pdf}; caption.md is its prose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from style import SERIES_STYLES, WIDTH_FULL, apply_style, save_figure  # noqa: E402

from decode_loadings import (  # noqa: E402
    FF_CACHE, FS_CACHE, GVKEY, firm_ids, month_grid,
)
from pelger import generalized_correlations  # noqa: E402
from subspace_alignment import J, N_PERM, emb_centroids, top_pcs  # noqa: E402

INK = "#333333"
GRAY = "#999999"
MODEL_COLOR = SERIES_STYLES["ijepa"]["color"]

# Panels 3 and 4 are SIBLINGS, not a chain, and their titles have to carry
# the contrast that used to be invisible: 3 decodes ONE loading axis from
# the WHOLE embedding; 4 asks how much of the WHOLE loading space the
# embedding's top-10 PCs span. Without "one" vs "all K_hat" in the titles,
# panel 4 reads as panel 3 with more bars.
TITLES = [
    "1 · correlation PCA",
    "2 · stocks in loading space",
    r"3 · decode one loading ($\lambda_1$)",
    r"4 · span all $\hat K$ loadings",
]
# --clean: identical geometry and identical numbers, no in-panel captions
# and no suptitle — the slide/paper version where the titles carry the
# whole explanation and caption.md carries the definitions. Same convention
# as fixed_panel/metric_diagram.py --clean.
CLEAN_TITLES = [
    "Realized Correlation PCA",
    "Stocks in Loading Space",
    r"Decode One Loading ($\lambda_1$)",
    r"Span All $\hat K$ Loadings",
]


def month_returns(ym):
    """Standardized increment matrix Z (M x N), exactly as estimate_factors
    builds it: drop the 09:30->09:35 increment, drop zero-QV names,
    standardize each stock by sqrt of its realized quadratic variation."""
    z = np.load(FS_CACHE / f"panel_{ym}.npz")
    mids = z["mids"]
    T, D, marks = mids.shape
    r = np.log(mids[:, :, 2:]) - np.log(mids[:, :, 1:-1])
    X = r.reshape(T, D * (marks - 2)).T
    q = (X ** 2).sum(0)
    keep = q > 0
    X, tickers = X[:, keep], z["tickers"][keep]
    return X / np.sqrt(q[keep])[None, :], tickers, z


def cv_predict(X, Y, folds):
    """decode_loadings.cv_r, returning the OOS predictions instead of r."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P = np.full_like(Y, np.nan)
    for f in range(5):
        tr, te = folds != f, folds == f
        pred = Ridge(alpha=100.0).fit(X[tr], Y[tr]).predict(X[te])
        P[te] = np.asarray(pred).reshape(int(te.sum()), -1)
    return P


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2020-08",
                    help="one of the 31 reported months; the 6-month SSL "
                         "embeddings are cached for those only")
    ap.add_argument("--model", default="ijepa_6mo")
    ap.add_argument("--model-label", default="I-JEPA")
    ap.add_argument("--clean", action="store_true",
                    help="titles only: drop the captions and the suptitle, "
                         "write factor_diagram_clean")
    args = ap.parse_args()
    ym = args.month
    clean = args.clean
    apply_style()

    # With the captions gone nothing competes for the panel interior, so the
    # clean variant can afford larger axis furniture.
    fs_label = 8.0 if clean else 6.5
    fs_leg = 6.5 if clean else 5.4
    fs_tick = 6.5 if clean else 5.5

    fz = np.load(FS_CACHE / f"factors_{ym}.npz")
    K = max(int(fz["k_total"][1]), 1)
    lam, evals, r2_k = fz["lam_total"], fz["evals_total"], fz["r2_k"]
    lam_lut = {t: i for i, t in enumerate(fz["tickers"])}

    Z, _, pz = month_returns(ym)
    C_mat = Z.T @ Z                       # realized correlation, diag = 1
    order = np.argsort(lam[:, 0])
    N = len(order)

    gm, t_codes, t_uniq, d_codes, d_uniq, mid_mat = month_grid(ym)
    n_t = len(t_uniq)
    days_per = np.bincount(t_codes, minlength=n_t)
    idx = np.array([i for i, t in enumerate(t_uniq) if t in lam_lut])
    rows = np.array([lam_lut[t_uniq[i]] for i in idx])
    Y1 = lam[rows, 0]
    firms = firm_ids(t_uniq[idx], ym, pd.read_parquet(GVKEY))
    uf, uf_inv = np.unique(firms, return_inverse=True)
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(uf)) % 5
    folds = folds[uf_inv]

    def month_mean_emb(series):
        return emb_centroids(ym, series, t_codes, n_t, days_per)[idx]

    E_model = month_mean_emb(args.model)
    E_floor = month_mean_emb("randvit_s0")
    Ycol = Y1[:, None]
    P_model = cv_predict(E_model, Ycol, folds)[:, 0]
    P_floor = cv_predict(E_floor, Ycol, folds)[:, 0]
    r_model, r_floor = pearson(Y1, P_model), pearson(Y1, P_floor)

    # subspace alignment, subspace_alignment.py's exact construction
    Lam = lam[rows, :K] - lam[rows, :K].mean(0)
    rng_p = np.random.default_rng(0)
    perms = [rng_p.permutation(len(idx)) for _ in range(N_PERM)]
    P_model_pcs = top_pcs(E_model, J)
    rho_model, tot_model, _, W_lam = generalized_correlations(
        P_model_pcs, Lam, return_vectors=True)
    rho_floor = np.mean([
        generalized_correlations(top_pcs(month_mean_emb(f"randvit_s{s}"), J),
                                 Lam)[0] ** 2
        for s in range(5)], axis=0)
    null = np.array([
        generalized_correlations(P_model_pcs, Lam[p])[0] ** 2 for p in perms])
    rhobar_model = tot_model / K
    rhobar_floor = rho_floor.sum() / K

    print(f"[{ym}] N={N} K={K}  decode tot1: {args.model} "
          f"{r_model:+.3f} vs floor {r_floor:+.3f}  "
          f"rhobar {rhobar_model:.3f} vs floor {rhobar_floor:.3f}")

    # Which raw loading axis does each canonical direction resemble? Panel 4's
    # bar 1 is NOT panel 3's lambda_1 — the numbers happen to look alike
    # (rho_1 = 0.86 against panel 3's r = 0.87) and a reader who squares the
    # bar concludes the panels restate each other. Printed for caption.md;
    # the mapping is encoder-specific (each series gets its own rotation),
    # so it is deliberately NOT drawn as a shared x-tick label.
    V = Lam @ W_lam
    print("  canonical direction -> nearest raw loading "
          f"({args.model_label}):")
    nearest_lam = []
    for k in range(K):
        c = np.array([abs(np.corrcoef(V[:, k], Lam[:, j])[0, 1])
                      for j in range(K)])
        nearest_lam.append(int(c.argmax()) + 1)
        print(f"    dir {k+1}: rho^2={rho_model[k] ** 2:.3f}  "
              f"~lam{int(c.argmax()) + 1} (|r|={c.max():.2f})  "
              + " ".join(f"lam{j+1}:{c[j]:.2f}" for j in range(K)))
    print(f"  chance (200 row permutations of Lambda): "
          f"{null.mean(0)[:K].sum():.3f} of {K} "
          f"(rhobar {null.mean(0)[:K].sum() / K:.4f}) -- computed and cited in "
          f"caption.md, no longer drawn")
    gap = rho_model[:K] ** 2 - rho_floor[:K]
    print(f"  rhobar gap {gap.sum() / K:+.3f}; directions 1-2 carry "
          f"{100 * gap[:2].sum() / gap.sum():.0f}% of it")

    # ── figure ────────────────────────────────────────────────────────────
    # Taller than the panels need: the fork that distributes panel 2 into
    # panels 3 and 4 runs in a band above the titles.
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_FULL, 2.55 if clean else 3.1))
    fig.subplots_adjust(wspace=0.26, top=0.78 if clean else 0.70,
                        bottom=0.145 if clean else 0.115,
                        left=0.012, right=0.995)

    def caption(ax, text, right=False, top=False):
        if clean:
            return
        x, ha = (0.98, "right") if right else (0.02, "left")
        y, va = (0.98, "top") if top else (0.02, "bottom")
        ax.text(x, y, text, transform=ax.transAxes, fontsize=5.5,
                color=INK, ha=ha, va=va, zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.5))

    titles = CLEAN_TITLES if clean else TITLES

    def title(ax, k):
        ax.set_title(titles[k], fontsize=9 if clean else 7.5, pad=4)

    # ---- panel 1: correlation matrix + scree with the K_hat cut ----
    ax = axes[0]
    ax.imshow(C_mat[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1,
              interpolation="nearest", aspect="auto", rasterized=True)
    ax.set_xticks([]), ax.set_yticks([])
    ins = ax.inset_axes([0.55, 0.40, 0.42, 0.34])
    top = 12
    ins.bar(np.arange(1, K + 1), evals[:K], color=INK, width=0.75)
    ins.bar(np.arange(K + 1, top + 1), evals[K:top], color="#bbbbbb",
            width=0.75)
    ins.axvline(K + 0.5, color="black", lw=0.7, ls="--")
    ins.text(K + 1.2, evals[0] * 0.66, rf"$\hat K={K}$", fontsize=5.8)
    ins.set_xticks([]), ins.set_yticks([])
    ins.patch.set_alpha(0.9)
    ccax = ax.inset_axes([0.04, 0.93, 0.30, 0.035])
    ccb = fig.colorbar(ax.images[0], cax=ccax, orientation="horizontal")
    ccb.set_ticks([-1, 0, 1])
    ccb.ax.tick_params(labelsize=fs_tick - 1, length=1.5, pad=1)
    ccb.outline.set_linewidth(0.4)
    ax.text(0.355, 0.945, "corr", transform=ax.transAxes,
            fontsize=fs_tick, ha="left", va="center")
    caption(ax, f"realized corr of 5-min returns\n"
                f"({N} stocks, sorted by $\\lambda_1$);\n"
                "the perturbed ER rule picks $\\hat K$")
    title(ax, 0)

    # ---- panel 2: the loading-space scatter ----
    ax = axes[1]
    sc = ax.scatter(lam[:, 0], lam[:, 1], c=r2_k, cmap="viridis",
                    s=7, alpha=0.8, linewidth=0, rasterized=True)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(r"$\lambda_1$", fontsize=fs_label, labelpad=1)
    ax.set_ylabel(r"$\lambda_2$", fontsize=fs_label, labelpad=1)
    cax = ax.inset_axes([0.70, 0.93, 0.26, 0.04])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([float(r2_k.min()), float(r2_k.max())])
    cb.ax.set_xticklabels([f"{r2_k.min():.1f}", f"{r2_k.max():.1f}"])
    cb.ax.tick_params(labelsize=fs_tick - 1, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    ax.text(0.67, 0.95, r"$r^2_{\hat K}$", transform=ax.transAxes,
            fontsize=7.0 if clean else 5.8, ha="right", va="center")
    caption(ax, "every stock is a point in\n"
                "loading space (hue: $r^2_{\\hat K}$, the\n"
                "variance the factors explain)")
    title(ax, 1)

    # ---- panel 3: OOS decode scatter ----
    ax = axes[2]
    lims = np.percentile(Y1, [0.5, 99.5])
    ax.scatter(Y1, P_floor, s=5, color=GRAY, alpha=0.45, linewidth=0,
               rasterized=True, label=f"Random ViT  r={r_floor:+.2f}")
    ax.scatter(Y1, P_model, s=5, color=MODEL_COLOR, alpha=0.55, linewidth=0,
               rasterized=True,
               label=f"{args.model_label}  r={r_model:+.2f}")
    ax.set_xlim(lims), ax.set_yticks([]), ax.set_xticks([])
    ax.set_xlabel(r"true $\lambda_1$", fontsize=fs_label, labelpad=1)
    ax.set_ylabel(r"OOS predicted $\lambda_1$", fontsize=fs_label, labelpad=1)
    # model first, floor second — the same reading order as panel 4's bars.
    # The legend title carries the half of the contrast with panel 4 that the
    # panel title cannot: this panel gets the WHOLE embedding, panel 4 only
    # its top-10 PCs.
    h, l = ax.get_legend_handles_labels()
    leg = ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=fs_leg,
                    frameon=False, handletextpad=0.2, borderaxespad=0.1,
                    title=r"full embedding $\rightarrow \lambda_1$",
                    title_fontsize=fs_leg)
    leg.get_title().set_color(INK)
    leg._legend_box.align = "left"
    caption(ax, "ridge from the month-mean\n"
                "embedding, fit and scored on\n"
                "disjoint firms (5 folds)")
    title(ax, 2)

    # ---- panel 4: how much of the K_hat-dim loading space is spanned ----
    # sum_k rho_k^2 is the effective number of dimensions the two spans share,
    # so rhobar = sum/K_hat is a FILLED FRACTION of a K_hat-wide box. Drawn
    # that way the panel answers in DIRECTIONS OUT OF K_hat, a quantity no
    # reader can confuse with panel 3's correlation. The previous form —
    # per-direction rho^2 bars — could not survive beside panel 3: sqrt of
    # its first bar was 0.86 against panel 3's r = 0.87, so the panel read as
    # a restatement of its neighbour. (It is not: that direction resembles
    # lambda_3, and lambda_1 is the axis these PCs capture WORST. See the
    # nearest-loading table this script prints, and caption.md.)
    ax = axes[3]
    # TWO ROWS, NOT THREE. The permutation null was dropped from the figure
    # (still computed, printed above, and cited in caption.md): both encoders
    # sit an order of magnitude above it, so it never was the operative
    # comparison -- the untrained floor is -- and it cost a third of the
    # panel. The space buys a subtitle saying what the bar IS, taller bars,
    # and room for the per-direction tags. If someone asks whether the
    # in-sample CCA manufactures the alignment, the answer is the printed
    # 0.06 of 5, said out loud.
    rows = [
        (1, rho_model[:K] ** 2, MODEL_COLOR, args.model_label, rhobar_model),
        (0, rho_floor[:K], GRAY, "Random ViT", rhobar_floor),
    ]
    bh = 0.52
    # A segment is a CANONICAL DIRECTION, and those are rotations of the
    # loading space -- direction 1 is not lambda_1. The model row therefore
    # carries the raw loading each of its directions most resembles, which is
    # the single thing a reader gets wrong about this panel: here direction 1
    # is ~lambda_3 and lambda_1 is the axis these PCs capture WORST. Only
    # segments wide enough to hold a tag get one; the tail is a tie with the
    # floor anyway (directions 1-2 carry 92% of the gap).
    LABEL_MIN = 0.15
    for y0, vals, col, lab, rbar in rows:
        left = 0.0
        is_model = rbar is not None and lab == args.model_label
        for k, v in enumerate(vals):
            # shade down the ordered directions so the segmentation reads as
            # 1st, 2nd, ... rather than as one undifferentiated bar
            ax.barh(y0, v, left=left, height=bh, color=col, zorder=2,
                    alpha=1.0 - 0.11 * k, edgecolor="white", lw=0.5)
            if is_model and v >= LABEL_MIN:
                # Staggered heights with leaders: the labelled segments are
                # the two widest and they are still only ~0.2 in apart, so
                # side-by-side tags overlap.
                xm = left + v / 2
                h = 0.12 + 0.19 * (k % 2)
                ax.plot([xm, xm], [y0 + bh / 2, y0 + bh / 2 + h],
                        color=col, lw=0.6, zorder=3, clip_on=False)
                ax.text(xm, y0 + bh / 2 + h + 0.03,
                        rf"$\approx\lambda_{{{nearest_lam[k]}}}$",
                        fontsize=fs_leg - 0.5, color=col, ha="center",
                        va="bottom", zorder=4)
            left += v
        ax.add_patch(Rectangle((0, y0 - bh / 2), K, bh, facecolor="none",
                               edgecolor=INK, lw=0.7, zorder=3))
        tag = lab if rbar is None else rf"{lab}   $\bar\rho$={rbar:.2f}"
        ax.text(0.04, y0 + bh / 2 + (0.62 if is_model else 0.10), tag,
                fontsize=fs_leg, color=INK if rbar is not None else GRAY,
                ha="left", va="bottom")
        ax.text(K - 0.12, y0, f"{vals.sum():.2f} of {K}", fontsize=fs_leg,
                color=INK if rbar is not None else GRAY,
                ha="right", va="center", zorder=4)
    # What the bar is, in the space the null row used to occupy. The clean
    # variant is otherwise caption-free by design; this one line is the
    # exception, because "1.42 of 5" is meaningless without it.
    ax.text(K / 2, 3.02, f"top-{J} embedding PCs vs.\n"
                         rf"the $\hat K$-dim loading space",
            fontsize=fs_leg, color=GRAY, ha="center", va="top",
            linespacing=1.35)
    ax.set_xlim(-0.04, K + 0.04)
    ax.set_ylim(-0.52, 3.05 if clean else 4.60)
    ax.set_xticks(np.arange(K + 1))
    ax.set_yticks([])
    ax.tick_params(labelsize=fs_tick, length=2)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Directions Captured", fontsize=fs_label, labelpad=1)
    caption(ax, "the top-10 embedding PCs vs\n"
                "the $\\hat K$-dim loading space;\n"
                "$\\bar\\rho=\\Sigma\\rho^2/\\hat K$ = filled share",
            right=True, top=True)
    title(ax, 3)

    # ---- flow: 1 -> 2 is a pipeline step, 2 -> {3, 4} is a FORK ----
    # Panels 3 and 4 are two INDEPENDENT questions about the same loading
    # space (decode_loadings.py and subspace_alignment.py, separate scripts,
    # neither consuming the other). The old chain of four identical arrows
    # said "pipeline" and made readers hunt for what panel 4 adds to panel 3.
    pos = [a.get_position() for a in axes]
    y = pos[0].y0 + 0.82 * (pos[0].y1 - pos[0].y0)

    def arrow(xy0, xy1):
        fig.add_artist(FancyArrowPatch(
            xy0, xy1, transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=9, color=INK, lw=1.2, shrinkA=0, shrinkB=0))

    arrow((pos[0].x1 + 0.004, y), (pos[1].x0 - 0.004, y))

    # The fork rises out of the 2|3 gap and runs over panel 3's title, so it
    # can drop into both siblings from the same stem.
    y_bracket = 0.945 if clean else 0.865
    y_head = y_bracket - 0.052
    x_stem = 0.5 * (pos[1].x1 + pos[2].x0)
    x3 = 0.5 * (pos[2].x0 + pos[2].x1)
    x4 = 0.5 * (pos[3].x0 + pos[3].x1)
    fig.add_artist(Line2D(
        [pos[1].x1 + 0.004, x_stem, x_stem, x4],
        [y, y, y_bracket, y_bracket],
        transform=fig.transFigure, color=INK, lw=1.2,
        solid_capstyle="round", solid_joinstyle="round"))
    for x in (x3, x4):
        arrow((x, y_bracket), (x, y_head))

    if not clean:
        fig.suptitle(
            f"One month ({ym}): the panel defines its own latent factors.\n"
            "Can the embedding decode a stock's loadings, and how much of "
            "the loading space does it span?",
            fontsize=8.0, y=0.995,
        )
    stem = "factor_diagram" + ("_clean" if clean else "")
    written = save_figure(fig, HERE / stem)
    plt.close(fig)
    for pth in written:
        print(f"Saved {pth}")


if __name__ == "__main__":
    main()
