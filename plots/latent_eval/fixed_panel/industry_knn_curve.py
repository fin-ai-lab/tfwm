"""Line plot: month-pooled same-industry precision@k over chance, as a
function of k (raw embedding space).

Same setup as industry_nn_sweep.py's pooled metric: per eval month, pool
the ~n_days*P*S drawn points; for each point take its k nearest neighbors
among OTHER stocks' points and score the same-industry fraction; chance =
the same-industry share of each point's pool (k-independent). The curve
shows mean over the MONTHS panel's per-month ratio, with a 95% CI band;
1.0 = chance. Ratio converges to 1 by construction as k approaches the
pool size.

Reads the fixed-panel caches industry_nn_sweep writes, so run that first.

Run:
    uv run plots/latent_eval/fixed_panel/industry_knn_curve.py
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import panel_lib as eg
from style import save_figure

from industry_nn_sweep import MODEL_ORDER, MODEL_SPECS, MONTHS

# Only the arms this curve is usually read for; everything else falls back to
# gray via .get(). Keys follow MODEL_ORDER (IC-era, manifest-resolved).
MODEL_COLORS = {
    "pair_rrc_final": "#1f77b4",
    "pair_k2_final": "#ff7f0e",
    "pair_k2ind_final": "#d62728",
    "sup_return_w8": "#2ca02c",
    "sup_multi_w8": "#9467bd",
}


def prec_curve(
    X: np.ndarray, tickers: np.ndarray, inds: np.ndarray, k_max: int,
) -> tuple[np.ndarray, float]:
    """(prec[k_max] for k=1..k_max, chance)."""
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    same_stock = tickers[:, None] == tickers[None, :]
    D[same_stock] = np.inf
    order = np.argsort(D, axis=1)[:, :k_max]
    same_ind = inds[:, None] == inds[None, :]
    hits = np.take_along_axis(same_ind, order, axis=1)  # (n, k_max)
    prec = hits.cumsum(1) / np.arange(1, k_max + 1)     # per-point p@k
    pool = ~same_stock
    chance = float(((same_ind & pool).sum(1) / pool.sum(1)).mean())
    return prec.mean(0), chance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_portfolios", type=int, default=3)
    p.add_argument("--n_stocks", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k_max", type=int, default=25)
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    P, S = args.n_portfolios, args.n_stocks
    models = args.models.split(",")
    eg.apply_variant("mixed")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )

    # ratios[model] = (n_months, k_max) per-month ratio curves
    ratios: dict[str, list[np.ndarray]] = {m: [] for m in models}
    for ym in MONTHS:
        ev_month, _, _ = eg.eval_window_t_plus_n(ym, 1)
        cache = eg.CACHE_DIR / f"indnn__{ev_month}__seed{args.seed}__P{P}S{S}.pkl"
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        day_sets, batches = blob["day_sets"], blob["batches"]
        for key in models:
            # The shared registry loader: manifest keys resolve by eval month,
            # glob keys by the month before it. Reaching into project_glob
            # here would KeyError on MODEL_ORDER's manifest defaults.
            backbone = eg.load_series_encoder(key, ev_month, device)
            if backbone is None:
                print(f"{ym}: no checkpoint for {key} — skipped")
                continue
            try:
                fwd = eg.forward_cached(backbone, batches, device, cap=10**9)
            finally:
                backbone.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            tk = np.asarray([str(t) for t in fwd["tickers"]], dtype=object)
            dt = np.asarray([str(d) for d in fwd["dates"]], dtype=object)
            keep = np.zeros(len(tk), dtype=bool)
            for d in sorted(day_sets):
                sel = dt == d
                if sel.sum() == P * S:
                    keep |= sel
            inds = np.asarray(
                [day_sets[d][t] for t, d in zip(tk[keep], dt[keep])]
            )
            prec, chance = prec_curve(fwd["X"][keep], tk[keep], inds, args.k_max)
            ratios[key].append(prec / chance)
        print(f"{ym} done", flush=True)

    ks = np.arange(1, args.k_max + 1)
    fig, ax = plt.subplots(figsize=(eg.WIDTH_FULL * 0.6, 2.9))
    for key in models:
        if not ratios[key]:
            continue
        R = np.stack(ratios[key])  # (months, k)
        mean = R.mean(0)
        se = R.std(0, ddof=1) / np.sqrt(len(R))
        color = MODEL_COLORS.get(key, "gray")
        ax.plot(ks, mean, color=color, lw=1.4, label=MODEL_SPECS[key]["label"])
        ax.fill_between(
            ks, mean - 1.96 * se, mean + 1.96 * se,
            color=color, alpha=0.12, lw=0,
        )
        print(
            f"[{key}] ratio @k=1: {mean[0]:.2f}  @3: {mean[2]:.2f}  "
            f"@5: {mean[4]:.2f}  @10: {mean[9]:.2f}  @{args.k_max}: {mean[-1]:.2f}"
        )
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("k (nearest neighbors)", fontsize=8)
    ax.set_ylabel("same-industry precision@k / chance", fontsize=8)
    ax.set_title(
        f"{args.n_portfolios} industries x {args.n_stocks} stocks per day, "
        f"month-pooled (14 months, 95% CI)",
        fontsize=9,
    )
    ax.legend(fontsize=7, frameon=False)
    out = eg.OUT_DIR / f"industry_knn_curve_P{P}S{S}"
    written = save_figure(fig, out)
    plt.close(fig)
    for pth in written:
        print(f"Saved {pth}")


if __name__ == "__main__":
    main()
