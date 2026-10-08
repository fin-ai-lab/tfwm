"""Decode each DAY's market return and factor returns from frozen embeddings.

decode_loadings asks whether a stock's factor EXPOSURE (lambda_i) can be read
off its embedding -- the thing an objective invariant across stocks is built
to discard. This is the complementary question: is the day's factor
REALIZATION f_t (what every stock shared that day) in the embedding.
The embeddings are full-day views ending at the close, so this is a
contemporaneous representation test, not a forecast.

Targets per (month, day), from the same factor_structure_cache panel and
factors the loading decode uses (09:35 -> 16:00, 77 increments):
  mkt     equal-weighted mean of the panel's daily log mid returns
  f1..f4  daily sum of the total-factor series fac_total[:, k]
          (sign-fixed so the mean loading is positive; f1 is ~the market)

Two readouts, both scored out of sample with leave-one-MONTH-out folds:
  A  day-level: Ridge from the cross-sectional MEAN embedding of the day to
     mkt. The direct question. f2..f4 are not decoded this way: the PCA
     rotates every month, so "factor 2" is a different portfolio in each
     month and one pooled map cannot apply.
  B  stock-level: Ridge from each (stock, day) embedding to that stock's
     vol-standardized daily return z_it, then aggregated with the month's
     own weights -- equal weights for mkt, loadings lambda_k for f_k (the
     factor is the loading-weighted portfolio of z, so a readout of z
     composes into a readout of f). The aggregation is checked on the TRUE
     z first; that ceiling is printed and should be ~1.

Score: OOS Pearson r between decoded and true daily series. For A, pooled
over all days and the mean of per-month r; for B, the mean of per-month r
(each month's factor has its own scale). Floor = mean over randvit_s0..4.

Usage:  uv run python plots/latent_eval/factors/decode_factor_returns.py [--tag _6mo]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
FF_CACHE = Path(os.environ.get(
    "FF_FULLDAY_CACHE",
    "lab/market-jepa-checkpoints/ff_fullday_cache"))
FS_CACHE = Path(os.environ.get(
    "FACTOR_STRUCTURE_CACHE",
    "lab/market-jepa-checkpoints/factor_structure_cache"))
# The fixed-panel table's rows (plots/core/fixed_panel_table.tex), at the
# readout that table reports.
SERIES = {
    "sup_return_w8": "Return", "sup_vol_w8": "Vol",
    "sup_spread_w8": "Spread", "sup_multi_w8": "Multihead",
    "pair_rrc_6mo": "Same Stock, Diff. View", "pair_warp_6mo": "Time Warping",
    "pair_noise_6mo": "Gaussian Noising", "pair_k2_6mo": "Cross Stock",
    "pair_k2ind_6mo": "C-S, Same Industry",
    "dino_6mo": "DINO", "byol_6mo": "BYOL", "cpc_6mo": "CPC",
    "ijepa_6mo": "I-JEPA", "mae_6mo": "MAE", "ts2vec_6mo": "TS2Vec",
    "cost_6mo": "CoST", "tfc_6mo": "TF-C", "timemae_6mo": "TimeMAE",
    "tsfm_chronos2_cmean_l12": "Chronos-2", "tsfm_kronos_cmean_l12": "Kronos",
    "tsfm_timesfm3_cmean_l20": "TimesFM 3.0",
}
FLOOR_SEEDS = [f"randvit_s{i}" for i in range(5)]
N_FAC = 4
ALPHAS = np.logspace(-1, 5, 13)
TARGETS = ["mkt", "f1", "f2", "f3", "f4"]


def load_month(ym, target="ret"):
    """Targets and the (stock, day) -> grid-row alignment for one month."""
    p = np.load(FS_CACHE / f"panel_{ym}.npz")
    fz = np.load(FS_CACHE / f"factors_{ym}.npz")
    tickers, dates = fz["tickers"], p["dates"]
    keep = np.isin(p["tickers"], tickers)
    assert (p["tickers"][keep] == tickers).all()
    mids = p["mids"][keep]                                   # (N, D, 79)
    inc = np.log(mids[:, :, 2:]) - np.log(mids[:, :, 1:-1])  # (N, D, 77)
    N, D, m = inc.shape
    r = inc.sum(2)                                           # (N, D)
    scale = np.sqrt((inc.reshape(N, -1) ** 2).sum(1))        # correlation_pca's
    z = r / scale[:, None]
    if target == "rv":
        # log realized variance of the day; "mkt" becomes its EW mean
        z = np.log((inc ** 2).sum(2) + 1e-12)
        r = z
    f = fz["fac_total"][:, :N_FAC].reshape(D, m, N_FAC).sum(1)   # (D, K)
    lam = fz["lam_total"][:, :N_FAC]                         # (N, K)

    gm = np.load(FF_CACHE / ym / "grid_meta.npz", allow_pickle=True)
    t_lut = {t: i for i, t in enumerate(tickers)}
    d_lut = {d: j for j, d in enumerate(dates)}
    row = np.full((N, D), -1)
    for g, (t, d) in enumerate(zip(gm["ticker"], gm["date"])):
        i, j = t_lut.get(t), d_lut.get(d)
        if i is not None and j is not None:
            row[i, j] = g
    return dict(ym=ym, r=r, z=z, f=f, lam=lam, row=row,
                mkt=r.mean(0), D=D)


def aggregate(M, P):
    """Panel (N, D) of z-like values -> (D, 1+K) of [EW mean, lam-weighted]."""
    ok = np.isfinite(P)
    P0 = np.where(ok, P, 0.0)
    ew = P0.sum(0) / np.maximum(ok.sum(0), 1)
    fk = M["lam"].T @ P0                                     # (K, D)
    return np.column_stack([ew, fk.T])


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def score_series(key, months):
    """{"A_pooled", "A_month", "B_<target>"} -> per-month list (+ pooled)."""
    embs, have = [], []
    for M in months:
        f = FF_CACHE / M["ym"] / f"emb_{key}.npz"
        if f.is_file():
            embs.append(np.load(f)["X_eval"].astype(np.float32))
            have.append(M)
    if len(have) < 4:
        return None

    # ---- A: mean embedding of the day -> EW market return
    Xd, yd, md = [], [], []
    for mi, (M, E) in enumerate(zip(have, embs)):
        for j in range(M["D"]):
            rows = M["row"][:, j]
            rows = rows[rows >= 0]
            if len(rows) < 20:
                continue
            Xd.append(E[rows].mean(0)); yd.append(M["mkt"][j]); md.append(mi)
    Xd, yd, md = np.asarray(Xd), np.asarray(yd), np.asarray(md)
    pa = np.full_like(yd, np.nan)
    for mi in np.unique(md):
        tr, te = md != mi, md == mi
        mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-9
        reg = RidgeCV(alphas=ALPHAS).fit((Xd[tr] - mu) / sd, yd[tr])
        pa[te] = reg.predict((Xd[te] - mu) / sd)
    out = {"A_pooled": corr(pa, yd),
           "A_month": [corr(pa[md == mi], yd[md == mi]) for mi in np.unique(md)]}

    # ---- B: (stock, day) embedding -> z_it, aggregated per month
    Xs, ys, ms, idx = [], [], [], []
    for mi, (M, E) in enumerate(zip(have, embs)):
        ii, jj = np.nonzero(M["row"] >= 0)
        Xs.append(E[M["row"][ii, jj]])
        ys.append(np.clip(M["z"][ii, jj], -30, 30))
        ms.append(np.full(len(ii), mi)); idx.append((ii, jj))
    Xs, ys, ms = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ms)
    ps = np.empty_like(ys)
    # Leave-one-month-out from per-month sufficient statistics: the same fit
    # as Ridge(alpha=100) on train-fold-standardized features, without
    # re-touching ~300k rows 32 times.
    stats = []
    for mi in range(len(have)):
        X = Xs[ms == mi].astype(np.float64)
        y = ys[ms == mi].astype(np.float64)
        stats.append((len(y), X.sum(0), X.T @ X, y.sum(), X.T @ y))
    tot = [sum(s[i] for s in stats) for i in range(5)]
    for mi in range(len(have)):
        n, sx, Q, sy, Xy = (t - s for t, s in zip(tot, stats[mi]))
        mu, ybar = sx / n, sy / n
        C = Q - n * np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(C), 0, None) / n) + 1e-9
        A = C / np.outer(sd, sd) + 100.0 * np.eye(len(mu))
        w = np.linalg.solve(A, (Xy - n * mu * ybar) / sd)
        te = ms == mi
        ps[te] = ((Xs[te] - mu) / sd) @ w + ybar
    for t in TARGETS:
        out[f"B_{t}"] = []
    out["B_stock"] = []
    for mi, M in enumerate(have):
        P = np.full_like(M["z"], np.nan)
        ii, jj = idx[mi]
        P[ii, jj] = ps[ms == mi]
        dec = aggregate(M, P)
        truth = np.column_stack([M["mkt"], M["f"]])
        for k, t in enumerate(TARGETS):
            out[f"B_{t}"].append(corr(dec[:, k], truth[:, k]))
        # stock-level: mean over days of the cross-sectional r(pred, z)
        Z = M["z"]
        out["B_stock"].append(np.nanmean([
            corr(P[:, j][np.isfinite(P[:, j])], Z[:, j][np.isfinite(P[:, j])])
            for j in range(M["D"])]))
    out["months"] = [M["ym"] for M in have]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--series", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--target", choices=["ret", "rv"], default="ret",
                    help="rv: decode daily log realized variance instead; "
                         "only the EW market column is scored")
    args = ap.parse_args()
    if args.months:
        yms = args.months
    else:
        d = json.load(open(HERE / "decode_loadings_6mo.json"))
        yms = d["months_used"]["pair_k2_6mo"]
    series = args.series or list(SERIES) + FLOOR_SEEDS

    global TARGETS
    if args.target == "rv":
        TARGETS = ["mkt"]
    months = [load_month(ym, args.target) for ym in yms]
    # Ceiling: does the aggregation recover the targets from the TRUE z?
    ceil = np.array([[corr(a, b) for a, b in zip(
        aggregate(M, M["z"]).T, np.column_stack([M["mkt"], M["f"]]).T)]
        for M in months])
    print(f"{len(months)} months, {sum(M['D'] for M in months)} days")
    print("aggregation ceiling on true z (mean r): "
          + "  ".join(f"{t}={c:.3f}" for t, c in zip(TARGETS, ceil.mean(0))))

    res = {}
    for s in series:
        o = score_series(s, months)
        if o is None:
            print(f"  {s}: too few months -- skipped")
            continue
        res[s] = o
        print(f"  {s:28s} A={o['A_pooled']:+.3f}  "
              + "  ".join(f"{t}={np.nanmean(o['B_'+t]):+.3f}" for t in TARGETS)
              + f"  stock={np.nanmean(o['B_stock']):+.3f}", flush=True)

    with open(HERE / f"decode_factor_returns{args.tag}.json", "w") as f:
        json.dump({"months": yms, "targets": TARGETS,
                   "ceiling": dict(zip(TARGETS, ceil.mean(0).tolist())),
                   "results": res}, f)
    print(f"wrote {HERE / f'decode_factor_returns{args.tag}.json'}")


if __name__ == "__main__":
    main()
