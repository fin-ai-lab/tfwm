"""Decode per-stock latent factor loadings from frozen embeddings.

Targets per (ticker, month), from estimate_factors.py (standardized units,
this month's own panel): loadings on total factors 1-4, the top-K_hat
factor-model R^2, continuous-factor loadings 1-2 and jump-factor loadings
1-2 (the a=3 split). Features per ticker: month-mean full-day embedding
from ff_fullday_cache. Harness: 5-fold cross-firm CV (folds grouped by
gvkey so dual listings never straddle a fold), Ridge(alpha=100), score =
OOS Pearson r, mean over months, t across months. The evidential
reference is the random-init ViT floor on the same months.

The FF experiment's 9-stat liquidity control (control / emb+ctl / resid
feature sets) was RETIRED 2026-08-26: an ad hoc construction that capped
the raw decode, and the model ordering was identical with and without it.
Results are still keyed ``<series>|emb`` so pre-retirement JSONs (which
carry the extra variants) stay readable by the same consumers.

Results are printed and dumped to decode_loadings.json next to this file.

Usage:  uv run python plots/latent_eval/factors/decode_loadings.py [--months ...]
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
# Overridable so the sweep can run one month per SLURM job on a node
# that does not mount lab/; unset, the path is exactly what it
# always was, so local runs are unchanged. Added 2026-08-20.
FF_CACHE = Path(os.environ.get(
    "FF_FULLDAY_CACHE",
    "lab/market-jepa-checkpoints/ff_fullday_cache"))
FS_CACHE = Path(os.environ.get(
    "FACTOR_STRUCTURE_CACHE",
    "lab/market-jepa-checkpoints/factor_structure_cache"))
GVKEY = Path(os.environ.get(
    "MJ_GVKEY_PATH",
    "lab/market-text-data/gvkey_ticker_history.parquet"))
SERIES = ["supervised_return", "supervised_vol", "supervised_spread",
          "multihead", "lejepa", "dino", "mae", "byol", "ijepa", "cpc",
          "randvit_s0", "randvit_s1", "randvit_s2", "randvit_s3", "randvit_s4"]
TARGETS = ["tot1", "tot2", "tot3", "tot4", "r2_k",
           "cont1", "cont2", "jump1", "jump2"]


def month_grid(ym):
    gm = np.load(FF_CACHE / ym / "grid_meta.npz", allow_pickle=True)
    t_codes, t_uniq = pd.factorize(gm["ticker"])
    d_codes, d_uniq = pd.factorize(gm["date"])
    order = np.argsort(d_uniq)
    d_uniq, d_codes = d_uniq[order], np.argsort(order)[d_codes]
    mid_mat = np.full((len(t_uniq), len(d_uniq)), np.nan, np.float32)
    mid_mat[t_codes, d_codes] = gm["mid"]
    return gm, t_codes, t_uniq, d_codes, d_uniq, mid_mat


def firm_ids(t_uniq, ym, hist):
    mid_date = pd.Timestamp(f"{ym}-15")
    h = hist[(hist.valid_start_date <= mid_date)
             & (hist.valid_end_date >= mid_date)]
    lut = h.drop_duplicates("ticker").set_index("ticker")["gvkey"]
    return np.array([f"G:{int(lut[t])}" if t in lut.index else f"T:{t}"
                     for t in t_uniq])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--series", nargs="*", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for the output json (e.g. _14mo)")
    args = ap.parse_args()
    series = args.series or SERIES
    months = args.months or sorted(
        p.name[8:15] for p in FS_CACHE.glob("factors_*.npz"))
    hist = pd.read_parquet(GVKEY)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")

    featsets = [f"{s}|emb" for s in series]
    res = {f: {t: [] for t in TARGETS} for f in featsets}
    # A SERIES NEED NOT COVER EVERY MONTH. An arm whose checkpoint for a month
    # has not landed has no embedding for it either, and the run exists to
    # score the arms that ARE complete -- so a missing npz drops that month for
    # that series and nothing else. Which months each series actually decoded
    # is recorded: the mean is over them, not over `months`, and a reader
    # comparing two rows needs to know they rest on different n.
    used: dict[str, list[str]] = {s: [] for s in series}
    rng = np.random.default_rng(0)

    for ym in months:
        fz = np.load(FS_CACHE / f"factors_{ym}.npz")
        lam_lut = {t: i for i, t in enumerate(fz["tickers"])}
        Yfull = np.column_stack([
            fz["lam_total"][:, :4], fz["r2_k"],
            fz["lam_cont"][:, :2], fz["lam_jump"][:, :2]])

        gm, t_codes, t_uniq, d_codes, d_uniq, mid_mat = month_grid(ym)
        n_t = len(t_uniq)
        days_per = np.bincount(t_codes, minlength=n_t)
        idx = np.array([i for i, t in enumerate(t_uniq) if t in lam_lut])
        if len(idx) < 50:
            print(f"[{ym}] only {len(idx)} matched tickers — skip")
            continue
        Y = np.stack([Yfull[lam_lut[t_uniq[i]]] for i in idx])
        firms = firm_ids(t_uniq[idx], ym, hist)
        uf, uf_inv = np.unique(firms, return_inverse=True)
        folds = rng.permutation(len(uf)) % 5
        folds = folds[uf_inv]

        def cv_r(X):
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            P = np.full_like(Y, np.nan)
            for f in range(5):
                tr, te = folds != f, folds == f
                P[te] = Ridge(alpha=100.0).fit(X[tr], Y[tr]).predict(X[te])
            return [np.corrcoef(Y[:, j], P[:, j])[0, 1]
                    for j in range(Y.shape[1])]

        for s in series:
            f_emb = FF_CACHE / ym / f"emb_{s}.npz"
            if not f_emb.is_file():
                continue
            E = np.load(f_emb)["X_eval"].astype(np.float32)
            tm = np.zeros((n_t, E.shape[1]))
            np.add.at(tm, t_codes, E)
            Ef = (tm / np.maximum(days_per, 1)[:, None])[idx]
            for t, r in zip(TARGETS, cv_r(Ef)):
                res[f"{s}|emb"][t].append(r)
            used[s].append(ym)
        n_here = sum(used[s][-1:] == [ym] for s in series)
        print(f"[{ym}] decoded {len(idx)} tickers, {n_here}/{len(series)} series",
              flush=True)

    short = {s: sorted(set(months) - set(used[s])) for s in series}
    for s, miss in short.items():
        if miss:
            print(f"  {s}: no embedding for {len(miss)} month(s) -- "
                  f"decoded on {len(used[s])}: {miss}")

    print("\nOOS Pearson r per target, mean over months (t across months)")
    print("series".ljust(20) + f"{'n':>4}" + "".join(f"{t:>12}" for t in TARGETS))
    for f in featsets:
        cells = []
        for t in TARGETS:
            a = np.asarray(res[f][t], dtype=float)
            a = a[np.isfinite(a)]
            if len(a) < 2:
                cells.append(f"{a.mean():+.2f}(--)")
            else:
                tt = a.mean() / (a.std() / np.sqrt(len(a)) + 1e-12)
                cells.append(f"{a.mean():+.2f}({tt:+.0f})")
        print(f.split("|")[0].ljust(20)
              + f"{len(used[f.split('|')[0]]):>4}"
              + "".join(f"{c:>12}" for c in cells))

    with open(HERE / f"decode_loadings{args.tag}.json", "w") as f:
        json.dump({"months": months, "targets": TARGETS, "series": series,
                   "months_used": used, "results": res}, f)
    print(f"\nwrote {HERE / f'decode_loadings{args.tag}.json'}")


if __name__ == "__main__":
    main()
