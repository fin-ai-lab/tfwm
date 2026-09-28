"""Subspace alignment: embedding geometry vs the latent loading space.

Per month, compare the cross-sectional span of the top-J PCs of the
ticker-centroid embeddings (N x J) with the span of the estimated total
factor loadings Lambda (N x K_hat, eps = 0.08) via canonical correlations
(both sides demeaned). Summary statistic: rhobar = sum(rho^2) / K_hat in
[0, 1] — the fraction of the loading space captured by the embedding
subspace. Calibration: 200 row permutations of Lambda give the chance
distribution; we report the permutation z-score too.

Usage:  uv run python plots/latent_eval/factors/subspace_alignment.py [--months ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decode_loadings import FF_CACHE, FS_CACHE, SERIES, month_grid  # noqa: E402
from pelger import generalized_correlations  # noqa: E402

N_PERM = 200
J = 10


def emb_centroids(ym, series, t_codes, n_t, days_per):
    E = np.load(FF_CACHE / ym / f"emb_{series}.npz")["X_eval"].astype(np.float32)
    tm = np.zeros((n_t, E.shape[1]))
    np.add.at(tm, t_codes, E)
    return tm / np.maximum(days_per, 1)[:, None]


def top_pcs(E, J):
    E = E - E.mean(0)
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    return U[:, :J] * s[:J]


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
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")

    res = {s: {"rhobar": [], "zperm": []} for s in series}
    # A SERIES NEED NOT COVER EVERY MONTH -- see decode_loadings. A month whose
    # embedding never got built (no checkpoint for that arm yet) is dropped for
    # that series alone, and the months each series did use are recorded so a
    # reader knows which rows rest on fewer of them.
    used: dict[str, list[str]] = {s: [] for s in series}
    rng = np.random.default_rng(0)
    for ym in months:
        fz = np.load(FS_CACHE / f"factors_{ym}.npz")
        K = max(int(fz["k_total"][1]), 1)
        lam_lut = {t: i for i, t in enumerate(fz["tickers"])}
        gm, t_codes, t_uniq, d_codes, d_uniq, mid_mat = month_grid(ym)
        n_t = len(t_uniq)
        days_per = np.bincount(t_codes, minlength=n_t)
        idx = np.array([i for i, t in enumerate(t_uniq) if t in lam_lut])
        Lam = np.stack([fz["lam_total"][lam_lut[t_uniq[i]], :K] for i in idx])
        Lam = Lam - Lam.mean(0)

        perms = [rng.permutation(len(idx)) for _ in range(N_PERM)]
        here = {}
        for s in series:
            if not (FF_CACHE / ym / f"emb_{s}.npz").is_file():
                continue
            E = emb_centroids(ym, s, t_codes, n_t, days_per)[idx]
            P = top_pcs(E, J)
            _, tot = generalized_correlations(P, Lam)
            null = np.array([
                generalized_correlations(P, Lam[p])[1] for p in perms])
            res[s]["rhobar"].append(tot / K)
            res[s]["zperm"].append((tot - null.mean()) / (null.std() + 1e-12))
            used[s].append(ym)
            here[s] = tot / K
        # Read off THIS month's values, not res[s][-1]: with a series skipped,
        # the last element of its list is some earlier month's number.
        print(f"[{ym}] K={K} n={len(idx)} {len(here)}/{len(series)} series  "
              + "  ".join(f"{s}:{here[s]:.2f}"
                          for s in series[:4] if s in here), flush=True)

    print(f"\nrhobar = sum(rho^2)/K over top-{J} embedding PCs, "
          "mean over months (t across months); z vs row permutations")
    for s in series:
        miss = sorted(set(months) - set(used[s]))
        if miss:
            print(f"  {s}: no embedding for {len(miss)} month(s) -- "
                  f"aligned on {len(used[s])}: {miss}")
    print("series".ljust(22) + f"{'n':>4}{'rhobar':>8}{'t':>7}{'z_perm':>9}")
    for s in series:
        a = np.asarray(res[s]["rhobar"])
        z = np.asarray(res[s]["zperm"])
        if not len(a):
            print(s.ljust(22) + f"{0:>4}" + "       --     --       --")
            continue
        tt = a.mean() / (a.std() / np.sqrt(len(a)) + 1e-12) if len(a) > 1 else np.nan
        print(s.ljust(22) + f"{len(a):>4}{a.mean():8.3f}{tt:7.1f}{z.mean():9.1f}")

    with open(HERE / f"subspace_alignment{args.tag}.json", "w") as f:
        json.dump({"months": months, "J": J, "series": series,
                   "months_used": used,
                   "results": {s: {k: list(map(float, v))
                                   for k, v in d.items()}
                               for s, d in res.items()}}, f)
    print(f"\nwrote {HERE / f'subspace_alignment{args.tag}.json'}")


if __name__ == "__main__":
    main()
