"""Per-month Pelger factor estimation from the 5-min panels.

For each month's panel: correlation-PCA on total increments, the perturbed
eigenvalue-ratio count (ERP1, eps in {0.05, 0.08, 0.20}), the a=3
continuous/jump split, and separate PCA on each piece. Loadings are in
standardized (correlation) units, each factor sign-fixed so its
cross-sectional mean loading is positive.

Per stock we also record r2_k: the share of its standardized variance
explained by the top-K_hat total factors.

Output: factor_structure_cache/factors_<ym>.npz
    tickers          U12 (zero-QV names dropped, logged)
    lam_total        f32 (N, 10)   lam_cont f32 (N, 10)   lam_jump f32 (N, 10)
    fac_total        f32 (M, 10)   fac_cont f32 (M, 10)   fac_jump f32 (M, 10)
    evals_total/cont/jump   f64 (N,)
    k_total/k_cont/k_jump   i32 (3,)  [eps = 0.05, 0.08, 0.20]
    r2_k             f32 (N,)  (top-K_hat(0.08) total factors)
    jump_frac, jump_qv_share  f64 scalars

Usage:  uv run python plots/latent_eval/factors/estimate_factors.py [2020-08 ...]
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pelger import (  # noqa: E402
    correlation_pca, jump_threshold, perturbed_er_count, split_continuous_jump,
)

# Overridable so the sweep can run one month per SLURM job on a node
# that does not mount /data/lab; unset, the path is exactly what it
# always was, so local runs are unchanged. Added 2026-08-20.
CACHE = Path(os.environ.get(
    "FACTOR_STRUCTURE_CACHE",
    "/data/lab/market-jepa-checkpoints/factor_structure_cache"))
KMAX = 10
EPS_GRID = (0.05, 0.08, 0.20)


def sign_fix(loadings: np.ndarray, factors: np.ndarray) -> None:
    s = np.where(loadings.mean(0) < 0, -1.0, 1.0)
    loadings *= s
    factors *= s


def ks(evals: np.ndarray) -> np.ndarray:
    return np.asarray(
        [perturbed_er_count(evals, eps=e)[0] for e in EPS_GRID], np.int32)


def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    out = correlation_pca(X, K=KMAX)
    lam, fac = out["loadings"].copy(), out["factors"].copy()
    sign_fix(lam, fac)
    return lam, fac, out["eigenvalues"], out


def estimate_month(ym: str) -> None:
    out_path = CACHE / f"factors_{ym}.npz"
    if out_path.exists():
        print(f"[{ym}] exists — skip", flush=True)
        return
    z = np.load(CACHE / f"panel_{ym}.npz")
    mids = z["mids"]
    T, D, marks = mids.shape
    # drop the 09:30->09:35 increment: opening quotes are unreliable
    # (one-sided books at the bell) — matches the paper's 9:35 start, 77/day
    r = np.log(mids[:, :, 2:]) - np.log(mids[:, :, 1:-1])
    m = marks - 2
    X = r.reshape(T, D * m).T
    q = (X ** 2).sum(0)
    keep = q > 0
    X, tickers = X[:, keep], z["tickers"][keep]
    N = X.shape[1]

    lam_t, fac_t, ev_t, out_t = run_pca(X)
    k_t = ks(ev_t)
    K_hat = max(int(k_t[1]), 1)
    # per-stock variance share of the top-K_hat factors (standardized units)
    Z = X / out_t["scale"][None, :]
    common_k = out_t["factors"][:, :K_hat] @ out_t["loadings"][:, :K_hat].T
    r2_k = 1.0 - ((Z - common_k) ** 2).sum(0) / (Z ** 2).sum(0)

    thr = np.empty_like(X)
    for i in range(N):
        thr[:, i] = jump_threshold(X[:, i].reshape(D, m), a=3.0).ravel()
    XC, XD, is_jump = split_continuous_jump(X, thr)

    def masked_pca(Xp):
        """PCA over the active columns only; inactive names get 0 loadings."""
        act = (Xp ** 2).sum(0) > 0
        lam = np.zeros((N, KMAX), np.float32)
        lam_a, fac, ev, _ = run_pca(Xp[:, act])
        lam[act] = lam_a
        return lam, fac, ev, ks(ev)

    lam_c, fac_c, ev_c, k_c = masked_pca(XC)
    lam_j, fac_j, ev_j, k_j = masked_pca(XD)

    np.savez_compressed(
        out_path,
        tickers=tickers,
        lam_total=lam_t.astype(np.float32),
        lam_cont=lam_c.astype(np.float32),
        lam_jump=lam_j,
        fac_total=fac_t.astype(np.float32),
        fac_cont=fac_c.astype(np.float32),
        fac_jump=fac_j.astype(np.float32),
        evals_total=ev_t, evals_cont=ev_c, evals_jump=ev_j,
        k_total=k_t, k_cont=k_c, k_jump=k_j,
        r2_k=r2_k.astype(np.float32),
        jump_frac=float(is_jump.mean()),
        jump_qv_share=float((XD ** 2).sum() / (X ** 2).sum()),
    )
    print(f"[{ym}] N={N} (dropped {int((~keep).sum())} zero-QV)  "
          f"K={k_t.tolist()} K_C={k_c.tolist()} K_D={k_j.tolist()}  "
          f"jump {100*is_jump.mean():.2f}% of cells / "
          f"{100*(XD**2).sum()/(X**2).sum():.1f}% of QV", flush=True)


def main():
    months = sys.argv[1:] or sorted(
        p.name[6:13] for p in CACHE.glob("panel_*.npz"))
    for ym in months:
        estimate_month(ym)


if __name__ == "__main__":
    main()
