"""Pelger (2018) high-frequency latent factor estimators.

Implements the estimation core of *Large-dimensional factor modeling based
on high-frequency observations*: PCA of the realized quadratic-correlation
matrix of intraday increments, the perturbed eigenvalue-ratio factor count,
the elementwise continuous/jump split with a TOD x bipower local-volatility
threshold, and generalized (canonical) correlations between factor spaces.

Conventions follow the paper's empirical specification:
  * correlation version (each asset standardized by sqrt of its realized
    quadratic variation) is the default,
  * loadings normalized as Lambda = sqrt(N) * U_K, factors F = X Lambda / N,
  * ERP1 perturbation g = sqrt(N) * median(eigenvalues), cutoff 1 + eps
    with eps = 0.08,
  * jump threshold u_{j,i} = a * n_incr^{-0.49} * sigma_hat_{j,i} with a = 3
    and sigma_hat^2_{j,i} = BV_{day(j),i} * TOD_{slot(j),i}.

The TOD estimator is a truncated mean-of-squares per intraday slot (the
paper defers to Bollerslev-Li-Todorov for the exact construction; this is a
documented implementation choice, not the paper's verbatim estimator).
"""
from __future__ import annotations

import numpy as np


def pca_factor_model(X: np.ndarray, K: int) -> dict:
    """PCA estimator on an M x N increment matrix (no demeaning).

    Returns eigenvalues of X'X/N (descending), loadings (N x K,
    Lambda = sqrt(N) U_K), factors (M x K, F = X Lambda / N), the fitted
    common component and residuals.
    """
    M, N = X.shape
    S = (X.T @ X) / N
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    U = evecs[:, :K]
    loadings = np.sqrt(N) * U
    factors = (X @ loadings) / N
    common = factors @ loadings.T
    return {
        "eigenvalues": evals,
        "loadings": loadings,
        "factors": factors,
        "common": common,
        "residuals": X - common,
    }


def correlation_pca(Y: np.ndarray, K: int) -> dict:
    """Paper's empirical variant: PCA on volatility-standardized increments.

    Assets with zero realized quadratic variation are not allowed (the paper
    gives no special rule) — drop them before calling.
    """
    q = np.sum(Y * Y, axis=0)
    if np.any(q <= 0):
        raise ValueError("zero realized variation — drop those assets first")
    scale = np.sqrt(q)
    Z = Y / scale[None, :]
    out = pca_factor_model(Z, K)
    out["scale"] = scale
    out["loadings_original_units"] = scale[:, None] * out["loadings"]
    return out


def perturbed_er_count(
    eigenvalues: np.ndarray,
    eps: float = 0.08,
    g_mode: str = "sqrtN_median",
) -> tuple[int, np.ndarray, float]:
    """Perturbed eigenvalue-ratio factor count from a descending spectrum.

    K_hat = max{k : (lam_k + g)/(lam_{k+1} + g) > 1 + eps}, 0 if none.
    Returns (K_hat, the ratio series, g).
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    N = len(lam)
    med = float(np.median(lam))
    if g_mode == "sqrtN_median":
        g = np.sqrt(N) * med
    elif g_mode == "logN_median":
        g = np.log(N) * med
    else:
        raise ValueError(f"unknown g_mode {g_mode!r}")
    if g <= 0:
        raise ValueError("median eigenvalue is nonpositive; paper gives no fallback")
    er = (lam[:-1] + g) / (lam[1:] + g)
    hits = np.flatnonzero(er > 1.0 + eps)
    K_hat = 0 if len(hits) == 0 else int(hits.max() + 1)
    return K_hat, er, g


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = (w // 2, w - 1 - w // 2)
    return np.convolve(np.pad(x, pad, mode="reflect"), np.ones(w) / w, "valid")


def local_vol(
    R_days: np.ndarray, trunc_mult: float = 5.0, tod_smooth: int = 5
) -> np.ndarray:
    """Spot-volatility estimate sigma_hat (D x m) for one asset.

    ``R_days``: D x m matrix of intraday increments (D days, m increments
    per day; NaN = missing). sigma_hat^2_{d,j} = BV_d * TOD_j where BV_d is
    the day's bipower variation (jump-robust integrated variance, per-day
    units) and TOD_j is the time-of-day profile normalized to mean 1,
    estimated as a truncated mean of squares (increments beyond
    ``trunc_mult`` global stds excluded) smoothed over ``tod_smooth``
    adjacent slots.
    """
    R = np.asarray(R_days, dtype=float)
    D, m = R.shape
    absr = np.abs(R)
    bv = (np.pi / 2.0) * np.nansum(absr[:, 1:] * absr[:, :-1], axis=1)
    bv *= m / np.maximum(m - 1, 1)          # small-sample scale for m-1 terms
    # fall back to the asset's median BV on degenerate days
    med_bv = np.median(bv[bv > 0]) if np.any(bv > 0) else 0.0
    bv = np.where(bv > 0, bv, med_bv)

    glob_sd = np.sqrt(max(med_bv, 1e-300) / m)
    keep = absr <= trunc_mult * glob_sd
    with np.errstate(invalid="ignore"):
        tod = np.nanmean(np.where(keep, R * R, np.nan), axis=0)
    good = np.isfinite(tod) & (tod > 0)
    tod = np.where(good, tod, tod[good].mean()) if good.any() else np.ones(m)
    tod = _smooth(tod, tod_smooth)
    tod = tod / tod.mean() if tod.mean() > 0 else np.ones(m)

    return np.sqrt(np.maximum(bv[:, None] * tod[None, :], 0.0))


def jump_threshold(
    R_days: np.ndarray, a: float = 3.0, omega: float = 0.49
) -> np.ndarray:
    """Elementwise threshold u_{d,j} = a * m^{-omega} * sigma_hat_{d,j}."""
    D, m = R_days.shape
    return a * (float(m) ** -omega) * local_vol(R_days)


def split_continuous_jump(
    X: np.ndarray, threshold: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X = XC + XD elementwise: XD keeps increments with |x| > threshold."""
    is_jump = np.abs(X) > threshold
    XD = np.where(is_jump, X, 0.0)
    XC = np.where(is_jump, 0.0, X)
    return XC, XD, is_jump


def generalized_correlations(
    F: np.ndarray, G: np.ndarray, return_vectors: bool = False
):
    """Generalized (canonical) correlations between the spans of F and G.

    No demeaning (paper convention for increment matrices; for
    cross-sectional loading/embedding matrices, demean the inputs yourself
    if that is the comparison you want). Returns (rho descending, sum rho^2).

    With ``return_vectors``, also returns the canonical weight matrices
    (WF, WG) whose k-th columns give the paired canonical variates F @ WF[:, k]
    and G @ WG[:, k], correlated at rho_k. The k-th canonical direction is a
    *rotation* of G's span, not G's k-th column — needed to say which raw
    column of G a given canonical direction actually resembles.
    """
    A = F.T @ F
    B = G.T @ G
    C = F.T @ G
    da, Ua = np.linalg.eigh(A)
    db, Ub = np.linalg.eigh(B)
    if da.min() <= 0 or db.min() <= 0:
        raise ValueError("singular Gram matrix — reduce the rank first")
    A_mh = Ua @ np.diag(da ** -0.5) @ Ua.T
    B_mh = Ub @ np.diag(db ** -0.5) @ Ub.T
    M = A_mh @ C @ B_mh
    if not return_vectors:
        rho = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
        return rho, float(np.sum(rho ** 2))
    U, rho, Vt = np.linalg.svd(M)
    rho = np.clip(rho, 0.0, 1.0)
    return rho, float(np.sum(rho ** 2)), A_mh @ U, B_mh @ Vt.T
