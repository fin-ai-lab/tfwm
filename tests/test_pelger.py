"""Unit tests for the Pelger high-frequency factor estimators."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots" / "latent_eval" / "factors"))

from pelger import (  # noqa: E402
    correlation_pca,
    generalized_correlations,
    jump_threshold,
    local_vol,
    pca_factor_model,
    perturbed_er_count,
    split_continuous_jump,
)


def synth_panel(seed=0, N=120, D=20, m=78, K=3, idio=0.5, jumps=0):
    """Factor-model increments: X = F Lam' + e, per-day layout D x m."""
    rng = np.random.default_rng(seed)
    M = D * m
    strengths = np.array([3.0, 2.0, 1.2])[:K]
    F = rng.standard_normal((M, K)) * strengths / np.sqrt(m)
    Lam = rng.standard_normal((N, K))
    Lam[:, 0] = np.abs(Lam[:, 0]) + 0.5          # market: all-positive loadings
    e = rng.standard_normal((M, N)) * idio / np.sqrt(m)
    X = F @ Lam.T + e
    jump_mask = np.zeros((M, N), dtype=bool)
    if jumps:
        # 10x each asset's own increment std, so detectability is uniform
        sd = np.sqrt((Lam ** 2 * strengths ** 2).sum(1) + idio ** 2) / np.sqrt(m)
        rows = rng.integers(0, M, jumps)
        cols = rng.integers(0, N, jumps)
        signs = rng.choice([-1.0, 1.0], jumps)
        X[rows, cols] += signs * 10.0 * sd[cols]
        jump_mask[rows, cols] = True
    return X, F, Lam, jump_mask


def test_pca_normalization_identities():
    X, _, _, _ = synth_panel()
    out = pca_factor_model(X, K=3)
    N = X.shape[1]
    np.testing.assert_allclose(
        out["loadings"].T @ out["loadings"] / N, np.eye(3), atol=1e-10)
    FtF = out["factors"].T @ out["factors"]
    np.testing.assert_allclose(FtF, np.diag(np.diag(FtF)), atol=1e-10)
    np.testing.assert_allclose(np.diag(FtF), out["eigenvalues"][:3], rtol=1e-10)
    np.testing.assert_allclose(out["common"] + out["residuals"], X, atol=1e-12)


def test_er_count_recovers_k():
    X, _, _, _ = synth_panel()
    out = correlation_pca(X, K=3)
    K_hat, er, g = perturbed_er_count(out["eigenvalues"], eps=0.08)
    assert K_hat == 3
    assert g > 0
    K_hat20, _, _ = perturbed_er_count(out["eigenvalues"], eps=0.20)
    assert K_hat20 == 3


def test_er_count_zero_on_noise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1500, 100))
    out = correlation_pca(X, K=1)
    K_hat, _, _ = perturbed_er_count(out["eigenvalues"], eps=0.08)
    assert K_hat == 0


def test_factor_space_recovery():
    X, F, Lam, _ = synth_panel()
    out = correlation_pca(X, K=3)
    rho, tot = generalized_correlations(out["factors"], F)
    assert rho.min() > 0.95
    # loadings of standardized returns span D^{-1/2} Lam
    lam_std = Lam / out["scale"][:, None]
    rho_l, _ = generalized_correlations(out["loadings"], lam_std)
    assert rho_l.min() > 0.95


def test_correlation_pca_rejects_zero_qv():
    X, _, _, _ = synth_panel()
    X[:, 0] = 0.0
    with pytest.raises(ValueError, match="zero realized variation"):
        correlation_pca(X, K=2)


def test_generalized_correlation_bounds():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((500, 3))
    rho, tot = generalized_correlations(F, F @ rng.standard_normal((3, 3)))
    np.testing.assert_allclose(rho, 1.0, atol=1e-8)
    assert abs(tot - 3.0) < 1e-6
    G = rng.standard_normal((500, 3))
    rho2, _ = generalized_correlations(F, G)
    assert rho2.max() < 0.3


def test_jump_split_catches_injected_jumps():
    X, _, _, jump_mask = synth_panel(seed=3, jumps=200, idio=0.3)
    D, m = 20, 78
    thr = np.empty_like(X)
    for i in range(X.shape[1]):
        thr[:, i] = jump_threshold(X[:, i].reshape(D, m), a=3.0).ravel()
    XC, XD, is_jump = split_continuous_jump(X, thr)
    np.testing.assert_allclose(XC + XD, X)
    assert (XC[is_jump] == 0).all() and (XD[~is_jump] == 0).all()
    hit = is_jump[jump_mask].mean()
    assert hit > 0.90                       # injected 20-sigma jumps flagged
    false_pos = is_jump[~jump_mask].mean()
    assert false_pos < 0.02                 # ~a=3 tail + factor moves only


def test_local_vol_recovers_tod_profile():
    rng = np.random.default_rng(4)
    D, m = 120, 78
    tod_true = 1.0 + 1.5 * np.cos(np.linspace(0, 2 * np.pi, m)) ** 2
    tod_true /= tod_true.mean()
    R = rng.standard_normal((D, m)) * np.sqrt(tod_true / m)
    sig = local_vol(R)
    assert sig.shape == (D, m) and (sig > 0).all()
    prof = (sig ** 2).mean(0)
    prof /= prof.mean()
    corr = np.corrcoef(prof, tod_true)[0, 1]
    assert corr > 0.95
