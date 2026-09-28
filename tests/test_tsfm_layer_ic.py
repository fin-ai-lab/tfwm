"""The streaming ridge must equal the one xs_ic_eval.score fits.

The TSFM layer sweep never materializes its design matrix — it accumulates
sufficient statistics and solves from them. That is only legitimate if the
weights it produces are the SAME weights ``StandardScaler`` + ``Ridge`` would
have produced on the full matrix, because that pair is the reported estimator
for every other method. These tests pin that equality, including the two places
it could silently break: the float32 GEMM feeding a float64 accumulator, and
the (m0, s0) origin the accumulation runs in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

from tsfm_layer_ic import LayerStats  # noqa: E402

ALPHAS = [1.0, 10.0, 1e3, 1e5]


def _reference(X, Y, alphas):
    """StandardScaler().fit(X) then Ridge(alpha).fit — the reported estimator."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    W = np.zeros((X.shape[1], len(alphas), Y.shape[1]))
    for j, a in enumerate(alphas):
        for k in range(Y.shape[1]):
            W[:, j, k] = Ridge(alpha=a).fit(Xs, Y[:, k]).coef_
    return sc.mean_, sc.scale_, W


def _stream(X, Y, alphas, chunk, warmup=64):
    """Feed X through LayerStats the way fit_pass does."""
    st = LayerStats(X.shape[1], Y.shape[1])
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(Y, dtype=torch.float32)
    st.set_origin(xt[:warmup])
    for s in range(0, len(X), chunk):
        e = min(len(X), s + chunk)
        st.add(st.u(xt[s:e]), yt[s:e])
    return st.solve(alphas)[:4]


@pytest.mark.parametrize("chunk", [37, 250, 10_000])
def test_matches_sklearn(chunk):
    """Same weights, same scaler, regardless of how the rows were chunked."""
    rng = np.random.default_rng(0)
    n, d, t = 4000, 40, 3
    # Features with wildly different means and scales: the case where the
    # textbook `A - n mu mu^T` centering loses its significant digits.
    X = (rng.normal(size=(n, d)) * rng.uniform(0.01, 5, d)
         + rng.uniform(-500, 500, d))
    Y = X[:, :t] @ rng.normal(size=(t, t)) + rng.normal(size=(n, t))

    mu_r, sd_r, W_r = _reference(X, Y, ALPHAS)
    mu, sd, W, ybar = _stream(X, Y, ALPHAS, chunk)

    assert np.allclose(mu, mu_r, rtol=1e-4, atol=1e-4)
    assert np.allclose(sd, sd_r, rtol=1e-4)
    assert np.allclose(ybar, Y.mean(axis=0), rtol=1e-4)
    assert np.allclose(W, W_r, rtol=2e-3, atol=2e-4)


def test_predictions_match_end_to_end():
    """What the eval pass computes equals scaler+ridge predictions."""
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(3000, 25)) * 3 + 100
    Ytr = Xtr[:, :2] @ rng.normal(size=(2, 2)) + rng.normal(size=(3000, 2))
    Xev = rng.normal(size=(500, 25)) * 3 + 100

    mu_r, sd_r, W_r = _reference(Xtr, Ytr, ALPHAS)
    mu, sd, W, ybar = _stream(Xtr, Ytr, ALPHAS, 512)

    got = ((Xev - mu) / sd) @ W.reshape(W.shape[0], -1)
    want = ((Xev - mu_r) / sd_r) @ W_r.reshape(W_r.shape[0], -1)
    # Rank IC only cares about ordering, but pin the values: a scaler bug that
    # preserved ranks on synthetic data would not on real data.
    assert np.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_rank_deficient_gram_still_solves():
    """Collinear features must not take the run down.

    Cholesky raised "leading minor is not positive definite" on a real Sundial
    layer and killed the job; the eigendecomposition clips the offending
    eigenvalues instead. Exact collinearity is the extreme case of what a
    9-channel concat produces naturally.
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1500, 20))
    X[:, 10:] = X[:, :10]              # rank 10 of 20 — exactly singular
    Y = X[:, :1] + rng.normal(size=(1500, 1))
    st = LayerStats(X.shape[1], 1)
    xt = torch.as_tensor(X, dtype=torch.float32)
    st.set_origin(xt[:128])
    st.add(st.u(xt), torch.as_tensor(Y, dtype=torch.float32))
    mu, sd, W, ybar, n_clipped = st.solve([1.0, 1e4])
    assert np.isfinite(W).all()
    # It still predicts: a duplicated column carries the same signal.
    pred = ((X - mu) / sd) @ W[:, 1, 0]
    assert np.corrcoef(pred, Y[:, 0])[0, 1] > 0.5


def test_constant_feature_is_dropped_not_nan():
    """A dead hidden unit must zero out, not poison every prediction."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 10))
    X[:, 3] = 7.0                     # constant: sd = 0
    Y = X[:, :1] + rng.normal(size=(800, 1))
    mu, sd, W, _ = _stream(X, Y, [10.0], 128)
    assert np.isfinite(W).all()
    assert np.isfinite(sd).all() and (sd > 0).all()
    assert W[3].max() == 0.0


def test_origin_choice_does_not_move_the_answer():
    """(m0, s0) is a numerical device, not part of the estimator."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 15)) * 2 + 50
    Y = rng.normal(size=(2000, 1)) + X[:, :1]

    a = _stream(X, Y, ALPHAS, 256, warmup=32)
    b = _stream(X, Y, ALPHAS, 256, warmup=1500)
    for u, v in zip(a[:3], b[:3]):
        assert np.allclose(u, v, rtol=1e-3, atol=1e-4)


def test_agrees_with_the_reported_scorer():
    """The streamed ridge must reproduce xs_ic_eval.score's IC exactly.

    xs_ic_eval.score is THE scorer — every arm of the paper reaches it. The
    layer sweep cannot call it (it never materializes X), so the one thing that
    keeps the two from drifting is this: same panel, same targets, same
    numbers, to the fourth decimal a result is quoted at.
    """
    import xs_ic_eval

    rng = np.random.default_rng(7)
    n_tr, n_ev, d = 3000, 900, 30
    names = [f"{t}_{h}" for t in xs_ic_eval.TARGET_TYPES
             for h in xs_ic_eval.HORIZONS]
    col = names.index("volatility_change_900")
    alpha = xs_ic_eval.ridge_alpha_for("volatility_change_900")

    def panel(n, seed):
        r = np.random.default_rng(seed)
        X = r.normal(size=(n, d)) * 4 + 20
        z = np.full((n, len(names)), np.nan)
        z[:, col] = X[:, 0] * 0.3 + r.normal(size=n)
        return {"X": X.astype(np.float32), "z": z.astype(np.float32),
                "date": np.array([f"2009-02-{1 + i % 5:02d}" for i in range(n)]),
                "anchor": np.array([50000 + 900 * (i % 6) for i in range(n)]),
                "target_names": np.array(names)}

    tr, ev = panel(n_tr, 1), panel(n_ev, 2)
    want = xs_ic_eval.score(tr, ev)["volatility_change_900"]["ic"]

    st = LayerStats(d, 1)
    xt = torch.as_tensor(tr["X"], dtype=torch.float32)
    st.set_origin(xt[:256])
    st.add(st.u(xt), torch.as_tensor(tr["z"][:, [col]], dtype=torch.float32))
    mu, sd, W, _, _ = st.solve([alpha])
    pred = ((ev["X"] - mu) / sd) @ W.reshape(d, -1)

    cells = np.char.add(np.char.add(ev["date"], "@"), ev["anchor"].astype(str))
    from market_jepa.eval.metrics import grouped_rank_ic
    got = grouped_rank_ic(pred[:, 0], ev["z"][:, col], cells)[0]
    assert abs(got - want) < 1e-4, f"{got} vs {want}"


def test_the_info_token_asymmetry_is_declared_at_both_call_sites():
    """TSFMs take nine channels and say so; the baselines take twenty and say so.

    THE ASYMMETRY IS DELIBERATE (2026-09-11): a frozen TSFM tokenizes a
    fixed-width series and has nowhere to route an information token, so it is
    withheld -- which is why a TSFM delta is a LOWER bound. The classical view
    learners do get it, because the level, spread width and activity that
    per_view normalization divides out are exactly what a classical predictor
    would keep.

    Both halves must be written down rather than inherited. panel_source
    defaults both flags to False, so the TSFM side reading correctly was
    indistinguishable from the argument simply being unset -- and the day that
    default flips (as StreamingMarketDataset's did on 2026-09-13) the frozen
    models silently gain eleven channels they cannot use.
    """
    import inspect
    from pathlib import Path

    src = Path("scripts/eval/tsfm_layer_ic.py").read_text()
    calls = src.count("panel_source(")
    # Every panel_source call in the TSFM scorer pins the flags off.
    assert calls == src.count("info_norm_stats=False, info_window=False"), (
        f"{calls} panel_source call(s) in tsfm_layer_ic.py but not all pin the "
        "information-token flags; 9-channel must be declared, not defaulted."
    )

    # And the baselines pin them ON, at their own read.
    tables = Path("plots/finance_baselines/panel_tables.py")
    if tables.is_file():                       # plots/ is optional at test time
        t = tables.read_text()
        assert "info_norm_stats=True, info_window=True" in t, (
            "the classical view learners must request the information token "
            "explicitly from panel_source"
        )


def test_panel_source_still_defaults_the_info_flags_off():
    """The default both sides are pinned AGAINST.

    If this ever flips, the TSFM pins above become load-bearing rather than
    merely explicit -- and anything else calling panel_source without the flags
    changes width silently. Asserted so the flip cannot be quiet.
    """
    import inspect
    import sys
    sys.path.insert(0, "scripts/eval")
    import xs_ic_eval

    sig = inspect.signature(xs_ic_eval.panel_source)
    assert sig.parameters["info_norm_stats"].default is False
    assert sig.parameters["info_window"].default is False
