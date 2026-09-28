"""The correlation loss must repel the degenerate constant solution.

This loss exists because both pointwise losses are minimized at the
conditional mean, and the cross-sectional rank target is zero-mean in every
cell by construction — so predicting a constant is very nearly optimal for
them. Measured against the analytic constant-0 loss, the best achievable
training loss was ~78% of trivial for spread_change, ~97% for
volatility_change and ~100% for return. Runs collapsed onto an identical
degenerate solution and return never escaped it at all.

Every test here pins a property that failing would restore that behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_jepa.modeling.modes.supervised import (  # noqa: E402
    LOSS_FNS, SupervisedModel, _regression_loss,
)


def _target(n=4096, seed=0):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


def test_corr_is_registered():
    assert "corr" in LOSS_FNS


@pytest.mark.parametrize("c", [0.0, 0.5, -1.0, 2.0])
def test_constant_prediction_is_not_near_optimal(c):
    """A constant scores 1.0 — the MIDPOINT, not ~0.

    Not the worst value (anti-correlation is 2.0); the point is that it is no
    longer NEARLY OPTIMAL the way it is under mse/smooth_l1, where the whole
    gap between trivial and achievable was 0-22%. Here a full unit of
    improvement is available. Critically it is independent of WHICH constant,
    so there is no constant the model can tune its way into.
    """
    t = _target()
    loss = _regression_loss(torch.full_like(t, c), t, "corr", 1.0)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.0, abs=1e-4)


def test_perfect_prediction_is_zero_and_anticorrelated_is_two():
    t = _target()
    assert _regression_loss(t, t, "corr", 1.0).item() == pytest.approx(0.0, abs=1e-5)
    assert _regression_loss(-t, t, "corr", 1.0).item() == pytest.approx(2.0, abs=1e-5)


def test_scale_free():
    """Rank IC reads ordering only, so the head's output scale must not matter.

    This is also what makes the loss safe to use with an unconstrained head:
    it cannot be gamed by shrinking or inflating the output.
    """
    t = _target()
    base = _regression_loss(t, t, "corr", 1.0)
    for k in (1e-3, 1e3):
        assert _regression_loss(t * k, t, "corr", 1.0).item() == pytest.approx(
            base.item(), abs=1e-4)


def test_monotone_in_signal_fraction():
    t = _target()
    noise = _target(seed=1)
    losses = [
        _regression_loss(t + w * noise, t, "corr", 1.0).item()
        for w in (0.0, 0.5, 1.0, 4.0)
    ]
    assert losses == sorted(losses), losses


def test_gradient_is_finite_and_bounded_at_collapse():
    """A correlation's gradient scales as 1/sd(pred) and is 0/0 at a constant.

    Unbounded here would be fatal rather than merely ugly: the collapse this
    loss targets is exactly the low-variance regime, so the gradient blows up
    precisely where the model spends its time. Without the floor this measured
    1.6e6 at sd=1e-8 and NaN at sd=0.
    """
    t = _target()
    prev = None
    for sd in (1.0, 1e-2, 1e-5, 1e-8, 0.0):
        p = (_target(seed=2) * sd).requires_grad_(True)
        _regression_loss(p, t, "corr", 1.0).backward()
        g = p.grad.norm().item()
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at sd={sd}"
        assert g < 1e3, f"gradient {g:.3e} unbounded at sd={sd}"
        prev = g
    assert prev is not None


def test_pointwise_losses_still_prefer_the_constant():
    """Guards the premise. If this ever fails, the diagnosis changed."""
    t = _target()
    const = torch.zeros_like(t)
    for fn in ("mse", "smooth_l1"):
        trivial = _regression_loss(const, t, fn, 1.0)
        perfect = _regression_loss(t, t, fn, 1.0)
        # A constant is BAD but not catastrophically so — within one unit of
        # perfect — which is the small gradient budget the collapse exploits.
        assert (trivial - perfect).item() < 1.0


# ── pairwise ranking loss ────────────────────────────────────────────────────


def test_pairwise_is_registered():
    assert "pairwise" in LOSS_FNS


@pytest.mark.parametrize("c", [0.0, 0.5, -1.0])
def test_pairwise_constant_scores_log2_for_any_constant(c):
    """Every pair is a tie, so every pair pays softplus(0) = log 2."""
    import math

    t = _target()
    loss = _regression_loss(torch.full_like(t, c), t, "pairwise", 1.0)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-4)


def test_pairwise_orders_correctly():
    t = _target()
    good = _regression_loss(t, t, "pairwise", 1.0).item()
    const = _regression_loss(torch.zeros_like(t), t, "pairwise", 1.0).item()
    bad = _regression_loss(-t, t, "pairwise", 1.0).item()
    assert good < const < bad


def test_pairwise_gradient_at_collapse_points_at_the_target_rank():
    """The property that motivates this loss over the correlation one.

    At a perfectly constant prediction the correlation loss is 0/0 and needs a
    variance floor; this one is smooth, and each sample's gradient is
    proportional to (n_below - n_above) — i.e. straight at its rank.
    """
    t = _target(n=512)
    p = torch.zeros(512, requires_grad=True)
    _regression_loss(p, t, "pairwise", 1.0).backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.norm().item() > 0
    # -grad is the direction of improvement; it should track the target.
    r = torch.corrcoef(torch.stack([-p.grad, t]))[0, 1].item()
    assert r > 0.9, f"gradient does not point at the rank (corr {r:.3f})"


def test_pairwise_handles_all_ties_without_nan():
    """A degenerate batch (every target equal) must not produce NaN."""
    t = torch.zeros(64)
    loss = _regression_loss(torch.randn(64), t, "pairwise", 1.0)
    assert torch.isfinite(loss) and loss.item() == 0.0


# ── grouped (within-cell) losses ─────────────────────────────────────────────


class _Grouped(SupervisedModel):
    """Bare instance exposing compute_group_loss without building a backbone."""

    def __init__(self, loss_fn):
        torch.nn.Module.__init__(self)
        self.target_col_idx = 0
        self.loss_fn = loss_fn
        self.smooth_l1_beta = 1.0


def _cell_batch(B=4, K=8, T=3, seed=0):
    return torch.randn(B, K, T, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("fn", ["pairwise", "corr"])
def test_grouped_orders_correctly(fn):
    tg = _cell_batch()
    y = tg[:, :, 0]
    m = _Grouped(fn)
    good = m.compute_group_loss(y.clone(), tg)[0].item()
    const = m.compute_group_loss(torch.zeros_like(y), tg)[0].item()
    bad = m.compute_group_loss(-y.clone(), tg)[0].item()
    assert good < const < bad


@pytest.mark.parametrize("fn", ["pairwise", "corr"])
def test_grouped_gradient_finite_at_collapse(fn):
    """The 0*inf trap: clamping OUTSIDE sqrt() made corr's gradient NaN here.

    torch.maximum backprops through both branches and multiplies the unselected
    one by 0; with sqrt() outside the clamp that branch's derivative is inf at
    zero variance, and 0*inf = NaN. The floor has to live inside the sqrt.
    """
    tg = _cell_batch()
    p = torch.zeros(tg.shape[0], tg.shape[1], requires_grad=True)
    loss, _ = _Grouped(fn).compute_group_loss(p, tg)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all(), f"{fn}: non-finite gradient at collapse"
    assert p.grad.norm() > 0


@pytest.mark.parametrize("fn", ["pairwise", "corr"])
def test_grouped_handles_partial_and_total_nan(fn):
    tg = _cell_batch()
    y = tg[:, :, 0].clone()
    partial = tg.clone()
    partial[:, 0, 0] = float("nan")          # one stock unlabelled per cell
    loss, n = _Grouped(fn).compute_group_loss(y, partial)
    assert torch.isfinite(loss) and n > 0

    total = tg.clone()
    total[:, :, 0] = float("nan")            # no labels at all
    loss, n = _Grouped(fn).compute_group_loss(y, total)
    assert n == 0 and loss.item() == 0.0


def test_grouped_pointwise_falls_through_to_the_flat_loss():
    """mse/smooth_l1 must NOT be grouped — that keeps "same batches, different
    loss" a controlled comparison instead of confounding loss with sampler."""
    tg = _cell_batch()
    y = tg[:, :, 0]
    loss, n = _Grouped("mse").compute_group_loss(y.clone(), tg)
    assert n == tg.shape[0] * tg.shape[1]
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
