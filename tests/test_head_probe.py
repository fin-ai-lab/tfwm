"""Tests for the offline head probe (``market_jepa.eval.head_probe``).

The load-bearing claims are (a) the fitted head consumes RAW embeddings, so it
is a drop-in for a checkpoint's ``head.pt`` and for ``EventPredictor``, and (b)
the fit is strong enough that a linear head reaches the ridge it is the SGD
twin of — without which the whole comparison would be measuring the optimizer.
"""

import numpy as np
import pytest
import torch

from market_jepa.eval.head_probe import (
    ProbeSpec, _fold_standardizer, build_head, fit_head_probe, head_scores,
)
from market_jepa.eval.metrics import grouped_rank_ic
from market_jepa.modeling.event_predictor import EventPredictor


def _panel(n_days=24, n_anchor=6, n_stock=60, d=32, seed=0):
    """A synthetic cross-sectional panel: linear signal + a nonlinear term."""
    rng = np.random.RandomState(seed)
    dates = np.repeat([f"2014-05-{i + 1:02d}" for i in range(n_days)],
                      n_anchor * n_stock)
    anchors = np.tile(np.repeat(np.arange(n_anchor), n_stock), n_days)
    cells = np.char.add(np.char.add(dates, "@"), anchors.astype(str))
    n = len(cells)
    # Deliberately off-centre and anisotropic, so a head that forgot the
    # standardizer fold would score visibly worse rather than by a rounding.
    X = (rng.randn(n, d) * rng.uniform(0.5, 3.0, d) + 4.0).astype(np.float32)
    w = rng.randn(d) / np.sqrt(d)
    y = (X @ w + 0.3 * np.tanh(X[:, 0]) + 2.0 * rng.randn(n)).astype(np.float32)
    for c in np.unique(cells):
        m = cells == c
        y[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-9)
    val = np.isin(dates, np.unique(dates)[-5:])
    return X, y, cells, val


def test_spec_parse_and_reject():
    assert ProbeSpec.parse("mlp:corr:cell").name == "mlp:corr:cell"
    assert ProbeSpec.parse("linear:mse").batching == "flat"
    with pytest.raises(ValueError):
        # A pointwise loss does not see the grouping, so a "cell" variant would
        # be the same fit under a second name.
        ProbeSpec.parse("mlp:mse:cell")
    with pytest.raises(ValueError):
        ProbeSpec.parse("mlp:hinge")


def test_standardizer_fold_is_exact():
    X, _, _, val = _panel()
    mu, sd = X[~val].mean(0), X[~val].std(0)
    head = build_head(ProbeSpec("mlp", "mse"), X.shape[1])
    with torch.no_grad():
        before = head(torch.from_numpy((X[:64] - mu) / sd).float()).numpy()
    _fold_standardizer(head, mu, sd)
    with torch.no_grad():
        after = head(torch.from_numpy(X[:64]).float()).numpy()
    assert np.abs(before - after).max() < 1e-5


def test_linear_head_reaches_the_ridge():
    """The SGD twin of the probe must land on the probe, or the grid is too small."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, y, cells, val = _panel()
    sc = StandardScaler().fit(X[~val])
    ridge = Ridge(alpha=1.0).fit(sc.transform(X[~val]), y[~val])
    ridge_ic, _, _ = grouped_rank_ic(
        ridge.predict(sc.transform(X[val])), y[val], cells[val])

    fit = fit_head_probe(ProbeSpec("linear", "mse"), X, y, cells, val,
                         device="cpu", epochs=25, batch_rows=1024)
    # Scored through the FOLDED head on raw features, which is also what makes
    # this a test of the fold and not only of the fit.
    ic, _, _ = grouped_rank_ic(
        head_scores(fit["head"], X[val], torch.device("cpu")), y[val], cells[val])
    assert ic > ridge_ic - 0.02, f"linear head {ic:.4f} vs ridge {ridge_ic:.4f}"


def test_cell_batching_beats_flat_for_a_ranking_loss():
    """Ranking inside a cross-section is the metric's own question.

    Flat batching ranks stocks drawn from different instants; on a panel whose
    labels are z-scored within the cell, that is a surrogate. This asserts the
    surrogate is not better, which is the premise the ``cell`` arm rests on.
    """
    X, y, cells, val = _panel()
    kw = dict(device="cpu", epochs=12, batch_rows=1024, batch_cells=4)
    flat = fit_head_probe(ProbeSpec("mlp", "corr", "flat"), X, y, cells, val, **kw)
    cell = fit_head_probe(ProbeSpec("mlp", "corr", "cell"), X, y, cells, val, **kw)
    assert cell["val_ic"] > flat["val_ic"]


def test_fitted_head_drops_into_event_predictor():
    """The point of fitting a head shape rather than any regressor."""
    X, y, cells, val = _panel()
    fit = fit_head_probe(ProbeSpec("mlp", "corr", "cell"), X, y, cells, val,
                         device="cpu", lrs=(1e-2,), epochs=2, batch_cells=4)
    pred = EventPredictor(fit["head"], emb_dim=64, use_market=True, use_text=True)
    cls = torch.from_numpy(X[:8]).float()
    market_only = pred(cls=cls, emb=torch.randn(8, 64))
    # The text branch is zero-init, so the predictor starts as the head itself.
    with torch.no_grad():
        direct = fit["head"](cls)
    assert torch.allclose(market_only.reshape(-1), direct.reshape(-1), atol=1e-5)


def test_fixed_recipe_uses_every_row_and_selects_nothing():
    """The deployment path: no hold-out, no per-month hyperparameter choice.

    Selecting one hyperparameter on one month's validation slice costs about
    0.003 of return IC (measured on the ridge, which has no other moving part),
    and the hold-out costs ~0.001 more in fitting rows. A recipe fixed once and
    applied unchanged pays neither, so it has to be a first-class path rather
    than something a caller improvises.
    """
    X, y, cells, _ = _panel()
    fit = fit_head_probe(ProbeSpec("mlp", "corr", "cell"), X, y, cells, None,
                         device="cpu", lrs=(1e-3,), epochs=3, batch_cells=4)
    assert fit["fixed"] is True
    assert fit["n_fit_rows"] == len(X)          # every row trains
    assert fit["epoch"] == 2                    # the prescribed last epoch
    assert np.isnan(fit["val_ic"])              # nothing was selected on
    assert not fit["at_lr_edge"] and not fit["at_epoch_edge"]


def test_fixed_recipe_needs_exactly_one_learning_rate():
    X, y, cells, _ = _panel()
    with pytest.raises(ValueError, match="one learning rate"):
        fit_head_probe(ProbeSpec("mlp", "mse"), X, y, cells, None,
                       device="cpu", lrs=(1e-3, 1e-2), epochs=1)


def test_linear_head_cannot_serve_event_conditioning():
    """A linear head has no hidden layer for the text branch to sum into.

    This is why the best RETURN readout on the panel (linear:pairwise:flat) is
    not a candidate for event conditioning: EventPredictor injects g(emb) at
    mlp[0]'s output, and a bare Linear(d, 1) has no such junction. Whatever
    head ships for the return channel has to come from the ``mlp`` family.
    """
    head = build_head(ProbeSpec("linear", "pairwise"), 32)
    assert len(head.mlp) < 5
    with pytest.raises(TypeError, match="torchvision-style"):
        EventPredictor(head, emb_dim=64)
