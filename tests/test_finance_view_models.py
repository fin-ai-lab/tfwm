"""The view ridge is solved from streamed normal equations, not by sklearn.

That solver is ~80 lines of Gram arithmetic standing in for one ``Ridge.fit``,
and every part of it is a place a silent wrong answer can hide: syrk only
accumulates in place on a Fortran-ordered C, it fills one triangle, the
centring is folded into the penalty rather than applied to the rows, and each
target is fitted on its own censoring shell. None of that shows up as an
exception -- it shows up as an IC that is merely plausible.

So the test is equivalence: on data small enough for the dense path, the
streamed solver must reproduce ``StandardScaler`` + ``Ridge`` exactly, INCLUDING
on targets censored to different subsets of rows.

The learners read a TAIL of the view (``panel_tables.VIEW_TAIL_TOKENS``), so
the width under test is ``view_features(tail)`` for a small tail rather than a
patched-down constant -- 47 columns here, the same arithmetic as 18,432.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "plots" / "finance_baselines"),
           str(_REPO / "scripts" / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# A tail small enough that the Gram is 47x47 rather than 18,432x18,432, and
# real: the solver sees a width it could actually be constructed with.
TAIL = 4


@pytest.fixture
def vm():
    import view_models
    return view_models


def _fit_streamed(vm, X, Y, names, buf_rows=128, n_blocks=11):
    tbl = {"Y": Y, "target_names": names, "ym": "t", "anchors_per_day": 1}
    r = vm.RidgeFullView("r", "R", "k", buf_rows=buf_rows, tail=TAIL)
    r._reset()
    r.begin_fit(tbl)
    off = 0
    for b in np.array_split(X.astype(np.float32), n_blocks):
        r.accumulate_fit(off, b)
        off += len(b)
    r.end_fit(tbl)
    return r


def _data(vm, seed=0, n=900, t=3):
    rng = np.random.default_rng(seed)
    p = vm.view_features(TAIL)
    # Wildly different per-column scales, because the centring/scaling is the
    # part being checked and equal scales would hide a bug in it.
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 50, p)
    Y = X @ rng.normal(size=(p, t)) + rng.normal(size=(n, t)) * 3
    return X, Y


def test_width_is_the_tail_the_learner_was_built_with(vm):
    """P comes from the tail, so the fixture is not lying about the width."""
    assert vm.view_features(TAIL) == TAIL * vm.N_FEATURES + 11
    assert vm.RidgeFullView("r", "R", "k", tail=TAIL).P == vm.view_features(TAIL)


def test_matches_dense_ridge_with_nested_censoring(vm):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, Y = _data(vm)
    # Nested shells, exactly the structure a horizon ladder produces: a longer
    # forward window fits on strictly fewer rows.
    Y[600:, 1] = np.nan
    Y[800:, 2] = np.nan
    names = ["a", "b", "c"]
    r = _fit_streamed(vm, X, Y, names)

    assert set(r.coefs) == set(names)
    for ti, name in enumerate(names):
        ok = np.isfinite(Y[:, ti])
        sc = StandardScaler().fit(X[ok])
        ref = Ridge(alpha=vm.RIDGE_ALPHA).fit(sc.transform(X[ok]), Y[ok, ti])
        want = ref.predict(sc.transform(X))
        v, mu_v, ybar = r.coefs[name]
        got = X @ v - mu_v + ybar
        # float32 is the stream dtype, so agreement is bounded by its epsilon
        # relative to the prediction scale, not by float64's.
        assert np.abs(got - want).max() < 1e-4 * max(1.0, np.abs(want).max())


def test_buffer_size_does_not_change_the_answer(vm):
    """A flush boundary must not be observable in the result."""
    X, Y = _data(vm, seed=1)
    Y[500:, 2] = np.nan
    names = ["a", "b", "c"]
    a = _fit_streamed(vm, X, Y, names, buf_rows=64, n_blocks=23)
    b = _fit_streamed(vm, X, Y, names, buf_rows=4096, n_blocks=2)
    for name in names:
        assert np.allclose(a.coefs[name][0], b.coefs[name][0], atol=1e-6)


def _ladder(n=2000, t=6, seed=0, rag_frac=0.015):
    """A horizon ladder plus the ragged edges a no-trade VWAP window leaves."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, t))
    for k in range(1, t):
        Y[n - k * 150:, k] = np.nan
    for r in rng.choice(n, int(n * rag_frac), replace=False):
        Y[r, rng.integers(0, t)] = np.nan
    return Y


def test_ragged_patterns_project_onto_the_common_shells(vm):
    """Over budget is projected, not refused -- and never fits a censored row.

    The ladder is six nested shells; the ragged rows push the pattern count
    past the budget. Those rows must join a family pattern CONTAINED IN their
    own finite mask, so nothing a row does not have enters that shell's Gram.
    """
    Y = _ladder()
    ok = np.isfinite(Y)
    assert len(np.unique(ok, axis=0)) > 8, "fixture no longer exceeds the budget"

    shell_of, fam = vm.RidgeFullView._patterns(Y, max_shells=8, verbose=False)

    assert len(fam) == 8
    kept = fam[np.maximum(shell_of, 0)] & (shell_of[:, None] >= 0)
    # THE INVARIANT: a row is only ever fitted on targets it really has.
    assert not (kept & ~ok).any()
    # And the projection is cheap, which is what makes it allowed at all.
    assert kept.sum() / ok.sum() > vm.MIN_PAIR_COVERAGE


def test_refuses_censoring_the_shells_cannot_describe(vm):
    """Below the coverage floor the pass stops rather than fits on a subset."""
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(2000, 6))
    # Random per-row censoring is not a handful of shells plus ragged edges:
    # most rows contain no family pattern at all and drop out entirely.
    Y[rng.random(Y.shape) < 0.3] = np.nan
    with pytest.raises(SystemExit, match=r"finite \(row, target\) pairs"):
        vm.RidgeFullView._patterns(Y, max_shells=8, verbose=False)


def test_registry_is_the_tails_and_the_full_view_is_kept_beside_it(vm):
    """The scored registry is tails; the full-view pair stays available."""
    assert [m.name for m in vm.default_view_learners()] == [
        "ridge_tail8", "ridge_tail24", "ridge_tail64", "hgb_tail24"]
    assert [m.name for m in vm.full_view_learners()] == ["ridge_full", "hgb_full"]

    # Every tail learner must say "tail", never "view" -- a tail is a lower
    # bound on the full view and the label is what stops it being quoted as one.
    for m in vm.default_view_learners():
        assert "tail" in m.label.lower() and m.tail is not None
    for m in vm.full_view_learners():
        assert m.tail is None


def test_one_pass_serves_every_tail(vm):
    """The tails share a stream at the widest of them, and slice their own."""
    learners = vm.default_view_learners()
    assert vm._stream_tail(learners) == 64

    wide = max(m.tail for m in learners)
    rng = np.random.default_rng(0)
    block = rng.normal(size=(5, vm.view_features(wide))).astype(np.float32)
    for m in learners:
        take = m._take(block, wide)
        assert take.shape[1] == m.P == vm.view_features(m.tail)
        # The slice is the last ``tail`` steps plus the eleven info columns.
        lo = (wide - m.tail) * vm.N_FEATURES
        assert np.array_equal(
            take, np.concatenate([block[:, lo:wide * vm.N_FEATURES],
                                  block[:, wide * vm.N_FEATURES:]], axis=1))

    # A full-view learner in a tail-streamed pass is a wiring bug, not a crop.
    with pytest.raises(SystemExit, match="streamed a"):
        vm.full_view_learners()[0]._take(block, wide)
