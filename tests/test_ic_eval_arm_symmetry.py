"""The SSL and supervised arms must reach the SAME scorer.

The whole comparison rests on one claim: the ridge-vs-head substitution is the
only difference between the arms. That claim is a property of the CODE, not of
the numbers, and it broke once already when the eval panel skipped the vwap
ffill that the training path performed. These tests pin the parts that a
lookalike reimplementation would silently break.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from xs_ic_eval import (  # noqa: E402
    RIDGE_ALPHA, RIDGE_ALPHAS, ridge_alpha_for, score,
)

HOOK = ROOT / "scripts/generic/post_train_ic_eval.py"


def _fake_cache(n, n_targets=2, d=8, seed=0):
    rng = np.random.default_rng(seed)
    # Two cells per date so grouped_rank_ic has something to group over.
    dates = np.array([f"2020-01-{1 + i % 5:02d}" for i in range(n)])
    anchors = np.array([[34200, 37800][i % 2] for i in range(n)])
    return {
        "X": rng.normal(size=(n, d)).astype(np.float32),
        "z": rng.normal(size=(n, n_targets)).astype(np.float32),
        "date": dates,
        "anchor": anchors,
        "ticker": np.array([f"T{i % 40}" for i in range(n)]),
        "target_names": np.array(["volatility_change_900", "spread_change_900"]),
    }


def test_hook_does_not_reimplement_scoring():
    """The hook must CALL xs_ic_eval.score, not carry its own copy.

    Two implementations that look alike drift: one gains a guard, a
    standardizer, or an alpha lookup and the other does not, and the arms stop
    being comparable while both keep producing plausible numbers.
    """
    tree = ast.parse(HOOK.read_text())

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "xs_ic_eval"
        for alias in node.names
    }
    assert "score" in imported, "the hook must import xs_ic_eval.score"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "score" in called, "the hook imports score but never calls it"

    # The giveaway that a second implementation has grown back.
    src = HOOK.read_text()
    for banned in ("Ridge(", "StandardScaler(", "grouped_rank_ic("):
        assert banned not in src, (
            f"{banned} appears in the hook — scoring is being reimplemented "
            f"instead of delegated to xs_ic_eval.score"
        )


def test_head_scores_require_a_task():
    """A head emits ONE scalar; scoring it against every target is nonsense."""
    ev = _fake_cache(400)
    with pytest.raises(ValueError, match="head_task"):
        score(_fake_cache(400), ev, head_scores=np.zeros(400))


def test_head_readout_is_additive_not_a_substitution():
    """A supervised checkpoint is REPORTED on its probe, like the SSL arm.

    The head is scored too, under a namespaced key, so "would the head have
    beaten its own probe?" stays a measured question. If the head ever
    replaced the ridge again, the supervised arm would be handed a trained
    predictor while the SSL arm got a linear one, and the headline comparison
    would stop being about the training objective.
    """
    tr, ev = _fake_cache(400, seed=1), _fake_cache(400, seed=2)
    names = [str(n) for n in ev["target_names"]]
    out = score(tr, ev, head_scores=np.arange(400, dtype=float),
                head_task=names[0])

    # Every target still gets its ridge column...
    assert set(names) <= set(out)
    # ...and the head adds exactly one more, for its own target only.
    assert set(out) - set(names) == {f"head:{names[0]}"}


def test_head_path_still_requires_a_train_cache():
    """The probe is the reported number for every arm, so the fit month is
    embedded even for a head checkpoint."""
    with pytest.raises(ValueError, match="train_cache"):
        score(None, _fake_cache(400), head_scores=np.arange(400, dtype=float),
              head_task="volatility_change_900")


def test_ridge_path_requires_a_train_cache():
    with pytest.raises(ValueError, match="train_cache"):
        score(None, _fake_cache(400))


def test_probe_column_is_identical_with_and_without_a_head():
    """Passing head_scores must not perturb the reported probe number."""
    tr, ev = _fake_cache(400, seed=3), _fake_cache(400, seed=4)
    names = [str(n) for n in ev["target_names"]]
    bare = score(tr, ev)
    with_head = score(tr, ev, head_scores=np.arange(400, dtype=float),
                      head_task=names[0])
    for n in names:
        assert bare[n]["ic"] == with_head[n]["ic"]
        assert bare[n]["n_cells"] == with_head[n]["n_cells"]


def test_alpha_is_per_task_with_a_fallback():
    """Resolution is by exact column first, then target type, then default."""
    assert ridge_alpha_for("volatility_change_900") == RIDGE_ALPHAS[
        "volatility_change"]
    # Horizons the dict does not name still resolve through the type.
    assert ridge_alpha_for("volatility_change_7200") == RIDGE_ALPHAS[
        "volatility_change"]
    assert ridge_alpha_for("something_unlisted_900") == RIDGE_ALPHA


def test_both_arms_share_one_eval_panel_geometry():
    """Eval anchors define the reported cells and must not vary by arm."""
    import xs_ic_eval

    hook_src = HOOK.read_text()
    # Both parsers must default the EVAL month to the same anchor count; the
    # fit month is deliberately larger and is allowed to differ from it.
    assert 'default=8,' in hook_src
    assert xs_ic_eval.TRAIN_ANCHORS_PER_DAY > 8
