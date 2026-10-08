"""The AUC's labels live on the RAW target, and the head is scored on the
reported partition rather than its own.

Both failures are silent. A head whose softmax is scored against z-score bins
still produces an AUC in [0, 1] that looks like a result; a head left at its
native k still fills every cell of a figure whose columns vary k. These tests
pin the two invariants that make those numbers mean what the figure says.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from market_jepa.eval.tasks import HORIZONS, TARGET_TYPES  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    _project_proba, macro_ovr_auc, raw_targets, score,
)

T, H = len(TARGET_TYPES), len(HORIZONS)
DATES = ["2020-05-04", "2020-05-05", "2020-05-06"]
ANCHORS = np.array([12000, 13200, 14400], dtype=np.int64)


def _table(tmp_path: Path, seed: int = 0) -> tuple[AnchorStats, np.ndarray, np.ndarray]:
    """A small but structurally real anchor table, with dispersed sigma."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1e-4, (len(DATES), len(ANCHORS), T, H)).astype(np.float32)
    # 3x spread in sigma across cells — the dispersion that pulls the raw and
    # z partitions apart in the first place.
    sigma = rng.uniform(1e-3, 3e-3, (len(DATES), len(ANCHORS), T, H)).astype(np.float32)
    p = tmp_path / "2020-05.npz"
    np.savez(p, dates=np.array(DATES), anchors=ANCHORS,
             types=np.array(TARGET_TYPES), horizons=np.array(HORIZONS, dtype=np.int32),
             mu=mu, sigma=sigma,
             count=np.full(mu.shape, 200, dtype=np.int32))
    return AnchorStats(p), mu, sigma


def _panel(stats: AnchorStats, n: int, seed: int) -> dict:
    """A cache-shaped panel whose z-scores came through stats.zscore()."""
    rng = np.random.default_rng(seed)
    dates = rng.choice(DATES, n)
    ancs = rng.choice(ANCHORS, n)
    z = rng.standard_normal((n, T * H))
    return {"z": z.astype(np.float32), "date": dates, "anchor": ancs,
            "ticker": np.array([f"T{i % 40}" for i in range(n)]),
            "target_names": np.array([f"{t}_{h}" for t in TARGET_TYPES
                                      for h in HORIZONS])}


# ── the inversion ────────────────────────────────────────────────────────────

def test_unstandardize_inverts_zscore_row_by_row(tmp_path):
    """raw_targets must agree with the per-sample zscore() it undoes."""
    stats, _, _ = _table(tmp_path)
    panel = _panel(stats, 200, seed=1)
    raw = raw_targets(panel, stats)

    for i in range(0, 200, 17):
        back = stats.zscore(raw[i], str(panel["date"][i]), int(panel["anchor"][i]),
                            TARGET_TYPES, HORIZONS).ravel()
        np.testing.assert_allclose(back, panel["z"][i], rtol=1e-5, atol=1e-6)


def test_raw_targets_prefers_values_stored_with_uniform_panels(tmp_path):
    stats, _, _ = _table(tmp_path)
    panel = _panel(stats, 20, seed=9)
    stored = np.random.default_rng(10).normal(size=panel["z"].shape)
    panel["raw"] = stored
    panel["target_transform"] = "uniform"
    np.testing.assert_array_equal(raw_targets(panel, stats), stored)


def test_unstandardize_is_nan_where_the_cell_is_missing(tmp_path):
    """An absent date or anchor NaNs the row rather than inventing a scale."""
    stats, _, _ = _table(tmp_path)
    panel = _panel(stats, 50, seed=2)
    panel["date"] = np.array(["1999-01-04"] * 25 + list(panel["date"][25:]))
    panel["anchor"] = np.concatenate([panel["anchor"][:25], np.full(25, 99999)])
    raw = raw_targets(panel, stats)
    assert np.isnan(raw).all(), "no cell in this panel exists in the table"


def test_raw_and_z_partitions_actually_disagree(tmp_path):
    """The premise: sigma dispersion moves rows between pooled bins.

    If this ever stopped being true the raw/z distinction would be cosmetic —
    so it is asserted, not assumed. Chance agreement at k=5 is 20%.
    """
    stats, _, _ = _table(tmp_path, seed=3)
    panel = _panel(stats, 4000, seed=4)
    raw = raw_targets(panel, stats)[:, 2]
    z = panel["z"][:, 2].astype(float)

    def q(v, k=5):
        return np.searchsorted(np.quantile(v, np.arange(1, k) / k), v)

    agree = float((q(raw) == q(z)).mean())
    assert 0.2 < agree < 0.9, f"same-bin rate {agree:.2f} — partitions collapsed"


# ── the projection ───────────────────────────────────────────────────────────

def test_projection_is_identity_at_the_reported_k():
    p = np.random.default_rng(0).dirichlet(np.ones(5), size=30)
    out = _project_proba(p, np.random.default_rng(1).standard_normal(2000), 5)
    assert out is p


def test_projection_nests_exactly_when_k_to_divides_k_from():
    """k=10 -> k=5 is a pure pairwise sum: bins 2j and 2j+1 land in bin j."""
    y = np.random.default_rng(2).standard_normal(20000)
    p = np.eye(10)                      # one row per source bin
    out = _project_proba(p, y, 5)
    expect = np.repeat(np.eye(5), 2, axis=0)
    np.testing.assert_allclose(out, expect, atol=1e-9)


def test_projection_preserves_mass_and_order_for_a_straddling_k():
    """k=21 -> k=5 has no clean nesting; mass must still be conserved and the
    map must stay monotone (source bin j cannot skew to a lower target bin
    than source bin j-1 does)."""
    y = np.random.default_rng(3).standard_normal(40000)
    out = _project_proba(np.eye(21), y, 5)
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-9)
    centre = out @ np.arange(5)
    assert np.all(np.diff(centre) >= -1e-9), "projection reversed the order"


def test_projection_of_a_confident_head_beats_a_flat_one():
    """A head that is right about the raw bin must score above chance after
    projection — the projection cannot be destroying the signal it carries."""
    rng = np.random.default_rng(4)
    y_fit = rng.standard_normal(20000)
    y_ev = rng.standard_normal(4000)
    edges = np.quantile(y_fit, np.arange(1, 21) / 21)
    true21 = np.searchsorted(edges, y_ev)
    confident = np.full((len(y_ev), 21), 0.002)
    confident[np.arange(len(y_ev)), true21] = 1.0 - 0.002 * 20
    lev = np.searchsorted(np.quantile(y_fit, np.arange(1, 5) / 5), y_ev)

    good = macro_ovr_auc(_project_proba(confident, y_fit, 5), lev)
    flat = macro_ovr_auc(np.full((len(y_ev), 5), 0.2), lev)
    assert good > 0.9 and 0.45 < flat < 0.55


# ── the AUC itself ───────────────────────────────────────────────────────────

def test_macro_ovr_auc_endpoints():
    lab = np.repeat(np.arange(4), 50)
    perfect = np.eye(4)[lab] * 0.9 + 0.025
    assert macro_ovr_auc(perfect, lab) == pytest.approx(1.0)
    assert macro_ovr_auc(np.full((200, 4), 0.25), lab) == pytest.approx(0.5)


def test_macro_ovr_auc_skips_absent_classes():
    """A class with no eval rows is skipped, not counted as a coin flip."""
    lab = np.repeat(np.arange(3), 40)          # class 3 never occurs
    proba = np.eye(4)[lab] * 0.9 + 0.025
    assert macro_ovr_auc(proba, lab) == pytest.approx(1.0)


# ── end to end through score() ───────────────────────────────────────────────

def _caches(stats, n_tr=1500, n_ev=1200, d=8, seed=5):
    """Panels whose embeddings genuinely predict the raw return, so the AUC
    has something to find and a broken labelling shows up as a collapse."""
    rng = np.random.default_rng(seed)
    out = []
    for n, s in ((n_tr, seed), (n_ev, seed + 1)):
        c = _panel(stats, n, s)
        raw = raw_targets(c, stats)
        X = rng.standard_normal((n, d))
        # First feature carries the raw return; z is then whatever the cell's
        # sigma makes of it, which is the point.
        X[:, 0] = raw[:, 2] / np.nanstd(raw[:, 2]) + 0.5 * rng.standard_normal(n)
        c["X"] = X.astype(np.float32)
        out.append(c)
    return out


def test_score_emits_auc_only_when_the_tables_are_given(tmp_path):
    stats, _, _ = _table(tmp_path, seed=6)
    tr, ev = _caches(stats)

    bare = score(tr, ev)
    assert not any("auc" in v for v in bare.values()), (
        "without the anchor tables the raw target is unrecoverable, so no AUC "
        "may be emitted — a z-binned one would silently mispair the head"
    )

    full = score(tr, ev, train_stats=stats, eval_stats=stats)
    assert full["return_900"]["auc"] > 0.6, "planted signal not recovered"
    assert full["return_900"]["auc_bins"] == 5
    # The IC path must be untouched by any of this.
    for k in bare:
        assert full[k]["ic"] == pytest.approx(bare[k]["ic"])


def test_head_is_scored_on_the_reported_partition_not_its_own(tmp_path):
    stats, _, _ = _table(tmp_path, seed=7)
    tr, ev = _caches(stats)
    raw_ev = raw_targets(ev, stats)[:, 2]

    rng = np.random.default_rng(8)
    edges = np.quantile(raw_targets(tr, stats)[:, 2], np.arange(1, 21) / 21)
    true21 = np.searchsorted(edges, raw_ev)
    proba = np.full((len(raw_ev), 21), 0.01)
    proba[np.arange(len(raw_ev)), true21] = 0.8

    res = score(tr, ev, head_scores=true21.astype(float), head_task="return_900",
                head_proba=proba, train_stats=stats, eval_stats=stats)
    head = res["head:return_900"]
    assert head["auc_bins"] == 5, "head must report on the figure's partition"
    assert head["head_bins"] == 21, "its native k must survive as provenance"
    assert head["auc"] > 0.9


def test_native_k_is_reported_next_to_the_projection(tmp_path):
    """A k=21 head must report BOTH numbers, each with a probe at its own k.

    The projected one is the only one that may be read across a figure whose
    axis is k; the native one is what the head was trained to separate. Losing
    either would be losing an argument, not a column.
    """
    stats, _, _ = _table(tmp_path, seed=9)
    tr, ev = _caches(stats)
    raw_ev = raw_targets(ev, stats)[:, 2]
    edges = np.quantile(raw_targets(tr, stats)[:, 2], np.arange(1, 21) / 21)
    true21 = np.searchsorted(edges, raw_ev)
    proba = np.full((len(raw_ev), 21), 0.01)
    proba[np.arange(len(raw_ev)), true21] = 0.8

    res = score(tr, ev, head_scores=true21.astype(float), head_task="return_900",
                head_proba=proba, train_stats=stats, eval_stats=stats)
    head, probe = res["head:return_900"], res["return_900"]

    assert head["auc_native_bins"] == 21 and head["auc_bins"] == 5
    assert probe["auc_native_bins"] == 21 and probe["auc_bins"] == 5
    # The two head numbers are on different partitions and must not be equal.
    assert abs(head["auc"] - head["auc_native"]) > 1e-6
    # A target the figure never reads gets neither, so the expensive logistic
    # is not paid 18 times over.
    assert "auc" not in res["return_300"] and "auc_native" not in res["return_300"]


def test_a_k5_head_reports_one_number_twice(tmp_path):
    """At k = auc_bins the projection is the identity, so native must agree
    exactly rather than being recomputed down a slightly different path."""
    stats, _, _ = _table(tmp_path, seed=10)
    tr, ev = _caches(stats)
    rng = np.random.default_rng(11)
    proba = rng.dirichlet(np.ones(5), size=len(ev["X"]))
    res = score(tr, ev, head_scores=proba @ np.arange(5), head_task="return_900",
                head_proba=proba, train_stats=stats, eval_stats=stats)
    head = res["head:return_900"]
    assert head["auc_native"] == head["auc"]
    assert res["return_900"]["auc_native"] == res["return_900"]["auc"]
