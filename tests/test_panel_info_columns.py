"""The eleven numbers the encoder gets in its information token reach the
classical baselines too.

Until 2026-08-25 ``panel_tables`` called ``iter_panel`` with neither half of
the information token, so every classical arm read 9 channels while the encoder
read 20. That is not a modelling choice a paper can defend -- it is the
baselines being handed strictly less information than the model they are meant
to bound. These tests pin the repair at all three places it has to hold:

  * the table CARRIES the eleven, at the end, untransformed;
  * the models CONSUME them (ARX / ARMAX / Ridge ARDL all widen their design
    matrices, and a planted signal in an info column is recovered);
  * the view stream appends them ONCE rather than 2048 times, which is the
    difference between an 18,443-wide block and a 40,960-wide one.
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

import panel_tables as pt  # noqa: E402
import models as fm  # noqa: E402


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_info_names_are_the_last_eleven_predictors():
    names = pt.predictor_names()
    assert names[-pt.N_INFO:] == pt.INFO_NAMES
    assert pt.N_INFO == 11
    # 8 norm stats + 3 window descriptors, in training.utils' order.
    assert pt.INFO_NAMES[:2] == ["norm_mu_g0", "norm_sigma_g0"]
    assert pt.INFO_NAMES[-3:] == [
        "view_start_frac", "view_end_frac", "log_agg"]


def test_view_features_counts_the_eleven_once():
    # 2048 x 20 would be 40,960 -- 22,517 of them exact duplicates.
    assert pt.VIEW_FEATURES == 2048 * pt.N_FEATURES + pt.N_INFO == 18_443


# ---------------------------------------------------------------------------
# _split_info
# ---------------------------------------------------------------------------


def _fake_views(B=4, T=64, info=None):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(B, T, pt.N_FEATURES)).astype(np.float32)
    if info is None:
        info = rng.normal(size=(B, pt.N_INFO))
    payload = np.zeros((B, T, pt.N_INFO), dtype=np.float32)
    payload[:, -1, :] = info
    return np.concatenate([v, payload], axis=2).astype(np.float32), v, info


def test_split_info_recovers_the_constants_and_leaves_the_series():
    views, v, info = _fake_views()
    got_v, got_info = pt._split_info(views)
    assert got_v.shape == v.shape
    np.testing.assert_allclose(got_v, v)
    np.testing.assert_allclose(got_info, info, rtol=1e-6)


def test_split_info_on_an_old_panel_reports_nan_not_zero():
    """A zero would read as a measured value and be fitted; NaN gets dropped."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 16, pt.N_FEATURES)).astype(np.float32)
    got_v, got_info = pt._split_info(v)
    assert got_v is v
    assert got_info.shape == (3, pt.N_INFO)
    assert np.isnan(got_info).all()


def test_view_predictors_appends_the_eleven_untransformed():
    views, _v, info = _fake_views(B=5, T=256)
    out = pt.view_predictors(views, agg=8)
    assert out.shape == (5, len(pt.PREDICTOR_NAMES))
    np.testing.assert_allclose(out[:, -pt.N_INFO:], info, rtol=1e-6)


# ---------------------------------------------------------------------------
# The models consume them
# ---------------------------------------------------------------------------


def _table(n=600, seed=0, with_info=True, planted=None):
    """A minimal panel table: real column names, synthetic numbers.

    ``planted`` names an info column the target is made a pure function of, so
    a model that ignores the block cannot score on it.
    """
    rng = np.random.default_rng(seed)
    names = list(pt.PREDICTOR_NAMES)
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    if not with_info:
        keep = [i for i, nm in enumerate(names) if nm not in pt.INFO_NAMES]
        names = [names[i] for i in keep]
        X = X[:, keep]
    targets = ["return_900"]
    if planted is not None:
        y = X[:, names.index(planted)] * 3.0 + rng.normal(scale=0.1, size=n)
    else:
        y = rng.normal(size=n)
    return {
        "X": X,
        "Y": y.astype(np.float32).reshape(n, 1),
        "predictor_names": names,
        "target_names": targets,
        "cell": np.repeat(np.arange(n // 20), 20).astype(str),
        "ym": "2009-06",
        "anchors_per_day": 8,
    }


def test_info_cols_are_found_and_absent_when_the_table_is_old():
    tbl = _table()
    cols = fm._info_cols(tbl)
    assert [tbl["predictor_names"][c] for c in cols] == pt.INFO_NAMES
    assert fm._info_cols(_table(with_info=False)) == []


def _design_width(model, target: str) -> int:
    """How many regressors the fitted model actually used.

    ARp/ARMApq keep a coefficient vector; RidgeARDL hands its columns to sklearn
    and
    keeps the surviving column list instead.
    """
    if hasattr(model, "cols") and target in getattr(model, "cols", {}):
        return len(model.cols[target])
    return len(model.coefs[target])


@pytest.mark.parametrize("model_factory", [fm.ARp, fm.ARMApq, fm.RidgeARDL])
def test_models_widen_their_design_matrix_by_eleven(model_factory):
    """Same fit twice, with and without the block: the design matrix grows."""
    wide = model_factory()
    wide.fit(_table(with_info=True))
    narrow = model_factory()
    narrow.fit(_table(with_info=False))
    t = "return_900"
    assert t in wide.coefs and t in narrow.coefs, "both fits must land"
    assert _design_width(wide, t) == _design_width(narrow, t) + pt.N_INFO


@pytest.mark.parametrize("model_factory", [fm.ARp, fm.ARMApq, fm.RidgeARDL])
def test_a_planted_info_signal_is_recovered(model_factory):
    """The block is not merely present in X -- the forecast moves with it."""
    tr = _table(seed=0, planted="norm_sigma_g2")
    ev = _table(seed=1, planted="norm_sigma_g2")
    m = model_factory()
    m.fit(tr)
    pred = m.predict(ev, "return_900")
    assert pred is not None
    y = ev["Y"][:, 0]
    ok = np.isfinite(pred) & np.isfinite(y)
    assert np.corrcoef(pred[ok], y[ok])[0, 1] > 0.5


def test_arma_keeps_its_lag_layout_when_the_block_is_appended():
    """The eleven go LAST, so ``_stage1_resid``/``predict`` still index lags."""
    tr, ev = _table(seed=2), _table(seed=3)
    m = fm.ARMApq()
    m.fit(tr)
    t = "return_900"
    p, q = m.order[t]
    # AR lags, then q reconstructed residuals, then the eleven.
    assert len(m.coefs[t]) == p + q + pt.N_INFO + 1  # +1 intercept
    assert m.predict(ev, t) is not None


# ---------------------------------------------------------------------------
# The cache cannot serve a stale table
# ---------------------------------------------------------------------------


def test_a_135_column_cache_is_refused(tmp_path, monkeypatch):
    stale = [n for n in pt.PREDICTOR_NAMES if n not in pt.INFO_NAMES]
    np.savez(
        pt.table_path("2009-06", 36, tmp_path),
        X=np.zeros((2, len(stale)), dtype=np.float32),
        Y=np.zeros((2, 1), dtype=np.float32),
        cell=np.asarray(["a", "a"]),
        predictor_names=np.asarray(stale),
        target_names=np.asarray(["return_900"]),
        # The CURRENT default, so the return-definition guard passes and the
        # predictor-width guard is the one under test.
        xs_anchor_stats_dir=np.asarray(
            "lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60"),
    )
    with pytest.raises(SystemExit, match="predictors"):
        pt.build_month_table("2009-06", anchors_per_day=36, cache_dir=tmp_path)
