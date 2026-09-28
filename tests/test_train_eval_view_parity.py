"""The eval panel must build the SAME view the training path would.

The reported metric's entire premise is that only the PREDICTOR differs between
arms. That silently stopped being true: `_panel_for_ticker_day` omitted the vwap
forward-fill training applies before `normalize_numpy`, so a single NaN bucket
poisoned the shared price group's mu/sigma and every price channel — bid, ask,
vwap, high, low — was normalized to NaN and then zeroed. It hit 100% of eval
views, and every IC measured before 2026-08-15 came from a model reading only
book-size, volume and trade count.

Shape and row-count checks cannot see this, because the panel was the right
shape and the right size the whole time. These assert the CONTENT.
"""

from __future__ import annotations

import numpy as np
import pytest

from market_jepa.training.utils import (
    build_norm_groups, ffill_vwap, normalize_numpy, prior_vwap,
)
from market_jepa.training.streaming_dataset import FEATURE_COLUMNS

PRICE_COLS = ["bid_price", "ask_price", "vwap_all", "high", "low"]


def _view_with_nan_vwap():
    rng = np.random.RandomState(0)
    v = rng.rand(64, len(FEATURE_COLUMNS)) + 1.0
    v[7, FEATURE_COLUMNS.index("vwap_all")] = np.nan   # one zero-volume bucket
    return v


def test_one_nan_vwap_would_zero_every_price_channel():
    """Without the ffill, normalize_numpy destroys the whole price group."""
    v = _view_with_nan_vwap()
    normalize_numpy(v, build_norm_groups(FEATURE_COLUMNS))
    np.nan_to_num(v, copy=False, nan=0.0)
    for c in PRICE_COLS:
        col = v[:, FEATURE_COLUMNS.index(c)]
        assert np.abs(col).max() == 0.0, f"{c} should be zeroed without the ffill"


def test_ffill_vwap_preserves_every_price_channel():
    """With the ffill — what both paths now do — the prices survive."""
    v = _view_with_nan_vwap()
    ffill_vwap(v, prior=None)
    assert np.isfinite(v).all(), "ffill must leave no NaN behind"
    normalize_numpy(v, build_norm_groups(FEATURE_COLUMNS))
    np.nan_to_num(v, copy=False, nan=0.0)
    for c in PRICE_COLS:
        col = v[:, FEATURE_COLUMNS.index(c)]
        assert np.abs(col).max() > 0.0, f"{c} must carry signal after the ffill"


def test_eval_panel_applies_the_ffill_before_normalizing():
    """Stable panel construction must ffill VWAP before normalization."""
    import inspect

    from stable_finance.dataset.panels import build_session_panel

    body = inspect.getsource(build_session_panel)
    assert "ffill_vwap(" in body, "stable panel dropped the vwap ffill"
    assert body.index("ffill_vwap(") < body.index("prepare_view("), (
        "ffill_vwap must run BEFORE normalization or the prices are zeroed"
    )


def test_prior_vwap_picks_the_last_valid_row():
    f = np.zeros((10, len(FEATURE_COLUMNS)))
    vw, nn = FEATURE_COLUMNS.index("vwap_all"), FEATURE_COLUMNS.index("n")
    f[:, vw] = np.nan
    f[2, vw], f[2, nn] = 5.0, 3
    f[6, vw], f[6, nn] = 9.0, 1
    assert prior_vwap(f, 8) == 9.0
    assert prior_vwap(f, 4) == 5.0
    assert prior_vwap(f, 0) is None
