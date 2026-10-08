from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

import panel_cache as pc  # noqa: E402
from stable_finance.dataset import PanelObservation, ViewMetadata


def test_model_neutral_cache_round_trips_targets_and_metadata(tmp_path):
    key = pc.panel_key(
        anchors_per_day=1,
        stats_tag="test_stats",
        has_rf=False,
        norm_groups=None,
        fixed_agg=None,
        seq_len=3,
    )
    # PINNED ON PURPOSE. The format is in the cache key, so a bump silently
    # retires every panel on disk -- hundreds of GB and hours of rebuild. That
    # should require editing this line, not just happen.
    # 2 (2026-09-10): panels carry the quote column.
    assert key["format"] == 2

    views = np.zeros((2, 3, 9), dtype=np.float32)
    uniform = np.array([[0.25, 0.75], [0.1, 0.9]], dtype=np.float32)
    raw = np.array([[0.001, -0.001], [0.02, -0.02]], dtype=np.float32)
    observations = []
    for index, ticker in enumerate(("A", "B")):
        observations.append(PanelObservation(
            date="2020-01-02", ticker=ticker, anchor=300,
            view=views[index], target=uniform[index], raw_target=raw[index],
            quote=np.array([10.0 + index, 10.5 + index]),
            metadata=ViewMetadata(
                start_seconds=100, end_seconds=103, aggregation_seconds=1,
                normalization_means=np.arange(4),
                normalization_scales=np.arange(4) + 1,
            ),
        ))

    pc.build(tmp_path, "2020-01", key, iter([(views, observations)]))
    loaded = pc.load(tmp_path, "2020-01", key)

    np.testing.assert_array_equal(loaded.targets, uniform)
    np.testing.assert_array_equal(loaded.raw_targets, raw)
    np.testing.assert_array_equal(loaded.normalization_means[0], np.arange(4))

    # THE QUOTE MUST SURVIVE. Without it the execution stage sees an unquoted
    # market: nothing fills, every spread charges zero, and a frictionless
    # Sharpe reports as a spread-charged one -- silently, because NaN is a
    # legitimate value for an absent quote.
    np.testing.assert_array_equal(
        loaded.quotes, np.array([[10.0, 10.5], [11.0, 11.5]]))
    for batch_views, batch_obs in pc.iter_cached(
            tmp_path, "2020-01", key, batch_size=2):
        assert batch_obs[0][-1] is not None, "iter_cached dropped the quote"
        np.testing.assert_array_equal(batch_obs[0][-1], np.array([10.0, 10.5]))
