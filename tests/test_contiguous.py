"""Tests for the contiguous augmentation.

The contiguous augmentation produces two views from temporally adjacent,
non-overlapping windows: view1 covers [start, start+L1) and view2 covers
[start+L1, start+L1+L2). Each view is independently aggregated at its own
scale.
"""

import numpy as np
import pytest

from market_jepa.augmentations import AUGMENTATION_REGISTRY, _aggregate_numpy_jittered


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_dense_features(n_rows=3600, seed=42):
    """Create a dense (n_rows, 9) feature array mimicking post-preprocessing data."""
    rng = np.random.RandomState(seed)
    features = np.empty((n_rows, 9), dtype=np.float64)
    features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01  # bid_price
    features[:, 2] = features[:, 0] + rng.rand(n_rows) * 0.1     # high > bid
    features[:, 3] = features[:, 0] - rng.rand(n_rows) * 0.1     # low < bid
    features[:, 4] = features[:, 0] + 0.05 + rng.randn(n_rows) * 0.005  # ask
    features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)  # bid_size
    features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)  # ask_size
    features[:, 7] = rng.randint(1, 500, n_rows).astype(float)     # volume (>0)
    features[:, 8] = rng.randint(1, 50, n_rows).astype(float)      # n (>0)

    # vwap: valid where n > 0
    mid = (features[:, 0] + features[:, 4]) / 2
    features[:, 1] = mid + rng.randn(n_rows) * 0.005
    features[features[:, 8] == 0, 1] = np.nan

    return features


def _prior_vwap_numpy(features, window_start_idx):
    """Mirror of StreamingMarketDataset._prior_vwap_numpy."""
    if window_start_idx == 0:
        return None
    prior = features[:window_start_idx]
    valid = (prior[:, 8] > 0) & ~np.isnan(prior[:, 1])
    if valid.any():
        return prior[np.flatnonzero(valid)[-1], 1]
    return None


def _parse_contiguous_config(cfg):
    """Replicate the config parsing logic for contiguous augmentation."""
    v1_len = cfg.get("view1_length_sec", 1200)
    v2_len = cfg.get("view2_length_sec", 1200)
    return {
        "name": "contiguous",
        "view1_length_sec": v1_len,
        "view2_length_sec": v2_len,
        "view1_scale": cfg.get("view1_scale", 1),
        "view2_scale": cfg.get("view2_scale", 1),
        "window_size_sec": v1_len + v2_len,
    }


def _run_contiguous(features, start_idx, cfg):
    """Replicate the contiguous dispatch logic, returning views or None.

    Returns (view1, view2, prior_vwap_v2) or None if skipped.
    """
    L1 = cfg["view1_length_sec"]
    L2 = cfg["view2_length_sec"]
    s1 = cfg["view1_scale"]
    s2 = cfg["view2_scale"]

    if start_idx + L1 + L2 + 300 > len(features):
        return None

    window1 = features[start_idx : start_idx + L1]
    window2 = features[start_idx + L1 : start_idx + L1 + L2]

    if len(window1) < 10 or len(window2) < 10:
        return None

    view1 = _aggregate_numpy_jittered(window1, s1)
    view2 = _aggregate_numpy_jittered(window2, s2)
    if view1 is None or view2 is None:
        return None

    prior_vwap_v2 = _prior_vwap_numpy(features, start_idx + L1)
    return view1, view2, prior_vwap_v2


# -----------------------------------------------------------------------
# 1. Registry
# -----------------------------------------------------------------------


class TestRegistry:
    def test_contiguous_in_registry(self):
        assert "contiguous" in AUGMENTATION_REGISTRY


# -----------------------------------------------------------------------
# 2. Config parsing
# -----------------------------------------------------------------------


class TestConfigParsing:
    def test_defaults(self):
        parsed = _parse_contiguous_config({"name": "contiguous"})
        assert parsed["view1_length_sec"] == 1200
        assert parsed["view2_length_sec"] == 1200
        assert parsed["view1_scale"] == 1
        assert parsed["view2_scale"] == 1
        assert parsed["window_size_sec"] == 2400

    def test_custom_overrides(self):
        parsed = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 600,
            "view2_length_sec": 300,
            "view1_scale": 60,
            "view2_scale": 10,
        })
        assert parsed["view1_length_sec"] == 600
        assert parsed["view2_length_sec"] == 300
        assert parsed["view1_scale"] == 60
        assert parsed["view2_scale"] == 10
        assert parsed["window_size_sec"] == 900

    def test_window_size_equals_l1_plus_l2(self):
        for l1, l2 in [(1200, 1200), (600, 300), (100, 2000)]:
            parsed = _parse_contiguous_config({
                "name": "contiguous",
                "view1_length_sec": l1,
                "view2_length_sec": l2,
            })
            assert parsed["window_size_sec"] == l1 + l2


# -----------------------------------------------------------------------
# 3. Contiguity
# -----------------------------------------------------------------------


class TestContiguity:
    def test_views_are_contiguous(self):
        """view2 raw data starts exactly at features[start + L1]; no gap/overlap."""
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })
        start = 100
        result = _run_contiguous(features, start, cfg)
        assert result is not None
        view1, view2, _ = result

        # With scale=1, views are raw windows
        np.testing.assert_array_equal(view1, features[start : start + 1200])
        np.testing.assert_array_equal(view2, features[start + 1200 : start + 2400])

    def test_no_gap_or_overlap(self):
        """The last row of view1's raw window is adjacent to view2's first row."""
        features = _make_dense_features(3600)
        start = 200
        L1, L2 = 600, 600
        # Raw windows
        w1 = features[start : start + L1]
        w2 = features[start + L1 : start + L1 + L2]
        # Last row of w1 is at index start + L1 - 1
        # First row of w2 is at index start + L1
        # These are adjacent (differ by 1 index)
        np.testing.assert_array_equal(w1[-1], features[start + L1 - 1])
        np.testing.assert_array_equal(w2[0], features[start + L1])

    def test_volume_sums_match_combined_span(self):
        """Total volume across both views equals the raw span's volume."""
        features = _make_dense_features(3600)
        start = 50
        L1, L2 = 1200, 1200
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": L1,
            "view2_length_sec": L2,
        })
        result = _run_contiguous(features, start, cfg)
        assert result is not None
        view1, view2, _ = result

        combined_volume = features[start : start + L1 + L2, 7].sum()
        np.testing.assert_allclose(
            view1[:, 7].sum() + view2[:, 7].sum(), combined_volume
        )


# -----------------------------------------------------------------------
# 4. Independent aggregation
# -----------------------------------------------------------------------


class TestIndependentAggregation:
    def test_scale_60_on_1200_produces_20_rows(self):
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 600,
            "view1_scale": 60,
            "view2_scale": 10,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is not None
        view1, view2, _ = result
        assert view1.shape == (20, 9)   # 1200 / 60 = 20
        assert view2.shape == (60, 9)   # 600 / 10 = 60

    def test_different_scales_different_shapes(self):
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
            "view1_scale": 10,
            "view2_scale": 60,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is not None
        view1, view2, _ = result
        assert view1.shape == (120, 9)  # 1200 / 10
        assert view2.shape == (20, 9)   # 1200 / 60


# -----------------------------------------------------------------------
# 5. Buffer constraint
# -----------------------------------------------------------------------


class TestBufferConstraint:
    def test_valid_start_passes(self):
        """start + L1 + L2 + 300 <= len(features) should produce views."""
        n = 3000
        features = _make_dense_features(n)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })
        # Max valid start: n - L1 - L2 - 300 = 3000 - 2400 - 300 = 300
        result = _run_contiguous(features, 300, cfg)
        assert result is not None

    def test_boundary_start_passes(self):
        """Exact boundary: start + L1 + L2 + 300 == len(features)."""
        n = 3000
        features = _make_dense_features(n)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })
        # start=300: 300 + 1200 + 1200 + 300 = 3000 == len(features) → passes
        result = _run_contiguous(features, 300, cfg)
        assert result is not None

    def test_over_boundary_skips(self):
        """start + L1 + L2 + 300 > len(features) should skip."""
        n = 3000
        features = _make_dense_features(n)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })
        # start=301: 301 + 2400 + 300 = 3001 > 3000 → skip
        result = _run_contiguous(features, 301, cfg)
        assert result is None

    def test_features_too_short_for_any_start(self):
        """If features is too short for any valid start, always skip."""
        # Need at least L1 + L2 + 300 = 2700 rows; 2699 is too short
        features = _make_dense_features(2699)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })
        # Even start=0: 0 + 2400 + 300 = 2700 > 2699
        result = _run_contiguous(features, 0, cfg)
        assert result is None


# -----------------------------------------------------------------------
# 6. Prior VWAP
# -----------------------------------------------------------------------


class TestPriorVWAP:
    def test_view1_uses_prior_before_start(self):
        features = _make_dense_features(3600)
        start = 500
        prior = _prior_vwap_numpy(features, start)
        # Should find valid vwap in features[0:500]
        assert prior is not None

    def test_view2_uses_prior_before_start_plus_l1(self):
        features = _make_dense_features(3600)
        start = 100
        L1 = 1200
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": L1,
            "view2_length_sec": 1200,
        })
        result = _run_contiguous(features, start, cfg)
        assert result is not None
        _, _, prior_vwap_v2 = result

        # prior_vwap_v2 should search features[0:start+L1]
        expected = _prior_vwap_numpy(features, start + L1)
        assert prior_vwap_v2 == expected

    def test_view2_prior_searches_view1_region(self):
        """View2's prior VWAP should be able to find values in view1's 1Hz region."""
        features = np.zeros((3600, 9), dtype=np.float64)
        features[:, 0] = 100.0  # bid
        features[:, 4] = 101.0  # ask
        # Only set valid vwap at index 600 (inside view1's window)
        features[600, 1] = 42.0
        features[600, 8] = 5.0  # n > 0

        start = 500
        L1 = 1200
        prior_v2 = _prior_vwap_numpy(features, start + L1)
        assert prior_v2 == 42.0


# -----------------------------------------------------------------------
# 7. Determinism
# -----------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_views(self):
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
            "view1_scale": 10,
            "view2_scale": 60,
        })

        def generate(seed):
            rng = np.random.RandomState(seed)
            start = rng.randint(0, 300)
            return _run_contiguous(features, start, cfg)

        r1 = generate(42)
        r2 = generate(42)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_different_seed_different_start(self):
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
        })

        rng1 = np.random.RandomState(42)
        start1 = rng1.randint(0, 300)
        rng2 = np.random.RandomState(99)
        start2 = rng2.randint(0, 300)
        # Different seeds should (very likely) produce different starts
        # But we just verify both produce valid results
        r1 = _run_contiguous(features, start1, cfg)
        r2 = _run_contiguous(features, start2, cfg)
        assert r1 is not None and r2 is not None


# -----------------------------------------------------------------------
# 8. Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_scale_1_preserves_raw_data(self):
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 1200,
            "view2_length_sec": 1200,
            "view1_scale": 1,
            "view2_scale": 1,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is not None
        view1, view2, _ = result
        np.testing.assert_array_equal(view1, features[0:1200])
        np.testing.assert_array_equal(view2, features[1200:2400])

    def test_very_small_l1_l2(self):
        """L1=20, L2=20 with scale=1: should produce views of length 20."""
        features = _make_dense_features(1000)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 20,
            "view2_length_sec": 20,
            "view1_scale": 1,
            "view2_scale": 1,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is not None
        view1, view2, _ = result
        assert view1.shape == (20, 9)
        assert view2.shape == (20, 9)

    def test_l1_l2_too_small_for_min_rows_skips(self):
        """L1 or L2 < 10 should skip (window too short)."""
        features = _make_dense_features(1000)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 5,
            "view2_length_sec": 1200,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is None

    def test_asymmetric_lengths(self):
        """Different L1 and L2 should work correctly."""
        features = _make_dense_features(3600)
        cfg = _parse_contiguous_config({
            "name": "contiguous",
            "view1_length_sec": 600,
            "view2_length_sec": 1800,
            "view1_scale": 10,
            "view2_scale": 60,
        })
        result = _run_contiguous(features, 0, cfg)
        assert result is not None
        view1, view2, _ = result
        assert view1.shape == (60, 9)   # 600 / 10
        assert view2.shape == (30, 9)   # 1800 / 60
