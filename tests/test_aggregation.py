"""Tests for aggregation logic — _aggregate_numpy and _aggregate_numpy_subset consistency.

These test the core assumption that _aggregate_numpy (hardcoded indices, used for main obs)
and _aggregate_numpy_subset (rule-based, used for risk factors) produce identical results
when given the same input and full column set.
"""

import numpy as np
import pytest

from market_jepa.augmentations import _aggregate_numpy_jittered
from market_jepa.training.streaming_dataset import FEATURE_COLUMNS, _AGG_RULES


def _make_features(n_rows, n_feat=9, seed=42):
    """Create realistic (n_rows, n_feat) feature array."""
    rng = np.random.RandomState(seed)
    features = np.empty((n_rows, n_feat), dtype=np.float64)
    features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01  # bid_price
    features[:, 1] = 100.025 + rng.randn(n_rows) * 0.005        # vwap_all
    features[:, 2] = 100.0 + rng.rand(n_rows) * 0.1             # high
    features[:, 3] = 99.9 + rng.rand(n_rows) * 0.1              # low
    features[:, 4] = 100.05 + rng.randn(n_rows).cumsum() * 0.01 # ask_price
    features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)  # bid_size
    features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)  # ask_size
    features[:, 7] = rng.randint(0, 500, n_rows).astype(float)     # volume
    features[:, 8] = rng.randint(0, 50, n_rows).astype(float)      # n
    return features


class _MockDataset:
    """Minimal mock to call _aggregate_numpy and _aggregate_numpy_subset."""

    def __init__(self, col_names=None):
        col_names = col_names or FEATURE_COLUMNS
        self._rf_agg_rules = [_AGG_RULES[c] for c in col_names]
        self._rf_volume_local_idx = (
            col_names.index("volume") if "volume" in col_names else None
        )

    def aggregate_hardcoded(self, features, scale_factor):
        """Use _aggregate_numpy_jittered(offset=0) — the single aggregation path."""
        return _aggregate_numpy_jittered(features, scale_factor, offset=0)

    def aggregate_subset(self, features, scale_factor):
        """Replicate _aggregate_numpy_subset logic (rule-based)."""
        from market_jepa.training.streaming_dataset import StreamingMarketDataset
        return StreamingMarketDataset._aggregate_numpy_subset(self, features, scale_factor)


# -----------------------------------------------------------------------
# _aggregate_numpy — basic correctness
# -----------------------------------------------------------------------


class TestAggregateNumpy:
    """Tests for the hardcoded _aggregate_numpy used for main observations."""

    def setup_method(self):
        self.ds = _MockDataset()

    def test_scale_1_is_identity_copy(self):
        features = _make_features(50)
        result = self.ds.aggregate_hardcoded(features, 1)
        np.testing.assert_array_equal(result, features)
        assert not np.shares_memory(result, features)

    def test_returns_none_for_single_bucket(self):
        features = _make_features(5)
        assert self.ds.aggregate_hardcoded(features, 10) is None

    def test_shape_exact_division(self):
        features = _make_features(60)
        result = self.ds.aggregate_hardcoded(features, 10)
        assert result.shape == (6, 9)

    def test_shape_with_remainder(self):
        features = _make_features(65)
        result = self.ds.aggregate_hardcoded(features, 10)
        # 6 full + 1 partial = 7
        assert result.shape == (7, 9)

    def test_bid_price_is_last(self):
        features = np.arange(30, dtype=np.float64).reshape(10, 3)
        # Need 9 columns
        features_9 = np.zeros((10, 9), dtype=np.float64)
        features_9[:, 0] = np.arange(10)  # bid_price
        features_9[:, 7] = 1.0  # volume (for vwap)
        features_9[:, 1] = 50.0  # vwap
        result = self.ds.aggregate_hardcoded(features_9, 5)
        # Bucket 0: rows 0-4, last = 4
        assert result[0, 0] == 4.0
        # Bucket 1: rows 5-9, last = 9
        assert result[1, 0] == 9.0

    def test_high_is_max(self):
        features = np.zeros((10, 9), dtype=np.float64)
        features[:, 2] = [1, 5, 3, 2, 4, 6, 2, 8, 1, 3]  # high
        features[:, 7] = 1.0  # volume for vwap
        features[:, 1] = 50.0  # vwap
        result = self.ds.aggregate_hardcoded(features, 5)
        assert result[0, 2] == 5.0  # max of [1,5,3,2,4]
        assert result[1, 2] == 8.0  # max of [6,2,8,1,3]

    def test_low_is_min(self):
        features = np.zeros((10, 9), dtype=np.float64)
        features[:, 3] = [5, 3, 7, 1, 4, 2, 8, 6, 9, 0]  # low
        features[:, 7] = 1.0
        features[:, 1] = 50.0
        result = self.ds.aggregate_hardcoded(features, 5)
        assert result[0, 3] == 1.0  # min of [5,3,7,1,4]
        assert result[1, 3] == 0.0  # min of [2,8,6,9,0]

    def test_volume_is_sum(self):
        features = np.zeros((10, 9), dtype=np.float64)
        features[:, 7] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        features[:, 1] = 50.0  # vwap
        result = self.ds.aggregate_hardcoded(features, 5)
        assert result[0, 7] == 150.0
        assert result[1, 7] == 400.0

    def test_vwap_is_volume_weighted(self):
        features = np.zeros((6, 9), dtype=np.float64)
        features[:, 1] = [10, 20, 30, 40, 50, 60]  # vwap
        features[:, 7] = [1, 2, 3, 4, 5, 6]         # volume
        result = self.ds.aggregate_hardcoded(features, 3)
        # Bucket 0: vwap * vol = 10*1 + 20*2 + 30*3 = 140, vol_sum = 6
        np.testing.assert_allclose(result[0, 1], 140.0 / 6.0)

    def test_vwap_nan_when_zero_volume(self):
        features = np.zeros((10, 9), dtype=np.float64)
        features[:, 1] = 50.0  # vwap
        features[:, 7] = 0.0   # volume = 0
        result = self.ds.aggregate_hardcoded(features, 5)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])


# -----------------------------------------------------------------------
# _aggregate_numpy vs _aggregate_numpy_subset — consistency
# -----------------------------------------------------------------------


class TestAggregateConsistency:
    """Verify _aggregate_numpy and _aggregate_numpy_subset produce identical results
    when given the same 9-column input (the full FEATURE_COLUMNS set)."""

    def setup_method(self):
        self.ds = _MockDataset(FEATURE_COLUMNS)

    @pytest.mark.parametrize("scale_factor", [1, 2, 3, 5, 10, 60, 300])
    def test_full_columns_match(self, scale_factor):
        features = _make_features(600)
        r_hard = self.ds.aggregate_hardcoded(features, scale_factor)
        r_rule = self.ds.aggregate_subset(features, scale_factor)

        if r_hard is None:
            assert r_rule is None
            return

        assert r_hard.shape == r_rule.shape
        np.testing.assert_allclose(r_hard, r_rule, atol=1e-10, equal_nan=True)

    @pytest.mark.parametrize("scale_factor", [5, 60])
    def test_with_remainder(self, scale_factor):
        """Test that partial trailing bucket matches between implementations."""
        n = 600 + scale_factor // 2  # Ensure remainder
        features = _make_features(n)
        r_hard = self.ds.aggregate_hardcoded(features, scale_factor)
        r_rule = self.ds.aggregate_subset(features, scale_factor)
        assert r_hard.shape == r_rule.shape
        np.testing.assert_allclose(r_hard, r_rule, atol=1e-10, equal_nan=True)

    def test_with_zero_volume_rows(self):
        """Both should produce NaN vwap when volume is zero."""
        features = _make_features(60)
        features[:, 7] = 0.0  # zero all volume
        r_hard = self.ds.aggregate_hardcoded(features, 10)
        r_rule = self.ds.aggregate_subset(features, 10)
        np.testing.assert_allclose(r_hard, r_rule, atol=1e-10, equal_nan=True)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_data(self, seed):
        """Random data fuzz test for consistency."""
        features = _make_features(np.random.RandomState(seed).randint(100, 1000), seed=seed)
        scale = np.random.RandomState(seed + 100).choice([2, 5, 10, 30, 60])
        r_hard = self.ds.aggregate_hardcoded(features, scale)
        r_rule = self.ds.aggregate_subset(features, scale)
        if r_hard is None:
            assert r_rule is None
            return
        np.testing.assert_allclose(r_hard, r_rule, atol=1e-10, equal_nan=True)


# -----------------------------------------------------------------------
# _aggregate_numpy_subset with RF column subsets
# -----------------------------------------------------------------------


class TestAggregateSubsetPresets:
    """Test _aggregate_numpy_subset with various RF column presets."""

    @pytest.mark.parametrize("preset_cols", [
        ["vwap_all"],
        ["vwap_all", "volume"],
        ["vwap_all", "volume", "bid_price", "ask_price", "bid_size", "ask_size"],
    ])
    def test_subset_shape(self, preset_cols):
        ds = _MockDataset(preset_cols)
        n_cols = len(preset_cols)
        features = _make_features(60)
        # Select the right columns
        col_indices = [FEATURE_COLUMNS.index(c) for c in preset_cols]
        subset = features[:, col_indices]
        result = ds.aggregate_subset(subset, 10)
        assert result.shape == (6, n_cols)

    def test_vwap_only_no_volume_uses_mean(self):
        """When only vwap is present (no volume column), vwap should fall back to mean."""
        ds = _MockDataset(["vwap_all"])
        features = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]], dtype=np.float64)
        result = ds.aggregate_subset(features, 3)
        # Should be mean: (10+20+30)/3 = 20, (40+50+60)/3 = 50
        np.testing.assert_allclose(result[0, 0], 20.0)
        np.testing.assert_allclose(result[1, 0], 50.0)

    def test_vwap_volume_uses_weighted_avg(self):
        """When volume is present, vwap should be volume-weighted."""
        ds = _MockDataset(["vwap_all", "volume"])
        features = np.array([
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [40.0, 4.0],
            [50.0, 5.0],
            [60.0, 6.0],
        ], dtype=np.float64)
        result = ds.aggregate_subset(features, 3)
        # Bucket 0: (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6
        np.testing.assert_allclose(result[0, 0], 140.0 / 6.0)
        # Volume: sum = 6
        assert result[0, 1] == 6.0
