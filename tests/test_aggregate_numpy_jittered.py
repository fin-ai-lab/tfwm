"""Tests for _aggregate_numpy_jittered, the shared uniform-bucket aggregator.

Every augmentation that emits uniform buckets routes through this kernel:
random_resized_crop, cross_stock, fast_timestamp_jittering, volume_noise,
price_jitter, and the Kronos bar coarsening.
"""

import numpy as np

from market_jepa.augmentations import _aggregate_numpy_jittered


# -----------------------------------------------------------------------
# _aggregate_numpy_jittered — unit tests
# -----------------------------------------------------------------------


class TestAggregateNumpyJittered:
    """Tests for the standalone _aggregate_numpy_jittered function."""

    def _make_features(self, n_rows, seed=42):
        """Helper: (n_rows, 9) random feature array with realistic values."""
        rng = np.random.RandomState(seed)
        features = np.empty((n_rows, 9), dtype=np.float64)
        features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01  # bid_price
        features[:, 1] = 100.025 + rng.randn(n_rows) * 0.005  # vwap_all
        features[:, 2] = 100.0 + rng.rand(n_rows) * 0.1  # high
        features[:, 3] = 99.9 + rng.rand(n_rows) * 0.1  # low
        features[:, 4] = 100.05 + rng.randn(n_rows).cumsum() * 0.01  # ask_price
        features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)  # bid_size
        features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)  # ask_size
        features[:, 7] = rng.randint(0, 500, n_rows).astype(float)  # volume
        features[:, 8] = rng.randint(0, 50, n_rows).astype(float)  # n
        return features

    def test_returns_none_when_too_few_rows(self):
        features = self._make_features(3)
        # scale_factor=5 => only 0 full buckets + 1 partial = 1 bucket < 2
        assert _aggregate_numpy_jittered(features, 5) is None

    def test_returns_none_when_offset_leaves_too_few(self):
        features = self._make_features(12)
        # 12 rows, offset=10 => 2 rows left, scale_factor=5 => 0 full + 1 partial = 1 bucket
        assert _aggregate_numpy_jittered(features, 5, offset=10) is None

    def test_scale_factor_1_is_copy(self):
        features = self._make_features(20)
        result = _aggregate_numpy_jittered(features, 1)
        np.testing.assert_array_equal(result, features)
        # Must be a copy, not a view
        assert not np.shares_memory(result, features)

    def test_scale_factor_1_with_offset(self):
        features = self._make_features(20)
        result = _aggregate_numpy_jittered(features, 1, offset=5)
        np.testing.assert_array_equal(result, features[5:])

    def test_output_shape_exact_division(self):
        features = self._make_features(20)
        result = _aggregate_numpy_jittered(features, 5)
        assert result.shape == (4, 9)

    def test_output_shape_with_remainder(self):
        features = self._make_features(23)
        result = _aggregate_numpy_jittered(features, 5)
        # 23 // 5 = 4 full + 1 partial = 5 buckets
        assert result.shape == (5, 9)

    def test_output_shape_with_offset(self):
        features = self._make_features(25)
        result = _aggregate_numpy_jittered(features, 5, offset=3)
        # 22 rows after offset, 22 // 5 = 4 full + 1 partial(2) = 5 buckets
        assert result.shape == (5, 9)

    def test_aggregation_rules_on_known_data(self):
        """Verify each column's aggregation rule on hand-crafted data."""
        # 2 buckets of scale_factor=3
        features = np.array(
            [
                # bid  vwap  high  low   ask   bsz   asz   vol   n
                [10.0, 50.0, 12.0, 8.0, 11.0, 100., 200., 10.,  3.],
                [10.5, 51.0, 13.0, 7.0, 11.5, 110., 210., 20.,  5.],
                [11.0, 52.0, 11.0, 9.0, 12.0, 120., 220., 30.,  2.],
                [11.5, 53.0, 14.0, 6.0, 12.5, 130., 230., 40.,  4.],
                [12.0, 54.0, 15.0, 5.0, 13.0, 140., 240., 50.,  6.],
                [12.5, 55.0, 10.0, 7.5, 13.5, 150., 250., 60.,  1.],
            ],
            dtype=np.float64,
        )
        result = _aggregate_numpy_jittered(features, 3)
        assert result.shape == (2, 9)

        # Bucket 0: rows 0-2
        assert result[0, 0] == 11.0  # bid_price: last
        assert result[0, 2] == 13.0  # high: max
        assert result[0, 3] == 7.0   # low: min
        assert result[0, 4] == 12.0  # ask_price: last
        assert result[0, 5] == 120.  # bid_size: last
        assert result[0, 6] == 220.  # ask_size: last
        assert result[0, 7] == 60.   # volume: sum (10+20+30)
        assert result[0, 8] == 10.   # n: sum (3+5+2)
        # vwap_all: volume-weighted = (50*10 + 51*20 + 52*30) / 60
        expected_vwap_0 = (50 * 10 + 51 * 20 + 52 * 30) / 60
        np.testing.assert_allclose(result[0, 1], expected_vwap_0)

        # Bucket 1: rows 3-5
        assert result[1, 0] == 12.5  # bid_price: last
        assert result[1, 2] == 15.0  # high: max
        assert result[1, 3] == 5.0   # low: min
        assert result[1, 4] == 13.5  # ask_price: last
        assert result[1, 5] == 150.  # bid_size: last
        assert result[1, 6] == 250.  # ask_size: last
        assert result[1, 7] == 150.  # volume: sum (40+50+60)
        assert result[1, 8] == 11.   # n: sum (4+6+1)
        expected_vwap_1 = (53 * 40 + 54 * 50 + 55 * 60) / 150
        np.testing.assert_allclose(result[1, 1], expected_vwap_1)

    def test_partial_trailing_bucket(self):
        """Partial bucket at the end aggregates correctly."""
        features = np.array(
            [
                [1., 10., 5., 1., 2., 100., 200., 10., 1.],
                [2., 20., 6., 2., 3., 110., 210., 20., 2.],
                [3., 30., 7., 3., 4., 120., 220., 30., 3.],
                [4., 40., 8., 0., 5., 130., 230., 40., 4.],
                [5., 50., 9., 1., 6., 140., 240., 50., 5.],
                # partial bucket: only 2 rows
                [6., 60., 4., 2., 7., 150., 250., 60., 6.],
                [7., 70., 3., 3., 8., 160., 260., 70., 7.],
            ],
            dtype=np.float64,
        )
        result = _aggregate_numpy_jittered(features, 5)
        assert result.shape == (2, 9)

        # Partial bucket (rows 5-6)
        assert result[1, 0] == 7.   # bid_price: last
        assert result[1, 2] == 4.   # high: max(4, 3)
        assert result[1, 3] == 2.   # low: min(2, 3)
        assert result[1, 7] == 130. # volume: sum(60, 70)
        assert result[1, 8] == 13.  # n: sum(6, 7)
        expected_vwap = (60 * 60 + 70 * 70) / 130
        np.testing.assert_allclose(result[1, 1], expected_vwap)

    def test_vwap_nan_when_zero_volume(self):
        """VWAP should be NaN when total volume in bucket is 0."""
        features = np.zeros((10, 9), dtype=np.float64)
        # All volume = 0
        result = _aggregate_numpy_jittered(features, 5)
        assert result.shape == (2, 9)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])

    def test_offset_shifts_buckets(self):
        """Offset should skip rows before bucketing starts."""
        features = self._make_features(30)
        r0 = _aggregate_numpy_jittered(features, 5, offset=0)
        r3 = _aggregate_numpy_jittered(features, 5, offset=3)
        # Different offsets should generally produce different results
        assert r0.shape[0] != r3.shape[0] or not np.allclose(r0, r3, equal_nan=True)

    def test_dtype_is_float64(self):
        features = self._make_features(20)
        result = _aggregate_numpy_jittered(features, 5)
        assert result.dtype == np.float64
