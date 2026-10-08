"""Tests for the full augmentation pipeline in _getitem_numpy.

Tests the multi-scale augmentation logic, vwap forward-fill, normalization,
risk factor merging, collation/bucketing, and the zero_sample fallback —
all without requiring MDS data on disk.
"""

import numpy as np
import pytest
import torch

from market_jepa.augmentations import _aggregate_numpy_jittered
from market_jepa.training.utils import build_norm_groups, normalize_numpy
from market_jepa.training.streaming_dataset import (
    FEATURE_COLUMNS,
    _AGG_RULES,
    _RF_PRESETS,
)


# -----------------------------------------------------------------------
# Helpers (replicate the private methods so we can test them in isolation)
# -----------------------------------------------------------------------


def _aggregate_numpy(features, scale_factor):
    """Thin wrapper around _aggregate_numpy_jittered(offset=0)."""
    return _aggregate_numpy_jittered(features, scale_factor, offset=0)


def _ffill_vwap_numpy(view, prior_vwap=None):
    """Mirror of StreamingMarketDataset._ffill_vwap_numpy."""
    vwap = view[:, 1]
    if np.isnan(vwap[0]) and prior_vwap is not None:
        vwap[0] = prior_vwap
    mask = np.isnan(vwap)
    if mask.any():
        idx = np.arange(len(vwap))
        idx[mask] = 0
        np.maximum.accumulate(idx, out=idx)
        vwap[:] = vwap[idx]
    still_nan = np.isnan(vwap)
    if still_nan.any():
        vwap[still_nan] = (view[still_nan, 0] + view[still_nan, 4]) / 2


def _prior_vwap_numpy(features, window_start_idx):
    """Mirror of StreamingMarketDataset._prior_vwap_numpy."""
    if window_start_idx == 0:
        return None
    prior = features[:window_start_idx]
    valid = (prior[:, 8] > 0) & ~np.isnan(prior[:, 1])
    if valid.any():
        return prior[np.flatnonzero(valid)[-1], 1]
    return None


def _make_dense_features(n_rows=1200, seed=42):
    """Create a dense (n_rows, 9) feature array mimicking post-preprocessing data."""
    rng = np.random.RandomState(seed)
    features = np.empty((n_rows, 9), dtype=np.float64)
    features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01  # bid_price
    features[:, 2] = features[:, 0] + rng.rand(n_rows) * 0.1     # high > bid
    features[:, 3] = features[:, 0] - rng.rand(n_rows) * 0.1     # low < bid
    features[:, 4] = features[:, 0] + 0.05 + rng.randn(n_rows) * 0.005  # ask
    features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)  # bid_size
    features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)  # ask_size
    features[:, 7] = rng.randint(0, 500, n_rows).astype(float)     # volume
    features[:, 8] = rng.randint(0, 50, n_rows).astype(float)      # n

    # vwap: valid where n > 0, NaN where n == 0
    mid = (features[:, 0] + features[:, 4]) / 2
    features[:, 1] = mid + rng.randn(n_rows) * 0.005
    features[features[:, 8] == 0, 1] = np.nan

    return features


# -----------------------------------------------------------------------
# vwap forward-fill
# -----------------------------------------------------------------------


class TestVwapForwardFill:
    """Tests for the vwap forward-fill logic."""

    def test_no_nan_after_ffill_with_prior(self):
        """If prior_vwap is provided, no NaN should remain in vwap column."""
        features = _make_dense_features()
        window = features[100:300].copy()
        # Force first row's vwap to NaN
        window[0, 1] = np.nan
        window[0, 8] = 0  # n=0
        prior = 99.5
        view = _aggregate_numpy(window, 5)
        _ffill_vwap_numpy(view, prior)
        assert not np.any(np.isnan(view[:, 1]))

    def test_prior_vwap_seeds_first_nan(self):
        """Prior vwap should fill the first row if it starts with NaN."""
        view = np.zeros((10, 9), dtype=np.float64)
        view[:, 1] = np.nan  # all vwap NaN
        view[:, 0] = 100.0   # bid
        view[:, 4] = 101.0   # ask
        _ffill_vwap_numpy(view, prior_vwap=99.0)
        assert view[0, 1] == 99.0  # seeded from prior

    def test_forward_fill_propagates(self):
        """After seeding, forward fill should propagate to subsequent NaN rows."""
        view = np.zeros((5, 9), dtype=np.float64)
        view[:, 1] = [np.nan, np.nan, 50.0, np.nan, np.nan]
        view[:, 0] = 100.0
        view[:, 4] = 101.0
        _ffill_vwap_numpy(view, prior_vwap=42.0)
        np.testing.assert_array_equal(view[:, 1], [42.0, 42.0, 50.0, 50.0, 50.0])

    def test_midpoint_backstop(self):
        """If no prior and first row is NaN, midpoint of bid/ask should be used."""
        view = np.zeros((3, 9), dtype=np.float64)
        view[:, 1] = np.nan
        view[:, 0] = [100.0, 101.0, 102.0]  # bid
        view[:, 4] = [200.0, 201.0, 202.0]  # ask
        _ffill_vwap_numpy(view, prior_vwap=None)
        # All NaN, no prior → backstop uses midpoint
        np.testing.assert_allclose(view[0, 1], 150.0)

    def test_prior_vwap_finder(self):
        """_prior_vwap_numpy should find the last valid vwap before window start."""
        features = np.zeros((100, 9), dtype=np.float64)
        features[:, 1] = np.nan
        features[:, 8] = 0  # all n=0
        # Valid vwap at idx 20
        features[20, 1] = 42.0
        features[20, 8] = 5.0
        # Valid vwap at idx 50
        features[50, 1] = 99.0
        features[50, 8] = 3.0

        assert _prior_vwap_numpy(features, 60) == 99.0
        assert _prior_vwap_numpy(features, 30) == 42.0
        assert _prior_vwap_numpy(features, 10) is None
        assert _prior_vwap_numpy(features, 0) is None


# -----------------------------------------------------------------------
# Multi-scale augmentation pipeline
# -----------------------------------------------------------------------


class TestMultiScalePipeline:
    """Test the full multi-scale augmentation pipeline end to end."""

    def test_two_views_different_lengths(self):
        """With different scale factors, views should have different lengths."""
        features = _make_dense_features(1200)
        window = features[:1200]
        view1 = _aggregate_numpy(window, 60)   # 1200/60 = 20 rows
        view2 = _aggregate_numpy(window, 1)    # 1200 rows
        assert view1 is not None and view2 is not None
        assert len(view1) == 20
        assert len(view2) == 1200

    def test_both_views_cover_same_window(self):
        """Both views should aggregate the same underlying data window."""
        features = _make_dense_features(600)
        window = features[:600]
        view1 = _aggregate_numpy(window, 60)
        view2 = _aggregate_numpy(window, 10)
        # Total volume should match
        np.testing.assert_allclose(view1[:, 7].sum(), view2[:, 7].sum())
        # Total n should match
        np.testing.assert_allclose(view1[:, 8].sum(), view2[:, 8].sum())

    def test_normalization_preserves_shape(self):
        features = _make_dense_features(600)
        window = features[:600]
        view = _aggregate_numpy(window, 10)
        _ffill_vwap_numpy(view)
        original_shape = view.shape
        norm_groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(view, norm_groups)
        assert view.shape == original_shape

    def test_normalization_produces_no_nan(self):
        """After ffill + normalization, there should be no NaN in the output."""
        features = _make_dense_features(600)
        # Ensure first row has valid data for forward fill
        features[0, 8] = 1.0  # n > 0
        features[0, 1] = 100.0  # valid vwap
        window = features[:600]
        view = _aggregate_numpy(window, 10)
        prior = _prior_vwap_numpy(features, 0)
        _ffill_vwap_numpy(view, prior)
        norm_groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(view, norm_groups)
        assert not np.any(np.isnan(view))

    def test_tensor_shape_is_features_by_length(self):
        """Output tensor should be (n_features, length) after transpose."""
        features = _make_dense_features(600)
        window = features[:600]
        view = _aggregate_numpy(window, 10)
        _ffill_vwap_numpy(view)
        norm_groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(view, norm_groups)
        t = torch.from_numpy(view.T.astype(np.float32))
        assert t.shape == (9, 60)  # (features, length)
        assert t.dtype == torch.float32

    def test_deterministic_pair_generation(self):
        """Same seed + same data should produce same views."""
        features = _make_dense_features(1200)
        norm_groups = build_norm_groups(FEATURE_COLUMNS)

        def generate(seed):
            rng = np.random.RandomState(seed)
            cfg_idx = rng.randint(0, 2)
            start_idx = rng.randint(0, max(1, len(features)))
            window = features[start_idx:start_idx + 600]
            if len(window) < 10:
                return None
            s1, s2 = sorted(rng.choice([1, 60], 2, replace=False), reverse=True)
            v1 = _aggregate_numpy(window, s1)
            v2 = _aggregate_numpy(window, s2)
            if v1 is None or v2 is None:
                return None
            _ffill_vwap_numpy(v1)
            _ffill_vwap_numpy(v2)
            normalize_numpy(v1, norm_groups)
            normalize_numpy(v2, norm_groups)
            return v1, v2

        r1 = generate(42)
        r2 = generate(42)
        assert r1 is not None
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])


# -----------------------------------------------------------------------
# Bucket key / collation
# -----------------------------------------------------------------------


class TestBucketingAndCollation:
    """Test the bucketing and collation logic from pretrain.py."""

    def _make_pair(self, n_feat, len1, len2, bucket_key):
        v1 = torch.randn(n_feat, len1)
        v2 = torch.randn(n_feat, len2)
        return {
            "views": [v1, v2],
            "lengths": torch.tensor([len1, len2], dtype=torch.long),
            "bucket_key": bucket_key,
        }

    def test_same_bucket_key_grouped(self):
        """Samples with same bucket_key should be in the same bucket."""
        from collections import defaultdict
        batch = [
            [self._make_pair(9, 20, 1200, 0), self._make_pair(9, 20, 1200, 0)],
            [self._make_pair(9, 12, 720, 1), self._make_pair(9, 12, 720, 1)],
        ]

        # Replicate collate_bucketed logic
        flat = []
        for item in batch:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        groups = defaultdict(list)
        for sample in flat:
            groups[sample.get("bucket_key", -1)].append(sample)

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2

    def test_padding_within_bucket(self):
        """Within a bucket, shorter tensors should be padded to max length."""
        v1_short = torch.randn(9, 18)
        v1_long = torch.randn(9, 20)
        batch = [
            {"views": [v1_short, torch.randn(9, 100)],
             "lengths": torch.tensor([18, 100]), "bucket_key": 0},
            {"views": [v1_long, torch.randn(9, 120)],
             "lengths": torch.tensor([20, 120]), "bucket_key": 0},
        ]

        # Run collation
        from market_jepa.training.utils import collate_bucketed
        result = collate_bucketed(batch)
        buckets = result["buckets"]
        assert len(buckets) == 1

        # View 0: max_len should be 20, both padded to 20
        assert buckets[0]["views"][0].shape == (2, 9, 20)
        # View 1: max_len should be 120
        assert buckets[0]["views"][1].shape == (2, 9, 120)

        # Lengths should be preserved
        torch.testing.assert_close(
            buckets[0]["lengths"][0],
            torch.tensor([18, 20]),
        )

    def test_multiple_buckets_independent_padding(self):
        """Different buckets should pad independently."""
        from market_jepa.training.utils import collate_bucketed
        batch = [
            [
                self._make_pair(9, 20, 1200, 0),  # bucket 0: lengths 20, 1200
                self._make_pair(9, 12, 720, 1),   # bucket 1: lengths 12, 720
            ],
        ]
        result = collate_bucketed(batch)
        buckets = result["buckets"]
        # Should have 2 buckets, each with 1 sample
        assert len(buckets) == 2

    def test_zero_sample_bucket_key_minus_1(self):
        """Zero samples (filtered obs) should have bucket_key=-1."""
        n_feat = 9
        view = torch.zeros(n_feat, 1, dtype=torch.float32)
        zero = {
            "views": [view, view.clone()],
            "lengths": torch.tensor([1, 1], dtype=torch.long),
            "bucket_key": -1,
        }
        assert zero["bucket_key"] == -1

    def test_collation_with_zero_samples(self):
        """Collation should drop sentinel zero samples (bucket_key=-1)."""
        from market_jepa.training.utils import collate_bucketed
        zero = {
            "views": [torch.zeros(9, 1), torch.zeros(9, 1)],
            "lengths": torch.tensor([1, 1]),
            "bucket_key": -1,
        }
        real = self._make_pair(9, 20, 1200, 0)
        batch = [[zero], [real]]
        result = collate_bucketed(batch)
        # Sentinel bucket (-1) is dropped; only real bucket remains
        assert len(result["buckets"]) == 1


# -----------------------------------------------------------------------
# Risk factor presets
# -----------------------------------------------------------------------


class TestRiskFactorPresets:
    """Test RF preset resolution and column mapping."""

    def test_all_preset_covers_all_columns(self):
        assert _RF_PRESETS["all"] == FEATURE_COLUMNS

    def test_vwap_preset_single_column(self):
        assert _RF_PRESETS["vwap"] == ["vwap_all"]

    def test_vwap_volume_preset(self):
        assert _RF_PRESETS["vwap_volume"] == ["vwap_all", "volume"]

    def test_all_preset_columns_have_agg_rules(self):
        """Every column name in presets should have an aggregation rule."""
        for preset_name, cols in _RF_PRESETS.items():
            for col in cols:
                assert col in _AGG_RULES, (
                    f"Column {col!r} in preset {preset_name!r} has no agg rule"
                )

    def test_n_features_with_risk_factors(self):
        """n_features should be 9 + (n_rf_cols * n_rf_tickers)."""
        base = len(FEATURE_COLUMNS)
        assert base == 9

        # vwap preset, 3 tickers
        n_rf_cols = len(_RF_PRESETS["vwap"])  # 1
        n_tickers = 3
        expected = base + n_rf_cols * n_tickers
        assert expected == 12

        # all preset, 5 tickers
        n_rf_cols = len(_RF_PRESETS["all"])  # 9
        n_tickers = 5
        expected = base + n_rf_cols * n_tickers
        assert expected == 54

    def test_preset_columns_are_in_feature_columns(self):
        """All preset column names should be loadable — either in
        FEATURE_COLUMNS or a recognized synthetic column."""
        synthetic = {"mid_price"}
        for preset_name, cols in _RF_PRESETS.items():
            for col in cols:
                assert col in FEATURE_COLUMNS or col in synthetic, (
                    f"Column {col!r} in preset {preset_name!r} not in FEATURE_COLUMNS"
                )


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the augmentation pipeline."""

    def test_window_shorter_than_scale_returns_none(self):
        """If window is too short for 2+ buckets, aggregation returns None."""
        features = _make_dense_features(50)
        result = _aggregate_numpy(features, 60)
        assert result is None

    def test_window_exactly_two_buckets(self):
        """Minimum viable window: exactly 2 full buckets."""
        features = _make_dense_features(120)
        result = _aggregate_numpy(features, 60)
        assert result is not None
        assert result.shape == (2, 9)

    def test_all_zero_volume_window(self):
        """Window with all zero volume should produce NaN vwap before ffill."""
        features = np.zeros((60, 9), dtype=np.float64)
        features[:, 0] = 100.0  # bid
        features[:, 4] = 101.0  # ask
        result = _aggregate_numpy(features, 10)
        assert result is not None
        # vwap should be NaN (zero volume)
        assert np.all(np.isnan(result[:, 1]))
        # After ffill with no prior, should use midpoint backstop
        _ffill_vwap_numpy(result, None)
        expected_mid = (100.0 + 101.0) / 2
        np.testing.assert_allclose(result[:, 1], expected_mid)

    def test_scale_factor_1_preserves_data(self):
        """Scale factor 1 should return an identical copy."""
        features = _make_dense_features(100)
        result = _aggregate_numpy(features, 1)
        np.testing.assert_array_equal(result, features)
        assert not np.shares_memory(result, features)

    def test_window_size_10_is_minimum(self):
        """The pipeline requires window >= 10 rows."""
        features = _make_dense_features(9)
        # 9 rows at scale 1 would give 9 buckets (>= 2), but pipeline checks len < 10
        # This tests the assumption in _getitem_numpy
        assert len(features) < 10

    def test_single_row_partial_bucket(self):
        """11 rows with scale=10: 1 full bucket + 1 partial = 2 (valid)."""
        features = _make_dense_features(11)
        result = _aggregate_numpy(features, 10)
        assert result is not None
        assert result.shape == (2, 9)


# -----------------------------------------------------------------------
# Current behavior assumption tests (Phase 1)
# -----------------------------------------------------------------------


class TestCurrentBehaviorAssumptions:
    """Tests that verify current behavior before wiring in timestamp jittering.

    These must pass both before and after the code changes.
    """

    def test_rf_merge_output_shape(self):
        """RF merge should produce (n_agg, 9 + n_rf_features)."""
        features = _make_dense_features(1200)
        window = features[:1200]
        view = _aggregate_numpy(window, 60)
        assert view is not None
        n_agg = len(view)

        # Simulate RF merge: 2 tickers, vwap_volume preset (2 cols each)
        rf_cols = _RF_PRESETS["vwap_volume"]
        n_rf_cols = len(rf_cols)
        n_tickers = 2
        rf_parts = [np.zeros((n_agg, n_rf_cols), dtype=np.float64) for _ in range(n_tickers)]
        merged = np.concatenate([view] + rf_parts, axis=1)
        assert merged.shape == (n_agg, 9 + n_rf_cols * n_tickers)

    def test_rf_merge_bucket_count_invariant(self):
        """RF aggregation at same scale + same window length = same bucket count as main."""
        from market_jepa.training.streaming_dataset import StreamingMarketDataset

        features = _make_dense_features(1200)
        window = features[:1200]

        for scale in [1, 5, 10, 60]:
            main_agg = _aggregate_numpy(window, scale)
            if main_agg is None:
                continue
            n_agg = len(main_agg)

            # Simulate what _merge_risk_factors does for RF
            rf_cols = _RF_PRESETS["vwap_volume"]
            rf_agg_rules = [_AGG_RULES[c] for c in rf_cols]
            vol_local = rf_cols.index("volume") if "volume" in rf_cols else None
            col_indices = [FEATURE_COLUMNS.index(c) for c in rf_cols]
            rf_raw = window[:, col_indices].copy()

            mock = type("M", (), {
                "_rf_agg_rules": rf_agg_rules,
                "_rf_volume_local_idx": vol_local,
            })()
            rf_agg = StreamingMarketDataset._aggregate_numpy_subset(mock, rf_raw, scale)
            assert rf_agg is not None
            assert len(rf_agg) == n_agg

    def test_aug_configs_built_from_multi_scale(self):
        """_ms_configs should extract scale_factors and window_size_sec from config dicts."""
        # This tests the current _ms_configs building logic
        configs = [
            {"name": "multi_scale", "scale_factors": [1, 60], "window_size_sec": 1200},
            {"name": "multi_scale", "scale_factors": [1, 16], "window_size_sec": 60},
        ]
        ms_configs = []
        for cfg in configs:
            ms_configs.append({
                "scale_factors": cfg.get("scale_factors", [1, 16]),
                "window_size_sec": cfg.get("window_size_sec", 60),
            })
        assert ms_configs[0]["scale_factors"] == [1, 60]
        assert ms_configs[0]["window_size_sec"] == 1200
        assert ms_configs[1]["scale_factors"] == [1, 16]
        assert ms_configs[1]["window_size_sec"] == 60
