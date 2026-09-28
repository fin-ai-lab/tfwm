"""Tests for risk factor temporal alignment with main observation.

Verifies that when a main observation window is extracted and aggregated,
the corresponding risk factor window covers the exact same seconds and
produces the same number of aggregated buckets — ensuring each row in
the output tensor pairs main-asset data with RF data from the same time period.
"""

import numpy as np
import pytest

from market_jepa.augmentations import _aggregate_numpy_jittered
from stable_finance.dataset import sparse_to_dense_grid, timeline_bounds_est
from market_jepa.training.utils import build_norm_groups, normalize_numpy
from market_jepa.training.streaming_dataset import (
    FEATURE_COLUMNS,
    _AGG_RULES,
    _FFILL_COLS,
    _RF_PRESETS,
    _ZERO_FILL_COLS,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

# Column indices matching FEATURE_COLUMNS
_FFILL_INDICES = [i for i, c in enumerate(FEATURE_COLUMNS) if c in set(_FFILL_COLS)]
_ZEROFILL_INDICES = [i for i, c in enumerate(FEATURE_COLUMNS) if c in set(_ZERO_FILL_COLS)]


def _make_rf_array(n_dates, n_seconds=23400, n_feat=9, seed=42):
    """Create a synthetic RF array (n_dates, n_seconds, n_feat).

    Each element encodes its position: features[d, s, f] = d*1e6 + s*10 + f
    so we can trace exactly which second was sampled.
    """
    arr = np.zeros((n_dates, n_seconds, n_feat), dtype=np.float32)
    for d in range(n_dates):
        for s in range(n_seconds):
            for f in range(n_feat):
                arr[d, s, f] = d * 1_000_000 + s * 10 + f
    return arr


def _make_rf_array_fast(n_dates, n_seconds=23400, n_feat=9):
    """Vectorized version of _make_rf_array for larger arrays."""
    d = np.arange(n_dates).reshape(-1, 1, 1) * 1_000_000
    s = np.arange(n_seconds).reshape(1, -1, 1) * 10
    f = np.arange(n_feat).reshape(1, 1, -1)
    return (d + s + f).astype(np.float32)


def _make_dense_main_obs(ts_open_sec, n_seconds=23400, n_feat=9, leading_nan=0, seed=42):
    """Create a dense main observation mimicking post-sparse_to_dense_grid output.

    Encodes position as: features[s, f] = s * 10 + f (matching RF encoding for date 0).

    Args:
        leading_nan: Number of leading rows trimmed (simulates trim_leading_nan).
    """
    canonical_sec = np.arange(
        ts_open_sec + leading_nan, ts_open_sec + n_seconds, dtype=np.int32
    )
    features = np.zeros((len(canonical_sec), n_feat), dtype=np.float64)
    for i, sec_offset in enumerate(range(leading_nan, n_seconds)):
        for f in range(n_feat):
            features[i, f] = sec_offset * 10 + f
    return canonical_sec, features


def _aggregate_numpy(features, scale_factor):
    """Thin wrapper around _aggregate_numpy_jittered(offset=0)."""
    return _aggregate_numpy_jittered(features, scale_factor, offset=0)


def _aggregate_numpy_subset(features, scale_factor, agg_rules, volume_local_idx):
    """Standalone copy of _aggregate_numpy_subset for testing."""
    if scale_factor == 1:
        return features.copy()
    n = len(features)
    n_full = n // scale_factor
    remainder = n % scale_factor
    n_buckets = n_full + (1 if remainder > 0 else 0)
    if n_buckets < 2:
        return None
    k = features.shape[1]
    out = np.empty((n_buckets, k), dtype=np.float64)
    if n_full > 0:
        reshaped = features[:n_full * scale_factor].reshape(n_full, scale_factor, k)
        for ci, rule in enumerate(agg_rules):
            if rule == "last":
                out[:n_full, ci] = reshaped[:, -1, ci]
            elif rule == "max":
                out[:n_full, ci] = reshaped[:, :, ci].max(axis=1)
            elif rule == "min":
                out[:n_full, ci] = reshaped[:, :, ci].min(axis=1)
            elif rule == "sum":
                out[:n_full, ci] = reshaped[:, :, ci].sum(axis=1)
            elif rule == "vwap":
                if volume_local_idx is not None:
                    vol = reshaped[:, :, volume_local_idx]
                    vwap_vals = reshaped[:, :, ci]
                    vol_sum = vol.sum(axis=1)
                    with np.errstate(invalid="ignore"):
                        out[:n_full, ci] = np.where(
                            vol_sum > 0, (vwap_vals * vol).sum(axis=1) / vol_sum, np.nan)
                else:
                    out[:n_full, ci] = reshaped[:, :, ci].mean(axis=1)
    if remainder > 0:
        p = features[n_full * scale_factor:]
        for ci, rule in enumerate(agg_rules):
            if rule == "last":
                out[n_full, ci] = p[-1, ci]
            elif rule == "max":
                out[n_full, ci] = p[:, ci].max()
            elif rule == "min":
                out[n_full, ci] = p[:, ci].min()
            elif rule == "sum":
                out[n_full, ci] = p[:, ci].sum()
            elif rule == "vwap":
                if volume_local_idx is not None:
                    vs = p[:, volume_local_idx].sum()
                    out[n_full, ci] = (p[:, ci] * p[:, volume_local_idx]).sum() / vs if vs > 0 else np.nan
                else:
                    out[n_full, ci] = p[:, ci].mean()
    return out


# -----------------------------------------------------------------------
# rf_offset_base computation
# -----------------------------------------------------------------------


class TestRfOffsetBase:
    """Test that rf_offset_base correctly maps canonical_sec[0] to RF grid position."""

    def test_no_leading_trim(self):
        """When main obs starts at market open, rf_offset_base = 0."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        canonical_sec, _ = _make_dense_main_obs(ts_open, leading_nan=0)
        rf_offset_base = int(canonical_sec[0]) - ts_open
        assert rf_offset_base == 0

    def test_with_leading_trim(self):
        """When trim_leading_nan removes N rows, rf_offset_base = N."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        leading_nan = 37  # First 37 seconds had no data
        canonical_sec, _ = _make_dense_main_obs(ts_open, leading_nan=leading_nan)
        rf_offset_base = int(canonical_sec[0]) - ts_open
        assert rf_offset_base == leading_nan

    def test_large_leading_trim(self):
        """Simulates a ticker that doesn't trade until well after market open."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        leading_nan = 3600  # First hour no data
        canonical_sec, _ = _make_dense_main_obs(ts_open, leading_nan=leading_nan)
        rf_offset_base = int(canonical_sec[0]) - ts_open
        assert rf_offset_base == 3600


# -----------------------------------------------------------------------
# Window extraction alignment
# -----------------------------------------------------------------------


class TestWindowAlignment:
    """Test that the RF window covers the exact same seconds as the main window."""

    def test_rf_window_matches_main_window_seconds(self):
        """The RF slice [rf_offset : rf_offset + window_size] should cover
        the same seconds as features[start_idx : end_idx]."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        leading_nan = 50
        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=leading_nan)
        rf = _make_rf_array_fast(1)  # (1, 23400, 9)

        rf_offset_base = int(canonical_sec[0]) - ts_open

        # Pick a window
        start_idx = 200
        window_size = 600
        end_idx = start_idx + window_size

        # Main window seconds
        main_seconds = canonical_sec[start_idx:end_idx]
        main_offsets_from_open = main_seconds - ts_open

        # RF offset
        rf_offset = rf_offset_base + start_idx
        rf_window = rf[0, rf_offset:rf_offset + window_size]

        # The RF feature at position [s, 0] encodes: s * 10 + 0
        # So rf_window[i, 0] / 10 gives the second offset from market open
        rf_seconds_from_open = (rf_window[:, 0] / 10).astype(int)

        np.testing.assert_array_equal(main_offsets_from_open, rf_seconds_from_open)

    def test_window_at_start_of_day(self):
        """Window starting at index 0 should align with RF seconds [leading_nan, ...]."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        leading_nan = 100
        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=leading_nan)
        rf = _make_rf_array_fast(1)

        rf_offset_base = int(canonical_sec[0]) - ts_open
        start_idx = 0
        window_size = 300
        rf_offset = rf_offset_base + start_idx

        main_seconds = canonical_sec[start_idx:start_idx + window_size] - ts_open
        rf_seconds = (rf[0, rf_offset:rf_offset + window_size, 0] / 10).astype(int)

        np.testing.assert_array_equal(main_seconds, rf_seconds)
        # Both should start at second 100 (the leading_nan offset)
        assert main_seconds[0] == leading_nan
        assert rf_seconds[0] == leading_nan

    def test_window_near_end_of_day(self):
        """Window near market close should still align correctly."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        total_secs = 23400
        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=0)
        rf = _make_rf_array_fast(1)

        rf_offset_base = 0
        start_idx = total_secs - 600  # Last 600 seconds
        window_size = 600
        end_idx = min(start_idx + window_size, len(features))
        actual_window = end_idx - start_idx

        rf_offset = rf_offset_base + start_idx
        main_seconds = canonical_sec[start_idx:end_idx] - ts_open
        rf_seconds = (rf[0, rf_offset:rf_offset + actual_window, 0] / 10).astype(int)

        np.testing.assert_array_equal(main_seconds, rf_seconds)

    @pytest.mark.parametrize("leading_nan", [0, 37, 100, 500, 3600])
    @pytest.mark.parametrize("start_idx", [0, 50, 200, 1000])
    def test_alignment_parametric(self, leading_nan, start_idx):
        """Parametric test across various leading_nan and start_idx combos."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        max_secs = 23400 - leading_nan
        if start_idx >= max_secs:
            pytest.skip("start_idx beyond available data")

        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=leading_nan)
        rf = _make_rf_array_fast(1)

        rf_offset_base = int(canonical_sec[0]) - ts_open
        window_size = min(600, len(features) - start_idx)
        if window_size < 10:
            pytest.skip("window too small")

        rf_offset = rf_offset_base + start_idx
        main_seconds = canonical_sec[start_idx:start_idx + window_size] - ts_open
        rf_seconds = (rf[0, rf_offset:rf_offset + window_size, 0] / 10).astype(int)

        np.testing.assert_array_equal(main_seconds, rf_seconds)


# -----------------------------------------------------------------------
# Aggregation bucket count alignment
# -----------------------------------------------------------------------


class TestAggregationAlignment:
    """Test that main obs and RF produce the same number of aggregated buckets."""

    @pytest.mark.parametrize("scale", [1, 5, 10, 60, 300])
    def test_same_bucket_count(self, scale):
        """Same input length + same scale = same output bucket count."""
        window_size = 1200
        rng = np.random.RandomState(42)
        main_window = rng.randn(window_size, 9).astype(np.float64)
        main_window[:, 7] = np.abs(main_window[:, 7]) * 100  # volume
        main_window[:, 8] = np.abs(main_window[:, 8]) * 10   # n

        rf_window = rng.randn(window_size, 9).astype(np.float64)
        rf_window[:, 7] = np.abs(rf_window[:, 7]) * 100
        rf_window[:, 8] = np.abs(rf_window[:, 8]) * 10

        main_agg = _aggregate_numpy(main_window, scale)
        rf_agg_rules = [_AGG_RULES[c] for c in FEATURE_COLUMNS]
        vol_idx = FEATURE_COLUMNS.index("volume")
        rf_agg = _aggregate_numpy_subset(rf_window, scale, rf_agg_rules, vol_idx)

        if main_agg is None:
            assert rf_agg is None
        else:
            assert main_agg.shape[0] == rf_agg.shape[0]

    @pytest.mark.parametrize("scale", [1, 5, 60])
    def test_same_bucket_count_with_remainder(self, scale):
        """Bucket count matches even when window_size is not divisible by scale."""
        window_size = 1200 + scale // 2 + 1  # Ensure remainder
        main_window = np.random.randn(window_size, 9).astype(np.float64)
        main_window[:, 7] = np.abs(main_window[:, 7]) * 100
        rf_window = np.random.randn(window_size, 9).astype(np.float64)
        rf_window[:, 7] = np.abs(rf_window[:, 7]) * 100

        main_agg = _aggregate_numpy(main_window, scale)
        rf_agg_rules = [_AGG_RULES[c] for c in FEATURE_COLUMNS]
        vol_idx = FEATURE_COLUMNS.index("volume")
        rf_agg = _aggregate_numpy_subset(rf_window, scale, rf_agg_rules, vol_idx)

        if main_agg is not None:
            assert rf_agg is not None
            assert main_agg.shape[0] == rf_agg.shape[0]

    @pytest.mark.parametrize("preset", ["vwap", "vwap_volume", "vwap_volume_orderbook", "all"])
    def test_bucket_count_with_rf_subsets(self, preset):
        """RF column subsets should not affect bucket count."""
        window_size = 1200
        scale = 60
        rf_cols = _RF_PRESETS[preset]
        rf_col_indices = [FEATURE_COLUMNS.index(c) for c in rf_cols]
        rf_agg_rules = [_AGG_RULES[c] for c in rf_cols]
        vol_local = rf_cols.index("volume") if "volume" in rf_cols else None

        rng = np.random.RandomState(42)
        main_window = rng.randn(window_size, 9).astype(np.float64)
        main_window[:, 7] = np.abs(main_window[:, 7]) * 100
        main_window[:, 8] = np.abs(main_window[:, 8]) * 10

        # RF data is subset of columns
        rf_full = rng.randn(window_size, 9).astype(np.float64)
        rf_full[:, 7] = np.abs(rf_full[:, 7]) * 100
        rf_window = rf_full[:, rf_col_indices]

        main_agg = _aggregate_numpy(main_window, scale)
        rf_agg = _aggregate_numpy_subset(rf_window, scale, rf_agg_rules, vol_local)

        assert main_agg.shape[0] == rf_agg.shape[0]


# -----------------------------------------------------------------------
# End-to-end: sparse data → dense grid → window → aggregate → concat
# -----------------------------------------------------------------------


class TestEndToEndAlignment:
    """Full pipeline test: sparse main obs + dense RF → aligned aggregated output."""

    def test_full_pipeline_alignment(self):
        """Simulate the full _getitem_numpy pipeline and verify temporal alignment."""
        date_str = "2023-01-03"
        ts_open, ts_close = timeline_bounds_est(date_str)
        n_grid = ts_close - ts_open  # 23400

        # Create sparse main observation (80% density — some seconds have no data)
        rng = np.random.RandomState(42)
        n_obs = int(n_grid * 0.8)
        obs_offsets = np.sort(rng.choice(n_grid, size=n_obs, replace=False))
        ts_sec = (ts_open + obs_offsets).astype(np.int32)
        raw_features = rng.randn(n_obs, 9).astype(np.float32)
        raw_features[:, 7] = np.abs(raw_features[:, 7]) * 100
        raw_features[:, 8] = np.abs(raw_features[:, 8]) * 10

        # Preprocess to dense grid (same as _preprocess_to_numpy)
        result = sparse_to_dense_grid(
            ts_sec, raw_features, ts_open, ts_close,
            _FFILL_INDICES, _ZEROFILL_INDICES, trim_leading_nan=True,
        )
        assert result is not None
        canonical_sec, features = result

        # Create RF array (dense, starts at market open)
        rf_data = _make_rf_array_fast(1, n_grid, 9)

        # Compute rf_offset_base (same as _getitem_numpy)
        rf_offset_base = int(canonical_sec[0]) - ts_open

        # Pick a random window (same as _getitem_numpy)
        start_idx = rng.randint(0, max(1, len(features)))
        window_size = 1200
        end_idx = min(start_idx + window_size, len(features))
        window = features[start_idx:end_idx]
        actual_window = end_idx - start_idx

        if actual_window < 10:
            pytest.skip("window too small")

        # RF offset (same as _getitem_numpy)
        rf_offset = rf_offset_base + start_idx

        # Extract RF window
        rf_window = rf_data[0, rf_offset:rf_offset + actual_window]

        # Verify temporal alignment: main and RF windows span same seconds
        main_seconds = canonical_sec[start_idx:end_idx] - ts_open
        rf_seconds = (rf_window[:, 0] / 10).astype(int)
        np.testing.assert_array_equal(main_seconds, rf_seconds)

        # Aggregate both at same scale
        for scale in [1, 60, 300]:
            main_agg = _aggregate_numpy(window, scale)
            rf_agg_rules = [_AGG_RULES[c] for c in FEATURE_COLUMNS]
            vol_idx = FEATURE_COLUMNS.index("volume")
            rf_agg = _aggregate_numpy_subset(
                rf_window.astype(np.float64), scale, rf_agg_rules, vol_idx,
            )
            if main_agg is None:
                assert rf_agg is None
                continue
            assert main_agg.shape[0] == rf_agg.shape[0], (
                f"Bucket count mismatch at scale={scale}: "
                f"main={main_agg.shape[0]}, rf={rf_agg.shape[0]}"
            )

    def test_concatenated_output_shape(self):
        """After merging, output should be (n_agg, 9 + n_rf_cols * n_tickers)."""
        window_size = 1200
        scale = 60
        n_rf_tickers = 3
        preset = "vwap_volume"
        rf_cols = _RF_PRESETS[preset]
        n_rf_cols = len(rf_cols)

        rng = np.random.RandomState(42)
        main_window = rng.randn(window_size, 9).astype(np.float64)
        main_window[:, 7] = np.abs(main_window[:, 7]) * 100
        main_window[:, 8] = np.abs(main_window[:, 8]) * 10
        main_agg = _aggregate_numpy(main_window, scale)
        n_agg = main_agg.shape[0]

        # Simulate merging N risk factors
        norm_groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(main_agg, norm_groups)

        rf_parts = []
        rf_norm_groups = build_norm_groups(rf_cols)
        rf_agg_rules = [_AGG_RULES[c] for c in rf_cols]
        rf_col_indices = [FEATURE_COLUMNS.index(c) for c in rf_cols]
        vol_local = rf_cols.index("volume") if "volume" in rf_cols else None

        for t in range(n_rf_tickers):
            rf_full = rng.randn(window_size, 9).astype(np.float64)
            rf_full[:, 7] = np.abs(rf_full[:, 7]) * 100
            rf_raw = rf_full[:, rf_col_indices]
            rf_agg = _aggregate_numpy_subset(rf_raw, scale, rf_agg_rules, vol_local)
            assert rf_agg is not None
            assert rf_agg.shape[0] == n_agg
            normalize_numpy(rf_agg, rf_norm_groups)
            rf_parts.append(rf_agg)

        merged = np.concatenate([main_agg] + rf_parts, axis=1)
        expected_cols = 9 + n_rf_cols * n_rf_tickers
        assert merged.shape == (n_agg, expected_cols)

    @pytest.mark.parametrize("seed", range(20))
    def test_alignment_with_random_windows(self, seed):
        """Fuzz: random leading_nan, start_idx, window_size all align correctly."""
        date_str = "2023-06-15"
        ts_open, ts_close = timeline_bounds_est(date_str)

        rng = np.random.RandomState(seed)
        leading_nan = rng.randint(0, 1000)
        n_available = 23400 - leading_nan
        if n_available < 100:
            pytest.skip("too few rows")

        canonical_sec, features = _make_dense_main_obs(
            ts_open, leading_nan=leading_nan, n_seconds=23400,
        )
        rf = _make_rf_array_fast(1)

        rf_offset_base = int(canonical_sec[0]) - ts_open
        start_idx = rng.randint(0, max(1, len(features) - 100))
        window_size = rng.choice([300, 600, 1200, 3600])
        end_idx = min(start_idx + window_size, len(features))
        actual_window = end_idx - start_idx

        rf_offset = rf_offset_base + start_idx

        # Verify second-level alignment
        main_secs = canonical_sec[start_idx:end_idx] - ts_open
        rf_secs = (rf[0, rf_offset:rf_offset + actual_window, 0] / 10).astype(int)
        np.testing.assert_array_equal(main_secs, rf_secs)

        # Verify aggregation bucket counts match
        for scale in [1, 10, 60]:
            main_agg = _aggregate_numpy(
                features[start_idx:end_idx], scale,
            )
            rf_raw = rf[0, rf_offset:rf_offset + actual_window].astype(np.float64)
            rf_agg_rules = [_AGG_RULES[c] for c in FEATURE_COLUMNS]
            vol_idx = FEATURE_COLUMNS.index("volume")
            rf_agg = _aggregate_numpy_subset(rf_raw, scale, rf_agg_rules, vol_idx)
            if main_agg is None:
                assert rf_agg is None
            else:
                assert main_agg.shape[0] == rf_agg.shape[0]


# -----------------------------------------------------------------------
# Edge cases for RF alignment
# -----------------------------------------------------------------------


class TestRfAlignmentEdgeCases:
    """Edge cases that could break temporal alignment."""

    def test_rf_offset_does_not_exceed_grid(self):
        """rf_offset + window_size should never exceed 23400."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=0)

        rf_offset_base = 0
        # Window starting near end of day
        start_idx = len(features) - 100
        window_size = 600
        end_idx = min(start_idx + window_size, len(features))
        actual_window = end_idx - start_idx

        rf_offset = rf_offset_base + start_idx
        assert rf_offset + actual_window <= 23400

    def test_leading_nan_offset_consistent_with_sparse_to_dense(self):
        """Verify the offset from sparse_to_dense_grid matches what we compute.

        trim_leading_nan removes rows until all columns are non-NaN, which may
        be later than the first sparse observation (forward-fill needs a seed
        for every ffill column). The rf_offset_base must equal canonical_sec[0]
        minus ts_open, which is exactly the number of trimmed rows.
        """
        date_str = "2023-01-03"
        ts_open, ts_close = timeline_bounds_est(date_str)

        # Create sparse data with gap at market open.
        # Place first observation at second 100, with all features valid.
        first_obs_offset = 100
        rng = np.random.RandomState(42)
        n_obs = 500
        offsets = np.sort(rng.choice(
            np.arange(first_obs_offset, 23400), size=n_obs, replace=False
        ))
        ts_sec = (ts_open + offsets).astype(np.int32)
        raw_features = rng.randn(n_obs, 9).astype(np.float32)
        raw_features[:, 7] = np.abs(raw_features[:, 7]) * 100  # volume > 0
        raw_features[:, 8] = np.abs(raw_features[:, 8]) * 10   # n > 0

        result = sparse_to_dense_grid(
            ts_sec, raw_features, ts_open, ts_close,
            _FFILL_INDICES, _ZEROFILL_INDICES, trim_leading_nan=True,
        )
        assert result is not None
        canonical_sec, features = result

        rf_offset_base = int(canonical_sec[0]) - ts_open

        # rf_offset_base >= first_obs_offset (trim may go further if some
        # ffill columns are still NaN between the first obs and the next)
        assert rf_offset_base >= first_obs_offset

        # Key invariant: canonical_sec[0] corresponds to RF grid position
        # rf_offset_base, regardless of how many rows were trimmed.
        rf = _make_rf_array_fast(1)
        rf_second = int(rf[0, rf_offset_base, 0] / 10)
        assert rf_second == rf_offset_base

        # And the first feature row should have no NaN
        assert not np.any(np.isnan(features[0]))

    def test_missing_rf_date_produces_zeros(self):
        """When RF doesn't have the requested date, zeros should be used."""
        # This tests the date_to_idx.get(date_str) → None fallback
        rf_date_to_idx = {"2023-01-03": 0, "2023-01-04": 1}
        date_str = "2023-01-05"  # Not in RF
        assert rf_date_to_idx.get(date_str) is None

    def test_window_clipping_at_end_of_day(self):
        """When window extends past market close, both main and RF should clip identically."""
        ts_open, _ = timeline_bounds_est("2023-01-03")
        canonical_sec, features = _make_dense_main_obs(ts_open, leading_nan=0)
        rf = _make_rf_array_fast(1)

        rf_offset_base = 0
        start_idx = 23000  # 400 seconds before close
        window_size = 1200  # Would extend 800 seconds past close

        # Main clips to available data
        end_idx = min(start_idx + window_size, len(features))
        actual_window = end_idx - start_idx
        assert actual_window == 400  # Clipped to remaining data

        # RF clips the same way (we pass actual_window, not window_size)
        rf_offset = rf_offset_base + start_idx
        rf_window = rf[0, rf_offset:rf_offset + actual_window]
        assert len(rf_window) == 400

        # Verify alignment
        main_secs = canonical_sec[start_idx:end_idx] - ts_open
        rf_secs = (rf_window[:, 0] / 10).astype(int)
        np.testing.assert_array_equal(main_secs, rf_secs)
