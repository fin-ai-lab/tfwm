"""Tests for market_jepa.training.normalization."""

import numpy as np
import pytest

from stable_finance.dataset.transforms import FEATURE_TYPES, LOG1P_TYPES
from market_jepa.training.utils import (
    EPS,
    build_norm_groups,
    normalize_numpy,
)
from market_jepa.training.streaming_dataset import FEATURE_COLUMNS


# -----------------------------------------------------------------------
# build_norm_groups
# -----------------------------------------------------------------------


class TestBuildNormGroups:
    """Tests for build_norm_groups."""

    def test_all_feature_columns_covered(self):
        """Every column in FEATURE_COLUMNS should appear in exactly one group."""
        groups = build_norm_groups(FEATURE_COLUMNS)
        all_indices = []
        for indices, _ in groups:
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(FEATURE_COLUMNS)))

    def test_price_columns_no_log1p(self):
        """Price group should have apply_log1p=False."""
        groups = build_norm_groups(FEATURE_COLUMNS)
        price_cols = set(FEATURE_TYPES["price"])
        for indices, apply_log1p in groups:
            col_names = [FEATURE_COLUMNS[i] for i in indices]
            if any(c in price_cols for c in col_names):
                assert not apply_log1p, f"Price columns {col_names} should not have log1p"

    def test_size_columns_have_log1p(self):
        """Size/count columns should have apply_log1p=True.

        The log1p set is read from LOG1P_TYPES rather than restated here. This
        test used to spell the three keys out against a duplicate FEATURE_TYPES
        that lived in training.utils; the duplicate had drifted (``size_ob``
        vs ``order_book_size``) without anything noticing, because nothing on
        the training path read it.
        """
        groups = build_norm_groups(FEATURE_COLUMNS)
        log1p_cols = set()
        for ft in LOG1P_TYPES:
            log1p_cols.update(FEATURE_TYPES[ft])
        for indices, apply_log1p in groups:
            col_names = [FEATURE_COLUMNS[i] for i in indices]
            if any(c in log1p_cols for c in col_names):
                assert apply_log1p, f"Size columns {col_names} should have log1p"

    def test_subset_columns(self):
        """Should work with arbitrary column subsets (e.g. RF preset)."""
        cols = ["vwap_all", "volume"]
        groups = build_norm_groups(cols)
        all_indices = []
        for indices, _ in groups:
            all_indices.extend(indices)
        assert sorted(all_indices) == [0, 1]

    def test_single_column(self):
        """Single column should produce one group."""
        groups = build_norm_groups(["vwap_all"])
        assert len(groups) == 1
        assert groups[0][0] == [0]

    def test_unknown_column_silently_skipped(self):
        """Unknown column names should not crash (just not appear in any group)."""
        groups = build_norm_groups(["unknown_col"])
        total_indices = sum(len(indices) for indices, _ in groups)
        assert total_indices == 0


# -----------------------------------------------------------------------
# normalize_numpy
# -----------------------------------------------------------------------


class TestNormalizeNumpy:
    """Tests for normalize_numpy."""

    def test_price_z_score_stats(self):
        """After normalization, price columns should have mean ~0 and std ~1."""
        rng = np.random.RandomState(42)
        features = rng.randn(1000, 9).astype(np.float64) * 10 + 100
        features[:, 7] = np.abs(features[:, 7])  # volume positive
        features[:, 8] = np.abs(features[:, 8])  # n positive

        groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(features, groups)

        # Check price columns (0=bid, 1=vwap, 2=high, 3=low, 4=ask) have mean~0
        price_indices = [0, 1, 2, 3, 4]
        price_vals = features[:, price_indices]
        assert abs(np.mean(price_vals)) < 0.1

    def test_in_place_modification(self):
        """normalize_numpy should modify the array in place."""
        features = np.ones((10, 9), dtype=np.float64) * 50.0
        features_id = id(features)
        groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(features, groups)
        assert id(features) == features_id  # same object
        # Values should have changed
        assert not np.all(features == 50.0)

    def test_constant_input_no_nan(self):
        """Constant input should not produce NaN (EPS prevents division by zero)."""
        features = np.ones((100, 9), dtype=np.float64) * 42.0
        groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(features, groups)
        assert not np.any(np.isnan(features))

    def test_log1p_applied_to_volume(self):
        """Volume column should have log1p applied before z-score."""
        features = np.zeros((10, 9), dtype=np.float64)
        features[:, 7] = 100.0  # volume = 100
        features[:, 8] = 10.0   # n = 10

        groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(features, groups)

        # After log1p + z-score on constant values, result should be ~0
        # (mean subtracted, constant so numerator = 0)
        assert abs(features[0, 7]) < 0.1

    def test_shared_stats_across_feature_type(self):
        """Z-score should use shared mean/std across all columns of same type."""
        features = np.zeros((100, 9), dtype=np.float64)
        rng = np.random.RandomState(42)
        # bid_price (idx 0) = 100 ± small noise
        features[:, 0] = 100.0 + rng.randn(100) * 0.01
        # ask_price (idx 4) = 200 ± small noise (very different from bid)
        features[:, 4] = 200.0 + rng.randn(100) * 0.01
        # All other price columns at 150
        features[:, 1] = 150.0
        features[:, 2] = 150.0
        features[:, 3] = 150.0

        groups = build_norm_groups(FEATURE_COLUMNS)
        normalize_numpy(features, groups)

        # bid and ask should have different signs (bid << mean, ask >> mean)
        assert features[0, 0] < 0  # bid below mean
        assert features[0, 4] > 0  # ask above mean
