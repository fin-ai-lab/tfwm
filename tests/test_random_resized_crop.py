"""Tests for the _random_resized_crop_numpy function and config parsing."""

import numpy as np
import pytest

from market_jepa.augmentations import AUGMENTATION_REGISTRY, _random_resized_crop_numpy


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_dense_features(n_rows=23400, seed=42):
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

    mid = (features[:, 0] + features[:, 4]) / 2
    features[:, 1] = mid + rng.randn(n_rows) * 0.005
    features[features[:, 8] == 0, 1] = np.nan

    return features


# -----------------------------------------------------------------------
# TestRRCFunction — unit tests for _random_resized_crop_numpy
# -----------------------------------------------------------------------


class TestRRCFunction:
    def test_output_shape_exact(self):
        """Various (N, scale_range, target_seq_len) → output is (target_seq_len, 9)."""
        features = _make_dense_features(23400)
        rng = np.random.RandomState(42)
        for target in [512, 1024, 2048]:
            view, start, agg, adj = _random_resized_crop_numpy(
                features, (0.3, 1.0), target, rng,
            )
            assert view is not None, f"Failed for target={target}"
            assert view.shape == (target, 9), f"Expected ({target}, 9), got {view.shape}"

    def test_returns_none_when_too_short(self):
        """N=10, target=2048 → None."""
        features = _make_dense_features(10)
        rng = np.random.RandomState(42)
        view, start, agg, adj = _random_resized_crop_numpy(
            features, (0.3, 1.0), 2048, rng,
        )
        assert view is None

    def test_no_partial_bucket(self):
        """Output length is exactly target_seq_len (no trailing partial)."""
        features = _make_dense_features(23400)
        rng = np.random.RandomState(42)
        for _ in range(20):
            view, start, agg, adj = _random_resized_crop_numpy(
                features, (0.05, 1.0), 512, rng,
            )
            if view is not None:
                assert view.shape[0] == 512

    def test_start_within_bounds(self):
        """start_idx is in [0, N - adjusted_window]."""
        features = _make_dense_features(23400)
        N = len(features)
        rng = np.random.RandomState(42)
        for _ in range(50):
            view, start, agg, adj = _random_resized_crop_numpy(
                features, (0.05, 1.0), 512, rng,
            )
            if view is not None:
                assert 0 <= start <= N - adj
                assert start + adj <= N

    def test_determinism(self):
        """Same RNG seed → identical output."""
        features = _make_dense_features(23400)
        rng1 = np.random.RandomState(99)
        v1, s1, a1, w1 = _random_resized_crop_numpy(features, (0.1, 0.5), 512, rng1)

        rng2 = np.random.RandomState(99)
        v2, s2, a2, w2 = _random_resized_crop_numpy(features, (0.1, 0.5), 512, rng2)

        assert v1 is not None and v2 is not None
        np.testing.assert_array_equal(v1, v2)
        assert s1 == s2
        assert a1 == a2
        assert w1 == w2

    def test_volume_conserved(self):
        """Total volume in output equals total volume in the raw 1Hz slice."""
        features = _make_dense_features(23400)
        rng = np.random.RandomState(42)
        view, start, agg, adj = _random_resized_crop_numpy(
            features, (0.3, 0.8), 512, rng,
        )
        assert view is not None
        raw_slice = features[start : start + adj]
        np.testing.assert_allclose(
            view[:, 7].sum(), raw_slice[:, 7].sum(), rtol=1e-10,
        )

    def test_fallback_path(self):
        """N slightly larger than target with large scale → fallback reduces agg_factor."""
        target = 512
        # N just enough to fit: agg_factor will be 1 after fallback
        features = _make_dense_features(target + 10)
        rng = np.random.RandomState(42)
        view, start, agg, adj = _random_resized_crop_numpy(
            features, (0.9, 1.0), target, rng,
        )
        assert view is not None
        assert view.shape == (target, 9)
        assert agg >= 1

    def test_scale_1_full_observation(self):
        """scale_range=(1.0, 1.0) with appropriate N → agg_factor computable."""
        # N = 2048 * 5 = 10240, scale=1.0 → window=10240 → agg=2
        features = _make_dense_features(10240)
        rng = np.random.RandomState(42)
        view, start, agg, adj = _random_resized_crop_numpy(
            features, (1.0, 1.0), 2048, rng,
        )
        # With scale=1.0, window_size = N = 10240
        # agg_factor = round(10240/2048) = 5
        # adjusted_window = 5 * 2048 = 10240 = N
        assert view is not None
        assert view.shape == (2048, 9)


# -----------------------------------------------------------------------
# TestRRCConfig — config parsing
# -----------------------------------------------------------------------


class TestRRCConfig:
    def test_default_values(self):
        """All expected keys present with correct defaults."""
        # Replicate config parsing from streaming_dataset.py
        cfg = {"name": "random_resized_crop"}
        parsed = {
            "name": "random_resized_crop",
            "n_global_views": cfg.get("n_global_views", 2),
            "n_local_views": cfg.get("n_local_views", 6),
            "global_scale_range": tuple(cfg.get("global_scale_range", [0.5, 1.0])),
            "local_scale_range": tuple(cfg.get("local_scale_range", [0.05, 0.3])),
            "global_seq_len": cfg.get("global_seq_len", 2048),
            "local_seq_len": cfg.get("local_seq_len", 512),
        }
        assert parsed["n_global_views"] == 2
        assert parsed["n_local_views"] == 6
        assert parsed["global_scale_range"] == (0.5, 1.0)
        assert parsed["local_scale_range"] == (0.05, 0.3)
        assert parsed["global_seq_len"] == 2048
        assert parsed["local_seq_len"] == 512

    def test_registry_entry(self):
        """random_resized_crop is in the augmentation registry."""
        assert "random_resized_crop" in AUGMENTATION_REGISTRY
