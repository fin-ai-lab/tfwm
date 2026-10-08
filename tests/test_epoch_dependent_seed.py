"""Tests for the counter-based augmentation seed feature.

Verifies that augmentation seeds vary (or stay fixed) across __getitem__
calls depending on the epoch_dependent_seed flag, using a per-call counter
and worker_id instead of the MosaicML epoch.
"""

import numpy as np
import pytest

from market_jepa.augmentations import _random_resized_crop_numpy


# -----------------------------------------------------------------------
# Helpers – replicate the seed derivation from StreamingMarketDataset
# -----------------------------------------------------------------------

_LCG_MUL = 6_364_136_223_846_793_005  # pair multiplier (from _getitem_numpy)
_EPOCH_MUL = 2_654_435_761             # counter multiplier (Knuth hash constant)


def _compute_base_seed(base_seed: int, idx: int, counter: int,
                       epoch_dependent: bool, worker_id: int = 0) -> int:
    """Mirror the seed computation in StreamingMarketDataset._getitem_numpy."""
    if epoch_dependent:
        return int((base_seed + idx + counter * _EPOCH_MUL + worker_id * 48271) % (2**32))
    else:
        return int((base_seed + idx) % (2**32))


def _compute_pair_seed(base_seed: int, pair_idx: int) -> int:
    return int((base_seed + pair_idx * _LCG_MUL) % (2**32))


def _make_dense_features(n_rows=23400, seed=42):
    """Dense (n_rows, 9) feature array mimicking post-preprocessing data."""
    rng = np.random.RandomState(seed)
    features = np.empty((n_rows, 9), dtype=np.float64)
    features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01
    features[:, 2] = features[:, 0] + rng.rand(n_rows) * 0.1
    features[:, 3] = features[:, 0] - rng.rand(n_rows) * 0.1
    features[:, 4] = features[:, 0] + 0.05 + rng.randn(n_rows) * 0.005
    features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)
    features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)
    features[:, 7] = rng.randint(1, 500, n_rows).astype(float)
    features[:, 8] = rng.randint(1, 50, n_rows).astype(float)
    mid = (features[:, 0] + features[:, 4]) / 2
    features[:, 1] = mid + rng.randn(n_rows) * 0.005
    return features


# -----------------------------------------------------------------------
# Seed arithmetic tests
# -----------------------------------------------------------------------


class TestSeedArithmetic:
    """Verify the seed derivation formula without needing MDS data."""

    def test_counter0_same_regardless_of_flag(self):
        """At counter=0 and worker_id=0, both modes produce base_seed + idx."""
        for idx in [0, 1, 42, 9999]:
            s_on = _compute_base_seed(42, idx, counter=0, epoch_dependent=True, worker_id=0)
            s_off = _compute_base_seed(42, idx, counter=0, epoch_dependent=False, worker_id=0)
            assert s_on == s_off

    def test_enabled_different_counters_different_seeds(self):
        """epoch_dependent_seed=True: same idx, different counters → different seeds."""
        seeds = {
            _compute_base_seed(42, idx=100, counter=c, epoch_dependent=True)
            for c in range(10)
        }
        assert len(seeds) == 10, "Expected 10 unique seeds across 10 counter values"

    def test_disabled_different_counters_same_seed(self):
        """epoch_dependent_seed=False: same idx, different counters → same seed."""
        seeds = {
            _compute_base_seed(42, idx=100, counter=c, epoch_dependent=False)
            for c in range(10)
        }
        assert len(seeds) == 1, "Expected 1 unique seed (flag off)"

    def test_different_idx_different_seeds(self):
        """Different sample indices produce different seeds (both modes)."""
        for epoch_dep in [True, False]:
            seeds = {
                _compute_base_seed(42, idx=i, counter=3, epoch_dependent=epoch_dep)
                for i in range(100)
            }
            assert len(seeds) == 100

    def test_pair_seeds_unique(self):
        """Multiple pairs from the same base seed are all distinct."""
        base = _compute_base_seed(42, idx=0, counter=0, epoch_dependent=True)
        pair_seeds = {_compute_pair_seed(base, p) for p in range(8)}
        assert len(pair_seeds) == 8

    def test_different_workers_different_seeds(self):
        """Same idx, same counter, different worker_ids → different seeds."""
        seeds = {
            _compute_base_seed(42, idx=100, counter=5, epoch_dependent=True, worker_id=w)
            for w in range(32)
        }
        assert len(seeds) == 32, "Expected 32 unique seeds across 32 workers"

    def test_same_idx_resampled_different_seeds(self):
        """Mega-epoch resampling: same idx at different counter values → different seeds.

        This is the core property needed for mega-epoch: when the same sample
        appears multiple times (due to epoch_size > natural dataset size),
        each occurrence gets a unique augmentation seed.
        """
        idx = 42
        seeds = {
            _compute_base_seed(42, idx=idx, counter=c, epoch_dependent=True)
            for c in [100, 500, 1000, 5000, 10000]
        }
        assert len(seeds) == 5, "Expected unique seeds for each resampled occurrence"


# -----------------------------------------------------------------------
# End-to-end: verify augmentation output actually differs across counters
# -----------------------------------------------------------------------


class TestAugmentationVariesByCounter:
    """Feed counter-varied seeds into _random_resized_crop_numpy and check
    that actual crop outputs differ."""

    def test_crops_differ_across_counters(self):
        """Same observation + different counter seeds → different crop windows."""
        features = _make_dense_features(23400)
        scale_range = (0.3, 1.0)
        target_seq_len = 512

        results = []
        for counter in range(3):
            base = _compute_base_seed(42, idx=100, counter=counter, epoch_dependent=True)
            pair_seed = _compute_pair_seed(base, pair_idx=0)
            rng = np.random.RandomState(pair_seed)
            view, start, agg, adj = _random_resized_crop_numpy(
                features, scale_range, target_seq_len, rng,
            )
            assert view is not None
            results.append((start, agg, adj))

        # At least two of the three counters should produce different windows
        assert len(set(results)) > 1, (
            f"All 3 counter values produced identical crop params: {results[0]}"
        )

    def test_crops_identical_when_disabled(self):
        """epoch_dependent_seed=False: same observation + any counter → same crop."""
        features = _make_dense_features(23400)
        scale_range = (0.3, 1.0)
        target_seq_len = 512

        results = []
        for counter in range(3):
            base = _compute_base_seed(42, idx=100, counter=counter, epoch_dependent=False)
            pair_seed = _compute_pair_seed(base, pair_idx=0)
            rng = np.random.RandomState(pair_seed)
            view, start, agg, adj = _random_resized_crop_numpy(
                features, scale_range, target_seq_len, rng,
            )
            assert view is not None
            results.append((start, agg, adj))

        assert len(set(results)) == 1, (
            f"Expected identical crops when flag is off, got: {results}"
        )

    def test_crops_match_at_counter0(self):
        """Both modes produce identical crops at counter=0, worker_id=0."""
        features = _make_dense_features(23400)
        scale_range = (0.3, 1.0)
        target_seq_len = 512

        views = []
        for epoch_dep in [True, False]:
            base = _compute_base_seed(42, idx=100, counter=0, epoch_dependent=epoch_dep, worker_id=0)
            pair_seed = _compute_pair_seed(base, pair_idx=0)
            rng = np.random.RandomState(pair_seed)
            view, start, agg, adj = _random_resized_crop_numpy(
                features, scale_range, target_seq_len, rng,
            )
            assert view is not None
            views.append(view)

        np.testing.assert_array_equal(views[0], views[1])
