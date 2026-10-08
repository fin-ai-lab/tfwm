"""Tests for the same-stock invariance augmentations (time_warp, gaussian_noise).

Both draw one shared wall-clock window and emit n_global_views copies of the
same stock, differing only by the transformation. The dataset-level tests need
the real 2023 mosaic months on local disk and are skipped elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from market_jepa.augmentations import (
    AUGMENTATION_REGISTRY,
    _add_volume_noise_numpy,
    _aggregate_numpy_jittered,
    _draw_channel_mask,
    _price_jitter_numpy,
    _warped_aggregate_numpy,
)

MOSAIC_DIR = Path("lab/market-jepa-mosaic/1Hz_mosaic_mnth")

needs_mosaic = pytest.mark.skipif(
    not (MOSAIC_DIR / "2023" / "01" / "index.json").is_file(),
    reason="2023 mosaic months not available locally",
)


def test_registered():
    for name in (
        "time_warp", "gaussian_noise", "volume_noise",
        "price_jitter", "channel_drop",
    ):
        assert name in AUGMENTATION_REGISTRY


def _synthetic_window(W, seed=0):
    """(W, 9) float64 window with the mosaic column layout, including NaN
    vwap on zero-volume rows (the case the vwap aggregation must handle)."""
    rng = np.random.RandomState(seed)
    mid = 100 + np.cumsum(rng.randn(W)) * 0.01
    f = np.empty((W, 9))
    f[:, 0] = mid - 0.01  # bid_price
    f[:, 2] = mid + 0.02  # high
    f[:, 3] = mid - 0.02  # low
    f[:, 4] = mid + 0.01  # ask_price
    f[:, 5] = rng.randint(1, 50, W)  # bid_size
    f[:, 6] = rng.randint(1, 50, W)  # ask_size
    f[:, 7] = rng.randint(0, 5, W).astype(float)  # volume
    f[:, 8] = (f[:, 7] > 0).astype(float) * rng.randint(1, 3, W)  # n
    f[:, 1] = np.where(f[:, 7] > 0, mid, np.nan)  # vwap NaN when no trades
    return f


class TestWarpedAggregate:
    def test_strength_zero_matches_uniform(self):
        f = _synthetic_window(8 * 512)
        rng = np.random.RandomState(7)
        warped = _warped_aggregate_numpy(f, 512, rng, n_knots=8, strength=0.0)
        uniform = _aggregate_numpy_jittered(f, 8)
        np.testing.assert_allclose(warped, uniform, equal_nan=True)

    def test_shape_and_totals_preserved(self):
        f = _synthetic_window(6 * 256, seed=3)
        for strength in (0.1, 0.25, 0.5, 1.0):
            rng = np.random.RandomState(11)
            out = _warped_aggregate_numpy(f, 256, rng, n_knots=8, strength=strength)
            assert out.shape == (256, 9)
            # Buckets tile the window exactly, so summed columns are conserved.
            assert out[:, 7].sum() == pytest.approx(f[:, 7].sum())
            assert out[:, 8].sum() == pytest.approx(f[:, 8].sum())
            # high >= low everywhere; prices stay inside the window's range.
            assert (out[:, 2] >= out[:, 3]).all()

    def test_views_differ_and_are_deterministic(self):
        f = _synthetic_window(8 * 512, seed=5)
        a = _warped_aggregate_numpy(f, 512, np.random.RandomState(1), 8, 0.25)
        b = _warped_aggregate_numpy(f, 512, np.random.RandomState(1), 8, 0.25)
        c = _warped_aggregate_numpy(f, 512, np.random.RandomState(2), 8, 0.25)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c, equal_nan=True)

    def test_degenerate_window_is_uniform(self):
        # W == seq_len leaves no room to warp: every bucket is one row.
        f = _synthetic_window(512, seed=9)
        out = _warped_aggregate_numpy(f, 512, np.random.RandomState(0), 8, 1.0)
        uniform = _aggregate_numpy_jittered(f, 1)
        np.testing.assert_allclose(out, uniform, equal_nan=True)

    def test_too_short_returns_none(self):
        f = _synthetic_window(100)
        assert _warped_aggregate_numpy(f, 512, np.random.RandomState(0), 8, 0.25) is None


class TestRawSpaceCorruptions:
    def test_volume_noise_touches_only_activity(self):
        f = _synthetic_window(1024, seed=2)
        v = f.copy()
        _add_volume_noise_numpy(v, np.random.RandomState(0), frac=0.5)
        np.testing.assert_array_equal(v[:, :7], f[:, :7])
        assert (v[:, 7] >= f[:, 7]).all() and (v[:, 7] > f[:, 7]).any()
        assert (v[:, 8] >= f[:, 8]).all() and (v[:, 8] > f[:, 8]).any()

    def test_price_jitter_never_crosses_book(self):
        f = _synthetic_window(1024, seed=4)
        v = f.copy()
        _price_jitter_numpy(v, np.random.RandomState(0), frac=4.0)
        # The ladder moves as one: spread preserved exactly, high >= low.
        np.testing.assert_allclose(v[:, 4] - v[:, 0], f[:, 4] - f[:, 0])
        assert (v[:, 4] >= v[:, 0]).all()
        assert (v[:, 2] >= v[:, 3]).all()
        assert (v[:, 0] != f[:, 0]).any()
        # Non-price channels untouched.
        np.testing.assert_array_equal(v[:, 5:], f[:, 5:])

    def test_price_jitter_locked_book_unchanged(self):
        f = _synthetic_window(256, seed=6)
        f[:, 4] = f[:, 0]  # zero spread everywhere
        v = f.copy()
        _price_jitter_numpy(v, np.random.RandomState(0), frac=2.0)
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(f))

    def test_channel_mask_never_blank(self):
        rng = np.random.RandomState(0)
        for _ in range(200):
            idx = _draw_channel_mask(rng, 9, drop_p=0.9)
            assert len(idx) < 9
            assert ((idx >= 0) & (idx < 9)).all()
        assert len(_draw_channel_mask(rng, 9, drop_p=0.0)) == 0


@needs_mosaic
@pytest.mark.parametrize("bad_p", [1.0, 1.5, -0.1])
def test_channel_drop_p_out_of_range_rejected(bad_p):
    """drop_p >= 1 would spin _draw_channel_mask forever in a worker."""
    with pytest.raises(ValueError, match=r"channel_drop_p must be in \[0, 1\)"):
        _make_dataset(
            augmentations=[{"name": "channel_drop", "channel_drop_p": bad_p}],
        )


@needs_mosaic
def test_time_warp_with_risk_factors_rejected_at_construction():
    """The warped grid breaks RF wall-clock alignment — fail before training."""
    with pytest.raises(ValueError, match="time_warp with risk factors"):
        _make_dataset(
            augmentations=[{"name": "time_warp"}],
            risk_factor_dir="/nonexistent",
            risk_factor_tickers=["IWM"],
        )


def _make_dataset(months=("2023/01",), **overrides):
    from streaming import Stream

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    kwargs = dict(
        augmentations=[{"name": "gaussian_noise"}],
        date_start="2023-01-01",
        date_end="2023-02-28",
        seed=1234,
        epoch_dependent_seed=False,
        # PINNED to the pre-2026-09-13 constructor defaults. These tests are
        # about augmentation mechanics -- grouping, warping, which channels a
        # noise op touches -- and assert a 9-row view. The information token
        # now defaults ON, so leaving these unset would silently retarget them
        # at a 20-row view and test something else.
        info_norm_stats=False,
        info_window=False,
        targets={"horizons": [300], "types": ["return"]},
        streams=[Stream(local=str(MOSAIC_DIR / m)) for m in months],
        shuffle=False,
        batch_size=4,
        allow_unsafe_types=True,
    )
    kwargs.update(overrides)
    return StreamingMarketDataset(**kwargs)


@needs_mosaic
class TestSameStockViewDataset:
    def test_gaussian_noise_structure(self):
        ds = _make_dataset(
            augmentations=[{"name": "gaussian_noise", "noise_sigma": 0.1}],
        )
        pair = ds[7][0]
        assert pair["n_global_views"] == 2
        assert len(pair["views"]) == 2
        v1, v2 = pair["views"]
        assert v1.shape == (9, 2048) and v2.shape == (9, 2048)
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
        assert pair["targets"].shape == (1,)
        # Same window + independent noise: difference is N(0, 2 sigma^2).
        diff = (v1 - v2).numpy().ravel()
        assert diff.std() == pytest.approx(0.1 * np.sqrt(2), rel=0.05)

    def test_gaussian_noise_sigma_zero_identical(self):
        ds = _make_dataset(
            augmentations=[{"name": "gaussian_noise", "noise_sigma": 0.0}],
        )
        v1, v2 = ds[7][0]["views"]
        assert torch.equal(v1, v2)

    def test_gaussian_noise_leaves_the_information_channels_alone(self):
        """The noise is for the SERIES; the info token is a per-window fact.

        Regression for a bug live until 2026-08-29: the draw was taken at the
        full view shape, so the 8 norm-stat and 3 window channels were noised
        too. They are window metadata and are never
        standardized, so sigma landed anywhere from 0.7x to 8x their natural
        spread -- tod_start, a fraction of the session, was reaching 2.717.

        The other tests in this class run with the info flags OFF, which is
        why the views there are (9, 2048) and why none of them caught it.
        """
        ds = _make_dataset(
            augmentations=[{"name": "gaussian_noise", "noise_sigma": 0.75}],
            info_norm_stats=True, info_window=True,
        )
        v1, v2 = ds[7][0]["views"]
        n_info = ds._n_info_features
        assert n_info == 11
        assert v1.shape == (9 + n_info, 2048)

        info1, info2 = v1[-n_info:], v2[-n_info:]
        # Only the final timestep carries the payload; it is not repeated.
        assert torch.count_nonzero(info1[:, :-1]) == 0
        assert torch.equal(info1, info2)

        # ...and the series channels are still noised, with both views drawing
        # independently: the difference is N(0, 2 sigma^2).
        diff = (v1[:9] - v2[:9]).numpy().ravel()
        assert diff.std() == pytest.approx(0.75 * np.sqrt(2), rel=0.05)

    def test_time_warp_structure(self):
        ds = _make_dataset(
            augmentations=[{"name": "time_warp", "warp_strength": 0.25}],
        )
        pair = ds[7][0]
        assert pair["n_global_views"] == 2
        v1, v2 = pair["views"]
        assert v1.shape == (9, 2048) and v2.shape == (9, 2048)
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
        assert not torch.equal(v1, v2)
        # Same underlying window: the slow price channels stay near-identical
        # under a local re-bucketing (fast microstructure channels — sizes,
        # volume, n — decorrelate, which is the augmentation's point).
        for ch in range(5):
            corr = np.corrcoef(v1.numpy()[ch], v2.numpy()[ch])[0, 1]
            assert corr > 0.8

    def test_time_warp_three_views(self):
        ds = _make_dataset(
            augmentations=[{"name": "time_warp", "n_global_views": 3}],
        )
        pair = ds[11][0]
        assert pair["n_global_views"] == 3
        assert len(pair["views"]) == 3
        assert pair["lengths"].tolist() == [2048] * 3

    def test_volume_noise_isolated_to_activity_channels(self):
        ds = _make_dataset(
            augmentations=[{"name": "volume_noise", "vol_noise_frac": 1.0}],
        )
        v1, v2 = ds[7][0]["views"]
        # volume and n normalize as their own groups, so post-norm the
        # corruption stays in channels 7-8; prices/sizes are identical.
        assert torch.equal(v1[:7], v2[:7])
        assert not torch.equal(v1[7], v2[7])
        assert not torch.equal(v1[8], v2[8])
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all()

    def test_price_jitter_isolated_to_price_channels(self):
        ds = _make_dataset(
            augmentations=[{"name": "price_jitter", "price_jitter_frac": 2.0}],
        )
        v1, v2 = ds[7][0]["views"]
        assert not torch.equal(v1[:5], v2[:5])
        assert torch.equal(v1[5:], v2[5:])
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all()

    def test_channel_drop_zeroes_whole_channels(self):
        ds = _make_dataset(
            augmentations=[{"name": "channel_drop", "channel_drop_p": 0.5}],
        )
        found_drop = False
        for i in (7, 11, 23):
            for v in ds[i][0]["views"]:
                zero_rows = (v == 0).all(dim=1)
                assert not zero_rows.all()
                found_drop = found_drop or bool(zero_rows.any())
        assert found_drop

    def test_channel_drop_p_zero_identical(self):
        ds = _make_dataset(
            augmentations=[{"name": "channel_drop", "channel_drop_p": 0.0}],
        )
        v1, v2 = ds[7][0]["views"]
        assert torch.equal(v1, v2)

    def test_deterministic(self):
        for name in (
            "time_warp", "gaussian_noise", "volume_noise",
            "price_jitter", "channel_drop",
        ):
            p1 = _make_dataset(augmentations=[{"name": name}])[11][0]
            p2 = _make_dataset(augmentations=[{"name": name}])[11][0]
            assert p1["ticker"] == p2["ticker"]
            for v1, v2 in zip(p1["views"], p2["views"]):
                assert torch.equal(v1, v2)
