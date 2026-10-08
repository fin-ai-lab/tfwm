"""Tests for DatasetConfig.zero_feature_columns (feature-ablation knob)."""

from pathlib import Path

import pytest
import torch

MOSAIC_DIR = Path("lab/market-jepa-mosaic/1Hz_mosaic_mnth")

needs_mosaic = pytest.mark.skipif(
    not (MOSAIC_DIR / "2023" / "01" / "index.json").is_file(),
    reason="2023 mosaic months not available locally",
)


def _make(zero_cols, augmentations=None):
    from streaming import Stream

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    return StreamingMarketDataset(
        augmentations=augmentations
        or [{"name": "random_resized_crop", "n_global_views": 2, "n_local_views": 2}],
        date_start="2023-01-01",
        date_end="2023-01-31",
        seed=7,
        epoch_dependent_seed=False,
        zero_feature_columns=zero_cols,
        streams=[Stream(local=str(MOSAIC_DIR / "2023" / "01"))],
        shuffle=False,
        batch_size=4,
        allow_unsafe_types=True,
    )


@needs_mosaic
def test_size_channels_zeroed_in_all_views():
    ds = _make(["bid_size", "ask_size"])
    checked = 0
    for idx in range(5):
        for pair in ds[idx]:
            if pair.get("bucket_key", -1) == -1:
                continue
            for v in pair["views"]:  # (9, L)
                assert torch.all(v[5] == 0), "bid_size not zeroed"
                assert torch.all(v[6] == 0), "ask_size not zeroed"
                # other channels still carry information
                assert v[0].abs().sum() > 0
                checked += 1
    assert checked > 0


@needs_mosaic
def test_no_zeroing_by_default():
    ds = _make([])
    pair = ds[0][0]
    assert any(pair["views"][0][5].abs().sum() > 0 for _ in [0])


def test_unknown_column_raises():
    with pytest.raises((ValueError, Exception)):
        _make(["not_a_column"])
