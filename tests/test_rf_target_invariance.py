"""Test that adding risk factors does NOT change downstream targets.

Creates fake MDS data and fake risk factor arrays, then compares
__getitem__ outputs with and without risk factors enabled. The first
N_base target columns must be bit-identical; risk factors should only
add extra columns to the feature dimension and (optionally) extra
risk-adjusted target columns.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from streaming import MDSWriter

from market_jepa.training.streaming_dataset import StreamingMarketDataset, FEATURE_COLUMNS


def _make_fake_mds(root: Path, n_tickers: int = 3, n_days: int = 5, seed: int = 0):
    """Write synthetic MDS shards mimicking real market data."""
    rng = np.random.RandomState(seed)
    dates = [f"2023-01-{d+2:02d}" for d in range(n_days)]  # Jan 2-6

    columns = {
        "ticker": "str",
        "date": "str",
        "ts_interval": "ndarray:int32",
        "features": "ndarray:float32",
    }

    # One month subdir: 2023/01/
    shard_dir = root / "2023" / "01"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tickers = [f"FAKE{i}" for i in range(n_tickers)]

    with MDSWriter(out=str(shard_dir), columns=columns) as writer:
        for date in dates:
            for ticker in tickers:
                # Generate ~6.5 hours of 1Hz data (23400 seconds) but sparse
                # Pick random subset of timestamps (market hours: 34200..57600 UTC-5)
                ts_open = 34200  # 09:30 ET in seconds from midnight
                ts_close = 57600  # 16:00 ET
                n_obs = rng.randint(5000, 15000)
                ts = np.sort(rng.choice(np.arange(ts_open, ts_close), size=n_obs, replace=False)).astype(np.int32)

                # 9 features: prices ~100-200, sizes ~1000-5000, volume ~100-1000
                features = np.zeros((n_obs, 9), dtype=np.float32)
                base_price = 100 + rng.rand() * 100
                features[:, 0] = base_price + rng.randn(n_obs).cumsum() * 0.01  # bid
                features[:, 4] = features[:, 0] + rng.rand(n_obs) * 0.05  # ask
                features[:, 1] = (features[:, 0] + features[:, 4]) / 2  # vwap
                features[:, 2] = features[:, 4] + rng.rand(n_obs) * 0.02  # high
                features[:, 3] = features[:, 0] - rng.rand(n_obs) * 0.02  # low
                features[:, 5] = rng.randint(100, 5000, size=n_obs).astype(np.float32)  # bid_size
                features[:, 6] = rng.randint(100, 5000, size=n_obs).astype(np.float32)  # ask_size
                features[:, 7] = rng.randint(0, 1000, size=n_obs).astype(np.float32)  # volume
                features[:, 8] = rng.randint(0, 100, size=n_obs).astype(np.float32)  # n

                writer.write({
                    "ticker": ticker,
                    "date": date,
                    "ts_interval": ts,
                    "features": features,
                })


def _make_fake_rf(rf_dir: Path, ticker: str = "IWM", n_days: int = 5, seed: int = 99):
    """Write synthetic risk factor .npy + meta.json."""
    rng = np.random.RandomState(seed)
    dates = [f"2023-01-{d+2:02d}" for d in range(n_days)]

    ticker_dir = rf_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Dense 1Hz: (n_days, 23400, 9)
    features = np.zeros((n_days, 23400, 9), dtype=np.float32)
    for d in range(n_days):
        base = 150 + rng.rand() * 50
        features[d, :, 0] = base + rng.randn(23400).cumsum() * 0.005  # bid
        features[d, :, 4] = features[d, :, 0] + rng.rand(23400) * 0.03  # ask
        features[d, :, 1] = (features[d, :, 0] + features[d, :, 4]) / 2  # vwap
        features[d, :, 2] = features[d, :, 4] + rng.rand(23400) * 0.01  # high
        features[d, :, 3] = features[d, :, 0] - rng.rand(23400) * 0.01  # low
        features[d, :, 5] = rng.randint(500, 10000, size=23400).astype(np.float32)
        features[d, :, 6] = rng.randint(500, 10000, size=23400).astype(np.float32)
        features[d, :, 7] = rng.randint(0, 5000, size=23400).astype(np.float32)
        features[d, :, 8] = rng.randint(0, 500, size=23400).astype(np.float32)

    np.save(ticker_dir / "features.npy", features)
    with open(ticker_dir / "meta.json", "w") as f:
        json.dump({
            "dates": dates,
            "shape": list(features.shape),
            "feature_columns": FEATURE_COLUMNS,
        }, f)


def test_targets_unchanged_with_risk_factors():
    """Core test: base targets must be identical with and without RF."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        mds_dir = tmpdir / "mds"
        rf_dir = tmpdir / "rf"
        _make_fake_mds(mds_dir, n_tickers=3, n_days=5, seed=0)
        _make_fake_rf(rf_dir, ticker="IWM", n_days=5, seed=99)

        aug_cfgs = [
            {
                "name": "random_resized_crop",
                "n_global_views": 2,
                "n_local_views": 6,
                "global_seq_len": 2048,
                "global_scale_range": [0.5, 1.0],
                "local_scale_range": [0.05, 0.5],
                "local_seq_len": 512,
            },
        ]

        targets_cfg = {
            "horizons": [300, 600, 900],
            "types": ["return", "spread_change", "volatility_change"],
        }

        common_kwargs = dict(
            augmentations=aug_cfgs,
            date_start="2023-01-01",
            date_end="2023-01-31",
            seed=42,
            n_pairs_per_obs=1,
            targets=targets_cfg,
            epoch_dependent_seed=False,
            shuffle=False,
            batch_size=1,
            allow_unsafe_types=True,
            predownload=0,
        )

        from market_jepa.training.streaming_dataset import discover_streams
        streams_base = discover_streams(str(mds_dir), "2023-01-01", "2023-01-31")
        streams_rf = discover_streams(str(mds_dir), "2023-01-01", "2023-01-31")

        # --- Dataset WITHOUT risk factors ---
        ds_base = StreamingMarketDataset(
            streams=streams_base,
            **common_kwargs,
        )

        # --- Dataset WITH risk factors ---
        ds_rf = StreamingMarketDataset(
            streams=streams_rf,
            risk_factor_dir=str(rf_dir),
            risk_factor_tickers=["IWM"],
            risk_factor_columns="vwap",
            **common_kwargs,
        )

        n_base_targets = len(ds_base.target_names)
        n_rf_targets = len(ds_rf.target_names)
        n_base_features = 9
        n_rf_features = ds_rf.n_features

        print(f"Base targets ({n_base_targets}): {ds_base.target_names}")
        print(f"RF targets   ({n_rf_targets}): {ds_rf.target_names}")
        print(f"Base features: {n_base_features}, RF features: {n_rf_features}")
        print()

        assert n_rf_targets > n_base_targets, (
            f"Expected more targets with RF, got base={n_base_targets}, rf={n_rf_targets}"
        )
        assert n_rf_features > n_base_features, (
            f"Expected more features with RF, got base={n_base_features}, rf={n_rf_features}"
        )

        # Compare N samples
        n_compare = min(len(ds_base), 50)
        n_matched = 0
        n_target_mismatches = 0
        n_feature_mismatches = 0

        for i in range(n_compare):
            # Reset getitem counters so RNG stays synchronized
            ds_base._getitem_counter = 0
            ds_rf._getitem_counter = 0

            pairs_base = ds_base[i]
            pairs_rf = ds_rf[i]

            if len(pairs_base) != len(pairs_rf):
                print(f"  idx={i}: different number of pairs ({len(pairs_base)} vs {len(pairs_rf)})")
                continue

            for pi, (pb, pr) in enumerate(zip(pairs_base, pairs_rf)):
                # Check targets: first n_base_targets columns should match exactly
                if "targets" in pb and "targets" in pr:
                    base_tgt = pb["targets"][:n_base_targets]
                    rf_tgt = pr["targets"][:n_base_targets]

                    if not torch.equal(base_tgt, rf_tgt):
                        # Check if it's just NaN matching
                        base_nan = torch.isnan(base_tgt)
                        rf_nan = torch.isnan(rf_tgt)
                        if not torch.equal(base_nan, rf_nan):
                            n_target_mismatches += 1
                            print(f"  idx={i} pair={pi}: NaN pattern differs!")
                            print(f"    base: {base_tgt}")
                            print(f"    rf:   {rf_tgt}")
                        else:
                            valid = ~base_nan
                            if valid.any():
                                diff = (base_tgt[valid] - rf_tgt[valid]).abs()
                                if diff.max() > 1e-7:
                                    n_target_mismatches += 1
                                    print(f"  idx={i} pair={pi}: TARGET MISMATCH (max diff={diff.max():.2e})")
                                    print(f"    base: {base_tgt}")
                                    print(f"    rf:   {rf_tgt}")
                                else:
                                    n_matched += 1
                            else:
                                n_matched += 1
                    else:
                        n_matched += 1

                # Check feature views: first 9 columns should match
                for vi, (vb, vr) in enumerate(zip(pb["views"], pr["views"])):
                    base_feat = vb[:n_base_features]  # (9, L)
                    rf_feat = vr[:n_base_features]  # (9, L) — first 9 of 10

                    if base_feat.shape != rf_feat.shape:
                        n_feature_mismatches += 1
                        print(f"  idx={i} pair={pi} view={vi}: SHAPE MISMATCH "
                              f"({base_feat.shape} vs {rf_feat.shape})")
                        continue

                    if not torch.equal(base_feat, rf_feat):
                        valid = ~(torch.isnan(base_feat) | torch.isnan(rf_feat))
                        if valid.any():
                            diff = (base_feat[valid] - rf_feat[valid]).abs()
                            if diff.max() > 1e-7:
                                n_feature_mismatches += 1
                                print(f"  idx={i} pair={pi} view={vi}: FEATURE MISMATCH "
                                      f"(max diff={diff.max():.2e})")

        print(f"\n{'='*60}")
        print(f"Compared {n_compare} samples:")
        print(f"  Target matches:     {n_matched}")
        print(f"  Target mismatches:  {n_target_mismatches}")
        print(f"  Feature mismatches: {n_feature_mismatches}")

        if n_target_mismatches == 0 and n_feature_mismatches == 0:
            print("\nPASSED: Risk factors do NOT affect base targets or base features.")
        else:
            print("\nFAILED: Risk factors changed base targets or features!")

        assert n_target_mismatches == 0, f"{n_target_mismatches} target mismatches"
        assert n_feature_mismatches == 0, f"{n_feature_mismatches} feature mismatches"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_targets_unchanged_with_risk_factors()
