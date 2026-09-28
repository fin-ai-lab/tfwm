"""Diagnose risk factor data quality issues that could explain case 6 underperformance.

Checks:
1. NaN prevalence in RF data vs main data
2. Date coverage gaps
3. Feature distributions before/after normalization
4. Whether RF features dominate or distort the normalized feature vector
5. Actual sample-level comparison: what the model sees with vs without RF
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import os

MOSAIC_DIR = os.environ.get("MOSAIC_DIR", "data/1Hz_mosaic_mnth")
RISK_FACTOR_DIR = os.environ.get("RISK_FACTOR_DIR", "data/1Hz_risk_factors")
from market_jepa.training.streaming_dataset import (
    FEATURE_COLUMNS,
    StreamingMarketDataset,
    _RF_PRESETS,
    discover_streams,
)
from market_jepa.training.utils import build_norm_groups, normalize_numpy


def diagnose_rf_files():
    """Check raw RF .npy files for NaN, zeros, coverage."""
    rf_dir = Path(RISK_FACTOR_DIR)
    tickers = ["IWM", "XLE", "VIXY"]

    print("=" * 70)
    print("1. RAW RISK FACTOR FILE DIAGNOSTICS")
    print("=" * 70)

    for ticker in tickers:
        ticker_dir = rf_dir / ticker
        if not ticker_dir.exists():
            print(f"\n  {ticker}: DIRECTORY NOT FOUND at {ticker_dir}")
            continue

        meta_path = ticker_dir / "meta.json"
        feat_path = ticker_dir / "features.npy"

        with open(meta_path) as f:
            meta = json.load(f)

        dates = meta["dates"]
        features = np.load(feat_path, mmap_mode="r")

        print(f"\n  {ticker}:")
        print(f"    Shape: {features.shape}")
        print(f"    Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        print(f"    Dtype: {features.dtype}")

        # Check a sample of days for NaN/zero patterns
        sample_days = np.linspace(0, len(dates) - 1, min(20, len(dates)), dtype=int)
        nan_fracs = []
        zero_fracs = []
        for di in sample_days:
            day_data = np.array(features[di])  # Load from mmap
            nan_frac = np.isnan(day_data).mean()
            zero_frac = (day_data == 0).mean()
            nan_fracs.append(nan_frac)
            zero_fracs.append(zero_frac)

        print(f"    NaN fraction: mean={np.mean(nan_fracs):.4f}, "
              f"min={np.min(nan_fracs):.4f}, max={np.max(nan_fracs):.4f}")
        print(f"    Zero fraction: mean={np.mean(zero_fracs):.4f}, "
              f"min={np.min(zero_fracs):.4f}, max={np.max(zero_fracs):.4f}")

        # Check feature-level stats for one day
        day_data = np.array(features[len(dates) // 2])  # Middle day
        date_str = dates[len(dates) // 2]
        print(f"    Per-feature stats (day={date_str}):")
        for fi, col in enumerate(FEATURE_COLUMNS):
            col_data = day_data[:, fi]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                print(f"      {col:>12s}: ALL NaN")
            else:
                print(f"      {col:>12s}: mean={valid.mean():12.4f}, std={valid.std():12.4f}, "
                      f"min={valid.min():12.4f}, max={valid.max():12.4f}, "
                      f"nan%={np.isnan(col_data).mean()*100:.1f}%, zero%={(col_data==0).mean()*100:.1f}%")

        # Check training date coverage
        train_dates = set()
        import datetime
        d = datetime.date(2023, 1, 1)
        end = datetime.date(2023, 1, 31)
        while d <= end:
            train_dates.add(d.isoformat())
            d += datetime.timedelta(days=1)
        rf_dates = set(dates)
        missing = train_dates - rf_dates
        if missing:
            print(f"    Missing training dates: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            print(f"    All training dates covered")


def diagnose_normalization():
    """Compare normalization behavior for main vs RF features."""
    print("\n" + "=" * 70)
    print("2. NORMALIZATION DIAGNOSTICS")
    print("=" * 70)

    rf_dir = Path(RISK_FACTOR_DIR)
    rf_cols_preset = "vwap_volume_orderbook"
    rf_cols = _RF_PRESETS[rf_cols_preset]
    rf_col_indices = [FEATURE_COLUMNS.index(c) for c in rf_cols]

    print(f"\n  Preset: {rf_cols_preset}")
    print(f"  RF columns: {rf_cols}")
    print(f"  RF column indices: {rf_col_indices}")

    # Build norm groups
    main_norm = build_norm_groups(FEATURE_COLUMNS)
    rf_norm = build_norm_groups(rf_cols)

    print(f"\n  Main norm groups:")
    for indices, log1p in main_norm:
        cols = [FEATURE_COLUMNS[i] for i in indices]
        print(f"    {cols} -> log1p={log1p}")

    print(f"\n  RF norm groups:")
    for indices, log1p in rf_norm:
        cols = [rf_cols[i] for i in indices]
        print(f"    {cols} -> log1p={log1p}")

    # Load a real RF day and normalize it
    for ticker in ["IWM", "XLE", "VIXY"]:
        ticker_dir = rf_dir / ticker
        if not ticker_dir.exists():
            continue
        with open(ticker_dir / "meta.json") as f:
            meta = json.load(f)
        features = np.load(ticker_dir / "features.npy", mmap_mode="r")

        mid_idx = len(meta["dates"]) // 2
        day = np.array(features[mid_idx]).astype(np.float64)

        # Simulate what _merge_risk_factors does:
        # 1. Extract RF column subset
        rf_subset = day[:, rf_col_indices].copy()

        # 2. Check pre-normalization stats
        print(f"\n  {ticker} (date={meta['dates'][mid_idx]}):")
        print(f"    Pre-normalization RF subset stats:")
        for ci, col in enumerate(rf_cols):
            v = rf_subset[:, ci]
            valid = v[~np.isnan(v)]
            if len(valid) > 0:
                print(f"      {col:>12s}: mean={valid.mean():12.4f}, std={valid.std():12.4f}, "
                      f"nan%={np.isnan(v).mean()*100:.1f}%")

        # 3. Normalize and check post-normalization
        rf_normed = rf_subset.copy()
        normalize_numpy(rf_normed, rf_norm)
        print(f"    Post-normalization RF subset stats:")
        for ci, col in enumerate(rf_cols):
            v = rf_normed[:, ci]
            valid = v[~np.isnan(v)]
            if len(valid) > 0:
                print(f"      {col:>12s}: mean={valid.mean():12.2f}, std={valid.std():12.2f}, "
                      f"nan%={np.isnan(v).mean()*100:.1f}%")

        # 4. Compare to main feature normalization on same day's data
        main_normed = day.copy()
        normalize_numpy(main_normed, main_norm)
        print(f"    Post-normalization MAIN feature stats (same day):")
        for ci, col in enumerate(FEATURE_COLUMNS):
            v = main_normed[:, ci]
            valid = v[~np.isnan(v)]
            if len(valid) > 0:
                print(f"      {col:>12s}: mean={valid.mean():12.2f}, std={valid.std():12.2f}")


def diagnose_real_samples():
    """Load real dataset and compare samples with/without RF."""
    print("\n" + "=" * 70)
    print("3. REAL SAMPLE COMPARISON (with vs without RF)")
    print("=" * 70)

    streams_base = discover_streams(MOSAIC_DIR, "2023-01-01", "2023-01-31")
    streams_rf = discover_streams(MOSAIC_DIR, "2023-01-01", "2023-01-31")

    aug_cfgs = [{
        "name": "random_resized_crop",
        "n_global_views": 2,
        "n_local_views": 0,
        "global_seq_len": 2048,
        "global_scale_range": [0.5, 1.0],
        "local_scale_range": [0.05, 0.5],
        "local_seq_len": 512,
    }]

    targets_cfg = {
        "horizons": [300, 600, 900],
        "types": ["return", "spread_change", "volatility_change"],
    }

    common = dict(
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

    ds_base = StreamingMarketDataset(streams=streams_base, **common)
    ds_rf = StreamingMarketDataset(
        streams=streams_rf,
        risk_factor_dir=RISK_FACTOR_DIR,
        risk_factor_tickers=["IWM", "XLE", "VIXY"],
        risk_factor_columns="vwap_volume_orderbook",
        **common,
    )

    print(f"\n  Base: {ds_base.n_features} features, {len(ds_base.target_names)} targets")
    print(f"  RF:   {ds_rf.n_features} features, {len(ds_rf.target_names)} targets")
    print(f"  RF target names: {ds_rf.target_names}")

    n_base_targets = len(ds_base.target_names)
    n_samples = min(len(ds_base), 30)

    target_diffs = []
    feature_norms_base = []
    feature_norms_rf_main = []
    feature_norms_rf_extra = []
    nan_counts_rf = []
    zero_counts_rf = []

    for i in range(n_samples):
        ds_base._getitem_counter = 0
        ds_rf._getitem_counter = 0

        pairs_base = ds_base[i]
        pairs_rf = ds_rf[i]

        if len(pairs_base) == 0 or len(pairs_rf) == 0:
            continue

        pb = pairs_base[0]
        pr = pairs_rf[0]

        if "targets" not in pb or "targets" not in pr:
            continue

        # Target comparison
        base_tgt = pb["targets"][:n_base_targets]
        rf_tgt = pr["targets"][:n_base_targets]
        valid = ~(torch.isnan(base_tgt) | torch.isnan(rf_tgt))
        if valid.any():
            diff = (base_tgt[valid] - rf_tgt[valid]).abs().max().item()
            target_diffs.append(diff)

        # Feature magnitude comparison (first global view)
        v_base = pb["views"][0]  # (9, L)
        v_rf = pr["views"][0]    # (9+18, L)

        # L2 norm per feature channel
        base_norms = v_base.norm(dim=-1)  # (9,)
        rf_main_norms = v_rf[:9].norm(dim=-1)  # (9,) — main features
        rf_extra_norms = v_rf[9:].norm(dim=-1)  # (18,) — RF features

        feature_norms_base.append(base_norms)
        feature_norms_rf_main.append(rf_main_norms)
        feature_norms_rf_extra.append(rf_extra_norms)

        # NaN and zero stats in RF features
        rf_extra = v_rf[9:]
        nan_counts_rf.append(torch.isnan(rf_extra).float().mean().item())
        zero_counts_rf.append((rf_extra == 0).float().mean().item())

    if target_diffs:
        print(f"\n  Target diffs (base columns, max abs): "
              f"mean={np.mean(target_diffs):.2e}, max={np.max(target_diffs):.2e}")

    if feature_norms_base:
        base_norms = torch.stack(feature_norms_base).mean(0)
        rf_main_norms = torch.stack(feature_norms_rf_main).mean(0)
        rf_extra_norms = torch.stack(feature_norms_rf_extra).mean(0)

        print(f"\n  Avg L2 norms per feature channel (across {len(feature_norms_base)} samples):")
        print(f"    Main features (base dataset):     {base_norms.tolist()}")
        print(f"    Main features (RF dataset):       {rf_main_norms.tolist()}")
        print(f"    RF features (extra {rf_extra_norms.shape[0]} channels): {rf_extra_norms.tolist()}")

        print(f"\n  Summary:")
        print(f"    Main feature avg norm: {base_norms.mean():.4f}")
        print(f"    RF extra feature avg norm: {rf_extra_norms.mean():.4f}")
        print(f"    Ratio (RF/main): {rf_extra_norms.mean() / base_norms.mean():.4f}")

    if nan_counts_rf:
        print(f"\n  RF features NaN%: mean={np.mean(nan_counts_rf)*100:.2f}%")
        print(f"  RF features Zero%: mean={np.mean(zero_counts_rf)*100:.2f}%")


if __name__ == "__main__":
    print("Risk Factor Diagnostics")
    print("Config: case 6 = IWM+XLE+VIXY, vwap_volume_orderbook")
    print()

    diagnose_rf_files()
    diagnose_normalization()
    diagnose_real_samples()
