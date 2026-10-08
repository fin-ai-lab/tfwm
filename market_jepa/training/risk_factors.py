"""Risk-factor input channels, shared by the training and eval paths.

A risk factor (e.g. IWM) is merged into a view as EXTRA INPUT CHANNELS: the
same wall-clock window, aggregated at the same scale, normalized in its own
feature groups, concatenated on the column axis. It is the only way a model
that otherwise sees one instance-normalized stock at a time can observe
anything about the market.

This lives outside StreamingMarketDataset because the synchronized-panel eval
(`scripts/eval/xs_ic_eval.py`) has to build byte-identical views. The
reported metric's whole design is that ONLY the predictor step differs between
arms, so a second implementation of the merge — with its own aggregation rules,
its own ffill, its own normalization grouping — would quietly break exactly the
comparison it exists to serve. One implementation, two callers.

Not to be confused with risk-factor TARGET columns (risk-adjusted returns),
which are a separate feature controlled by ``DatasetConfig.risk_factor_targets``
and are incompatible with the cross-sectional anchor tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import build_norm_groups, normalize_numpy


class RiskFactorMerger:
    """Loads risk-factor series and concatenates them onto aggregated views.

    Args:
        risk_factor_dir: directory holding ``<TICKER>/{meta.json,features.npy}``.
        tickers: risk-factor tickers, in the order their columns are appended.
        columns: output column names (a ``_RF_PRESETS`` value already resolved
            to a list). ``mid_price`` is synthesized from bid/ask.
        feature_columns: the raw on-disk column order of ``features.npy``.
        agg_rules: column name -> aggregation rule, as used for the main view.
    """

    def __init__(self, risk_factor_dir, tickers, columns, feature_columns,
                 agg_rules):
        self.col_names = list(columns)
        self.n_features = len(self.col_names) * len(tickers)

        # Extraction plan: which raw column indices to read, and how each
        # output column is derived from them (passthrough or synthesized mid).
        self.raw_indices: list[int] = []
        self.post_specs: list[tuple] = []
        pos_of: dict[int, int] = {}

        def _ensure(raw_idx: int) -> int:
            pos = pos_of.get(raw_idx)
            if pos is None:
                pos = len(self.raw_indices)
                self.raw_indices.append(raw_idx)
                pos_of[raw_idx] = pos
            return pos

        for c in self.col_names:
            if c == "mid_price":
                b = _ensure(feature_columns.index("bid_price"))
                a = _ensure(feature_columns.index("ask_price"))
                self.post_specs.append(("mid", b, a))
            else:
                self.post_specs.append(("col", _ensure(feature_columns.index(c))))

        self.agg_rules = [agg_rules[c] for c in self.col_names]
        self.norm_groups = build_norm_groups(self.col_names)
        self.vwap_idx = (self.col_names.index("vwap_all")
                         if "vwap_all" in self.col_names else None)
        self.volume_idx = (self.col_names.index("volume")
                           if "volume" in self.col_names else None)

        self.series: dict[str, dict] = {}
        for ticker in tickers:
            d = Path(risk_factor_dir) / ticker
            with open(d / "meta.json") as f:
                meta = json.load(f)
            self.series[ticker] = {
                "features": np.load(d / "features.npy", mmap_mode="r"),
                "date_to_idx": {dt: i for i, dt in enumerate(meta["dates"])},
            }

    # ---- aggregation ----------------------------------------------------

    def _aggregate(self, features: np.ndarray, scale: int) -> np.ndarray | None:
        if scale == 1:
            return features.copy()
        n = len(features)
        n_full = n // scale
        remainder = n % scale
        n_buckets = n_full + (1 if remainder else 0)
        if n_buckets < 2:
            return None
        k = features.shape[1]
        out = np.empty((n_buckets, k), dtype=np.float64)

        if n_full > 0:
            r = features[: n_full * scale].reshape(n_full, scale, k)
            for ci, rule in enumerate(self.agg_rules):
                if rule == "last":
                    out[:n_full, ci] = r[:, -1, ci]
                elif rule == "max":
                    out[:n_full, ci] = r[:, :, ci].max(axis=1)
                elif rule == "min":
                    out[:n_full, ci] = r[:, :, ci].min(axis=1)
                elif rule == "sum":
                    out[:n_full, ci] = r[:, :, ci].sum(axis=1)
                elif rule == "vwap":
                    if self.volume_idx is not None:
                        vol = r[:, :, self.volume_idx]
                        vs = vol.sum(axis=1)
                        with np.errstate(invalid="ignore"):
                            out[:n_full, ci] = np.where(
                                vs > 0, (r[:, :, ci] * vol).sum(axis=1) / vs, np.nan)
                    else:
                        out[:n_full, ci] = r[:, :, ci].mean(axis=1)

        if remainder > 0:
            p = features[n_full * scale:]
            for ci, rule in enumerate(self.agg_rules):
                if rule == "last":
                    out[n_full, ci] = p[-1, ci]
                elif rule == "max":
                    out[n_full, ci] = p[:, ci].max()
                elif rule == "min":
                    out[n_full, ci] = p[:, ci].min()
                elif rule == "sum":
                    out[n_full, ci] = p[:, ci].sum()
                elif rule == "vwap":
                    if self.volume_idx is not None:
                        vs = p[:, self.volume_idx].sum()
                        out[n_full, ci] = (
                            (p[:, ci] * p[:, self.volume_idx]).sum() / vs
                            if vs > 0 else np.nan)
                    else:
                        out[n_full, ci] = p[:, ci].mean()
        return out

    @staticmethod
    def _ffill(arr: np.ndarray, col: int) -> None:
        vals = arr[:, col]
        mask = np.isnan(vals)
        if not mask.any():
            return
        idx = np.arange(len(vals))
        idx[mask] = 0
        np.maximum.accumulate(idx, out=idx)
        vals[:] = vals[idx]
        still = np.isnan(vals)
        if still.any():
            vals[still] = 0.0

    # ---- the merge ------------------------------------------------------

    def merge(self, view: np.ndarray, date_str: str, rf_offset: int,
              window_size: int, scale: int) -> np.ndarray:
        """Concatenate risk-factor columns onto an aggregated, normalized view.

        ``rf_offset`` indexes the dense RF grid from the 09:30 open. A negative
        offset would silently slice from the END of the day in numpy, so it is
        refused rather than allowed to produce a plausible wrong window.
        """
        n_agg = len(view)
        n_out = len(self.post_specs)
        if rf_offset < 0:
            raise ValueError(
                f"rf_offset={rf_offset} is before the 09:30 RF grid origin; "
                "risk factors do not cover the extended session.")

        parts = []
        for rf in self.series.values():
            date_idx = rf["date_to_idx"].get(date_str)
            if date_idx is None:
                parts.append(np.zeros((n_agg, n_out), dtype=np.float64))
                continue
            win = rf["features"][date_idx, rf_offset: rf_offset + window_size]
            loaded = win[:, self.raw_indices].astype(np.float64)
            raw = np.empty((len(loaded), n_out), dtype=np.float64)
            for ci, spec in enumerate(self.post_specs):
                if spec[0] == "mid":
                    raw[:, ci] = (loaded[:, spec[1]] + loaded[:, spec[2]]) / 2.0
                else:
                    raw[:, ci] = loaded[:, spec[1]]
            agg = self._aggregate(raw, scale)
            if agg is None or len(agg) < n_agg:
                parts.append(np.zeros((n_agg, n_out), dtype=np.float64))
                continue
            agg = agg[:n_agg]
            if self.vwap_idx is not None:
                self._ffill(agg, self.vwap_idx)
            normalize_numpy(agg, self.norm_groups)
            parts.append(agg)
        return np.concatenate([view] + parts, axis=1)
