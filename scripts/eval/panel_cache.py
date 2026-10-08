"""market-jepa adapter for stable-finance's model-neutral panel cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from market_jepa.augmentations import encode_view_metadata
from stable_finance.dataset import (
    CacheClaimed as Claimed,
    PanelCache,
    panel_cache_key,
)


def cache_root() -> Path | None:
    value = os.environ.get("MJ_PANEL_CACHE")
    return Path(value) if value else None


def panel_key(*, anchors_per_day: int, stats_tag: str, has_rf: bool,
              norm_groups, fixed_agg, seq_len,
              info_norm_stats=None, info_window=None) -> dict:
    """Identify the model-neutral panel; token encoding flags are not data."""
    return panel_cache_key(
        anchors_per_day=int(anchors_per_day),
        stats_tag=str(stats_tag),
        has_rf=bool(has_rf),
        norm_groups=("default" if norm_groups is None
                     else [[list(map(int, group[0])), bool(group[1])]
                           for group in norm_groups]),
        fixed_agg=None if fixed_agg is None else int(fixed_agg),
        seq_len=int(seq_len),
    )


def _cache(root) -> PanelCache:
    return PanelCache(root)


def panel_dir(root: Path, ym: str, key: dict) -> Path:
    return _cache(root).directory(ym, key)


def is_built(root: Path, ym: str, key: dict) -> bool:
    return _cache(root).is_built(ym, key)


def load(root: Path, ym: str, key: dict):
    """Return the stable-finance cache object for specialized consumers."""
    return _cache(root).load(ym, key)


def iter_cached(root: Path, ym: str, key: dict, batch_size: int, *,
                info_norm_stats: bool = False, info_window: bool = False):
    """Adapt natural metadata to market-jepa's optional information token."""
    for views, observations in _cache(root).iter_batches(ym, key, batch_size):
        if info_norm_stats or info_window:
            encoded = np.stack([
                encode_view_metadata(
                    row.metadata,
                    include_normalization=info_norm_stats,
                    include_window=info_window,
                )
                for row in observations
            ])
            payload = np.zeros(
                (len(views), views.shape[1], encoded.shape[1]), dtype=np.float32,
            )
            payload[:, -1, :] = encoded
            views = np.concatenate([views, payload], axis=2)
        # The quote is PERSISTED as of cache format 2. It was not, and the
        # framing here used to call that transitional -- wrongly: the WRITER
        # omitted it too, so newly built panels carried none either, and
        # panel_key had no way to retire the ones that did. A cached panel
        # therefore handed the execution stage an all-NaN market permanently:
        # nothing fills, every spread charges zero, and a frictionless Sharpe
        # reports as a spread-charged one. NaN still means "this row had no
        # bid/ask", which is a real state; what it no longer means is "this
        # cache format cannot say".
        metas = [
            (row.target, row.date, row.anchor, row.ticker, row.raw_target,
             row.quote)
            for row in observations
        ]
        yield views, metas


def build(root: Path, ym: str, key: dict, panel_iter, *, overwrite=False):
    """Build from stable-finance ``(views, PanelObservation[])`` batches."""
    return _cache(root).build(ym, key, panel_iter, overwrite=overwrite)
