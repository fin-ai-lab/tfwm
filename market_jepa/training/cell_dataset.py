"""Supervised cells served from the day-major store.

THE CONTRACT IS THE cross_stock BRANCH OF ``StreamingMarketDataset``: each
item is one cell -- K stocks at one (date, anchor, resolution) window -- as
the same dict that branch emits (K ``(C, L)`` view tensors, ``lengths``,
``targets`` of ``(K, n_targets)``, the four-way ``target_metadata``,
``xs_group``, the probe labels), so ``collate_bucketed`` and
``SupervisedModel.training_step`` do not know which loader fed them.
tests/test_cell_dataset.py holds the two loaders' views against each other
on one synthetic month.

WHAT IS DIFFERENT IS THE COST. The MDS branch fetches K shuffled records,
rebuilds nothing (the mosaic is dense) but decodes, crops, forward-fills and
normalizes each of the K views in its own Python round trip, computes K
forward targets from the 1 Hz grid and looks each up in a month-wide anchor
table. Here the day's cross-section is one memmap, the K windows are K
contiguous slices of it, ``stable_finance.dataset.cells`` aggregates and
normalizes the whole cell in one vectorized pass, and the targets -- raw and
all four transforms -- were computed by the writer and are read as a
``(K, T, H)`` block. Six loader CPUs per GPU is the constraint this is
written for.

THE EPOCH. One cell per ticker-day of the span, so ``len(dataset)`` -- and
with it ``steps_per_epoch`` and every ``num_epochs`` recipe -- means what it
meant on the mosaic: a 12-month span at 7 passes is still 84 month-passes.
Each epoch is a fresh permutation over the WHOLE span (a 6- or 12-month
shuffle, not a per-shard block), the cell draws are seeded by (seed, epoch,
slot) so a run is reproducible for any worker count, and worker ``w`` of
``W`` takes slots ``w::W``. The iterator does not stop at an epoch boundary:
the trainer counts optimizer steps and stops itself, exactly as the
mega-epoch sizing arranged for the mosaic loader.
"""

from __future__ import annotations

import datetime
import logging
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stable_finance.dataset import MARKET_SCHEMA, build_norm_groups, get_target_names
from stable_finance.dataset.anchors import ANCHOR_STEP
from stable_finance.dataset.cells import CellGeometry, build_cell_views, draw_cell
from stable_finance.dataset.daystore import DayRecord, discover_days

from ..augmentations import (
    _EPS as INFO_EPS, SESSION_SECONDS as INFO_SESSION, N_WINDOW_INFO,
)

logger = logging.getLogger(__name__)

TRANSFORMS = ("raw", "zscore", "uniform", "rank")
MAX_DRAW_ATTEMPTS = 8


def _mix(*parts: int) -> int:
    """A deterministic 32-bit seed from a few integers (splitmix-style)."""
    x = 0x9E3779B97F4A7C15
    for p in parts:
        x = (x ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return int(x & 0xFFFFFFFF)


class DayStoreCellDataset(IterableDataset):
    def __init__(
        self,
        *,
        daystore_dir,
        date_start,
        date_end,
        cell: dict,
        targets: dict,
        xs_target: str = "uniform",
        seed: int = 42,
        feature_columns=None,
        zero_feature_columns=None,
        norm_mode: str = "per_view",
        info_norm_stats: bool = True,
        info_window: bool = True,
        schedule=None,
        max_open_days: int = 48,
    ):
        """
        Args:
            cell: the canonicalized ``cross_stock`` config (``n_stocks``,
                ``global_seq_len``, ``global_scale_range``, ``global_agg_range``).
            targets: ``{"horizons": [...], "types": [...]}``; the slack the
                draw reserves before the close follows from it exactly as the
                anchor-table setup pins it (one horizon -> h + 60 s).
        """
        super().__init__()
        if cell.get("name") != "cross_stock":
            raise ValueError(f"the daystore loader serves cross_stock cells, got {cell.get('name')!r}")
        if int(cell.get("n_local_views", 0) or 0) or cell.get("industry_table"):
            raise ValueError("local views and industry tables are not served from the daystore")
        self.n_stocks = int(cell["n_stocks"])
        if self.n_stocks < 2:
            raise ValueError("a cell needs at least 2 stocks")
        horizons = [int(h) for h in targets.get("horizons", [900])]
        types = [str(t) for t in targets.get("types", ["return"])]
        self._horizons, self._types = horizons, types
        self._target_names = get_target_names(horizons, types)
        if xs_target not in TRANSFORMS:
            raise ValueError(f"xs_target must be one of {TRANSFORMS}, got {xs_target!r}")
        self._xs_target = xs_target
        agg_range = cell.get("global_agg_range")
        self.geometry = CellGeometry(
            sequence_length=int(cell["global_seq_len"]),
            scale_range=tuple(cell.get("global_scale_range", (0.5, 1.0))),
            aggregation_seconds=(None if agg_range is None else (int(agg_range[0]), int(agg_range[1]))),
            anchor_step=ANCHOR_STEP,
            slack_seconds=CellGeometry.slack_for(horizons),
        )
        self.feature_columns = list(feature_columns or MARKET_SCHEMA.columns)
        if self.feature_columns != list(MARKET_SCHEMA.columns):
            raise ValueError("the daystore carries the market schema's nine columns in order")
        zero = list(zero_feature_columns or [])
        unknown = [c for c in zero if c not in self.feature_columns]
        if unknown:
            raise ValueError(f"zero_feature_columns not in feature_columns: {unknown}")
        self._zero_indices = [self.feature_columns.index(c) for c in zero]
        if norm_mode not in ("per_view", "none"):
            raise ValueError(f"unknown norm_mode: {norm_mode!r}")
        if info_norm_stats and norm_mode == "none":
            raise ValueError("info_norm_stats needs a normalization to report")
        self._norm_groups = [] if norm_mode == "none" else build_norm_groups(self.feature_columns)
        self.info_norm_stats, self.info_window = bool(info_norm_stats), bool(info_window)
        self._n_info_features = (2 * len(self._norm_groups) if self.info_norm_stats else 0) \
            + (N_WINDOW_INFO if self.info_window else 0)
        self._seed = int(seed)
        self._schedule = schedule
        self._max_open = int(max_open_days)

        self._date_start = datetime.date.fromisoformat(str(date_start))
        self._date_end = datetime.date.fromisoformat(str(date_end))
        paths = discover_days(daystore_dir, str(self._date_start), str(self._date_end))
        if schedule is not None:
            paths = [p for p in paths if not schedule.is_closed(p.name)]
        self._day_paths = paths
        # Cells per day = tickers per day, read from the month indexes so no
        # record is opened here (this object is pickled into every worker).
        self._day_tickers = np.asarray([_n_tickers(p) for p in paths], dtype=np.int64)
        self._slot_bounds = np.concatenate([[0], np.cumsum(self._day_tickers)])
        self._n_cells = int(self._slot_bounds[-1])
        if not self._n_cells:
            raise ValueError(f"no ticker-days in {daystore_dir} for [{date_start}, {date_end}]")
        self._records: OrderedDict[int, DayRecord] = OrderedDict()
        self._small_dates: set[str] = set()
        # Slots that produced no cell this epoch. A slot whose day simply has
        # too few tickers is reported once per date by the warning in _cell;
        # this counts the OTHER case, where the day is big enough but no draw
        # in MAX_DRAW_ATTEMPTS found n_stocks feasible tickers, which is
        # otherwise a silent shortfall -- the slot is skipped, the epoch is
        # quietly smaller than epoch_size, and nothing says so.
        self._n_skipped = 0
        # For the trainer's mega-epoch message and progress accounting.
        self.epoch_size = self._n_cells

    # -- the surface StreamingMarketDataset exposes ------------------------------

    def __len__(self) -> int:
        return self._n_cells

    @property
    def n_features(self) -> int:
        return len(self.feature_columns) + self._n_info_features

    @property
    def target_names(self) -> list[str]:
        return list(self._target_names)

    @property
    def n_days(self) -> int:
        return len(self._day_paths)

    # -- records -----------------------------------------------------------------

    def _record(self, day_idx: int) -> DayRecord:
        rec = self._records.get(day_idx)
        if rec is None:
            rec = DayRecord(self._day_paths[day_idx])
            self._records[day_idx] = rec
            while len(self._records) > self._max_open:
                self._records.popitem(last=False)
        else:
            self._records.move_to_end(day_idx)
        return rec

    # -- iteration ---------------------------------------------------------------

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        epoch = 0
        while True:
            order = np.random.RandomState(_mix(self._seed, epoch, 0x5EED)).permutation(self._n_cells)
            self._n_skipped = 0
            n_slots = 0
            for slot in order[wid::nw]:
                n_slots += 1
                item = self._cell(int(slot), epoch)
                if item is not None:
                    yield [item]
            if self._n_skipped:
                logger.warning(
                    "daystore: worker %d/%d yielded %d of %d slots in epoch %d "
                    "(%.2f%% skipped: no draw in %d attempts found %d feasible "
                    "tickers)", wid, nw, n_slots - self._n_skipped, n_slots,
                    epoch, 100.0 * self._n_skipped / max(n_slots, 1),
                    MAX_DRAW_ATTEMPTS, self.n_stocks)
            epoch += 1

    def cell_at(self, slot: int, epoch: int = 0, *, with_draw: bool = False):
        """The item slot ``slot`` of ``epoch`` yields -- for tests and tools.

        ``with_draw`` also returns ``(day, ticker indices, draw)`` so a caller
        can rebuild the same cell through another path and compare.
        """
        return self._cell(int(slot), int(epoch), with_draw=with_draw)

    def _cell(self, slot: int, epoch: int, *, with_draw: bool = False):
        day_idx = int(np.searchsorted(self._slot_bounds, slot, side="right") - 1)
        day = self._record(day_idx)
        rng = np.random.RandomState(_mix(self._seed, epoch, slot))
        K = self.n_stocks
        for _ in range(MAX_DRAW_ATTEMPTS):
            draw = draw_cell(rng, self.geometry, day.rows, day.open_tod)
            if draw is None:
                continue
            candidates = np.flatnonzero(day.feasible_from(draw.start_row))
            if len(candidates) < K:
                if day.n_tickers < K and day.date not in self._small_dates:
                    self._small_dates.add(day.date)
                    logger.warning("daystore: %s has %d tickers, fewer than n_stocks=%d; "
                                   "no cell can be drawn on this date", day.date, day.n_tickers, K)
                continue
            idx = candidates[rng.permutation(len(candidates))[:K]]
            item = self._build(day, idx, draw)
            return (item, day, idx, draw) if with_draw else item
        self._n_skipped += 1
        return (None, day, None, None) if with_draw else None

    def _build(self, day: DayRecord, idx: np.ndarray, draw) -> dict:
        window = day.window(idx, draw.start_row, draw.window)
        if self._zero_indices:
            window[:, self._zero_indices, :] = 0.0
        views, stats = build_cell_views(
            day.features, idx, draw, groups=self._norm_groups, window=window)
        K, C, L = views.shape
        out = np.zeros((K, C + self._n_info_features, L), dtype=np.float32)
        out[:, :C, :] = views
        if self._n_info_features:
            out[:, C:, -1] = self._info(stats, day.open_tod + draw.start_row, draw.aggregation, L)
        tensor = torch.from_numpy(out)
        anchor_tod = draw.anchor_tod
        metadata = {
            name: torch.from_numpy(day.targets(name, anchor_tod, idx, self._types, self._horizons))
            for name in TRANSFORMS
        }
        tod_sec = day.open_tod + draw.start_row
        try:
            weekday = datetime.date.fromisoformat(day.date).weekday()
        except ValueError:
            weekday = -1
        return {
            "views": [tensor[k] for k in range(K)],
            "lengths": torch.full((K,), L, dtype=torch.long),
            "bucket_key": 0,
            "n_global_views": K,
            "ticker": str(day.tickers[idx[0]]),
            "date": day.date,
            "tod_sec": int(tod_sec),
            "tod_bucket": int(min(12, max(0, tod_sec // 1800))),
            "weekday": int(weekday),
            "agg_factor": int(draw.aggregation),
            "target_metadata": metadata,
            "targets": metadata[self._xs_target],
            "xs_group": K,
        }

    def _info(self, stats: np.ndarray, tod_start: int, agg: int, n_tokens: int) -> np.ndarray:
        """The information token per view, as ``augmentations.encode_view_metadata``
        lays it out: (sign(mu) log1p|mu|, log sigma) per group, then the window."""
        K = stats.shape[0]
        parts = []
        if self.info_norm_stats:
            mean, scale = stats[:, :, 0], stats[:, :, 1]
            enc = np.empty((K, 2 * stats.shape[1]), dtype=np.float64)
            enc[:, 0::2] = np.sign(mean) * np.log1p(np.abs(mean))
            enc[:, 1::2] = np.log(scale + INFO_EPS)
            parts.append(enc)
        if self.info_window:
            end = tod_start + n_tokens * agg
            row = np.array([tod_start / INFO_SESSION, end / INFO_SESSION,
                            np.log(max(float(agg), INFO_EPS))], dtype=np.float64)
            parts.append(np.broadcast_to(row, (K, N_WINDOW_INFO)))
        return np.concatenate(parts, axis=1).astype(np.float32)


def _n_tickers(day_path) -> int:
    import json

    return len(json.loads((day_path / "meta.json").read_text())["tickers"])
