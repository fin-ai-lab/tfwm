"""The daystore loader emits the cross_stock branch's cells, view for view.

One synthetic month is written twice -- as a shuffled dense MDS month and as
the day-major store built from it -- and the same cell (day, anchor,
resolution, K tickers) is built through ``StreamingMarketDataset``'s per-view
helpers and through ``DayStoreCellDataset``. Views, information token and raw
targets must agree; the dict must collate and train unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("streaming")

from stable_finance.dataset import compute_pair_targets
from stable_finance.dataset.calendar import standard_open_est, timeline_bounds_est
from stable_finance.dataset.daystore import convert_month
from stable_finance.dataset.write_mds import DENSE_MDS_COLUMNS

from market_jepa.training.cell_dataset import DayStoreCellDataset
from market_jepa.training.utils import collate_bucketed, supervised_cell_view

DATES = ("2023-01-03", "2023-01-04", "2023-01-05")
TICKERS = tuple(f"T{i:02d}" for i in range(8))
ROWS = 23_400
ONE_GLOBAL = {"global_scale_range": [0.5, 1.0], "global_seq_len": 512, "global_agg_range": None}


def _session(rng, n_rows, seed_price):
    mid = seed_price * np.exp(rng.normal(0, 0.005, n_rows).cumsum())
    f = np.empty((n_rows, 9), dtype=np.float32)
    f[:, 0], f[:, 4] = mid - 0.01, mid + 0.01
    vol = rng.poisson(20, n_rows) * (rng.random(n_rows) < 0.5)
    f[:, 7], f[:, 8] = vol, np.minimum(vol, 2)
    idx = np.arange(n_rows); src = np.where(vol > 0, idx, 0); np.maximum.accumulate(src, out=src)
    f[:, 1] = np.where(src > 0, mid[src], mid)
    f[:, 2], f[:, 3] = mid + 0.02, mid - 0.02
    f[:, 5], f[:, 6] = rng.poisson(100, n_rows) + 1, rng.poisson(100, n_rows) + 1
    return f


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    from streaming import MDSWriter

    root = tmp_path_factory.mktemp("cells")
    mds = root / "mds" / "2023" / "01"
    rng = np.random.default_rng(11)
    records = []
    for d, date in enumerate(DATES):
        ts_open, _ = timeline_bounds_est(date)
        for t, ticker in enumerate(TICKERS):
            first = 4_000 if t == 3 else 0
            records.append({"ticker": ticker, "date": date, "grid_start": ts_open + first,
                            "features": _session(rng, ROWS - first, 30.0 + 5 * t + d)})
    with MDSWriter(out=str(mds), columns=DENSE_MDS_COLUMNS, compression="zstd",
                   size_limit=1 << 22) as w:
        for i in rng.permutation(len(records)):
            w.write(records[i])
    days = root / "days"
    convert_month(mds, days, None, verify_every=1, log=lambda m: None)
    return {"mds": mds, "days": days}


def _mds_dataset(stores, k):
    from streaming import Stream

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    cell = supervised_cell_view(ONE_GLOBAL, k)
    cell.update({"end_grid_sec": 300, "end_min_slack_sec": 960})
    return StreamingMarketDataset(
        augmentations=[cell], date_start="2023-01-01", date_end="2023-01-31", seed=3,
        targets={"horizons": [900], "types": ["return"]},
        streams=[Stream(local=str(stores["mds"]))], shuffle=False, batch_size=2,
        info_norm_stats=True, info_window=True,
        allow_unsafe_types=True,
    )


def _days_dataset(stores, k, **kw):
    return DayStoreCellDataset(
        daystore_dir=stores["days"], date_start="2023-01-01", date_end="2023-01-31",
        cell=supervised_cell_view(ONE_GLOBAL, k),
        targets={"horizons": [900], "types": ["return"]}, seed=3, **kw)


def test_the_epoch_is_one_cell_per_ticker_day(stores):
    ds = _days_dataset(stores, 4)
    assert len(ds) == len(DATES) * len(TICKERS) and ds.n_days == len(DATES)
    assert ds.n_features == 9 + 8 + 3 and ds.target_names == ["return_900"]


def test_cells_match_the_streaming_dataset_view_for_view(stores):
    k = 4
    days, mds = _days_dataset(stores, k), _mds_dataset(stores, k)
    checked = 0
    for slot in range(0, len(days), 5):
        item, day, idx, draw = days.cell_at(slot, epoch=1, with_draw=True)
        if item is None:
            continue
        std_open = standard_open_est(day.date)
        for j, i in enumerate(idx):
            rec = day.record(int(i))
            sec, feat = mds._preprocess_to_numpy(rec)
            tod_offset = int(sec[0]) - std_open
            p_start = (day.open_tod + draw.start_row) - tod_offset
            assert p_start >= 0
            view = mds._build_aligned_view(feat, p_start, draw.window, draw.aggregation,
                                           day.date, 0, tod_start_sec=tod_offset + p_start)
            torch.testing.assert_close(item["views"][j], view, rtol=1e-5, atol=1e-5)
            raw = compute_pair_targets(feat, p_start + draw.window - 1, [900], ["return"])
            np.testing.assert_allclose(item["target_metadata"]["raw"][j].numpy(), raw,
                                       rtol=1e-5, equal_nan=True)
        assert item["targets"].shape == (k, 1) and item["xs_group"] == k
        assert item["lengths"].tolist() == [ONE_GLOBAL["global_seq_len"]] * k
        checked += 1
    assert checked >= 3


def test_items_are_reproducible_and_change_across_epochs(stores):
    a, b = _days_dataset(stores, 3), _days_dataset(stores, 3)
    x, y = a.cell_at(7, epoch=2), b.cell_at(7, epoch=2)
    assert x["date"] == y["date"] and x["ticker"] == y["ticker"]
    for u, v in zip(x["views"], y["views"]):
        assert torch.equal(u, v)
    z = a.cell_at(7, epoch=3)
    assert not all(torch.equal(u, v) for u, v in zip(x["views"], z["views"]))


def test_worker_streams_partition_the_epoch_and_collate(stores):
    ds = _days_dataset(stores, 3)
    items = []
    for item in ds:
        items.append(item[0])
        if len(items) == 6:
            break
    batch = collate_bucketed(items)
    bucket = batch["buckets"][0]
    assert len(bucket["views"]) == 3 and bucket["views"][0].shape == (6, 20, 512)
    assert bucket["targets"].shape == (6, 3, 1)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2,
                                         collate_fn=collate_bucketed)
    seen = []
    for i, batch in enumerate(loader):
        seen.append(batch["buckets"][0]["targets"].shape[0])
        if i == 3:
            break
    assert seen == [4, 4, 4, 4]


def test_the_cell_trains_the_supervised_head(stores):
    from market_jepa.modeling.modes.supervised import SupervisedModel

    class _Backbone(torch.nn.Module):
        d_embedding = 4

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(20, 4)

        def forward(self, x, lengths=None):
            return self.proj(x.mean(dim=-1))

    # Eight names is under MIN_NAMES, so the cross-sectional transforms are
    # NaN on this month; the raw label ranks.
    ds = _days_dataset(stores, 4, xs_target="raw")
    batch = collate_bucketed([ds.cell_at(s, epoch=0) for s in (0, 9, 15)])
    m = SupervisedModel(_Backbone(), task="return_900", loss_fn="pairwise")
    m.target_col_idx = 0
    out = m.training_step(batch, torch.device("cpu"))
    assert out is not None and out["metrics"]["train/cells_per_step"] == 3


def test_both_datasets_spell_the_info_width_the_same_way(stores):
    """The attribute pretrain.py reads must exist on BOTH dataset classes.

    THIS IS THE BUG THIS TEST EXISTS FOR. pretrain.py takes the backbone's
    n_info_channels from the dataset -- only the dataset knows it -- and read it
    as getattr(full_dataset, "_n_info_features", 0). StreamingMarketDataset has
    that attribute; DayStoreCellDataset called the same quantity `_n_info`. So
    every dataset.backend=days run built n_info_channels=0 against a 20-column
    view: the 11 information columns went through the patch embedding as if they
    were eleven more time series, no info_proj was constructed, and nothing
    raised, because the WIDTH was right and only the routing was wrong.

    Checking the views agree element by element (above) cannot catch it -- the
    views DO agree. What differed was what the backbone was told they meant. So
    this asserts the surface, not the contents.
    """
    days, mds = _days_dataset(stores, 4), _mds_dataset(stores, 4)
    for ds in (days, mds):
        assert hasattr(ds, "_n_info_features"), (
            f"{type(ds).__name__} must expose _n_info_features: pretrain.py "
            "reads the backbone's n_info_channels off it by that name."
        )
        # It is the part of n_features that is NOT a time series, so the two
        # numbers have to be consistent or the split is meaningless.
        assert ds.n_features - len(ds.feature_columns) == ds._n_info_features
    assert days._n_info_features == mds._n_info_features == 8 + 3


def test_pretrain_refuses_a_dataset_that_cannot_state_its_info_width(stores):
    """No silent zero-default on the n_info_channels read.

    The getattr default was the mechanism above: a third dataset class, or a
    rename, resurrects the identical failure and again says nothing. There is no
    safe default for this number, so the read raises instead of guessing.
    """
    import inspect

    from market_jepa.training import pretrain

    src = inspect.getsource(pretrain.train)
    assert 'getattr(full_dataset, "_n_info_features", 0)' not in src
    assert 'hasattr(full_dataset, "_n_info_features")' in src


def test_skipped_slots_are_counted_not_silently_dropped(stores, caplog):
    """A slot that yields no cell must leave a trace.

    ``__iter__`` drops a slot whose day could not produce ``n_stocks``
    feasible tickers in MAX_DRAW_ATTEMPTS draws. That makes the epoch quietly
    smaller than ``epoch_size``; without a counter the shortfall is invisible,
    which is the shape of every pair-starvation bug this dataset has had.
    """
    ok = _days_dataset(stores, 4)
    for slot in range(len(ok)):
        assert ok.cell_at(slot) is not None
    assert ok._n_skipped == 0

    # More stocks than the day has tickers: nothing can ever be drawn.
    impossible = _days_dataset(stores, len(TICKERS) + 1)
    with caplog.at_level("WARNING"):
        for slot in range(len(impossible)):
            assert impossible.cell_at(slot) is None
    assert impossible._n_skipped == len(impossible)
    assert "fewer than n_stocks" in caplog.text
