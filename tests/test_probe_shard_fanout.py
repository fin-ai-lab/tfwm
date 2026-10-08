"""A probe shard is only reusable at the fan-out it was cut at.

Shard files are named ``<month>_a<anchors>_<shard>.npz``, which says nothing
about how many shards the month was split into. The examples' --smoke embeds
at 4 shards and the full run at 32 in the same cache, so the full run reused
smoke's shards 0-3 (each a quarter month) beside its own 4-31 (each 1/32),
and the released encoder was reduced on a panel that was mostly duplicates.

These pin the two guards: a shard stamped at another fan-out is stale, and the
reducer refuses a panel whose (date, anchor, ticker) rows repeat.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import probe_fit_size as pfs  # noqa: E402

STATS = "xs_anchor_stats_fwdvwap60"


def _shard(path, tickers, **stamp):
    n = len(tickers)
    np.savez(path, X=np.zeros((n, 4), np.float32), z=np.zeros((n, 1), np.float32),
             date=np.array(["2020-01-02"] * n), anchor=np.zeros(n, np.int64),
             ticker=np.array(tickers), target_names=np.array(["return_900"]),
             xs_anchor_stats=STATS, readout="last", **stamp)
    return path


def test_other_fanout_is_stale(tmp_path):
    p = _shard(tmp_path / "2020-01_a8_003.npz", ["A"], num_shards=4)
    assert pfs._shard_is_current(p, STATS, readout="last", num_shards=4)
    assert not pfs._shard_is_current(p, STATS, readout="last", num_shards=32)


def test_unstamped_shard_stays_current(tmp_path):
    # Caches written before the stamp must not all be re-embedded.
    p = _shard(tmp_path / "2020-01_a8_003.npz", ["A"])
    assert pfs._shard_is_current(p, STATS, readout="last", num_shards=32)


def test_reduce_refuses_duplicate_rows(tmp_path):
    a = _shard(tmp_path / "2020-01_a8_000.npz", ["A", "B"])
    b = _shard(tmp_path / "2020-01_a8_001.npz", ["B", "C"])
    with pytest.raises(SystemExit, match="repeat"):
        pfs._load_group([a, b])


def test_reduce_accepts_a_partition(tmp_path):
    a = _shard(tmp_path / "2020-01_a8_000.npz", ["A", "B"])
    b = _shard(tmp_path / "2020-01_a8_001.npz", ["C"])
    assert len(pfs._load_group([a, b])["X"]) == 3
