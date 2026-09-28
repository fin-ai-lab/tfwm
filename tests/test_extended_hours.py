"""Tests for extended-hours session bounds and the absolute resolution band.

Extended hours change the dense grid from a constant 23,400 rows to a
37.8k-57.6k range, which is why resolution has to be pinned in seconds rather
than as a fraction of the session. These tests hold both halves of that.
"""

from __future__ import annotations

import numpy as np
import pytest

from market_jepa.augmentations import _random_resized_crop_numpy, parse_aggregation_range
from stable_finance.dataset import MarketSchedule, timeline_bounds_est

RTH_SECONDS = 23_400  # 09:30-16:00
EXT_SECONDS = 57_600  # 04:00-20:00


# ── session bounds ───────────────────────────────────────────────────────────


def test_default_is_unchanged():
    """Regular hours must stay bit-identical — every existing run depends on it."""
    for date in ("2023-01-03", "2023-06-15", "2023-11-06"):
        assert timeline_bounds_est(date) == timeline_bounds_est(date, extended_hours=False)
        o, c = timeline_bounds_est(date)
        assert c - o == RTH_SECONDS


@pytest.mark.parametrize("date", ["2023-01-03", "2023-06-15", "2022-03-14"])
def test_extended_spans_four_to_eight(date):
    o, c = timeline_bounds_est(date, extended_hours=True)
    assert c - o == EXT_SECONDS


def test_extended_starts_before_and_ends_after_regular(date="2023-06-15"):
    r_o, r_c = timeline_bounds_est(date)
    e_o, e_c = timeline_bounds_est(date, extended_hours=True)
    assert e_o == r_o - 5 * 3600 - 30 * 60  # 04:00 vs 09:30
    assert e_c == r_c + 4 * 3600            # 20:00 vs 16:00


def test_extended_survives_dst(tmp_path):
    """DST days must still be exactly 16h — the bug that ate the opening hour."""
    for date in ("2023-03-12", "2023-11-05", "2023-03-13", "2023-11-06"):
        o, c = timeline_bounds_est(date, extended_hours=True)
        assert c - o == EXT_SECONDS


def test_extended_respects_short_days(tmp_path):
    """A 13:00 half day closes post-market at 17:00, not 20:00."""
    csv = tmp_path / "holidays.csv"
    csv.write_text(
        "date,status,start_time,end_time,name\n"
        "2023-11-24,short day,09:30,13:00,Day After Thanksgiving\n"
    )
    sched = MarketSchedule(str(csv))

    o, c = timeline_bounds_est("2023-11-24", schedule=sched, extended_hours=True)
    assert c - o == 13 * 3600  # 04:00 -> 17:00
    r_o, r_c = timeline_bounds_est("2023-11-24", schedule=sched)
    assert c == r_c + 4 * 3600


def test_extended_still_rejects_closed_days(tmp_path):
    csv = tmp_path / "holidays.csv"
    csv.write_text("date,status,start_time,end_time,name\n2023-12-25,closed,,,Christmas\n")
    sched = MarketSchedule(str(csv))
    with pytest.raises(ValueError, match="closed"):
        timeline_bounds_est("2023-12-25", schedule=sched, extended_hours=True)


# ── absolute resolution band ─────────────────────────────────────────────────


def _feats(n_sec: int, n_feat: int = 9) -> np.ndarray:
    return np.tile(np.arange(n_sec, dtype=np.float64)[:, None], (1, n_feat))


def test_agg_range_none_is_the_old_behaviour():
    """Omitting agg_range must reproduce the fractional path exactly."""
    feats = _feats(RTH_SECONDS)
    a = _random_resized_crop_numpy(feats, (0.5, 1.0), 2048, np.random.RandomState(7))
    b = _random_resized_crop_numpy(
        feats, (0.5, 1.0), 2048, np.random.RandomState(7), agg_range=None,
    )
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


@pytest.mark.parametrize("n_sec", [37_800, 43_198, 50_000, EXT_SECONDS])
def test_agg_range_pins_resolution_regardless_of_session_length(n_sec):
    """The whole point: resolution must not track the grid length."""
    rng = np.random.RandomState(0)
    feats = _feats(n_sec)
    for _ in range(200):
        view, _, agg, window = _random_resized_crop_numpy(
            feats, (0.5, 1.0), 2048, rng, agg_range=(6, 11),
        )
        assert 6 <= agg <= 11
        assert len(view) == 2048
        assert window == agg * 2048


def test_fractional_path_escapes_the_band_on_extended_grids():
    """Documents why agg_range exists — the fractional path cannot hold 6-11."""
    rng = np.random.RandomState(0)
    aggs = set()
    for n_sec in (37_800, EXT_SECONDS):
        feats = _feats(n_sec)
        for _ in range(200):
            _, _, agg, _ = _random_resized_crop_numpy(feats, (0.5, 1.0), 2048, rng)
            aggs.add(agg)
    assert max(aggs) > 11


def test_agg_range_covers_its_whole_span():
    rng = np.random.RandomState(3)
    feats = _feats(EXT_SECONDS)
    seen = {_random_resized_crop_numpy(feats, (0.5, 1.0), 2048, rng, agg_range=(6, 11))[2]
            for _ in range(500)}
    assert seen == {6, 7, 8, 9, 10, 11}


def test_agg_range_falls_back_when_window_exceeds_the_grid():
    """A short grid must degrade gracefully, not return a too-long window."""
    feats = _feats(10_000)
    out = _random_resized_crop_numpy(
        feats, (0.5, 1.0), 2048, np.random.RandomState(0), agg_range=(6, 11),
    )
    view, _, agg, window = out
    assert window <= len(feats)
    assert agg == 10_000 // 2048
    assert len(view) == 2048


def test_local_band_matches_the_paper():
    rng = np.random.RandomState(1)
    feats = _feats(EXT_SECONDS)
    for _ in range(200):
        _, _, agg, _ = _random_resized_crop_numpy(
            feats, (0.05, 0.5), 512, rng, agg_range=(2, 23),
        )
        assert 2 <= agg <= 23


# ── config parsing ───────────────────────────────────────────────────────────


def test_parse_agg_range():
    assert parse_aggregation_range(None) is None
    assert parse_aggregation_range([6, 11]) == (6, 11)
    assert parse_aggregation_range((6, 6)) == (6, 6)


@pytest.mark.parametrize("bad", [[11, 6], [0, 5], [-1, 3]])
def test_parse_agg_range_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="aggregation range"):
        parse_aggregation_range(bad)
