"""Tests for MDS directory structure, stream discovery, and date alignment.

Covers:
- stable_finance.dataset.write_mds: get_period_key, enumerate_periods,
  _period_discovery_range, discover_and_convert_period
  (these lived in scripts/data_prep/convert_to_mosaic.py until it was retired;
  the test has always imported them from write_mds)
- market_jepa/training/streaming_dataset.py: _infer_freq, _validate_date_alignment,
  discover_streams
"""

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from market_jepa.training.streaming_dataset import (
    _infer_freq,
    _validate_date_alignment,
    discover_streams,
)

# ---------------------------------------------------------------------------
# Import the storage writer from stable-finance.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def convert_mod():
    """Return stable-finance's MDS writer module."""
    from stable_finance.dataset import write_mds

    return write_mds


# ===================================================================
# _infer_freq
# ===================================================================

class TestInferFreq:
    def test_monthly_standard(self):
        assert _infer_freq("/data/1Hz_mosaic_mnth") == "mnth"

    def test_quarterly_standard(self):
        assert _infer_freq("/data/1Hz_mosaic_qtr") == "qtr"

    def test_monthly_is_default(self):
        """Anything without 'qtr' in the name defaults to monthly."""
        assert _infer_freq("/data/1Hz_mosaic_custom") == "mnth"
        assert _infer_freq("/data/mosaic") == "mnth"

    def test_qtr_substring_matches(self):
        """'qtr' anywhere in the basename triggers quarterly."""
        assert _infer_freq("/data/mosaic_qtr_v2") == "qtr"
        assert _infer_freq("/data/qtr_mosaic") == "qtr"

    def test_path_object(self):
        assert _infer_freq(Path("/data/1Hz_mosaic_qtr")) == "qtr"
        assert _infer_freq(Path("/data/1Hz_mosaic_mnth")) == "mnth"

    def test_parent_dir_with_qtr_does_not_match(self):
        """Only the final component (basename) is checked."""
        assert _infer_freq("/qtr_data/1Hz_mosaic_mnth") == "mnth"


# ===================================================================
# _validate_date_alignment
# ===================================================================

class TestValidateDateAlignment:
    """Tests for month and quarter alignment validation."""

    # --- Monthly: valid cases ---

    def test_mnth_single_month(self):
        _validate_date_alignment(
            datetime.date(2023, 3, 1), datetime.date(2023, 3, 31), "mnth"
        )

    def test_mnth_multi_month(self):
        _validate_date_alignment(
            datetime.date(2023, 1, 1), datetime.date(2023, 6, 30), "mnth"
        )

    def test_mnth_february_non_leap(self):
        _validate_date_alignment(
            datetime.date(2023, 2, 1), datetime.date(2023, 2, 28), "mnth"
        )

    def test_mnth_february_leap(self):
        _validate_date_alignment(
            datetime.date(2024, 2, 1), datetime.date(2024, 2, 29), "mnth"
        )

    def test_mnth_cross_year(self):
        _validate_date_alignment(
            datetime.date(2022, 11, 1), datetime.date(2023, 2, 28), "mnth"
        )

    # --- Monthly: invalid cases ---

    def test_mnth_start_not_first(self):
        with pytest.raises(ValueError, match="must be the 1st"):
            _validate_date_alignment(
                datetime.date(2023, 3, 2), datetime.date(2023, 3, 31), "mnth"
            )

    def test_mnth_end_not_last_day(self):
        with pytest.raises(ValueError, match="must be the last day"):
            _validate_date_alignment(
                datetime.date(2023, 3, 1), datetime.date(2023, 3, 30), "mnth"
            )

    def test_mnth_end_feb_28_in_leap_year_is_invalid(self):
        """Feb has 29 days in 2024, so 28 is not the last day."""
        with pytest.raises(ValueError, match="must be the last day"):
            _validate_date_alignment(
                datetime.date(2024, 2, 1), datetime.date(2024, 2, 28), "mnth"
            )

    # --- Quarterly: valid cases ---

    def test_qtr_single_quarter(self):
        _validate_date_alignment(
            datetime.date(2023, 1, 1), datetime.date(2023, 3, 31), "qtr"
        )

    def test_qtr_q2(self):
        _validate_date_alignment(
            datetime.date(2023, 4, 1), datetime.date(2023, 6, 30), "qtr"
        )

    def test_qtr_q3(self):
        _validate_date_alignment(
            datetime.date(2023, 7, 1), datetime.date(2023, 9, 30), "qtr"
        )

    def test_qtr_q4(self):
        _validate_date_alignment(
            datetime.date(2023, 10, 1), datetime.date(2023, 12, 31), "qtr"
        )

    def test_qtr_full_year(self):
        _validate_date_alignment(
            datetime.date(2023, 1, 1), datetime.date(2023, 12, 31), "qtr"
        )

    def test_qtr_cross_year(self):
        _validate_date_alignment(
            datetime.date(2022, 10, 1), datetime.date(2023, 3, 31), "qtr"
        )

    # --- Quarterly: invalid cases ---

    def test_qtr_start_mid_quarter(self):
        with pytest.raises(ValueError, match="not quarter-aligned"):
            _validate_date_alignment(
                datetime.date(2023, 2, 1), datetime.date(2023, 3, 31), "qtr"
            )

    def test_qtr_end_mid_quarter(self):
        with pytest.raises(ValueError, match="not quarter-aligned"):
            _validate_date_alignment(
                datetime.date(2023, 1, 1), datetime.date(2023, 2, 28), "qtr"
            )

    def test_qtr_start_month_5(self):
        with pytest.raises(ValueError, match="Nearest valid start month: 04"):
            _validate_date_alignment(
                datetime.date(2023, 5, 1), datetime.date(2023, 6, 30), "qtr"
            )

    def test_qtr_end_month_11(self):
        with pytest.raises(ValueError, match="Nearest valid end month: 12"):
            _validate_date_alignment(
                datetime.date(2023, 10, 1), datetime.date(2023, 11, 30), "qtr"
            )

    def test_qtr_nearest_hint_start_month_8(self):
        """Month 8 is nearest to quarter-start month 7."""
        with pytest.raises(ValueError, match="Nearest valid start month: 07"):
            _validate_date_alignment(
                datetime.date(2023, 8, 1), datetime.date(2023, 9, 30), "qtr"
            )

    def test_qtr_nearest_hint_end_month_1(self):
        """Month 1 end is nearest to quarter-end month 3."""
        with pytest.raises(ValueError, match="Nearest valid end month: 03"):
            _validate_date_alignment(
                datetime.date(2023, 1, 1), datetime.date(2023, 1, 31), "qtr"
            )

    def test_qtr_start_not_first_caught_before_quarter_check(self):
        """Month-alignment is checked before quarter-alignment."""
        with pytest.raises(ValueError, match="must be the 1st"):
            _validate_date_alignment(
                datetime.date(2023, 2, 15), datetime.date(2023, 3, 31), "qtr"
            )


# ===================================================================
# discover_streams
# ===================================================================

def _make_mds_dir(base: Path, year: int, mm: str) -> Path:
    """Create a fake MDS period directory with a minimal index.json."""
    d = base / str(year) / mm
    d.mkdir(parents=True, exist_ok=True)
    # discover_streams only checks is_dir(), no need for real MDS files
    return d


class TestDiscoverStreamsMonthly:
    """discover_streams with monthly mosaic dirs."""

    def test_single_month(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "03")
        streams = discover_streams(mosaic, "2023-03-01", "2023-03-31")
        assert len(streams) == 1
        assert streams[0].local.endswith("03")

    def test_multi_month(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        for mm in ["01", "02", "03"]:
            _make_mds_dir(mosaic, 2023, mm)
        streams = discover_streams(mosaic, "2023-01-01", "2023-03-31")
        assert len(streams) == 3

    def test_missing_middle_month(self, tmp_path):
        """A gap in the data produces fewer streams, not an error."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "01")
        # 02 missing
        _make_mds_dir(mosaic, 2023, "03")
        streams = discover_streams(mosaic, "2023-01-01", "2023-03-31")
        assert len(streams) == 2

    def test_cross_year_boundary(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2022, "11")
        _make_mds_dir(mosaic, 2022, "12")
        _make_mds_dir(mosaic, 2023, "01")
        _make_mds_dir(mosaic, 2023, "02")
        streams = discover_streams(mosaic, "2022-11-01", "2023-02-28")
        assert len(streams) == 4

    def test_extra_dirs_outside_range_ignored(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        for mm in ["01", "02", "03", "04", "05"]:
            _make_mds_dir(mosaic, 2023, mm)
        streams = discover_streams(mosaic, "2023-02-01", "2023-04-30")
        assert len(streams) == 3
        locals_ = [s.local for s in streams]
        assert not any("01" == Path(l).name for l in locals_)
        assert not any("05" == Path(l).name for l in locals_)

    def test_no_dirs_raises(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        mosaic.mkdir()
        with pytest.raises(ValueError, match="No MDS period directories"):
            discover_streams(mosaic, "2023-01-01", "2023-01-31")

    def test_chronological_order(self, tmp_path):
        """Streams are returned in chronological order."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        # Create in reverse order
        for mm in ["06", "03", "01", "12"]:
            yr = 2022 if mm == "12" else 2023
            _make_mds_dir(mosaic, yr, mm)
        streams = discover_streams(mosaic, "2022-12-01", "2023-06-30")
        names = [Path(s.local).name for s in streams]
        assert names == ["12", "01", "03", "06"]

    def test_rejects_non_month_aligned_start(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "03")
        with pytest.raises(ValueError, match="must be the 1st"):
            discover_streams(mosaic, "2023-03-15", "2023-03-31")

    def test_rejects_non_month_aligned_end(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "03")
        with pytest.raises(ValueError, match="must be the last day"):
            discover_streams(mosaic, "2023-03-01", "2023-03-30")


class TestDiscoverStreamsQuarterly:
    """discover_streams with quarterly mosaic dirs."""

    def test_single_quarter(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        _make_mds_dir(mosaic, 2023, "03")
        streams = discover_streams(mosaic, "2023-01-01", "2023-03-31")
        assert len(streams) == 1

    def test_full_year(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        for mm in ["03", "06", "09", "12"]:
            _make_mds_dir(mosaic, 2023, mm)
        streams = discover_streams(mosaic, "2023-01-01", "2023-12-31")
        assert len(streams) == 4

    def test_cross_year(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        _make_mds_dir(mosaic, 2022, "12")
        _make_mds_dir(mosaic, 2023, "03")
        streams = discover_streams(mosaic, "2022-10-01", "2023-03-31")
        assert len(streams) == 2

    def test_ignores_non_quarter_dirs(self, tmp_path):
        """Even if monthly dirs exist, quarterly discovery only looks at 03/06/09/12."""
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        for mm in ["01", "02", "03", "04", "05", "06"]:
            _make_mds_dir(mosaic, 2023, mm)
        streams = discover_streams(mosaic, "2023-01-01", "2023-06-30")
        assert len(streams) == 2
        names = [Path(s.local).name for s in streams]
        assert names == ["03", "06"]

    def test_rejects_non_quarter_aligned_start(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        _make_mds_dir(mosaic, 2023, "03")
        with pytest.raises(ValueError, match="not quarter-aligned"):
            discover_streams(mosaic, "2023-02-01", "2023-03-31")

    def test_rejects_non_quarter_aligned_end(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        _make_mds_dir(mosaic, 2023, "03")
        with pytest.raises(ValueError, match="not quarter-aligned"):
            discover_streams(mosaic, "2023-01-01", "2023-02-28")

    def test_missing_quarter_still_finds_others(self, tmp_path):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        _make_mds_dir(mosaic, 2023, "03")
        # Q2 dir missing
        _make_mds_dir(mosaic, 2023, "09")
        _make_mds_dir(mosaic, 2023, "12")
        streams = discover_streams(mosaic, "2023-01-01", "2023-12-31")
        assert len(streams) == 3


# ===================================================================
# get_period_key (convert_to_mosaic)
# ===================================================================

class TestGetPeriodKey:
    def test_mnth_january(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 1, 15), "mnth") == (2023, "01")

    def test_mnth_december(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 12, 31), "mnth") == (2023, "12")

    def test_mnth_preserves_year(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2022, 6, 1), "mnth") == (2022, "06")

    def test_qtr_q1_start(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 1, 1), "qtr") == (2023, "03")

    def test_qtr_q1_end(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 3, 31), "qtr") == (2023, "03")

    def test_qtr_q2(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 5, 15), "qtr") == (2023, "06")

    def test_qtr_q3(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 7, 1), "qtr") == (2023, "09")

    def test_qtr_q4(self, convert_mod):
        assert convert_mod.get_period_key(datetime.date(2023, 12, 25), "qtr") == (2023, "12")

    def test_qtr_all_months_map_correctly(self, convert_mod):
        """Every month maps to the correct quarter-end month."""
        expected = {
            1: "03", 2: "03", 3: "03",
            4: "06", 5: "06", 6: "06",
            7: "09", 8: "09", 9: "09",
            10: "12", 11: "12", 12: "12",
        }
        for month, exp_mm in expected.items():
            result = convert_mod.get_period_key(datetime.date(2023, month, 15), "qtr")
            assert result == (2023, exp_mm), f"month {month} -> {result}, expected (2023, {exp_mm})"

    def test_unknown_freq_raises(self, convert_mod):
        with pytest.raises(ValueError, match="Unknown cal_freq"):
            convert_mod.get_period_key(datetime.date(2023, 1, 1), "wk")


# ===================================================================
# Partition scanning (via discover_and_convert_period)
# ===================================================================

class TestPartitionScanning:
    """Test partition discovery logic inside discover_and_convert_period."""

    def test_finds_all_partitions(self, tmp_path, convert_mod):
        base = tmp_path / "1Hz"
        _make_raw_data(base, "2023-03-15", n_partitions=3)

        with patch.object(convert_mod, "convert_period", return_value=0) as mock_cp:
            n_days, n_parts, _ = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "03"), "mnth",
            )
        assert n_days == 1
        assert n_parts == 3
        # Verify entries passed to convert_period have correct partition IDs
        entries = mock_cp.call_args[0][2]
        pids = sorted(e[1] for e in entries)
        assert pids == [0, 1, 2]

    def test_partitions_sorted_by_id(self, tmp_path, convert_mod):
        """Partition dirs are iterated in sorted order before shuffling."""
        base = tmp_path / "1Hz"
        _make_raw_data(base, "2023-03-15", n_partitions=10)

        with patch.object(convert_mod, "convert_period", return_value=0):
            n_days, n_parts, _ = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "03"), "mnth",
            )
        assert n_parts == 10


# ===================================================================
# Round-trip: get_period_key ↔ discover_streams consistency
# ===================================================================

class TestConvertDiscoverConsistency:
    """Verify that directories produced by convert match what discover_streams expects."""

    def test_monthly_round_trip(self, tmp_path, convert_mod):
        """Directories named by get_period_key are discoverable by discover_streams."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        dates = [
            datetime.date(2023, 1, 15),
            datetime.date(2023, 2, 10),
            datetime.date(2023, 3, 20),
        ]
        # Simulate what convert_period would create
        for d in dates:
            year, mm = convert_mod.get_period_key(d, "mnth")
            out = mosaic / str(year) / mm
            out.mkdir(parents=True, exist_ok=True)

        streams = discover_streams(mosaic, "2023-01-01", "2023-03-31")
        assert len(streams) == 3

    def test_quarterly_round_trip(self, tmp_path, convert_mod):
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        dates = [
            datetime.date(2023, 1, 5),
            datetime.date(2023, 4, 10),
            datetime.date(2023, 7, 20),
            datetime.date(2023, 10, 1),
        ]
        for d in dates:
            year, mm = convert_mod.get_period_key(d, "qtr")
            out = mosaic / str(year) / mm
            out.mkdir(parents=True, exist_ok=True)

        streams = discover_streams(mosaic, "2023-01-01", "2023-12-31")
        assert len(streams) == 4

    def test_quarterly_multi_month_data_single_dir(self, tmp_path, convert_mod):
        """Three months of trading days all map to one quarterly dir."""
        mosaic = tmp_path / "1Hz_mosaic_qtr"
        all_keys = set()
        for month in [1, 2, 3]:
            key = convert_mod.get_period_key(datetime.date(2023, month, 15), "qtr")
            all_keys.add(key)
            year, mm = key
            (mosaic / str(year) / mm).mkdir(parents=True, exist_ok=True)

        # All three months should map to the same key
        assert len(all_keys) == 1
        assert all_keys == {(2023, "03")}

        streams = discover_streams(mosaic, "2023-01-01", "2023-03-31")
        assert len(streams) == 1


# ===================================================================
# CLI argument validation (convert_to_mosaic)
# ===================================================================

class TestCLIArgs:
    def test_wk_freq_rejected(self, convert_mod):
        """The 'wk' frequency is no longer a valid choice."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cal-freq", choices=["mnth", "qtr"], default="mnth")
        with pytest.raises(SystemExit):
            parser.parse_args(["--cal-freq", "wk"])

    def test_default_freq_is_mnth(self, convert_mod):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cal-freq", choices=["mnth", "qtr"], default="mnth")
        args = parser.parse_args([])
        assert args.cal_freq == "mnth"


# ===================================================================
# enumerate_periods (convert_to_mosaic)
# ===================================================================

class TestEnumeratePeriods:
    def test_single_month(self, convert_mod):
        keys = convert_mod.enumerate_periods("2023-03", "2023-03", "mnth")
        assert keys == [(2023, "03")]

    def test_full_year_monthly(self, convert_mod):
        keys = convert_mod.enumerate_periods("2023-01", "2023-12", "mnth")
        assert len(keys) == 12
        assert keys[0] == (2023, "01")
        assert keys[-1] == (2023, "12")

    def test_full_year_quarterly(self, convert_mod):
        keys = convert_mod.enumerate_periods("2023-01", "2023-12", "qtr")
        assert keys == [(2023, "03"), (2023, "06"), (2023, "09"), (2023, "12")]

    def test_cross_year_monthly(self, convert_mod):
        keys = convert_mod.enumerate_periods("2022-11", "2023-02", "mnth")
        assert keys == [(2022, "11"), (2022, "12"), (2023, "01"), (2023, "02")]

    def test_cross_year_quarterly(self, convert_mod):
        keys = convert_mod.enumerate_periods("2022-10", "2023-03", "qtr")
        assert keys == [(2022, "12"), (2023, "03")]

    def test_no_duplicates_quarterly(self, convert_mod):
        """Three months in the same quarter produce one key, not three."""
        keys = convert_mod.enumerate_periods("2023-01", "2023-03", "qtr")
        assert keys == [(2023, "03")]

    def test_two_years_monthly(self, convert_mod):
        keys = convert_mod.enumerate_periods("2022-01", "2023-12", "mnth")
        assert len(keys) == 24


# ===================================================================
# _period_discovery_range (convert_to_mosaic)
# ===================================================================

class TestPeriodDiscoveryRange:
    def test_monthly(self, convert_mod):
        start, end = convert_mod._period_discovery_range(2023, "03", "mnth")
        assert start == "2023-03"
        assert end == "2023-03"

    def test_quarterly_q1(self, convert_mod):
        start, end = convert_mod._period_discovery_range(2023, "03", "qtr")
        assert start == "2023-01"
        assert end == "2023-03"

    def test_quarterly_q2(self, convert_mod):
        start, end = convert_mod._period_discovery_range(2023, "06", "qtr")
        assert start == "2023-04"
        assert end == "2023-06"

    def test_quarterly_q3(self, convert_mod):
        start, end = convert_mod._period_discovery_range(2023, "09", "qtr")
        assert start == "2023-07"
        assert end == "2023-09"

    def test_quarterly_q4(self, convert_mod):
        start, end = convert_mod._period_discovery_range(2023, "12", "qtr")
        assert start == "2023-10"
        assert end == "2023-12"


# ===================================================================
# discover_and_convert_period (convert_to_mosaic) — integration
# ===================================================================

def _make_raw_data(base_path: Path, date_str: str, n_partitions: int = 1):
    """Create fake raw data dirs matching what discover_trading_days expects.

    Structure: base_path/{year}/{MM}/{date_str}.parquet/partition=N/0.parquet
    """
    d = datetime.date.fromisoformat(date_str)
    day_dir = base_path / str(d.year) / f"{d.month:02d}" / f"{date_str}.parquet"
    for p in range(n_partitions):
        part = day_dir / f"partition={p}"
        part.mkdir(parents=True, exist_ok=True)
        (part / "0.parquet").touch()
    return day_dir


class TestDiscoverAndConvertPeriod:
    """Integration tests for the combined discover+scan+convert pipeline.

    These test discovery and partition scanning only (convert_period will fail
    on fake parquet data, so we mock it).
    """

    def test_discovers_correct_days_monthly(self, tmp_path, convert_mod):
        base = tmp_path / "1Hz"
        _make_raw_data(base, "2023-03-01")
        _make_raw_data(base, "2023-03-15")
        _make_raw_data(base, "2023-03-31")
        # Different month — should NOT be discovered for March
        _make_raw_data(base, "2023-04-01")

        with patch.object(convert_mod, "convert_period", return_value=42) as mock_cp:
            n_days, n_parts, n_obs = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "03"), "mnth",
            )
        assert n_days == 3
        assert n_parts == 3
        assert n_obs == 42
        # Verify convert_period was called with the right period key
        call_args = mock_cp.call_args
        assert call_args[0][1] == (2023, "03")
        # Verify entries have 3 items
        assert len(call_args[0][2]) == 3

    def test_discovers_correct_days_quarterly(self, tmp_path, convert_mod):
        base = tmp_path / "1Hz"
        _make_raw_data(base, "2023-01-15")
        _make_raw_data(base, "2023-02-15")
        _make_raw_data(base, "2023-03-15")
        # Q2 — should NOT be discovered for Q1
        _make_raw_data(base, "2023-04-01")

        with patch.object(convert_mod, "convert_period", return_value=100) as mock_cp:
            n_days, n_parts, n_obs = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "03"), "qtr",
            )
        assert n_days == 3
        assert n_parts == 3
        assert n_obs == 100

    def test_multiple_partitions_counted(self, tmp_path, convert_mod):
        base = tmp_path / "1Hz"
        _make_raw_data(base, "2023-06-15", n_partitions=5)

        with patch.object(convert_mod, "convert_period", return_value=0):
            n_days, n_parts, _ = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "06"), "mnth",
            )
        assert n_days == 1
        assert n_parts == 5

    def test_empty_period_returns_zeros(self, tmp_path, convert_mod):
        """Period with no trading days still works (returns 0s)."""
        base = tmp_path / "1Hz"
        base.mkdir()

        with patch.object(convert_mod, "convert_period", return_value=0):
            n_days, n_parts, n_obs = convert_mod.discover_and_convert_period(
                base, tmp_path / "out", (2023, "07"), "mnth",
            )
        assert n_days == 0
        assert n_parts == 0
        assert n_obs == 0


# ===================================================================
# enumerate_periods ↔ discover_and_convert consistency
# ===================================================================

class TestEnumerateDiscoverConsistency:
    """Verify enumerate_periods produces keys that cover exactly the right months."""

    def test_monthly_keys_match_months(self, convert_mod):
        """Each monthly key's discovery range covers exactly that month."""
        keys = convert_mod.enumerate_periods("2023-01", "2023-12", "mnth")
        for year, mm in keys:
            start, end = convert_mod._period_discovery_range(year, mm, "mnth")
            assert start == end == f"{year}-{mm}"

    def test_quarterly_keys_cover_three_months(self, convert_mod):
        """Each quarterly key's discovery range spans exactly 3 months."""
        keys = convert_mod.enumerate_periods("2023-01", "2023-12", "qtr")
        expected_ranges = [
            ("2023-01", "2023-03"),
            ("2023-04", "2023-06"),
            ("2023-07", "2023-09"),
            ("2023-10", "2023-12"),
        ]
        for key, expected in zip(keys, expected_ranges):
            year, mm = key
            actual = convert_mod._period_discovery_range(year, mm, "qtr")
            assert actual == expected

    def test_full_coverage_no_gaps(self, convert_mod):
        """Monthly enumeration + discovery ranges cover every month in the input range."""
        keys = convert_mod.enumerate_periods("2022-06", "2023-05", "mnth")
        all_months = set()
        for year, mm in keys:
            start, end = convert_mod._period_discovery_range(year, mm, "mnth")
            all_months.add(start)
        # Should have 12 months: 2022-06 through 2023-05
        assert len(all_months) == 12


# ===================================================================
# Edge cases and regression tests
# ===================================================================

class TestEdgeCases:
    def test_discover_streams_string_dates(self, tmp_path):
        """discover_streams accepts string dates (not just date objects)."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "06")
        streams = discover_streams(str(mosaic), "2023-06-01", "2023-06-30")
        assert len(streams) == 1

    def test_discover_streams_date_objects(self, tmp_path):
        """discover_streams also accepts datetime.date objects."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "06")
        streams = discover_streams(
            mosaic,
            datetime.date(2023, 6, 1),
            datetime.date(2023, 6, 30),
        )
        assert len(streams) == 1

    def test_single_month_full_year_scan(self, tmp_path):
        """Scanning a full year with only one month of data works."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "07")
        streams = discover_streams(mosaic, "2023-01-01", "2023-12-31")
        assert len(streams) == 1

    def test_discover_streams_does_not_look_beyond_range(self, tmp_path):
        """Only dirs within [date_start, date_end] months are returned."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        _make_mds_dir(mosaic, 2023, "01")
        _make_mds_dir(mosaic, 2023, "02")
        _make_mds_dir(mosaic, 2023, "03")
        streams = discover_streams(mosaic, "2023-02-01", "2023-02-28")
        assert len(streams) == 1
        assert Path(streams[0].local).name == "02"

    def test_discover_streams_multi_year_range(self, tmp_path):
        """A range spanning 3+ years finds all matching dirs."""
        mosaic = tmp_path / "1Hz_mosaic_mnth"
        for year in [2021, 2022, 2023]:
            _make_mds_dir(mosaic, year, "06")
        streams = discover_streams(mosaic, "2021-01-01", "2023-12-31")
        assert len(streams) == 3

    def test_get_period_key_month_boundary(self, convert_mod):
        """Last day of month stays in that month, first day of next month advances."""
        assert convert_mod.get_period_key(datetime.date(2023, 1, 31), "mnth") == (2023, "01")
        assert convert_mod.get_period_key(datetime.date(2023, 2, 1), "mnth") == (2023, "02")

    def test_get_period_key_quarter_boundary(self, convert_mod):
        """Mar 31 → Q1, Apr 1 → Q2."""
        assert convert_mod.get_period_key(datetime.date(2023, 3, 31), "qtr") == (2023, "03")
        assert convert_mod.get_period_key(datetime.date(2023, 4, 1), "qtr") == (2023, "06")
