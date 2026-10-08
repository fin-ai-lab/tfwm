"""Tests for stable-finance targets and their market-jepa integration."""

import numpy as np
import pytest
import torch

from stable_finance.dataset import get_target_names, compute_pair_targets


# -----------------------------------------------------------------------
# Helper to create synthetic focal features (N, 9)
# -----------------------------------------------------------------------

def _make_features(n_rows, bid=100.0, ask=100.10):
    """Create (n_rows, 9) features with constant bid/ask and zeros elsewhere."""
    features = np.zeros((n_rows, 9), dtype=np.float64)
    features[:, 0] = bid       # bid_price
    features[:, 4] = ask       # ask_price
    features[:, 1] = (bid + ask) / 2  # vwap_all
    # VOLUME IS NOT DECORATION HERE. The return is measured between two
    # forward VWAP windows and the window is volume-weighted, so a fixture
    # with zero volume has no traded price and every return is NaN.
    features[:, 7] = 100.0     # volume
    return features


# =======================================================================
# Unit tests for get_target_names
# =======================================================================


class TestGetTargetNames:

    def test_standard_only(self):
        names = get_target_names([300, 600, 900], ["return", "spread_change", "volatility_change"])
        assert len(names) == 9
        assert names == [
            "return_300", "return_600", "return_900",
            "spread_change_300", "spread_change_600", "spread_change_900",
            "volatility_change_300", "volatility_change_600", "volatility_change_900",
        ]

    def test_with_rf(self):
        names = get_target_names(
            [300, 600, 900],
            ["return", "spread_change", "volatility_change"],
            rf_tickers=["IWM", "SPY"],
        )
        assert len(names) == 15  # 9 standard + 6 risk-adjusted
        # Check RF names
        assert "return_adj__IWM__300" in names
        assert "return_adj__IWM__600" in names
        assert "return_adj__IWM__900" in names
        assert "return_adj__SPY__300" in names
        assert "return_adj__SPY__600" in names
        assert "return_adj__SPY__900" in names
        # RF names come after standard
        assert names.index("return_adj__IWM__300") > names.index("volatility_change_900")

    def test_single_type_single_horizon(self):
        names = get_target_names([300], ["return"])
        assert names == ["return_300"]

    def test_empty_rf_tickers_list(self):
        names = get_target_names([300], ["return"], rf_tickers=[])
        assert names == ["return_300"]

    def test_none_rf_tickers(self):
        names = get_target_names([300], ["return"], rf_tickers=None)
        assert names == ["return_300"]


# =======================================================================
# Unit tests for compute_pair_targets
# =======================================================================


class TestComputePairTargets:

    def _ret(self, features, t_idx=500, h=300):
        return compute_pair_targets(
            focal_features=features, t_idx=t_idx, horizons=[h],
            types=["return"], rf_data=None, rf_price_mode=None,
            date_str=None, rf_t_idx=None,
        )

    def test_basic_return(self):
        """Ratio of two FORWARD vwap windows, not two quote midpoints."""
        features = _make_features(1000)
        # The whole forward window at t+300, not just the instant.
        features[800:860, 1] = 101.05

        result = self._ret(features)

        assert result.dtype == np.float32
        assert len(result) == 1
        expected = (101.05 / 100.05) - 1.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_return_looks_forward_not_backward(self):
        """THE PROPERTY THE WHOLE CHANGE EXISTS FOR.

        A trailing base would put the already-realized [t-w, t) move inside
        the target, so anything that knows the price at t could read it back
        out -- measured at rank IC -0.0407, worse than the midpoint target it
        replaced. Rewriting history strictly before t must leave the target
        untouched.
        """
        a = _make_features(1000)
        b = a.copy()
        b[:500, 1] = 42.0          # everything strictly before t
        assert self._ret(a)[0] == pytest.approx(self._ret(b)[0])

        # ... and the base window [t, t+60) MUST matter.
        c = a.copy()
        c[500:560, 1] = 50.0
        assert self._ret(c)[0] != pytest.approx(self._ret(a)[0])

    def test_return_is_volume_weighted(self):
        """Tradeless seconds carry volume 0 and must not enter the average.

        The 1 Hz grid forward-fills vwap through seconds with no trade, so a
        plain mean would average stale prices. Weighting by volume drops them
        on its own.
        """
        f = _make_features(1000)
        f[500:560, 1] = 999.0      # stale ffilled price ...
        f[500:560, 7] = 0.0        # ... with no volume behind it
        f[505, 1] = 100.05
        f[505, 7] = 10.0           # the one real print
        assert self._ret(f)[0] == pytest.approx(0.0, abs=1e-6)

    def test_return_is_nan_without_room_for_the_forward_window(self):
        f = _make_features(1000)
        assert np.isnan(self._ret(f, t_idx=980)[0])

    def test_spread_change(self):
        """Mean spread over two FORWARD windows, not two instants."""
        features = _make_features(1000, bid=100.0, ask=100.10)   # spread 0.10
        features[800:860, 4] = 100.20                            # widen at t+300
        result = compute_pair_targets(
            focal_features=features, t_idx=500, horizons=[300],
            types=["spread_change"], rf_data=None, rf_price_mode=None,
            date_str=None, rf_t_idx=None,
        )
        assert len(result) == 1
        np.testing.assert_allclose(result[0], 0.10, atol=1e-6)

    def test_clamp_to_close_now_yields_nan(self):
        """The close clamp no longer manufactures a value, and should not.

        fut_idx is still min(t+h, N-1), but the forward VWAP window then needs
        [N-1, N-1+w) and runs off the end, so the target is NaN instead of a
        return measured against a horizon that was silently shortened. This is
        the no-clamp policy the eval cross-sections already applied as a row
        filter; the sampler's end_min_slack_sec keeps training draws inside the
        region where it does not bite.
        """
        features = _make_features(1000)
        result = compute_pair_targets(
            focal_features=features,
            t_idx=900,
            horizons=[300],  # 900 + 300 = 1200 > 999
            types=["return"],
            rf_data=None,
            rf_price_mode=None,
            date_str=None,
            rf_t_idx=None,
        )
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_rf_vwap_mode(self):
        features = _make_features(1000)
        # Focal: mid_t = 100.05, future mid at t+300 stays 100.05 → gross return = 1.0

        rf_arr = np.zeros((1, 23400, 9), dtype=np.float64)
        rf_arr[0, 500, 1] = 400.0    # vwap at rf_t_idx=500
        rf_arr[0, 800, 1] = 404.0    # vwap at rf_t_idx+300=800

        rf_data = {
            "SPY": {
                "features": rf_arr,
                "date_to_idx": {"2023-01-03": 0},
            }
        }

        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300],
            types=["return"],
            rf_data=rf_data,
            rf_price_mode="vwap",
            date_str="2023-01-03",
            rf_t_idx=500,
        )

        # result: [return_300, return_adj__SPY__300]
        assert len(result) == 2
        # Focal return ≈ 0 (constant price)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)
        # Risk-adjusted: focal_gross - rf_gross = 1.0 - (404/400) = -0.01
        np.testing.assert_allclose(result[1], 1.0 - 404.0 / 400.0, rtol=1e-5)

    def test_rf_mid_mode(self):
        features = _make_features(1000)

        rf_arr = np.zeros((1, 23400, 9), dtype=np.float64)
        rf_arr[0, 500, 0] = 399.0    # bid at t
        rf_arr[0, 500, 4] = 401.0    # ask at t  → mid = 400.0
        rf_arr[0, 800, 0] = 403.0    # bid at t+300
        rf_arr[0, 800, 4] = 405.0    # ask at t+300  → mid = 404.0

        rf_data = {
            "SPY": {
                "features": rf_arr,
                "date_to_idx": {"2023-01-03": 0},
            }
        }

        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300],
            types=["return"],
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-03",
            rf_t_idx=500,
        )

        assert len(result) == 2
        # Risk-adjusted: 1.0 - (404/400)
        np.testing.assert_allclose(result[1], 1.0 - 404.0 / 400.0, rtol=1e-5)

    def test_rf_span_mismatch_is_nan(self):
        # Extended-hours shape: the focal grid clamps the forward window while
        # the RF grid does not (or vice versa) — the two legs would measure
        # different horizons, so the adjusted target must be NaN, not a
        # mostly-unadjusted number.
        features = _make_features(1000)
        rf_arr = np.zeros((1, 23400, 9), dtype=np.float64)
        rf_arr[0, 900, 0] = 399.0
        rf_arr[0, 900, 4] = 401.0
        rf_arr[0, 1200, 0] = 403.0
        rf_arr[0, 1200, 4] = 405.0
        rf_data = {"SPY": {"features": rf_arr, "date_to_idx": {"2023-01-03": 0}}}

        result = compute_pair_targets(
            focal_features=features,
            t_idx=900,               # focal span = min(1200, 999) - 900 = 99
            horizons=[300],          # rf span    = 1200 - 900          = 300
            types=["return"],
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-03",
            rf_t_idx=900,
        )
        assert len(result) == 2
        # The plain return used to keep the clamped value here. It no longer
        # does: the forward VWAP window has no room past the clamp, so the row
        # has no target at all. The SUBJECT of this test is unchanged -- the
        # adjusted leg must refuse a span mismatch rather than return a
        # mostly-unadjusted number.
        assert np.isnan(result[0])
        assert np.isnan(result[1])      # adjusted leg refuses the mismatch

    def test_rf_symmetric_clamp_still_computed(self):
        # When BOTH legs truncate identically (regular-hours close), the
        # adjusted target is still well-defined and must not be NaN'd.
        features = _make_features(1000)
        rf_arr = np.zeros((1, 1000, 9), dtype=np.float64)
        rf_arr[0, 900, 0] = 399.0
        rf_arr[0, 900, 4] = 401.0
        rf_arr[0, 999, 0] = 403.0
        rf_arr[0, 999, 4] = 405.0
        rf_data = {"SPY": {"features": rf_arr, "date_to_idx": {"2023-01-03": 0}}}

        result = compute_pair_targets(
            focal_features=features,
            t_idx=900,               # both spans clamp to 99
            horizons=[300],
            types=["return"],
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-03",
            rf_t_idx=900,
        )
        assert len(result) == 2
        assert np.isfinite(result[1])

    def test_rf_missing_date(self):
        features = _make_features(1000)

        rf_data = {
            "SPY": {
                "features": np.zeros((1, 23400, 9), dtype=np.float64),
                "date_to_idx": {"2023-01-03": 0},  # Only has Jan 3
            }
        }

        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300],
            types=["return"],
            rf_data=rf_data,
            rf_price_mode="vwap",
            date_str="2023-01-04",  # Missing date
            rf_t_idx=500,
        )

        # return_300 (standard) + return_adj__SPY__300 (NaN due to missing date)
        assert len(result) == 2
        assert not np.isnan(result[0])  # Standard return still works
        assert np.isnan(result[1])      # RF target is NaN

    def test_rf_none(self):
        features = _make_features(1000)

        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300, 600, 900],
            types=["return"],
            rf_data=None,
            rf_price_mode=None,
            date_str=None,
            rf_t_idx=None,
        )

        # No RF targets: just 3 returns
        assert len(result) == 3

    def test_multiple_types_and_horizons(self):
        features = _make_features(2000)
        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300, 600, 900],
            types=["return", "spread_change", "volatility_change"],
            rf_data=None,
            rf_price_mode=None,
            date_str=None,
            rf_t_idx=None,
        )

        assert result.dtype == np.float32
        assert len(result) == 9

    def test_volatility_change(self):
        """RV over [t+h, t+h+w) minus RV over [t, t+w) -- both FORWARD."""
        from stable_finance.dataset.anchors import RETURN_VWAP_WINDOW as W
        from stable_finance.dataset.outcomes import window_realized_volatility

        rng = np.random.RandomState(42)
        features = _make_features(2000)
        px = 100.0 + np.cumsum(rng.randn(2000) * 0.01)
        features[:, 0] = px - 0.05
        features[:, 4] = px + 0.05

        result = compute_pair_targets(
            focal_features=features, t_idx=500, horizons=[300],
            types=["volatility_change"], rf_data=None, rf_price_mode=None,
            date_str=None, rf_t_idx=None,
        )
        expected = window_realized_volatility(
            features, 800, W
        ) - window_realized_volatility(features, 500, W)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_volatility_change_ignores_history(self):
        """THE BACKWARD LEG IS GONE, and that is the point of the change.

        It ran over [t-h, t), entirely inside what the model can see, so
        -bwd_vol forecast the target at rank IC ~0.32 on the view with no
        market model at all. Rewriting everything before t must now change
        nothing, and t_idx=0 -- which used to be NaN for want of history -- is
        now a perfectly good sample.
        """
        rng = np.random.RandomState(7)
        a = _make_features(1000)
        px = 100.0 + np.cumsum(rng.randn(1000) * 0.01)
        a[:, 0] = px - 0.05
        a[:, 4] = px + 0.05
        b = a.copy()
        b[:500, 0] += 5.0          # violent, entirely historical
        b[:500, 4] += 5.0
        kw = dict(horizons=[300], types=["volatility_change"], rf_data=None,
                  rf_price_mode=None, date_str=None, rf_t_idx=None)
        ra = compute_pair_targets(focal_features=a, t_idx=500, **kw)[0]
        rb = compute_pair_targets(focal_features=b, t_idx=500, **kw)[0]
        np.testing.assert_allclose(ra, rb, rtol=1e-9)

        # t_idx=0 has no history at all and is now usable.
        assert np.isfinite(compute_pair_targets(focal_features=a, t_idx=0, **kw)[0])


# =======================================================================
# Risk-factor-adjusted volatility change / spread change
# =======================================================================


def _make_rf_data(ticker="SPY", n_days=1, bid=400.0, ask=400.05, date="2023-01-03"):
    """Dense RF arrays with constant positive quotes everywhere."""
    rf_arr = np.zeros((n_days, 23400, 9), dtype=np.float64)
    rf_arr[:, :, 0] = bid
    rf_arr[:, :, 4] = ask
    rf_arr[:, :, 1] = (bid + ask) / 2
    return {ticker: {"features": rf_arr, "date_to_idx": {date: 0}}}


class TestRfAdjustedTargets:

    def test_adjusted_nan_on_missing_date(self):
        features = _make_features(1000)
        rf_data = _make_rf_data(date="2023-01-03")

        result = compute_pair_targets(
            focal_features=features,
            t_idx=500,
            horizons=[300],
            types=["return", "spread_change", "volatility_change"],
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-04",  # missing from RF archive
            rf_t_idx=500,
        )

        names = get_target_names(
            [300], ["return", "spread_change", "volatility_change"], ["SPY"]
        )
        assert len(result) == len(names)
        for name in names:
            if "_adj__" in name:
                assert np.isnan(result[names.index(name)]), name
        assert not np.isnan(result[names.index("return_300")])

    @pytest.mark.parametrize("rf_t_idx", [-100, 23400, 30000])
    def test_adjusted_nan_outside_rf_session(self, rf_t_idx):
        """Extended-hours t outside the RF grid → all RF targets NaN.

        A negative rf_t_idx previously wrapped around and silently indexed
        from the end of the RF day; past-close indices raised IndexError.
        """
        features = _make_features(40000)
        rf_data = _make_rf_data()

        result = compute_pair_targets(
            focal_features=features,
            t_idx=20000,
            horizons=[300],
            types=["return", "spread_change", "volatility_change"],
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-03",
            rf_t_idx=rf_t_idx,
        )

        names = get_target_names(
            [300], ["return", "spread_change", "volatility_change"], ["SPY"]
        )
        assert len(result) == len(names)
        for name in names:
            if "_adj__" in name:
                assert np.isnan(result[names.index(name)]), name
            else:
                assert not np.isnan(result[names.index(name)]), name

    def test_layout_matches_names_multi_ticker(self):
        """Value count and order line up with get_target_names for 2 tickers."""
        features = _make_features(5000)
        rf_data = {
            **_make_rf_data("IWM", bid=190.0, ask=190.02),
            **_make_rf_data("SPY", bid=400.0, ask=400.05),
        }

        horizons = [300, 600, 900]
        types = ["return", "spread_change", "volatility_change"]
        result = compute_pair_targets(
            focal_features=features,
            t_idx=2000,
            horizons=horizons,
            types=types,
            rf_data=rf_data,
            rf_price_mode="mid",
            date_str="2023-01-03",
            rf_t_idx=2000,
        )
        names = get_target_names(horizons, types, ["IWM", "SPY"])
        assert len(result) == len(names)


# =======================================================================
# Task registry wiring
# =======================================================================


class TestTaskRegistry:

    def test_base_regression_tasks_registered(self):
        """Every (type, horizon) pair is registered exactly once — there is no
        k-suffix any more, since every task is a z-score regression."""
        from market_jepa.eval.tasks import (
            DAY_HORIZON, HORIZONS, TARGET_TYPES, TASK_REGISTRY,
        )

        # HORIZONS is the REPORTED sweep and the probe's column set;
        # DAY_HORIZON is registered as a task but deliberately kept out of it,
        # so the registry carries one extra row per target type.
        assert DAY_HORIZON not in HORIZONS
        assert len(TASK_REGISTRY) == len(TARGET_TYPES) * (len(HORIZONS) + 1)
        for t in TARGET_TYPES:
            for h in (*HORIZONS, DAY_HORIZON):
                spec = TASK_REGISTRY[f"{t}_{h}"]
                assert spec.target_type == t
                assert spec.horizon == h
                assert spec.target_col == f"{t}_{h:03d}"


# =======================================================================
# Integration: collate_bucketed with targets
# =======================================================================


class TestCollateBucketedTargets:

    def test_with_targets(self):
        from market_jepa.training.utils import collate_bucketed

        n_targets = 9
        batch = []
        for _ in range(4):
            pair = {
                "views": [
                    torch.randn(9, 20),
                    torch.randn(9, 15),
                ],
                "lengths": torch.tensor([20, 15], dtype=torch.long),
                "bucket_key": 0,
                "targets": torch.randn(n_targets),
            }
            batch.append([pair])

        result = collate_bucketed(batch)
        assert len(result["buckets"]) == 1
        bucket = result["buckets"][0]
        assert "targets" in bucket
        assert bucket["targets"].shape == (4, n_targets)

    def test_without_targets(self):
        from market_jepa.training.utils import collate_bucketed

        batch = []
        for _ in range(4):
            pair = {
                "views": [
                    torch.randn(9, 20),
                    torch.randn(9, 15),
                ],
                "lengths": torch.tensor([20, 15], dtype=torch.long),
                "bucket_key": 0,
            }
            batch.append([pair])

        result = collate_bucketed(batch)
        assert len(result["buckets"]) == 1
        bucket = result["buckets"][0]
        assert "targets" not in bucket

    def test_all_target_representations_are_collated_as_metadata(self):
        from market_jepa.training.utils import collate_bucketed

        names = ("raw", "zscore", "uniform", "rank")
        batch = []
        for row in range(3):
            metadata = {
                name: torch.full((2,), float(row + offset))
                for offset, name in enumerate(names)
            }
            batch.append([{
                "views": [torch.randn(9, 8), torch.randn(9, 8)],
                "lengths": torch.tensor([8, 8]),
                "bucket_key": 0,
                "targets": metadata["uniform"],
                "target_metadata": metadata,
            }])

        bucket = collate_bucketed(batch)["buckets"][0]
        assert tuple(bucket["target_metadata"]) == names
        for name in names:
            assert bucket["target_metadata"][name].shape == (3, 2)
        torch.testing.assert_close(
            bucket["targets"], bucket["target_metadata"]["uniform"]
        )

    def test_multiple_buckets_with_targets(self):
        from market_jepa.training.utils import collate_bucketed

        n_targets = 6
        batch = []
        for bk in [0, 0, 1, 1]:
            pair = {
                "views": [
                    torch.randn(9, 20),
                    torch.randn(9, 15),
                ],
                "lengths": torch.tensor([20, 15], dtype=torch.long),
                "bucket_key": bk,
                "targets": torch.randn(n_targets),
            }
            batch.append([pair])

        result = collate_bucketed(batch)
        assert len(result["buckets"]) == 2
        for bucket in result["buckets"]:
            assert "targets" in bucket
            assert bucket["targets"].shape == (2, n_targets)
