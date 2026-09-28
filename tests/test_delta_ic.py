"""Delta IC: the paired month-clustered difference against the untrained floor.

The reported headline is not a raw IC. Months differ in how rankable they are
by more than methods differ from each other, so a table of raw ICs measured on
different months mostly reads regime. These tests pin the two properties that
make the subtraction worth doing: it is PAIRED within a month, and the month
stays the unit of clustering.
"""

import numpy as np
import pytest

from market_jepa.eval.metrics import delta_ic, pooled_month_ic


def test_delta_is_the_mean_of_paired_differences():
    method = [0.05, 0.08, 0.03, 0.06]
    baseline = [0.02, 0.05, 0.01, 0.03]
    out = delta_ic(method, baseline)
    assert out["mean"] == pytest.approx(np.mean(np.array(method) - np.array(baseline)))
    assert out["n_months"] == 4


def test_pairing_cancels_the_month_effect():
    """The whole point: a shared regime shift must not reach the estimate.

    Both arms are shifted by the same large per-month offset. A paired delta
    is untouched by it; differencing the pooled means would be too, but its
    STANDARD ERROR would absorb the offset's variance and balloon.
    """
    rng = np.random.default_rng(0)
    month_effect = rng.normal(0, 0.05, 12)      # regime, shared by both arms
    edge = 0.01                                  # the real, constant advantage
    baseline = month_effect
    method = month_effect + edge

    out = delta_ic(method, baseline)
    assert out["mean"] == pytest.approx(edge)
    # Zero residual spread once the shared effect cancels.
    assert out["se"] == pytest.approx(0.0, abs=1e-12)

    # Unpaired, the month effect is still in both terms and swamps the edge.
    unpaired_se = np.hypot(
        pooled_month_ic(method)["se"], pooled_month_ic(baseline)["se"],
    )
    assert unpaired_se > 100 * max(out["se"], 1e-12)


def test_month_is_the_clustering_unit():
    """delta_ic must delegate to pooled_month_ic, not re-derive an SE."""
    method = [0.05, 0.08, 0.03, 0.06]
    baseline = [0.02, 0.05, 0.01, 0.03]
    diffs = np.array(method) - np.array(baseline)
    assert delta_ic(method, baseline) == pooled_month_ic(diffs)


def test_a_method_at_the_untrained_floor_scores_zero():
    """The failure mode the baseline exists to catch: structure that a random
    projection already had."""
    ics = [0.04, 0.02, 0.07, 0.03]
    out = delta_ic(ics, ics)
    assert out["mean"] == pytest.approx(0.0)


def test_unpaired_months_are_refused():
    with pytest.raises(ValueError, match="one baseline per method month"):
        delta_ic([0.1, 0.2, 0.3], [0.1, 0.2])


def test_nan_months_drop_out_of_the_pool():
    out = delta_ic([0.05, float("nan"), 0.03], [0.02, 0.01, 0.01])
    assert out["n_months"] == 2
    assert out["mean"] == pytest.approx(np.mean([0.03, 0.02]))


def test_single_month_reports_no_standard_error():
    out = delta_ic([0.05], [0.02])
    assert out["mean"] == pytest.approx(0.03)
    assert np.isnan(out["se"])
