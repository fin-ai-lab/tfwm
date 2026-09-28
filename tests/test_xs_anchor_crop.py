"""Anchor-snapped crops and the cross-sectional z-score target.

The z-score only exists on the 5-minute anchor lattice, so these two pieces
are one mechanism: a view whose last row misses the lattice has no (mu, sigma)
to divide by and yields no label at all.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from market_jepa.augmentations import _random_resized_crop_numpy
from stable_finance.dataset.anchors import ANCHOR_STEP, SESSION_LEN
from stable_finance.dataset.outcomes import (
    ANCHOR_TARGET_TYPES as TARGET_TYPES,
    anchor_indices,
    anchor_targets,
)
from stable_finance.dataset.targets import AnchorTargetStats as AnchorStats
from stable_finance.dataset.targets import accumulate, finalize

SEQ = 64
N = SESSION_LEN


def _features(n=N, seed=0):
    """A plausible (n, 9) 1 Hz session: random-walk mid with a live book."""
    rng = np.random.RandomState(seed)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-5))
    half = 0.01 + np.abs(rng.standard_normal(n)) * 0.001
    f = np.zeros((n, 9))
    f[:, 0] = mid - half          # bid
    f[:, 4] = mid + half          # ask
    f[:, 1] = mid                 # vwap
    f[:, 2] = mid + half          # high
    f[:, 3] = mid - half          # low
    f[:, 5] = f[:, 6] = 100.0     # sizes
    f[:, 7] = rng.poisson(50, n)  # volume
    f[:, 8] = rng.poisson(5, n)   # n
    return f


# ── crop: end snapping ───────────────────────────────────────────────────────


@pytest.mark.parametrize("offset", [0, 1, 137, 299])
def test_end_lands_on_lattice(offset):
    """(end_offset + t_idx) is always a multiple of the grid, for any offset."""
    f = _features()
    rng = np.random.RandomState(7)
    for _ in range(200):
        view, start, agg, window = _random_resized_crop_numpy(
            f, (0.5, 1.0), SEQ, rng, end_grid_sec=ANCHOR_STEP, end_offset=offset,
        )
        if view is None:
            continue
        t_idx = start + window - 1
        assert (offset + t_idx) % ANCHOR_STEP == 0
        assert 0 <= start and t_idx < len(f)
        assert view.shape == (SEQ, 9)


def test_slack_is_respected_and_rejects_rather_than_shrinks():
    """A window too long for the deadline is dropped, not re-scaled.

    Shrinking would pile probability onto the coarsest feasible resolution;
    the sampler must keep the scale it drew or emit nothing.
    """
    f = _features()
    rng = np.random.RandomState(3)
    slack = 7200
    aggs, rejects = [], 0
    for _ in range(400):
        view, start, agg, window = _random_resized_crop_numpy(
            f, (0.5, 1.0), 2048, rng,
            end_grid_sec=ANCHOR_STEP, end_offset=0, end_min_slack=slack,
        )
        if view is None:
            rejects += 1
            continue
        assert start + window - 1 <= len(f) - 1 - slack
        aggs.append(agg)
    assert rejects > 0, "a 2 h deadline must reject the coarse draws"
    # Only scales whose whole window fits before the deadline may appear.
    assert max(aggs) * 2048 <= len(f) - slack


def test_no_grid_is_unchanged():
    """Without end_grid_sec the sampler keeps its original uniform-start draw."""
    f = _features()
    a = _random_resized_crop_numpy(f, (0.5, 1.0), SEQ, np.random.RandomState(1))
    b = _random_resized_crop_numpy(f, (0.5, 1.0), SEQ, np.random.RandomState(1),
                                   end_grid_sec=None, end_offset=99, end_min_slack=99)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


# ── targets: no clamp ────────────────────────────────────────────────────────


def test_targets_are_nan_past_the_close():
    """The forward window is never truncated — it is simply absent."""
    f = _features()
    hs = [300, 7200]
    y = anchor_targets(f, hs, anchor_indices())
    anchors = anchor_indices()
    for hi, h in enumerate(hs):
        valid = np.isfinite(y[:, TARGET_TYPES.index("return"), hi])
        np.testing.assert_array_equal(valid, anchors + h <= len(f) - 1)


def test_return_matches_direct_computation():
    """Forward VWAP windows at both ends, not quote midpoints."""
    from stable_finance.dataset.anchors import RETURN_VWAP_WINDOW
    from stable_finance.dataset.outcomes import forward_vwap

    f = _features()
    y = anchor_targets(f, [900], np.array([3000]))
    w = RETURN_VWAP_WINDOW

    def vwap(a):
        return (f[a:a + w, 1] * f[a:a + w, 7]).sum() / f[a:a + w, 7].sum()

    assert y[0, TARGET_TYPES.index("return"), 0] == pytest.approx(
        vwap(3900) / vwap(3000) - 1
    )
    # And it is NOT the old midpoint ratio, which is the entire point.
    mid = (f[:, 0] + f[:, 4]) / 2
    assert y[0, TARGET_TYPES.index("return"), 0] != pytest.approx(
        mid[3900] / mid[3000] - 1
    )


def test_training_and_eval_targets_agree():
    """compute_pair_targets (train) vs anchor_targets (score), same number.

    A model trained on one definition and scored on the other is not a milder
    version of the change, it is a different experiment -- so the two paths go
    through the same forward_vwap helper and this pins that they still do.
    """
    from stable_finance.dataset import compute_pair_targets

    f = _features()
    for t in (1200, 3000, 6000):
        a = anchor_targets(f, [900], np.array([t]))[0]
        b = compute_pair_targets(f, t, [900], list(TARGET_TYPES),
                                 None, None, None, None)
        for i, name in enumerate(TARGET_TYPES):
            assert a[i, 0] == pytest.approx(b[i], rel=1e-5), name


def test_every_target_is_measured_between_two_forward_windows():
    """No target may subtract a quantity the model has already seen.

    The old definitions each did: the return divided by mid(t), spread_change
    subtracted spread(t), and volatility_change subtracted vol over [t-h, t).
    All three are knowable at t, so a model that had merely read its own input
    could forecast them -- measured at rank IC +0.0279, ~0.18 and ~0.32
    respectively. Rewriting history strictly before t must now leave every
    target untouched.
    """
    a = _features()
    b = a.copy()
    b[:3000] = _features(seed=99)[:3000]        # everything before t
    t = 3000
    ya = anchor_targets(a, [900], np.array([t]))[0]
    yb = anchor_targets(b, [900], np.array([t]))[0]
    for i, name in enumerate(TARGET_TYPES):
        assert ya[i, 0] == pytest.approx(yb[i, 0], rel=1e-9), name


# ── z-score round trip ───────────────────────────────────────────────────────


def _table(tmp_path, n_names=40, hs=(300, 900)):
    """Build a one-date table from a synthetic cross-section of n_names."""
    anchors = anchor_indices()
    shape = (len(anchors), len(TARGET_TYPES), len(hs))
    sums, sqs, cnts = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64)
    ys = []
    for k in range(n_names):
        y = anchor_targets(_features(seed=k), list(hs), anchors)
        ys.append(y)
        accumulate(sums, sqs, cnts, y)
    mu, sigma = finalize(sums, sqs, cnts)
    p = tmp_path / "2023-01.npz"
    np.savez(
        p,
        dates=np.array(["2023-01-03"]),
        anchors=anchors.astype(np.int32),
        types=np.array(TARGET_TYPES),
        horizons=np.array(hs, dtype=np.int32),
        mu=mu[None], sigma=sigma[None], count=cnts[None].astype(np.int32),
    )
    return AnchorStats(p), ys


def test_zscores_of_a_cross_section_are_standardized(tmp_path):
    """Standardizing every member against the shared cell gives mean 0, std 1."""
    stats, ys = _table(tmp_path)
    zs = np.stack([
        stats.zscore(y[6], "2023-01-03", 1800, TARGET_TYPES, [300, 900])
        for y in ys
    ])
    assert np.isfinite(zs).all()
    np.testing.assert_allclose(zs.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(zs.std(axis=0), 1, rtol=1e-4)


def test_thin_cross_sections_are_dropped(tmp_path):
    """Below MIN_NAMES the cell is unusable, so every member scores NaN."""
    stats, ys = _table(tmp_path, n_names=5)
    z = stats.zscore(ys[0][6], "2023-01-03", 1800, TARGET_TYPES, [300, 900])
    assert np.isnan(z).all()


def test_unknown_anchor_or_date_is_nan(tmp_path):
    stats, ys = _table(tmp_path)
    raw = ys[0][6]
    assert np.isnan(stats.zscore(raw, "2023-01-03", 1799, TARGET_TYPES, [300, 900])).all()
    assert np.isnan(stats.zscore(raw, "1999-01-04", 1800, TARGET_TYPES, [300, 900])).all()


_TMP = ["_xs_a", "_xs_b"]


def _write_table(stem):
    """A minimal table carrying both empirical transforms, on disk."""
    import tempfile
    import numpy as np

    from stable_finance.dataset.outcomes import anchor_indices

    A = len(anchor_indices()); T, H, Q = 3, 6, 64
    N = 21 if stem.endswith("a") else 24
    d = pathlib.Path(tempfile.gettempdir()) / f"{stem}.npz"
    rng = np.random.RandomState(abs(hash(stem)) % 2**31)
    np.savez_compressed(
        d,
        dates=np.array([f"2013-04-0{1 if stem.endswith('a') else 2}"]),
        anchors=anchor_indices().astype(np.int32),
        types=np.array(["return", "spread_change", "volatility_change"]),
        horizons=np.array([300, 600, 900, 1800, 3600, 7200], dtype=np.int32),
        mu=rng.randn(1, A, T, H).astype(np.float32),
        sigma=np.abs(rng.randn(1, A, T, H)).astype(np.float32) + 0.1,
        count=np.full((1, A, T, H), N, dtype=np.int32),
        quantiles=np.sort(rng.randn(1, A, T, H, Q).astype(np.float32), axis=-1),
        quantile_levels=np.linspace(0, 1, Q).astype(np.float32),
        sorted_values=np.sort(
            rng.randn(1, A, T, H, N).astype(np.float32), axis=-1,
        ),
    )
    return str(d)


def test_load_months_carries_every_attribute():
    """AnchorStats.load_months bypasses __init__, so it must set ALL state.

    Forgetting the rank target's quantile arrays here made every rank job die
    with AttributeError on its first batch while the single-month path worked
    fine, so the gap only appeared on the cluster.
    """
    import numpy as np

    from stable_finance.dataset.targets import AnchorTargetStats as AnchorStats

    single = AnchorStats.__new__(AnchorStats)
    expected = set(vars(AnchorStats(_write_table(_TMP[0]))))
    merged = AnchorStats.load_months([_write_table(_TMP[0]), _write_table(_TMP[1])])
    missing = expected - set(vars(merged))
    assert not missing, f"load_months dropped attributes: {sorted(missing)}"
    # date axis must grow for every per-date array
    for name in ("mu", "sigma", "count", "quantiles", "sorted_values"):
        arr = getattr(merged, name)
        if arr is not None:
            assert arr.shape[0] == len(merged.dates), f"{name} date axis wrong"
    del single


def test_uniform_score_is_exact_rankdata_over_n_plus_one(tmp_path):
    """One-based ranks, average ties, and open-unit-interval endpoints."""
    values = np.array(
        [-2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
         7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        dtype=np.float32,
    )
    p = tmp_path / "uniform.npz"
    np.savez(
        p,
        dates=np.array(["2023-01-03"]),
        anchors=np.array([300], dtype=np.int32),
        types=np.array(["return"]),
        horizons=np.array([900], dtype=np.int32),
        mu=np.zeros((1, 1, 1, 1), dtype=np.float32),
        sigma=np.ones((1, 1, 1, 1), dtype=np.float32),
        count=np.full((1, 1, 1, 1), len(values), dtype=np.int32),
        sorted_values=values.reshape(1, 1, 1, 1, -1),
    )
    stats = AnchorStats(p)

    def score(v):
        return stats.uniform_score(
            np.array([[v]]), "2023-01-03", 300, ["return"], [900],
        ).item()

    assert score(-2.0) == 1 / 21
    assert score(-1.0) == 2.5 / 21
    assert score(16.0) == 20 / 21
