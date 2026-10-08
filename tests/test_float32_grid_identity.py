"""The float32 dense grid must be bitwise identical to the float64 one.

``convert_to_mosaic`` writes ``features`` as ``ndarray:float32``, so the
float64 grid the training path used to build carried no information the
float32 one does not. ``sparse_to_dense_grid`` only MOVES values — scatter,
forward-fill, zero-fill, trim — so building it narrow is exact; the arithmetic
starts in the aggregators, which pin their accumulators to float64.

These tests are the contract that lets the grid be cached at half the memory:
if any of them fail, a float32 grid is silently changing training targets.
"""

from pathlib import Path

import numpy as np
import pytest

from market_jepa.augmentations import (
    _aggregate_numpy_jittered, _warped_aggregate_numpy,
)
from stable_finance.dataset import MARKET_SCHEMA, sparse_to_dense_grid

FFILL = [0, 1, 2, 3, 4, 5, 6]
ZEROFILL = [7, 8]
N_ROWS = 23_400


def _sparse_day(density=0.6, seed=0, with_nan=True):
    """Sparse (ts, features) for one session, features float32 as MDS stores them."""
    rng = np.random.default_rng(seed)
    open_s = 1_600_000_000
    n = int(N_ROWS * density)
    ts = np.sort(
        rng.choice(np.arange(open_s - 1800, open_s + N_ROWS), n, replace=False)
    ).astype(np.int32)
    f = np.empty((n, 9), dtype=np.float32)
    f[:, 0] = 100.0 + rng.normal(0, 0.01, n).cumsum()          # bid_price
    f[:, 1] = f[:, 0] + 0.025                                  # vwap_all
    f[:, 2] = f[:, 0] + rng.random(n) * 0.1                    # high
    f[:, 3] = f[:, 0] - rng.random(n) * 0.1                    # low
    f[:, 4] = f[:, 0] + 0.05                                   # ask_price
    f[:, 5] = rng.integers(100, 1000, n)                       # bid_size
    f[:, 6] = rng.integers(100, 1000, n)                       # ask_size
    f[:, 7] = rng.integers(0, 500, n)                          # volume
    f[:, 8] = rng.integers(0, 50, n)                           # n
    if with_nan:
        # vwap is NaN wherever nothing traded — the real gap pattern.
        f[rng.random(n) < 0.15, 1] = np.nan
    return ts, f, open_s, open_s + N_ROWS


def _both_grids(**kw):
    ts, f, o, c = _sparse_day(**kw)
    g64 = sparse_to_dense_grid(ts, f, o, c, FFILL, ZEROFILL)
    g32 = sparse_to_dense_grid(ts, f, o, c, FFILL, ZEROFILL, dtype=np.float32)
    assert g64 is not None and g32 is not None
    return g64, g32


def test_the_grid_itself_carries_identical_values():
    (sec64, full64), (sec32, full32) = _both_grids()
    assert full32.dtype == np.float32 and full64.dtype == np.float64
    assert np.array_equal(sec64, sec32)
    assert np.array_equal(full64, full32.astype(np.float64), equal_nan=True)


def test_the_grid_is_half_the_bytes():
    (_, full64), (_, full32) = _both_grids()
    assert full32.nbytes * 2 == full64.nbytes


@pytest.mark.parametrize("scale", [1, 2, 5, 7, 30, 60])
def test_uniform_aggregation_is_bitwise_identical(scale):
    (_, full64), (_, full32) = _both_grids()
    # A window length that is NOT a multiple of scale, to exercise the
    # partial trailing bucket alongside the reshaped ones.
    w = 9_001
    a = _aggregate_numpy_jittered(full64[:w], scale, offset=0)
    b = _aggregate_numpy_jittered(full32[:w], scale, offset=0)
    assert a is not None and b is not None
    assert b.dtype == np.float64
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("offset", [0, 1, 13])
def test_uniform_aggregation_identical_at_every_offset(offset):
    (_, full64), (_, full32) = _both_grids(density=0.35, seed=3)
    a = _aggregate_numpy_jittered(full64[:6_000], 17, offset=offset)
    b = _aggregate_numpy_jittered(full32[:6_000], 17, offset=offset)
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("seq_len", [64, 256])
def test_warped_aggregation_is_bitwise_identical(seq_len):
    (_, full64), (_, full32) = _both_grids(density=0.8, seed=7)
    w = 12_000
    a = _warped_aggregate_numpy(
        full64[:w], seq_len, np.random.RandomState(11), n_knots=8, strength=0.3)
    b = _warped_aggregate_numpy(
        full32[:w], seq_len, np.random.RandomState(11), n_knots=8, strength=0.3)
    assert a is not None and b is not None
    assert b.dtype == np.float64
    assert np.array_equal(a, b, equal_nan=True)


def test_an_all_nan_vwap_column_still_matches():
    """Zero-volume buckets divide by zero on both paths — identically."""
    ts, f, o, c = _sparse_day(density=0.5, seed=5)
    f[:, 1] = np.nan
    f[:, 7] = 0.0
    # trim_leading_nan would reject the whole day (no row is fully valid once
    # vwap is NaN everywhere), so keep the untrimmed grid.
    g64 = sparse_to_dense_grid(
        ts, f, o, c, FFILL, ZEROFILL, trim_leading_nan=False)[1]
    g32 = sparse_to_dense_grid(
        ts, f, o, c, FFILL, ZEROFILL, trim_leading_nan=False, dtype=np.float32)[1]
    a = _aggregate_numpy_jittered(g64[:5_000], 25, offset=0)
    b = _aggregate_numpy_jittered(g32[:5_000], 25, offset=0)
    assert np.array_equal(a, b, equal_nan=True)


def test_float64_stays_the_default():
    """Every existing caller must keep getting float64 without asking."""
    (_, full64), _ = _both_grids()
    assert full64.dtype == np.float64


# ---------------------------------------------------------------------------
# The dense-grid cache built on top of that guarantee.
# ---------------------------------------------------------------------------

from stable_finance.dataset import SessionPreprocessor  # noqa: E402


class _StubDataset:
    """Adapter around the stable-finance session preprocessing boundary."""

    def __init__(self, cache_gb, open_s, close_s, zero_cols=()):
        zero_names = tuple(MARKET_SCHEMA.columns[i] for i in zero_cols)
        self.processor = SessionPreprocessor(
            zero_features=zero_names,
            cache_bytes=int(cache_gb * (1 << 30)),
            bounds=lambda _date: (open_s, close_s),
        )

    def run(self, sample):
        session = self.processor.transform(sample)
        if session is None:
            return None
        return session.timestamps, session.features


def _sample(ticker="AAPL", date="2020-05-01", **kw):
    ts, f, o, c = _sparse_day(**kw)
    return {"ticker": ticker, "date": date, "ts_interval": ts, "features": f}, o, c


def test_the_cache_returns_exactly_what_building_returns():
    s, o, c = _sample()
    cold = _StubDataset(0.0, o, c).run(s)          # cache disabled -> float64 build
    ds = _StubDataset(1.0, o, c)
    warm_miss = ds.run(s)                          # cached path, first touch
    warm_hit = ds.run(s)                           # cached path, served
    for got in (warm_miss, warm_hit):
        assert got[1].dtype == np.float64
        assert np.array_equal(cold[0], got[0])
        assert np.array_equal(cold[1], got[1], equal_nan=True)
    info = ds.processor.cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_a_hit_hands_back_a_fresh_array_that_cannot_poison_the_cache():
    s, o, c = _sample()
    ds = _StubDataset(1.0, o, c)
    first = ds.run(s)[1]
    first[:] = -999.0                              # a caller mutating its view
    second = ds.run(s)[1]
    assert not np.array_equal(second, first)
    assert np.isfinite(second).any()


def test_zeroed_feature_columns_survive_a_round_trip():
    s, o, c = _sample()
    ds = _StubDataset(1.0, o, c, zero_cols=[5, 6])
    miss = ds.run(s)[1]
    hit = ds.run(s)[1]
    assert (miss[:, [5, 6]] == 0.0).all()
    assert np.array_equal(miss, hit, equal_nan=True)


def test_eviction_holds_the_byte_cap():
    _, o, c = _sample()
    cap = 2 * (23_400 * 9 * 4 + 23_400 * 4)
    ds = _StubDataset(cap / (1 << 30), o, c)
    for i in range(6):
        s, _, _ = _sample(date=f"2020-05-0{i + 1}", seed=i)
        ds.run(s)
    info = ds.processor.cache_info()
    assert info.bytes <= info.max_bytes
    assert info.entries <= 2
    # The oldest key went first: asking for it is another miss.
    misses = info.misses
    ds.run(_sample(date="2020-05-01", seed=0)[0])
    assert ds.processor.cache_info().misses == misses + 1


def test_the_cache_keys_on_ticker_as_well_as_date():
    sa, o, c = _sample(ticker="AAPL", seed=1)
    sb, _, _ = _sample(ticker="MSFT", seed=2)
    ds = _StubDataset(1.0, o, c)
    a = ds.run(sa)[1]
    b = ds.run(sb)[1]
    assert not np.array_equal(a, b, equal_nan=True)
    assert ds.processor.cache_info().entries == 2
    assert np.array_equal(a, ds.run(sa)[1], equal_nan=True)


def test_disabled_cache_stays_on_the_float64_path():
    s, o, c = _sample()
    ds = _StubDataset(0.0, o, c)
    out = ds.run(s)
    assert ds.processor.cache_info().max_bytes == 0
    assert out[1].dtype == np.float64


def test_a_blank_ticker_never_shares_a_cache_slot():
    """The one way keying on (ticker, date) could silently swap stocks."""
    a, o, c = _sample(ticker="", date="2020-05-01", seed=1)
    b, _, _ = _sample(ticker="", date="2020-05-01", seed=2)
    ds = _StubDataset(1.0, o, c)
    ga = ds.run(a)[1]
    gb = ds.run(b)[1]
    assert not np.array_equal(ga, gb, equal_nan=True)   # b did NOT get a's grid
    assert ds.processor.cache_info().entries == 0       # nothing was cached


# ---------------------------------------------------------------------------
# End to end, on real shards: the cache must not perturb a single view or
# target. The unit tests above pin the grid; this pins what training sees.
# ---------------------------------------------------------------------------

MOSAIC_DIR = Path("lab/market-jepa-mosaic/1Hz_mosaic_mnth")

# THE CACHE TESTS NEED A SPARSE DATASET, and the canonical mosaic is no longer
# one. A dense record is adopted as-is by SessionPreprocessor, which bypasses
# the cache entirely -- there is no grid rebuild left to amortize -- so the
# cache can only be exercised against the sparse copy kept beside it.
SPARSE_MOSAIC_DIR = Path("lab/market-jepa-mosaic/1Hz_mosaic_mnth_sparse")

needs_mosaic = pytest.mark.skipif(
    not (MOSAIC_DIR / "2023" / "01" / "index.json").is_file(),
    reason="2023 mosaic months not available locally",
)

needs_sparse_mosaic = pytest.mark.skipif(
    not (SPARSE_MOSAIC_DIR / "2023" / "01" / "index.json").is_file(),
    reason="sparse 2023 mosaic months not available locally",
)


def _real_dataset(grid_cache_gb, root=MOSAIC_DIR):
    from streaming import Stream

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    return StreamingMarketDataset(
        augmentations=[{"name": "cross_stock", "n_stocks": 2}],
        date_start="2023-01-01",
        date_end="2023-01-31",
        seed=1234,
        epoch_dependent_seed=False,      # augmentations keyed on idx alone
        targets={"horizons": [300, 900], "types": ["return", "volatility_change"]},
        streams=[Stream(local=str(root / "2023/01"))],
        shuffle=False,
        batch_size=4,
        allow_unsafe_types=True,
        grid_cache_gb=grid_cache_gb,
    )


def _flatten(pairs):
    """Every tensor a pair carries, in a stable order."""
    out = []
    for p in pairs:
        out.extend(np.asarray(v, dtype=np.float64) for v in p["views"])
        for key in ("targets", "lengths"):
            if key in p:
                out.append(np.asarray(p[key], dtype=np.float64))
    return out


@needs_sparse_mosaic
def test_cached_and_uncached_datasets_agree_bit_for_bit():
    """Same indices, same everything — including on a cache HIT.

    The second read of each index is served from the cache, so this covers the
    miss path and the hit path against the same uncached reference.
    """
    plain = _real_dataset(0.0, SPARSE_MOSAIC_DIR)
    cached = _real_dataset(4.0, SPARSE_MOSAIC_DIR)
    assert cached._session_preprocessor.cache_info().max_bytes > 0
    assert plain._session_preprocessor.cache_info().max_bytes == 0

    checked = 0
    for idx in (0, 1, 5, 11, 23):
        ref = _flatten(plain[idx])
        miss = _flatten(cached[idx])       # builds and stores
        hit = _flatten(cached[idx])        # served from RAM
        assert len(ref) == len(miss) == len(hit) and ref
        for a, b, c in zip(ref, miss, hit):
            assert np.array_equal(a, b, equal_nan=True), f"miss path differs at idx {idx}"
            assert np.array_equal(a, c, equal_nan=True), f"hit path differs at idx {idx}"
        checked += len(ref)
    assert cached._session_preprocessor.cache_info().hits > 0, "no cache hit was exercised"
    assert checked > 0


@needs_sparse_mosaic
def test_the_cache_does_not_narrow_what_training_sees():
    """Breadth is untouched: fresh crops every visit, and the SAME fresh crops.

    The cache holds the dense 1 Hz DAY, which is a deterministic function of
    (ticker, date) — rebuilding it never produced new data. Every source of
    variety (which window, how wide, at what aggregation, which partner stock)
    is drawn downstream from a seed keyed on a per-worker counter that the
    cache neither reads nor perturbs.

    So two things have to hold, and neither is implied by the other:
      1. repeated visits to one index still differ  -> breadth is preserved
      2. cached and uncached differ IDENTICALLY     -> and it is the same
                                                       breadth as before
    """
    plain = _real_dataset(0.0, SPARSE_MOSAIC_DIR)
    cached = _real_dataset(4.0, SPARSE_MOSAIC_DIR)
    for ds in (plain, cached):
        ds._epoch_dependent_seed = True          # the training default

    # Same call sequence on both, so the per-worker counters stay in lockstep.
    seq = [3, 3, 3, 8, 3, 8]
    plain_out = [_flatten(plain[i]) for i in seq]
    cached_out = [_flatten(cached[i]) for i in seq]

    # 1. Revisits to index 3 are genuinely different draws, not a replay.
    visits = [plain_out[i] for i, idx in enumerate(seq) if idx == 3]
    assert len(visits) == 4
    distinct = sum(
        1 for j in range(1, len(visits))
        if not np.array_equal(visits[0][0], visits[j][0], equal_nan=True)
    )
    assert distinct == len(visits) - 1, (
        "repeated visits produced identical views — augmentation is not varying"
    )

    # 2. ...and the cached dataset draws exactly the same sequence.
    for n, (p, c) in enumerate(zip(plain_out, cached_out)):
        assert len(p) == len(c) and p
        for a, b in zip(p, c):
            assert np.array_equal(a, b, equal_nan=True), (
                f"call {n} (idx {seq[n]}) diverged between cached and uncached"
            )
    assert cached._session_preprocessor.cache_info().hits >= 4, "the repeats were not served from cache"


@needs_mosaic
def test_a_dense_dataset_never_touches_the_cache():
    """The canonical mosaic stores reconstructed sessions, so asking for a
    cache must get you nothing -- not a cache that quietly fills.

    This is the invariant that lets GRID_CACHE_GB default to 0. The cache
    existed to amortize the sparse->dense rebuild, which was 71% of a worker's
    per-sample CPU; a dense record is adopted as-is, so a non-zero cache would
    only take memory from the reader. Reading the SAME index twice is what
    would populate it on the sparse path.
    """
    dense = _real_dataset(4.0)
    for idx in (0, 1, 0, 1):
        assert _flatten(dense[idx])
    info = dense._session_preprocessor.cache_info()
    assert info.hits == 0 and info.misses == 0 and info.entries == 0, (
        f"dense reads went through the cache: {info}")
