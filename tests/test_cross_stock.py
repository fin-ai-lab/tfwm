"""Tests for the cross_stock augmentation (same wall-clock window, K tickers).

The dataset-level tests need the real 2023 mosaic months on local disk (the data host)
and are skipped elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from market_jepa.augmentations import AUGMENTATION_REGISTRY

MOSAIC_DIR = Path("lab/market-jepa-mosaic/1Hz_mosaic_mnth")

needs_mosaic = pytest.mark.skipif(
    not (MOSAIC_DIR / "2023" / "01" / "index.json").is_file(),
    reason="2023 mosaic months not available locally",
)


def test_registered():
    assert "cross_stock" in AUGMENTATION_REGISTRY


def _make_dataset(cls=None, months=("2023/01",), n_stocks=4, **overrides):
    from streaming import Stream

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    cls = cls or StreamingMarketDataset
    kwargs = dict(
        augmentations=[{"name": "cross_stock", "n_stocks": n_stocks}],
        date_start="2023-01-01",
        date_end="2023-02-28",
        seed=1234,
        epoch_dependent_seed=False,
        # PINNED to the pre-2026-09-13 constructor defaults. These tests are
        # about augmentation mechanics -- grouping, warping, which channels a
        # noise op touches -- and assert a 9-row view. The information token
        # now defaults ON, so leaving these unset would silently retarget them
        # at a 20-row view and test something else.
        info_norm_stats=False,
        info_window=False,
        targets={"horizons": [300], "types": ["return"]},
        streams=[Stream(local=str(MOSAIC_DIR / m)) for m in months],
        shuffle=False,
        batch_size=4,
        allow_unsafe_types=True,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


@needs_mosaic
class TestCrossStockDataset:
    def test_basic_group_structure(self):
        ds = _make_dataset(n_stocks=4)
        pairs = ds[7]
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["bucket_key"] == 0
        assert pair["n_global_views"] == 4
        assert len(pair["views"]) == 4
        for v in pair["views"]:
            assert v.shape == (9, 2048)
            assert torch.isfinite(v).all()
        assert pair["lengths"].tolist() == [2048] * 4
        # (K, n_targets), not the focal's single row: 1c0ea79 began labelling
        # every stock in the group at the shared anchor so a ranking loss has
        # a real cross-section to rank within. xs_group carries K to the loss.
        assert pair["targets"].shape == (4, 1)
        assert pair["xs_group"] == 4
        # Views come from different tickers — same-window crops of distinct
        # stocks must not be identical.
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.equal(pair["views"][i], pair["views"][j])

    def test_deterministic(self):
        ds1 = _make_dataset(n_stocks=3)
        ds2 = _make_dataset(n_stocks=3)
        p1, p2 = ds1[11][0], ds2[11][0]
        assert p1["ticker"] == p2["ticker"]
        for v1, v2 in zip(p1["views"], p2["views"]):
            assert torch.equal(v1, v2)

    def test_wall_clock_alignment(self):
        """Every view in a group must cover the identical wall-clock window."""
        from market_jepa.training.streaming_dataset import StreamingMarketDataset

        recorded: list[tuple[int, int, int]] = []  # (abs_start_sec, window, agg)
        sec0_by_features_id: dict[int, int] = {}

        class Instrumented(StreamingMarketDataset):
            def _preprocess_to_numpy(self, sample):
                result = super()._preprocess_to_numpy(sample)
                if result is not None:
                    sec, feat = result
                    sec0_by_features_id[id(feat)] = int(sec[0])
                return result

            def _build_aligned_view(self, features, start_idx, window, agg,
                                    date_str, rf_offset, **kw):
                sec0 = sec0_by_features_id.get(id(features))
                assert sec0 is not None, "features array not seen by preprocessing"
                recorded.append((sec0 + start_idx, window, agg))
                return super()._build_aligned_view(
                    features, start_idx, window, agg, date_str, rf_offset, **kw,
                )

        ds = _make_dataset(cls=Instrumented, n_stocks=5)
        for idx in [3, 42, 199]:
            recorded.clear()
            pairs = ds[idx]
            assert len(pairs) == 1 and pairs[0]["bucket_key"] == 0
            # One record per accepted view plus possibly rejected partners is
            # not possible: _build_aligned_view is only called for partners
            # that already passed the coverage check, and it only fails on
            # aggregation failure (window multiple of agg → never here).
            assert len(recorded) == 5
            abs_starts = {r[0] for r in recorded}
            windows = {r[1] for r in recorded}
            aggs = {r[2] for r in recorded}
            assert len(abs_starts) == 1, f"views start at different times: {abs_starts}"
            assert len(windows) == 1 and len(aggs) == 1

    def test_multi_stream_global_index_order(self):
        """Sidecar concat order must match StreamingDataset global sample ids."""
        from streaming import StreamingDataset

        from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar

        ds = _make_dataset(months=("2023/01", "2023/02"), n_stocks=2)
        tickers, dates = [], []
        for m in ("2023/01", "2023/02"):
            meta = ensure_ticker_date_sidecar(MOSAIC_DIR / m)
            tickers.extend(meta["tickers"])
            dates.extend(meta["dates"])
        assert len(tickers) == ds.num_samples
        rng = np.random.RandomState(0)
        for gidx in rng.randint(0, ds.num_samples, 12):
            sample = StreamingDataset.__getitem__(ds, int(gidx))
            assert sample["ticker"] == tickers[gidx]
            assert sample["date"] == dates[gidx]

    def test_collate(self):
        from market_jepa.training.utils import collate_bucketed

        ds = _make_dataset(n_stocks=4)
        batch = collate_bucketed([ds[i] for i in range(8)])
        assert len(batch["buckets"]) == 1
        bucket = batch["buckets"][0]
        assert bucket["n_global_views"] == 4
        assert len(bucket["views"]) == 4
        for v in bucket["views"]:
            assert v.shape == (8, 9, 2048)


class TestEncodeFastPath:
    def test_matches_per_view_loop(self):
        from market_jepa.modeling.backbones import create_backbone
        from market_jepa.modeling.backbones.transformer import TransformerConfig
        from market_jepa.modeling.modes.lejepa import LeJEPA

        torch.manual_seed(0)
        config = TransformerConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, patch_size=8, drop_path_rate=0.0,
        )
        backbone = create_backbone("transformer", n_features=9, d_embedding=32, config=config)
        model = LeJEPA(backbone=backbone, proj_dim=16)
        model.eval()

        views = [torch.randn(3, 9, 128) for _ in range(4)]
        lengths = [torch.full((3,), 128, dtype=torch.long) for _ in range(4)]

        with torch.no_grad():
            fast = model.encode(views, lengths)["embeddings"]
            slow = torch.stack(
                [backbone(v, l) for v, l in zip(views, lengths)], dim=1,
            )
        assert fast.shape == (3, 4, 32)
        assert torch.allclose(fast, slow, atol=1e-5)

    def test_unequal_shapes_fall_back(self):
        from market_jepa.modeling.backbones import create_backbone
        from market_jepa.modeling.backbones.transformer import TransformerConfig
        from market_jepa.modeling.modes.lejepa import LeJEPA

        config = TransformerConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, patch_size=8, drop_path_rate=0.0,
        )
        backbone = create_backbone("transformer", n_features=9, d_embedding=32, config=config)
        model = LeJEPA(backbone=backbone, proj_dim=16)
        model.eval()
        views = [torch.randn(3, 9, 128), torch.randn(3, 9, 64)]
        with torch.no_grad():
            out = model.encode(views, None)["embeddings"]
        assert out.shape == (3, 2, 32)


class TestPairWeightedLoss:
    def _model(self):
        from market_jepa.modeling.backbones import create_backbone
        from market_jepa.modeling.backbones.transformer import TransformerConfig
        from market_jepa.modeling.modes.lejepa import LeJEPA

        config = TransformerConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, patch_size=8, drop_path_rate=0.0,
        )
        backbone = create_backbone("transformer", n_features=9, d_embedding=32, config=config)
        return LeJEPA(backbone=backbone, proj_dim=16)

    def test_all_ones_graph_reduces_to_unweighted(self):
        """A fully connected edge graph == the flat pull-to-the-mean loss."""
        torch.manual_seed(0)
        model = self._model()
        proj = torch.randn(8, 4, 16)
        base = model.compute_loss(proj, n_global_views=4)
        weighted = model.compute_loss(
            proj, n_global_views=4, pair_weights=torch.ones(8, 4, 4),
        )
        assert torch.allclose(base["inv_loss"], weighted["inv_loss"], atol=1e-6)
        assert torch.allclose(weighted["pair_w_mean"], torch.tensor(1.0), atol=1e-6)

    def test_dropped_edges_ignored(self):
        """Pairs with zero weight contribute nothing to the invariance term."""
        torch.manual_seed(0)
        model = self._model()
        proj = torch.zeros(2, 2, 16)
        proj[0, 1, 0] = 2.0  # group 0: the only nonzero embedding gap
        proj[1, 1, 0] = 4.0  # group 1: a larger gap, edge dropped below
        w = torch.ones(2, 2, 2)
        w[1] = 0.0
        kept = model.compute_loss(proj, pair_weights=w)["inv_loss"]
        # Same loss as the group-0-only batch scaled by the mean-1
        # renormalization: the dropped group's large gap never enters.
        full = model.compute_loss(proj, pair_weights=torch.ones(2, 2, 2))["inv_loss"]
        assert kept < full
        assert torch.allclose(
            model.compute_loss(proj, pair_weights=w)["pair_w_mean"],
            torch.tensor(0.5),
        )


@needs_mosaic
class TestCrossStockLocals:
    def _ds(self, structured, cls=None):
        return _make_dataset(
            cls=cls,
            n_stocks=2,
            augmentations=[{
                "name": "cross_stock", "n_stocks": 2,
                "cross_stock_local_views": 6, "local_seq_len": 512,
                "local_scale_range": [0.1, 0.5],
                "structured_matching": structured,
            }],
        )

    def test_flat_view_structure(self):
        ds = self._ds(structured=False)
        pair = ds[7][0]
        assert pair["bucket_key"] == 0
        assert pair["n_global_views"] == 2
        assert len(pair["views"]) == 8  # 2 globals + 3 slots x 2 stocks
        for v in pair["views"][:2]:
            assert v.shape == (9, 2048)
        for v in pair["views"][2:]:
            assert v.shape == (9, 512)
        assert "pair_weights" not in pair

    def test_structured_edge_matrix(self):
        ds = self._ds(structured=True)
        pair = ds[7][0]
        w = pair["pair_weights"]
        assert w.shape == (8, 8)
        assert torch.equal(w, w.T)
        assert torch.all(w.diagonal() == 0)
        # global<->global edge
        assert w[0, 1] == 1
        # matched local pairs: slots at (2,3), (4,5), (6,7)
        for base in (2, 4, 6):
            assert w[base, base + 1] == 1
            # local <-> own global
            assert w[base, 0] == 1 and w[base + 1, 1] == 1
            # local NOT matched to other stock's global
            assert w[base, 1] == 0 and w[base + 1, 0] == 0
        # locals of different slots not connected
        assert w[2, 4] == 0 and w[3, 5] == 0
        # total edges (undirected): 1 gg + 3 ll + 6 lg = 10 -> 20 entries
        assert int(w.sum().item()) == 20

    def test_locals_wall_clock_matched(self):
        """Slot j's two views must cover the identical wall-clock sub-window."""
        from market_jepa.training.streaming_dataset import StreamingMarketDataset

        recorded = []
        sec0_by_id = {}

        class Instrumented(StreamingMarketDataset):
            def _preprocess_to_numpy(self, sample):
                r = super()._preprocess_to_numpy(sample)
                if r is not None:
                    sec0_by_id[id(r[1])] = int(r[0][0])
                return r

            def _build_aligned_view(self, features, start_idx, window, agg,
                                    date_str, rf_offset, **kw):
                recorded.append((sec0_by_id[id(features)] + start_idx, window, agg))
                return super()._build_aligned_view(
                    features, start_idx, window, agg, date_str, rf_offset, **kw,
                )

        ds = self._ds(structured=True, cls=Instrumented)
        recorded.clear()
        pair = ds[13][0]
        assert pair["bucket_key"] == 0
        assert len(recorded) == 8
        g_abs, g_win, _ = recorded[0]
        assert recorded[1][:2] == (g_abs, g_win)  # partner global matches
        for base in (2, 4, 6):
            a, b = recorded[base], recorded[base + 1]
            assert a == b, f"slot views differ: {a} vs {b}"
            # local lies inside the shared window
            assert g_abs <= a[0] and a[0] + a[1] <= g_abs + g_win

    def test_collate_and_loss_shapes(self):
        from market_jepa.modeling.backbones import create_backbone
        from market_jepa.modeling.backbones.transformer import TransformerConfig
        from market_jepa.modeling.modes.lejepa import LeJEPA
        from market_jepa.training.utils import collate_bucketed

        ds = self._ds(structured=True)
        items = [ds[i] for i in range(6)]
        items = [it for it in items if it[0].get("bucket_key", -1) != -1]
        batch = collate_bucketed(items)
        bucket = batch["buckets"][0]
        assert bucket["pair_weights"].shape == (len(items), 8, 8)

        config = TransformerConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, patch_size=8, drop_path_rate=0.0,
        )
        backbone = create_backbone("transformer", n_features=9, d_embedding=32, config=config)
        model = LeJEPA(backbone=backbone, proj_dim=16)
        out = model.training_step({"buckets": [
            {k: bucket[k] for k in ("views", "lengths", "n_global_views", "pair_weights")}
        ]}, device="cpu")
        assert out is not None
        assert np.isfinite(out["loss"])
        assert "train/pair_w_mean" in out["metrics"]  # edge-density diagnostic


REPO_ROOT = Path(__file__).resolve().parents[1]
INDUSTRY_MAP = REPO_ROOT / "data" / "industry_map.parquet"


def _instrumented_cls():
    """Subclass that records the ticker of every features array passed to
    _build_aligned_view (the focal is recorded first, then each accepted-
    coverage partner attempt)."""
    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    seen: list[str] = []
    ticker_by_features_id: dict[int, str] = {}

    class Instrumented(StreamingMarketDataset):
        def _preprocess_to_numpy(self, sample):
            result = super()._preprocess_to_numpy(sample)
            if result is not None:
                ticker_by_features_id[id(result[1])] = str(sample["ticker"])
            return result

        def _build_aligned_view(self, features, start_idx, window, agg,
                                    date_str, rf_offset, **kw):
            seen.append(ticker_by_features_id[id(features)])
            return super()._build_aligned_view(
                features, start_idx, window, agg, date_str, rf_offset, **kw,
            )

    return Instrumented, seen


@needs_mosaic
class TestSameIndustryDataset:
    """Hard same-industry pairing via industry_table."""

    def test_missing_month_raises(self, tmp_path):
        import pandas as pd

        bad = tmp_path / "map.parquet"
        pd.DataFrame({"month": ["1999-01"], "ticker": ["AAPL"], "ff49": [35]}).to_parquet(bad)
        with pytest.raises(ValueError, match="lacks dataset months"):
            _make_dataset(
                n_stocks=2,
                augmentations=[{
                    "name": "cross_stock", "n_stocks": 2,
                    "industry_table": str(bad),
                }],
            )

    def _focal_in_big_industry(self, ind_by_ticker):
        """Pick a global idx whose focal ticker sits in an industry with many
        same-date peers (so the restricted pass cannot plausibly fail)."""
        from collections import Counter

        from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar

        meta = ensure_ticker_date_sidecar(MOSAIC_DIR / "2023" / "01")
        date0 = meta["dates"][0]
        day_tickers = [
            t for t, d in zip(meta["tickers"], meta["dates"]) if d == date0
        ]
        by_ind = Counter(
            ind_by_ticker[t] for t in day_tickers if t in ind_by_ticker
        )
        big_ind, n = by_ind.most_common(1)[0]
        assert n >= 20, f"biggest industry on {date0} has only {n} members"
        for i, (t, d) in enumerate(zip(meta["tickers"], meta["dates"])):
            if d == date0 and ind_by_ticker.get(t) == big_ind:
                return i, big_ind
        raise AssertionError("unreachable")

    def test_partner_same_industry(self):
        import pandas as pd

        df = pd.read_parquet(INDUSTRY_MAP)
        jan = df[df.month == "2023-01"]
        ind_by_ticker = dict(zip(jan.ticker, jan.ff49))
        idx, big_ind = self._focal_in_big_industry(ind_by_ticker)

        cls, seen = _instrumented_cls()
        ds = _make_dataset(
            cls=cls, n_stocks=2,
            augmentations=[{
                "name": "cross_stock", "n_stocks": 2,
                "industry_table": str(INDUSTRY_MAP),
            }],
        )
        pairs = ds[idx]
        assert pairs[0]["bucket_key"] == 0
        assert len(pairs[0]["views"]) == 2
        # Every partner whose coverage check passed was same-industry.
        assert len(seen) >= 2
        for t in seen:
            assert ind_by_ticker.get(t) == big_ind, (
                f"partner {t} (ff49={ind_by_ticker.get(t)}) != focal industry {big_ind}"
            )

    def test_singleton_industry_falls_back(self, tmp_path):
        import pandas as pd

        from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar

        meta = ensure_ticker_date_sidecar(MOSAIC_DIR / "2023" / "01")
        focal_idx = 7
        focal = meta["tickers"][focal_idx]
        tickers = sorted(set(meta["tickers"]))
        table = tmp_path / "map.parquet"
        pd.DataFrame({
            "month": ["2023-01"] * len(tickers),
            "ticker": tickers,
            "ff49": [99 if t == focal else 1 for t in tickers],
        }).to_parquet(table)

        cls, seen = _instrumented_cls()
        ds = _make_dataset(
            cls=cls, n_stocks=2,
            augmentations=[{
                "name": "cross_stock", "n_stocks": 2,
                "industry_table": str(table),
            }],
        )
        pairs = ds[focal_idx]
        # No same-industry peer exists; the unrestricted pass still fills
        # the group.
        assert pairs[0]["bucket_key"] == 0
        assert len(pairs[0]["views"]) == 2
        partner = [t for t in seen if t != focal]
        assert partner, "no partner was ever attempted"

    def test_unrestricted_matches_plain_k2_when_unmapped(self, tmp_path):
        """A focal absent from the table behaves exactly like plain K=2."""
        import pandas as pd

        from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar

        meta = ensure_ticker_date_sidecar(MOSAIC_DIR / "2023" / "01")
        focal_idx = 11
        focal = meta["tickers"][focal_idx]
        others = sorted(set(meta["tickers"]) - {focal})
        table = tmp_path / "map.parquet"
        pd.DataFrame({
            "month": ["2023-01"] * len(others),
            "ticker": others,
            "ff49": [1] * len(others),
        }).to_parquet(table)

        ds_ind = _make_dataset(
            n_stocks=2,
            augmentations=[{
                "name": "cross_stock", "n_stocks": 2,
                "industry_table": str(table),
            }],
        )
        ds_plain = _make_dataset(n_stocks=2)
        p_ind, p_plain = ds_ind[focal_idx][0], ds_plain[focal_idx][0]
        assert p_ind["ticker"] == p_plain["ticker"]
        for v1, v2 in zip(p_ind["views"], p_plain["views"]):
            assert torch.equal(v1, v2)
