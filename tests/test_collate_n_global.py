"""Tests for n_global_views propagation through collate_bucketed."""

import torch
import pytest

from market_jepa.training.utils import collate_bucketed


def _make_sample(n_views, seq_len, n_features=9, bucket_key=0, n_global_views=None):
    """Create a sample dict mimicking streaming_dataset output."""
    views = [torch.randn(n_features, seq_len) for _ in range(n_views)]
    lengths = torch.tensor([seq_len] * n_views, dtype=torch.long)
    d = {"views": views, "lengths": lengths, "bucket_key": bucket_key}
    if n_global_views is not None:
        d["n_global_views"] = n_global_views
    return d


class TestCollateNGlobal:
    def test_n_global_propagated(self):
        """Bucket carries n_global_views from input dicts."""
        batch = [
            _make_sample(8, 512, n_global_views=2, bucket_key=0),
            _make_sample(8, 512, n_global_views=2, bucket_key=0),
        ]
        result = collate_bucketed(batch)
        assert len(result["buckets"]) == 1
        assert result["buckets"][0]["n_global_views"] == 2

    def test_absent_for_old_augs(self):
        """2-view dicts without the key → no n_global_views in bucket."""
        batch = [
            _make_sample(2, 1200, bucket_key=0),
            _make_sample(2, 1200, bucket_key=0),
        ]
        result = collate_bucketed(batch)
        assert "n_global_views" not in result["buckets"][0]

    def test_8_view_padding(self):
        """Collation handles 8 views with 2 different lengths correctly."""
        # Global views: seq_len=2048, local views: seq_len=512
        def make_rrc_sample(bucket_key=0):
            views = []
            lengths = []
            for i in range(8):
                sl = 2048 if i < 2 else 512
                views.append(torch.randn(9, sl))
                lengths.append(sl)
            return {
                "views": views,
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "bucket_key": bucket_key,
                "n_global_views": 2,
            }

        batch = [make_rrc_sample(), make_rrc_sample()]
        result = collate_bucketed(batch)
        bucket = result["buckets"][0]

        assert len(bucket["views"]) == 8
        # Global views (idx 0, 1) should have max_len=2048
        assert bucket["views"][0].shape == (2, 9, 2048)
        assert bucket["views"][1].shape == (2, 9, 2048)
        # Local views (idx 2-7) should have max_len=512
        for v in range(2, 8):
            assert bucket["views"][v].shape == (2, 9, 512)
        assert bucket["n_global_views"] == 2
