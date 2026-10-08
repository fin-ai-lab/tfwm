"""Tests for mixing several augmentations in one LeJEPA run.

Covers the three pieces that make p%-per-batch mixing work:
- weighted config sampling (AugmentationConfig.weight -> _pick_aug_idx),
- collate_bucketed producing one bucket per config for heterogeneous
  view counts (RRC 2g+6l alongside cross_stock K=2),
- LeJEPA per-bucket loss: _combine_bucket_outputs weighting and a full
  training_step over a mixed batch.
"""

import numpy as np
import pytest
import torch

from market_jepa.modeling import LeJEPA, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.training.streaming_dataset import StreamingMarketDataset
from market_jepa.training.utils import collate_bucketed


# ── weighted sampling ───────────────────────────────────────────────────────


def _bare_dataset(weights):
    """StreamingMarketDataset shell with just the aug-sampling attrs set."""
    ds = object.__new__(StreamingMarketDataset)
    w = np.asarray(weights, dtype=np.float64)
    ds._aug_configs = [{"name": f"aug{i}"} for i in range(len(w))]
    ds._aug_uniform = bool(np.all(w == w[0]))
    ds._aug_cum_probs = np.cumsum(w / w.sum())
    return ds


def test_uniform_keeps_randint_stream():
    """Equal weights must reproduce the legacy rng.randint draw exactly."""
    ds = _bare_dataset([1.0, 1.0, 1.0])
    assert ds._aug_uniform
    for seed in range(20):
        assert ds._pick_aug_idx(np.random.RandomState(seed)) == \
            np.random.RandomState(seed).randint(0, 3)


def test_weighted_proportions():
    """3:1 weights -> ~75/25 split over many draws."""
    ds = _bare_dataset([3.0, 1.0])
    assert not ds._aug_uniform
    rng = np.random.RandomState(0)
    draws = np.array([ds._pick_aug_idx(rng) for _ in range(20_000)])
    frac0 = (draws == 0).mean()
    assert abs(frac0 - 0.75) < 0.01
    assert set(np.unique(draws)) == {0, 1}


def test_weighted_four_way():
    ds = _bare_dataset([0.5, 0.25, 0.125, 0.125])
    rng = np.random.RandomState(1)
    draws = np.array([ds._pick_aug_idx(rng) for _ in range(40_000)])
    for i, p in enumerate([0.5, 0.25, 0.125, 0.125]):
        assert abs((draws == i).mean() - p) < 0.01


# ── mixed-bucket collate + loss ─────────────────────────────────────────────


def _sample(n_views, seq_lens, bucket_key, n_global_views, n_features=9):
    views = [torch.randn(n_features, sl) for sl in seq_lens]
    return {
        "views": views,
        "lengths": torch.tensor(seq_lens, dtype=torch.long),
        "bucket_key": bucket_key,
        "n_global_views": n_global_views,
    }


def _mixed_batch(n_rrc=3, n_cs=2):
    """RRC samples (2 globals @2048 + 6 locals @512) + cross_stock K=2."""
    batch = [
        _sample(8, [2048] * 2 + [512] * 6, bucket_key=0, n_global_views=2)
        for _ in range(n_rrc)
    ] + [
        _sample(2, [2048] * 2, bucket_key=1, n_global_views=2)
        for _ in range(n_cs)
    ]
    return collate_bucketed(batch)


def test_collate_heterogeneous_buckets():
    out = _mixed_batch(n_rrc=3, n_cs=2)
    assert len(out["buckets"]) == 2
    by_views = {len(b["views"]): b for b in out["buckets"]}
    assert by_views[8]["views"][0].shape[0] == 3   # rrc bucket batch
    assert by_views[2]["views"][0].shape[0] == 2   # cross_stock bucket batch
    assert by_views[8]["n_global_views"] == 2
    assert by_views[2]["n_global_views"] == 2


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = TransformerConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, patch_size=8, drop_path_rate=0.0,
    )
    backbone = create_backbone(
        "transformer", n_features=9, d_embedding=32, config=config,
    )
    return LeJEPA(backbone=backbone, proj_dim=32, lamb=0.02)


def test_combine_single_bucket_identity(model):
    proj = torch.randn(4, 8, 32)
    out = model.compute_loss(proj, n_global_views=2)
    combined = model._combine_bucket_outputs([out], [4])
    for k in ("lejepa_loss", "sigreg_loss", "inv_loss",
              "sigreg_loss_normalized", "lejepa_loss_normalized"):
        torch.testing.assert_close(combined[k], out[k])


def test_combine_weights_by_batch_size(model):
    o1 = model.compute_loss(torch.randn(6, 8, 32), n_global_views=2)
    o2 = model.compute_loss(torch.randn(2, 2, 32), n_global_views=2)
    combined = model._combine_bucket_outputs([o1, o2], [6, 2])
    expected = o1["inv_loss"] * 0.75 + o2["inv_loss"] * 0.25
    torch.testing.assert_close(combined["inv_loss"], expected)


def test_training_step_mixed_batch(model):
    """Full training_step over an RRC + cross_stock mixed batch."""
    batch = _mixed_batch(n_rrc=3, n_cs=2)
    model.train()
    result = model.training_step(batch, device="cpu")
    assert result is not None
    assert np.isfinite(result["loss"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_training_step_single_bucket_unchanged(model):
    """Homogeneous batch (one bucket) still trains, matching legacy path."""
    batch = collate_bucketed(
        [_sample(8, [2048] * 2 + [512] * 6, 0, 2) for _ in range(4)]
    )
    result = model.training_step(batch, device="cpu")
    assert result is not None
    assert np.isfinite(result["loss"])


def test_eval_step_mixed_batch(model):
    out = model.eval_step([_mixed_batch(n_rrc=2, n_cs=2)], device="cpu")
    assert np.isfinite(out["eval/jepa_loss"])
