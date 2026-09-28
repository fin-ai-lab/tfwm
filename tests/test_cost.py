"""Smoke tests for the CoST mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from market_jepa.modeling import CoST, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig


def _tiny_backbone(n_features: int = 9, pool: str = "last") -> torch.nn.Module:
    cfg = TransformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        patch_size=8,
        drop_path_rate=0.0,
    )
    return create_backbone(
        backbone_type="transformer",
        n_features=n_features,
        d_embedding=64,
        config=cfg,
        pool=pool,
        max_seq_len=256,
    )


def _tiny_cost(**kwargs) -> CoST:
    kwargs.setdefault("kernels", [1, 2, 4])
    kwargs.setdefault("queue_size", 32)
    kwargs.setdefault("fourier_length", 16)
    return CoST(backbone=_tiny_backbone(), **kwargs)


def test_cost_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        CoST(backbone=torch.nn.Linear(8, 8))


def test_cost_rejects_cls_pool():
    with pytest.raises(ValueError, match="pool"):
        CoST(backbone=_tiny_backbone(pool="cls"))


def test_cost_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        _tiny_cost(temperature=0.0)


def test_encode_shape_is_trend_plus_season():
    model = _tiny_cost().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    # component_dims = d_embedding // 2 each for trend and season.
    assert out["embeddings"].shape == (4, 1, model.d_embedding)


def test_forward_loss_finite():
    torch.manual_seed(0)
    model = _tiny_cost().eval()
    x = torch.randn(4, 9, 128)
    out = model(x)
    assert out is not None
    assert torch.isfinite(out["cost_loss"])
    assert torch.isfinite(out["cost_time_loss"])
    assert torch.isfinite(out["cost_season_loss"])


def test_forward_returns_none_for_singleton_batch():
    model = _tiny_cost().eval()
    x = torch.randn(1, 9, 128)
    assert model(x) is None


def test_queue_advances_only_in_training():
    torch.manual_seed(0)
    model = _tiny_cost()
    x = torch.randn(4, 9, 128)
    model.train()
    ptr0 = int(model.queue_ptr.item())
    model(x)
    ptr1 = int(model.queue_ptr.item())
    assert ptr1 == (ptr0 + 4) % model.queue_size
    model.eval()
    model(x)
    assert int(model.queue_ptr.item()) == ptr1


def test_momentum_branch_is_frozen_and_ema_updates():
    torch.manual_seed(0)
    model = _tiny_cost(ema_momentum=0.5)
    assert all(not p.requires_grad for p in model.backbone_k.parameters())
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(1.0)
    p_q = next(model.backbone.parameters()).detach().clone()
    p_k_before = next(model.backbone_k.parameters()).detach().clone()
    model.post_training_step(1, 10)
    p_k_after = next(model.backbone_k.parameters()).detach().clone()
    assert torch.allclose(p_k_after, 0.5 * p_k_before + 0.5 * p_q, atol=1e-6)


def test_training_step_returns_loss_and_metrics():
    model = _tiny_cost()
    model.train()
    batch = {
        "buckets": [
            {
                "views": [torch.randn(4, 9, 128)],
                "lengths": [torch.full((4,), 128, dtype=torch.long)],
            }
        ]
    }
    out = model.training_step(batch, torch.device("cpu"))
    assert out is not None
    assert "train/loss" in out["metrics"]
    assert "train/cost_acc" in out["metrics"]


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = _tiny_cost().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
        loaded = CoST.from_pretrained(tmp).eval()

    assert cfg["class"] == "CoST"
    assert cfg["kernels"] == [1, 2, 4]
    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]
    assert torch.allclose(emb_before, emb_after, atol=1e-5)
