"""Smoke tests for the TimeMAE mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from market_jepa.modeling import TimeMAE, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig


def _tiny_backbone(n_features: int = 9, pool: str = "mean") -> torch.nn.Module:
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


def _tiny_timemae(**kwargs) -> TimeMAE:
    kwargs.setdefault("vocab_size", 32)
    kwargs.setdefault("reg_layers", 2)
    return TimeMAE(backbone=_tiny_backbone(), **kwargs)


def test_timemae_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        TimeMAE(backbone=torch.nn.Linear(8, 8))


def test_timemae_rejects_cls_pool():
    with pytest.raises(ValueError, match="pool"):
        TimeMAE(backbone=_tiny_backbone(pool="cls"))


def test_timemae_rejects_bad_mask_ratio():
    with pytest.raises(ValueError, match="mask_ratio"):
        _tiny_timemae(mask_ratio=0.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        _tiny_timemae(mask_ratio=1.0)


def test_encode_shape_matches_other_modes():
    model = _tiny_timemae().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    assert out["embeddings"].shape == (4, 1, model.d_embedding)


def test_forward_loss_finite_and_components():
    torch.manual_seed(0)
    model = _tiny_timemae().eval()
    x = torch.randn(4, 9, 128)
    out = model(x)
    assert out is not None
    assert torch.isfinite(out["timemae_loss"])
    assert torch.isfinite(out["timemae_align_loss"])
    assert torch.isfinite(out["timemae_reconstruct_loss"])
    assert 0.0 <= out["timemae_token_acc"].item() <= 1.0


def test_forward_returns_none_when_too_short():
    # 8 timesteps = 1 patch → cannot split into visible + masked.
    model = _tiny_timemae().eval()
    assert model(torch.randn(2, 9, 8)) is None


def test_momentum_encoder_frozen_and_ema_updates():
    torch.manual_seed(0)
    model = _tiny_timemae(ema_momentum=0.5)
    assert all(not p.requires_grad for p in model.momentum_backbone.parameters())
    # At init the momentum encoder is an exact copy.
    p_q0 = next(model.backbone.parameters()).detach().clone()
    p_m0 = next(model.momentum_backbone.parameters()).detach().clone()
    assert torch.allclose(p_q0, p_m0)
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(1.0)
    model.post_training_step(1, 10)
    p_m1 = next(model.momentum_backbone.parameters()).detach().clone()
    p_q1 = next(model.backbone.parameters()).detach().clone()
    assert torch.allclose(p_m1, 0.5 * p_m0 + 0.5 * p_q1, atol=1e-6)


def test_training_step_returns_loss_and_metrics():
    model = _tiny_timemae()
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
    assert "train/timemae_token_acc" in out["metrics"]


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = _tiny_timemae().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
        loaded = TimeMAE.from_pretrained(tmp).eval()

    assert cfg["class"] == "TimeMAE"
    assert cfg["vocab_size"] == 32
    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]
    assert torch.allclose(emb_before, emb_after, atol=1e-5)
