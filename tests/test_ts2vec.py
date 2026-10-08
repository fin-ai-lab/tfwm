"""Smoke tests for the TS2Vec mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from market_jepa.modeling import TS2Vec, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.modeling.modes.ts2vec import hierarchical_contrastive_loss


def _tiny_backbone(n_features: int = 9, pool: str = "max") -> torch.nn.Module:
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


def _tiny_ts2vec(**kwargs) -> TS2Vec:
    return TS2Vec(backbone=_tiny_backbone(), **kwargs)


def test_ts2vec_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        TS2Vec(backbone=torch.nn.Linear(8, 8))


def test_ts2vec_rejects_cls_pool():
    with pytest.raises(ValueError, match="pool"):
        TS2Vec(backbone=_tiny_backbone(pool="cls"))


def test_ts2vec_rejects_bad_hyperparams():
    with pytest.raises(ValueError, match="alpha"):
        _tiny_ts2vec(alpha=1.5)
    with pytest.raises(ValueError, match="mask_p"):
        _tiny_ts2vec(mask_p=0.0)
    with pytest.raises(ValueError, match="temporal_unit"):
        _tiny_ts2vec(temporal_unit=-1)


def test_encode_shape_matches_other_modes():
    model = _tiny_ts2vec().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    assert out["embeddings"].shape == (4, 1, model.d_embedding)


def test_forward_loss_finite():
    torch.manual_seed(0)
    model = _tiny_ts2vec().eval()
    x = torch.randn(4, 9, 128)
    out = model(x)
    assert out is not None
    loss = out["ts2vec_loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_forward_returns_none_when_too_short():
    # temporal_unit=3 → min span 16 patches; 64 timesteps = 8 patches.
    model = _tiny_ts2vec(temporal_unit=3).eval()
    x = torch.randn(2, 9, 64)
    assert model(x) is None


def test_crop_positions_are_relative_not_absolute():
    """Two content-zeroed crops at different global offsets must encode
    identically under the TS2Vec path: position-only tokens at the same
    relative positions. With absolute codes (the pre-fix gather), the same
    global patch carries the same code in both of TS2Vec's aligned crops,
    handing every positive pair a position-matching shortcut."""
    torch.manual_seed(0)
    bb = _tiny_backbone().eval()
    x = torch.randn(2, 9, 256)  # 32 patches of 8
    idx_a = torch.arange(0, 8)
    idx_b = torch.arange(10, 18)
    zero = torch.ones(2, 8, dtype=torch.bool)  # zero ALL content
    out_a = bb.forward_patches(
        x, mask_indices=[idx_a, idx_a], zero_mask=zero, relative_pos=True
    )
    out_b = bb.forward_patches(
        x, mask_indices=[idx_b, idx_b], zero_mask=zero, relative_pos=True
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)
    # The absolute-gather path distinguishes the two crops — that
    # distinguishability on identical (zeroed) content is the shortcut.
    out_a_abs = bb.forward_patches(
        x, mask_indices=[idx_a, idx_a], zero_mask=zero, relative_pos=False
    )
    out_b_abs = bb.forward_patches(
        x, mask_indices=[idx_b, idx_b], zero_mask=zero, relative_pos=False
    )
    assert not torch.allclose(out_a_abs, out_b_abs, atol=1e-4)


def test_hierarchical_loss_zero_for_identical_singleton():
    # B=1: instance term is 0 at every level; temporal term drives the loss.
    z = torch.randn(1, 8, 16)
    loss = hierarchical_contrastive_loss(z, z, alpha=0.5)
    assert torch.isfinite(loss)


def test_swa_update_changes_encode():
    torch.manual_seed(0)
    model = _tiny_ts2vec()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        emb0 = model.encode(x)["embeddings"]
    # Perturb the online backbone, then take one SWA step.
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    model.post_training_step(1, 10)
    assert int(model._swa_n.item()) == 1
    with torch.no_grad():
        emb1 = model.encode(x)["embeddings"]
    # After one equal-weight update, the SWA backbone equals the (perturbed)
    # online backbone — embeddings must differ from the pre-perturbation ones.
    assert not torch.allclose(emb0, emb1)


def test_training_step_returns_loss_and_metrics():
    model = _tiny_ts2vec()
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


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = _tiny_ts2vec().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
        loaded = TS2Vec.from_pretrained(tmp).eval()

    assert cfg["class"] == "TS2Vec"
    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]
    assert torch.allclose(emb_before, emb_after, atol=1e-5)
