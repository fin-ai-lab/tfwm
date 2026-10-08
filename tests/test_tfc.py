"""Smoke tests for the TF-C mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from market_jepa.modeling import TFC, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.modeling.modes.tfc import nt_xent_poly


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


def _tiny_tfc(**kwargs) -> TFC:
    kwargs.setdefault("proj_dim", 16)
    kwargs.setdefault("proj_hidden", 32)
    return TFC(backbone=_tiny_backbone(), **kwargs)


def test_tfc_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        TFC(backbone=torch.nn.Linear(8, 8))


def test_tfc_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        _tiny_tfc(temperature=0.0)


def test_nt_xent_poly_prefers_aligned_pairs():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    loss_aligned = nt_xent_poly(z, z.clone())
    loss_random = nt_xent_poly(z, torch.randn(8, 16))
    assert loss_aligned < loss_random


def test_encode_is_concat_of_projections():
    model = _tiny_tfc().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    assert out["embeddings"].shape == (4, 1, 2 * model.proj_dim)
    assert model.d_embedding == 2 * model.proj_dim


def test_forward_loss_finite():
    torch.manual_seed(0)
    model = _tiny_tfc().eval()
    x = torch.randn(4, 9, 128)
    out = model(x)
    assert out is not None
    for key in ("tfc_loss", "tfc_loss_t", "tfc_loss_f", "tfc_loss_tf"):
        assert torch.isfinite(out[key]), key


def test_forward_returns_none_for_singleton_batch():
    model = _tiny_tfc().eval()
    assert model(torch.randn(1, 9, 128)) is None


def test_freq_branch_has_independent_weights():
    model = _tiny_tfc()
    p_t = next(model.backbone.parameters())
    p_f = next(model.freq_backbone.parameters())
    assert p_t.shape == p_f.shape
    assert not torch.allclose(p_t, p_f)


def test_training_step_returns_loss_and_metrics():
    model = _tiny_tfc()
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
    model = _tiny_tfc().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
        loaded = TFC.from_pretrained(tmp).eval()

    assert cfg["class"] == "TFC"
    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]
    assert torch.allclose(emb_before, emb_after, atol=1e-5)
