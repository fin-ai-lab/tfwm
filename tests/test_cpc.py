"""Smoke tests for the CPC mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from market_jepa.modeling import CPC, create_backbone
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


def _tiny_cpc(n_predictions: int = 4, min_context_frac: float = 0.25) -> CPC:
    return CPC(
        backbone=_tiny_backbone(),
        gru_hidden_size=32,
        gru_num_layers=1,
        n_predictions=n_predictions,
        min_context_frac=min_context_frac,
        temperature=1.0,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_cpc_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        CPC(backbone=torch.nn.Linear(8, 8))


def test_cpc_rejects_cls_pool():
    with pytest.raises(ValueError, match="pool"):
        CPC(backbone=_tiny_backbone(pool="cls"))


def test_cpc_rejects_bad_hyperparams():
    with pytest.raises(ValueError, match="n_predictions"):
        CPC(backbone=_tiny_backbone(), n_predictions=0)
    with pytest.raises(ValueError, match="min_context_frac"):
        CPC(backbone=_tiny_backbone(), min_context_frac=0.0)
    with pytest.raises(ValueError, match="min_context_frac"):
        CPC(backbone=_tiny_backbone(), min_context_frac=1.0)
    with pytest.raises(ValueError, match="temperature"):
        CPC(backbone=_tiny_backbone(), temperature=0.0)


# ---------------------------------------------------------------------------
# Shape + encode + forward
# ---------------------------------------------------------------------------


def test_encode_shape_matches_other_modes():
    """encode() must return (B, 1, d_embedding) for downstream probe eval."""
    model = _tiny_cpc().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    assert "embeddings" in out
    assert out["embeddings"].shape == (4, 1, model.d_embedding)


def test_forward_loss_finite():
    model = _tiny_cpc().eval()
    x = torch.randn(4, 9, 128)
    out = model(x)
    assert out is not None
    loss = out["cpc_loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_forward_prediction_shapes():
    """Predictions stacked as (B, K, z_dim); context is (B, gru_hidden_size)."""
    K = 5
    model = _tiny_cpc(n_predictions=K).eval()
    B, T = 3, 128
    x = torch.randn(B, 9, T)
    out = model(x)
    assert out is not None
    H = model.backbone.config.hidden_size
    assert out["_predictions"].shape == (B, K, H)
    assert out["_c"].shape == (B, model.gru_hidden_size)
    # t_c must leave room for K targets
    assert out["t_c"] + K <= out["_z"].shape[1]


def test_forward_returns_none_when_sequence_too_short():
    """If even one sample has fewer than ~K patches the batch is skipped."""
    # patch_size=8, K=8 → need min_valid >= ~10 patches = 80 timesteps.
    # Set one sample to have only 32 timesteps (4 patches) — should return None.
    model = _tiny_cpc(n_predictions=8, min_context_frac=0.25).eval()
    x = torch.randn(3, 9, 128)
    lengths = torch.tensor([128, 128, 32], dtype=torch.long)
    assert model(x, lengths=lengths) is None


# ---------------------------------------------------------------------------
# InfoNCE semantics
# ---------------------------------------------------------------------------


def test_infonce_logits_use_in_batch_negatives():
    """At each predicted step, the logit matrix is (B, B) and the positive
    lies on the diagonal (label = anchor index)."""
    model = _tiny_cpc(n_predictions=3).eval()
    B, T = 6, 128
    x = torch.randn(B, 9, T)
    out = model(x)
    assert out is not None
    z = out["_z"]
    preds = out["_predictions"]  # (B, K, H)
    t_c = out["t_c"]
    K = model.n_predictions
    for k in range(K):
        pred_k = preds[:, k, :]
        pos_k = z[:, t_c + k, :]
        logits = pred_k @ pos_k.t() / model.temperature  # (B, B)
        # Diagonal should be exactly <pred_i, pos_i>
        diag = torch.einsum("bh,bh->b", pred_k, pos_k) / model.temperature
        assert torch.allclose(logits.diagonal(), diag, atol=1e-5)


# ---------------------------------------------------------------------------
# Training step integration
# ---------------------------------------------------------------------------


def test_training_step_returns_loss_and_metrics():
    model = _tiny_cpc()
    model.train()
    device = torch.device("cpu")
    batch = {
        "buckets": [
            {
                "views": [torch.randn(4, 9, 128)],
                "lengths": [torch.full((4,), 128, dtype=torch.long)],
            }
        ]
    }
    out = model.training_step(batch, device)
    assert out is not None
    assert "loss" in out
    assert "metrics" in out
    assert "train/loss" in out["metrics"]
    assert "train/cpc_acc" in out["metrics"]


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = _tiny_cpc(n_predictions=3).eval()
    x = torch.randn(2, 9, 128)

    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = CPC.from_pretrained(tmp).eval()

    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]

    assert torch.allclose(emb_before, emb_after, atol=1e-5)
    assert loaded.n_predictions == 3


def test_save_pretrained_writes_class_discriminator():
    import json
    import os

    model = _tiny_cpc()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
    assert cfg["class"] == "CPC"
    assert cfg["n_predictions"] == 4