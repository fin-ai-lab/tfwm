"""Smoke tests for the MAE mode.

Uses tiny synthetic batches on CPU — no real data, no autocast, no W&B.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from market_jepa.modeling import MAE, create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.modeling.modes.mae import _mae_random_masking, _patchify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _tiny_mae(mask_ratio: float = 0.75) -> MAE:
    return MAE(
        backbone=_tiny_backbone(),
        decoder_embed_dim=32,
        decoder_depth=2,
        decoder_num_heads=4,
        mask_ratio=mask_ratio,
        norm_pix_loss=True,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_mae_requires_transformer_backbone():
    with pytest.raises(ValueError, match="TransformerBackbone"):
        MAE(backbone=torch.nn.Linear(8, 8))


def test_mae_rejects_cls_pool():
    bb = _tiny_backbone(pool="cls")
    with pytest.raises(ValueError, match="pool"):
        MAE(backbone=bb)


def test_mae_rejects_bad_mask_ratio():
    bb = _tiny_backbone()
    with pytest.raises(ValueError, match="mask_ratio"):
        MAE(backbone=bb, mask_ratio=0.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        MAE(backbone=_tiny_backbone(), mask_ratio=1.0)


# ---------------------------------------------------------------------------
# Shape + encode
# ---------------------------------------------------------------------------


def test_encode_shape_matches_other_modes():
    """encode() must return (B, 1, d_embedding) so downstream probe eval and
    future MAE+LeJEPA concatenation work unchanged."""
    model = _tiny_mae().eval()
    x = torch.randn(4, 9, 128)
    out = model.encode(x)
    assert "embeddings" in out
    assert out["embeddings"].shape == (4, 1, model.d_embedding)


def test_forward_loss_finite():
    model = _tiny_mae()
    x = torch.randn(4, 9, 128)
    out = model(x)
    loss = out["mae_loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Masking semantics
# ---------------------------------------------------------------------------


def test_mask_coverage_matches_ratio():
    """With no padding, the fraction of masked patches should match mask_ratio."""
    B, n_patches = 64, 16
    n_valid = torch.full((B,), n_patches, dtype=torch.long)
    _, _, mask, _ = _mae_random_masking(n_valid, n_patches, mask_ratio=0.75, device="cpu")
    frac_masked = mask.mean().item()
    # With mask_ratio=0.75 and n_patches=16, len_keep = int(16*0.25) = 4,
    # so exactly 12/16 = 0.75 are masked.
    assert abs(frac_masked - 0.75) < 1e-6


def test_masking_respects_padding():
    """Padded patches must never appear in ids_keep."""
    B, n_patches = 8, 16
    # Samples with 4, 8, 12, 16 valid patches respectively
    n_valid = torch.tensor([4, 8, 12, 16, 4, 8, 12, 16], dtype=torch.long)
    ids_keep, _, _, valid_mask = _mae_random_masking(
        n_valid, n_patches, mask_ratio=0.75, device="cpu"
    )
    for b in range(B):
        # ids_keep for sample b must all be < n_valid[b]
        assert (ids_keep[b] < n_valid[b]).all(), (
            f"Sample {b}: ids_keep={ids_keep[b].tolist()} but n_valid={n_valid[b].item()}"
        )
        # valid_mask must be True exactly on positions < n_valid[b]
        expected = torch.arange(n_patches) < n_valid[b]
        assert torch.equal(valid_mask[b], expected)


def test_len_keep_uses_min_valid():
    """len_keep is clamped to min(n_valid) * (1 - mask_ratio) for rectangular batching."""
    n_valid = torch.tensor([4, 16, 16], dtype=torch.long)
    ids_keep, _, _, _ = _mae_random_masking(n_valid, 16, mask_ratio=0.75, device="cpu")
    # min_valid = 4, len_keep = int(4 * 0.25) = 1, clamped to max(1, ...) = 1
    assert ids_keep.shape == (3, 1)


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------


def test_patchify_roundtrip_shape():
    x = torch.randn(2, 9, 64)
    patched = _patchify(x, patch_size=8, n_patches=8)
    assert patched.shape == (2, 8, 9 * 8)


def test_patchify_pads_short_sequences():
    x = torch.randn(2, 9, 60)  # not divisible by 8
    patched = _patchify(x, patch_size=8, n_patches=8)  # 8*8=64 > 60
    assert patched.shape == (2, 8, 9 * 8)


# ---------------------------------------------------------------------------
# Padding handling end-to-end
# ---------------------------------------------------------------------------


def test_forward_with_padding_skips_padded_loss():
    """When a sample is mostly padding, the reconstruction loss weight on its
    padded patches should be zero."""
    model = _tiny_mae().eval()
    B = 4
    T = 128
    x = torch.randn(B, 9, T)
    # One sample has very short length
    lengths = torch.tensor([T, T // 2, T // 4, T // 8], dtype=torch.long)
    out = model(x, lengths=lengths)
    # Padded positions in valid_mask must be False
    for b in range(B):
        n_valid = (lengths[b] + model.patch_size - 1) // model.patch_size
        assert (out["valid_mask"][b, :n_valid]).all()
        assert not (out["valid_mask"][b, n_valid:]).any()


# ---------------------------------------------------------------------------
# Training step integration
# ---------------------------------------------------------------------------


def test_training_step_returns_loss_and_metrics():
    """training_step must return {'loss': float, 'metrics': dict}."""
    model = _tiny_mae()
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
    # Disable autocast for CPU test by monkey-patching (forward uses autocast
    # but autocast on CPU with bfloat16 is a no-op in practice).
    out = model.training_step(batch, device)
    assert out is not None
    assert "loss" in out
    assert "metrics" in out
    assert "train/loss" in out["metrics"]
    assert "train/mae_valid_frac" in out["metrics"]


def test_overfit_one_batch_loss_decreases():
    """Overfitting a fixed small batch should drive reconstruction loss down."""
    torch.manual_seed(0)
    model = _tiny_mae(mask_ratio=0.5)  # less aggressive so it learns faster
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 9, 128)
    lengths = torch.full((4,), 128, dtype=torch.long)

    initial_losses = []
    for _ in range(5):
        out = model(x, lengths)
        initial_losses.append(out["mae_loss"].item())
    initial = float(np.mean(initial_losses))

    for _ in range(200):
        opt.zero_grad()
        loss = model(x, lengths)["mae_loss"]
        loss.backward()
        opt.step()

    final_losses = []
    with torch.no_grad():
        for _ in range(5):
            final_losses.append(model(x, lengths)["mae_loss"].item())
    final = float(np.mean(final_losses))

    # The stochastic mask means we can't demand monotonic decrease, but after
    # 200 steps on a 4-sample batch the model should cut loss significantly.
    assert final < 0.5 * initial, f"final={final:.3f} >= 0.5*initial={0.5*initial:.3f}"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = _tiny_mae(mask_ratio=0.6).eval()
    x = torch.randn(2, 9, 128)

    with torch.no_grad():
        emb_before = model.encode(x)["embeddings"]

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = MAE.from_pretrained(tmp).eval()

    with torch.no_grad():
        emb_after = loaded.encode(x)["embeddings"]

    assert torch.allclose(emb_before, emb_after, atol=1e-5)
    assert loaded.mask_ratio == 0.6


def test_save_pretrained_writes_class_discriminator():
    """Offline eval dispatches on config['class']; make sure we write it."""
    import json
    import os

    model = _tiny_mae()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "config.json")) as f:
            cfg = json.load(f)
    assert cfg["class"] == "MAE"
    assert cfg["mask_ratio"] == 0.75
