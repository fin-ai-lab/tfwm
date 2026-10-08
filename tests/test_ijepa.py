"""Tests for I-JEPA: masking, predictor, model forward/encode, and forward_patches."""

import pytest
import torch

from market_jepa.modeling import (
    IJEPA,
    IJEPAPredictor,
    TransformerBackbone,
    TransformerConfig,
    create_backbone,
)
from market_jepa.modeling.modes.ijepa import _sample_1d_block_masks, _sample_block_sizes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**overrides) -> TransformerConfig:
    defaults = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        patch_size=4,
        drop_path_rate=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_backbone(pool="mean", d_embedding=16, **config_kw) -> TransformerBackbone:
    cfg = _small_config(**config_kw)
    return create_backbone(
        "transformer", n_features=9, d_embedding=d_embedding, config=cfg, pool=pool
    )


def _make_ijepa(pool="mean", d_embedding=16, **model_kw) -> IJEPA:
    backbone = _make_backbone(pool=pool, d_embedding=d_embedding)
    defaults = dict(pred_depth=2, pred_emb_dim=32, n_targets=2, target_scale=[0.15, 0.3])
    defaults.update(model_kw)
    return IJEPA(backbone=backbone, **defaults)


# ---------------------------------------------------------------------------
# 1. _sample_1d_block_masks
# ---------------------------------------------------------------------------


class TestSample1DBlockMasks:
    def test_returns_correct_types(self):
        ctx, tgt_blocks = _sample_1d_block_masks(n_patches=64, n_targets=4)
        assert isinstance(ctx, torch.Tensor)
        assert isinstance(tgt_blocks, list)
        assert all(isinstance(t, torch.Tensor) for t in tgt_blocks)

    def test_correct_number_of_target_blocks(self):
        for n_targets in [1, 4, 8]:
            _, tgt_blocks = _sample_1d_block_masks(n_patches=100, n_targets=n_targets)
            assert len(tgt_blocks) == n_targets

    def test_context_does_not_overlap_targets(self):
        ctx, tgt_blocks = _sample_1d_block_masks(
            n_patches=64, n_targets=4, context_crop_max=0.0
        )
        ctx_set = set(ctx.tolist())
        for tgt in tgt_blocks:
            tgt_set = set(tgt.tolist())
            assert ctx_set.isdisjoint(tgt_set), "Context overlaps with a target block"

    def test_all_indices_in_range(self):
        n_patches = 50
        ctx, tgt_blocks = _sample_1d_block_masks(n_patches=n_patches, n_targets=4)
        assert ctx.min() >= 0
        assert ctx.max() < n_patches
        for tgt in tgt_blocks:
            assert tgt.min() >= 0
            assert tgt.max() < n_patches

    def test_target_blocks_are_contiguous(self):
        _, tgt_blocks = _sample_1d_block_masks(n_patches=100, n_targets=4)
        for tgt in tgt_blocks:
            diffs = tgt[1:] - tgt[:-1]
            assert (diffs == 1).all(), "Target block indices are not contiguous"

    def test_context_is_sorted(self):
        ctx, _ = _sample_1d_block_masks(n_patches=100, n_targets=4)
        assert (ctx[1:] >= ctx[:-1]).all(), "Context indices are not sorted"

    def test_context_crop_reduces_context(self):
        """With max crop, context should generally be smaller than without."""
        torch.manual_seed(0)
        sizes_no_crop = []
        sizes_with_crop = []
        for _ in range(50):
            ctx_no, _ = _sample_1d_block_masks(
                n_patches=100, n_targets=2, context_crop_max=0.0
            )
            ctx_yes, _ = _sample_1d_block_masks(
                n_patches=100, n_targets=2, context_crop_max=0.15
            )
            sizes_no_crop.append(len(ctx_no))
            sizes_with_crop.append(len(ctx_yes))
        # On average, cropping should reduce context size
        assert sum(sizes_with_crop) < sum(sizes_no_crop)

    def test_small_n_patches(self):
        """Should work even with very few patches."""
        ctx, tgt_blocks = _sample_1d_block_masks(n_patches=3, n_targets=2, target_scale=(0.1, 0.5))
        assert len(tgt_blocks) == 2
        for tgt in tgt_blocks:
            assert len(tgt) >= 1


# ---------------------------------------------------------------------------
# 2. IJEPAPredictor
# ---------------------------------------------------------------------------


class TestIJEPAPredictor:
    def test_output_shape(self):
        B, N_ctx, N_tgt, hidden = 4, 20, 8, 32
        pred = IJEPAPredictor(
            hidden_size=hidden, pred_emb_dim=16, depth=2, num_heads=4, max_patches=64
        )
        ctx_tokens = torch.randn(B, N_ctx, hidden)
        ctx_indices = torch.arange(N_ctx).unsqueeze(0).expand(B, -1)
        tgt_indices = torch.arange(N_ctx, N_ctx + N_tgt).unsqueeze(0).expand(B, -1)
        out = pred(ctx_tokens, ctx_indices, tgt_indices)
        assert out.shape == (B, N_tgt, hidden)

    def test_gradient_flows(self):
        B, N_ctx, N_tgt, hidden = 2, 10, 5, 32
        pred = IJEPAPredictor(
            hidden_size=hidden, pred_emb_dim=16, depth=2, num_heads=4, max_patches=32
        )
        ctx_tokens = torch.randn(B, N_ctx, hidden, requires_grad=True)
        ctx_indices = torch.arange(N_ctx).unsqueeze(0).expand(B, -1)
        tgt_indices = torch.arange(N_ctx, N_ctx + N_tgt).unsqueeze(0).expand(B, -1)
        out = pred(ctx_tokens, ctx_indices, tgt_indices)
        out.sum().backward()
        assert ctx_tokens.grad is not None
        assert ctx_tokens.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. TransformerBackbone.forward_patches
# ---------------------------------------------------------------------------


class TestForwardPatches:
    def test_output_shape_no_mask(self):
        backbone = _make_backbone(pool="mean")
        x = torch.randn(2, 9, 32)  # 32 / patch_size=4 = 8 patches
        out = backbone.forward_patches(x)
        assert out.shape == (2, 8, 32)  # (B, n_patches, hidden_size)

    def test_output_shape_with_mask(self):
        backbone = _make_backbone(pool="mean")
        x = torch.randn(2, 9, 32)
        mask = [torch.tensor([0, 2, 4, 6])] * 2  # keep 4 of 8 patches
        out = backbone.forward_patches(x, mask_indices=mask)
        assert out.shape == (2, 4, 32)

    def test_masked_subset_of_full(self):
        """Masked patches should use same position embeddings as full."""
        backbone = _make_backbone(pool="mean", drop_path_rate=0.0)
        backbone.eval()
        x = torch.randn(1, 9, 32)

        full = backbone.forward_patches(x)  # (1, 8, hidden)
        kept = [0, 3, 7]
        mask = [torch.tensor(kept)]
        partial = backbone.forward_patches(x, mask_indices=mask)  # (1, 3, hidden)

        # Without attention interactions, results won't match exactly
        # but shapes should be correct
        assert partial.shape == (1, len(kept), 32)

    def test_with_lengths(self):
        backbone = _make_backbone(pool="mean")
        x = torch.randn(2, 9, 32)
        lengths = torch.tensor([16, 32])  # 4 and 8 patches
        out = backbone.forward_patches(x, lengths=lengths)
        assert out.shape == (2, 8, 32)

    def test_cls_pool_backbone_works_in_forward(self):
        """Regular forward should still work with cls pooling."""
        backbone = _make_backbone(pool="cls", d_embedding=16)
        x = torch.randn(2, 9, 32)
        out = backbone(x)
        assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# 4. IJEPA model
# ---------------------------------------------------------------------------


class TestIJEPA:
    def test_forward_produces_finite_loss(self):
        model = _make_ijepa()
        x = torch.randn(4, 9, 64)
        lengths = torch.full((4,), 64)
        out = model(x, lengths)
        assert "ijepa_loss" in out
        assert torch.isfinite(out["ijepa_loss"])

    def test_forward_without_lengths(self):
        model = _make_ijepa()
        x = torch.randn(4, 9, 64)
        out = model(x)
        assert torch.isfinite(out["ijepa_loss"])

    def test_backward(self):
        model = _make_ijepa()
        model.train()
        x = torch.randn(4, 9, 64)
        out = model(x)
        out["ijepa_loss"].backward()
        # Check gradients flow to backbone
        grad_norms = [p.grad.abs().sum() for p in model.backbone.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert sum(grad_norms) > 0

    def test_ema_target_encoder_is_frozen(self):
        model = _make_ijepa()
        for p in model.ema.ema_model.parameters():
            assert not p.requires_grad

    def test_post_training_step_returns_ema_momentum(self):
        model = _make_ijepa()
        metrics = model.post_training_step(0, 1000)
        assert "train/ema_momentum" in metrics

    def test_rejects_cls_pooling(self):
        backbone = _make_backbone(pool="cls")
        with pytest.raises(ValueError, match="cls"):
            IJEPA(backbone=backbone)

    def test_describe_parameters(self):
        model = _make_ijepa()
        counts, summary = model.describe_parameters()
        assert "total" in counts
        assert "backbone" in counts
        assert "predictor" in counts
        assert counts["total"] > 0
        assert isinstance(summary, str)

    def test_mode_metadata(self):
        model = _make_ijepa()
        assert model.mode_label == "I-JEPA"
        assert model.uses_multi_view is False


# ---------------------------------------------------------------------------
# 5. IJEPA.encode (for probe eval)
# ---------------------------------------------------------------------------


class TestIJEPAEncode:
    def test_encode_single_view(self):
        model = _make_ijepa(d_embedding=16)
        model.eval()
        x = torch.randn(4, 9, 64)
        with torch.no_grad():
            out = model.encode(x)
        assert "embeddings" in out
        assert out["embeddings"].shape == (4, 1, 16)

    def test_encode_multi_view_tensor(self):
        model = _make_ijepa(d_embedding=16)
        model.eval()
        x = torch.randn(4, 3, 9, 64)  # 3 views
        with torch.no_grad():
            out = model.encode(x)
        assert out["embeddings"].shape == (4, 3, 16)

    def test_encode_multi_view_list(self):
        model = _make_ijepa(d_embedding=16)
        model.eval()
        views = [torch.randn(4, 9, 64), torch.randn(4, 9, 32)]
        with torch.no_grad():
            out = model.encode(views)
        assert out["embeddings"].shape == (4, 2, 16)


# ---------------------------------------------------------------------------
# 6. IJEPA.training_step / eval_step
# ---------------------------------------------------------------------------


class TestIJEPATrainingStep:
    def _make_batch(self, batch_size=4, n_features=9, seq_len=64):
        views = [torch.randn(batch_size, n_features, seq_len)]
        lengths = [torch.full((batch_size,), seq_len)]
        return {"buckets": [{"views": views, "lengths": lengths}]}

    def test_training_step_returns_loss(self):
        model = _make_ijepa()
        model.train()
        batch = self._make_batch()
        result = model.training_step(batch, torch.device("cpu"))
        assert result is not None
        assert "loss" in result
        assert "metrics" in result
        assert "train/loss" in result["metrics"]

    def test_training_step_accumulates_gradients(self):
        model = _make_ijepa()
        model.train()
        batch = self._make_batch()
        model.zero_grad()
        result = model.training_step(batch, torch.device("cpu"), grad_accum_steps=4)
        assert result is not None
        # Gradients should exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_grad_accum_scales_gradients(self):
        """Gradient with accum=2 should be exactly half of accum=1 for same input.

        Seeds torch RNG before each forward pass so the stochastic masking
        produces identical masks, isolating the effect of loss scaling.
        """
        torch.manual_seed(42)
        model = _make_ijepa()
        model.train()
        batch = self._make_batch()

        # Run with accum=1
        model.zero_grad()
        torch.manual_seed(99)  # seed masking RNG
        model.training_step(batch, torch.device("cpu"), grad_accum_steps=1)
        grads_1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        # Run with accum=2 using same masking seed
        model.zero_grad()
        torch.manual_seed(99)  # same masking RNG
        model.training_step(batch, torch.device("cpu"), grad_accum_steps=2)
        grads_2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        # grads_2 should be exactly half of grads_1
        for g1, g2 in zip(grads_1, grads_2):
            torch.testing.assert_close(g2, g1 / 2, rtol=1e-4, atol=1e-6)

    def test_eval_step(self):
        model = _make_ijepa()
        batch = self._make_batch()
        metrics = model.eval_step([batch], torch.device("cpu"))
        assert "eval/ijepa_loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["eval/ijepa_loss"]))


# ---------------------------------------------------------------------------
# 7. Per-sample masking (Fix #1)
# ---------------------------------------------------------------------------


class TestPerSampleMasking:
    def test_block_sizes_shared_positions_differ(self):
        """Block sizes should be uniform but positions should differ across samples."""
        torch.manual_seed(42)
        n_patches = 64
        block_sizes = _sample_block_sizes(n_patches, n_targets=4)

        masks = [
            _sample_1d_block_masks(n_patches, block_sizes=block_sizes)
            for _ in range(10)
        ]

        # All should have same block sizes
        for _, tgt_blocks in masks:
            for i, tgt in enumerate(tgt_blocks):
                assert len(tgt) == block_sizes[i]

        # Positions should differ across at least some samples
        starts = [tuple(tgt[0].item() for tgt in m[1]) for m in masks]
        assert len(set(starts)) > 1, "All samples got identical mask positions"

    def test_forward_per_sample_masking(self):
        """Forward pass should work with per-sample masking."""
        model = _make_ijepa()
        x = torch.randn(4, 9, 64)
        out = model(x)
        assert torch.isfinite(out["ijepa_loss"])


# ---------------------------------------------------------------------------
# 8. Test gaps: backward checks predictor gradients
# ---------------------------------------------------------------------------


class TestBackwardGradientFlow:
    def test_backward_predictor_has_gradients(self):
        """Verify gradients flow to the predictor (not just backbone)."""
        model = _make_ijepa()
        model.train()
        x = torch.randn(4, 9, 64)
        out = model(x)
        out["ijepa_loss"].backward()
        pred_grads = [
            p.grad.abs().sum()
            for p in model.predictor.parameters()
            if p.grad is not None
        ]
        assert len(pred_grads) > 0, "No predictor parameters received gradients"
        assert sum(pred_grads) > 0, "All predictor gradients are zero"


# ---------------------------------------------------------------------------
# 9. Test gap: EMA params actually move after post_training_step
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_ema_params_move_after_post_training_step(self):
        """EMA target encoder params should change after update."""
        model = _make_ijepa()
        model.train()

        # Snapshot EMA params before
        ema_before = {
            name: p.clone()
            for name, p in model.ema.ema_model.named_parameters()
        }

        # Do a forward + backward to change the online model
        x = torch.randn(4, 9, 64)
        out = model(x)
        out["ijepa_loss"].backward()

        # Simulate an optimizer step on the online model
        with torch.no_grad():
            for p in model.backbone.parameters():
                if p.grad is not None:
                    p.add_(-0.01 * p.grad)

        # Run EMA update
        model.post_training_step(0, 1000)

        # Check that at least some EMA params changed
        changed = False
        for name, p in model.ema.ema_model.named_parameters():
            if not torch.equal(p, ema_before[name]):
                changed = True
                break
        assert changed, "EMA target encoder params did not change after update"

    def test_ema_momentum_is_linear(self):
        """Verify the momentum schedule is linear from ema_start to ema_end."""
        model = _make_ijepa(ema_start=0.9, ema_end=1.0)
        max_steps = 100

        # Check at various points
        metrics_0 = model.post_training_step(0, max_steps)
        metrics_half = model.post_training_step(max_steps // 2, max_steps)
        metrics_end = model.post_training_step(max_steps - 1, max_steps)

        # Should be approximately linear
        assert abs(metrics_0["train/ema_momentum"] - 0.9) < 1e-6
        assert abs(metrics_half["train/ema_momentum"] - 0.95) < 0.01
        assert abs(metrics_end["train/ema_momentum"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 10. Test gap: masked_subset_of_full semantic validation
# ---------------------------------------------------------------------------


class TestMaskedSubsetSemantic:
    def test_masked_positions_use_correct_embeddings(self):
        """Verify masked forward uses same position embeddings as full.

        With a single patch kept and no attention interactions (depth=0-like),
        the position embedding at that index should match.
        """
        cfg = _small_config(num_hidden_layers=0, drop_path_rate=0.0)
        backbone = create_backbone(
            "transformer", n_features=9, d_embedding=16, config=cfg, pool="mean"
        )
        backbone.eval()

        x = torch.randn(1, 9, 32)  # 8 patches

        with torch.no_grad():
            full = backbone.forward_patches(x)  # (1, 8, hidden)
            for kept_idx in [0, 3, 7]:
                mask = [torch.tensor([kept_idx])]
                partial = backbone.forward_patches(x, mask_indices=mask)
                # With 0 transformer layers, output = layernorm(patch_embed + pos_embed)
                # The kept patch should match the corresponding full output
                torch.testing.assert_close(
                    partial[0, 0], full[0, kept_idx],
                    rtol=1e-4, atol=1e-5,
                    msg=f"Mismatch at patch index {kept_idx}",
                )


# ---------------------------------------------------------------------------
# 11. Test gap: loss_fn="smooth_l1" variant
# ---------------------------------------------------------------------------


class TestSmoothL1Loss:
    def test_forward_with_smooth_l1(self):
        model = _make_ijepa(loss_fn="smooth_l1")
        x = torch.randn(4, 9, 64)
        out = model(x)
        assert torch.isfinite(out["ijepa_loss"])

    def test_backward_with_smooth_l1(self):
        model = _make_ijepa(loss_fn="smooth_l1")
        model.train()
        x = torch.randn(4, 9, 64)
        out = model(x)
        out["ijepa_loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad


# ---------------------------------------------------------------------------
# 12. Test gap: gradient_checkpointing=True
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    def test_forward_with_gradient_checkpointing(self):
        model = _make_ijepa(gradient_checkpointing=True)
        model.train()
        x = torch.randn(4, 9, 64)
        out = model(x)
        assert torch.isfinite(out["ijepa_loss"])

    def test_backward_with_gradient_checkpointing(self):
        model = _make_ijepa(gradient_checkpointing=True)
        model.train()
        x = torch.randn(4, 9, 64)
        out = model(x)
        out["ijepa_loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# 13. Test gap: non-finite loss handling (None return path)
# ---------------------------------------------------------------------------


class TestNonFiniteLossHandling:
    def test_training_step_returns_none_on_nonfinite(self):
        """If the model produces non-finite loss, training_step returns None."""
        model = _make_ijepa()
        model.train()

        # Create a batch with extreme values that will produce NaN/inf loss
        batch_size, n_features, seq_len = 4, 9, 64
        x = torch.full((batch_size, n_features, seq_len), float("inf"))
        views = [x]
        lengths = [torch.full((batch_size,), seq_len)]
        batch = {"buckets": [{"views": views, "lengths": lengths}]}

        result = model.training_step(batch, torch.device("cpu"))
        # Should return None because loss is non-finite
        assert result is None


# ---------------------------------------------------------------------------
# 14. Test: encode uses target encoder (Fix #2)
# ---------------------------------------------------------------------------


class TestEncodeUsesTargetEncoder:
    def test_encode_uses_ema_model(self):
        """IJEPA.encode() should use the EMA target encoder, not the context encoder."""
        model = _make_ijepa(d_embedding=16)

        # Make EMA model different from online model by modifying online weights
        with torch.no_grad():
            for p in model.backbone.parameters():
                p.add_(10.0)

        model.eval()
        x = torch.randn(2, 9, 64)

        with torch.no_grad():
            ijepa_emb = model.encode(x)["embeddings"][:, 0, :]
            # Direct call to EMA model
            ema_emb = model.ema.ema_model(x)
            # Direct call to online backbone
            online_emb = model.backbone(x)

        # encode() should match EMA, not online
        torch.testing.assert_close(ijepa_emb, ema_emb, rtol=1e-4, atol=1e-5)
        assert not torch.allclose(ijepa_emb, online_emb, atol=0.1)
