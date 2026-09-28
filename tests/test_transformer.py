"""Tests for the vanilla PyTorch Transformer backbone (TransformerConfig, ViTBlock, TransformerBackbone)."""

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from market_jepa.modeling import (
    PatchEmbedding1D,
    LeJEPA,
    TransformerBackbone,
    TransformerConfig,
    ViTBlock,
    create_backbone,
)


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
        drop_path_rate=0.1,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_input(batch: int, n_features: int, length: int) -> torch.Tensor:
    return torch.randn(batch, n_features, length)


# ---------------------------------------------------------------------------
# 1. TestTransformerConfig
# ---------------------------------------------------------------------------


class TestTransformerConfig:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TransformerConfig)
        names = {f.name for f in dataclasses.fields(TransformerConfig)}
        for expected in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "patch_size",
            "drop_path_rate",
        ):
            assert expected in names

    def test_defaults(self):
        cfg = TransformerConfig()
        assert cfg.hidden_size == 384
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 6
        assert cfg.intermediate_size == 1536
        assert cfg.patch_size == 8
        assert cfg.drop_path_rate == pytest.approx(0.1)

    def test_custom_values(self):
        cfg = TransformerConfig(hidden_size=64, num_hidden_layers=4)
        assert cfg.hidden_size == 64
        assert cfg.num_hidden_layers == 4


# ---------------------------------------------------------------------------
# 2. TestPatchEmbedding1D
# ---------------------------------------------------------------------------


class TestPatchEmbedding1D:
    def test_output_shape_exact_division(self):
        pe = PatchEmbedding1D(n_features=3, patch_size=4, d_model=16)
        out = pe(_make_input(2, 3, 16))
        assert out.shape == (2, 4, 16)

    def test_output_shape_with_padding(self):
        pe = PatchEmbedding1D(n_features=3, patch_size=4, d_model=16)
        out = pe(_make_input(2, 3, 17))
        assert out.shape == (2, math.ceil(17 / 4), 16)  # 5 patches

    def test_no_pad_when_exact(self):
        pe = PatchEmbedding1D(n_features=5, patch_size=4, d_model=8)
        out = pe(_make_input(1, 5, 8))
        assert out.shape == (1, 2, 8)

    @pytest.mark.parametrize("length", [1, 2, 3, 4])
    def test_single_patch(self, length):
        pe = PatchEmbedding1D(n_features=2, patch_size=4, d_model=8)
        out = pe(_make_input(1, 2, length))
        assert out.shape == (1, 1, 8)

    def test_batch_independence(self):
        pe = PatchEmbedding1D(n_features=3, patch_size=4, d_model=16)
        for bs in (1, 4, 8):
            out = pe(_make_input(bs, 3, 16))
            assert out.shape[0] == bs


# ---------------------------------------------------------------------------
# 3. TestViTBlock
# ---------------------------------------------------------------------------


class TestViTBlock:
    def test_output_shape(self):
        block = ViTBlock(hidden_size=32, num_attention_heads=4, intermediate_size=64)
        x = torch.randn(2, 10, 32)
        assert block(x).shape == x.shape

    def test_residual_connection(self):
        block = ViTBlock(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            drop_path_rate=0.0,
        )
        # Zero out attn and mlp weights AND biases so their contribution is zero
        with torch.no_grad():
            nn.init.zeros_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        x = torch.randn(2, 5, 32)
        out = block(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_key_padding_mask_isolates_positions(self):
        block = ViTBlock(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            drop_path_rate=0.0,
        )
        block.eval()
        x = torch.randn(1, 6, 32)
        # True = IGNORE in key_padding_mask; mask last 2 positions
        mask = torch.tensor([[False, False, False, False, True, True]])
        out_base = block(x, key_padding_mask=mask)

        # Alter masked positions — unmasked output should be identical
        x2 = x.clone()
        x2[:, 4:, :] = 999.0
        out_altered = block(x2, key_padding_mask=mask)
        torch.testing.assert_close(
            out_base[:, :4], out_altered[:, :4], atol=1e-5, rtol=1e-5
        )

    def test_drop_path_active_in_train(self):
        block = ViTBlock(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            drop_path_rate=1.0,  # always drop
        )
        block.train()
        x = torch.randn(2, 5, 32)
        out = block(x)
        # Both branches dropped → output == input
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_drop_path_inactive_in_eval(self):
        block = ViTBlock(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            drop_path_rate=1.0,
        )
        block.eval()
        x = torch.randn(2, 5, 32)
        out = block(x)
        # In eval, StochasticDepth is identity → branches contribute
        assert not torch.allclose(out, x, atol=1e-5)

    def test_two_independent_drop_paths(self):
        block = ViTBlock(
            hidden_size=32, num_attention_heads=4, intermediate_size=64, drop_path_rate=0.5
        )
        assert block.drop_path1 is not block.drop_path2


# ---------------------------------------------------------------------------
# 4. TestBackboneOutputShape
# ---------------------------------------------------------------------------


class TestBackboneOutputShape:
    @pytest.mark.parametrize("pool", ["cls", "mean", "max"])
    def test_shape_per_pooling_mode(self, pool):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool=pool)
        model.eval()
        out = model(_make_input(4, 3, 32))
        assert out.shape == (4, 16)

    def test_batch_size_one(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        model.eval()
        out = model(_make_input(1, 3, 32))
        assert out.shape == (1, 16)

    def test_very_short_input_one_patch(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="mean")
        model.eval()
        out = model(_make_input(2, 3, 3))  # length < patch_size
        assert out.shape == (2, 16)

    def test_variable_length_input(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        model.eval()
        for length in (8, 16, 32, 64):
            out = model(_make_input(2, 3, length))
            assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# 5. TestBackboneMasking
# ---------------------------------------------------------------------------


class TestBackboneMasking:
    def test_lengths_changes_output(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="mean")
        model.eval()
        x = torch.randn(2, 3, 32)
        out_short = model(x, lengths=torch.tensor([8, 8]))
        out_long = model(x, lengths=torch.tensor([32, 32]))
        assert not torch.allclose(out_short, out_long, atol=1e-4)

    def test_full_length_matches_no_length(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="mean")
        model.eval()
        x = torch.randn(2, 3, 32)
        out_none = model(x, lengths=None)
        out_full = model(x, lengths=torch.tensor([32, 32]))
        torch.testing.assert_close(out_none, out_full, atol=1e-5, rtol=1e-5)

    def test_cls_always_valid_in_mask(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        model.eval()
        x = torch.randn(2, 3, 32)
        out = model(x, lengths=torch.tensor([1, 1]))  # very short
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 6. TestBackbonePooling — "poison padded region" technique
# ---------------------------------------------------------------------------


class TestBackbonePooling:
    def _build(self, pool):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool=pool)
        model.eval()
        return model

    def test_mean_excludes_padded_positions(self):
        model = self._build("mean")
        x = torch.randn(2, 3, 32)
        x_poison = x.clone()
        x_poison[:, :, 16:] = 1e6  # poison padding region
        lengths = torch.tensor([16, 16])
        out_clean = model(x, lengths=lengths)
        out_poison = model(x_poison, lengths=lengths)
        torch.testing.assert_close(out_clean, out_poison, atol=1e-4, rtol=1e-4)

    def test_max_excludes_padded_positions(self):
        model = self._build("max")
        x = torch.randn(2, 3, 32)
        x_poison = x.clone()
        x_poison[:, :, 16:] = 1e6
        lengths = torch.tensor([16, 16])
        out_clean = model(x, lengths=lengths)
        out_poison = model(x_poison, lengths=lengths)
        torch.testing.assert_close(out_clean, out_poison, atol=1e-4, rtol=1e-4)

    def test_cls_not_contaminated_by_padding(self):
        model = self._build("cls")
        x = torch.randn(2, 3, 32)
        x_poison = x.clone()
        x_poison[:, :, 16:] = 1e6
        lengths = torch.tensor([16, 16])
        out_clean = model(x, lengths=lengths)
        out_poison = model(x_poison, lengths=lengths)
        torch.testing.assert_close(out_clean, out_poison, atol=1e-4, rtol=1e-4)

    def test_unknown_pool_raises(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        model.pool = "bogus"  # force invalid pool after construction
        with pytest.raises(ValueError, match="Unknown pooling"):
            model(_make_input(1, 3, 16))


# ---------------------------------------------------------------------------
# 7. TestBackboneWeightInit
# ---------------------------------------------------------------------------


class TestBackboneWeightInit:
    @pytest.fixture()
    def model(self):
        cfg = _small_config(drop_path_rate=0.0)
        return TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")

    def test_linear_weights_small_std(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                assert m.weight.std().item() < 0.06, f"{name} std too large"

    def test_layernorm_weight_ones_bias_zeros(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                torch.testing.assert_close(
                    m.weight, torch.ones_like(m.weight), atol=1e-6, rtol=0
                )
                torch.testing.assert_close(
                    m.bias, torch.zeros_like(m.bias), atol=1e-6, rtol=0
                )

    def test_mha_in_proj_weight_initialized(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
                assert m.in_proj_weight.std().item() < 0.06, f"{name} std too large"

    def test_linear_bias_zeros(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.testing.assert_close(
                    m.bias, torch.zeros_like(m.bias), atol=1e-6, rtol=0
                )


# ---------------------------------------------------------------------------
# 8. TestBackboneDeterminism
# ---------------------------------------------------------------------------


class TestBackboneDeterminism:
    def test_eval_mode_deterministic(self):
        cfg = _small_config(drop_path_rate=0.1)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        model.eval()
        x = torch.randn(2, 3, 32)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 9. TestDropPathScaling
# ---------------------------------------------------------------------------


class TestDropPathScaling:
    def test_linearly_scaled_rates(self):
        n_layers = 4
        rate = 0.2
        cfg = _small_config(num_hidden_layers=n_layers, drop_path_rate=rate)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        expected = torch.linspace(0, rate, n_layers).tolist()
        for i, block in enumerate(model.blocks):
            actual_p = block.drop_path1.p
            assert actual_p == pytest.approx(expected[i], abs=1e-6), (
                f"Block {i}: expected {expected[i]}, got {actual_p}"
            )

    def test_first_block_zero_rate(self):
        cfg = _small_config(num_hidden_layers=4, drop_path_rate=0.2)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        assert model.blocks[0].drop_path1.p == pytest.approx(0.0)

    def test_last_block_max_rate(self):
        cfg = _small_config(num_hidden_layers=4, drop_path_rate=0.2)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="cls")
        assert model.blocks[-1].drop_path1.p == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. TestPositionEmbeddings
# ---------------------------------------------------------------------------


class TestPositionEmbeddings:
    def test_cls_adds_one_extra_position(self):
        max_seq = 128
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(
            cfg, n_features=3, d_embedding=16, pool="cls", max_seq_len=max_seq
        )
        assert model.position_embeddings.shape[1] == max_seq + 1

    def test_no_cls_exact_positions(self):
        max_seq = 128
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(
            cfg, n_features=3, d_embedding=16, pool="mean", max_seq_len=max_seq
        )
        assert model.position_embeddings.shape[1] == max_seq

    def test_shorter_input_slices_correctly(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(
            cfg, n_features=3, d_embedding=16, pool="cls", max_seq_len=512
        )
        model.eval()
        # Input produces far fewer patches than max_seq_len — should not error
        out = model(_make_input(1, 3, 16))
        assert out.shape == (1, 16)


# ---------------------------------------------------------------------------
# 11. TestGradientFlow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("pool", ["cls", "mean", "max"])
    def test_all_params_get_gradients(self, pool):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool=pool)
        model.train()
        x = _make_input(2, 3, 32)
        out = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradients_with_lengths(self):
        cfg = _small_config(drop_path_rate=0.0)
        model = TransformerBackbone(cfg, n_features=3, d_embedding=16, pool="mean")
        model.train()
        x = _make_input(2, 3, 32)
        lengths = torch.tensor([16, 24])
        out = model(x, lengths=lengths)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# 12. TestCreateJepaTransformer
# ---------------------------------------------------------------------------


class TestCreateJepaTransformer:
    def test_factory_creates_model(self):
        cfg = _small_config(drop_path_rate=0.0)
        backbone = create_backbone("transformer", n_features=3, d_embedding=16, config=cfg)
        model = LeJEPA(backbone, proj_dim=8)
        x = torch.randn(2, 2, 3, 32)  # (batch, n_views, n_features, length)
        result = model(x, return_loss=True)
        assert "lejepa_loss" in result
        assert result["embeddings"].shape == (2, 2, 16)

    def test_factory_default_config(self):
        backbone = create_backbone("transformer", n_features=3, d_embedding=16)
        model = LeJEPA(backbone, proj_dim=8)
        assert model.backbone.config == TransformerConfig()

    def test_end_to_end_backward(self):
        cfg = _small_config(drop_path_rate=0.0)
        backbone = create_backbone("transformer", n_features=3, d_embedding=16, config=cfg)
        model = LeJEPA(backbone, proj_dim=8)
        model.train()
        x = torch.randn(4, 2, 3, 32)
        result = model(x, return_loss=True)
        result["lejepa_loss"].backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize("pos_embed", ["learned", "sinusoidal", "rope"])
def test_a_causal_mask_actually_blocks_the_future(pos_embed):
    """Perturbing the LAST patch must leave every earlier token untouched.

    Not a formality: the RoPE path folded the bool causal mask into SDPA with
    ``.to(dtype)``, which turns True into +1.0 -- an additive bias that mildly
    ENCOURAGES attention instead of blocking it. Causal masking was therefore
    a no-op there. It stayed hidden because the recency prior's caller always
    folded the mask into a float bias first, so the bare bool only reached
    that line once the prior was retired.

    Exact equality is the right assertion. A causal token cannot see the
    future at all, so the drift is 0, not merely small -- and "small" is what
    the bug produced (2.1e-03).
    """
    import torch

    from market_jepa.modeling.backbones.transformer import (
        TransformerBackbone, TransformerConfig)

    torch.manual_seed(0)
    cfg = TransformerConfig(pos_embed=pos_embed, hidden_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128)
    bb = TransformerBackbone(cfg, pool="mean", causal=True, n_features=9,
                             max_seq_len=256).eval()
    x = torch.randn(1, 9, 2048)
    y = x.clone()
    y[:, :, -cfg.patch_size:] += 10.0          # the final patch only
    lengths = torch.tensor([2048])
    with torch.no_grad():
        a = bb.forward_patches(x, lengths, causal=True)
        b = bb.forward_patches(y, lengths, causal=True)
    drift = (a - b).abs()
    assert torch.equal(drift[:, :-1], torch.zeros_like(drift[:, :-1])), (
        f"{pos_embed}: an earlier token moved when only the last patch "
        f"changed (max {drift[:, :-1].max():.3e}) — the causal mask is not "
        "blocking the future")
    assert drift[:, -1].max() > 1e-3, "the last token should see the change"
