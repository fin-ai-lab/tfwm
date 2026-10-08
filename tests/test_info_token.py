"""One information token for the facts that belong to the window, not the clock.

The per-view (mu, sigma) used to be broadcast along all 2048 timesteps and
concatenated onto the features, so 8 constants were re-read 256 times and
n_features moved 9 -> 17. The compute was negligible; the coupling was not,
because n_features shapes the checkpoint's first projection, so a conditioning
flag invalidated every checkpoint. These pin the replacement.
"""
import pytest
import torch

from market_jepa.eval.checkpoints import _widths_from_state_dict
from market_jepa.modeling.backbones import create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig

N_FEAT, N_INFO, T = 17, 8, 2048


def build(n_info=N_INFO, pool="cls", causal=False, **kw):
    return create_backbone(
        backbone_type="transformer", n_features=N_FEAT, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        max_seq_len=T, pool=pool, causal=causal, state_token=True,
        **({"n_info_channels": n_info} if n_info else {}), **kw)


def _x():
    x = torch.randn(2, N_FEAT, T)
    x[:, N_FEAT - N_INFO:, :-1] = 0
    return x


def _emb(bb, x):
    o = bb(x, lengths=torch.full((2,), T, dtype=torch.long))
    return o["embeddings"] if isinstance(o, dict) else o


def test_the_info_channels_leave_the_patch_embedding():
    bb = build()
    assert bb.patch_embed.proj.weight.shape[1] == N_FEAT - N_INFO
    assert bb.info_proj.weight.shape[1] == N_INFO


def test_off_by_default_and_byte_identical_to_the_old_shape():
    bb = build(n_info=0)
    assert bb.info_proj is None
    assert bb.patch_embed.proj.weight.shape[1] == N_FEAT


def test_the_token_is_actually_read():
    bb = build().eval()
    x = _x()
    y = x.clone(); y[:, N_FEAT - N_INFO:, :] += 1.0
    with torch.no_grad():
        assert (_emb(bb, x) - _emb(bb, y)).abs().max() > 1e-3


def test_only_the_metadata_payload_is_read():
    """Corrupting reserved columns outside the payload must change nothing."""
    bb = build().eval()
    x = _x()
    z = x.clone(); z[:, N_FEAT - N_INFO:, :-1] = 999.0
    with torch.no_grad():
        assert torch.equal(_emb(bb, x), _emb(bb, z))


def test_it_carries_no_position_embedding():
    """n_pos counts patches + cls + state and NOT the info token, because the
    token is appended after the position embeddings are added."""
    a, b = build(n_info=0), build(n_info=N_INFO)
    assert a.position_embeddings.shape == b.position_embeddings.shape


def test_every_pooling_mode_ignores_the_token_itself():
    """It is an input to attention, not a member of the sequence. Under
    cls_at_end a trailing info token would BE the readout."""
    x = _x()
    for pool in ("cls", "mean", "max", "last"):
        for causal in (False, True):
            if causal and pool == "cls":
                continue          # cls_at_end path, covered by pool="last"
            bb = build(pool=pool, causal=causal).eval()
            with torch.no_grad():
                e = _emb(bb, x)
            assert e.shape == (2, 384) and torch.isfinite(e).all(), (pool, causal)


def test_under_a_causal_mask_the_readout_can_still_see_it():
    bb = build(pool="last", causal=True).eval()
    x = _x()
    y = x.clone(); y[:, N_FEAT - N_INFO:, :] += 1.0
    with torch.no_grad():
        assert (_emb(bb, x) - _emb(bb, y)).abs().max() > 1e-4


def test_both_widths_are_recoverable_from_the_weights_alone():
    sd = build().state_dict()
    assert _widths_from_state_dict(sd) == (N_FEAT, N_INFO)
    assert _widths_from_state_dict(build(n_info=0).state_dict()) == (N_FEAT, 0)


# ── frozen-identity patch embedding ────────────────────────────────────────

def _square(identity):
    """patch 32 / hidden 288: 9 real channels x 32 steps = 288, exactly square."""
    return create_backbone(
        backbone_type="transformer", n_features=N_FEAT + 3, d_embedding=384,
        config=TransformerConfig(hidden_size=288, intermediate_size=1152,
                                 num_attention_heads=6, patch_size=32,
                                 patch_embed_identity=identity),
        max_seq_len=T, pool="cls", state_token=True, n_info_channels=11)


def test_the_identity_projection_is_exactly_the_flattened_patch():
    """A SQUARE projection is still a learned matrix and tests nothing about
    whether the projection earns its place. This makes it the identity."""
    bb = _square(True)
    feat = torch.randn(2, 9, T)
    want = (feat.reshape(2, 9, T // 32, 32).permute(0, 2, 1, 3)
                .reshape(2, T // 32, 288))
    assert torch.equal(bb.patch_embed(feat), want)


def test_the_identity_is_frozen_and_takes_no_gradient():
    bb = _square(True)
    w = bb.patch_embed.proj.weight
    assert not w.requires_grad
    assert int((w != 0).sum()) == 9 * 32 and bool(((w == 0) | (w == 1)).all())
    x = torch.randn(2, N_FEAT + 3, T)
    x[:, 9:, :] = x[:, 9:, -1:].expand(-1, -1, T)
    o = bb(x, lengths=torch.full((2,), T, dtype=torch.long))
    (o["embeddings"] if isinstance(o, dict) else o).pow(2).mean().backward()
    assert w.grad is None


def test_a_non_square_hidden_size_is_refused_rather_than_silently_projected():
    with pytest.raises(ValueError, match="patch_embed_identity"):
        create_backbone(
            backbone_type="transformer", n_features=N_FEAT + 3, d_embedding=384,
            config=TransformerConfig(hidden_size=384, patch_size=32,
                                     patch_embed_identity=True),
            max_seq_len=T, pool="cls", n_info_channels=11)


def test_the_input_widths_survive_even_with_a_frozen_projection():
    """checkpoints.py reads the width off patch_embed.proj.weight, which is why
    this is a frozen Conv1d and not a reshape."""
    assert _widths_from_state_dict(_square(True).state_dict()) == (N_FEAT + 3, 11)


def test_identity_is_off_by_default():
    bb = _square(False)
    assert bb.patch_embed.proj.weight.requires_grad


# ── the OTHER entry point ───────────────────────────────────────────────────
#
# forward() strips the info columns; forward_patches() did not, for the eleven
# months between the token landing (2026-09-13) and the ssl-6mo wave failing on
# it. Six modes reach the backbone only through forward_patches -- cost, cpc,
# ijepa, mae, timemae, ts2vec -- so they handed a 20-column view to a patch
# embedding built for 9 and died in conv1d, while byol/dino/tfc/lejepa and the
# supervised arm went through forward() and were fine. A wave that fails at
# exactly two thirds of its arms is the signature; these pin the contract so it
# cannot come back quietly.


def _lens(n, t=T):
    return torch.full((n,), t, dtype=torch.long)


def test_forward_patches_strips_the_info_columns():
    bb = build(pool="mean")
    out = bb.forward_patches(_x(), _lens(2))
    assert out.shape == (2, T // 8, 384)


def test_forward_patches_returns_one_row_per_patch_not_a_token_more():
    """I-JEPA's predictor, MAE's decoder and TS2Vec's alignment all index this
    output positionally against the patch grid, so the token must come off."""
    bb = build(pool="mean")
    assert bb.forward_patches(_x(), _lens(2)).shape[1] == T // 8


def test_forward_patches_gathers_only_the_masked_patches():
    bb = build(pool="mean")
    idx = [torch.arange(5), torch.arange(5)]
    assert bb.forward_patches(_x(), _lens(2), mask_indices=idx).shape == (2, 5, 384)


def test_the_token_conditions_the_patch_output():
    bb = build(pool="mean").eval()
    x = _x()
    a = bb.forward_patches(x, _lens(2))
    x2 = x.clone()
    x2[:, -N_INFO:, -1] += 5.0
    assert not torch.allclose(a, bb.forward_patches(x2, _lens(2)), atol=1e-6)


def test_forward_patches_reads_the_payload_at_the_last_valid_step():
    """Right-padding must not be able to replace the payload with zeros."""
    bb = build(pool="mean").eval()
    x = _x()
    lengths = torch.tensor([T, T // 2])
    a = bb.forward_patches(x, lengths)
    x2 = x.clone()
    x2[:, -N_INFO:, T // 2:] += 5.0          # past the second row's length
    b = bb.forward_patches(x2, lengths)
    assert not torch.allclose(a[0], b[0], atol=1e-6)   # row 0 saw the change
    assert torch.allclose(a[1], b[1], atol=1e-6)       # row 1 could not


def test_info_proj_is_trained_through_forward_patches():
    """Otherwise the checkpoint's eval path adds a random projection: forward()
    uses info_proj whether or not training ever put a gradient through it."""
    bb = build(pool="mean")
    bb.forward_patches(_x(), _lens(2)).pow(2).mean().backward()
    assert bb.info_proj.weight.grad is not None
    assert bb.info_proj.weight.grad.abs().sum() > 0
