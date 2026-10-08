"""Tests for the event-conditioned predictor."""

from __future__ import annotations

import pytest
import torch

from market_jepa.eval.heads import RegressionHead
from market_jepa.modeling.event_predictor import (
    EventPredictor,
    pool_chunk_embeddings,
)

D_EMB = 384
# RegressionHead emits one scalar (the cross-sectional z-score). EventPredictor
# splices into head.mlp, so its output keeps the trailing dim rather than being
# squeezed the way RegressionHead.forward does.
OUT_DIM = 1
EMB_DIM = 3072


def _head() -> RegressionHead:
    torch.manual_seed(0)
    h = RegressionHead(D_EMB)
    # Move off the init point so "identical to the pretrained head" is a real claim.
    with torch.no_grad():
        for p in h.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return h.eval()


def test_matches_pretrained_head_at_init():
    """Zero-init g means the full model starts exactly at the market-only baseline."""
    head = _head()
    cls = torch.randn(8, D_EMB)
    emb = torch.randn(8, EMB_DIM)

    expected = head.mlp(cls)
    model = EventPredictor(_head(), EMB_DIM, bottleneck=16).eval()
    got = model(cls=cls, emb=emb)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_embedding_changes_output_once_g_is_nonzero():
    model = EventPredictor(_head(), EMB_DIM, bottleneck=16).eval()
    cls = torch.randn(4, D_EMB)
    e1, e2 = torch.randn(4, EMB_DIM), torch.randn(4, EMB_DIM)

    assert torch.equal(model(cls=cls, emb=e1), model(cls=cls, emb=e2))  # zero-init
    with torch.no_grad():
        model.g.fc2.weight.normal_(0, 0.02)
    assert not torch.equal(model(cls=cls, emb=e1), model(cls=cls, emb=e2))


def test_first_layer_frozen_rest_trainable():
    model = EventPredictor(_head(), EMB_DIM, bottleneck=16)
    assert not model.first.weight.requires_grad
    assert not model.first.bias.requires_grad
    for p in model.g.parameters():
        assert p.requires_grad
    for p in model.rest.parameters():
        assert p.requires_grad

    # mlp[0] holds the bulk of the head, so freezing must actually shrink the budget.
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert frozen == D_EMB * 768 + 768


def test_no_grad_reaches_frozen_first_layer():
    model = EventPredictor(_head(), EMB_DIM, bottleneck=16)
    out = model(cls=torch.randn(4, D_EMB), emb=torch.randn(4, EMB_DIM))
    out.sum().backward()
    assert model.first.weight.grad is None
    assert model.g.fc1.weight.grad is not None


def test_bottleneck_keeps_parameter_count_small():
    """The bottleneck is the whole reason this is trainable on ~1e2 events."""
    model = EventPredictor(_head(), EMB_DIM, bottleneck=16)
    g_params = sum(p.numel() for p in model.g.parameters())
    dense = EMB_DIM * 768 + 768
    assert g_params < dense / 30


@pytest.mark.parametrize("bottleneck", [4, 8, 16, 64])
def test_bottleneck_widths_run(bottleneck):
    model = EventPredictor(_head(), EMB_DIM, bottleneck=bottleneck).eval()
    out = model(cls=torch.randn(3, D_EMB), emb=torch.randn(3, EMB_DIM))
    assert out.shape == (3, OUT_DIM)


def test_text_only_arm():
    """No market pathway: g must NOT be zero-init, or it starts at a dead point."""
    model = EventPredictor(_head(), EMB_DIM, use_market=False, bottleneck=16).eval()
    assert not model.g.fc2.weight.eq(0).all()

    e1, e2 = torch.randn(4, EMB_DIM), torch.randn(4, EMB_DIM)
    o1, o2 = model(emb=e1), model(emb=e2)
    assert o1.shape == (4, OUT_DIM)
    assert torch.isfinite(o1).all()
    assert not torch.equal(o1, o2)


def test_baseline_arm_is_the_plain_head():
    head = _head()
    cls = torch.randn(6, D_EMB)
    model = EventPredictor(_head(), EMB_DIM, use_text=False).eval()
    torch.testing.assert_close(model(cls=cls), head.mlp(cls), rtol=0, atol=0)
    assert model.g is None


def test_missing_input_raises():
    m_both = EventPredictor(_head(), EMB_DIM)
    with pytest.raises(ValueError, match="cls is None"):
        m_both(emb=torch.randn(2, EMB_DIM))
    with pytest.raises(ValueError, match="emb is None"):
        m_both(cls=torch.randn(2, D_EMB))


def test_both_arms_disabled_raises():
    with pytest.raises(ValueError, match="at least one"):
        EventPredictor(_head(), EMB_DIM, use_market=False, use_text=False)


def test_rejects_non_head_module():
    with pytest.raises(TypeError):
        EventPredictor(torch.nn.Linear(4, 4), EMB_DIM)


# ── chunk pooling ────────────────────────────────────────────────────────────


def test_pool_chunks_masked_mean():
    chunks = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(
        pool_chunk_embeddings(chunks, mask), torch.tensor([[2.0, 2.0]])
    )


def test_pool_chunks_unmasked_is_plain_mean():
    chunks = torch.randn(4, 7, 16)
    torch.testing.assert_close(pool_chunk_embeddings(chunks), chunks.mean(dim=1))


def test_pool_chunks_no_valid_chunk_is_zero_not_nan():
    chunks = torch.randn(2, 3, 5)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    out = pool_chunk_embeddings(chunks, mask)
    assert torch.isfinite(out).all()
    assert out.abs().sum() == 0


def test_pool_chunks_shape_validation():
    with pytest.raises(ValueError, match="batch, n_chunks, emb_dim"):
        pool_chunk_embeddings(torch.randn(4, 16))
    with pytest.raises(ValueError, match="mask must be"):
        pool_chunk_embeddings(torch.randn(2, 3, 4), torch.ones(2, 5, dtype=torch.bool))
