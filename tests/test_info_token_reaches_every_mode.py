"""Every mode that touches the patch grid, against a 20-column view.

THE BUG THIS REPLACES A DAY OF. The information token (2026-09-13) moved the
trailing per-window columns out of the patch embedding, and ``forward()`` was
taught to split them off. Nothing else was. Two more paths reach the patch
embedding -- ``forward_patches`` (cost, cpc, ijepa, mae, timemae, ts2vec) and
TimeMAE's direct ``backbone.patch_embed`` call for its tokenizer -- and both
handed conv1d a 20-column view built for 9. The ssl-6mo wave lost six of nine
arms on all 31 months to the first, and then lost timemae again to the second
after the first was fixed, because the fix was verified on ONE arm.

So this is parametrized over every mode, and asserts on a real step rather
than on a shape: an arm that constructs and then dies in its loss is exactly
what shipped twice.
"""
from __future__ import annotations

import pytest
import torch

from market_jepa.modeling import create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig

N_INFO, N_DATA, T = 11, 9, 256
N_FEAT = N_DATA + N_INFO


def _backbone(pool="mean"):
    return create_backbone(
        backbone_type="transformer", n_features=N_FEAT, d_embedding=64,
        config=TransformerConfig(hidden_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, intermediate_size=128,
                                 patch_size=8, drop_path_rate=0.0),
        pool=pool, max_seq_len=T, n_info_channels=N_INFO)


def _view(b=4):
    """A view shaped like the real one: the payload only on the last step."""
    x = torch.randn(b, N_FEAT, T)
    x[:, N_DATA:, :-1] = 0
    return x


def _build(name):
    from market_jepa.modeling import CPC, MAE, IJEPA, TimeMAE, TS2Vec
    from market_jepa.modeling.modes.cost import CoST
    if name == "timemae":
        return TimeMAE(backbone=_backbone(), vocab_size=32, reg_layers=2)
    if name == "ts2vec":
        return TS2Vec(backbone=_backbone())
    if name == "cost":
        return CoST(backbone=_backbone())
    if name == "cpc":
        return CPC(backbone=_backbone())
    if name == "mae":
        return MAE(backbone=_backbone())
    if name == "ijepa":
        return IJEPA(backbone=_backbone())
    raise AssertionError(name)


MODES = ["cost", "cpc", "ijepa", "mae", "timemae", "ts2vec"]


@pytest.mark.parametrize("name", MODES)
def test_the_mode_runs_a_step_on_a_twenty_column_view(name):
    model = _build(name)
    x = _view()
    lengths = torch.full((x.shape[0],), T, dtype=torch.long)
    try:
        out = model(x, lengths=lengths)
    except TypeError:
        out = model(x)
    assert out is not None, f"{name} returned nothing"


@pytest.mark.parametrize("name", MODES)
def test_the_patch_embedding_is_never_handed_the_info_columns(name):
    """Asserts on the CONV's input width wherever it is reached from."""
    model = _build(name)
    seen = []
    bb = model.backbone
    orig = bb.patch_embed.forward
    bb.patch_embed.forward = lambda t, _o=orig: (seen.append(t.shape[1]), _o(t))[1]
    x = _view()
    lengths = torch.full((x.shape[0],), T, dtype=torch.long)
    try:
        model(x, lengths=lengths)
    except TypeError:
        model(x)
    assert seen, f"{name} never reached the patch embedding"
    assert set(seen) == {N_DATA}, f"{name} fed widths {sorted(set(seen))}"


def test_patch_channels_is_what_the_projection_expects():
    bb = _backbone()
    assert bb.patch_channels(_view()).shape[1] == bb.patch_embed.proj.weight.shape[1]
