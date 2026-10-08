"""Knobs that change the PARAMETER SET must survive save/load.

No mode's ``save_pretrained`` ever recorded ``state_token``, ``diff_channels``
or ``n_info_channels`` -- only ``n_features`` and the inner TransformerConfig.
So a checkpoint trained with an information token would rebuild without one,
put the per-window columns through the patch embedding, and fail on a size
mismatch AFTER training finished. That is the same failure that cost 75 jobs
on 2026-08-24 in the supervised loader; these pin the SSL side of it.

They are recovered from the weights rather than from a config migration, so
checkpoints written before any of the knobs existed still load: a missing
tensor means the knob was off.
"""
import tempfile

import pytest
import torch

from market_jepa.modeling.backbones import (
    backbone_kwargs_from_state_dict, create_backbone,
)
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.modeling.modes.lejepa import LeJEPA

CASES = [
    ({}, {"n_features": 20}),
    ({"state_token": True}, {"n_features": 20, "state_token": True}),
    ({"n_info_channels": 11}, {"n_features": 20, "n_info_channels": 11}),
    ({"state_token": True, "n_info_channels": 11},
     {"n_features": 20, "state_token": True, "n_info_channels": 11}),
]


def _bb(**kw):
    return create_backbone(
        backbone_type="transformer", n_features=20, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        max_seq_len=2048, pool="cls", **kw)


@pytest.mark.parametrize("built,want", CASES)
def test_the_knobs_are_recoverable_from_the_weights(built, want):
    sd = {f"backbone.{k}": v for k, v in _bb(**built).state_dict().items()}
    assert backbone_kwargs_from_state_dict(sd) == want


def test_a_non_patched_backbone_says_nothing_rather_than_guessing():
    assert backbone_kwargs_from_state_dict({"backbone.other.weight": torch.zeros(3)}) == {}


def test_a_lejepa_checkpoint_with_an_info_token_round_trips():
    m = LeJEPA(backbone=_bb(n_info_channels=11), proj_dim=64,
               n_projections=256, lamb=0.01)
    x = torch.randn(2, 20, 2048)
    x[:, 9:, :] = x[:, 9:, -1:].expand(-1, -1, 2048)
    lens = [torch.full((2,), 2048, dtype=torch.long)]
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = LeJEPA.from_pretrained(d)
        assert m2.backbone.patch_embed.proj.weight.shape[1] == 9
        assert m2.backbone.info_proj.weight.shape[1] == 11
        # eval(), because the last block's StochasticDepth (drop_path_rate
        # 0.1) is a live Bernoulli in train mode: two forwards of the SAME
        # weights agree only when the two masks happen to match, which is
        # about 45% of the time here. What is under test is the restored
        # parameter set, not drop_path.
        m.eval(), m2.eval()
        with torch.no_grad():
            a = m.encode([x], lens)["embeddings"]
            b = m2.encode([x], lens)["embeddings"]
        assert torch.equal(a, b)


def test_the_config_still_does_not_record_them():
    """If save_pretrained ever starts writing these, the recovery path above
    becomes dead code and this test should be revisited deliberately."""
    import json
    from pathlib import Path
    m = LeJEPA(backbone=_bb(n_info_channels=11), proj_dim=64,
               n_projections=256, lamb=0.01)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        cfg = json.loads((Path(d) / "config.json").read_text())
    assert "n_info_channels" not in cfg and "state_token" not in cfg
