"""The CALLERS of backbone_block, not the resolver.

tests/test_backbone_ownership.py locks the rule. It passed green while four
call sites still ignored it, two of them producing a cls/learned model for a
last/rope checkpoint. A test that locks a resolver is not a test that locks
its callers, so these exercise the call sites themselves.

The failure mode is never a missing block. The top-level block is PRESENT and
carries pool=None / pos_embed=None, which the backbone resolves to "cls" and
"learned" -- so reading the wrong one silently yields a different architecture
rather than an error.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

from market_jepa.backbone_config import backbone_block  # noqa: E402
from market_jepa.eval.checkpoints import load_backbone  # noqa: E402
from market_jepa.modeling.backbones import create_backbone  # noqa: E402


def _supervised_cfg():
    """A real supervised run's shape: mode owns last/rope, top level is empty."""
    return {
        "mode": {
            "_target_": "market_jepa.modeling.modes.supervised.SupervisedModel",
            "backbone": {
                "_target_": "market_jepa.modeling.backbones.transformer.TransformerBackbone",
                "d_embedding": 384, "pool": "last", "max_seq_len": 2048,
                "config": {"pos_embed": "rope"},
            },
        },
        # Present, and empty of answers -- this is what a sweep leaves behind.
        "backbone": {
            "_target_": "market_jepa.modeling.backbones.transformer.TransformerBackbone",
            "d_embedding": 384, "pool": None, "max_seq_len": 2048,
            "config": {"pos_embed": None},
        },
        "n_features": 9,
    }


@pytest.fixture
def last_rope_checkpoint(tmp_path):
    """Save a genuine last/rope backbone plus the train_meta.json beside it."""
    from market_jepa.eval.checkpoints import _transformer_config_from
    cfg = _supervised_cfg()
    bb = backbone_block(cfg)
    # Built from the block the MODE declares, through the same helper the
    # loader uses, so the state dict is exactly what a real run would save.
    backbone = create_backbone(
        backbone_type="transformer", n_features=cfg["n_features"],
        d_embedding=bb["d_embedding"], pool=bb["pool"],
        config=_transformer_config_from(bb["config"]),
    )
    d = tmp_path / "run"
    d.mkdir()
    torch.save(backbone.state_dict(), d / "backbone.pt")
    (d / "train_meta.json").write_text(json.dumps({"config": cfg}))
    return d, backbone


def test_load_backbone_reads_the_trained_block(last_rope_checkpoint):
    """The whole latent-eval suite enters through here."""
    d, original = last_rope_checkpoint
    loaded = load_backbone(d)          # must not raise
    assert loaded.pool == "last", f"loaded a {loaded.pool}-pooled model"
    assert sum(p.numel() for p in loaded.parameters()) == \
        sum(p.numel() for p in original.parameters())


def test_load_backbone_does_not_invent_a_cls_token(last_rope_checkpoint):
    """cls/learned is what the EMPTY top-level block resolves to."""
    d, _ = last_rope_checkpoint
    loaded = load_backbone(d)
    assert not hasattr(loaded, "cls_token") or loaded.cls_token is None, \
        "built a cls-pooled model for a last-pooled checkpoint"


def test_predict_readout_reaches_an_ssl_mode(monkeypatch):
    """PREDICT_POOL was a no-op for every SSL mode: pinned on the wrong block."""
    from xs_ic_eval import PREDICT_POOL, _predict_readout
    cfg = _supervised_cfg()
    cfg["mode"]["_target_"] = "market_jepa.modeling.modes.lejepa.LeJEPA"
    cfg["mode"]["backbone"]["pool"] = "mean"
    cfg["mode"]["backbone"]["config"]["pos_embed"] = "sinusoidal"
    out = _predict_readout(cfg)
    assert backbone_block(out)["pool"] == PREDICT_POOL, \
        "the readout pin never reached the block the encoder is built from"


def test_predict_readout_leaves_the_input_untouched():
    """It deep-copies; a scorer must be able to reuse the original config."""
    from xs_ic_eval import _predict_readout
    cfg = _supervised_cfg()
    cfg["mode"]["backbone"]["pool"] = "mean"
    _predict_readout(cfg)
    assert cfg["mode"]["backbone"]["pool"] == "mean"


def test_predict_readout_does_not_touch_position_encoding():
    """It pins the TOKEN, not the architecture: the floor must stay matched."""
    from xs_ic_eval import _predict_readout
    cfg = _supervised_cfg()
    cfg["mode"]["backbone"]["config"]["pos_embed"] = "sinusoidal"
    out = _predict_readout(cfg)
    assert backbone_block(out)["config"]["pos_embed"] == "sinusoidal"
