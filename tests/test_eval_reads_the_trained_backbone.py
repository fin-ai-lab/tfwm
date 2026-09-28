"""The scorer must build the architecture the mode TRAINED, not a stale default.

Every supervised checkpoint is last/rope with 2048 positions and fused qkv.
The loader used to pick its backbone block with ``"IJEPA" in mode_target``,
which sent every non-IJEPA mode to the TOP-LEVEL ``cfg["backbone"]``. That
block is not absent -- it carries the composed default, ``pool="cls"`` with
learned positions -- so while sweeps still pinned a backbone at the top level
the misread was invisible. Once modes owned their backbone and sweeps stopped
pinning, the loader began building a cls-pooled 2049-position model for a
last/rope 2048-position checkpoint and every post-train IC eval died in
load_state_dict.

These lock the ONE RULE the trainer uses: a mode that declares a backbone owns
it.
"""

from market_jepa.eval.checkpoints import _backbone_block, architecture_signature


def _cfg(mode_pool, mode_pos, top_pool="cls", top_pos="learned"):
    """A config shaped like a real supervised run's train_meta.json."""
    return {
        "mode": {
            "_target_": "market_jepa.modeling.modes.supervised.SupervisedModel",
            "backbone": {
                "_target_": "market_jepa.modeling.backbones.TransformerBackbone",
                "pool": mode_pool,
                "d_embedding": 384,
                "config": {"pos_embed": mode_pos},
            },
        },
        # The leftover the trainer ignores and the loader used to read.
        "backbone": {
            "_target_": "market_jepa.modeling.backbones.TransformerBackbone",
            "pool": top_pool,
            "d_embedding": 384,
            "config": {"pos_embed": top_pos},
        },
    }


def test_mode_backbone_wins_over_stale_top_level():
    block = _backbone_block(_cfg("last", "rope"))
    assert block["pool"] == "last", "read the stale top-level pool"
    assert block["config"]["pos_embed"] == "rope"


def test_top_level_is_used_when_the_mode_declares_none():
    cfg = _cfg("last", "rope")
    del cfg["mode"]["backbone"]
    assert _backbone_block(cfg)["pool"] == "cls"


def test_a_bare_parameter_count_does_not_outrank_the_mode():
    """Some runs logged cfg.backbone as a bare int rather than a block."""
    cfg = _cfg("last", "rope")
    cfg["backbone"] = 22_000_000
    assert _backbone_block(cfg)["pool"] == "last"


def test_a_bare_parameter_count_with_no_mode_block_is_not_a_block():
    """The int must not reach the loader as though it were a config.

    This is the case the type test exists for, and it was the one the old
    implementation left uncovered -- its int branch was written as a fallback
    to mode.backbone, which is unreachable here by construction.
    """
    cfg = _cfg("last", "rope")
    del cfg["mode"]["backbone"]
    cfg["backbone"] = 22_000_000
    assert _backbone_block(cfg) == {}


def test_an_empty_mode_block_does_not_shadow_the_top_level():
    """A mode that declares `backbone: {}` has declared nothing."""
    cfg = _cfg("last", "rope")
    cfg["mode"]["backbone"] = {}
    assert _backbone_block(cfg)["pool"] == "cls"


def test_signature_distinguishes_the_two_readouts():
    """The floor a checkpoint is scored against depends on this block."""
    last = architecture_signature(_cfg("last", "rope"))
    mean = architecture_signature(_cfg("mean", "sinusoidal"))
    assert last != mean, "last/rope and mean/sinusoidal share a signature"


def test_signature_ignores_the_stale_top_level_block():
    """Two runs identical in what they trained must sign identically."""
    a = architecture_signature(_cfg("last", "rope", top_pool="cls"))
    b = architecture_signature(_cfg("last", "rope", top_pool="mean"))
    assert a == b, "the signature moved with a block the trainer never read"
