"""Every mode owns its readout, and the top level can never quietly win.

The backbone block a run trains is decided by ONE rule -- a mode that declares
its own backbone owns it -- and that rule was previously re-implemented at ten
call sites. Two of them got it wrong: the eval loader built a cls-pooled
2049-position model for last/rope checkpoints (loud, every IC eval died) and
architecture_signature matched SSL checkpoints to the wrong random-init floor
(silent, the figures just read wrong).

These tests exist so the rule cannot drift again: the schema stays uniform, the
resolver answers identically whatever shape it is handed, and an override that
would be ignored is fatal rather than logged.
"""

import dataclasses
import inspect

import pytest

import market_jepa.schemas as schemas
from market_jepa.backbone_config import (
    MODES_WITHOUT_A_BACKBONE,
    assert_no_ignored_backbone_overrides,
    backbone_block,
    owns_backbone,
)

# The supervised family reads the LAST token; everything else pools the MEAN.
# The POSITION ENCODING is now uniform -- sinusoidal everywhere -- so that a
# supervised and an SSL checkpoint scored at the prediction readout share one
# architecture_signature and therefore one random-init floor.
LAST_POOL_MODES = {"SupervisedModeConfig", "MultiTaskSupervisedModeConfig"}


def _mode_configs():
    for name, obj in vars(schemas).items():
        if (inspect.isclass(obj) and dataclasses.is_dataclass(obj)
                and name.endswith("ModeConfig")):
            yield name, obj


def test_there_are_mode_configs_to_check():
    """A guard on the collector itself: a rename must not empty this suite."""
    assert len(list(_mode_configs())) >= 10


@pytest.mark.parametrize("name,cls", list(_mode_configs()))
def test_every_mode_owns_a_backbone_or_is_listed(name, cls):
    """A new mode cannot join the no-backbone set by omission."""
    declares = bool(getattr(cls(), "backbone", None))
    if name in MODES_WITHOUT_A_BACKBONE:
        assert not declares, f"{name} is listed as having no backbone but declares one"
    else:
        assert declares, (
            f"{name} declares no backbone, so it would silently fall back to "
            f"the top-level block. Give it one, or add it to "
            f"MODES_WITHOUT_A_BACKBONE with a reason."
        )


@pytest.mark.parametrize("name,cls", [
    (n, c) for n, c in _mode_configs() if n not in MODES_WITHOUT_A_BACKBONE
])
def test_the_readout_is_uniform_across_methods(name, cls):
    """The POOL is the only axis a mode may differ on."""
    bb = cls().backbone
    pool, pos = bb.pool, bb.config.pos_embed
    assert pos == "sinusoidal", (
        f"{name} uses {pos} positions. The encoding is uniform so that the "
        f"supervised and SSL arms share one floor; a mode that differs here "
        f"silently splits that floor in two."
    )
    want = "last" if name in LAST_POOL_MODES else "mean"
    assert pool == want, f"{name} pools {pool}, expected {want}"


@pytest.mark.parametrize("name,cls", [
    (n, c) for n, c in _mode_configs() if n not in MODES_WITHOUT_A_BACKBONE
])
def test_the_position_encoding_is_uniform(name, cls):
    """One encoding across every method, which is what unifies the floor."""
    assert cls().backbone.config.pos_embed == "sinusoidal"


def test_the_resolver_prefers_the_mode_over_the_top_level():
    cfg = {"mode": {"_target_": "m.Supervised", "backbone": {"pool": "last"}},
           "backbone": {"pool": "cls"}}
    assert backbone_block(cfg)["pool"] == "last"
    assert owns_backbone(cfg)


def test_an_empty_mode_block_is_not_a_declaration():
    cfg = {"mode": {"_target_": "m.X", "backbone": {}}, "backbone": {"pool": "cls"}}
    assert backbone_block(cfg)["pool"] == "cls"
    assert not owns_backbone(cfg)


def test_the_resolver_agrees_across_config_shapes():
    """The trainer hands it a DictConfig; the scorer hands it a dict."""
    from omegaconf import OmegaConf
    raw = {"mode": {"_target_": "m.Supervised",
                    "backbone": {"pool": "last", "config": {"pos_embed": "rope"}}},
           "backbone": {"pool": "cls", "config": {"pos_embed": "learned"}}}
    as_dict = backbone_block(raw)
    as_omega = backbone_block(OmegaConf.create(raw))
    assert as_dict["pool"] == as_omega["pool"] == "last"
    assert as_dict["config"]["pos_embed"] == as_omega["config"]["pos_embed"] == "rope"


def test_an_ignored_top_level_override_is_fatal(monkeypatch):
    """The trap: composes cleanly, logs your value, trains something else."""
    import market_jepa.backbone_config as bc
    monkeypatch.setattr(bc, "ignored_top_level_overrides",
                        lambda: ["backbone.config.pos_embed=sinusoidal"])
    cfg = {"mode": {"_target_": "m.SupervisedModel", "backbone": {"pool": "last"}}}
    with pytest.raises(ValueError, match="owns its backbone"):
        bc.assert_no_ignored_backbone_overrides(cfg)


def test_the_error_names_the_key_that_would_have_worked():
    import market_jepa.backbone_config as bc
    original = bc.ignored_top_level_overrides
    bc.ignored_top_level_overrides = lambda: ["backbone.pool=mean"]
    try:
        cfg = {"mode": {"_target_": "m.SupervisedModel", "backbone": {"pool": "last"}}}
        with pytest.raises(ValueError) as excinfo:
            bc.assert_no_ignored_backbone_overrides(cfg)
        assert "mode.backbone.pool=mean" in str(excinfo.value)
    finally:
        bc.ignored_top_level_overrides = original


def test_a_mode_without_a_backbone_may_override_the_top_level(monkeypatch):
    """FinanceBaseline/TSFM legitimately use the top level."""
    import market_jepa.backbone_config as bc
    monkeypatch.setattr(bc, "ignored_top_level_overrides",
                        lambda: ["backbone.pool=mean"])
    assert_no_ignored_backbone_overrides({"mode": {"_target_": "m.Finance"}})


def test_a_group_selection_is_not_a_field_override():
    """Every sweep passes backbone=transformer; it must stay harmless."""
    import market_jepa.backbone_config as bc
    original = bc.ignored_top_level_overrides
    bc.ignored_top_level_overrides = lambda: []
    try:
        cfg = {"mode": {"_target_": "m.SupervisedModel", "backbone": {"pool": "last"}}}
        assert bc.assert_no_ignored_backbone_overrides(cfg) is None
    finally:
        bc.ignored_top_level_overrides = original
