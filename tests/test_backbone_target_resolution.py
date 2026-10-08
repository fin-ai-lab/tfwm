"""A checkpoint must come back as the architecture it was trained as.

``load_model`` picks the backbone class from the hydra ``_target_`` recorded
with the run. That lookup was keyed on full dotted paths under
``market_jepa.modeling.*`` -- a layout that has not existed since the backbones
moved to ``market_jepa.modeling.backbones.*``. Every lookup missed, and the
miss fell through to a ``"transformer"`` default, so a ResNet checkpoint was
rebuilt as a ViT and died on a state dict with no overlapping keys.

The transformer only ever passed because it WAS the wrong default. That is why
these tests check the non-ViT backbones: a regression here is invisible until a
sweep has already spent its GPU hours.
"""

from __future__ import annotations

import pytest

from market_jepa import schemas
from market_jepa.eval.checkpoints import _backbone_type_from_target


@pytest.mark.parametrize(
    "target,expected",
    [
        ("market_jepa.modeling.backbones.resnet.ResNetBackbone", "resnet"),
        ("market_jepa.modeling.backbones.transformer.TransformerBackbone", "transformer"),
        # The legacy spelling post_train_ic_eval._load_cfg still writes.
        ("market_jepa.modeling.transformer.TransformerBackbone", "transformer"),
        ("market_jepa.modeling.resnet.ResNetBackbone", "resnet"),
    ],
)
def test_both_module_layouts_resolve(target, expected):
    assert _backbone_type_from_target(target) == expected


def test_absent_target_still_defaults_to_the_vit():
    """Old wandb configs carry no target at all, and every one is a ViT."""
    assert _backbone_type_from_target("") == "transformer"
    assert _backbone_type_from_target(None) == "transformer"


def test_an_unknown_target_raises_instead_of_becoming_a_vit():
    """The silent fallback is the bug; a present-but-unknown target must fail.

    Defaulting here is what turned a ResNet into a ViT after a full training
    run. A crash at load is recoverable; a wrong architecture that happens to
    load is not.
    """
    with pytest.raises(ValueError, match="unknown backbone _target_"):
        _backbone_type_from_target("some.other.pkg.MysteryBackbone")


def test_every_backbone_schema_target_resolves():
    """The map must cover every backbone the config store can actually select.

    Pinned against the schemas rather than a hand-written list, so a new
    backbone config cannot be added without this failing.
    """
    for name in dir(schemas):
        if not name.endswith("BackboneConfig"):
            continue
        target = getattr(schemas, name)()._target_
        # Resolves without raising, and does NOT quietly become a ViT unless
        # it really is one.
        kind = _backbone_type_from_target(target)
        assert kind != "transformer" or "Transformer" in target, (
            f"{name} ({target}) silently resolved to a ViT"
        )
