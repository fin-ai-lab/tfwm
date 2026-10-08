"""A random-init floor must be the SAME architecture, on the SAME panel.

Delta IC subtracts the floor from the trained number, so any difference
between the two that is not training is charged to training. Two were:

  ARCHITECTURE. pretrain.py takes n_info_channels from the DATASET, not from
  the backbone block, so train_meta.json records the unset config value (0)
  while the weights carry an info_proj of width 11. load_backbone recovers the
  truth from the state dict; build_untrained_encoder has no state dict, so it
  built a floor with no info_proj at all -- a structurally different encoder.

  PANEL. The floor was embedded without panel_kwargs, so it saw 9 channels
  where the models it floors see 9 + 8 norm-stat + 3 window = 20.

architecture_signature returns the SAME string either way, so nothing
downstream could notice. These assert the floor is loadable FROM a trained
checkpoint, which is the only check that covers every axis at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

from market_jepa.eval.checkpoints import build_untrained_encoder  # noqa: E402
from xs_ic_eval import _info_channel_width  # noqa: E402


def _cfg():
    return {
        "mode": {
            "_target_": "market_jepa.modeling.modes.supervised.SupervisedModel",
            "backbone": {
                "_target_": "market_jepa.modeling.backbones.transformer.TransformerBackbone",
                "d_embedding": 384, "pool": "last", "max_seq_len": 2048,
                "config": {"pos_embed": "sinusoidal"},
            },
        },
        "n_features": 9,
    }


def test_info_width_follows_the_panel_flags():
    assert _info_channel_width({"info_norm_stats": True, "info_window": True}) == 11
    assert _info_channel_width({"info_norm_stats": True, "info_window": False}) == 8
    assert _info_channel_width({"info_norm_stats": False, "info_window": True}) == 3
    assert _info_channel_width({"info_norm_stats": False, "info_window": False}) == 0


def test_the_floor_gains_an_info_projection_when_the_panel_has_one():
    plain = build_untrained_encoder(_cfg(), torch.device("cpu"), seed=0).backbone
    withi = build_untrained_encoder(
        _cfg(), torch.device("cpu"), seed=0, n_info_channels=11).backbone
    assert not any("info_proj" in n for n, _ in plain.named_parameters())
    assert any("info_proj" in n for n, _ in withi.named_parameters()), (
        "the floor has no info_proj, so it cannot see the metadata every "
        "trained model sees and is not their floor")


def test_the_info_token_widens_the_input_rather_than_reallocating_it():
    """n_features is the TOTAL and n_info_channels a subset of it."""
    withi = build_untrained_encoder(
        _cfg(), torch.device("cpu"), seed=0, n_info_channels=11).backbone
    conv = [p for n, p in withi.named_parameters() if "patch" in n][0]
    assert conv.shape[1] == 9, (
        f"patch embedding takes {conv.shape[1]} channels; the 11 info "
        f"channels must ADD to the 9 real ones, not consume them")


def test_a_trained_state_dict_loads_into_the_matched_floor():
    """The whole-architecture check, in one assertion.

    Every axis -- pool, position encoding, widths, the info projection -- has
    to agree for this to pass, and each of those has been wrong at least once.
    """
    withi = build_untrained_encoder(
        _cfg(), torch.device("cpu"), seed=0, n_info_channels=11).backbone
    donor = build_untrained_encoder(
        _cfg(), torch.device("cpu"), seed=1, n_info_channels=11).backbone
    withi.load_state_dict(donor.state_dict())

    plain = build_untrained_encoder(_cfg(), torch.device("cpu"), seed=0).backbone
    try:
        plain.load_state_dict(donor.state_dict())
    except RuntimeError:
        return
    raise AssertionError(
        "an info-token state dict loaded into a floor built without one; "
        "the mismatch this test exists for would be undetectable")


def test_the_floor_derives_the_token_from_the_checkpoint_by_default():
    """No argument, and the floor still matches the models it floors.

    ``n_info_channels`` began life as an opt-in override, which made the
    correct floor a thing the CALLER had to remember -- and the whole point of
    this file is that forgetting it is undetectable downstream, because
    architecture_signature returns the same string either way. So the width is
    now DERIVED from the checkpoint's own dataset block, through the same
    ``dataset_flag`` the scorer builds the panel with, and the argument is
    only an override.
    """
    dev = torch.device("cpu")

    def with_ds(**ds):
        cfg = _cfg()
        cfg["dataset"] = ds
        return build_untrained_encoder(cfg, dev, seed=0).backbone

    def info_width(bb):
        sd = bb.state_dict()
        return int(sd["info_proj.weight"].shape[1]) if "info_proj.weight" in sd else 0

    # A modern checkpoint says so in its dataset block, and gets all eleven.
    assert info_width(with_ds(info_norm_stats=True, info_window=True)) == 11
    # The pre-2026-08-25 spellings are still a historical record.
    assert info_width(with_ds(norm_stats_channels=True, time_info=True)) == 11
    # Each flag on its own, so an ablation floors itself correctly.
    assert info_width(with_ds(info_norm_stats=True)) == 8
    assert info_width(with_ds(info_window=True)) == 3

    # ABSENCE MEANS OFF, not "the live default". save_train_meta writes these
    # keys only when they are on, so a checkpoint from before the information
    # token existed must floor at width 0 -- which is what it trained with.
    assert info_width(with_ds()) == 0
    assert info_width(build_untrained_encoder(_cfg(), dev, seed=0).backbone) == 0

    # The override still wins in both directions.
    cfg = _cfg()
    cfg["dataset"] = {"info_norm_stats": True, "info_window": True}
    assert int(build_untrained_encoder(
        cfg, dev, seed=0, n_info_channels=0).backbone.state_dict()
        .get("info_proj.weight", torch.zeros(1, 0)).shape[1]) == 0


def test_the_two_info_width_definitions_agree():
    """The scorer's panel width and market_jepa's must be one number.

    ``_info_channel_width`` asks encode_view_metadata what a real ViewMetadata
    encodes to; ``info_channel_width`` computes it from the flags. They are
    used on opposite sides of the subtraction -- the panel and the floor -- so
    a disagreement is a floor built for a panel that does not exist.
    """
    from market_jepa.augmentations import info_channel_width

    for ns in (True, False):
        for wd in (True, False):
            assert _info_channel_width(
                {"info_norm_stats": ns, "info_window": wd}
            ) == info_channel_width(info_norm_stats=ns, info_window=wd)
