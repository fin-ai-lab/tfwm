"""A checkpoint's input width must survive the round trip to the scorer.

n_features is NOT a knob. It is computed by the dataset from several knobs
that are -- the information token adds 8 columns for the norm stats and 3 for
the window descriptors, diff_channels doubles the width, risk-factor tickers
add a block -- and the resolved number was only
ever written into the WANDB config. train_meta.json carries the hydra config,
which does not have it, so a checkpoint rebuilt from train_meta was built 9
wide and refused to load. That failure happens at SCORING time, after the full
training run, and it took out 75 of 105 jobs in one sweep.

These tests pin the fix: the width is read off the checkpoint's own first
projection, so it cannot disagree with the weights being loaded into it.
"""
import json
import torch

from market_jepa.eval.checkpoints import _n_features_from_state_dict, load_model
from market_jepa.modeling.backbones import create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig


def _save_supervised_ckpt(d, *, n_features, state_token):
    bb = create_backbone(
        backbone_type="transformer", n_features=n_features, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        pool="cls", **({"state_token": True} if state_token else {}))
    torch.save(bb.state_dict(), d / "backbone.pt")
    # A REAL head: load_model reads head.pt's own shape to decide whether the
    # run was binned, and a hand-rolled stand-in fails that check before the
    # backbone width is ever exercised.
    from market_jepa.eval.heads import RegressionHead
    torch.save(RegressionHead(384).state_dict(), d / "head.pt")
    # Exactly what save_train_meta writes now: the hydra config, which has the
    # knobs but NOT the width they imply.
    (d / "train_meta.json").write_text(json.dumps({
        "task": "return_900", "run_name": "t", "xs_target": "rank",
        "config": {
            "mode": {"_target_": "market_jepa.modeling.modes.supervised."
                                 "SupervisedModel", "task": "return_900",
                     "loss_fn": "pairwise"},
            # DELIBERATELY THE OLD KEY. Every checkpoint this regression is
            # about was written before the 2026-08-25 rename, so its config
            # really does say norm_stats_channels; the fixture would stop
            # reproducing the on-disk shape if it were modernized.
            "dataset": {"norm_stats_channels": n_features > 9},
            "backbone": {"_target_": "market_jepa.modeling.backbones."
                                     "transformer.TransformerBackbone",
                         "pool": "cls", "state_token": state_token,
                         "d_embedding": 384,
                         "config": {"hidden_size": 384, "num_hidden_layers": 2,
                                    "num_attention_heads": 6,
                                    "patch_size": 8}},
        }}))


def test_the_width_is_read_from_the_weights_not_the_config():
    sd = create_backbone(
        backbone_type="transformer", n_features=17, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        pool="cls").state_dict()
    assert _n_features_from_state_dict(sd) == 17


def test_a_state_dict_without_a_patch_embed_yields_none_not_a_guess():
    assert _n_features_from_state_dict({"other.weight": torch.zeros(3)}) is None


def test_normtok_checkpoint_loads_although_its_config_never_records_17(tmp_path):
    """The regression itself. The (mu, sigma) half of the information token
    means 9 + 8 = 17, and nothing in the hydra config says so."""
    _save_supervised_ckpt(tmp_path, n_features=17, state_token=True)
    import sys
    sys.path.insert(0, "scripts/generic")
    from post_train_ic_eval import _load_cfg

    cfg = _load_cfg(tmp_path, None)
    assert "n_features" not in cfg, "the config still must not carry the width"
    model = load_model(str(tmp_path), cfg, torch.device("cpu"))
    bb = getattr(model, "backbone", model)
    assert bb.patch_embed.proj.weight.shape[1] == 17
    assert bb.state_proj.weight.shape[1] == 17


def test_a_plain_9_channel_checkpoint_is_unaffected(tmp_path):
    _save_supervised_ckpt(tmp_path, n_features=9, state_token=False)
    import sys
    sys.path.insert(0, "scripts/generic")
    from post_train_ic_eval import _load_cfg

    model = load_model(str(tmp_path), _load_cfg(tmp_path, None),
                       torch.device("cpu"))
    bb = getattr(model, "backbone", model)
    assert bb.patch_embed.proj.weight.shape[1] == 9
