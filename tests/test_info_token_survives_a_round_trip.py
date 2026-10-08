"""An information-token checkpoint must come back as the model that was saved.

THE COMPANION TO test_info_token_reaches_every_mode. That file pins the
TRAINING side of the 2026-09-13 information token: every mode must run a step
on a 20-column view. This one pins the LOADING side, which was still wrong
after training was fixed, and failed in the place a failure costs most --
after 31 months x 14 arms had already trained.

WHAT WAS WRONG, IN TWO PLACES. The token takes the trailing per-window
constants OUT of the patch embedding, so a checkpoint's patch_embed is 9 wide
and info_proj takes the other 11. No save_pretrained records that split;
``backbone_kwargs_from_state_dict`` recovers it from the tensors. Six modes
CALLED that helper and then dropped everything but n_features on the floor --
so from_pretrained rebuilt a 20-channel patch embedding with no info_proj and
died on ``Unexpected key(s): "info_proj.weight"``. And ``load_backbone``'s
configless branch never called it at all, which is the branch IJEPA lands in:
it defines no save_pretrained, so its checkpoints ship model.pt and
train_meta.json and nothing else.

The assertions are on a REBUILT MODEL'S OUTPUT, not on a successful load. A
rebuild that silently picks a different architecture (the configless branch
used TransformerConfig() defaults, whose pos_embed is "learned" against the
SSL family's "sinusoidal") is the failure mode that does not raise.
"""
from __future__ import annotations

import json

import pytest
import torch

from market_jepa.eval.checkpoints import load_encoder
from market_jepa.modeling import create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig

N_INFO, N_DATA, T = 11, 9, 256
N_FEAT = N_DATA + N_INFO

# SINUSOIDAL ON PURPOSE: it is what every arm in the six-month wave trains, and
# it differs from TransformerConfig's default ("learned") in the PARAMETER SET,
# so a rebuild that ignores the saved config cannot load these weights at all.
_CFG = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=128, patch_size=8, drop_path_rate=0.0,
            pos_embed="sinusoidal")


def _backbone():
    return create_backbone(
        backbone_type="transformer", n_features=N_FEAT, d_embedding=64,
        config=TransformerConfig(**_CFG), pool="mean", max_seq_len=T,
        n_info_channels=N_INFO)


def _view(b=2):
    """A view shaped like the real one: the payload only on the last step."""
    torch.manual_seed(0)
    x = torch.randn(b, N_FEAT, T)
    x[:, N_DATA:, :-1] = 0
    return x


def _build(name):
    from market_jepa.modeling import CPC, IJEPA, MAE, TimeMAE, TS2Vec
    from market_jepa.modeling.modes.byol import BYOL
    from market_jepa.modeling.modes.cost import CoST
    from market_jepa.modeling.modes.dino import DINO
    from market_jepa.modeling.modes.lejepa import LeJEPA
    from market_jepa.modeling.modes.tfc import TFC
    return {
        "byol": lambda: BYOL(backbone=_backbone()),
        "cost": lambda: CoST(backbone=_backbone()),
        "cpc": lambda: CPC(backbone=_backbone()),
        "dino": lambda: DINO(backbone=_backbone()),
        "lejepa": lambda: LeJEPA(backbone=_backbone()),
        "mae": lambda: MAE(backbone=_backbone()),
        "tfc": lambda: TFC(backbone=_backbone()),
        "timemae": lambda: TimeMAE(backbone=_backbone(), vocab_size=32,
                                   reg_layers=2),
        "ts2vec": lambda: TS2Vec(backbone=_backbone()),
    }[name]()


# Every mode that writes a config.json. IJEPA is absent BECAUSE it defines no
# save_pretrained -- it gets its own test below, on the layout it does produce.
SAVING_MODES = ["byol", "cost", "cpc", "dino", "lejepa", "mae", "tfc",
                "timemae", "ts2vec"]


@pytest.mark.parametrize("name", SAVING_MODES)
def test_the_saved_checkpoint_reloads_and_forwards(name, tmp_path):
    model = _build(name).eval()
    out = tmp_path / name
    model.save_pretrained(str(out))
    assert (out / "config.json").is_file(), (
        f"{name}.save_pretrained wrote no config.json; if that is deliberate, "
        f"move it to the configless test below")

    enc = load_encoder(str(out)).eval()
    x, lengths = _view(), torch.full((2,), T, dtype=torch.long)
    with torch.no_grad():
        emb = enc(x[:, :enc.n_features], lengths)
    assert emb.shape[0] == 2 and emb.ndim == 2
    assert enc.n_features == N_FEAT, (
        f"{name} came back {enc.n_features} wide, not {N_FEAT}: the "
        f"9/11 information split was not recovered from the weights")


@pytest.mark.parametrize("name", SAVING_MODES)
def test_the_reloaded_backbone_is_the_one_that_was_saved(name, tmp_path):
    """Bit-identical backbone weights, not merely a load that did not raise."""
    model = _build(name).eval()
    out = tmp_path / name
    model.save_pretrained(str(out))
    enc = load_encoder(str(out)).eval()

    got = getattr(enc, "model", enc)          # unwrap EncodeAdapter
    got = getattr(got, "backbone", got)
    want = model.backbone.state_dict()
    assert set(got.state_dict()) == set(want), (
        f"{name}: the rebuilt backbone has a different parameter set")
    for k, v in want.items():
        assert torch.equal(got.state_dict()[k], v), f"{name}: {k} differs"


def test_a_configless_checkpoint_rebuilds_from_train_meta(tmp_path):
    """The IJEPA layout: model.pt + train_meta.json, no config.json.

    IJEPA defines no save_pretrained, so this is what all 27 of the
    2026-09-15 six-month I-JEPA checkpoints look like on disk. Before the fix
    this branch built TransformerConfig() defaults and died on info_proj.
    """
    from market_jepa.modeling import IJEPA

    model = IJEPA(backbone=_backbone()).eval()
    out = tmp_path / "ijepa"
    out.mkdir()
    torch.save(model.state_dict(), out / "model.pt")
    # The MODE's block, which is where the trainer records the real one; the
    # top-level block is deliberately wrong here, as it is on disk.
    (out / "train_meta.json").write_text(json.dumps({"config": {
        "backbone": {"pool": None, "config": {"pos_embed": None}},
        "mode": {"backbone": {"d_embedding": 64, "pool": "mean",
                              "max_seq_len": T, "n_info_channels": 0,
                              "config": _CFG}},
    }}))
    assert not (out / "config.json").exists()

    enc = load_encoder(str(out), "ssl-6mo-ijepa-206b47-2008-02-01-2008-07-31",
                       pool="mean").eval()
    assert enc.n_features == N_FEAT
    assert enc.n_info_channels == N_INFO, (
        "n_info_channels is recorded as 0 in train_meta even for a run that "
        "trained WITH the token, so the weights have to outrank it")
    for k, v in model.backbone.state_dict().items():
        assert torch.equal(enc.state_dict()[k], v), f"{k} differs"


def test_the_configless_branch_honours_the_saved_position_encoding(tmp_path):
    """A negative control on the one knob whose default is wrong.

    Without train_meta the branch falls back to TransformerConfig(), whose
    pos_embed is "learned" -- a different parameter set from the sinusoidal
    table every wave arm trains. That must fail loudly rather than load
    something else, so the test above is testing the meta and not the default.
    """
    from market_jepa.modeling import IJEPA

    model = IJEPA(backbone=_backbone()).eval()
    out = tmp_path / "ijepa_nometa"
    out.mkdir()
    torch.save(model.state_dict(), out / "model.pt")

    with pytest.raises(RuntimeError):
        load_encoder(str(out), "ssl-6mo-ijepa-206b47-2008-02-01-2008-07-31")


@pytest.mark.parametrize("name", SAVING_MODES)
def test_from_pretrained_itself_recovers_the_split(name, tmp_path):
    """Directly, because load_encoder does not reach every mode's own loader.

    Only the four ENCODE_MODE classes are dispatched to ``from_pretrained``;
    cpc and mae go through ``load_backbone`` instead, so their copy of the same
    dropped-kwargs bug is invisible to the tests above. It is still a bug --
    resume and any direct caller hit it -- so it is asserted where it lives.
    """
    model = _build(name).eval()
    out = tmp_path / name
    model.save_pretrained(str(out))

    back = type(model).from_pretrained(str(out))
    assert back.backbone.n_features == N_FEAT
    assert back.backbone.n_info_channels == N_INFO
