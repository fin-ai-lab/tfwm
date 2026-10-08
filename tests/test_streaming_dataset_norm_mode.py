"""The two knobs that make an arm's panel differ from the default one.

``dataset.norm_mode`` is expressed as an EMPTY GROUP LIST, on purpose.

``normalize_numpy`` is called from eight places in the dataset -- global views,
local views, the pair path, the probe path -- and a boolean checked at each one
is a boolean that will eventually be forgotten at one of them. Emptying the
group list makes every call site a no-op at once, so the ablation cannot be
half-applied. This test pins that representation, because a future refactor
that "cleans it up" into a per-call-site flag would reintroduce exactly the
risk it was chosen to remove.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from market_jepa.training.streaming_dataset import (
    FEATURE_COLUMNS,
    StreamingMarketDataset,
)
from market_jepa.training.utils import build_norm_groups, normalize_numpy


def _sample():
    """Positive features: three of the four groups are log1p'd first."""
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 1e5, size=(64, len(FEATURE_COLUMNS)))


def test_empty_groups_make_normalize_a_noop():
    a = _sample()
    b = a.copy()
    normalize_numpy(b, [])
    assert np.array_equal(a, b)


def test_default_groups_do_normalize():
    a = _sample()
    b = a.copy()
    normalize_numpy(b, build_norm_groups(FEATURE_COLUMNS))
    assert np.isfinite(b).all()
    # Raw features span 1..1e5; standardized ones sit within a few sigma.
    assert np.abs(b).max() < 10 < np.abs(a).max()


def test_dataset_accepts_the_knob_and_rejects_a_typo():
    """The parameter exists and only the two documented values are legal."""
    sig = inspect.signature(StreamingMarketDataset.__init__)
    assert sig.parameters["norm_mode"].default == "per_view"
    src = inspect.getsource(StreamingMarketDataset.__init__)
    assert 'if norm_mode not in ("per_view", "none"):' in src
    assert "[] if norm_mode == \"none\"" in src


def test_norm_groups_for_maps_the_dataset_knob():
    """A norm_mode=none checkpoint must be SCORED without normalization.

    The training knob and the eval knob are separate code paths that must
    agree; this is the join between them.
    """
    import sys
    from pathlib import Path

    p = str(Path(__file__).resolve().parents[1] / "scripts" / "eval")
    if p not in sys.path:
        sys.path.insert(0, p)
    from xs_ic_eval import norm_groups_for

    assert norm_groups_for(None) is None
    assert norm_groups_for({"dataset": {}}) is None
    assert norm_groups_for({"dataset": {"norm_mode": "per_view"}}) is None
    assert norm_groups_for({"dataset": {"norm_mode": "none"}}) == []
    with pytest.raises(ValueError):
        norm_groups_for({"dataset": {"norm_mode": "global"}})


def _panel(cfg):
    import sys
    from pathlib import Path

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "eval")
    if q not in sys.path:
        sys.path.insert(0, q)
    from xs_ic_eval import panel_kwargs_for
    return panel_kwargs_for(cfg)


def _aug(a):
    return {"dataset": {"augmentations": {"0": a}}}


def test_fixed_agg_reads_the_seconds_band_first():
    """A pinned global_agg_range is the resolution, verbatim."""
    assert _panel(_aug({"global_agg_range": [8, 8]}))["fixed_agg"] == 8
    # A real band is not a pinned point.
    assert _panel(_aug({"global_agg_range": [6, 11]}))["fixed_agg"] is None


def test_fixed_agg_falls_back_to_a_degenerate_scale_range():
    """Older runs pinned the fraction; convert on the STANDARD session.

    global_scale_range is a fraction of the trimmed grid length of THAT
    ticker-day, so a late-opening stock would get a different resolution from
    a full-session one in the same cross-section. The standard session is the
    only reading under which a cell has a single resolution.
    """
    assert _panel(_aug({"global_scale_range": [0.75, 0.75]}))["fixed_agg"] == 9
    assert _panel(_aug({"global_scale_range": [0.5, 1.0]}))["fixed_agg"] is None


DEFAULT_PANEL = {"norm_groups": None, "fixed_agg": None, "info_norm_stats": False,
                 "seq_len": 2048, "info_window": False}


def test_default_config_asks_for_the_default_panel():
    assert _panel({"dataset": {}}) == DEFAULT_PANEL
    assert _panel(None) == DEFAULT_PANEL


def test_the_cluster_scoring_path_replays_the_panel_too(tmp_path):
    """probe_fit_size is what our cluster runs; it must not be a weaker loader.

    It used to build its own ViT-384/9-feature config and drop the dataset
    block entirely, so every panel knob vanished on the cluster path: a run
    pinned to one resolution, or to a shorter view, was embedded on the DEFAULT
    panel and scored on views it never trained on. No error, plausible number.
    Local scoring replayed the panel and cluster scoring did not, which is the
    worst version -- the two disagree only for the arms that need it.
    """
    import sys
    from pathlib import Path as _P
    from market_jepa.training.utils import save_train_meta
    # ONLY the directory the module itself lives in. probe_fit_size resolves
    # everything else it imports from its own ROOT-relative inserts, and a test
    # that pre-seeds those hides a missing one -- which is exactly how the
    # scripts/generic insert went missing and left the module unimportable on
    # the cluster while this test still passed.
    x = str(_P(__file__).resolve().parents[1] / "scripts" / "eval")
    if x not in sys.path:
        sys.path.insert(0, x)
    from probe_fit_size import _panel_for

    d = tmp_path / "run"
    d.mkdir()
    save_train_meta({"mode": {"task": "return_900"},
                     "dataset": {"augmentations": {"0": {
                         "n_global_views": 1, "n_local_views": 0,
                         "global_agg_range": [1, 1], "global_seq_len": 256}}}},
                    d, task_name="return_900")
    _, panel = _panel_for(d)
    assert panel["fixed_agg"] == 1, "cluster path dropped the pinned resolution"
    assert panel["seq_len"] == 256, "cluster path dropped the shortened view"


def test_every_panel_kwarg_is_accepted_by_the_functions_it_is_splatted_into():
    """panel_kwargs_for is passed as ``**kwargs`` -- adding a key breaks callers.

    ``seq_len`` was added to panel_kwargs_for and to iter_panel but NOT to
    embed_month, which post_train_ic_eval splats it into. Training was
    unaffected, so 65 jobs trained for hours and then died at the scoring step
    with ``embed_month() got an unexpected keyword argument 'seq_len'`` --
    after the compute was spent, and only visible in a per-job log.
    """
    import inspect
    from xs_ic_eval import embed_month, embed_month_many, iter_panel, panel_kwargs_for

    keys = set(panel_kwargs_for(None))
    for fn in (iter_panel, embed_month, embed_month_many):
        params = set(inspect.signature(fn).parameters)
        missing = keys - params
        assert not missing, f"{fn.__name__} does not accept {sorted(missing)}"


def test_seq_len_is_read_and_defaults_to_the_full_view():
    """The OTHER way to shorten the context, and it must reach the panel.

    global_agg_range holds the token count and shrinks each token; this holds
    the token size and feeds fewer of them. A scorer that replays one but not
    the other builds 2048-token views for a model trained on 256 -- eight times
    the context it ever saw, scored without an error anywhere.
    """
    assert _panel(_aug({"global_seq_len": 256}))["seq_len"] == 256
    assert _panel(_aug({}))["seq_len"] == 2048
    # A degenerate scale range converts against the run's OWN token count, not
    # the module default, or a short view reports the wrong resolution.
    assert _panel(_aug({"global_scale_range": [0.75, 0.75],
                        "global_seq_len": 256}))["fixed_agg"] == 69


def test_a_pinned_agg_costs_eval_anchors_and_says_which():
    """8 s/token spans 16,384 s, so the earliest anchors admit no view.

    Pinned because it is the one thing that makes a fixed-agg arm's IC not
    directly comparable to a free-scale arm's: fewer cells, and specifically
    the LATER ones. A paired comparison has to restrict the control to these.
    """
    import sys
    from pathlib import Path

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "eval")
    if q not in sys.path:
        sys.path.insert(0, q)
    from xs_ic_eval import GLOBAL_SEQ_LEN, day_anchors

    anchors = [int(a) for a in day_anchors(8)]
    fits = [a for a in anchors if 8 * GLOBAL_SEQ_LEN <= a + 1]
    assert len(anchors) == 8
    assert len(fits) == 5
    assert fits == anchors[-5:]        # the five LATEST, not an arbitrary five


def test_panel_knobs_survive_the_train_meta_round_trip(tmp_path):
    """save_train_meta -> train_meta.json -> _load_cfg -> panel_kwargs_for.

    THE JOIN THAT MATTERS, and the one that was broken. train_meta.json does
    not carry a hydra config -- it carries a flat schema -- so an arm's panel
    knobs reach the in-job scorer only if they are written into it explicitly.
    Without this, `nonorm` is scored on normalized views and a pinned-agg run
    is scored across the 6-11 band, and neither raises: both just return a
    worse number that reads as a result.
    """
    import json
    import sys
    from pathlib import Path

    from market_jepa.training.utils import save_train_meta

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    r = str(Path(__file__).resolve().parents[1] / "scripts" / "eval")
    for x in (q, r):
        if x not in sys.path:
            sys.path.insert(0, x)
    from post_train_ic_eval import _load_cfg
    from xs_ic_eval import panel_kwargs_for

    def roundtrip(dataset):
        d = tmp_path / dataset.get("_name", "run")
        d.mkdir(exist_ok=True)
        save_train_meta({"mode": {"task": "return_900"}, "dataset": dataset},
                        d, task_name="return_900")
        assert json.loads((d / "train_meta.json").read_text())
        return panel_kwargs_for(_load_cfg(d, None))

    aug = {"0": {"n_global_views": 1, "n_local_views": 0}}
    assert roundtrip({"_name": "a", "augmentations": aug}) == {
        "norm_groups": None, "fixed_agg": None, "info_norm_stats": False,
        "seq_len": 2048, "info_window": False}
    assert roundtrip({"_name": "b", "augmentations": aug,
                      "norm_mode": "none"}) == {
        "norm_groups": [], "fixed_agg": None, "info_norm_stats": False,
        "seq_len": 2048, "info_window": False}
    pinned = {"0": {"n_global_views": 1, "n_local_views": 0,
                    "global_agg_range": [8, 8]}}
    assert roundtrip({"_name": "c", "augmentations": pinned}) == {
        "norm_groups": None, "fixed_agg": 8, "info_norm_stats": False,
        "seq_len": 2048, "info_window": False}
    # A SHORT view has to survive the round trip for the same reason a pinned
    # agg does: it is the other way to change the context, and missing it is
    # silent rather than fatal.
    short = {"0": {"n_global_views": 1, "n_local_views": 0,
                   "global_agg_range": [6, 11], "global_seq_len": 256}}
    assert roundtrip({"_name": "e", "augmentations": short}) == {
        "norm_groups": None, "fixed_agg": None, "info_norm_stats": False,
        "seq_len": 256, "info_window": False}
    # The third knob is not merely a panel knob -- it changes the input WIDTH,
    # so the round trip has to carry it or the backbone is built 9-wide and
    # the checkpoint will not load at all.
    assert roundtrip({"_name": "d", "augmentations": aug,
                      "info_norm_stats": True}) == {
        "norm_groups": None, "fixed_agg": None, "info_norm_stats": True,
        "seq_len": 2048, "info_window": False}
    # dataset.info_window is the same class of knob and worse if lost: it widens
    # the tensor by 3, and the backbone strips a FIXED count off the end. Emit
    # them at eval for a model trained without them and the count is right
    # while the contents are not -- which scores silently wrong rather than
    # failing to load.
    assert roundtrip({"_name": "f", "augmentations": aug,
                      "info_norm_stats": True, "info_window": True}) == {
        "norm_groups": None, "fixed_agg": None, "info_norm_stats": True,
        "seq_len": 2048, "info_window": True}


def test_train_meta_reads_an_omegaconf_config(tmp_path):
    """The config a hydra run passes is a DictConfig, not a dict.

    ``isinstance(DictConfig, dict)`` is False, so the augmentation mapping was
    never converted to a list: iterating it yielded its string keys, and every
    checkpoint recorded ``n_global_views: 0``. Verified against the archive --
    every supervised-bins-penalty run has 0. The panel knobs sit in the same
    block and would have crashed on ``augs[0]``, which is how it surfaced.
    """
    import json

    from omegaconf import OmegaConf

    from market_jepa.training.utils import save_train_meta

    cfg = OmegaConf.create({
        "mode": {"task": "return_900", "loss_fn": "cross_entropy", "n_bins": 11},
        "dataset": {
            "norm_mode": "per_view",
            "augmentations": {"0": {"n_global_views": 1, "n_local_views": 0,
                                    "global_agg_range": [8, 8]}},
        },
    })
    save_train_meta(cfg, tmp_path, task_name="return_900")
    meta = json.loads((tmp_path / "train_meta.json").read_text())
    assert meta["n_global_views"] == 1
    assert meta["global_agg_range"] == [8, 8]
    assert "norm_mode" not in meta          # default is not recorded


def test_train_meta_prefers_the_resolved_augmentations(tmp_path):
    """mode.dataset_overrides rewrites the view counts after compose.

    The supervised modes drop to one global view and no locals, so the config
    says 2 and the run used 1. Recording 2 would be a plausible wrong number,
    which is worse than the obviously-broken 0 it replaced.
    """
    import json

    from omegaconf import OmegaConf

    from market_jepa.training.utils import save_train_meta

    cfg = OmegaConf.create({
        "mode": {"task": "return_900"},
        "dataset": {"augmentations": {"0": {"n_global_views": 2,
                                            "n_local_views": 6}}},
    })
    resolved = [{"n_global_views": 1, "n_local_views": 0,
                 "global_agg_range": [8, 8]}]
    save_train_meta(cfg, tmp_path, task_name="return_900",
                    augmentations=resolved)
    meta = json.loads((tmp_path / "train_meta.json").read_text())
    assert meta["n_global_views"] == 1
    assert meta["n_local_views"] == 0
    assert meta["num_views"] == 1
    assert meta["global_agg_range"] == [8, 8]


def test_pool_survives_the_round_trip(tmp_path):
    """A pool=last checkpoint cannot be LOADED as pool=cls.

    The backbone allocates a cls_token only under "cls" and sizes its
    positional embedding 2049 vs 2048, so guessing wrong is not a mis-score --
    it is a state_dict failure. _load_cfg hardcoded "cls" because nothing else
    had been trained; this pins the read-back that makes that safe.
    """
    import json
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf

    from market_jepa.training.utils import save_train_meta

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    if q not in sys.path:
        sys.path.insert(0, q)
    from post_train_ic_eval import _load_cfg

    aug = [{"n_global_views": 1, "n_local_views": 0}]
    for pool, expected in (("last", "last"), ("cls", "cls"), (None, "cls")):
        d = tmp_path / f"p_{pool}"
        d.mkdir()
        bb = {} if pool is None else {"pool": pool}
        save_train_meta(OmegaConf.create({"mode": {"task": "return_900"},
                                          "backbone": bb, "dataset": {}}),
                        d, task_name="return_900", augmentations=aug)
        meta = json.loads((d / "train_meta.json").read_text())
        # Only the non-default is recorded; the default is the fallback.
        assert ("pool" in meta) == (pool == "last")
        assert _load_cfg(d, None)["backbone"]["pool"] == expected


def test_the_two_poolings_are_load_incompatible():
    """Pins WHY pool has to be recorded, not just that it is."""
    import pytest

    from market_jepa.modeling.backbones.transformer import TransformerBackbone
    from market_jepa.schemas import TransformerInnerConfig

    def mk(pool):
        return TransformerBackbone(n_features=9, d_embedding=384,
                                    config=TransformerInnerConfig(), pool=pool)

    assert "cls_token" in mk("cls").state_dict()
    assert "cls_token" not in mk("last").state_dict()
    with pytest.raises(RuntimeError):
        mk("cls").load_state_dict(mk("last").state_dict())


def test_causal_moves_cls_to_the_end():
    """A prepended CLS under a causal mask is a CONSTANT, not a weak readout.

    triu(diagonal=1) means position i attends to j <= i, so CLS at index 0
    attends to nothing but itself and two different inputs give a bit-identical
    embedding -- which trains to a flat curve and reads as a modelling failure
    rather than a config error. Appending it keeps it a learned query over the
    whole view, and keeps a causal run a ONE-KNOB change from the control.
    """
    import torch

    from market_jepa.modeling.backbones.transformer import TransformerBackbone
    from market_jepa.schemas import TransformerInnerConfig

    def mk(**kw):
        return TransformerBackbone(n_features=9, d_embedding=384,
                                    config=TransformerInnerConfig(), **kw).eval()

    assert mk(pool="cls", causal=True).cls_at_end is True
    assert mk(pool="cls").cls_at_end is False
    assert mk(pool="last", causal=True).cls_at_end is False

    x1, x2 = torch.randn(2, 9, 256), torch.randn(2, 9, 256)
    lens = torch.full((2,), 256, dtype=torch.long)
    m = mk(pool="cls", causal=True)
    with torch.no_grad():
        assert (m(x1, lens) - m(x2, lens)).abs().max() > 1e-3
        # And it must see the EARLIEST patch, not just recent ones -- that is
        # the whole point of appending rather than prepending.
        x3 = x1.clone()
        x3[:, :, :8] += 5.0
        assert (m(x1, lens) - m(x3, lens)).abs().max() > 1e-4
        # The mask has to actually bite: same weights, causal vs not.
        b = mk(pool="cls")
        b.load_state_dict(m.state_dict())
        assert (m(x1, lens) - b(x1, lens)).abs().max() > 1e-3
    # Ragged lengths keep CLS valid at the final index.
    assert m(x1, torch.tensor([256, 128])).shape == (2, 384)


def test_causal_is_silent_on_a_wrong_load_so_it_must_be_recorded(tmp_path):
    """Causal masking adds NO parameters, unlike pool.

    A causal checkpoint therefore loads into a bidirectional backbone without
    complaint and is scored as something it is not -- no exception, just a
    worse number. That is why train_meta has to carry it.
    """
    import json
    import sys
    from pathlib import Path

    import torch
    from omegaconf import OmegaConf

    from market_jepa.modeling.backbones.transformer import TransformerBackbone
    from market_jepa.schemas import TransformerInnerConfig
    from market_jepa.training.utils import save_train_meta

    def mk(**kw):
        return TransformerBackbone(n_features=9, d_embedding=384,
                                    config=TransformerInnerConfig(), **kw)

    # The silent part: state dicts are interchangeable.
    mk(pool="last").load_state_dict(mk(pool="last", causal=True).state_dict())

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    if q not in sys.path:
        sys.path.insert(0, q)
    from post_train_ic_eval import _load_cfg

    aug = [{"n_global_views": 1, "n_local_views": 0}]
    for causal in (True, False):
        d = tmp_path / f"c_{causal}"
        d.mkdir()
        save_train_meta(
            OmegaConf.create({"mode": {"task": "return_900"},
                              "backbone": {"pool": "last", "causal": causal},
                              "dataset": {}}),
            d, task_name="return_900", augmentations=aug)
        assert ("causal" in json.loads((d / "train_meta.json").read_text())) is causal
        bb = _load_cfg(d, None)["backbone"]
        assert bb["causal"] is causal and bb["pool"] == "last"


def test_state_token_gives_the_anchor_its_own_path(tmp_path):
    """The raw final timestep, projected into its own appended token.

    patch_size timesteps share a patch, so the value AT the anchor is mixed
    with its neighbours -- invertible in principle (72 numbers into 384 is
    over-complete) but something the encoder must learn to invert, while the
    best classical predictor of return_900 is an instantaneous function of
    exactly that row.
    """
    import json

    import torch
    from omegaconf import OmegaConf

    from market_jepa.modeling.backbones.transformer import TransformerBackbone
    from market_jepa.schemas import TransformerInnerConfig
    from market_jepa.training.utils import save_train_meta

    def mk(**kw):
        return TransformerBackbone(n_features=9, d_embedding=384,
                                    config=TransformerInnerConfig(),
                                    max_seq_len=256, **kw).eval()

    base, st = mk(), mk(state_token=True)
    # One extra position, and a projection that did not exist before.
    assert st.position_embeddings.shape[1] == base.position_embeddings.shape[1] + 1
    assert "state_proj.weight" in st.state_dict()
    assert "state_proj.weight" not in base.state_dict()

    x = torch.randn(2, 9, 256)
    lens = torch.full((2,), 256, dtype=torch.long)
    with torch.no_grad():
        assert st(x, lens).shape == (2, 384)
        # It must read the ANCHOR row, not some other one.
        x2 = x.clone(); x2[:, :, -1] += 3.0
        assert (st(x, lens) - st(x2, lens)).abs().max() > 1e-3
    # Ragged lengths: every special token stays valid, only padding is masked.
    assert st(x, torch.tensor([256, 128])).shape == (2, 384)
    # Composes with both poolings and with causal masking.
    for kw in ({"pool": "last"}, {"causal": True}):
        with torch.no_grad():
            assert mk(state_token=True, **kw)(x, lens).shape == (2, 384)

    # And it survives the round trip, like every other knob that changes the
    # model or the view.
    import sys
    from pathlib import Path

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    if q not in sys.path:
        sys.path.insert(0, q)
    from post_train_ic_eval import _load_cfg

    save_train_meta(
        OmegaConf.create({"mode": {"task": "return_900"},
                          "backbone": {"state_token": True}, "dataset": {}}),
        tmp_path, task_name="return_900",
        augmentations=[{"n_global_views": 1, "n_local_views": 0}])
    assert json.loads((tmp_path / "train_meta.json").read_text())["state_token"]
    assert _load_cfg(tmp_path, None)["backbone"]["state_token"] is True


def test_diff_channels_double_the_input_and_match_torch_diff(tmp_path):
    """First differences alongside levels, computed in the backbone.

    Every classical predictor that works on this panel is a difference or a
    position within a range -- Ridge ARDL's return_900 coefficients are -ask(t),
    -bid(t), +high(t), +low(t) -- and the encoder is handed levels. Computed
    in the backbone rather than the dataset because it is a pure function of
    the view: nothing to plumb through iter_panel, and no way for train and
    eval to disagree about it.
    """
    import json

    import torch
    from omegaconf import OmegaConf

    from market_jepa.modeling.backbones.transformer import TransformerBackbone
    from market_jepa.schemas import TransformerInnerConfig
    from market_jepa.training.utils import save_train_meta

    def mk(**kw):
        return TransformerBackbone(n_features=9, d_embedding=384,
                                    config=TransformerInnerConfig(),
                                    max_seq_len=256, **kw).eval()

    assert mk().patch_embed.proj.weight.shape[1] == 9
    assert mk(diff_channels=True).patch_embed.proj.weight.shape[1] == 18

    x = torch.randn(2, 9, 256)
    lens = torch.full((2,), 256, dtype=torch.long)
    m = mk(diff_channels=True)
    with torch.no_grad():
        assert m(x, lens).shape == (2, 384)
    # The leading column is ZERO rather than dropped, so the sequence length --
    # and therefore the patch grid and every position embedding -- is unchanged.
    d = torch.zeros_like(x)
    d[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    assert torch.allclose(d[:, :, 1:], x.diff(dim=-1))
    assert (d[:, :, 0] == 0).all()
    assert m.position_embeddings.shape[1] == mk().position_embeddings.shape[1]

    # Composes with every other knob.
    for kw in ({"state_token": True}, {"pool": "last"}, {"causal": True}):
        with torch.no_grad():
            assert mk(diff_channels=True, **kw)(x, lens).shape == (2, 384)

    import sys
    from pathlib import Path

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    if q not in sys.path:
        sys.path.insert(0, q)
    from post_train_ic_eval import _load_cfg

    save_train_meta(
        OmegaConf.create({"mode": {"task": "return_900"},
                          "backbone": {"diff_channels": True}, "dataset": {}}),
        tmp_path, task_name="return_900",
        augmentations=[{"n_global_views": 1, "n_local_views": 0}])
    assert json.loads((tmp_path / "train_meta.json").read_text())["diff_channels"]
    assert _load_cfg(tmp_path, None)["backbone"]["diff_channels"] is True


def test_create_backbone_forwards_the_transformer_knobs():
    """The transformer branch DROPPED **backbone_kwargs until 2026-08-22.

    Every other branch forwarded them; this one listed its arguments and
    silently discarded the rest. Training never noticed, because hydra
    instantiates the backbone from the config directly -- only the eval path
    goes through create_backbone. So a knob set at train time was simply
    absent at score time.

    For state_token and diff_channels that failed loudly (the parameter set
    differs). For CAUSAL it did not: causal masking adds no parameters, so the
    checkpoint loaded and was scored as if it had been trained bidirectionally
    -- a wrong number with no error anywhere.
    """
    from market_jepa.modeling.backbones import create_backbone
    from market_jepa.modeling.backbones.transformer import TransformerConfig

    def mk(**kw):
        pool = kw.pop("pool", "cls")
        return create_backbone(backbone_type="transformer", n_features=9,
                               d_embedding=384, config=TransformerConfig(),
                               pool=pool, **kw)

    assert mk().patch_embed.proj.weight.shape[1] == 9
    assert mk(diff_channels=True).patch_embed.proj.weight.shape[1] == 18
    assert mk(state_token=True).state_proj is not None
    assert mk(causal=True, pool="last").causal is True
    assert mk().causal is False


# ---------------------------------------------------------------------------
# info_norm_stats: handing back the (mu, sigma) standardization discards
# ---------------------------------------------------------------------------


def test_the_widening_path_normalizes_in_place():
    """The default path must be the SAME OBJECT and the same numbers.

    Every call site in the dataset rebinds to this function's return value, so
    a copy here would be a silent doubling of the per-view allocation on the
    hot path.
    """
    from market_jepa.augmentations import prepare_augmented_view

    a = _sample()
    b = a.copy()
    groups = build_norm_groups(FEATURE_COLUMNS)
    normalize_numpy(a, groups)
    out, _meta = prepare_augmented_view(
        b, groups, start_seconds=0.0, aggregation_seconds=1.0,
    )
    assert out is b
    assert np.array_equal(a, b)


def test_the_widening_path_widens_and_leaves_the_signal_alone():
    from market_jepa.augmentations import (
        encode_view_metadata, prepare_augmented_view,
    )
    from market_jepa.training.utils import append_view_info

    n_feat = len(FEATURE_COLUMNS)
    a = _sample()
    b = a.copy()
    groups = build_norm_groups(FEATURE_COLUMNS)
    normalize_numpy(a, groups)
    b, meta = prepare_augmented_view(
        b, groups, start_seconds=11700.0, aggregation_seconds=8.0,
    )
    out = append_view_info(b, encode_view_metadata(
        meta, include_normalization=True, include_window=False))

    assert out.shape == (len(a), n_feat + 2 * len(groups))
    # The standardized channels are untouched: this arm ADDS information, it
    # does not trade any away. If these ever differ, normstats stops being
    # comparable to the control on anything but its own terms.
    assert np.array_equal(out[:, :n_feat], a)
    # Window metadata occupies only its reserved final-row payload.
    stats = out[:, n_feat:]
    assert np.count_nonzero(stats[:-1]) == 0
    assert np.isfinite(stats[-1]).all()


def test_encoded_norm_stats_are_on_a_usable_scale():
    """A $400 stock and a $4 stock must not differ by two orders of magnitude.

    Raw mu would put a price level (~400) in the same vector as a log-count
    (~2), and the patch projection would see one channel that dwarfs the nine
    standardized ones. The signed log1p / log mapping is what makes the extra
    channels information rather than a scaling accident.
    """
    from stable_finance.dataset import ViewMetadata
    from market_jepa.augmentations import encode_view_metadata

    def stats(mean, scale):
        meta = ViewMetadata(
            start_seconds=0.0, end_seconds=1.0, aggregation_seconds=1.0,
            normalization_means=np.array([mean], dtype=np.float64),
            normalization_scales=np.array([scale], dtype=np.float64),
        )
        return encode_view_metadata(
            meta, include_normalization=True, include_window=False)

    cheap = stats(4.0, 0.02)
    rich = stats(400.0, 2.0)
    assert np.abs(np.concatenate([cheap, rich])).max() < 10
    # Still ORDERED: the mapping compresses, it does not erase.
    assert rich[0] > cheap[0]
    assert rich[1] > cheap[1]
    # Negative mu (a centred group) stays negative rather than folding over.
    assert stats(-4.0, 1.0)[0] < 0


def test_stats_out_reports_exactly_what_was_divided_out():
    from market_jepa.training.utils import normalize_numpy as nn_

    groups = build_norm_groups(FEATURE_COLUMNS)
    a = _sample()
    stats: list[tuple[float, float]] = []
    nn_(a, groups, stats_out=stats)
    assert len(stats) == len(groups) == 4
    # Recover the group: standardized * sigma + mu is the log1p'd original.
    raw = _sample()
    for (indices, log1p), (mu, sd) in zip(groups, stats):
        want = np.log1p(raw[:, indices]) if log1p else raw[:, indices]
        assert np.allclose(a[:, indices] * sd + mu, want)


def test_dataset_widens_n_features_and_rejects_the_meaningless_pair():
    """9 -> 17, and norm_mode=none + stats is refused rather than ignored.

    With no normalization there is no (mu, sigma) to report, so the flag would
    widen the view by ZERO -- a model built for 17 channels fed 9. That fails
    at load, three days into a sweep, which is why it is refused here.
    """
    src = inspect.getsource(StreamingMarketDataset.__init__)
    sig = inspect.signature(StreamingMarketDataset.__init__)
    # ON BY DEFAULT (flipped 2026-09-13), and asserted here because the default
    # is the whole point: DatasetConfig and DayStoreCellDataset have always
    # said True, and this constructor saying False made the view width a
    # property of which class you happened to construct.
    assert sig.parameters["info_norm_stats"].default is True
    assert sig.parameters["info_window"].default is True
    assert "info_norm_stats and norm_mode == \"none\"" in src
    assert "2 * len(self._norm_groups) if self.info_norm_stats" in src
    n_features_src = inspect.getsource(StreamingMarketDataset.n_features.fget)
    assert "_n_info_features" in n_features_src


def test_every_call_site_goes_through_the_widening_helper():
    """No bare normalize_numpy left in the dataset.

    Concatenation is not in-place, so a call site that did not REBIND would
    silently keep the 9-channel view while the rest of the run assumed 17 --
    the same class of half-applied ablation the empty-group-list trick exists
    to prevent, but this one is a shape mismatch rather than a no-op.
    """
    import market_jepa.training.streaming_dataset as sd

    src = inspect.getsource(sd)
    body = src.split("def _normalize_view")[0]
    assert "normalize_numpy(" not in body
    assert body.count("self._normalize_view(") == 7


def test_the_eval_panel_uses_the_same_helper_in_the_same_place():
    """Training and eval must widen identically, and BEFORE the rf merge.

    Two copies of the concatenation would drift into a channel-order
    mismatch, which does not crash -- the widths agree -- it just feeds the
    encoder its price level where it expects its depth.
    """
    import sys
    from pathlib import Path

    r = str(Path(__file__).resolve().parents[1] / "scripts" / "eval")
    if r not in sys.path:
        sys.path.insert(0, r)
    import xs_ic_eval

    src = inspect.getsource(xs_ic_eval._panel_for_ticker_day)
    i_norm = src.index("build_sample_panel(")
    i_rf = src.index("rf_merger.merge(")
    assert i_norm < i_rf
    assert "normalize_numpy(" not in src
    assert xs_ic_eval.info_norm_stats_for({"dataset": {"info_norm_stats": True}})
    assert not xs_ic_eval.info_norm_stats_for({"dataset": {}})
    assert not xs_ic_eval.info_norm_stats_for(None)


def test_n_features_reaches_the_backbone_through_the_scorer():
    """post_train_ic_eval sizes the backbone off the flag, not off a constant.

    This is the join that decides whether the arm can be scored at all: the
    9 is hardcoded there, and a 17-channel checkpoint loaded into a 9-channel
    patch embedding is a state-dict shape error at eval time.
    """
    import json
    import sys
    from pathlib import Path

    q = str(Path(__file__).resolve().parents[1] / "scripts" / "generic")
    if q not in sys.path:
        sys.path.insert(0, q)
    from post_train_ic_eval import _load_cfg

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "train_meta.json").write_text(json.dumps(
            {"task": "return_900", "pool": "last", "info_norm_stats": True}))
        assert _load_cfg(d, None)["n_features"] == 17
        (d / "train_meta.json").write_text(json.dumps(
            {"task": "return_900", "pool": "last"}))
        assert _load_cfg(d, None)["n_features"] == 9


def test_the_cluster_entry_points_import_under_a_bare_interpreter():
    """The cluster runs these as scripts, not as pytest imports.

    ``uv run scripts/eval/probe_fit_size.py embed-many`` gets a
    sys.path holding the script's own directory and nothing else, so every
    cross-tree import has to be reachable from the module's own ROOT-relative
    inserts. Importing from inside pytest proves nothing: the session has
    already put half the tree on sys.path. Run them the way our cluster does.
    """
    import subprocess
    import sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    for rel in ("scripts/eval/probe_fit_size.py",
                "scripts/eval/xs_score_many.py",
                "scripts/eval/xs_ic_series.py"):
        r = subprocess.run([sys.executable, str(root / rel), "--help"],
                           cwd=str(root), capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, f"{rel} is not runnable as a script:\n{r.stderr[-2000:]}"
