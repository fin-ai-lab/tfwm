"""Tests for SSL finetuning: pretrained-backbone init, the ridge-initialized
head, and the StreamingMarketDataset data_fraction subset.

The head-only arm (``freeze_backbone``) and the multihead finetune path were
retired 2026-09-16 and their tests with them; what replaces them is the ridge
init below, whose whole claim is that the head's step-0 output IS the probe.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from market_jepa.modeling.backbones.transformer import (
    TransformerBackbone,
    TransformerConfig,
)
from market_jepa.modeling.modes.lejepa import LeJEPA
from market_jepa.eval.heads import SkipRegressionHead
from market_jepa.modeling.modes.supervised import SupervisedModel


TINY = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    patch_size=8,
)


def _tiny_backbone(seed: int) -> TransformerBackbone:
    torch.manual_seed(seed)
    return TransformerBackbone(
        n_features=9, d_embedding=32, pool="cls", max_seq_len=256,
        config=TransformerConfig(**TINY),
    )


@pytest.fixture
def lejepa_ckpt(tmp_path):
    """A LeJEPA save_pretrained checkpoint with a tiny transformer."""
    model = LeJEPA(backbone=_tiny_backbone(seed=0), proj_dim=8, n_projections=4)
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return model, str(path)


def test_init_backbone_from_loads_weights(lejepa_ckpt):
    ssl_model, path = lejepa_ckpt
    m = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=path
    )
    for k, v in ssl_model.backbone.state_dict().items():
        assert torch.equal(m.backbone.state_dict()[k], v), k
    # e2e default: everything trainable
    assert all(p.requires_grad for p in m.parameters())


def test_init_backbone_from_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900",
            init_backbone_from=str(tmp_path / "nope"),
        )




# ── The ridge-initialized head ─────────────────────────────────────────────


def _write_ridge(tmp_path, d=32, task="return_900", seed=0, readout="last"):
    """A head-init directory shaped like fit_ridge_head_init.py's output."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d).astype(np.float32)
    d_dir = tmp_path / "ridge"
    d_dir.mkdir(exist_ok=True)
    # pred_std is the probe's spread on its fit pool; 0.02 is the measured
    # order of magnitude (return_900 on pair_warp_6mo/2008-08 is 0.0146).
    kw = {} if readout is None else {"readout": readout}
    np.savez(d_dir / f"{task}.npz", weight=w, bias=np.float32(0.5),
             pred_std=np.float32(0.02), n_rows=np.int64(1000),
             alpha=np.float32(10.0), **kw)
    return d_dir, w


def test_skip_head_step0_is_the_ridge(tmp_path):
    """THE CLAIM: at step 0 the head reproduces the probe, exactly."""
    d_dir, w = _write_ridge(tmp_path)
    head = SkipRegressionHead(32)
    head.init_from_ridge(torch.from_numpy(w), 0.5)
    x = torch.randn(64, 32)
    assert torch.allclose(head(x), x @ torch.from_numpy(w) + 0.5, atol=1e-5)


def test_skip_head_mlp_branch_is_zero_but_trainable(tmp_path):
    """The MLP output layer starts at zero; its earlier layers do not.

    A fully zeroed branch could never leave zero -- every weight's gradient
    runs through the output layer -- so only the last layer is zeroed.
    """
    _, w = _write_ridge(tmp_path)
    head = SkipRegressionHead(32)
    head.init_from_ridge(torch.from_numpy(w), 0.0)
    last = [m for m in head.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert torch.count_nonzero(last.weight) == 0
    first = [m for m in head.mlp if isinstance(m, torch.nn.Linear)][0]
    assert torch.count_nonzero(first.weight) > 0

    head(torch.randn(8, 32)).sum().backward()
    assert last.weight.grad is not None
    assert torch.count_nonzero(last.weight.grad) > 0


def test_ridge_init_scale_unit_normalizes_output_not_weights(lejepa_ckpt, tmp_path):
    """'unit' scales the OUTPUT by 1/pred_std and leaves the weights alone.

    This is the whole fix. The rescale has to reach the predictions, because
    the pairwise loss saturates on an IC-sized spread; it must NOT reach the
    weights, because Adam steps a parameter by ~lr regardless of the
    parameter's magnitude, so a 50x heavier skip is a 50x slower head. Wave
    047986 ran with it folded into the weight and the head did not move at
    all: cos(init, final) = 1.000000, with the norm change fully accounted for
    by weight decay.
    """
    _, ckpt = lejepa_ckpt
    d_dir, w = _write_ridge(tmp_path)
    raw = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
        init_head_from=str(d_dir), head_init_scale="raw",
    )
    unit = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
        init_head_from=str(d_dir), head_init_scale="unit",
    )
    # THE WEIGHTS ARE THE PROBE'S OWN, under both scales.
    for m in (raw, unit):
        assert torch.allclose(m.head.skip.weight.flatten(),
                              torch.from_numpy(w), atol=1e-6)
    assert torch.allclose(unit.head.skip.weight, raw.head.skip.weight,
                          rtol=1e-6)
    assert unit.head.skip.bias.item() == pytest.approx(
        raw.head.skip.bias.item(), rel=1e-6)

    # THE OUTPUT CARRIES THE SCALE. pred_std=0.02 -> 50x.
    assert raw.head.gain.item() == pytest.approx(1.0, rel=1e-6)
    assert unit.head.gain.item() == pytest.approx(50.0, rel=1e-5)
    x = torch.randn(16, 32)
    assert torch.allclose(unit.head(x), raw.head(x) * 50.0, rtol=1e-5)


def test_ridge_head_is_scale_matched_to_the_backbone(lejepa_ckpt, tmp_path):
    """The head must not be orders of magnitude heavier than the encoder.

    THE REGRESSION THIS EXISTS FOR. Adam moves a parameter by ~lr per step no
    matter how large the parameter is, so a head whose weights are 28x the
    backbone's RMS takes a 28x smaller RELATIVE step and is, in practice,
    frozen. Wave 047986 measured exactly that: skip RMS 3.78 against a
    backbone RMS of 0.136, cos(w_init, w_final) = 1.000000 over 3,526 steps,
    and a norm change of 0.99837 against 0.998362 predicted by weight decay
    alone. The encoder then drifted out from under a readout that could not
    follow, which is the IC valley the figure shows.

    The supervised arm this finetune is compared against sits at head/backbone
    ~= 0.43, so its head adapts slightly FASTER than its encoder. A pred_std
    of 0.02 is a realistic 50x rescale and must not move this ratio at all.
    """
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path)
    m = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
        init_head_from=str(d_dir), head_init_scale="unit",
    )

    def rms(ts):
        v = torch.cat([t.detach().float().reshape(-1) for t in ts])
        return v.pow(2).mean().sqrt().item()

    # THE PROPERTY, stated so the fixture's arbitrary weight scale cannot
    # decide it: shrinking pred_std by 50x must not change the WEIGHTS by a
    # thing. Under the old fold-it-in code this made the skip 50x heavier and
    # the head 50x slower, while every step-0 prediction stayed identical --
    # which is why it went unnoticed through two waves.
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    import shutil
    shutil.copy(d_dir / "return_900.npz", tiny / "return_900.npz")
    z = dict(np.load(tiny / "return_900.npz"))
    z["pred_std"] = np.float32(0.02 / 50)
    np.savez(tiny / "return_900.npz", **z)
    hotter = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
        init_head_from=str(tiny), head_init_scale="unit",
    )
    assert rms([hotter.head.skip.weight]) == pytest.approx(
        rms([m.head.skip.weight]), rel=1e-6), (
        "skip weights moved with pred_std: the output scale is still being "
        "folded into the parameters, so the head's relative step is that "
        "factor smaller and it will not train"
    )
    assert hotter.head.gain.item() == pytest.approx(2500.0, rel=1e-4)
    # And the gain is where the scale went -- not trainable, so no optimizer
    # sees it and weight decay cannot shrink it.
    assert m.head.gain.item() == pytest.approx(50.0, rel=1e-5)
    assert not isinstance(m.head.gain, torch.nn.Parameter)
    assert "gain" in dict(m.head.named_buffers())


def test_skip_head_loads_a_checkpoint_written_before_gain_existed(tmp_path):
    """gain=1.0 is exactly right for an old state_dict, which folded it in.

    Pre-fix checkpoints carry the rescale inside `skip.weight` and have no
    `gain` key. Defaulting the buffer to 1.0 means such a head loads with
    `strict=False` and still predicts what it predicted, so a missing key is
    not a silent rescale of somebody's saved model.
    """
    from market_jepa.eval.heads import SkipRegressionHead

    old = SkipRegressionHead(32)
    with torch.no_grad():
        old.skip.weight.normal_()
    sd = {k: v for k, v in old.state_dict().items() if k != "gain"}

    fresh = SkipRegressionHead(32)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert list(missing) == ["gain"] and not unexpected
    assert fresh.gain.item() == 1.0
    x = torch.randn(8, 32)
    old.eval(); fresh.eval()
    assert torch.allclose(old(x), fresh(x), atol=1e-6)


def test_ridge_init_swaps_the_head_type(lejepa_ckpt, tmp_path):
    """Without init_head_from the reported arms keep their plain MLP head."""
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path)
    plain = SupervisedModel(_tiny_backbone(seed=1), task="return_900")
    assert not isinstance(plain.head, SkipRegressionHead)
    ft = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
        init_head_from=str(d_dir),
    )
    assert isinstance(ft.head, SkipRegressionHead)
    assert all(p.requires_grad for p in ft.parameters())   # e2e, always


def test_ridge_init_without_backbone_init_raises(tmp_path):
    d_dir, _ = _write_ridge(tmp_path)
    with pytest.raises(ValueError, match="init_head_from"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900", init_head_from=str(d_dir)
        )


def test_ridge_init_missing_task_raises(lejepa_ckpt, tmp_path):
    """The init is per TASK; asking for one that was not fitted must not pass."""
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path, task="return_900")
    with pytest.raises(FileNotFoundError, match="spread_change_900"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="spread_change_900",
            init_backbone_from=ckpt, init_head_from=str(d_dir),
        )


def test_ridge_init_bad_scale_raises(lejepa_ckpt, tmp_path):
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path)
    with pytest.raises(ValueError, match="head_init_scale"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900", init_backbone_from=ckpt,
            init_head_from=str(d_dir), head_init_scale="nope",
        )


def test_ridge_init_width_mismatch_raises(lejepa_ckpt, tmp_path):
    """A probe from a different-width encoder must fail loudly, not broadcast."""
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path, d=64)
    with pytest.raises(ValueError, match="ridge weight"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900",
            init_backbone_from=ckpt, init_head_from=str(d_dir),
        )


class _FractionOnly:
    """Exercise _remap_fraction_idx without constructing a StreamingDataset."""

    _remap_fraction_idx = None  # replaced below

    def __init__(self, num_samples, frac, seed=42):
        self.num_samples = num_samples
        self._data_fraction = float(frac)
        self._base_seed = seed
        self._fraction_subset = None


from market_jepa.training.streaming_dataset import StreamingMarketDataset  # noqa: E402

_FractionOnly._remap_fraction_idx = StreamingMarketDataset._remap_fraction_idx


def test_data_fraction_identity_at_one():
    d = _FractionOnly(1000, 1.0)
    assert [d._remap_fraction_idx(i) for i in (0, 5, 999)] == [0, 5, 999]


def test_data_fraction_subset_size_and_determinism():
    d1 = _FractionOnly(1000, 0.1)
    d2 = _FractionOnly(1000, 0.1)
    mapped1 = {d1._remap_fraction_idx(i) for i in range(5000)}
    mapped2 = {d2._remap_fraction_idx(i) for i in range(5000)}
    assert mapped1 == mapped2
    assert len(mapped1) == 100
    assert all(0 <= m < 1000 for m in mapped1)


def test_data_fraction_seed_changes_subset():
    a = _FractionOnly(1000, 0.1, seed=1)
    b = _FractionOnly(1000, 0.1, seed=2)
    sa = {a._remap_fraction_idx(i) for i in range(5000)}
    sb = {b._remap_fraction_idx(i) for i in range(5000)}
    assert sa != sb


def test_data_fraction_uniform_coverage():
    """Every subset element is hit with roughly equal frequency."""
    d = _FractionOnly(1000, 0.05)
    counts: dict[int, int] = {}
    for i in range(1000):
        m = d._remap_fraction_idx(i)
        counts[m] = counts.get(m, 0) + 1
    assert len(counts) == 50
    assert max(counts.values()) - min(counts.values()) <= 1


def test_data_fraction_tiny_fraction_keeps_at_least_one():
    d = _FractionOnly(1000, 0.0001)
    assert d._remap_fraction_idx(123) == d._remap_fraction_idx(123)
    assert len({d._remap_fraction_idx(i) for i in range(100)}) == 1


# ── The finetune's checkpoint has to survive the trip to the scorer ────────


def _save_finetune_ckpt(d, head):
    """The on-disk layout load_model's supervised branch expects."""
    import json

    from market_jepa.modeling.backbones import create_backbone

    bb = create_backbone(
        backbone_type="transformer", n_features=9, d_embedding=32,
        config=TransformerConfig(**TINY), pool="cls")
    torch.save(bb.state_dict(), d / "backbone.pt")
    torch.save(head.state_dict(), d / "head.pt")
    (d / "train_meta.json").write_text(json.dumps({
        "task": "return_900", "run_name": "ft", "xs_target": "uniform",
        "config": {
            "mode": {"_target_": "market_jepa.modeling.modes.supervised."
                                 "SupervisedModel", "task": "return_900",
                     "loss_fn": "pairwise"},
            "dataset": {"norm_stats_channels": False},
            "backbone": {"_target_": "market_jepa.modeling.backbones."
                                     "transformer.TransformerBackbone",
                         "pool": "cls", "d_embedding": 32,
                         "config": dict(TINY)},
        }}))


def test_finetune_checkpoint_reloads_with_its_skip_head(tmp_path):
    """A SkipRegressionHead must come back AS ONE, not as a plain MLP.

    load_model rebuilds the head from the config, which says only
    "supervised" -- so before this it always built the plain MLP and the skip
    keys failed the strict load. That failure lands at SCORING time, after the
    whole training run, which is the expensive place to find it.
    """
    from market_jepa.eval.checkpoints import load_model
    head = SkipRegressionHead(32)
    rng = np.random.default_rng(3)
    head.init_from_ridge(torch.from_numpy(rng.normal(size=32).astype(np.float32)), 0.25)
    # Make the MLP branch nonzero so the reload cannot pass by accident.
    with torch.no_grad():
        for p in head.mlp.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    d = tmp_path / "ckpt"
    d.mkdir()
    _save_finetune_ckpt(d, head)

    cfg = json.loads((d / "train_meta.json").read_text())["config"]
    model = load_model(str(d), cfg, torch.device("cpu"))
    assert isinstance(model.head, SkipRegressionHead)
    assert not getattr(model, "_head_is_random", False)

    x = torch.randn(16, 32)
    head.eval()
    with torch.no_grad():
        assert torch.allclose(model.head(x), head(x), atol=1e-6)


# ── The readout the probe was fit at ──────────────────────────────────────


def test_mean_pooled_init_is_refused(lejepa_ckpt, tmp_path):
    """THE REGRESSION THAT COST A WAVE.

    Prediction is scored at the last token. A probe fit on MEAN-pooled
    embeddings describes a different feature space, but it is the same SHAPE,
    so it loads, trains, and produces a plausible curve for the wrong reason.
    The first wave of head inits was mean-pooled and had to be thrown away.
    """
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path, readout="mean")
    with pytest.raises(ValueError, match="readout mean"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900",
            init_backbone_from=ckpt, init_head_from=str(d_dir),
        )


def test_unstamped_init_is_refused(lejepa_ckpt, tmp_path):
    """Unstamped files predate the fix; for an SSL arm that means mean."""
    _, ckpt = lejepa_ckpt
    d_dir, _ = _write_ridge(tmp_path, readout=None)
    with pytest.raises(ValueError, match="unstamped/unknown"):
        SupervisedModel(
            _tiny_backbone(seed=1), task="return_900",
            init_backbone_from=ckpt, init_head_from=str(d_dir),
        )


def test_last_pooled_init_is_accepted(lejepa_ckpt, tmp_path):
    _, ckpt = lejepa_ckpt
    d_dir, w = _write_ridge(tmp_path, readout="last")
    m = SupervisedModel(
        _tiny_backbone(seed=1), task="return_900",
        init_backbone_from=ckpt, init_head_from=str(d_dir),
        head_init_scale="raw",
    )
    assert torch.allclose(m.head.skip.weight.flatten(), torch.from_numpy(w),
                          atol=1e-6)


# ── THE FITTER'S SHARD LISTING ─────────────────────────────────────────────
#
# pathlib.glob matches a leading dot where the shell's glob does not, so the
# writers' `.<name>.<pid>.tmp.npz` staging files were picked up as shards. Two
# abandoned ones (a job killed mid-write on 2026-09-17) took the head-init
# refit down with BadZipFile after 26 of 31 months.

import importlib.util  # noqa: E402

_FITTER = Path(__file__).resolve().parents[1] / "scripts/eval/fit_ridge_head_init.py"


def _load_fitter():
    spec = importlib.util.spec_from_file_location("_fit_ridge_head_init", _FITTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_groups_ignores_in_progress_temporaries(tmp_path, monkeypatch):
    """The actual crash path, which a month-anchored glob does NOT reproduce.

    `resolve_groups` lists with the OPEN-ENDED `*_a36_*.npz` to discover which
    months a run has. That pattern's leading `*` matches the dot, so the
    temporary became a month named ".2021-03", and group_readout then opened
    it by that name and died. The month-anchored pattern used everywhere else
    never matches the file, so a test written against `shards()` alone passes
    with and without the fix.
    """
    fitter = _load_fitter()
    run_dir = tmp_path / "pgpu006" / "pblast-part26" / "93hkqfoe"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "2021-03_a36_000.npz", X=np.zeros((2, 4), np.float32),
             z=np.zeros((2, 1), np.float32), target_names=np.array(["return_900"]),
             readout="last")
    # 400 MB of half-written zip in the real case; any non-zip will do here.
    (run_dir / ".2021-03_a36_000.854472.tmp.npz").write_bytes(b"not a zip")
    monkeypatch.setattr(fitter, "STAGING", tmp_path)

    got = fitter.resolve_groups("93hkqfoe", series="pair_warp_6mo")

    assert set(got) == {"2021-03"}, "a temporary must not become a month"


def test_shards_skips_in_progress_temporaries(tmp_path):
    fitter = _load_fitter()
    (tmp_path / "2021-03_a36_000.npz").write_bytes(b"")
    (tmp_path / ".2021-03_a36_000.854472.tmp.npz").write_bytes(b"not a zip")
    # Reached when the bogus ".2021-03" month key is passed on, as it was.
    assert fitter.shards(tmp_path, ".2021-03", 36) == []
    assert [p.name for p in fitter.shards(tmp_path, "2021-03", 36)] == [
        "2021-03_a36_000.npz"]


def test_shards_are_sorted(tmp_path):
    fitter = _load_fitter()
    for i in (2, 0, 1):
        (tmp_path / f"2021-03_a36_{i:03d}.npz").write_bytes(b"")
    assert [p.name for p in fitter.shards(tmp_path, "2021-03", 36)] == [
        "2021-03_a36_000.npz", "2021-03_a36_001.npz", "2021-03_a36_002.npz"]


# ── THE HEAD-INIT STAGING DIGEST ───────────────────────────────────────────
#
# ssl_base_lib.sh validates a reused /hpc_temp head init against this digest.
# Two implementations compute it -- Python here, `cat ... | sha256sum` there --
# so the thing worth testing is that they agree, byte order included.

_MANIFEST_GEN = (Path(__file__).resolve().parents[1]
                 / "scripts/sweeps/ssl_finetune/make_ssl_base_manifest.py")


def _load_manifest_gen():
    spec = importlib.util.spec_from_file_location(
        "_make_ssl_base_manifest", _MANIFEST_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_head_init(d: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for t in ("return_900", "spread_change_900", "volatility_change_900"):
        np.savez(d / f"{t}.npz", weight=rng.normal(size=8).astype(np.float32),
                 bias=np.float32(0.0), readout="last")


def test_head_digest_matches_the_shell_sha256sum(tmp_path):
    """The shell side concatenates the .npz files in lexicographic order."""
    import subprocess
    gen = _load_manifest_gen()
    _fake_head_init(tmp_path, seed=0)
    shell = subprocess.run(
        f'cat "{tmp_path}"/return_900.npz "{tmp_path}"/spread_change_900.npz '
        f'"{tmp_path}"/volatility_change_900.npz | sha256sum | cut -c1-16',
        shell=True, capture_output=True, text=True, check=True).stdout.strip()
    assert gen.head_digest(tmp_path) == shell


def test_head_digest_changes_when_the_weights_do(tmp_path):
    """The whole point: a refit IN PLACE must not look like the old content."""
    gen = _load_manifest_gen()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _fake_head_init(a, seed=0)
    _fake_head_init(b, seed=1)
    assert gen.head_digest(a) != gen.head_digest(b)


def test_head_digest_covers_every_file_the_head_loads(tmp_path):
    """A digest over only some tasks would miss a refit of the others."""
    gen = _load_manifest_gen()
    base = tmp_path / "base"
    base.mkdir()
    _fake_head_init(base, seed=0)
    before = gen.head_digest(base)
    for t in ("return_900", "spread_change_900", "volatility_change_900"):
        d = tmp_path / f"only_{t}"
        d.mkdir()
        _fake_head_init(d, seed=0)
        np.savez(d / f"{t}.npz", weight=np.arange(8, dtype=np.float32),
                 bias=np.float32(1.0), readout="last")
        assert gen.head_digest(d) != before, f"{t} is not covered by the digest"


_COLLECTOR = (Path(__file__).resolve().parents[1]
              / "scripts/eval/collect_ssl_finetune_breadth.py")
_BREADTH_SH = (Path(__file__).resolve().parents[1]
               / "scripts/sweeps/ssl_finetune/ssl_finetune_breadth.sh")


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "_collect_ssl_finetune_breadth", _COLLECTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collector_ladder_matches_the_sweep():
    """The two ends of the budget axis, which drifted apart once already.

    The sweep picks the checkpoint steps from SWEEP_TARGET_ROWS; the collector
    snaps each checkpoint's obs_seen back onto LADDER to decide which rung a
    point belongs to. When the sweep grew from six rungs to nine the collector
    kept the old list, so three real rungs had no nominal to snap to and were
    dropped, and a retired rung (524288) sat in the collector matching nothing.
    Neither side errors -- the figure just silently loses points.
    """
    import re as _re

    m = _re.search(r'^SWEEP_TARGET_ROWS="\$\{SWEEP_TARGET_ROWS:-([^}]*)\}"',
                   _BREADTH_SH.read_text(), _re.M)
    assert m, "SWEEP_TARGET_ROWS default not found in ssl_finetune_breadth.sh"
    swept = tuple(int(t) for t in m.group(1).split())

    assert _load_collector().LADDER == swept


def test_collector_ladder_rungs_cannot_be_confused():
    """Every rung is further from its neighbours than the snap tolerance.

    `rung()` takes the NEAREST nominal and accepts it within RUNG_TOL, so two
    rungs closer together than 2*RUNG_TOL would let a checkpoint land on the
    wrong budget rather than be dropped -- a wrong point, not a missing one.
    """
    mod = _load_collector()
    ladder = mod.LADDER
    assert sorted(ladder) == list(ladder), "LADDER must be ascending"
    for lo, hi in zip(ladder, ladder[1:]):
        assert hi - lo > mod.RUNG_TOL * (lo + hi), f"{lo} and {hi} can be confused"


def _fake_wave(root, commit, month_span, run, task, blr, steps=100):
    """A checkpoint tree shaped like the collector's input, minimal but real."""
    proj = root / f"ssl-ft-breadth-pair-warp-{commit}-{month_span}"
    d = proj / run
    d.mkdir(parents=True)
    (d / "train_meta.json").write_text(json.dumps({
        "run_name": f"ft_{task}_wsd_blr{blr}", "max_train_steps": steps,
        "completed_steps": steps, "obs_seen": steps * 256,
        "config": {"mode": {"task": task}, "optimizer": {"blr": blr},
                   "dataset": {"backend": "mds"}},
    }))
    (d / "xs_ic.json").write_text(json.dumps({
        f"xs_ic/head:{task}": 0.05, f"xs_ic_se/head:{task}": 0.01,
        f"xs_ic_cells/head:{task}": 100, f"xs_ic_rows/head:{task}": 1000,
    }))


def test_collector_names_the_wave_it_collected(tmp_path, capsys):
    """The label must be the SELECTED commit, not the first one present.

    It used to print commits[0] -- the first of every commit in the tree --
    so collecting a new pilot while the previous wave was still on disk
    announced the OLD hash over the NEW numbers. The rows were correct and
    the banner was not, which is the more dangerous way round: it is the line
    you quote when you say which recipe a result came from.
    """
    for commit in ("047986", "653736"):
        _fake_wave(tmp_path, commit, "2012-10-01-2013-03-31",
                   f"run{commit}", "return_900", 1e-5)

    mod = _load_collector()
    out = tmp_path / "out.json"
    mod.main_argv = None
    import sys
    argv = ["collect", "--ckpt-root", str(tmp_path), "--commit", "653736",
            "--out", str(out)]
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old

    printed = capsys.readouterr().out
    assert "653736" in printed, printed
    assert "047986" not in printed, printed
