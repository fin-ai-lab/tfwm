"""The ViT-scale sweep: its table, its ladder, its FLOP axis and its collector.

Each test pins a way this experiment could silently draw the wrong figure:
a "small" rung that is not the reported model, a ladder rung past the
shortest month's stable phase (a curve that quietly loses its top), a FLOP
count that drifts from the architecture, a knob that never reaches the node.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "scripts/sweeps/supervised_scaling.sh"
LR_SWEEP = ROOT / "scripts/sweeps/supervised_scaling_lr.sh"
COLLECTOR = ROOT / "scripts/eval/collect_supervised_scaling.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sweep_default(path: Path, name: str) -> str:
    m = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', path.read_text(), re.M)
    assert m, f"{name} default not found in {path.name}"
    return m.group(1)


# ── the table ──────────────────────────────────────────────────────────────

def test_small_is_the_reported_model():
    """The middle rung must be TransformerInnerConfig's own defaults, so the
    scaling curve passes through the paper's arm rather than beside it."""
    from market_jepa.schemas import VIT_SCALES, TransformerInnerConfig
    d = TransformerInnerConfig()
    assert VIT_SCALES["small"] == {
        "hidden_size": d.hidden_size,
        "num_attention_heads": d.num_attention_heads,
        "intermediate_size": d.intermediate_size,
    }


def test_every_scale_is_a_plain_vit_shape():
    """head_dim 64, MLP 4x: width, heads and MLP are the only knobs."""
    from market_jepa.schemas import VIT_SCALES
    for name, d in VIT_SCALES.items():
        assert d["hidden_size"] // d["num_attention_heads"] == 64, name
        assert d["intermediate_size"] == 4 * d["hidden_size"], name
        assert set(d) == {"hidden_size", "num_attention_heads", "intermediate_size"}, name


def test_small_never_needs_a_measured_lr():
    """Small's LR is the recipe's; a table entry for it would be a second copy."""
    from market_jepa.schemas import VIT_SCALE_BLR, VIT_SCALES
    assert "small" not in VIT_SCALE_BLR
    assert set(VIT_SCALE_BLR) <= set(VIT_SCALES)


# ── the ladder ─────────────────────────────────────────────────────────────

def _anneal_steps(scale: str) -> list[int]:
    r = _bash(f'source {SWEEP}; scaling_anneal_steps {scale}', {})
    assert r.returncode == 0, r.stderr
    return [int(t) for t in r.stdout.split()]


def test_targets_are_half_decades_of_flops_as_steps_per_scale():
    """A target is a step count per scale, the same compute in every month;
    the run is the target's length, not the recipe's, so nothing ties the
    top to the shortest month any more."""
    from market_jepa.eval.flops import shape_for_scale, training_flops_per_view
    views = int(_sweep_default(SWEEP, "SCALING_VIEWS_PER_STEP"))
    grid = [1e16, 3.33e16, 1e17, 3.33e17, 1e18, 3.33e18]
    tops = {"tiny": 3.33e17, "small": 1e18, "base": 3.33e18}
    for scale, top in tops.items():
        steps = _anneal_steps(scale)
        assert steps == sorted(set(steps)), scale
        per_step = training_flops_per_view(shape_for_scale(scale)) * views
        assert steps[-1] * per_step == pytest.approx(top, rel=0.01), scale
        # Each step count lands on a grid value, the user's 1, 3.33 decades.
        for t in steps:
            assert any(t * per_step == pytest.approx(g, rel=0.01) for g in grid), (scale, t)


def test_views_per_step_is_the_recipe_batch():
    import dataclasses
    from market_jepa.schemas import SupervisedModeConfig
    c = dataclasses.asdict(SupervisedModeConfig())
    eff = c["training_overrides"]["effective_batch_size"]

    def find(d, key):
        if isinstance(d, dict):
            if key in d and d[key] is not None:
                return d[key]
            for v in d.values():
                got = find(v, key)
                if got is not None:
                    return got
        if isinstance(d, (list, tuple)):
            for v in d:
                got = find(v, key)
                if got is not None:
                    return got
        return None
    k = find(c["dataset_overrides"], "n_stocks")
    assert eff * k == int(_sweep_default(SWEEP, "SCALING_VIEWS_PER_STEP"))


def test_per_view_cost_matches_the_waves_checkpoints():
    """shape_for_scale builds the recipe's model from the width table; hold
    it to the per-view forward the collector read off real checkpoints on
    2026-09-19 (tiny 3.34, small 12.15, base 46.12 GF)."""
    from market_jepa.eval.flops import forward_flops, shape_for_scale
    for scale, gf in (("tiny", 3.34), ("small", 12.15), ("base", 46.12)):
        assert forward_flops(shape_for_scale(scale))["total"] / 1e9 == pytest.approx(gf, rel=0.01)


def test_warmup_bound_targets_are_dropped_not_drawn():
    """Base at 1e16 is 18 steps and at 3.33e16 is 59: both branch inside the
    128-step warmup and are left out; the first base dot is 1e17."""
    from market_jepa.eval.flops import steps_for_flops
    all_six = [1e16, 3.33e16, 1e17, 3.33e17, 1e18, 3.33e18]
    base = steps_for_flops("base", all_six, views_per_step=4096, warmup_steps=128, anneal_frac=0.1)
    assert len(base) == 4 and base[0] == 176
    tiny = steps_for_flops("tiny", all_six, views_per_step=4096, warmup_steps=128, anneal_frac=0.1)
    assert len(tiny) == 6 and tiny[0] == 243
    assert _anneal_steps("base")[0] == 176


def test_warmup_is_pinned_in_steps_and_before_every_first_branch():
    """A fraction would put a target at a different LR in every month."""
    warm = int(_sweep_default(SWEEP, "SCALING_WARMUP_STEPS"))
    body = "\n".join(l for l in SWEEP.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    assert "optimizer.warmup_steps=${SCALING_WARMUP_STEPS}" in body
    frac = float(_sweep_default(SWEEP, "SCALING_ANNEAL_FRAC"))
    for scale in ("tiny", "small", "base"):
        first = _anneal_steps(scale)[0]
        assert first - round(frac * first) >= warm, scale


def test_warmup_steps_pin_beats_warmup_frac():
    import torch
    from market_jepa.training.utils import build_lr_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched, n = build_lr_scheduler(opt, 100, 0.5, 0.1, schedule="wsd",
                                  decay_frac=0.1, warmup_steps=4)
    assert n == 4
    lrs = []
    for _ in range(6):
        opt.step(); sched.step(); lrs.append(opt.param_groups[0]["lr"])
    assert lrs[3] == pytest.approx(1.0) and lrs[4] == pytest.approx(1.0)
    assert lrs[0] < lrs[1] < lrs[2] < 1.0
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, 100, 0.5, 0.1, schedule="wsd", warmup_steps=100)


# ── the sweep files ────────────────────────────────────────────────────────

def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, **env})


def test_ladder_sweep_values_cover_every_scale_and_task():
    r = _bash(f'source {SWEEP}; printf "%s\\n" "${{SWEEP_VALUES[@]}}"', {})
    assert r.returncode == 0, r.stderr
    vals = r.stdout.split()
    from market_jepa.schemas import VIT_SCALES
    assert len(vals) == 3 * len(VIT_SCALES)
    assert "small:return_900" in vals and "base:spread_change_900" in vals


def test_ladder_sweep_args_carry_the_scale_and_the_ladder():
    r = _bash(f'source {SWEEP}; sweep_train_args base:return_900',
              {"DAYSTORE": "1", "POST_TRAIN_PROBE": "0", "TRAIN_END": "2013-04-30", "SCALING_BLR": "1e-4"})
    assert r.returncode == 0, r.stderr
    args = r.stdout.split()
    from market_jepa.schemas import VIT_SCALES
    b = VIT_SCALES["base"]
    assert f"mode.backbone.config.hidden_size={b['hidden_size']}" in args
    assert f"mode.backbone.d_embedding={b['hidden_size']}" in args
    assert f"mode.backbone.config.num_attention_heads={b['num_attention_heads']}" in args
    assert f"mode.backbone.config.intermediate_size={b['intermediate_size']}" in args
    assert "optimizer.lr_schedule=stable" in args
    assert "optimizer.anneal_frac=0.1" in args
    assert "optimizer.blr=1e-4" in args
    steps = _anneal_steps("base")
    assert f"checkpoint.anneal_steps=[{','.join(map(str, steps))}]" in args
    # The run is the last target's length, in steps, not the recipe's passes.
    assert f"training.max_train_steps={steps[-1]}" in args
    assert "training.num_epochs=null" in args
    assert "checkpoint.save_steps" not in r.stdout
    assert "decay_frac" not in r.stdout
    assert "wandb.project=supervised-scaling-base" in args


def test_ladder_sweep_refuses_an_unmeasured_lr_and_inherits_smalls():
    from market_jepa.schemas import VIT_SCALE_BLR
    env = {"DAYSTORE": "1", "POST_TRAIN_PROBE": "0", "TRAIN_END": "2013-04-30"}
    for scale, blr in VIT_SCALE_BLR.items():
        r = _bash(f'source {SWEEP}; sweep_train_args {scale}:return_900', env)
        if blr is None:
            assert r.returncode != 0, f"{scale} ran with no measured LR"
            assert "supervised_scaling_lr.sh" in r.stderr
        else:
            assert r.returncode == 0 and f"optimizer.blr={blr}" in r.stdout.split()
    r = _bash(f'source {SWEEP}; sweep_train_args small:return_900', env)
    assert r.returncode == 0, r.stderr
    assert not any(a.startswith("optimizer.blr=") for a in r.stdout.split()), \
        "small inherits the recipe's LR from the mode, not from the sweep"


def test_ladder_sweep_refuses_the_mds_backend():
    r = _bash(f'source {SWEEP}; sweep_train_args small:return_900',
              {"DAYSTORE": "0", "POST_TRAIN_PROBE": "0", "TRAIN_END": "2013-04-30"})
    assert r.returncode != 0 and "DAYSTORE=1" in r.stderr


def test_scale_sweeps_refuse_to_score_the_probe():
    """Twelve a36 embeddings a run for a readout the figure does not draw."""
    for sweep, val in ((SWEEP, "small:return_900"), (LR_SWEEP, "tiny:2e-4:return_900")):
        r = _bash(f'source {sweep}; sweep_train_args {val}',
                  {"DAYSTORE": "1", "TRAIN_END": "2013-04-30"})
        assert r.returncode != 0 and "POST_TRAIN_PROBE=0" in r.stderr, sweep.name


def test_lr_sweep_grids_each_scale_and_passes_the_rate_as_the_axis():
    r = _bash(f'source {LR_SWEEP}; printf "%s\\n" "${{SWEEP_VALUES[@]}}"', {})
    assert r.returncode == 0, r.stderr
    vals = r.stdout.split()
    scales = {v.split(":")[0] for v in vals}
    assert scales == {"tiny", "base"}, "small is not swept: its LR is the recipe's"
    for s in scales:
        lrs = {v.split(":")[1] for v in vals if v.startswith(s + ":")}
        assert "2e-4" in lrs, f"{s}'s grid must include the recipe's rate for a paired read"
        assert len(lrs) >= 4
    r = _bash(f'source {LR_SWEEP}; sweep_train_args tiny:4e-4:volatility_change_900',
              {"DAYSTORE": "1", "POST_TRAIN_PROBE": "0", "TRAIN_END": "2009-06-30"})
    assert r.returncode == 0, r.stderr
    args = r.stdout.split()
    assert "optimizer.blr=4e-4" in args and "mode.task=volatility_change_900" in args
    assert not any(a.startswith("checkpoint.save_steps") for a in args), \
        "the grid scores endpoints only"


# ── the FLOP axis ──────────────────────────────────────────────────────────

def test_flop_count_matches_torch_up_to_the_attention_products():
    """torch's FlopCounterMode reproduces every term of forward_flops except
    the QK^T / AV products, which it does not see through SDPA. Hold the
    count to the counter PLUS that term, on the tiny rung at the real view
    geometry, with the shape read the way the collector reads it."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from market_jepa.eval.flops import forward_flops, shape_from_state_dicts
    from market_jepa.eval.heads import RegressionHead
    from market_jepa.modeling.backbones.transformer import (
        TransformerBackbone, TransformerConfig)
    from market_jepa.schemas import VIT_SCALES

    t = VIT_SCALES["tiny"]
    cfg = TransformerConfig(hidden_size=t["hidden_size"],
                            num_attention_heads=t["num_attention_heads"],
                            intermediate_size=t["intermediate_size"],
                            pos_embed="sinusoidal")
    bb = TransformerBackbone(cfg, n_features=17, d_embedding=t["hidden_size"],
                             pool="last", n_info_channels=8).train()
    head = RegressionHead(t["hidden_size"]).train()
    shape = shape_from_state_dicts(bb.state_dict(), head.state_dict(), seq_len=2048)
    assert (shape.n_info_channels, shape.n_patch_features, shape.patch_size,
            shape.num_hidden_layers, shape.n_tokens) == (8, 9, 8, 12, 257)
    parts = forward_flops(shape)
    with FlopCounterMode(display=False) as fc:
        head(bb(torch.randn(1, 17, 2048)))
    measured = fc.get_total_flops()
    assert measured == pytest.approx(parts["total"] - parts["attention"], rel=2e-3)
    assert parts["attention"] > 0


def test_training_flops_scale_with_views_and_the_reported_widths_order():
    from market_jepa.eval.flops import TransformerShape, training_flops
    from market_jepa.schemas import VIT_SCALES

    def shape(name):
        d = VIT_SCALES[name]
        return TransformerShape(
            hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"],
            num_hidden_layers=12, patch_size=8, n_patch_features=9,
            n_info_channels=8, seq_len=2048, d_embedding=d["hidden_size"],
            head_hidden=2 * d["hidden_size"])
    tiny, small, base = (training_flops(shape(n), 4096) for n in ("tiny", "small", "base"))
    assert tiny < small < base
    assert 3.3 < small / tiny < 4.0 and 3.5 < base / small < 4.0
    assert training_flops(shape("small"), 10) == 10 * training_flops(shape("small"), 1)


# ── the collector ──────────────────────────────────────────────────────────

def _fake_run(root: Path, project: str, run: str, task: str, steps: list[int],
              max_steps: int, ic: float, branched: bool = False):
    import torch
    from market_jepa.eval.heads import RegressionHead
    from market_jepa.modeling.backbones.transformer import (
        TransformerBackbone, TransformerConfig)
    cfg = TransformerConfig(hidden_size=64, num_attention_heads=1,
                            intermediate_size=256, num_hidden_layers=2,
                            pos_embed="sinusoidal")
    bb = TransformerBackbone(cfg, n_features=17, d_embedding=64, pool="last",
                             n_info_channels=8)
    d = root / project / run
    d.mkdir(parents=True)
    torch.save(bb.state_dict(), d / "backbone.pt")
    torch.save(RegressionHead(64).state_dict(), d / "head.pt")
    config = {"dataset": {"backend": "days",
                          "augmentations": {"0": {"name": "cross_stock", "n_stocks": 16,
                                                  "global_seq_len": 2048}}},
              "optimizer": {"blr": None, "warmup_steps": 128},
              "checkpoint": {"anneal_steps": list(steps) if branched else []},
              "mode": {"training_overrides": {"blr": 2e-4},
                       "backbone": {"config": {"pos_embed": "sinusoidal"}}}}

    def meta(step):
        return {"task": task, "run_name": f"x_{task}", "config": config,
                "max_train_steps": max_steps, "completed_steps": step,
                "obs_seen": step * 256}

    def xs(step):
        return {f"xs_ic/{task}": ic + step * 1e-5, f"xs_ic_se/{task}": 0.001,
                f"xs_ic/head:{task}": ic + 0.01, f"xs_ic_se/head:{task}": 0.002}
    (d / "train_meta.json").write_text(json.dumps(meta(max_steps)))
    (d / "xs_ic.json").write_text(json.dumps(xs(max_steps)))
    for s in steps:
        sd = d / str(s)
        sd.mkdir()
        (sd / "train_meta.json").write_text(json.dumps(meta(s)))
        (sd / "xs_ic.json").write_text(json.dumps(xs(s)))


def test_collector_bills_every_checkpoint_from_the_roots_weights(tmp_path):
    mod = _load(COLLECTOR)
    _fake_run(tmp_path, "supervised-scaling-tiny-abc123-2013-01-01-2013-06-30",
              "r1", "return_900", [64, 128], 2400, 0.02)
    rows = mod.collect(tmp_path)
    assert len(rows) == 3
    by_step = {r["step"]: r for r in rows}
    assert set(by_step) == {64, 128, 2400}
    root = by_step[2400]
    assert root["annealed"] and not by_step[64]["annealed"]
    assert root["endpoint"] and not by_step[64]["endpoint"]
    assert root["eval_month"] == "2013-07" and root["scale"] == "tiny"
    assert root["views"] == 2400 * 256 * 16, "obs_seen counts cells; x16 rows"
    assert by_step[128]["flops"] == 2 * by_step[64]["flops"]
    assert root["flops_per_view"] == by_step[64]["flops_per_view"]
    assert root["blr"] == 2e-4 and root["warmup_steps"] == 128
    assert root["ic_head"] == pytest.approx(0.03)
    assert root["hidden_size"] == 64
    # The fixed sinusoidal table (2048 x 64) is not a parameter and is not
    # counted; everything else is.
    import torch
    sd = torch.load(tmp_path / "supervised-scaling-tiny-abc123-2013-01-01-2013-06-30"
                    / "r1" / "backbone.pt", weights_only=True)
    trainable = sum(v.numel() for k, v in sd.items() if k != "position_embeddings")
    assert root["params_backbone"] == trainable
    assert sd["position_embeddings"].numel() == 2048 * 64


def test_collector_keeps_branch_checkpoints_and_drops_their_root(tmp_path):
    """Under checkpoint.anneal_steps every step dir is a finished model and
    the root is the last one again: no endpoint, no duplicate."""
    root = tmp_path / "ckpt"
    _fake_run(root, "supervised-scaling-small-abc123-2013-01-01-2013-06-30",
              "r1", "return_900", [223, 670, 2231], 2231, 0.02, branched=True)
    mod = _load(COLLECTOR)
    rows = mod.collect(root)
    assert sorted(r["step"] for r in rows) == [223, 670, 2231]
    assert all(r["annealed"] and not r["endpoint"] for r in rows)


def test_collector_refuses_to_pool_two_waves(tmp_path):
    mod = _load(COLLECTOR)
    _fake_run(tmp_path, "supervised-scaling-tiny-abc123-2013-01-01-2013-06-30",
              "r1", "return_900", [64], 2400, 0.02)
    _fake_run(tmp_path, "supervised-scaling-tiny-def456-2013-01-01-2013-06-30",
              "r2", "return_900", [64], 2400, 0.02)
    with pytest.raises(SystemExit, match="must not be pooled"):
        mod.collect(tmp_path)
    assert len(mod.collect(tmp_path, commit="def456")) == 2


def test_collector_ignores_the_lr_grid(tmp_path):
    mod = _load(COLLECTOR)
    _fake_run(tmp_path, "supervised-scaling-lr-tiny-abc123-2013-01-01-2013-06-30",
              "r1", "return_900", [], 2400, 0.02)
    assert mod.collect(tmp_path) == []


# ── the figure ─────────────────────────────────────────────────────────────

def test_figure_draws_shared_rungs_only_and_the_endpoint(tmp_path):
    mod = _load(ROOT / "plots/scaling/supervised_scaling.py")
    rows = []
    for ym, top in (("2013-07", 2400), ("2014-02", 3000)):
        for step, ann in ((64, False), (128, False), (top, True)):
            rows.append({"eval_month": ym, "task": "return_900", "scale": "small",
                         "step": step, "annealed": ann, "flops": step * 1e12,
                         "ic_probe": 0.02, "ic_head": 0.03, "params_backbone": 21e6})
    # a rung one month never reached must not be drawn
    rows.append({"eval_month": "2013-07", "task": "return_900", "scale": "small",
                 "step": 256, "annealed": False, "flops": 256e12,
                 "ic_probe": 0.05, "ic_head": 0.05, "params_backbone": 21e6})
    xs, mu, se, months, end = mod.series(rows, "return_900", "small", "head")
    assert list(xs) == [64e12, 128e12] and months == ["2013-07", "2014-02"]
    assert end == pytest.approx((2700e12, 0.03))
    assert mod.series(rows, "return_900", "base", "head") is None
    out = tmp_path / "f.png"
    assert mod.draw(rows, "head", out)
    assert out.exists() and out.with_suffix(".pdf").exists()


# ── the star: the reported model's own compute ─────────────────────────────

def test_star_sits_on_the_curve_at_the_reported_models_compute(tmp_path):
    """The star is a place on the x axis, not a score: its height is the
    curve's at that compute, and it is drawn only for the reported scale."""
    mod = _load(ROOT / "plots/scaling/supervised_scaling.py")
    assert mod.on_curve([1e17, 1e18], [0.02, 0.04], 1e17) == pytest.approx(0.02)
    # halfway in LOG compute, not in linear compute
    assert mod.on_curve([1e17, 1e18], [0.02, 0.04], 10**17.5) == pytest.approx(0.03)
    # past the last rung the sweep has nothing to say
    assert mod.on_curve([1e17, 1e18], [0.02, 0.04], 5e18) is None
    assert mod.on_curve([1e17], [0.02], 1e17) is None


def test_star_is_drawn_only_where_the_curve_reaches(tmp_path, monkeypatch):
    mod = _load(ROOT / "plots/scaling/supervised_scaling.py")
    rows = []
    for ym in ("2013-07", "2014-02"):
        for scale in ("small", "base"):
            for step in (223, 6700):
                rows.append({"eval_month": ym, "task": "return_900",
                             "scale": scale, "step": step, "annealed": True,
                             "endpoint": False, "flops": step * 1e14,
                             "ic_probe": None, "ic_head": 0.03,
                             "params_backbone": 21e6})
    seen: list[list[str]] = []
    monkeypatch.setattr(mod, "add_bottom_legend",
                        lambda fig, h, labels, **kw: seen.append(list(labels)))

    star = {"scale": "small", "label": "Reported model", "flops": 1e17}
    assert mod.draw(rows, "head", tmp_path / "f.png", default=star)
    assert "Reported model" in seen[-1]

    # the same model, if the sweep had not yet reached its compute
    assert mod.draw(rows, "head", tmp_path / "f.png",
                    default={**star, "flops": 1e20})
    assert "Reported model" not in seen[-1]

    # and it belongs to one curve: the scale it is
    assert mod.draw(rows, "head", tmp_path / "f.png",
                    default={**star, "scale": "tiny"})
    assert "Reported model" not in seen[-1]


def test_default_model_is_the_recipe_not_a_ladder_run():
    """The step count is read off any run of the recipe, so the filter is the
    recipe: a ladder run trains a target's length, not twelve passes."""
    mod = _load(ROOT / "scripts/eval/default_model_flops.py")
    cfg = {"dataset": {"backend": "days", "train_span_months": 6},
           "training": {"num_epochs": 12, "max_train_steps": None},
           "checkpoint": {"anneal_steps": []},
           "backbone": {"d_embedding": 384},
           "mode": {"training_overrides": {"effective_batch_size": 256}}}
    assert mod.is_the_recipe(cfg, 384)
    assert not mod.is_the_recipe(cfg, 768)          # another scale's curve
    assert not mod.is_the_recipe({**cfg, "checkpoint": {"anneal_steps": [223]}}, 384)
    assert not mod.is_the_recipe(
        {**cfg, "training": {"num_epochs": 12, "max_train_steps": 6700}}, 384)
    assert not mod.is_the_recipe(
        {**cfg, "dataset": {"backend": "mds", "train_span_months": 6}}, 384)
    assert not mod.is_the_recipe(
        {**cfg, "dataset": {"backend": "days", "train_span_months": 1}}, 384)


def test_reported_ic_comes_from_one_wave_of_specialists():
    """Two waves of specialists are two recipes' scores; the widest wins and
    the other is not averaged into it."""
    mod = _load(ROOT / "scripts/eval/default_model_flops.py")
    ic = {("return_900", "581eb2"): {"2013-07": 0.03, "2013-08": 0.05},
          ("return_900", "3e5087"): {"2013-07": 0.90}}
    got = mod.best_ic(ic)["return_900"]
    assert got["commit"] == "581eb2" and got["n_months"] == 2
    assert got["mean"] == pytest.approx(0.04)
