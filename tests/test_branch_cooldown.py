"""Branch cooldowns (checkpoint.anneal_steps): one stable run, a finished
model at every FLOPs target.

The plan resolver, the LR line a cooldown walks, and the snapshot that puts
the stable run back are tested here on toy models; pretrain.py wires them
at each branch point (scripts/sweeps/supervised_scaling.sh is the caller).
"""
from __future__ import annotations

import pytest
import torch

from market_jepa.training.utils import (
    build_lr_scheduler, cooldown_factor, resolve_anneal_plan,
    restore_snapshot, take_snapshot)


# ── the plan ───────────────────────────────────────────────────────────────

def test_plan_names_each_checkpoint_by_its_total_and_branches_early():
    plan = resolve_anneal_plan([223, 670, 2231], 0.1, 2231, 128, "stable")
    assert plan == {201: (223, 22), 603: (670, 67), 2008: (2231, 223)}


def test_plan_is_empty_when_nothing_is_asked():
    assert resolve_anneal_plan([], 0.1, 100, 10, "cosine") == {}
    assert resolve_anneal_plan(None, 0.1, 100, 10, "wsd") == {}


@pytest.mark.parametrize("bad, msg", [
    (dict(schedule="wsd"), "stable"),
    (dict(anneal_steps=[130, 130, 500]), "duplicates"),
    (dict(anneal_steps=[130, 400]), "max_train_steps"),
    (dict(anneal_steps=[130, 500], warmup=200), "warmup"),
    (dict(anneal_frac=1.5), "anneal_frac"),
])
def test_plan_refuses_what_is_not_a_finished_model(bad, msg):
    kw = dict(anneal_steps=[130, 500], anneal_frac=0.1, max_train_steps=500,
              warmup=10, schedule="stable")
    kw.update(bad)
    with pytest.raises(ValueError, match=msg):
        resolve_anneal_plan(kw["anneal_steps"], kw["anneal_frac"],
                            kw["max_train_steps"], kw["warmup"], kw["schedule"])


def test_two_totals_on_one_branch_point_are_refused():
    # 100 -> branch 90 (n=10); 101 -> branch 91; 109 -> n=11, branch 98;
    # 111 -> n=11, branch 100. Pick a pair that collides: 105 (n=10 -> 95)
    # and 106 (n=11 -> 95).
    with pytest.raises(ValueError, match="both branch"):
        resolve_anneal_plan([105, 106, 500], 0.1, 500, 10, "stable")


# ── the line ───────────────────────────────────────────────────────────────

def test_cooldown_walks_linearly_to_min_and_never_starts_at_peak():
    n, ratio = 10, 0.1
    fs = [cooldown_factor(k, n, ratio) for k in range(n)]
    assert fs[-1] == pytest.approx(ratio)
    assert fs[0] < 1.0
    steps = [a - b for a, b in zip(fs, fs[1:])]
    assert all(s == pytest.approx(steps[0]) for s in steps)
    with pytest.raises(ValueError):
        cooldown_factor(n, n, ratio)


def test_stable_schedule_holds_the_peak_to_the_end():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched, n = build_lr_scheduler(opt, 50, 0.1, 0.1, schedule="stable", warmup_steps=4)
    assert n == 4
    lrs = []
    for _ in range(50):
        opt.step(); sched.step(); lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[1] < lrs[2] < 1.0
    assert all(lr == pytest.approx(1.0) for lr in lrs[3:])


# ── the snapshot ───────────────────────────────────────────────────────────

def _toy():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    return m, opt


def _step(m, opt, lr_factor=1.0):
    for g in opt.param_groups:
        g["lr"] = 1e-2 * lr_factor
    x = torch.randn(16, 4)
    loss = (m(x) - x.sum(1, keepdim=True)).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def _flat(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def test_restore_puts_weights_and_moments_back_exactly():
    m, opt = _toy()
    for _ in range(5):
        _step(m, opt)
    before = _flat(m).clone()
    moments = {k: {kk: vv.clone() for kk, vv in v.items()}
               for k, v in opt.state_dict()["state"].items()}
    snap = take_snapshot(m, opt)
    for k in range(10):                        # a cooldown moves the model
        _step(m, opt, cooldown_factor(k, 10, 0.1))
    assert not torch.allclose(_flat(m), before)
    restore_snapshot(m, opt, snap)
    assert torch.equal(_flat(m), before)
    for k, v in opt.state_dict()["state"].items():
        for kk, vv in v.items():
            assert torch.equal(vv, moments[k][kk]), (k, kk)


def test_snapshot_is_a_copy_not_a_view():
    m, opt = _toy()
    _step(m, opt)
    snap = take_snapshot(m, opt)
    _step(m, opt)
    ref = snap["model"]["0.weight"]
    assert not torch.equal(ref, m[0].weight.detach())


def test_the_stable_run_continues_as_if_the_branch_never_happened():
    """Same data after the branch -> same weights as a run that never
    branched; the branch changes nothing but the batches it consumed."""
    a, opt_a = _toy()
    b, opt_b = _toy()
    b.load_state_dict(a.state_dict())
    torch.manual_seed(1); _step(a, opt_a)
    torch.manual_seed(1); _step(b, opt_b)
    snap = take_snapshot(b, opt_b)
    for k in range(3):
        _step(b, opt_b, cooldown_factor(k, 3, 0.1))
    restore_snapshot(b, opt_b, snap)
    torch.manual_seed(2); _step(a, opt_a)
    torch.manual_seed(2); _step(b, opt_b)
    assert torch.equal(_flat(a), _flat(b))
