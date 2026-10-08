"""The WSD schedule, and the silent no-op it is easy to write instead.

WSD exists so that ONE run yields a label-budget ladder: the stable phase
holds the LR constant, so checkpoints taken inside it were all trained under
the same optimizer state and are comparable to each other. Under cosine they
are not, and a budget curve needs one run per budget -- the difference between
93 runs and 558.

The trap these pin: LinearLR.__init__ applies start_factor to the optimizer
IMMEDIATELY, so a peak-LR reading taken after the warmup scheduler is built
returns peak * warmup_start_frac. Computing the decay's end_factor from that
makes it 1.0 -- a decay that runs for the right number of steps and changes
nothing. Nothing raises; the LR is simply flat to the end of training.
"""
import pytest
import torch
from torch.optim import SGD

from market_jepa.training.utils import build_lr_scheduler

PEAK, MIN_LR, TOTAL = 1e-5, 1e-7, 100


def trace(schedule, total=TOTAL, warmup_frac=0.05, decay_frac=0.2, min_lr=MIN_LR):
    model = torch.nn.Linear(2, 1)
    opt = SGD(model.parameters(), lr=PEAK)
    sched, n_warm = build_lr_scheduler(
        opt, total, warmup_frac, min_lr, schedule=schedule, decay_frac=decay_frac)
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs, n_warm


def test_wsd_stable_phase_is_actually_constant():
    """The whole point: every checkpoint in the stable phase sees one LR."""
    lrs, n_warm = trace("wsd")
    stable = lrs[n_warm:80]
    assert len(stable) > 50
    assert max(stable) == pytest.approx(PEAK, rel=1e-9)
    assert min(stable) == pytest.approx(PEAK, rel=1e-9)


def test_wsd_actually_decays():
    """THE REGRESSION. A decay whose end_factor is 1.0 is flat and silent."""
    lrs, _ = trace("wsd")
    assert lrs[-1] < PEAK / 10, f"WSD never decayed: final LR {lrs[-1]:.2e}"
    # Monotonically down through the decay, not a single step change.
    tail = lrs[81:]
    assert all(b <= a for a, b in zip(tail, tail[1:]))
    assert len(set(tail)) > 5


def test_wsd_reaches_about_min_lr():
    lrs, _ = trace("wsd")
    # The last recorded LR is one step short of the end, so allow a step's gap.
    assert lrs[-1] < 10 * MIN_LR


def test_cosine_is_unchanged_and_never_constant():
    """The reported recipe must not move: cosine differs at every step."""
    lrs, n_warm = trace("cosine")
    body = lrs[n_warm:]
    assert all(b < a for a, b in zip(body, body[1:]))
    assert lrs[-1] < 10 * MIN_LR


def test_warmup_is_the_same_in_both():
    c, nc = trace("cosine")
    w, nw = trace("wsd")
    assert nc == nw
    assert c[:nc] == pytest.approx(w[:nw])
    assert c[nc] == pytest.approx(PEAK) and w[nw] == pytest.approx(PEAK)


def test_short_run_still_gets_a_stable_step():
    """Warmup + decay can exceed a very short run; stable must not vanish."""
    lrs, n_warm = trace("wsd", total=6)
    assert len(lrs) == 6
    assert max(lrs) == pytest.approx(PEAK)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_bad_decay_frac_raises(bad):
    with pytest.raises(ValueError, match="decay_frac"):
        trace("wsd", decay_frac=bad)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="lr_schedule"):
        trace("triangular")
