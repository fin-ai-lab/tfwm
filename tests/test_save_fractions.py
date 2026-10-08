"""checkpoint.save_fractions must mean the same point in every arm.

The SSL finetune breadth sweep varies the LABEL BUDGET, so every arm has a
different max_train_steps -- steps_per_epoch is n_train_avail // (batch x
accum), which moves with train_data_fraction AND with the month's
cross-section. A literal step list would therefore sit at a different point on
each arm's training curve, which is exactly what a progress plot must not do.

These pin the resolution: the rounding, the clamps, and the de-duplication
that a short arm needs.
"""
import pytest

from market_jepa.training.utils import resolve_save_steps


def resolve(fractions, max_train_steps, save_steps=()):
    """THE REAL FUNCTION, sorted for readable assertions.

    Deliberately not a local reimplementation: a test that re-derives the
    rounding it is checking passes whatever pretrain.py goes on to do.
    """
    return sorted(resolve_save_steps(save_steps, fractions, max_train_steps))


FRACS = [0.01, 0.10, 0.25, 0.50, 1.0]


def test_full_budget_arm_spreads_over_the_run():
    # ~3,484 steps: the full six-month pool at effective batch 256, 12 passes.
    assert resolve(FRACS, 3484) == [35, 348, 871, 1742, 3484]


def test_the_last_fraction_is_the_final_step():
    for n in (12, 36, 171, 1509, 3484):
        assert resolve(FRACS, n)[-1] == n


def test_short_arm_collapses_rather_than_saving_twice():
    """36 steps is the 32,768-row rung. Several fractions land on one step.

    Without the de-duplication this writes the same weights under two names
    and the progress plot gets two points that are the same model.
    """
    assert resolve(FRACS, 36) == [1, 4, 9, 18, 36]
    # 12 steps: 1% and 10% both round below 1 and are clamped onto step 1.
    assert resolve(FRACS, 12) == [1, 3, 6, 12]


def test_a_fraction_never_rounds_to_step_zero():
    """completed_steps starts at 1, so a step-0 entry would never fire."""
    assert 0 not in resolve([0.001], 100)
    assert resolve([0.001], 100) == [1]


def test_a_fraction_never_exceeds_the_run():
    assert resolve([1.0], 50) == [50]


def test_explicit_steps_and_fractions_merge():
    assert resolve([0.5], 100, save_steps=[10, 50]) == [10, 50]
    assert resolve([0.5], 100, save_steps=[10, 99]) == [10, 50, 99]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_out_of_range_fraction_raises(bad):
    with pytest.raises(ValueError, match="save_fractions"):
        resolve([bad], 100)


def test_empty_is_empty():
    """The default must add nothing: every other sweep inherits it."""
    assert resolve([], 3484) == []


def test_fractions_need_a_known_run_length():
    """max_train_steps is None when neither epochs nor a step cap is set."""
    with pytest.raises(ValueError, match="known run length"):
        resolve([0.5], None)
    # ...but explicit steps alone are fine without one.
    assert resolve([], None, save_steps=[10]) == [10]
