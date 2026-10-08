"""The MULTIHEAD must rank within cells too -- it is a different class.

00fbb35 made collate_bucketed carry xs_cell and SupervisedModel use it, and
the commit message said the blast radius was "supervised + multihead (which
shares the loss)". The multihead does NOT share the loss.
MultiTaskSupervisedModel never defined compute_loss and never read xs_cell: it
went straight to _per_task_loss -> _regression_loss -> _pairwise_rank_loss,
which takes no cells and pairs across the whole batch.

So the multihead kept training the flat cross-cell surrogate for the entire
life of the fix that removed it from the specialists, and a 203-month campaign
was trained that way before anyone noticed. The check that would have caught
it -- train/loss_units_per_step -- was added to SupervisedModel only, so it
returned "absent" for the one class that needed it and that read as a logging
gap rather than as the answer.

These assert the wiring on the class that was missed, and that every class
computing a supervised loss can receive cells at all.
"""
from __future__ import annotations

import inspect

import torch

from market_jepa.modeling.modes.supervised import (
    MultiTaskSupervisedModel,
    SupervisedModel,
    _within_cell_rank_loss,
)


def test_within_cell_loss_is_module_level_so_both_classes_can_use_it():
    """A method on one class is how the multihead came to miss it."""
    assert callable(_within_cell_rank_loss)
    assert not inspect.ismethod(_within_cell_rank_loss)


def test_every_supervised_loss_entry_point_accepts_cells():
    """The structural guard: one representative is not the family."""
    for cls, name in ((SupervisedModel, "compute_loss"),
                      (MultiTaskSupervisedModel, "_per_task_loss")):
        fn = getattr(cls, name, None)
        assert fn is not None, f"{cls.__name__} has no {name}"
        params = inspect.signature(fn).parameters
        assert "cells" in params, (
            f"{cls.__name__}.{name} cannot receive cell ids, so it can only "
            f"ever rank across the whole batch")


class _BareMulti(MultiTaskSupervisedModel):
    """_per_task_loss without building a backbone or heads."""

    def __init__(self, loss_fn="pairwise"):
        torch.nn.Module.__init__(self)
        self.loss_fn = loss_fn
        self.smooth_l1_beta = 1.0


def test_cells_change_the_multihead_loss():
    """Same rows, same targets: only the cell partition differs."""
    model = _BareMulti()
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([0.1, 0.9, 0.2, 0.8])
    valid = torch.ones(4, dtype=torch.bool)
    cells = torch.tensor([7, 7, 8, 8], dtype=torch.int64)

    flat, n_flat = model._per_task_loss(pred, y, valid, None, None, "t")
    within, n_within = model._per_task_loss(pred, y, valid, cells, None, "t")

    assert n_flat == 4, "the flat path weights by ROWS"
    assert n_within == 2, "two same-cell ordered pairs, not six"
    assert not torch.isclose(flat, within), (
        "cells did not change the loss -- the multihead is still ranking "
        "across the whole batch")


def test_cross_cell_pairs_are_excluded_not_downweighted():
    """A batch where every row is its own cell has nothing comparable."""
    model = _BareMulti()
    pred = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = torch.tensor([0.1, 0.9, 0.5])
    valid = torch.ones(3, dtype=torch.bool)
    cells = torch.tensor([1, 2, 3], dtype=torch.int64)
    loss, n = model._per_task_loss(pred, y, valid, cells, None, "t")
    assert n == 0, "distinct cells must yield no pairs"


def test_the_multihead_weights_by_pairs_when_cells_are_present():
    """n is PAIRS within cells and ROWS flat -- the caller weights by it."""
    model = _BareMulti()
    pred = torch.arange(6, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    valid = torch.ones(6, dtype=torch.bool)
    one_cell = torch.zeros(6, dtype=torch.int64)
    _, n = model._per_task_loss(pred, y, valid, one_cell, None, "t")
    assert n == 15, f"6 rows in one cell is 15 ordered pairs, got {n}"


# ── the wiring, at training_step level ──────────────────────────────────────
#
# The unit tests above prove _per_task_loss uses cells WHEN GIVEN THEM. That is
# not what broke: training_step never read the key. These drive the real entry
# point, which is the only level at which the original bug was visible.

class _ToyBackbone(torch.nn.Module):
    d_embedding = 4

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, self.d_embedding)

    def forward(self, x, lengths=None):
        return self.proj(x.mean(dim=-1))


_TASKS = ["return_900", "volatility_change_900", "spread_change_900"]


def _model(loss_fn="pairwise"):
    # PAIRWISE, which is what the campaign trains. The constructor defaults to
    # "mse", and MSE is not a ranking loss, so a model built without this
    # argument takes the flat path CORRECTLY and the test would pass while
    # proving nothing.
    torch.manual_seed(31)
    m = MultiTaskSupervisedModel(
        _ToyBackbone(), _TASKS, loss_fn=loss_fn, gradient_norm_ema_decay=0.0,
        gradient_norm_min=1e-8, gradient_norm_max_scale=1e6,
    )
    m.target_col_idx = {t: i for i, t in enumerate(_TASKS)}
    return m


def _batch(with_cells: bool):
    g = torch.Generator().manual_seed(73)
    n = 10
    bucket = {
        "views": [torch.randn(n, 2, 3, generator=g)],
        "lengths": torch.full((n,), 3),
        "targets": torch.randn(n, 3, generator=g),
    }
    if with_cells:
        # Two cross-sections of five, so within-cell pairs exist and are
        # strictly fewer than the flat path's row count.
        bucket["xs_cell"] = torch.tensor([0] * 5 + [1] * 5, dtype=torch.int64)
    return {"buckets": [bucket]}


def test_training_step_reads_xs_cell_from_the_bucket():
    """The failure was here: the key was collated and never read."""
    out = _model().training_step(_batch(with_cells=True), torch.device("cpu"))
    assert out is not None
    m = out["metrics"]
    assert "train/loss_units_per_step" in m, (
        "the multihead logs no loss-path metric, which is what made a "
        "203-month flat-loss campaign invisible")
    # TWO cells of five, so 10 ordered pairs each: 20 pairs against 10 rows.
    # NOT "pairs < rows" -- that holds in production (256 rows spread over
    # ~720 monthly cells gives ~75 pairs) but it is a property of the cell
    # SIZE, not of the code path, and asserting it here would encode a
    # production coincidence as an invariant.
    assert m["train/rows_per_step"] == 10
    assert m["train/loss_units_per_step"] == 20, (
        f"expected 20 within-cell pairs, got "
        f"{m['train/loss_units_per_step']}: at 10 rows the flat path reports "
        f"10, so this says which branch ran")


def test_an_mse_multihead_ignores_cells_by_design():
    """MSE is not a ranking loss, so the partition cannot apply to it."""
    out = _model(loss_fn="mse").training_step(
        _batch(with_cells=True), torch.device("cpu"))
    m = out["metrics"]
    assert m["train/loss_units_per_step"] == m["train/rows_per_step"]


def test_training_step_without_cells_weights_by_rows():
    """The fallback must still work and must be distinguishable."""
    out = _model().training_step(_batch(with_cells=False), torch.device("cpu"))
    m = out["metrics"]
    assert m["train/loss_units_per_step"] == m["train/rows_per_step"]


def test_cells_change_what_training_step_optimizes():
    """Same rows and targets; only the partition differs."""
    a = _model().training_step(_batch(with_cells=False), torch.device("cpu"))
    b = _model().training_step(_batch(with_cells=True), torch.device("cpu"))
    assert a["metrics"]["train/loss"] != b["metrics"]["train/loss"]
