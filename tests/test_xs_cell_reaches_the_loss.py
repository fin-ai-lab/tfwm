"""The cross-section id must survive collation and select the within-cell loss.

THREE PIECES IMPLEMENT ONE BEHAVIOUR and they were disconnected for the whole
campaign. StreamingMarketDataset emits ``xs_cell`` -- one id per (date,
wall-clock anchor) -- so SupervisedModel can rank WITHIN a cross-section, which
is the quantity the reported rank IC measures. ``collate_bucketed`` did not
copy the key, so ``bucket.get("xs_cell")`` was always None, ``compute_loss``
took its cells-is-None branch, and ``_within_cell_loss`` never ran: every
supervised head trained on the flat surrogate that ranks stocks against other
stocks at DIFFERENT instants.

Nothing failed. The loss went down, the runs finished, the code read as though
it ranked within cells. Only the head-vs-probe gap showed it, and only after
the loss path was traced by hand. These tests assert the wiring end to end so
the next disconnection is a red test rather than a silent recipe change.
"""
from __future__ import annotations

import torch

from market_jepa.modeling.modes.supervised import SupervisedModel
from market_jepa.training.utils import collate_bucketed


def _sample(cell: int, target: float, n_ch: int = 9, t: int = 16) -> dict:
    return {
        "views": [torch.zeros(n_ch, t)],
        "lengths": torch.tensor([t]),
        "bucket_key": 0,
        "targets": torch.tensor([target]),
        "xs_cell": torch.tensor(cell, dtype=torch.int64),
    }


def test_collate_preserves_xs_cell():
    batch = collate_bucketed([_sample(100, 0.1), _sample(100, 0.9),
                              _sample(200, 0.5)])
    bucket = batch["buckets"][0]
    assert "xs_cell" in bucket, (
        "collate dropped xs_cell; compute_loss will silently fall back to the "
        "flat cross-cell surrogate")
    assert bucket["xs_cell"].dtype == torch.int64
    assert bucket["xs_cell"].tolist() == [100, 100, 200]


class _Bare(SupervisedModel):
    """compute_loss without building a backbone."""

    def __init__(self, loss_fn="pairwise"):
        torch.nn.Module.__init__(self)
        self.loss_fn = loss_fn
        self.smooth_l1_beta = 1.0
        self.target_col_idx = 0


def test_cells_select_the_within_cell_loss_and_change_the_number():
    """With cells, only same-cell pairs count -- so the loss must differ."""
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    targets = torch.tensor([[0.1], [0.9], [0.2], [0.8]])
    cells = torch.tensor([7, 7, 8, 8], dtype=torch.int64)
    m = _Bare()
    flat, n_flat = m.compute_loss(pred, targets, cells=None)
    grouped, n_pairs = m.compute_loss(pred, targets, cells=cells)
    # The flat path returns a ROW count; the within-cell path returns PAIRS.
    assert n_flat == 4
    # Only (0,1) and (2,3) share a cell and are correctly ordered.
    assert n_pairs == 2
    assert not torch.isclose(flat, grouped), (
        "cells made no difference; the within-cell branch did not run")


def test_cross_cell_pairs_are_excluded_not_downweighted():
    """A cell with one member contributes nothing at all."""
    pred = torch.tensor([0.0, 1.0, 5.0])
    targets = torch.tensor([[0.1], [0.9], [0.5]])
    m = _Bare()
    both = m.compute_loss(pred, targets,
                          cells=torch.tensor([1, 1, 2], dtype=torch.int64))
    # Dropping the singleton cell entirely must not move the loss.
    only = m.compute_loss(pred[:2], targets[:2],
                          cells=torch.tensor([1, 1], dtype=torch.int64))
    assert both[1] == only[1] == 1
    assert torch.isclose(both[0], only[0])
