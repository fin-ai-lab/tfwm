"""The supervised specialist trains on CELLS, and says how much of a batch works.

The 2026-09-11 audit of the 3e5087 campaign found a recipe under which nothing
trained: one random crop per row, so the within-cell ranking loss could only
pair rows that collided into a (date, anchor) by chance -- ~74 pairs per
256-row batch, more than half of every batch in no pair at all -- for 1,800
steps at an LR the sweep had picked because it perturbed the random features
least. The saved backbones ended 0.15% from their init. The loss (log 2 for
any weak ranking signal) and the ridge probe (reads random features fine)
both looked normal.

These pin the three things that make the failure visible or impossible:

  * the train view is a cross_stock CELL of K labelled stocks, so every row
    has K-1 partners (``supervised_cell_view``, ``training_step`` on (B, K));
  * ``train/rows_with_grad_frac`` reports the share of rows in a usable pair;
  * ``ParamDrift`` reports how far the encoder has moved from its init.
"""
from __future__ import annotations

import torch

from market_jepa.modeling.modes.supervised import (
    MultiTaskSupervisedModel,
    SupervisedModel,
    _rows_in_pairs,
)
from market_jepa.training.utils import ParamDrift, supervised_cell_view


class _ToyBackbone(torch.nn.Module):
    d_embedding = 4

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, self.d_embedding)

    def forward(self, x, lengths=None):
        return self.proj(x.mean(dim=-1))


def _specialist():
    torch.manual_seed(5)
    m = SupervisedModel(_ToyBackbone(), task="return_900", loss_fn="pairwise")
    m.target_col_idx = 0
    return m


def _cell_batch(b: int = 3, k: int = 4):
    """(b) cells of (k) stocks: views[i] is stock i of every cell."""
    g = torch.Generator().manual_seed(11)
    bucket = {
        "views": [torch.randn(b, 2, 3, generator=g) for _ in range(k)],
        "lengths": [torch.full((b,), 3) for _ in range(k)],
        "targets": torch.rand(b, k, 1, generator=g),
    }
    return {"buckets": [bucket]}


def _row_batch_with_singletons():
    """Six rows in six different cells: nothing is comparable."""
    g = torch.Generator().manual_seed(12)
    n = 6
    bucket = {
        "views": [torch.randn(n, 2, 3, generator=g)],
        "lengths": [torch.full((n,), 3)],
        "targets": torch.rand(n, 1, generator=g),
        "xs_cell": torch.arange(n, dtype=torch.int64),
    }
    return {"buckets": [bucket]}


def test_the_cell_view_borrows_the_probe_geometry_and_adds_stocks():
    one_global = {"global_scale_range": [0.5, 1.0], "global_seq_len": 2048,
                  "global_agg_range": None}
    v = supervised_cell_view(one_global, 16)
    assert v["name"] == "cross_stock" and v["n_stocks"] == 16
    assert v["global_seq_len"] == 2048 and v["global_scale_range"] == [0.5, 1.0]
    # A random subset of the cross-section: no industry restriction, no locals.
    assert v["industry_table"] is None
    assert v["n_local_views"] == 0 and v["cross_stock_local_views"] == 0


def test_a_cell_batch_gives_every_row_a_gradient():
    out = _specialist().training_step(_cell_batch(b=3, k=4), torch.device("cpu"))
    m = out["metrics"]
    assert m["train/rows_per_step"] == 12
    # k(k-1)/2 ordered pairs per cell, three cells.
    assert m["train/loss_units_per_step"] == 3 * 6
    assert m["train/rows_with_grad_frac"] == 1.0
    assert m["train/cells_per_step"] == 3


def test_singleton_cells_report_zero_coverage_instead_of_a_finished_step():
    out = _specialist().training_step(_row_batch_with_singletons(),
                                      torch.device("cpu"))
    # No pair anywhere: the step is skipped rather than logged as converged.
    assert out is None


def test_rows_in_pairs_counts_only_rows_with_a_usable_partner():
    y = torch.tensor([0.1, 0.9, 0.5, float("nan"), 0.3])
    cells = torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64)
    # Row 0 and 1 pair; row 2's partner is NaN; row 4 is alone.
    assert _rows_in_pairs(y, cells) == 2
    # Grouped (B, K): a cell of equal labels has no ordering.
    yk = torch.tensor([[0.2, 0.8, 0.5], [0.4, 0.4, 0.4]])
    assert _rows_in_pairs(yk, None) == 3


def test_the_multihead_ranks_inside_cells_too():
    torch.manual_seed(2)
    tasks = ["return_900", "volatility_change_900"]
    m = MultiTaskSupervisedModel(
        _ToyBackbone(), tasks, loss_fn="pairwise", gradient_norm_ema_decay=0.0,
        gradient_norm_min=1e-8, gradient_norm_max_scale=1e6)
    m.target_col_idx = {t: i for i, t in enumerate(tasks)}
    g = torch.Generator().manual_seed(3)
    b, k = 2, 5
    bucket = {
        "views": [torch.randn(b, 2, 3, generator=g) for _ in range(k)],
        "lengths": [torch.full((b,), 3) for _ in range(k)],
        "targets": torch.rand(b, k, 2, generator=g),
    }
    out = m.training_step({"buckets": [bucket]}, torch.device("cpu"))
    assert out is not None
    # 10 ordered pairs per cell of five, two cells -- within cells, not the
    # 45 pairs the flat surrogate would count over 10 rows.
    assert out["metrics"]["train/loss_units_per_step"] == 2 * 10
    assert out["metrics"]["train/rows_per_step"] == 10


def test_param_drift_is_zero_at_init_and_moves_with_the_weights():
    torch.manual_seed(1)
    bb = _ToyBackbone()
    drift = ParamDrift(bb)
    assert drift() == 0.0
    with torch.no_grad():
        for p in bb.parameters():
            p.add_(0.01 * torch.ones_like(p))
    assert drift() > 0.0
    # Relative: scaling every weight by 1.1 is exactly 0.1.
    torch.manual_seed(1)
    bb2 = _ToyBackbone()
    d2 = ParamDrift(bb2)
    with torch.no_grad():
        for p in bb2.parameters():
            p.mul_(1.1)
    assert abs(d2() - 0.1) < 1e-5


# ── the anchor draw ─────────────────────────────────────────────────────────
#
# The head is scored on a panel whose anchors are spread evenly over the
# feasible band. Drawing the window first and then an anchor it fits before
# piles training cells toward the close (58% in the last two hours vs the
# panel's 35% on 2012-12) and cost the spread head half its panel IC.

def test_anchor_first_draw_matches_the_panel_band_and_feasibility():
    import numpy as np
    from stable_finance.dataset.anchors import SESSION_LEN
    from market_jepa.training.streaming_dataset import _draw_anchor_first

    cfg = {"global_scale_range": (0.5, 1.0), "global_agg_range": None}
    rng = np.random.RandomState(0)
    n_rows, seq_len, grid, slack = SESSION_LEN, 2048, 300, 960
    anchors, aggs = [], []
    for _ in range(4000):
        d = _draw_anchor_first(rng, cfg, n_rows, seq_len, 0, grid, slack)
        assert d is not None
        anchor, agg, window, start = d
        assert anchor % grid == 0
        assert window == agg * seq_len and start == anchor - window + 1 and start >= 0
        assert 6 <= agg <= 11 and agg * seq_len <= anchor + 1
        assert anchor <= SESSION_LEN - 1 - slack
        anchors.append(anchor); aggs.append(agg)
    anchors = np.array(anchors)
    # The panel's band: first lattice point admitting a 6 s/token view.
    assert anchors.min() == 12300
    # Uniform over the band, not piled at the close: the first and last
    # thirds of the band hold about the same share of cells.
    lo, hi = anchors.min(), anchors.max()
    third = (hi - lo) / 3
    early = np.mean(anchors < lo + third); late = np.mean(anchors > hi - third)
    assert abs(early - late) < 0.06, (early, late)
    # Under a 960 s slack the last lattice anchor is 22200, which admits at
    # most 10 s/token (10 * 2048 <= 22201); 11 needs the last 15 minutes the
    # slack reserves. With no slack the coarsest resolution comes back.
    assert max(aggs) == 10
    rng = np.random.RandomState(1)
    free = [_draw_anchor_first(rng, cfg, n_rows, seq_len, 0, grid, 0)[1] for _ in range(2000)]
    assert max(free) == 11


def test_the_cell_view_draws_anchors_first():
    v = supervised_cell_view({"global_scale_range": [0.5, 1.0],
                              "global_seq_len": 2048, "global_agg_range": None}, 16)
    assert v["anchor_uniform"] is True


def test_the_parser_keeps_anchor_uniform_off_unless_asked():
    from market_jepa.augmentations import canonicalize_augmentations
    plain, on = canonicalize_augmentations([
        {"name": "cross_stock", "n_stocks": 2},
        {"name": "cross_stock", "n_stocks": 16, "anchor_uniform": True},
    ])
    assert plain["anchor_uniform"] is False and on["anchor_uniform"] is True
