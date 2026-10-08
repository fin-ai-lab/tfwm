"""Task registry for supervised baselines.

Every task defaults to a **regression onto the empirical-uniform rank** of a
forward target:

    uniform_i(t, h) = rankdata(y_cell)_i / (n_cell + 1)

where the cell is all stocks at the same date and decision instant. Exact order
statistics come from the precomputed anchor tables (see
``stable_finance.dataset.targets``), so the dataset only performs a lookup.

Consequences that motivate the design:

  * The empirical-uniform target is dimensionless and bounded strictly inside
    (0, 1), so extreme raw returns do not dominate MSE.
  * Ranking it within one instant is identical to ranking the raw target, so
    the training objective is aligned with the reported metric: Spearman rank
    IC of the prediction against the realized target, computed per cross-section.

Because the cross-sectional tables only exist on the anchor grid, the view's
END must be snapped to that grid (``augmentations.*.end_grid_sec``); otherwise
the lookup would use a different instant than the one the return is measured
from.

Registered names follow ``{target_type}_{horizon}``, e.g.::

    return_900, volatility_change_300, spread_change_7200

The task no longer encodes a head shape or a loss. A supervised run picks
those with ``mode.loss_fn``, and the paper's supervised ablation is exactly
that axis held against one fixed task. Binned losses need no per-task setting
at all: ``market_jepa.eval.discretize`` fits equal-count quantile bins and
spreads tied values across the bins they straddle, which works the same way
for every target type -- there is no longer a zero-bin scheme to select.
"""

from dataclasses import dataclass

from stable_finance.dataset.anchors import (
    DEFAULT_TARGET_HORIZONS,
    RETURN_VWAP_WINDOW,
    SESSION_LEN,
)
from stable_finance.dataset.outcomes import ANCHOR_TARGET_TYPES


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single supervised prediction task."""

    name: str
    target_type: str          # "return" | "volatility_change" | "spread_change"
    horizon: int              # forward horizon in seconds

    @property
    def target_col(self) -> str:
        """Column name produced by ``targets.get_target_names``.

        The dataset emits its configured target transform under this name.
        """
        return f"{self.target_type}_{self.horizon:03d}"


TASK_REGISTRY: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> TaskSpec:
    TASK_REGISTRY[spec.name] = spec
    return spec


TARGET_TYPES = ANCHOR_TARGET_TYPES

# The reported horizon sweep, which is stable-finance's default set --
# imported rather than restated, because a horizon is a TARGET DEFINITION and
# that package owns those. 7200 (2h) doubles as the event-conditioning long
# horizon. Anchors with less than h seconds of session left get a NaN target
# for that horizon (no close clamp — the row is simply dropped), so the usable
# sample count falls as h grows.
HORIZONS = DEFAULT_TARGET_HORIZONS

for _t in TARGET_TYPES:
    for _h in HORIZONS:
        register_task(TaskSpec(name=f"{_t}_{_h}", target_type=_t, horizon=_h))

# ── The whole-session horizon ────────────────────────────────────────────────
#
# DERIVED, not a constant: open -> close is the longest horizon whose forward
# window still closes on the session, i.e. SESSION_LEN - RETURN_VWAP_WINDOW.
# Both ends are then measured exactly as every other horizon measures them --
# vwap[0, 60) -> vwap[23340, 23400) -- and the last window ends on the bell
# rather than one second past it. Spelling it in stable-finance's own session
# constants is what keeps that true if the session or the measurement window
# is ever redefined; a literal 23340 would quietly become wrong.
#
# It is deliberately NOT in HORIZONS: that tuple is the reported sweep and the
# probe's column set, and one horizon 3x the next longest would dominate both.
# stable-finance builds it on request (``--horizons ... 23340``) like any
# other; nothing about it is special to the target machinery.
#
# This is the DECODER target of the event world model (plots/event_conditioning):
# a full-day view is encoded to one latent, and the decoder reads that day's
# cross-sectional return rank back off it. Because the label is measured
# FORWARD from the session open, a view carrying it is anchored at its FIRST
# row, not its last — see dataset.label_at_view_start.
DAY_HORIZON = SESSION_LEN - RETURN_VWAP_WINDOW

for _t in TARGET_TYPES:
    register_task(TaskSpec(name=f"{_t}_{DAY_HORIZON}", target_type=_t,
                           horizon=DAY_HORIZON))
