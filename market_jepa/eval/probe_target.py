"""Cross-sectional target transforms for panels with NO anchor table.

STABLE-FINANCE OWNS THESE DEFINITIONS. ``AnchorTargetStats.transform`` returns
all four cross-sectional representations -- ``raw`` | ``zscore`` | ``uniform``
| ``rank`` -- computed from the exact per-cell order statistics, and every
anchor-backed consumer must read them from there rather than re-deriving them.
``scripts/eval/xs_ic_eval.probe_fit_targets`` does exactly that.

THIS MODULE IS THE FALLBACK FOR THE ONE PANEL THAT HAS NO TABLE. The event
world model (``plots/event_conditioning``) builds a DAY-level panel out of
daily bars, not out of the 1 Hz mosaic, so no anchor table describes its cells
and there is nothing to read a transform off. The names and definitions below
are stable-finance's, so the two paths cannot drift into two vocabularies:

    uniform   average-tie rank / (n + 1)          <- the default
    rank      Phi^-1 of the empirical percentile
    zscore    the caller's target, untouched

``rank`` is offered only for completeness. It re-inflates the tail that
ranking just compressed, which is the thing the transform exists to fix.

WHY THE PROBE IS FIT ON A RANK AT ALL. The reported metric is a within-cell
Spearman rank IC, and Spearman IC IS the Pearson correlation of ranks -- but a
squared-error fit on a moment z-score puts a day's biggest movers at 3.5
sigma, where they dominate the normal equations while being exactly what the
metric cannot see. Measured on the day-level panel (fit 2018-22, scored
2023-24), every target rescaled to unit variance so the ridge penalty means
the same thing across transforms:

    fit target      alpha=1    alpha=10   alpha=100
    zscore          +0.7727    +0.7799    +0.7642
    rank            +0.8103    +0.8105    +0.7872
    uniform         +0.8167    +0.8152    +0.7905

Monotone at every penalty, so it is the transform and not extra shrinkage.
The between-encoder CONTRAST barely moves (+0.0451 -> +0.0465), which is why
every readout comparison survives the switch -- but the ABSOLUTE levels do
not, and numbers fit under two settings must not share a table.

ONLY THE FIT TARGET MOVES. The eval target is untouched, so there is no
leakage: this changes how the probe is ESTIMATED, never what it is judged
against. Set ``XS_PROBE_FIT_TARGET=zscore`` to reproduce anything reported
before 2026-09-06.
"""

import os

import numpy as np
from stable_finance.dataset.targets import TARGET_TRANSFORMS

PROBE_FIT_TARGET = os.environ.get("XS_PROBE_FIT_TARGET", "uniform")


def probe_fit_target(y, cells, kind: str | None = None):
    """Monotone within-CELL transform of a probe's FIT target.

    Args:
        y: (n,) targets, already cross-sectionally standardized.
        cells: (n,) cell id per row -- the unit the IC is computed within.
            Ranking pooled across cells would fold back in the day effect the
            standardization exists to remove.
        kind: a member of stable-finance's ``TARGET_TRANSFORMS``; defaults to
            ``PROBE_FIT_TARGET``. ``raw`` is not meaningful here -- the caller
            has already standardized -- and is rejected.

    Returns:
        (n,) transformed target, NaN preserved. Rescaled to unit variance so a
        fixed ridge ``alpha`` keeps its meaning across transforms.
    """
    kind = kind or PROBE_FIT_TARGET
    y = np.asarray(y, dtype=np.float64)
    if kind == "zscore":
        return y
    if kind not in ("uniform", "rank"):
        raise ValueError(
            f"unknown probe fit target: {kind!r}; stable-finance defines "
            f"{TARGET_TRANSFORMS} and only zscore/uniform/rank apply to an "
            "already-standardized target"
        )

    from scipy.stats import norm, rankdata

    cells = np.asarray(cells)
    out = np.full(y.shape, np.nan)
    for c in np.unique(cells):
        m = cells == c
        v = y[m]
        ok = np.isfinite(v)
        if ok.sum() < 5:          # too thin to rank; the caller's NaN filter drops it
            continue
        r = rankdata(v[ok]) / (ok.sum() + 1)
        vv = np.full(v.shape, np.nan)
        vv[ok] = r if kind == "uniform" else norm.ppf(r)
        out[m] = vv
    sd = np.nanstd(out)
    return (out - np.nanmean(out)) / (sd if sd > 0 else 1.0)
