"""Evaluation metrics: Spearman rank IC.

The reported metric everywhere is the **information coefficient** — the
Spearman rank correlation between a model's prediction and the realized
cross-sectional z-score.

Two aggregations, both computed from the same arrays:

  * :func:`rank_ic` — pooled over whatever rows it is handed. Used for the
    in-training probe, whose samples are independent draws rather than
    synchronized cross-sections.
  * :func:`grouped_rank_ic` — Spearman within each decision instant, then
    averaged over instants. This is the finance convention and the number the
    paper reports; it requires synchronized views (all stocks cropped to the
    same wall-clock end).

Spearman is computed directly from ranks rather than via ``scipy.stats`` so a
whole batch of targets can be ranked in one vectorized pass — the eval nodes
run 8 CPUs against an H100 and per-column scipy calls dominated the budget.
"""

from __future__ import annotations

import numpy as np
from stable_finance import (
    grouped_rank_ic_by_label as _stable_grouped_rank_ic,
    pooled_estimates as _stable_pooled_estimates,
)

# A cross-section smaller than this is dropped rather than scored: Spearman on
# a handful of names is mostly noise and would widen the average's variance
# without adding signal.
MIN_CELL_N = 20


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks of a 1-D array — scipy's ``rankdata(method='average')``.

    Fully vectorized (no per-tie-run Python loop): ties matter here because
    the zero-return mass in ``spread_change`` and wide-quote names is large,
    and mid-rank averaging is what keeps those from biasing the correlation.
    """
    n = a.size
    order = np.argsort(a, kind="stable")
    srt = a[order]
    # Start of each run of equal values, then the dense rank of every position.
    obs = np.r_[True, srt[1:] != srt[:-1]]
    dense = obs.cumsum()
    # count[j] = number of elements strictly before distinct value j.
    count = np.r_[np.flatnonzero(obs), n]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = 0.5 * (count[dense] + count[dense - 1] + 1)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if denom <= 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def rank_ic(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled Spearman correlation between prediction and realized z.

    Rows where either side is NaN are dropped. Returns NaN when fewer than
    :data:`MIN_CELL_N` rows survive or either side is constant.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    result = _stable_grouped_rank_ic(
        pred,
        np.asarray(true, dtype=np.float64).ravel(),
        np.zeros(len(pred), dtype=np.int8),
        min_assets=MIN_CELL_N,
    )
    return result.mean


def grouped_rank_ic(
    pred: np.ndarray, true: np.ndarray, group: np.ndarray,
) -> tuple[float, float, int]:
    """Spearman within each group, averaged over groups.

    Args:
        pred: (N,) predictions.
        true: (N,) realized z-scores.
        group: (N,) integer group id — one decision instant (date, anchor).

    Returns:
        ``(mean_ic, se_of_mean, n_cells)``. Cells with fewer than
        :data:`MIN_CELL_N` usable rows, or a constant side, are skipped. The
        standard error treats cells as independent, which OVERSTATES precision
        when anchors are spaced closer than the horizon (their forward windows
        overlap) — cluster by day before quoting a t-stat.
    """
    result = _stable_grouped_rank_ic(
        pred, true, group, min_assets=MIN_CELL_N,
    )
    return result.mean, result.standard_error, result.observations


def pooled_month_ic(month_ics, month_ses=None):
    """Pool per-month ICs with the MONTH as the unit of clustering.

    This is the reporting standard for every multi-month result. A single
    month's IC is not a reportable number on its own for the weaker targets:
    per-cell IC dispersion is sigma ~ 0.078, and cells stay independent only
    while anchors are spaced at least one horizon apart, which caps a month at
    ~12 anchors x ~20 days ~ 240 cells and therefore SE ~ 0.005. A month is
    significant only if its own IC clears ~0.010, which return never does.

    Why the month and not the cell. Pooling all cells from all months and
    dividing by sqrt(total cells) assumes every cell is an exchangeable draw
    from one distribution. It is not: each month has its OWN checkpoint and
    its own regime, and the between-month dispersion is real signal about how
    the method behaves, not noise to be averaged away. Measured on the k2ind
    series, cell-pooling understates the standard error by 2.0x for
    volatility_change and 3.8x for spread_change (whose monthly ICs range
    +0.036 to +0.108). It is roughly correct only for return, and only because
    return's IC is zero in every month, so there is no between-month variation
    to miss. Choosing the estimator that happens to be right for the null case
    would be exactly backwards.

    ``month_ses`` (the within-month SEs) is optional and used only to report
    the inflation factor, so the cost of clustering stays visible.

    Returns a dict: mean, se, t, n_months, and inflation when SEs are given.
    """
    result = _stable_pooled_estimates(list(month_ics))
    n = result.observations
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "n_months": 0}
    mean, se = result.mean, result.standard_error
    out = {"mean": mean, "se": se, "t": mean / se if se else float("nan"),
           "n_months": n}
    if month_ses is not None:
        s = np.asarray(list(month_ses), dtype=np.float64)
        s = s[np.isfinite(s)]
        if len(s) == n and n > 0:
            se_cell = float(np.sqrt((s ** 2).sum()) / n)
            out["se_cell_pooled"] = se_cell
            out["inflation"] = se / se_cell if se_cell else float("nan")
    return out


def delta_ic(method_ics, baseline_ics, month_ses=None):
    """Pool the PAIRED per-month differences method - random-init baseline.

    This is the reported headline. Raw IC is already centred at zero under the
    null, so the subtraction is not there to remove a bias — it is there to
    remove a MONTH EFFECT. Some months are simply easier to rank than others,
    by more than the spread between methods, and a table of raw ICs across
    different months is mostly reading regime rather than method. Subtracting
    the untrained floor measured on the SAME month, with the SAME probe and
    panel, is what makes two months' numbers comparable and keeps a method
    that merely inherits a random projection's structure from looking like a
    result.

    Pairing matters and is why this is not ``mean(method) - mean(baseline)``:
    the difference is taken WITHIN a month and only then pooled, so the
    between-month variance cancels instead of being added twice. On this data
    that is the difference between a usable standard error and a useless one.

    Args:
        method_ics: per-month IC of the method, aligned with baseline_ics.
        baseline_ics: per-month IC of the random-init encoder, SAME months.
        month_ses: optional within-month SEs, forwarded to report inflation.

    Returns the dict :func:`pooled_month_ic` returns, computed on the
    differences — the month stays the unit of clustering (see that function
    for why the cell is the wrong unit).
    """
    a = np.asarray(list(method_ics), dtype=np.float64)
    b = np.asarray(list(baseline_ics), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"delta_ic needs one baseline per method month, got {a.shape} "
            f"and {b.shape} — an unpaired subtraction would compare a method "
            f"on one set of months against a floor measured on another."
        )
    return pooled_month_ic(a - b, month_ses)
