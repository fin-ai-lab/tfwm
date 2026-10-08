"""Fitted parameters for the classical finance baselines, estimated per month.

The baselines in :mod:`.finance_baselines` are three textbook models — AR(1) for
returns, GARCH(1,1) for volatility, AR(1) for the spread. Their *state* (where a
given window sits right now) is per-sample, but their *parameters* are pooled:
one set of coefficients shared across every asset, refit per calendar month.
That mirrors how these models are used in practice — you do not fit a separate
GARCH per stock per window — and it makes the fitted object a small, inspectable
artifact you can save, diff and re-load, exactly like a model checkpoint.

Why parameters are keyed by ``(month, agg_factor)``
---------------------------------------------------
A view is not a 1 Hz series: ``_random_resized_crop_numpy`` aggregates a random
fraction of the session down to a fixed token count, so one token is ~6-11
seconds and **the factor is resampled per sample**. Persistence is a function of
the sampling interval — an AR(1) coefficient at 6 s/token is not the same number
as at 11 s/token (roughly ``phi_11 ~ phi_6 ** 1.8``) — so pooling across
aggregation scales would blur two different quantities together. Rather than
approximate a continuous-time reparameterisation, parameters are simply
estimated separately per ``(month, agg_factor)`` cell. ``agg_factor`` takes only
a handful of integer values, so the cells stay well populated.

Everything is estimated on **per-window standardized** series
-------------------------------------------------------------
Views are per-sample z-scored (:func:`market_jepa.training.utils.normalize_numpy`),
which leaves normalized mid as an affine map of raw mid with an unknown
per-sample scale ``1/sigma``. Increments and spreads therefore carry a
per-sample factor that is meaningless across samples. Each series is consequently
standardized by its own mean and standard deviation before anything is fitted:

  * ``z = (dmid - mean) / std``   — unit-variance return innovations
  * ``w = (spread - mean) / std`` — unit-variance spread level

The unknown ``1/sigma`` cancels, so coefficients are comparable across assets and
poolable. It also makes GARCH **variance targeting** exact: ``Var(z) = 1`` by
construction, so the intercept is pinned at ``omega = 1 - alpha - beta`` and only
two parameters are free.

The cost is that the absolute *level* of volatility is unrecoverable from a
single normalized window, so the volatility baseline predicts a **relative** vol
change (in units of the window's own return std). That is a property of the
input, not of the baseline — the learned encoders see exactly the same
normalized views, so the comparison stays fair.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import torch

# Focal feature-channel indices (see FEATURE_COLUMNS in streaming_dataset.py).
# The focal stock's 9 columns always occupy 0..8; optional risk-factor columns
# are appended after, so these stay valid regardless of risk-factor config.
_BID = 0
_ASK = 4

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The fitted object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineParams:
    """One pooled parameter set: GARCH(1,1) plus per-horizon AR(1) coefficients.

    GARCH ``(alpha, beta)`` is a property of the sampling interval, shared across
    horizons. The AR(1) coefficients are **per horizon**: ``phi_return_by_h[h]``
    is the predictive correlation between the trailing-``h`` return and the next-
    ``h`` return, and likewise for the spread change. They are estimated at the
    horizon frequency — one ``(trailing-h, next-h)`` pair per calibration sample
    — because iterating a token-scale AR(1) out to 15 or 120 minutes is not what
    an AR return model represents; the coefficient at 15 minutes is a different
    number from the one at 3 seconds.

    Attributes:
        alpha: GARCH news-impact coefficient.
        beta: GARCH persistence-of-variance coefficient.
        phi_return_by_h: ``{horizon_sec: corr}`` for the return AR(1).
        phi_spread_by_h: ``{horizon_sec: corr}`` for the spread-change AR(1).
        n_series: Number of windows the cell was fitted on (diagnostic).
        n_steps: Total number of time steps fitted on (diagnostic).
    """

    alpha: float
    beta: float
    phi_return_by_h: dict = field(default_factory=dict)
    phi_spread_by_h: dict = field(default_factory=dict)
    n_series: int = 0
    n_steps: int = 0

    def phi_return(self, horizon: int) -> float:
        """Return AR(1) coefficient for ``horizon`` seconds (0.0 if unfitted)."""
        return float(self.phi_return_by_h.get(int(horizon), 0.0))

    def phi_spread(self, horizon: int) -> float:
        """Spread-change AR(1) coefficient for ``horizon`` seconds (0.0 default)."""
        return float(self.phi_spread_by_h.get(int(horizon), 0.0))

    @property
    def omega(self) -> float:
        """Variance-targeted intercept: unit unconditional variance."""
        return 1.0 - self.alpha - self.beta

    @property
    def persistence(self) -> float:
        """``alpha + beta`` — the rate the variance forecast reverts to 1."""
        return self.alpha + self.beta


def _month_of(date_str: str) -> str:
    """``"2023-04-17"`` -> ``"2023-04"``. Empty/malformed dates -> ``""``."""
    return date_str[:7] if len(date_str) >= 7 else ""


def _params_from_row(row: dict) -> BaselineParams:
    """Build :class:`BaselineParams` from a JSON row, restoring int horizon keys.

    JSON stringifies the ``phi_*_by_h`` dict keys (``300`` -> ``"300"``); convert
    them back so ``phi_return(300)`` hits.
    """
    def _int_keys(d: dict | None) -> dict:
        return {int(k): float(v) for k, v in (d or {}).items()}

    return BaselineParams(
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        phi_return_by_h=_int_keys(row.get("phi_return_by_h")),
        phi_spread_by_h=_int_keys(row.get("phi_spread_by_h")),
        n_series=int(row.get("n_series", 0)),
        n_steps=int(row.get("n_steps", 0)),
    )


class MonthlyBaselineParams:
    """A ``(month, agg_factor) -> BaselineParams`` table plus a global fallback.

    Persisted as a single JSON file so a fitted baseline is as portable as a
    checkpoint. Use :meth:`fit` (via ``scripts/fit_finance_baselines.py``) to
    build one, :meth:`save` / :meth:`load` to round-trip it, and :meth:`lookup`
    at encode time.
    """

    def __init__(
        self,
        cells: dict[tuple[str, int], BaselineParams],
        fallback: BaselineParams,
    ):
        self.cells = dict(cells)
        self.fallback = fallback
        self._lookup_cache: dict[tuple[str, int, str | None], BaselineParams] = {}

    # -- lookup ---------------------------------------------------------

    def lookup(
        self, month: str, agg: int, as_of: str | None = None
    ) -> BaselineParams:
        """Best available parameters for a ``(month, agg)`` cell.

        Resolution order, each step widening only after the previous fails:

        1. the exact cell;
        2. same month, nearest ``agg_factor``;
        3. same ``agg_factor``, nearest *earlier* month (preferred over a later
           one so a baseline never uses parameters fitted on its own future),
           then nearest later month;
        4. the pooled global fallback.

        ``as_of`` caps resolution at that month, inclusive: cells from later
        months are invisible to every step. Without it, steps 1-2 hand an
        eval-month sample the parameters fitted on that month's own future
        returns whenever the artifact covers the eval month — in-sample
        forecasts the learned-encoder side never gets. Pass the *train*
        month when scoring (``FinanceBaseline.params_as_of``); the cap makes
        the earlier-month preference in step 3 an actual guarantee.
        """
        key = (month, int(agg), as_of)
        hit = self._lookup_cache.get(key)
        if hit is not None:
            return hit

        resolved = self._resolve(month, int(agg), as_of)
        self._lookup_cache[key] = resolved
        return resolved

    def _resolve(
        self, month: str, agg: int, as_of: str | None = None
    ) -> BaselineParams:
        cells = (
            self.cells
            if as_of is None
            else {k: p for k, p in self.cells.items() if k[0] <= as_of}
        )
        exact = cells.get((month, agg))
        if exact is not None:
            return exact

        same_month = [(a, p) for (m, a), p in cells.items() if m == month]
        if same_month:
            return min(same_month, key=lambda ap: abs(ap[0] - agg))[1]

        same_agg = [(m, p) for (m, a), p in cells.items() if a == agg]
        if same_agg:
            earlier = [mp for mp in same_agg if mp[0] <= month]
            if earlier:
                return max(earlier, key=lambda mp: mp[0])[1]
            return min(same_agg, key=lambda mp: mp[0])[1]

        return self.fallback

    # -- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        """Write the table to ``path`` as JSON (parent dirs created)."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "version": 2,
            "fallback": asdict(self.fallback),
            "cells": [
                {"month": m, "agg_factor": a, **asdict(p)}
                for (m, a), p in sorted(self.cells.items())
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MonthlyBaselineParams":
        with open(path) as f:
            payload = json.load(f)
        cells = {}
        for row in payload["cells"]:
            row = dict(row)
            month = row.pop("month")
            agg = int(row.pop("agg_factor"))
            cells[(month, agg)] = _params_from_row(row)
        return cls(cells=cells, fallback=_params_from_row(dict(payload["fallback"])))

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        months = sorted({m for m, _ in self.cells})
        aggs = sorted({a for _, a in self.cells})
        span = f"{months[0]}..{months[-1]}" if months else "empty"
        return (
            f"MonthlyBaselineParams({len(self.cells)} cells, "
            f"months={span}, agg_factors={aggs})"
        )


# ---------------------------------------------------------------------------
# Series extraction — shared by the fitter and the model
# ---------------------------------------------------------------------------


@dataclass
class StandardizedSeries:
    """Per-window standardized return and spread series.

    All tensors are float64 and ``(B, ...)``. Masked-out positions are zero, so
    masked sums need no further gating.

    Attributes:
        z: ``(B, T-1)`` standardized return innovations, unit variance.
        z_mask: ``(B, T-1)`` bool, valid increments.
        w: ``(B, T)`` standardized spread level, unit variance.
        w_mask: ``(B, T)`` bool, valid levels.
        drift: ``(B,)`` per-step return drift in standardized units
            (``mean(dmid) / std(dmid)``) — the AR(1) constant.
        w_last: ``(B,)`` standardized spread at the window's last valid step.
    """

    z: torch.Tensor
    z_mask: torch.Tensor
    w: torch.Tensor
    w_mask: torch.Tensor
    drift: torch.Tensor
    w_last: torch.Tensor


def _masked_moments(
    s: torch.Tensor, m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked ``(mean, std)`` of ``(B, T)`` over time. Returns ``(B,)`` each."""
    denom = m.sum(1).clamp(min=1).to(s.dtype)
    mean = (s * m).sum(1) / denom
    ex2 = (s * s * m).sum(1) / denom
    std = (ex2 - mean * mean).clamp(min=0).sqrt()
    return mean, std


def extract_series(
    view: torch.Tensor, lengths: torch.Tensor | None,
) -> StandardizedSeries:
    """Build standardized return/spread series from one normalized view.

    Args:
        view: ``(B, C, T)`` normalized view; focal bid/ask at channels 0 and 4.
        lengths: ``(B,)`` valid length per sample, or ``None`` for full ``T``.

    Returns:
        A :class:`StandardizedSeries`. Float64 throughout: ``spread = ask - bid``
        subtracts two O(1) normalized channels, and in float32 that cancellation
        costs enough significant digits to visibly perturb the spread AR(1).
    """
    v = view.double()
    B, _C, T = v.shape
    device = v.device

    if lengths is None:
        lengths = torch.full((B,), T, device=device, dtype=torch.long)
    else:
        lengths = lengths.to(device).long().clamp(min=1, max=T)

    mid = (v[:, _BID] + v[:, _ASK]) * 0.5
    spr = v[:, _ASK] - v[:, _BID]

    t_idx = torch.arange(T, device=device)
    w_mask = t_idx[None, :] < lengths[:, None]

    # Increments live on T-1 positions; sample b has lengths[b] - 1 valid ones.
    Tr = max(T - 1, 0)
    r = mid[:, 1:] - mid[:, :-1] if Tr else mid[:, :0]
    r_idx = torch.arange(Tr, device=device)
    z_mask = r_idx[None, :] < (lengths - 1).clamp(min=0)[:, None]

    r_mean, r_std = _masked_moments(r, z_mask)
    z = ((r - r_mean[:, None]) / (r_std[:, None] + _EPS)) * z_mask

    s_mean, s_std = _masked_moments(spr, w_mask)
    w = ((spr - s_mean[:, None]) / (s_std[:, None] + _EPS)) * w_mask

    last_pos = (lengths - 1).clamp(min=0)
    w_last = w.gather(1, last_pos[:, None]).squeeze(1)

    return StandardizedSeries(
        z=z,
        z_mask=z_mask,
        w=w,
        w_mask=w_mask,
        drift=r_mean / (r_std + _EPS),
        w_last=w_last,
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trailing-h predictors — shared by the fitter and the model
# ---------------------------------------------------------------------------
#
# The AR(1) predictor for horizon h is the realized change over the trailing h,
# so 5-min forecasts read the last 5 min, 120-min the last 120 min — matching how
# the GARCH vol term reads trailing-h realized vol. Both predictors are scale-
# invariant: the return one sums unit-variance increments (the per-window price
# scale cancels), the spread one differences the standardized spread level.


def trailing_return(
    z: torch.Tensor, z_mask: torch.Tensor, n: torch.Tensor,
) -> torch.Tensor:
    """Sum of the last ``n`` valid **de-meaned** increments per row. Returns ``(B,)``.

    ``z`` is ``(B, Tr)`` (unit-variance, zero-mean increments, right-padded with
    zeros); ``n`` is ``(B,)`` clamped to each row's valid length. Note this is the
    drift-*removed* cumulative — :func:`trailing_return_predictor` adds the drift
    back to form the actual return predictor. Kept separate so both are testable.
    """
    B, Tr = z.shape
    if Tr == 0:
        return z.new_zeros(B)
    idx = torch.arange(Tr, device=z.device)[None, :]
    valid_len = z_mask.sum(1)                                  # (B,)
    n_eff = torch.minimum(n.to(valid_len), valid_len).clamp(min=0)
    start = (valid_len - n_eff).clamp(min=0)
    window = (idx >= start[:, None]) & (idx < valid_len[:, None])
    return (z * window).sum(1)


def trailing_return_predictor(
    s: "StandardizedSeries", n: torch.Tensor,
) -> torch.Tensor:
    """Trailing-``n``-step return in per-window volatility units. Returns ``(B,)``.

    Equals ``(mid_last - mid_{last-n}) / std(increments)`` — scale-invariant (the
    per-window price scale cancels) and drift-*preserving*, which matters because
    a steady trend is the strongest momentum signal and de-meaning would erase it.
    Built as the de-meaned cumulative plus ``n * drift`` since ``extract_series``
    de-means the increments into ``z``; ``n`` is clamped to the valid length to
    match :func:`trailing_return`.
    """
    valid_len = s.z_mask.sum(1)
    n_eff = torch.minimum(n.to(valid_len), valid_len).clamp(min=0)
    return trailing_return(s.z, s.z_mask, n) + n_eff.to(s.drift) * s.drift


def trailing_spread_change(
    w: torch.Tensor, w_mask: torch.Tensor, n: torch.Tensor,
) -> torch.Tensor:
    """Standardized spread at the last valid step minus its value ``n`` steps back.

    ``w`` is ``(B, T)`` (standardized spread level); ``n`` is ``(B,)``, clamped so a
    horizon longer than the window reaches back to the window's first step.
    """
    valid_len = w_mask.sum(1)
    last = (valid_len - 1).clamp(min=0)
    n_eff = torch.minimum(n.to(valid_len), (valid_len - 1).clamp(min=0)).clamp(min=0)
    prev = (last - n_eff).clamp(min=0)
    w_last = w.gather(1, last[:, None]).squeeze(1)
    w_prev = w.gather(1, prev[:, None]).squeeze(1)
    return w_last - w_prev


def steps_for_horizon(horizon_sec: int, agg: float) -> int:
    """Horizon in seconds -> number of aggregated tokens (at least one)."""
    return max(1, round(horizon_sec / max(agg, 1.0)))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def corr_from_stats(
    n: float, sx: float, sy: float, sxy: float, sxx: float, syy: float,
) -> float:
    """Pearson correlation from streamed moments, clipped to the AR(1) range.

    Used to fit the per-horizon AR coefficient as the predictive correlation of
    the trailing-h predictor with the next-h target — sign is what matters for
    the probe ranking; magnitude is a bounded, reportable coefficient.
    """
    if n < 2:
        return 0.0
    cov = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom <= 0.0 or not math.isfinite(cov) or not math.isfinite(denom):
        return 0.0
    return max(-0.999, min(0.999, cov / denom))


def garch11_nll(
    z: torch.Tensor,
    mask: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood of GARCH(1,1) over a grid of parameters.

    Variance-targeted: ``omega = 1 - alpha - beta``, which is exact because ``z``
    is standardized to unit variance. The recursion is
    ``h_{t+1} = omega + alpha * z_t^2 + beta * h_t`` seeded at the unconditional
    variance ``h_0 = 1``, and is frozen past each window's valid length so
    right-padding cannot leak into the likelihood.

    Args:
        z: ``(B, Tr)`` standardized returns.
        mask: ``(B, Tr)`` bool validity.
        alpha: ``(G,)`` candidate news-impact coefficients.
        beta: ``(G,)`` candidate persistence coefficients, same length as alpha.

    Returns:
        ``(G,)`` NLL (dropping the constant ``log 2*pi`` term), summed over all
        valid observations.
    """
    G = alpha.shape[0]
    omega = (1.0 - alpha - beta)[:, None]      # (G, 1)
    a = alpha[:, None]
    b = beta[:, None]

    h = torch.ones(G, z.shape[0], dtype=z.dtype, device=z.device)   # (G, B)
    nll = torch.zeros(G, dtype=z.dtype, device=z.device)

    for t in range(z.shape[1]):
        zt = z[:, t][None, :]                  # (1, B)
        mt = mask[:, t][None, :]               # (1, B)
        # Score z_t under the variance forecast h formed strictly before t.
        contrib = torch.log(h) + (zt * zt) / h
        nll = nll + (contrib * mt).sum(dim=1)
        h_next = omega + a * zt * zt + b * h
        h = torch.where(mt, h_next, h)

    return nll


def fit_garch11(
    z: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_coarse: int = 16,
    n_fine: int = 12,
) -> tuple[float, float]:
    """Fit ``(alpha, beta)`` by variance-targeted MLE over a coarse-to-fine grid.

    A grid search rather than an optimiser: with only two free parameters on a
    bounded, well-behaved domain it is fully deterministic, cannot fail to
    converge, and needs no gradients through a 2000-step recursion.

    The grid is laid out in ``(persistence, alpha)`` rather than ``(alpha, beta)``
    because at 6-11 second sampling the persistence sits very close to 1, and a
    uniform grid in ``beta`` would spend nearly all its points in a region the
    data rules out. ``1 - persistence`` is therefore spaced logarithmically.

    Returns:
        ``(alpha, beta)`` at the grid minimum.
    """
    device, dtype = z.device, z.dtype

    def _search(p_lo: float, p_hi: float, a_lo: float, a_hi: float, n: int):
        # persistence spaced log-uniformly in its distance from 1
        gap = torch.logspace(
            math.log10(max(1.0 - p_hi, 1e-5)),
            math.log10(max(1.0 - p_lo, 1e-4)),
            n, device=device, dtype=dtype,
        )
        p_grid = 1.0 - gap
        a_grid = torch.logspace(
            math.log10(a_lo), math.log10(a_hi), n, device=device, dtype=dtype,
        )
        pp, aa = torch.meshgrid(p_grid, a_grid, indexing="ij")
        pp, aa = pp.reshape(-1), aa.reshape(-1)
        # alpha must leave room for a non-negative beta.
        ok = aa < pp
        pp, aa = pp[ok], aa[ok]
        if pp.numel() == 0:
            return None
        nll = garch11_nll(z, mask, aa, pp - aa)
        best = int(torch.argmin(nll))
        return float(aa[best]), float(pp[best] - aa[best])

    coarse = _search(0.50, 0.99999, 1e-3, 0.45, n_coarse)
    if coarse is None:
        return 0.05, 0.90
    a0, b0 = coarse
    p0 = a0 + b0

    # Refine in a neighbourhood of the coarse optimum.
    fine = _search(
        max(0.10, 1.0 - (1.0 - p0) * 4.0),
        min(0.999999, 1.0 - (1.0 - p0) / 4.0),
        max(1e-4, a0 / 3.0),
        min(0.60, a0 * 3.0),
        n_fine,
    )
    return fine if fine is not None else coarse


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


# Correlation sufficient statistics: [n, Sx, Sy, Sxy, Sxx, Syy].
_CORR_STATS = 6


class BaselineFitter:
    """Accumulates streamed buckets into ``(month, agg_factor)`` cells and fits.

    Two estimators run per cell:

    * **GARCH(1,1)** — its likelihood is a recursion over each return series, so
      series are retained (up to ``max_series_per_cell``) and the MLE runs at
      ``fit()``. The cap bounds memory; a few hundred thousand observations pin
      ``(alpha, beta)`` far tighter than the month-to-month variation.
    * **Per-horizon AR(1)** — fitted as a *predictive regression*: each sample
      contributes one ``(trailing-h predictor, next-h target)`` pair, and the
      coefficient is their pooled correlation. This needs only streamed moments
      (no retention) and, crucially, stays well-populated even at 120 min, where
      a single window holds too little to form within-window AR lags.

    The calibration dataset must therefore expose targets for ``return`` and
    ``spread_change`` at ``horizons`` (see ``scripts/fit_finance_baselines.py``).

    Usage::

        fitter = BaselineFitter(horizons=[300, 600, 900, 1800, 3600, 7200])
        for batch in loader:
            for bucket in batch["buckets"]:
                fitter.add_bucket(bucket)
        params = fitter.fit()
        params.save("garch_ar_monthly.json")
    """

    def __init__(self, horizons, max_series_per_cell: int = 512):
        from stable_finance.dataset import get_target_names

        self.horizons = [int(h) for h in horizons]
        self.max_series_per_cell = max_series_per_cell
        # Target column layout the calibration dataset must produce.
        names = get_target_names(self.horizons, ["return", "spread_change"])
        self._ret_col = {h: names.index(f"return_{h:03d}") for h in self.horizons}
        self._spr_col = {h: names.index(f"spread_change_{h:03d}") for h in self.horizons}
        self._n_target_cols = len(names)

        # (month, agg) -> retained GARCH series / counts
        self._series: dict[tuple[str, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._n_series: dict[tuple[str, int], int] = {}
        self._n_steps: dict[tuple[str, int], int] = {}
        # (month, agg) -> {"ret"|"spr": {h: [n, Sx, Sy, Sxy, Sxx, Syy]}}
        self._corr: dict[tuple[str, int], dict[str, dict[int, list[float]]]] = {}

    def add_bucket(self, bucket: dict) -> None:
        """Accumulate one collated bucket. Silently skips buckets lacking meta."""
        if "dates" not in bucket or "agg_factors" not in bucket:
            return
        if not bucket.get("views") or "targets" not in bucket:
            return

        view = bucket["views"][0]
        lengths = bucket["lengths"][0]
        targets = bucket["targets"]
        dates = bucket["dates"]
        aggs = bucket["agg_factors"].tolist()

        by_cell: dict[tuple[str, int], list[int]] = {}
        for i, (d, a) in enumerate(zip(dates, aggs)):
            month = _month_of(d)
            if not month:
                continue
            by_cell.setdefault((month, int(a)), []).append(i)

        for cell, idxs in by_cell.items():
            sel = torch.tensor(idxs, dtype=torch.long)
            s = extract_series(view[sel], lengths[sel])
            self._accumulate_garch(cell, s)
            self._accumulate_ar(cell, s, targets[sel])

    def _accumulate_garch(self, cell: tuple[str, int], s: StandardizedSeries) -> None:
        self._n_series[cell] = self._n_series.get(cell, 0) + int(s.z.shape[0])
        self._n_steps[cell] = self._n_steps.get(cell, 0) + int(s.z_mask.sum())
        held = self._series.setdefault(cell, [])
        room = self.max_series_per_cell - sum(t.shape[0] for t, _ in held)
        if room > 0:
            held.append((s.z[:room].cpu(), s.z_mask[:room].cpu()))

    def _accumulate_ar(
        self, cell: tuple[str, int], s: StandardizedSeries, tgt: torch.Tensor,
    ) -> None:
        agg = float(cell[1])
        cell_stats = self._corr.setdefault(
            cell, {"ret": {}, "spr": {}},
        )
        for h in self.horizons:
            n = torch.full((s.z.shape[0],), steps_for_horizon(h, agg), dtype=torch.long)
            x_ret = trailing_return_predictor(s, n)
            x_spr = trailing_spread_change(s.w, s.w_mask, n)
            y_ret = tgt[:, self._ret_col[h]].double()
            y_spr = tgt[:, self._spr_col[h]].double()
            _accum_corr(cell_stats["ret"].setdefault(h, [0.0] * _CORR_STATS), x_ret, y_ret)
            _accum_corr(cell_stats["spr"].setdefault(h, [0.0] * _CORR_STATS), x_spr, y_spr)

    def fit(self) -> MonthlyBaselineParams:
        """Fit every populated cell, plus a pooled global fallback."""
        cells: dict[tuple[str, int], BaselineParams] = {}
        for cell in sorted(self._series):
            cells[cell] = self._fit_cell(cell)

        # Global fallback: subsample the GARCH pool (one chunk per cell — the
        # convergence study showed (alpha, beta) pin down long before the full
        # hundreds-of-thousands pool, whose recursion would dominate fit()), and
        # pool the AR correlation moments across every cell per horizon.
        pooled = [parts[0] for parts in self._series.values() if parts]
        alpha, beta = self._fit_garch(pooled)
        fallback = BaselineParams(
            alpha=alpha,
            beta=beta,
            phi_return_by_h=self._pooled_phi("ret"),
            phi_spread_by_h=self._pooled_phi("spr"),
            n_series=sum(self._n_series.values()),
            n_steps=sum(self._n_steps.values()),
        )
        return MonthlyBaselineParams(cells=cells, fallback=fallback)

    def _fit_cell(self, cell: tuple[str, int]) -> BaselineParams:
        alpha, beta = self._fit_garch(self._series.get(cell, []))
        stats = self._corr.get(cell, {"ret": {}, "spr": {}})
        return BaselineParams(
            alpha=alpha,
            beta=beta,
            phi_return_by_h={h: corr_from_stats(*v) for h, v in stats["ret"].items()},
            phi_spread_by_h={h: corr_from_stats(*v) for h, v in stats["spr"].items()},
            n_series=self._n_series[cell],
            n_steps=self._n_steps[cell],
        )

    def _pooled_phi(self, kind: str) -> dict[int, float]:
        pooled: dict[int, list[float]] = {h: [0.0] * _CORR_STATS for h in self.horizons}
        for stats in self._corr.values():
            for h, v in stats[kind].items():
                for i in range(_CORR_STATS):
                    pooled[h][i] += v[i]
        return {h: corr_from_stats(*v) for h, v in pooled.items() if v[0] >= 2}

    @staticmethod
    def _fit_garch(
        chunks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float]:
        if not chunks:
            return 0.05, 0.90
        width = max(z.shape[1] for z, _ in chunks)
        zs, ms = [], []
        for z, m in chunks:
            pad = width - z.shape[1]
            if pad:
                z = torch.nn.functional.pad(z, (0, pad))
                m = torch.nn.functional.pad(m, (0, pad), value=False)
            zs.append(z)
            ms.append(m)
        return fit_garch11(torch.cat(zs), torch.cat(ms))


def _accum_corr(acc: list[float], x: torch.Tensor, y: torch.Tensor) -> None:
    """Add finite ``(x, y)`` pairs into a ``[n, Sx, Sy, Sxy, Sxx, Syy]`` accumulator."""
    ok = torch.isfinite(x) & torch.isfinite(y)
    if not bool(ok.any()):
        return
    x, y = x[ok], y[ok]
    acc[0] += float(x.numel())
    acc[1] += float(x.sum())
    acc[2] += float(y.sum())
    acc[3] += float((x * y).sum())
    acc[4] += float((x * x).sum())
    acc[5] += float((y * y).sum())
