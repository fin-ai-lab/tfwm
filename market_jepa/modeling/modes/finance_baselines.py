"""Classical finance baselines as a non-parametric training mode.

The "did the learned representation beat textbook finance?" reference point.
Three standard models, one per target family, each producing an actual forecast
of the quantity the probe is scored on:

  ===================  ====================  ==================================
  probe target         model                 forecast
  ===================  ====================  ==================================
  ``return_h``         AR(1) on returns      cumulative return over ``h``
  ``volatility_``      GARCH(1,1)            forward vol over ``h`` minus
  ``change_h``                               trailing vol over ``h``
  ``spread_change_h``  AR(1) on spread       mean-reverting move over ``h``
  ===================  ====================  ==================================

The volatility and spread targets are **changes**, not levels
(:mod:`stable_finance.dataset`: ``spread_change`` is ``spr(t+h) - spr(t)`` and
``volatility_change`` is forward vol minus backward vol), so both baselines
forecast a change. For volatility that is precisely what GARCH(1,1) is for: its
``h``-step-ahead variance forecast reverts toward the long-run level at rate
``alpha + beta``, so the predicted change is large and negative after a burst of
volatility and positive after a lull. A model without mean reversion — a
RiskMetrics EWMA, say — forecasts no change at all and is useless here.

Design — scored by the probe, not by a head
--------------------------------------------
A finance baseline has nothing to learn by gradient descent. Instead of inventing
a bespoke scorer, :meth:`FinanceBaseline.encode` emits the model forecasts as an
"embedding" and lets the existing probe-eval pipeline
(``eval/probe_eval_worker.py``) fit a ridge probe on them and report
``probe/ridge_ic_<target>_<horizon>`` — the identical metric, in the identical
namespace, used for every learned representation. A baseline run therefore drops
straight onto the same axis as a LeJEPA/I-JEPA run with no plotting changes.

Note the probe sees the *whole* forecast bank when scoring any one target, just
as it sees a whole 768-d embedding for a learned encoder. It is "probe on
classical forecasts" vs "probe on a learned embedding" — the same protocol on
both sides, which is what makes the comparison fair.

Parameters are fitted offline, per month
-----------------------------------------
Model *state* is per-window, but the coefficients ``(alpha, beta, phi_return,
phi_spread)`` are pooled across assets and refit per calendar month by
``scripts/fit_finance_baselines.py``, which writes a JSON artifact loaded here.
See :mod:`.finance_baseline_params` for why they are additionally keyed by
``agg_factor``, and for the per-window standardization that makes them poolable.
Without an artifact the mode falls back to conventional textbook values, which
is fine for a smoke test and wrong for a headline number.
"""

from __future__ import annotations

import json
import logging
import os

import torch

from .base import TrainingModel
from .finance_baseline_params import (
    BaselineParams,
    MonthlyBaselineParams,
    _month_of,
    extract_series,
    steps_for_horizon,
    trailing_return_predictor,
    trailing_spread_change,
)

logger = logging.getLogger(__name__)

# Matches plots/metrics/metrics.py HORIZONS — 5/10/15/30/60/120 minutes.
DEFAULT_HORIZONS = [300, 600, 900, 1800, 3600, 7200]

# Used when no fitted artifact is supplied. Conventional daily-equity GARCH
# values; the AR coefficients default to 0 (no signal) so an uncalibrated run
# emits flat return/spread forecasts rather than a wrong sign.
_FALLBACK = BaselineParams(
    alpha=0.05, beta=0.90, phi_return_by_h={}, phi_spread_by_h={},
)

_EPS = 1e-12

_FAMILIES = ("return", "volatility", "spread")


class FinanceBaseline(TrainingModel):
    """Textbook AR(1)/GARCH(1,1) forecasts exposed as a probe-scored bank.

    Trains nothing: ``training_step`` is a no-op and the recommended invocation
    sets ``num_epochs=0`` (see ``FinanceBaselineModeConfig``), so the only work
    is the initial/final probe eval over :meth:`encode`'s forecasts.

    The ``backbone`` argument is required by the training harness (it always
    instantiates ``cfg.backbone`` and passes it) but is **unused** — it is stored
    only to satisfy the :class:`TrainingModel` interface and to give the
    optimizer a non-empty parameter list. With ``num_epochs=0`` it is never
    touched.

    Args:
        backbone: Unused; stored for interface compatibility (may be ``None``
            when reconstructed via :meth:`from_pretrained`).
        features: Which baseline families to emit, any subset of
            ``{"return", "volatility", "spread"}``.
        horizons: Forecast horizons in seconds. Must match the probe targets'
            horizons for the numbers to line up.
        params_path: Path to the JSON written by
            ``scripts/fit_finance_baselines.py``. ``None`` uses ``_FALLBACK``
            everywhere and logs a warning.
        params_as_of: Cap parameter resolution at this month (``"YYYY-MM"``,
            inclusive). Set it to the *train* month whenever the artifact
            covers the eval month, so eval samples fall through to the
            nearest earlier month instead of resolving to parameters fitted
            on their own month's future returns. ``None`` (default) keeps
            the uncapped lookup and logs a warning when a fitted table is
            loaded.
    """

    mode_label: str = "FinBaseline"
    uses_multi_view: bool = False
    # Opt into the metadata channel in collect_probe_data: the forecasts need
    # each sample's date (to select that month's parameters) and agg_factor (to
    # convert a horizon in seconds into a number of steps).
    wants_metadata: bool = True

    def __init__(
        self,
        backbone,
        features: list[str] | None = None,
        horizons: list[int] | None = None,
        params_path: str | None = None,
        params_as_of: str | None = None,
    ):
        super().__init__()
        if features is None:
            features = list(_FAMILIES)
        unknown = [f for f in features if f not in _FAMILIES]
        if unknown:
            raise ValueError(
                f"Unknown finance-baseline families {unknown}; valid: {list(_FAMILIES)}"
            )
        if not features:
            raise ValueError("FinanceBaseline needs at least one feature family")

        self.backbone = backbone
        self.families: list[str] = [f for f in _FAMILIES if f in features]
        self.horizons: list[int] = list(horizons) if horizons else list(DEFAULT_HORIZONS)
        self.params_path = params_path
        self.params_as_of = params_as_of

        if params_path:
            self.params = MonthlyBaselineParams.load(params_path)
            logger.info("Loaded finance-baseline parameters: %r", self.params)
            if params_as_of is None:
                logger.warning(
                    "FinanceBaseline has no params_as_of; samples resolve to "
                    "their own month's cells first, so eval-month forecasts "
                    "are in-sample wherever the artifact covers the eval "
                    "month. Set mode.params_as_of to the train month for "
                    "reported numbers."
                )
        else:
            self.params = MonthlyBaselineParams(cells={}, fallback=_FALLBACK)
            logger.warning(
                "FinanceBaseline has no params_path; using uncalibrated defaults "
                "(alpha=%.3f beta=%.3f). Fit with scripts/fit_finance_baselines.py "
                "before reporting numbers.",
                _FALLBACK.alpha, _FALLBACK.beta,
            )

        self.feature_names: list[str] = [
            f"{prefix}_{h}"
            for fam in self.families
            for prefix in _FORECAST_PREFIX[fam]
            for h in self.horizons
        ]
        self.d_embedding = len(self.feature_names)

    @property
    def mode_str(self) -> str:
        return f"finance_baseline: {'+'.join(self.families)}"

    # ------------------------------------------------------------------
    # Parameter resolution
    # ------------------------------------------------------------------

    def _param_rows(
        self, dates: list[str] | None, aggs: torch.Tensor | None, B: int,
    ) -> list[BaselineParams]:
        """Per-sample fitted parameters, one :class:`BaselineParams` per row.

        Looks each sample's ``(month, agg_factor)`` cell up in the fitted table.
        Missing metadata degrades to the global fallback rather than failing, so
        a unit test or an ad-hoc forward pass still runs.
        """
        if dates is None or aggs is None:
            return [self.params.fallback] * B
        return [
            self.params.lookup(_month_of(d), int(a), as_of=self.params_as_of)
            for d, a in zip(dates, aggs.tolist())
        ]

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    @staticmethod
    def _garch_filter(
        z: torch.Tensor,
        mask: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """One-step-ahead conditional variance at each window's last valid step.

        Runs ``h_{t+1} = omega + alpha*z_t^2 + beta*h_t`` from the unconditional
        variance ``h_0 = 1``, freezing the state past each window's valid length
        so right-padding cannot contaminate the final state. Returns ``(B,)``.
        """
        omega = 1.0 - alpha - beta
        h = torch.ones_like(alpha)
        for t in range(z.shape[1]):
            zt = z[:, t]
            h = torch.where(mask[:, t], omega + alpha * zt * zt + beta * h, h)
        return h

    @staticmethod
    def _mean_reversion_factor(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """``(1 - p^n) / (n * (1 - p))`` — the average of ``p^k`` for k in [0,n).

        This is how much of today's deviation from the long-run level survives
        on average across the next ``n`` steps. Guarded at ``p -> 1``, where the
        limit is 1 (a unit-root process forecasts no reversion at all).
        """
        gap = 1.0 - p
        near_unit = gap.abs() < 1e-9
        safe_gap = torch.where(near_unit, torch.ones_like(gap), gap)
        factor = (1.0 - _pow_n(p, n)) / (n * safe_gap)
        return torch.where(near_unit, torch.ones_like(factor), factor)

    @staticmethod
    def _trailing_std(
        s: torch.Tensor, mask: torch.Tensor, n: torch.Tensor,
    ) -> torch.Tensor:
        """Std of the last ``n`` valid steps of each row. ``n`` is ``(B,)``."""
        T = s.shape[1]
        idx = torch.arange(T, device=s.device)[None, :]
        valid_len = mask.sum(1, keepdim=True)
        window = (idx >= (valid_len - n[:, None]).clamp(min=0)) & mask
        denom = window.sum(1).clamp(min=1).to(s.dtype)
        mean = (s * window).sum(1) / denom
        ex2 = (s * s * window).sum(1) / denom
        return (ex2 - mean * mean).clamp(min=0).sqrt()

    def _compute_forecasts(
        self,
        view: torch.Tensor,
        lengths: torch.Tensor | None,
        dates: list[str] | None,
        aggs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Model forecasts for one view. Returns ``(B, d)``, finite, float64."""
        B, _C, T = view.shape
        device = view.device
        if T < 3:
            return torch.zeros(B, self.d_embedding, dtype=torch.float64, device=device)

        s = extract_series(view, lengths)
        rows = self._param_rows(dates, aggs, B)

        # Horizon in seconds -> steps, using each sample's own aggregation scale.
        if aggs is None:
            agg = torch.ones(B, dtype=torch.float64, device=device)
        else:
            agg = aggs.to(device).double().clamp(min=1.0)

        def phi_col(getter, h: int) -> torch.Tensor:
            return torch.tensor(
                [getter(p, h) for p in rows], dtype=torch.float64, device=device,
            )

        cols: list[torch.Tensor] = []

        if "return" in self.families:
            # Horizon-matched AR(1): predict the next-h return from the trailing-h
            # return, scaled by the fitted per-horizon coefficient. The trailing
            # window equals the horizon (5->5, 120->120), mirroring the vol term.
            for h in self.horizons:
                n = _steps(h, agg).long()
                x = trailing_return_predictor(s, n)
                cols.append(phi_col(BaselineParams.phi_return, h) * x)

        if "volatility" in self.families:
            h_next = self._garch_filter(
                s.z, s.z_mask, phi_col(_alpha, 0), phi_col(_beta, 0),
            )
            p = phi_col(_alpha, 0) + phi_col(_beta, 0)
            for h in self.horizons:
                n = _steps(h, agg)
                # Mean forward variance over the next n steps, reverting to the
                # unconditional variance of 1.
                fwd_var = 1.0 + (h_next - 1.0) * self._mean_reversion_factor(p, n)
                bwd_vol = self._trailing_std(s.z, s.z_mask, n)
                cols.append(fwd_var.clamp(min=0).sqrt() - bwd_vol)

        if "spread" in self.families:
            # Horizon-matched AR(1) on the spread *change*: predict the next-h
            # change from the trailing-h change, scaled by the fitted coefficient.
            for h in self.horizons:
                n = _steps(h, agg).long()
                x = trailing_spread_change(s.w, s.w_mask, n)
                cols.append(phi_col(BaselineParams.phi_spread, h) * x)

        out = torch.stack(cols, dim=1)
        # The probe's LogisticRegression rejects NaN/inf; clamp + scrub so one
        # degenerate window can never poison the whole probe fit.
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.clamp(min=-1e4, max=1e4)

    # ------------------------------------------------------------------
    # TrainingModel interface
    # ------------------------------------------------------------------

    def encode(self, x, lengths=None, metadata=None) -> dict[str, torch.Tensor]:
        """Return ``{"embeddings": (B, n_views, d)}`` of model forecasts.

        Mirrors :meth:`TrainingModel.encode`'s input handling so the probe's
        ``collect_probe_data`` (which passes a list of views) works unchanged.

        Args:
            x: ``(B, n_views, C, T)`` tensor, a list of ``(B, C, T)`` views, or a
                single ``(B, C, T)`` view.
            lengths: Matching valid lengths, or ``None``.
            metadata: The collated bucket dict, supplying ``dates`` and
                ``agg_factors``. ``None`` falls back to global parameters and
                1 second per token — correct only for un-aggregated views.
        """
        self._validate_instance_attrs()
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            views = [x[:, i, :, :] for i in range(x.shape[1])]
            view_lengths = (
                [lengths] * x.shape[1] if lengths is not None else [None] * x.shape[1]
            )
        elif isinstance(x, list):
            views = x
            view_lengths = lengths if lengths is not None else [None] * len(views)
        else:
            views = [x]
            view_lengths = [lengths]

        dates = metadata.get("dates") if metadata else None
        aggs = metadata.get("agg_factors") if metadata else None

        feats = [
            self._compute_forecasts(v, vl, dates, aggs)
            for v, vl in zip(views, view_lengths)
        ]
        return {"embeddings": torch.stack(feats, dim=1)}

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the ``(B, d)`` forecast bank for a single view (interface req.)."""
        return self._compute_forecasts(x, lengths, None, None)

    def training_step(self, batch, device, grad_accum_steps: int = 1):
        # Nothing to train. Returning None makes the harness advance the step
        # counter without an optimizer step (see pretrain.py training loop).
        return None

    @torch.no_grad()
    def eval_step(self, eval_batches, device) -> dict[str, float]:
        """Cheap sanity diagnostic — the real numbers come from the probe.

        Reports the cross-sample std of each forecast (averaged), so a collapsed
        or constant bank is visible at a glance.
        """
        self.eval()
        collected: list[torch.Tensor] = []
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                if bucket.get("n_global_views", 0) < 1:
                    continue
                v = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                collected.append(
                    self._compute_forecasts(
                        v, lengths, bucket.get("dates"), bucket.get("agg_factors"),
                    ).cpu()
                )
        if not collected:
            return {"eval/baseline_feature_std": float("nan")}
        feats = torch.cat(collected, dim=0)
        return {
            "eval/baseline_feature_std": float(feats.std(dim=0).mean()),
            "eval/baseline_n": float(feats.shape[0]),
        }

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(self)
        fitted = (
            f"{len(self.params)} monthly cells from {self.params_path}"
            if self.params_path
            else "UNCALIBRATED defaults"
        )
        summary = (
            f"Finance baseline (no trainable objective):\n"
            f"  Families: {', '.join(self.families)}\n"
            f"  Horizons: {', '.join(str(h) for h in self.horizons)} sec\n"
            f"  Parameters: {fitted}\n"
            f"  Forecasts ({self.d_embedding}): {', '.join(self.feature_names)}\n"
            f"  Scored by the probe -> probe/ridge_ic_<target>_<horizon>"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        return "__".join(
            ["mode=finance_baseline", f"feats={'+'.join(self.families)}"]
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        """Persist the baseline config and a copy of the fitted parameters.

        The parameter table is copied in rather than referenced by path, so a
        saved baseline reproduces its own numbers without the calibration
        artifact still being on disk.
        """
        os.makedirs(path, exist_ok=True)
        self.params.save(os.path.join(path, "baseline_params.json"))
        config = {
            "class": "FinanceBaseline",
            "features": self.families,
            "horizons": self.horizons,
            "feature_names": self.feature_names,
            "params_as_of": self.params_as_of,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "FinanceBaseline":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        params_file = os.path.join(path, "baseline_params.json")
        # The backbone is unused; reconstruct the forecaster directly.
        return cls(
            backbone=kwargs.get("backbone"),
            features=config["features"],
            horizons=config.get("horizons"),
            params_path=params_file if os.path.exists(params_file) else None,
            params_as_of=config.get("params_as_of"),
        )


# Forecast-name prefixes per family, in fixed output order.
_FORECAST_PREFIX: dict[str, tuple[str, ...]] = {
    "return": ("ar1_return",),
    "volatility": ("garch_vol_change",),
    "spread": ("ar1_spread_change",),
}


def _steps(horizon_sec: int, agg: torch.Tensor) -> torch.Tensor:
    """Horizon in seconds -> steps, per sample. At least one step."""
    return (horizon_sec / agg).round().clamp(min=1.0)


def _pow_n(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """``p ** n`` for real ``p`` and whole-number ``n`` held as a float tensor.

    ``torch.pow`` returns NaN for a negative base with a floating exponent, so
    split into magnitude and parity (exact for integral ``n``). Used by the GARCH
    mean-reversion factor, where the persistence base is positive but the helper
    stays robust regardless.
    """
    magnitude = p.abs().clamp(max=1.0) ** n
    odd = torch.remainder(n, 2.0) == 1.0
    return torch.where((p < 0) & odd, -magnitude, magnitude)


def _alpha(p: BaselineParams, _h: int) -> float:
    """GARCH alpha getter (horizon-independent); shaped like the phi getters."""
    return p.alpha


def _beta(p: BaselineParams, _h: int) -> float:
    return p.beta
