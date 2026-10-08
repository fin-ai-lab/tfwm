"""Classical forecasting baselines, fitted on the synchronized anchor panel.

Every predictor these models read is a function of the VIEW the encoder is fed
(see ``panel_tables``), so a classical model here has exactly the encoder's
information. Every model fits against ``Z`` — the cross-sectional z-score that
the reported rank IC is computed on — and emits one score per row. Only the
WITHIN-CELL ORDERING of that score is ever used, which is what makes the
protocol so much lighter than the retired AUC one: rank IC is invariant to any
monotone transform, so a forecast never has to be calibrated into label units,
there is no discretizer, and there is no probe.

Contract: ``fit(train_table)`` then ``predict(table, target_name)`` returning a
score array, or ``None`` for targets the model does not address (HAR-RV and
GARCH only speak to volatility; mean-reversion only to the two change targets).

  * :class:`MeanReversion` — the change regressed on its own current level.
  * :class:`ARp`           — AR(p) on the horizon-matched series, p by CV.
  * :class:`ARMApq`        — ARMA(p,q) via Hannan-Rissanen, (p,q) by CV.
  * :class:`HARRV`         — heterogeneous autoregressive realized volatility.
  * :class:`GARCH11`       — variance-targeted GARCH(1,1), persistence by fit.
  * :class:`RidgeARDL`     — single-equation ARDL on all backbone channels.

RETIRED WITH THE AUC METRIC, and not for tidiness — under a cross-sectional
rank metric they are undefined, not merely uninteresting:

  * ``NaiveZero``      constant within a cell, so every row ties and the rank
                       correlation has no value at all. Under AUC it scored
                       exactly 0.500 and served as the floor; here the floor is
                       the zero line, drawn rather than estimated.
  * ``TimeOfDay``      likewise constant within a cell (one anchor is one
                       instant). It existed only to expose the close-clamp
                       artifact, and ``dataset.outcomes.anchor_targets`` emits NaN
                       instead of clamping, so the artifact is gone too.
  * ``PriorDaySpread`` yesterday's spread at the same clock time is not inside
                       a 2048-token window. Under the view-only rule it is no
                       more available to a baseline than to the encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from market_jepa.eval.metrics import grouped_rank_ic
from panel_tables import (
    EWMA_DECAYS, FEATURE_LAGS, HAR_WINDOWS, INFO_NAMES, N_LAGS,
)

_MAX_P = 3
_MAX_Q = 2
_N_FOLDS = 5

# Minimum usable training rows before a candidate is considered. Deep lags at
# long horizons reach past the session open and go NaN, leaving a handful of rows
# whose OLS fit extrapolates wildly on the eval month; CV alone does not catch it
# because the folds are equally small.
_MIN_TRAIN_ROWS = 200


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares with an intercept prepended. Returns coefficients."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _apply(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def _clean(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[m], y[m]


def cv_mse(X: np.ndarray, y: np.ndarray, n_folds: int = _N_FOLDS) -> float:
    """K-fold CV mean squared error of an OLS fit. ``inf`` if unfittable."""
    X, y = _clean(X, y)
    if len(y) < max(_MIN_TRAIN_ROWS, n_folds * 5) or X.shape[1] == 0:
        return float("inf")
    idx = np.arange(len(y))
    folds = np.array_split(idx, n_folds)
    errs = []
    for f in folds:
        tr = np.setdiff1d(idx, f, assume_unique=True)
        if len(tr) < X.shape[1] + 2:
            continue
        try:
            coef = _ols(X[tr], y[tr])
        except np.linalg.LinAlgError:
            return float("inf")
        errs.append(float(np.mean((_apply(coef, X[f]) - y[f]) ** 2)))
    return float(np.mean(errs)) if errs else float("inf")


# A candidate must survive on at least this share of the rows the target
# itself has. See :func:`coverage` for why a deep order would otherwise buy
# accuracy by quietly changing which cross-sections it is scored on.
MIN_COVERAGE = 0.6


def coverage(X: np.ndarray, cols, y: np.ndarray) -> float:
    """Share of the target's usable rows on which ``cols`` are all finite.

    THIS IS A COMPARABILITY GUARD, not a numerical one. A 2048-token view
    spans ``2048 * agg`` seconds, so the deep lags of a long horizon exist
    only at the coarser resolutions — ``rv_h7200_lag3`` is NaN at every agg
    below 11. A model free to pick that lag would fit and then predict on the
    agg-11 cells ALONE, and ``grouped_rank_ic`` would happily score it there:
    a higher number on a different, smaller panel than every other series in
    the figure. Requiring the panel back is cheaper than explaining it.
    """
    n_y = int(np.isfinite(y).sum())
    if n_y == 0:
        return 0.0
    ok = np.isfinite(y) & np.all(np.isfinite(X[:, cols]), axis=1)
    return float(ok.sum()) / n_y


def usable_cols(X: np.ndarray, cols, y: np.ndarray,
                min_coverage: float = MIN_COVERAGE) -> list[int]:
    """Drop the individual columns that would shrink the panel."""
    return [c for c in cols if coverage(X, [c], y) >= min_coverage]


def select_by_cv(
    X_full: np.ndarray, y: np.ndarray, candidates: dict,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[object, np.ndarray | None]:
    """Pick the candidate column-set with the lowest CV error, then refit on all.

    ``candidates`` maps a label (e.g. ``p`` or ``(p, q)``) to column indices.
    Candidates that do not cover ``min_coverage`` of the target's rows are not
    considered at all. Returns ``(best_label, coefficients)``.
    """
    best_key, best_err = None, float("inf")
    for key, cols in candidates.items():
        if len(cols) == 0 or coverage(X_full, cols, y) < min_coverage:
            continue
        err = cv_mse(X_full[:, cols], y)
        if err < best_err:
            best_key, best_err = key, err
    if best_key is None:
        return None, None
    Xc, yc = _clean(X_full[:, candidates[best_key]], y)
    if len(yc) < _MIN_TRAIN_ROWS:
        return None, None
    return best_key, _ols(Xc, yc)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    name: str
    label: str
    color: str
    # Dashed marks a reference that is not a market model (the clock).
    linestyle: str = "-"

    # Per-model fitted state, cleared before every fit. The scorer reuses one
    # model instance across months, and every ``fit`` here writes into dicts
    # keyed by target; without an explicit reset a target that fails to fit on
    # month M would silently keep month M-1's coefficients and be scored with
    # them. Subclasses list their state attributes here.
    _state_attrs: tuple[str, ...] = ()

    def _reset(self) -> None:
        for attr in self._state_attrs:
            getattr(self, attr).clear()

    def fit(self, table: dict) -> None:                     # pragma: no cover
        raise NotImplementedError

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        raise NotImplementedError                            # pragma: no cover

    # -- shared plumbing -------------------------------------------------
    @staticmethod
    def _pidx(table: dict, name: str) -> int:
        return table["predictor_names"].index(name)

    @staticmethod
    def _tidx(table: dict, name: str) -> int:
        return table["target_names"].index(name)

    @staticmethod
    def _parse(target: str) -> tuple[str, int]:
        """``"volatility_change_1800"`` -> ``("volatility_change", 1800)``."""
        m = re.match(r"^(.*)_(\d+)$", target)
        return m.group(1), int(m.group(2))


class MeanReversion(Baseline):
    """Regress the CHANGE on the current LEVEL of the same statistic.

    Both change targets subtract a quantity that sits inside the observation
    window:

        volatility_change = fwd_vol[t, t+h) - bwd_vol[t-h, t)
        spread_change     = spread(t+h)     - spread(t)

    so ``-bwd_vol`` and ``-spread(t)`` forecast them with no market model at
    all. Read off THE VIEW, over the 13 standard months, that is rank IC
    0.32 -> 0.02 across the horizons for volatility and a flat ~0.18 for
    spread (plots/metrics/mechanical_baseline.py). On the raw book both are
    far higher -- spread reaches ~0.9 -- and the gap is exactly what
    ``normalize_numpy`` takes away from every model on this panel, encoder
    included. A reader is entitled to know how much of a change target is
    just its own level coming back, and no other baseline answers that
    cleanly:
    ARp and ARMApq regress on lagged CHANGES, which cannot express a level, and
    the level only enters Ridge ARDL buried among 40+ ridge regressors.

    The coefficient is FITTED, not pinned at -1. Full reversion to a common
    mean implies -1; the cross-sectional regression says about -0.81, and a
    baseline that overstates reversion would understate itself.

    ``return`` is skipped deliberately. It is a ratio with the cross-section
    z-scored out and carries no analogous level term — which is exactly why it
    is the one target where a trained encoder's lead does not close with the
    horizon.
    """

    coefs: dict = field(default_factory=dict)

    def __init__(self):
        super().__init__("mean_reversion", "Mean reversion", "#17becf")
        self.coefs = {}
        self._state_attrs = ("coefs",)

    def _col(self, table: dict, kind: str, h: int) -> int | None:
        """The level this target is a change IN."""
        if kind == "spread_change":
            return self._pidx(table, "spread_level")
        if kind == "volatility_change":
            # rv_h{h}_lag1 IS bwd_vol: the trailing realized vol over the
            # h-window ending at t, the exact term the target subtracts.
            cols = _lag_cols(table, kind, h, 1)
            return cols[0] if cols else None
        return None

    def fit(self, table: dict) -> None:
        self._reset()
        X, Y = table["X"], table["Y"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            col = self._col(table, kind, h)
            if col is None:
                continue
            Xc, yc = _clean(X[:, [col]], Y[:, self._tidx(table, target)])
            if len(yc) < 10:
                continue
            # With an intercept, so the fit is "reversion toward the sample
            # mean" rather than toward zero.
            self.coefs[target] = _ols(np.column_stack([Xc, np.ones(len(Xc))]), yc)

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.coefs:
            return None
        kind, h = self._parse(target)
        col = self._col(table, kind, h)
        x = table["X"][:, [col]]
        return _apply(self.coefs[target], np.column_stack([x, np.ones(len(x))]))


# ---------------------------------------------------------------------------
# AR / ARMA
# ---------------------------------------------------------------------------


def _info_cols(table: dict) -> list[int]:
    """Indices of the eleven information-token predictors, or [] if absent.

    THESE ARE EXOGENOUS, not lags. The encoder receives the per-view
    (mu, sigma) and the three window descriptors in a token of their own, so a
    classical arm that never sees them is being compared on strictly less
    information. Appending them to a model's design matrix is what makes an
    AR(p) an ARX(p) -- the same objective, the same lag structure, plus the
    conditioning the encoder already had.

    Returns [] for a table built before the columns existed, so an old cached
    panel degrades to the previous behaviour instead of raising.
    """
    names = list(table["predictor_names"])
    return [names.index(n) for n in INFO_NAMES if n in names]


def _lag_cols(table: dict, kind: str, h: int, n: int) -> list[int]:
    """Column indices of the first ``n`` horizon-matched lags for a statistic."""
    stem = {
        "return": f"ret_h{h}_lag",
        "spread_change": f"dspr_h{h}_lag",
        "volatility_change": f"rv_h{h}_lag",
    }[kind]
    names = table["predictor_names"]
    out = []
    for lag in range(1, n + 1):
        col = f"{stem}{lag}"
        if col in names:
            out.append(names.index(col))
    return out


@dataclass
class ARp(Baseline):
    """AR(p) on the horizon-matched series, ``p`` chosen by K-fold CV.

    For horizon ``h`` the regressors are the trailing ``h``-returns (or
    ``h``-spread-changes, or trailing realized vols) at lags 1..p, so the model
    operates at the frequency it forecasts — an AR on 15-minute returns, not a
    tick-scale AR iterated 300 steps.
    """

    order: dict = field(default_factory=dict)
    coefs: dict = field(default_factory=dict)

    def __init__(self, max_p: int = _MAX_P):
        super().__init__("ar_p", "AR(p)", "#6a3d9a")
        self.max_p = max_p
        self.order, self.coefs = {}, {}
        self._state_attrs = ("order", "coefs")

    def fit(self, table: dict) -> None:
        self._reset()
        X, Y = table["X"], table["Y"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            y = Y[:, self._tidx(table, target)]
            # ARX: the lag structure is unchanged and the eleven ride along as
            # exogenous regressors. The length check still counts LAGS only, so
            # a p is rejected for missing a lag, never for missing info.
            info = _info_cols(table)
            cands = {
                p: _lag_cols(table, kind, h, p)
                for p in range(1, self.max_p + 1)
            }
            cands = {p: c for p, c in cands.items() if len(c) == p}
            cands = {p: c + info for p, c in cands.items()}
            p, coef = select_by_cv(X, y, cands)
            if coef is not None:
                self.order[target], self.coefs[target] = p, coef

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.coefs:
            return None
        kind, h = self._parse(target)
        cols = _lag_cols(table, kind, h, self.order[target]) + _info_cols(table)
        return _apply(self.coefs[target], table["X"][:, cols])


@dataclass
class ARMApq(Baseline):
    """ARMA(p,q) fitted by Hannan-Rissanen, ``(p,q)`` chosen by K-fold CV.

    The samples are scattered ``(day, t)`` points rather than one contiguous
    series, so an MA term cannot read a stored residual. Instead the residual at
    lag ``k`` is *reconstructed* from the lag structure: a long AR predicts the
    value at lag ``k`` from lags ``k+1, k+2``, and the shortfall is that step's
    innovation. That is exactly the Hannan-Rissanen two-stage estimator, and it
    is why the table keeps ``N_LAGS = 5``.
    """

    def __init__(self, max_p: int = _MAX_P, max_q: int = _MAX_Q):
        super().__init__("arma_pq", "ARMA(p,q)", "#c994c7")
        self.max_p, self.max_q = max_p, max_q
        self.order, self.coefs, self.stage1 = {}, {}, {}
        self._state_attrs = ("order", "coefs", "stage1")

    # Stage 1: a long AR used only to generate innovations.
    def _stage1_resid(
        self, table: dict, kind: str, h: int, coef: np.ndarray, k: int,
    ) -> np.ndarray | None:
        """Reconstructed innovation at lag ``k``: value(k) - AR_pred(k+1, k+2)."""
        names = table["predictor_names"]
        need = [f"{_stem(kind, h)}{k + j}" for j in range(0, 3)]
        if any(nm not in names for nm in need):
            return None
        cols = [names.index(nm) for nm in need]
        value = table["X"][:, cols[0]]
        pred = _apply(coef, table["X"][:, cols[1:]])
        return value - pred

    def fit(self, table: dict) -> None:
        self._reset()
        X, Y = table["X"], table["Y"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            y = Y[:, self._tidx(table, target)]
            # Stage 1: AR(2) mapping lags (k+1, k+2) -> lag k, giving innovations.
            s1_cols = _lag_cols(table, kind, h, 3)
            if len(s1_cols) < 3:
                continue
            Xs, ys = _clean(X[:, s1_cols[1:]], X[:, s1_cols[0]])
            if len(ys) < 20:
                continue
            s1 = _ols(Xs, ys)
            self.stage1[target] = s1

            # Stage 2: candidates over (p, q).
            cand_cols, extra = {}, {}
            for p in range(1, self.max_p + 1):
                for q in range(0, self.max_q + 1):
                    ar_cols = _lag_cols(table, kind, h, p)
                    if len(ar_cols) < p:
                        continue
                    res = []
                    for k in range(1, q + 1):
                        r = self._stage1_resid(table, kind, h, s1, k)
                        if r is None:
                            break
                        res.append(r)
                    if len(res) < q:
                        continue
                    extra[(p, q)] = res
                    cand_cols[(p, q)] = ar_cols

            best_key, best_err, best_coef = None, float("inf"), None
            # ARMAX: the exogenous eleven go LAST, after the AR lags and the
            # reconstructed MA residuals, so the coefficient vector's lag block
            # keeps the layout _stage1_resid and predict() assume.
            info = _info_cols(table)
            info_block = [X[:, info]] if info else []
            for key, ar_cols in cand_cols.items():
                Xk = np.column_stack([X[:, ar_cols]] + extra[key] + info_block)
                # Same panel guard as select_by_cv: a reconstructed MA residual
                # at lag k reads lags k+1, k+2, so a q>0 candidate reaches two
                # lags deeper than its p suggests and dies first at long h.
                if coverage(Xk, np.arange(Xk.shape[1]), y) < MIN_COVERAGE:
                    continue
                err = cv_mse(Xk, y)
                if err < best_err:
                    Xc, yc = _clean(Xk, y)
                    if len(yc) < Xk.shape[1] + 2:
                        continue
                    best_key, best_err, best_coef = key, err, _ols(Xc, yc)
            if best_coef is not None:
                self.order[target], self.coefs[target] = best_key, best_coef

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.coefs:
            return None
        kind, h = self._parse(target)
        p, q = self.order[target]
        ar_cols = _lag_cols(table, kind, h, p)
        res = []
        for k in range(1, q + 1):
            r = self._stage1_resid(table, kind, h, self.stage1[target], k)
            if r is None:
                return None
            res.append(r)
        info = _info_cols(table)
        Xk = np.column_stack([table["X"][:, ar_cols]] + res
                             + ([table["X"][:, info]] if info else []))
        return _apply(self.coefs[target], Xk)


def _stem(kind: str, h: int) -> str:
    return {
        "return": f"ret_h{h}_lag",
        "spread_change": f"dspr_h{h}_lag",
        "volatility_change": f"rv_h{h}_lag",
    }[kind]


# ---------------------------------------------------------------------------
# Volatility specialists
# ---------------------------------------------------------------------------


class HARRV(Baseline):
    """Heterogeneous AR of realized volatility (Corsi): short + medium + long.

    Classical HAR-RV regresses the FORWARD realized vol on trailing vols over
    three cascading windows, then differences against the trailing vol to reach
    a change. That round trip needs the forward vol in the predictors' units,
    and on this panel it does not exist: the label is a dollar-unit change
    z-scored across the cross-section, while the predictors live in each view's
    own normalized units.

    So the change is regressed directly, with ``rv_h{h}_lag1`` — the very term
    the classical form subtracts — added to the cascade. The function class is
    unchanged: "level regression minus trailing vol" is the special case where
    that coefficient is pinned at -1, and here it is free. Volatility only.
    """

    def __init__(self):
        super().__init__("har_rv", "HAR-RV", "#1f77b4")
        self.coefs = {}
        self._state_attrs = ("coefs",)

    def _cols(self, table: dict, h: int) -> list[int]:
        return ([self._pidx(table, f"har_rv_{w}") for w in HAR_WINDOWS]
                + [self._pidx(table, f"rv_h{h}_lag1")])

    def fit(self, table: dict) -> None:
        self._reset()
        X, Y = table["X"], table["Y"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            if kind != "volatility_change":
                continue
            y = Y[:, self._tidx(table, target)]
            if coverage(X, self._cols(table, h), y) < MIN_COVERAGE:
                continue
            Xc, yc = _clean(X[:, self._cols(table, h)], y)
            if len(yc) < _MIN_TRAIN_ROWS:
                continue
            self.coefs[target] = _ols(Xc, yc)

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.coefs:
            return None
        _, h = self._parse(target)
        return _apply(self.coefs[target], table["X"][:, self._cols(table, h)])


class GARCH11(Baseline):
    """Variance-targeted GARCH(1,1) on actual returns, differenced to a change.

    The conditional variance of a GARCH(1,1) is an EWMA of squared returns, so
    the precomputed EWMA columns *are* the filter state; the fit selects the
    persistence (the decay whose h-step forecast best matches the training
    month) and the long-run variance. The h-step forecast reverts toward that
    long-run level at rate ``beta``, is converted to a volatility, and then
    differenced against trailing realized vol so it lands in the units of the
    ``volatility_change`` label. Volatility targets only.
    """

    def __init__(self):
        # Not a red: the supervised specialist owns red on every panel.
        super().__init__("garch11", "GARCH(1,1)", "#ff7f0e")
        self.decay = {}
        self.long_run = {}
        self._state_attrs = ("decay", "long_run")

    @staticmethod
    def _fwd_vol(v_t: np.ndarray, lr_var: float, beta: float, h: int) -> np.ndarray:
        """Mean forecast variance over the next ``h`` seconds -> a volatility."""
        # Average of beta^k for k in [0, h): how much of today's deviation from
        # the long-run variance survives across the horizon.
        gap = 1.0 - beta
        factor = (1.0 - beta ** h) / (h * gap) if gap > 1e-12 else 1.0
        var = lr_var + (v_t - lr_var) * factor
        return np.sqrt(np.clip(var, 0.0, None))

    def fit(self, table: dict) -> None:
        """Pick the persistence, and the long-run variance that goes with it.

        Selection is by RANK IC on the fit month, not by MSE. This model is the
        one baseline whose forecast is constructed rather than regressed, so it
        comes out in the view's own volatility units while the label is a
        z-score — an MSE between them is a comparison of two different scales
        and would rank the decays by their magnitude as much as by their fit.
        Rank IC is the metric this is ultimately scored by and is scale-free,
        which removes the question entirely.
        """
        self._reset()
        X, Y = table["X"], table["Y"]
        cell = table["cell"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            if kind != "volatility_change":
                continue
            y = Y[:, self._tidx(table, target)]
            trail = X[:, self._pidx(table, f"rv_h{h}_lag1")]
            best, best_ic = None, -np.inf
            for b in EWMA_DECAYS:
                v = X[:, self._pidx(table, f"ewma_var_{b}")]
                m = np.isfinite(v) & np.isfinite(y) & np.isfinite(trail)
                if m.sum() < _MIN_TRAIN_ROWS:
                    continue
                lr_var = float(np.mean(v[m]))
                pred = self._fwd_vol(v[m], lr_var, b, h) - trail[m]
                ic, _, n = grouped_rank_ic(pred, y[m], cell[m])
                if n and np.isfinite(ic) and ic > best_ic:
                    best, best_ic = (b, lr_var), ic
            if best is not None:
                self.decay[target], self.long_run[target] = best

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.decay:
            return None
        _, h = self._parse(target)
        b = self.decay[target]
        v = table["X"][:, self._pidx(table, f"ewma_var_{b}")]
        trail = table["X"][:, self._pidx(table, f"rv_h{h}_lag1")]
        return self._fwd_vol(v, self.long_run[target], b, h) - trail


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------


class RidgeARDL(Baseline):
    """Single-equation autoregressive distributed-lag regression, ridge-fit.

    The target is regressed on the 9 raw market features at several fixed lags
    (levels, sizes, volume, trade counts) together with its own horizon-matched
    lags. Own lags plus a distributed lag of every other channel is an ARDL.
    Ridge-regularized, since 40+ correlated regressors on a month of samples
    would otherwise overfit.

    NOT A VAR, though this class was called VARX until 2026-09-22. A VARX is a
    SYSTEM: a vector of endogenous series regressed jointly on lags of the whole
    vector, plus exogenous terms. This fits ONE SCALAR EQUATION per target,
    independently; the dependent variable is a FORWARD target that is not among
    the nine channels being lagged, so it is not an equation of any VAR system;
    and ``FEATURE_LAGS`` opens at lag 0, a contemporaneous regressor a
    reduced-form VAR equation does not carry. One equation, own lags, other
    series entering as a distributed lag -- that is an ARDL, and the old name
    claimed a system this never estimated.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__("ardl", "Ridge ARDL", "#2ca02c")
        self.alpha = alpha
        self.coefs, self.scaler, self.cols = {}, {}, {}
        self._state_attrs = ("coefs", "scaler", "cols")

    def _cols(self, table: dict, kind: str, h: int) -> list[int]:
        names = table["predictor_names"]
        cols = [
            names.index(f"feat{i}_lag{lag}")
            for lag in FEATURE_LAGS for i in range(9)
        ]
        cols += _lag_cols(table, kind, h, 2)
        cols.append(names.index("spread_level"))
        cols += _info_cols(table)
        return cols

    def fit(self, table: dict) -> None:
        self._reset()
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, Y = table["X"], table["Y"]
        for target in table["target_names"]:
            kind, h = self._parse(target)
            y = Y[:, self._tidx(table, target)]
            # Ridge's NaN imputation at predict time would hide a regressor
            # that is absent from most of the panel, so it is dropped at fit
            # time instead and the SAME set is used to predict.
            cols = usable_cols(X, self._cols(table, kind, h), y)
            if not cols:
                continue
            Xc, yc = _clean(X[:, cols], y)
            if len(yc) < len(cols) + 5:
                continue
            sc = StandardScaler().fit(Xc)
            model = Ridge(alpha=self.alpha).fit(sc.transform(Xc), yc)
            self.coefs[target] = model
            self.scaler[target] = sc
            self.cols[target] = cols

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self.coefs:
            return None
        cols = self.cols[target]
        Xq = table["X"][:, cols]
        # Ridge cannot consume NaN; impute with the training means via the scaler.
        sc = self.scaler[target]
        Xq = np.where(np.isfinite(Xq), Xq, sc.mean_)
        return self.coefs[target].predict(sc.transform(Xq))


def default_baselines() -> list[Baseline]:
    """Registry. Add a model here and it appears in the scorer and the plot."""
    return [
        MeanReversion(), ARp(), ARMApq(), HARRV(), GARCH11(), RidgeARDL(),
    ]
