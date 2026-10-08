"""Tests for the classical finance baselines.

Three layers, cheapest first:

  * **Mechanics** — shapes, finiteness, length masking, save/load.
  * **Invariance** — every forecast must survive a per-sample affine map of the
    price channels, because the view a mode receives is per-sample z-scored
    while targets are computed on the raw series.
  * **Econometrics** — the estimators recover known parameters from simulated
    GARCH(1,1) data, the per-horizon AR(1) coefficients are recovered as the
    predictive correlation of the trailing-h predictor with the next-h target,
    and the forecasts move in the direction the fitted coefficients say.

All CPU, all synthetic — no real data, no W&B.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from market_jepa.modeling.modes import FinanceBaseline
from market_jepa.modeling.modes.finance_baseline_params import (
    _ASK,
    _BID,
    BaselineFitter,
    BaselineParams,
    MonthlyBaselineParams,
    corr_from_stats,
    extract_series,
    fit_garch11,
    steps_for_horizon,
    trailing_return,
    trailing_return_predictor,
    trailing_spread_change,
)

_H = [300, 600, 900, 1800, 3600, 7200]

# A single well-populated cell, so lookups are deterministic in tests.
_PARAMS = BaselineParams(
    alpha=0.10,
    beta=0.85,
    phi_return_by_h={h: -0.05 for h in _H},   # slight short-horizon reversal
    phi_spread_by_h={h: -0.30 for h in _H},   # spread-change reversal
)


def _model(**kwargs) -> FinanceBaseline:
    kwargs.setdefault("backbone", torch.nn.Identity())
    model = FinanceBaseline(**kwargs)
    model.params = MonthlyBaselineParams(cells={}, fallback=_PARAMS)
    return model


def _synthetic_window(
    B: int = 6, C: int = 9, T: int = 256, dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A realistic post-normalization window: O(1) z-scored mid, O(0.1) spread."""
    torch.manual_seed(0)
    steps = torch.randn(B, T, dtype=dtype) * 0.08
    mid = steps.cumsum(dim=1)
    spread = 0.10 + 0.05 * torch.rand(B, T, dtype=dtype)
    x = torch.zeros(B, C, T, dtype=dtype)
    x[:, _BID] = mid - spread / 2
    x[:, _ASK] = mid + spread / 2
    x[:, 1] = mid
    x[:, 5:] = torch.rand(B, 4, T, dtype=dtype)
    lengths = torch.full((B,), T, dtype=torch.long)
    lengths[0] = 120  # exercise the length mask
    return x, lengths


def _window_from_returns(r: torch.Tensor, spread_level: torch.Tensor | None = None):
    """Wrap a (B, T) return series into a (B, 9, T+1) view with a flat spread."""
    B, T = r.shape
    mid = torch.cat([torch.zeros(B, 1, dtype=r.dtype), r.cumsum(dim=1)], dim=1)
    spread = (
        spread_level
        if spread_level is not None
        else torch.full((B, T + 1), 0.1, dtype=r.dtype)
    )
    x = torch.zeros(B, 9, T + 1, dtype=r.dtype)
    x[:, _BID] = mid - spread / 2
    x[:, _ASK] = mid + spread / 2
    return x, torch.full((B,), T + 1, dtype=torch.long)


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_forecast_bank_shape_and_finite():
    model = _model()
    x, lengths = _synthetic_window()
    out = model._compute_forecasts(x, lengths, None, None)
    assert out.shape == (x.shape[0], model.d_embedding)
    assert model.d_embedding == 3 * len(model.horizons)
    assert torch.isfinite(out).all()


def test_feature_names_cover_families_and_horizons():
    model = _model(horizons=[300, 900])
    assert model.feature_names == [
        "ar1_return_300", "ar1_return_900",
        "garch_vol_change_300", "garch_vol_change_900",
        "ar1_spread_change_300", "ar1_spread_change_900",
    ]


def test_length_masking_ignores_padding():
    model = _model()
    x, lengths = _synthetic_window()
    out = model._compute_forecasts(x, lengths, None, None)
    xpad = x.clone()
    xpad[0, :, int(lengths[0]):] = 999.0  # garbage beyond the valid length
    out_pad = model._compute_forecasts(xpad, lengths, None, None)
    assert torch.allclose(out_pad[0], out[0], atol=1e-9)


def test_degenerate_window_is_finite():
    model = _model()
    x, _ = _synthetic_window()
    short = torch.ones(x.shape[0], dtype=torch.long)  # length-1 windows
    out = model._compute_forecasts(x, short, None, None)
    assert torch.isfinite(out).all()


def test_encode_returns_per_view_embeddings():
    model = _model()
    x, lengths = _synthetic_window()
    enc = model.encode([x, x], [lengths, lengths])
    assert enc["embeddings"].shape == (x.shape[0], 2, model.d_embedding)


def test_model_opts_into_the_metadata_channel():
    # collect_probe_data gates on this attribute; without it the baseline would
    # silently receive no dates and fall back to global parameters everywhere.
    assert FinanceBaseline.wants_metadata is True


def test_training_step_is_noop_and_eval_step_reports():
    model = _model()
    x, lengths = _synthetic_window()
    batch = {"buckets": [{"views": [x], "lengths": [lengths], "n_global_views": 2}]}
    assert model.training_step(batch, torch.device("cpu")) is None
    metrics = model.eval_step([batch], torch.device("cpu"))
    assert "eval/baseline_feature_std" in metrics
    assert metrics["eval/baseline_n"] == float(x.shape[0])


def test_family_subsetting_and_validation():
    model = _model(features=["volatility"], horizons=[300])
    assert model.feature_names == ["garch_vol_change_300"]
    with pytest.raises(ValueError):
        FinanceBaseline(backbone=None, features=["not_a_family"])
    with pytest.raises(ValueError):
        FinanceBaseline(backbone=None, features=[])


def test_save_pretrained_roundtrip_carries_fitted_params():
    model = _model(features=["return", "spread"], horizons=[300, 1800])
    model.params = MonthlyBaselineParams(
        cells={("2023-04", 8): _PARAMS}, fallback=_PARAMS,
    )
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert os.path.exists(os.path.join(d, "baseline_params.json"))
        reloaded = FinanceBaseline.from_pretrained(d)
    assert reloaded.feature_names == model.feature_names
    assert reloaded.horizons == [300, 1800]
    # The parameter table travels with the checkpoint, not by reference.
    got = reloaded.params.lookup("2023-04", 8)
    assert got.alpha == pytest.approx(_PARAMS.alpha)
    assert got.phi_spread(300) == pytest.approx(_PARAMS.phi_spread(300))


# ---------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------


def test_scale_invariance_under_affine_price_map():
    """forecast(a*price + b) == forecast(price) up to the eps regularizers.

    Views are per-sample z-scored, so the same market state can arrive with any
    positive scale and any offset. A forecast that moved under that map would be
    ranking samples by their normalization constant, not by their dynamics.
    """
    model = _model()
    x, lengths = _synthetic_window()
    out = model._compute_forecasts(x, lengths, None, None)

    a, b = 3.7, -2.1  # a>0; per-sample z-score scale + price offset
    xa = x.clone()
    xa[:, _BID] = a * x[:, _BID] + b
    xa[:, _ASK] = a * x[:, _ASK] + b
    out_a = model._compute_forecasts(xa, lengths, None, None)

    assert (out - out_a).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Econometrics — estimator recovery
# ---------------------------------------------------------------------------


def _simulate_garch(B: int, T: int, alpha: float, beta: float, seed: int = 0):
    """Simulate GARCH(1,1) with unit unconditional variance."""
    g = torch.Generator().manual_seed(seed)
    omega = 1.0 - alpha - beta
    h = torch.ones(B, dtype=torch.float64)
    out = torch.zeros(B, T, dtype=torch.float64)
    for t in range(T):
        eps = torch.randn(B, generator=g, dtype=torch.float64)
        z = h.sqrt() * eps
        out[:, t] = z
        h = omega + alpha * z * z + beta * h
    return out


def test_fit_garch11_recovers_known_parameters():
    alpha_true, beta_true = 0.10, 0.85
    z = _simulate_garch(B=48, T=2000, alpha=alpha_true, beta=beta_true)
    z = (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True)
    mask = torch.ones_like(z, dtype=torch.bool)

    alpha, beta = fit_garch11(z, mask)

    assert abs((alpha + beta) - (alpha_true + beta_true)) < 0.03
    assert abs(alpha - alpha_true) < 0.05


def test_corr_from_stats_matches_numpy():
    torch.manual_seed(5)
    x = torch.randn(500, dtype=torch.float64)
    y = 0.6 * x + torch.randn(500, dtype=torch.float64)
    n = float(x.numel())
    got = corr_from_stats(
        n, float(x.sum()), float(y.sum()), float((x * y).sum()),
        float((x * x).sum()), float((y * y).sum()),
    )
    expected = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    assert got == pytest.approx(expected, abs=1e-6)


def test_corr_from_stats_degenerate_is_zero():
    assert corr_from_stats(1, 0, 0, 0, 0, 0) == 0.0          # too few points
    assert corr_from_stats(10, 0, 0, 0, 0, 5) == 0.0         # x has no variance


def test_extract_series_standardizes_to_zero_mean_unit_variance():
    x, lengths = _synthetic_window()
    s = extract_series(x, lengths)
    n = s.z_mask.sum(1).clamp(min=1)
    mean = (s.z * s.z_mask).sum(1) / n
    var = (s.z * s.z * s.z_mask).sum(1) / n - mean * mean
    assert mean.abs().max().item() < 1e-9
    assert (var - 1.0).abs().max().item() < 1e-6


def test_fitter_recovers_predictive_correlation():
    """When the next-h target IS the trailing-h predictor, the fitted AR(1)
    coefficient must come back as (clipped) unit correlation for both families.
    """
    from stable_finance.dataset import get_target_names

    h, agg, B, T = 900, 6, 64, 512
    x, lengths = _synthetic_window(B=B, T=T)
    s = extract_series(x, lengths)
    n = torch.full((B,), steps_for_horizon(h, agg), dtype=torch.long)
    x_ret = trailing_return_predictor(s, n)
    x_spr = trailing_spread_change(s.w, s.w_mask, n)

    names = get_target_names([h], ["return", "spread_change"])
    targets = torch.zeros(B, len(names), dtype=torch.float32)
    targets[:, names.index(f"return_{h:03d}")] = x_ret.float()          # target == predictor
    targets[:, names.index(f"spread_change_{h:03d}")] = x_spr.float()

    bucket = {
        "views": [x],
        "lengths": [lengths],
        "agg_factors": torch.full((B,), agg, dtype=torch.long),
        "dates": ["2020-07-15"] * B,
        "targets": targets,
    }
    fitter = BaselineFitter(horizons=[h])
    fitter.add_bucket(bucket)
    params = fitter.fit()
    cell = params.lookup("2020-07", agg)
    assert cell.phi_return(h) == pytest.approx(0.999, abs=1e-3)
    assert cell.phi_spread(h) == pytest.approx(0.999, abs=1e-3)


def test_fitter_recovers_sign_of_a_planted_relationship():
    """A negative trailing->next relationship must yield a negative coefficient."""
    from stable_finance.dataset import get_target_names

    h, agg, B, T = 900, 6, 128, 512
    x, lengths = _synthetic_window(B=B, T=T)
    s = extract_series(x, lengths)
    n = torch.full((B,), steps_for_horizon(h, agg), dtype=torch.long)
    x_ret = trailing_return_predictor(s, n)

    names = get_target_names([h], ["return", "spread_change"])
    targets = torch.zeros(B, len(names), dtype=torch.float32)
    targets[:, names.index(f"return_{h:03d}")] = (-0.8 * x_ret).float()  # reversal

    bucket = {
        "views": [x], "lengths": [lengths],
        "agg_factors": torch.full((B,), agg, dtype=torch.long),
        "dates": ["2020-07-15"] * B, "targets": targets,
    }
    fitter = BaselineFitter(horizons=[h])
    fitter.add_bucket(bucket)
    assert fitter.fit().lookup("2020-07", agg).phi_return(h) == pytest.approx(-0.999, abs=1e-3)


# ---------------------------------------------------------------------------
# Trailing-h predictors
# ---------------------------------------------------------------------------


def test_trailing_return_reads_only_the_last_n_steps():
    """The horizon-matched lookback: a short window and a long one can disagree.

    Last 5 increments are +1, the 95 before are -1. The 5-step trailing return is
    +5; the 100-step trailing return is dominated by the earlier down-move.
    """
    z = torch.cat([-torch.ones(1, 95), torch.ones(1, 5)], dim=1).double()
    mask = torch.ones_like(z, dtype=torch.bool)
    short = trailing_return(z, mask, torch.tensor([5]))
    long = trailing_return(z, mask, torch.tensor([100]))
    assert float(short) == pytest.approx(5.0)
    assert float(long) == pytest.approx(-90.0)


def test_trailing_return_clamps_horizon_to_window():
    z = torch.ones(1, 10).double()
    mask = torch.ones_like(z, dtype=torch.bool)
    # n far beyond the window -> whole-window cumulative return.
    assert float(trailing_return(z, mask, torch.tensor([9999]))) == pytest.approx(10.0)


def test_trailing_spread_change_is_last_minus_n_back():
    w = torch.arange(20, dtype=torch.float64)[None, :]  # w[t] = t
    mask = torch.ones_like(w, dtype=torch.bool)
    assert float(trailing_spread_change(w, mask, torch.tensor([5]))) == pytest.approx(5.0)
    assert float(trailing_spread_change(w, mask, torch.tensor([1]))) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Forecast behaviour
# ---------------------------------------------------------------------------


def test_return_forecast_follows_phi_sign():
    """forecast_return_h = phi_h * trailing_h_return; sign is phi * predictor."""
    torch.manual_seed(3)
    up = 0.05 * torch.ones(1, 400, dtype=torch.float64)   # steady positive drift
    x, lengths = _window_from_returns(up)

    pos = BaselineParams(alpha=0.1, beta=0.85, phi_return_by_h={300: +0.3}, phi_spread_by_h={})
    neg = BaselineParams(alpha=0.1, beta=0.85, phi_return_by_h={300: -0.3}, phi_spread_by_h={})

    m_pos = _model(features=["return"], horizons=[300])
    m_pos.params = MonthlyBaselineParams(cells={}, fallback=pos)
    m_neg = _model(features=["return"], horizons=[300])
    m_neg.params = MonthlyBaselineParams(cells={}, fallback=neg)

    f_pos = float(m_pos._compute_forecasts(x, lengths, None, None)[0, 0])
    f_neg = float(m_neg._compute_forecasts(x, lengths, None, None)[0, 0])
    assert f_pos > 0.0        # momentum: positive trailing return -> positive forecast
    assert f_neg < 0.0        # reversal: opposite sign
    assert f_pos == pytest.approx(-f_neg, rel=1e-6)


def test_garch_forecasts_negative_vol_change_after_a_burst():
    """GARCH mean-reversion: vol forecast falls after a burst, rises after a lull."""
    torch.manual_seed(7)
    T, tail = 600, 150
    calm = torch.randn(1, T, dtype=torch.float64)
    burst = calm.clone(); burst[:, -tail:] *= 4.0
    lull = calm.clone(); lull[:, -tail:] *= 0.2

    model = _model(features=["volatility"], horizons=[tail])
    outs = []
    for series in (burst, lull):
        x, lengths = _window_from_returns(series)
        outs.append(float(model._compute_forecasts(x, lengths, None, None)[0, 0]))
    assert outs[0] < 0.0, "vol should be forecast to fall after a burst"
    assert outs[1] > 0.0, "vol should be forecast to rise after a lull"


def test_horizon_matched_lookback_can_flip_the_return_forecast():
    """A recent up-move over a longer down-trend: the short and long horizons read
    opposite trailing windows, so with the same phi the forecasts differ in sign.
    """
    down_then_up = torch.cat(
        [-0.02 * torch.ones(1, 380), 0.20 * torch.ones(1, 20)], dim=1,
    ).double()
    x, lengths = _window_from_returns(down_then_up)   # agg defaults to 1 s/token

    params = BaselineParams(
        alpha=0.1, beta=0.85,
        phi_return_by_h={20: +0.3, 400: +0.3}, phi_spread_by_h={},
    )
    model = _model(features=["return"], horizons=[20, 400])
    model.params = MonthlyBaselineParams(cells={}, fallback=params)
    out = model._compute_forecasts(x, lengths, None, None)[0]
    assert float(out[0]) > 0.0, "20s horizon sees the recent up-move"
    assert float(out[1]) < 0.0, "400s horizon integrates the earlier down-trend"


def test_agg_factor_converts_horizon_into_steps():
    """A 300 s horizon is 300 steps at 1 s/token but 30 at 10 s/token.

    Same phi at both horizons, so any difference is the step-count conversion.
    """
    x, lengths = _synthetic_window(B=2, T=256)
    # Equal coefficient at h=30 and h=300 isolates the n = h/agg conversion.
    params = BaselineParams(
        alpha=0.1, beta=0.85, phi_return_by_h={},
        phi_spread_by_h={30: 0.5, 300: 0.5},
    )
    fine = _model(features=["spread"], horizons=[300])
    coarse = _model(features=["spread"], horizons=[30])
    fine.params = MonthlyBaselineParams(cells={}, fallback=params)
    coarse.params = MonthlyBaselineParams(cells={}, fallback=params)

    aggs_10 = torch.full((2,), 10, dtype=torch.long)
    aggs_1 = torch.ones(2, dtype=torch.long)
    dates = ["2020-07-15", "2020-07-15"]

    at_10s = fine._compute_forecasts(x, lengths, dates, aggs_10)   # 300/10 = 30 steps
    at_1s = coarse._compute_forecasts(x, lengths, dates, aggs_1)   # 30/1  = 30 steps
    assert torch.allclose(at_10s, at_1s, atol=1e-12)
    # agg genuinely changes the answer: at 1 s/token the 300 s horizon is 300 steps.
    assert not torch.allclose(at_10s, fine._compute_forecasts(x, lengths, dates, aggs_1))


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------


def test_monthly_params_save_load_roundtrip():
    cells = {
        ("2023-01", 6): BaselineParams(
            0.1, 0.85, {300: 0.01, 900: -0.02}, {300: 0.97}, n_series=10, n_steps=99,
        ),
        ("2023-02", 8): BaselineParams(0.2, 0.70, {300: -0.05}, {900: 0.90}),
    }
    table = MonthlyBaselineParams(cells=cells, fallback=_PARAMS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "params.json")
        table.save(path)
        loaded = MonthlyBaselineParams.load(path)
    assert len(loaded) == 2
    got = loaded.lookup("2023-01", 6)
    assert got.n_steps == 99
    assert got.phi_return(900) == pytest.approx(-0.02)   # int horizon keys survive JSON
    assert loaded.lookup("2023-02", 8).phi_return(300) == pytest.approx(-0.05)
    assert loaded.fallback.beta == pytest.approx(_PARAMS.beta)


def test_lookup_fallback_chain():
    a = BaselineParams(0.1, 0.85, {}, {})
    b = BaselineParams(0.2, 0.70, {}, {})
    fallback = BaselineParams(0.3, 0.60, {}, {})
    table = MonthlyBaselineParams(
        cells={("2023-01", 6): a, ("2023-05", 11): b}, fallback=fallback,
    )
    assert table.lookup("2023-01", 6) is a            # exact
    assert table.lookup("2023-01", 9) is a            # same month, nearest agg
    assert table.lookup("2023-09", 11) is b           # same agg, nearest earlier month
    assert table.lookup("2023-09", 3) is fallback     # nothing close -> fallback


def test_lookup_prefers_earlier_month_over_later():
    early = BaselineParams(0.1, 0.85, {}, {})
    late = BaselineParams(0.2, 0.70, {}, {})
    table = MonthlyBaselineParams(
        cells={("2023-01", 7): early, ("2023-12", 7): late}, fallback=_PARAMS,
    )
    # June sits between the two; the earlier one wins so a baseline never uses
    # parameters fitted on its own future.
    assert table.lookup("2023-06", 7) is early


def test_lookup_as_of_blocks_own_month_cells():
    train = BaselineParams(0.1, 0.85, {}, {})
    eval_ = BaselineParams(0.2, 0.70, {}, {})
    table = MonthlyBaselineParams(
        cells={("2023-01", 7): train, ("2023-02", 7): eval_}, fallback=_PARAMS,
    )
    # Uncapped, an eval-month sample hits its own month's cell — parameters
    # fitted on that month's own future returns (the leak the cap closes).
    assert table.lookup("2023-02", 7) is eval_
    # Capped at the train month, resolution falls through to the train cell,
    # matching what the plots-pipeline adapter enforces by pinning dates.
    assert table.lookup("2023-02", 7, as_of="2023-01") is train
    # The cap also blinds the same-month-nearest-agg step (step 2): uncapped
    # this resolves within the eval month; capped it can't see any 2023-02
    # cell, and with no earlier agg-9 cell either, it lands on the fallback.
    assert table.lookup("2023-02", 9) is eval_
    assert table.lookup("2023-02", 9, as_of="2023-01") is _PARAMS
    # With no eligible cells at all, the fallback still applies.
    empty_cap = table.lookup("2023-02", 7, as_of="2022-12")
    assert empty_cap is _PARAMS
    # The cache keys on as_of, so capped and uncapped lookups don't collide.
    assert table.lookup("2023-02", 7) is eval_


def test_finance_baseline_threads_params_as_of(tmp_path):
    train = BaselineParams(0.1, 0.85, {300: -0.02}, {300: 0.9})
    eval_ = BaselineParams(0.2, 0.70, {300: -0.05}, {300: 0.8})
    table = MonthlyBaselineParams(
        cells={("2023-01", 7): train, ("2023-02", 7): eval_}, fallback=_PARAMS,
    )
    path = str(tmp_path / "params.json")
    table.save(path)
    fb = FinanceBaseline(
        backbone=None, params_path=path, params_as_of="2023-01",
    )
    rows = fb._param_rows(
        ["2023-02-15"], torch.tensor([7.0]), B=1,
    )
    assert rows[0].alpha == pytest.approx(train.alpha)
    # Round-trips through save/from_pretrained.
    ckpt = str(tmp_path / "ckpt")
    fb.save_pretrained(ckpt)
    reloaded = FinanceBaseline.from_pretrained(ckpt)
    assert reloaded.params_as_of == "2023-01"


def test_variance_targeting_pins_the_intercept():
    p = BaselineParams(alpha=0.1, beta=0.85, phi_return_by_h={}, phi_spread_by_h={})
    assert p.omega == pytest.approx(0.05)
    assert p.persistence == pytest.approx(0.95)
