"""Does a checkpoint's rank IC predict the Sharpe of a portfolio built from it?

IC and Sharpe answer different questions. IC asks whether the forecast orders
the cross-section; Sharpe asks whether a portfolio built from that ordering
makes money per unit of risk. They are related but not the same number: a
signal can rank well and still lose to the spread, and a covariance model can
turn a mediocre ranking into a decent portfolio by sizing it. This script
measures both on the SAME forecast so the relationship can be read off rather
than assumed.

THE FORECAST IS SHARED, DELIBERATELY. One ridge probe is fit per checkpoint on
the training month's embeddings against the empirical-uniform return target --
the production recipe from xs_ic_eval.score, reproducing xs_ic.json. Its
prediction on the eval month is scored for IC and is also, after
cross-sectional demeaning, the ``mu`` handed to the allocator. There is no
second probe, so a difference between a checkpoint's IC and its Sharpe cannot
be a difference in what was forecast.

THE PIPELINE, in stable-finance's stages:

    embeddings -> [ridge probe] -> forward returns -> [MV allocator]
        -> weights -> [orders] -> [bid/ask fills] -> Sharpe

These are supervised volatility-change models, so they enter at embeddings:
their trained head predicts the wrong quantity and is never read here.

TWO STAGES, because only the first one is expensive. ``embed`` runs the GPU
pass and writes each checkpoint's panel -- embeddings, targets, quotes, keys --
to disk. ``portfolio`` reads those panels and does the probe, the allocation
and the metrics on CPU in seconds. Every choice worth second-guessing (the
universe, the shrinkage, the exposure, the horizon) lives in the second stage,
so revisiting one costs nothing. A month decode costs about half an hour and
there are 64 of them.

MU IS IN RANK UNITS, and Sigma is in return units. The mean-variance solution
Sigma^-1 mu / lambda is therefore only defined up to the global constant that
relates them, and ``lambda`` is set per (checkpoint, horizon) so the cost-free
solution has a chosen mean gross exposure. That constant is what makes the
proportional transaction cost comparable across checkpoints -- without it a
model whose predictions happen to span a wider range would pay a
systematically smaller spread per unit of position. The consequence to keep in
mind: cross-sections are not individually calibrated, so a decision whose
realized dispersion was small still gets a full-sized bet.

A SIGNAL-ONLY PORTFOLIO rides along as the control: weights proportional to
the demeaned cross-sectional rank of the same forecast, at the same gross
exposure, with no covariance model at all. If IC tracks that Sharpe but not
the allocator's, the allocator is what broke the relationship.

Usage:
    uv run scripts/eval/portfolio_ic_sharpe.py embed \
        --ckpt-glob '/data/lab/market-jepa-checkpoints/supervised-full-month-vol-change-*/*/' \
        --panel-dir /data/lab/portfolio_ic_sharpe/panels

    uv run scripts/eval/portfolio_ic_sharpe.py portfolio \
        --panel-dir /data/lab/portfolio_ic_sharpe/panels \
        --out-dir /data/lab/portfolio_ic_sharpe/results
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/generic"))

from stable_finance import (  # noqa: E402
    ColumnwiseRidge,
    ForwardReturns,
    MeanVarianceAllocator,
    PortfolioWeights,
    evaluate_fills,
    cross_spread_sharpe,
    evaluate_weights,
    grouped_rank_ic_by_label,
    mid_price_sharpe,
    orders_from_weights,
    positions_from_fills,
    simulate_fills,
)

# The target family the portfolio trades. The checkpoints are volatility-change
# specialists, but a portfolio is priced in returns and nothing else here is.
TARGET_TYPE = "return"

# Anchors per day on the eval month -- the reported cross-sections. Matches
# xs_ic_eval's default so the IC computed here is the IC on disk.
EVAL_ANCHORS_PER_DAY = 8


# ── shared ──────────────────────────────────────────────────────────────────

def next_month(ym: str) -> str:
    year, month = (int(part) for part in ym.split("-"))
    return f"{year + (month == 12)}-{(month % 12) + 1:02d}"


def train_month_of(run_dir: Path) -> str:
    """The month a run's training window ENDS on; the eval month is the next.

    The sweep names each project ``...-<train start>-<train end>``, and this
    used to return the START -- the same month, and so the same answer, only
    while a run trained one calendar month. The locked recipe trains
    ``DatasetConfig.train_span_months`` of data ending at the eval-adjacent
    month, so ``...-2008-02-01-2008-07-31`` is ONE run whose eval month is
    2008-08, and reading the start would have embedded 2008-02/2008-03: a month
    the model did train on, scored as if it were held out, five months early.
    (plots/style.py fixed the same bug in the IC readers.)

    THE CONFIG IS AUTHORITATIVE, the directory name is the fallback: the
    project suffix is built by the launcher, the window is what training
    actually used, and only ``train_date_end`` records it.
    """
    meta_path = run_dir / "train_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        cfg = meta.get("config", meta)
        end = str((cfg.get("dataset") or {}).get("train_date_end") or "")
        if len(end) >= 7 and end[4] == "-":
            return end[:7]
    match = re.search(r"-\d{4}-\d{2}-\d{2}-(\d{4}-\d{2})-\d{2}$", run_dir.parent.name)
    if not match:
        raise ValueError(f"cannot read a training month out of {run_dir.parent.name!r}")
    return match.group(1)


def panel_name(run_dir: Path) -> str:
    return f"{run_dir.parent.name}__{run_dir.name}"


def cell_ids(dates, anchors) -> np.ndarray:
    """One id per (date, anchor) -- the unit a cross-section is defined over."""
    return np.char.add(np.char.add(dates.astype(str), "@"), anchors.astype(str))


# ── stage 1: embed ──────────────────────────────────────────────────────────

def embed_one(run_dir: Path, args, device, schedule) -> dict:
    """One checkpoint's two months of panel, ready for any downstream analysis.

    Both months keep their full embedding matrix, not just the probe's output:
    refitting a probe is milliseconds and re-decoding a month is half an hour,
    so the expensive artifact is the one worth storing.
    """
    from stable_finance.dataset.targets import AnchorTargetStats as AnchorStats
    from market_jepa.eval.checkpoints import load_model
    from post_train_ic_eval import _load_cfg
    from xs_ic_eval import (
        HORIZONS, day_anchors, embed_month, head_readout, panel_kwargs_for,
        raw_targets,
    )

    train_ym = train_month_of(run_dir)
    eval_ym = next_month(train_ym)
    stats_dir = Path(args.xs_anchor_stats_dir)
    mosaic = Path(args.mosaic_dir)

    cfg = _load_cfg(run_dir, None)
    model = load_model(str(run_dir), cfg, device)
    model.eval()
    panel_kw = panel_kwargs_for(cfg)

    stored: dict[str, np.ndarray] = {}
    for tag, ym, per_day in (("train", train_ym, args.train_anchors_per_day),
                             ("eval", eval_ym, args.anchors_per_day)):
        year, month = ym.split("-")
        stats = AnchorStats(stats_dir / f"{ym}.npz")
        cache = embed_month(
            model, mosaic / year / month, ym, stats, schedule,
            day_anchors(per_day), device, args.batch_size,
            num_shards=max(args.smoke_shards, 1), **panel_kw,
        )
        names = [str(name) for name in cache["target_names"]]
        columns = [names.index(f"{TARGET_TYPE}_{h}") for h in HORIZONS]
        stored[f"{tag}_X"] = cache["X"].astype(np.float32)
        # Only the return family is kept: the volatility and spread columns are
        # not what a portfolio trades and would triple the file.
        stored[f"{tag}_uniform"] = cache["z"][:, columns].astype(np.float32)
        stored[f"{tag}_raw"] = raw_targets(cache, stats)[:, columns].astype(np.float32)
        stored[f"{tag}_quote"] = cache["quote"].astype(np.float32)
        stored[f"{tag}_date"] = cache["date"].astype("U10")
        stored[f"{tag}_anchor"] = cache["anchor"].astype(np.int32)
        # NATIVE WIDTH, not a fixed one. A ticker truncated to fit would not
        # error -- it would silently merge two symbols into one asset and put
        # a position in a name that does not exist.
        stored[f"{tag}_ticker"] = np.asarray(cache["ticker"])

        # THE HEAD'S OWN READOUT, on the same rows as the embeddings. These
        # return specialists were trained on return_900, so unlike the
        # volatility arm their head predicts the quantity a portfolio trades
        # and is a second, independent entry point into the same downstream
        # pipeline. A checkpoint whose head is missing or randomly initialized
        # returns nothing here and is scored on the probe alone.
        for task, (scores, _proba) in head_readout(model, cache["X"], device).items():
            stored[f"{tag}_head:{task}"] = np.asarray(scores, dtype=np.float32)

    stored["horizons"] = np.asarray(HORIZONS, dtype=np.int32)
    stored["months"] = np.asarray([train_ym, eval_ym])
    return stored


def run_embed(args):
    import torch
    from stable_finance.dataset import MarketSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)
    panel_dir = Path(args.panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)

    dirs = resolve_dirs(args)
    print(f"==> embedding {len(dirs)} checkpoint(s) on {device}", flush=True)
    for index, run_dir in enumerate(dirs, start=1):
        destination = panel_dir / f"{panel_name(run_dir)}.npz"
        if destination.exists() and not args.smoke_shards:
            print(f"[{index}/{len(dirs)}] {run_dir.name}: panel exists", flush=True)
            continue
        started = time.time()
        try:
            stored = embed_one(run_dir, args, device, schedule)
        except Exception as error:  # noqa: BLE001 - one bad month must not end the sweep
            print(f"[{index}/{len(dirs)}] {run_dir.name}: FAILED {error!r}", flush=True)
            continue
        if args.smoke_shards:
            print(f"[{index}/{len(dirs)}] {run_dir.name}: SMOKE, "
                  f"{len(stored['eval_X'])} eval rows, nothing written", flush=True)
            continue
        # Write beside the target and rename, so an interrupted run never
        # leaves a half-written panel that looks complete to the next pass.
        scratch = destination.with_suffix(".partial.npz")
        np.savez(scratch, **stored)
        scratch.rename(destination)
        print(f"[{index}/{len(dirs)}] {run_dir.name} "
              f"{stored['months'][0]}->{stored['months'][1]}  "
              f"{len(stored['train_X'])} train / {len(stored['eval_X'])} eval rows"
              f"  ({time.time() - started:.0f}s)", flush=True)
    print("==> embed done", flush=True)


# ── stage 2: portfolio ──────────────────────────────────────────────────────

def densify(rows, cols, values, n_decisions, n_assets, n_columns):
    """Scatter flat panel rows onto the ``(decision, asset, horizon)`` grid.

    Cells no row fills stay NaN, which every downstream stage reads as "not
    observed" rather than as zero.
    """
    keep = (rows >= 0) & (cols >= 0)
    dense = np.full((n_decisions, n_assets, n_columns), np.nan)
    dense[rows[keep], cols[keep]] = np.asarray(values, dtype=np.float64)[keep]
    return dense


def half_spread_of(quote: np.ndarray) -> np.ndarray:
    """Quoted half-spread over mid; NaN where the market is not two-sided."""
    bid, ask = quote[:, 0].astype(np.float64), quote[:, 1].astype(np.float64)
    broken = ~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask < bid)
    mid = (bid + ask) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(broken, np.nan, (ask - bid) / 2 / mid)


def choose_universe(panel, args):
    """The traded universe: names priced often enough in BOTH months.

    Ranked by presence and then by median quoted half-spread, so the cap keeps
    the names that are actually continuously tradable rather than an arbitrary
    alphabetical slice.
    """
    presence: dict[str, list[float]] = {}
    for tag in ("train", "eval"):
        cells = cell_ids(panel[f"{tag}_date"], panel[f"{tag}_anchor"])
        n_cells = len(np.unique(cells))
        priced = np.isfinite(panel[f"{tag}_raw"]).any(axis=1)
        tickers, counts = np.unique(panel[f"{tag}_ticker"][priced], return_counts=True)
        seen = dict(zip(tickers.tolist(), counts / n_cells))
        for ticker in np.unique(panel[f"{tag}_ticker"]).tolist():
            presence.setdefault(ticker, []).append(seen.get(ticker, 0.0))

    spread = half_spread_of(panel["eval_quote"])
    tickers = panel["eval_ticker"]
    cost = {}
    for ticker in np.unique(tickers).tolist():
        rows = spread[tickers == ticker]
        rows = rows[np.isfinite(rows)]
        cost[ticker] = float(np.median(rows)) if len(rows) else np.inf

    eligible = [
        ticker for ticker, seen in presence.items()
        if len(seen) == 2 and min(seen) >= args.min_presence
    ]
    eligible.sort(key=lambda t: (-min(presence[t]), cost.get(t, np.inf), t))
    return np.array(sorted(eligible[:args.max_assets]))


def fit_probe(panel, horizons):
    """The production ridge probe, and the IC of its prediction per horizon.

    One ridge per return horizon on the empirical-uniform target, fit on the
    training month and applied to the eval month -- ``xs_ic_eval.score``'s
    recipe, so the IC score_pathway computes from these predictions reproduces
    ``xs_ic/return_<h>`` on disk. Scoring lives in score_pathway so the probe
    and the head cannot be measured two different ways.
    """
    from xs_ic_eval import ridge_alpha_for

    predictions = np.full((len(panel["eval_X"]), len(horizons)), np.nan)
    train_predictions = np.full((len(panel["train_X"]), len(horizons)), np.nan)
    for slot, horizon in enumerate(horizons):
        y_train = panel["train_uniform"][:, slot].astype(np.float64)
        if np.isfinite(y_train).sum() < 100:
            continue
        probe = ColumnwiseRidge(
            alpha=ridge_alpha_for(f"{TARGET_TYPE}_{horizon}"), min_samples=100,
        ).fit(panel["train_X"], y_train[:, None])
        predictions[:, slot] = probe.predict(panel["eval_X"])[:, 0]
        # In-sample on the month the probe was fit, and used only to calibrate
        # the forecast's SCALE. A ridge with 384 features against ~400k rows
        # has effective degrees of freedom under 0.1% of the sample, so the
        # optimism in this slope is far below the precision it is read at.
        train_predictions[:, slot] = probe.predict(panel["train_X"])[:, 0]
    return predictions, train_predictions


def demeaned(values: np.ndarray) -> np.ndarray:
    """Cross-sectionally demean each (decision, horizon), ignoring NaN.

    A long-short portfolio is a bet on the ORDER of a cross-section, so the
    level of the forecast -- which for a uniform target is ~0.5 everywhere and
    carries no information -- must not become a market-wide position.
    """
    with np.errstate(invalid="ignore"):
        centre = np.nanmean(values, axis=1, keepdims=True)
    return values - np.where(np.isfinite(centre), centre, 0.0)


def return_units(signal: np.ndarray, realized: np.ndarray) -> float:
    """Expected return per unit of forecast, from the training month.

    WHY THIS EXISTS. The probe predicts an empirical-uniform rank, so its
    output is a number in [0, 1] and its cross-sectional spread is ~0.3
    whatever the month's returns did. Sigma and the quoted half-spread are in
    return units. Handing the allocator a rank-unit mu against a return-unit
    cost makes the cost term about three orders of magnitude too small to
    matter, and the transaction-cost machinery -- the entire reason the
    allocator is path-dependent -- silently does nothing.

    The scale cannot be recovered by tuning risk_aversion. The objective is
    positively homogeneous: scaling lambda by k scales the whole solution path
    by 1/k and leaves its SHAPE, including how far the cost term damps each
    trade, exactly where it was. Only the ratio of cost to mu moves it.

    So mu is put into return units here: regress the cross-sectionally
    demeaned realized return on the demeaned forecast over the training
    month's decisions, through the origin. Both sides are already demeaned, so
    the slope is the expected return earned per unit of forecast -- the number
    that makes "is this trade worth the spread?" a well-posed question.
    """
    x, y = demeaned(signal), demeaned(realized)
    usable = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.sum(x[usable] * x[usable]))
    if denominator <= 0:
        return np.nan
    return float(np.sum(x[usable] * y[usable]) / denominator)


def rank_weights(mu: np.ndarray, gross: float) -> np.ndarray:
    """Signal-only control: weights proportional to the demeaned forecast rank.

    No covariance, no cost term, no path dependence -- the portfolio a reader
    would write down from the ranking alone, scaled to the same gross exposure
    the allocator is normalized to.
    """
    weights = np.zeros(mu.shape)
    for decision in range(mu.shape[0]):
        observed = np.isfinite(mu[decision, :, 0])
        if observed.sum() < 2:
            continue
        order = np.argsort(np.argsort(mu[decision, observed, 0])).astype(np.float64)
        centred = order - order.mean()
        scale = np.abs(centred).sum()
        if scale > 0:
            weights[decision, observed, 0] = gross * centred / scale
    return weights


def risk_aversion_for(mu: np.ndarray, sigma: np.ndarray, gross: float) -> float:
    """lambda such that the COST-FREE solution averages ``gross`` exposure.

    mu is a rank score and Sigma a return covariance, so their ratio has no
    natural scale; fixing the exposure is what makes one checkpoint's spread
    bill comparable with another's. Solved on the closed form Sigma^-1 mu,
    which is the cost-free optimum at lambda = 1.
    """
    raw = np.linalg.solve(sigma, np.nan_to_num(mu, nan=0.0).T).T
    mean_gross = float(np.abs(raw).sum(axis=1).mean())
    return max(mean_gross / gross, 1e-12)


def head_task_of(panel) -> str | None:
    """The task a stored head readout belongs to, or None for no usable head."""
    tasks = sorted(key.split("head:", 1)[1] for key in panel
                   if key.startswith("eval_head:"))
    if not tasks:
        return None
    if len(tasks) > 1:
        raise RuntimeError(f"expected a single-task head, found {tasks}")
    return tasks[0]


def score_pathway(ctx, flat_eval, dense_eval, dense_train, slot, horizon) -> dict:
    """IC and portfolio Sharpe for one forecast at one horizon.

    THE TWO ENTRY POINTS SHARE THIS FUNCTION. A probe forecast and a head
    forecast differ only in the arrays handed in; the universe, the
    covariance, the return-unit calibration, the exposure normalization, the
    cost, the fills and every metric are literally the same code below. That is
    what makes "the head beats the probe" a statement about the forecast
    rather than about two evaluation paths that drifted.

    Both ICs are recomputed here rather than taken from the probe fit, for the
    same reason.
    """
    args, assets = ctx["args"], ctx["assets"]
    eval_decisions, train_decisions = ctx["eval_decisions"], ctx["train_decisions"]
    axis = np.asarray([horizon])
    realized_eval = ctx["realized_eval"][:, :, slot, None]
    realized_train = ctx["realized_train"][:, :, slot, None]
    outcome = ForwardReturns(realized_eval, eval_decisions, assets, axis)

    # All eval rows, against the empirical-uniform target -- the reported
    # metric, and for the probe pathway the one that reproduces xs_ic.json.
    y_eval = ctx["eval_uniform"][:, slot].astype(np.float64)
    ok = np.isfinite(y_eval) & np.isfinite(flat_eval)
    all_rows = grouped_rank_ic_by_label(flat_eval[ok], y_eval[ok], ctx["cells"][ok])
    # RESTRICTED TO THE TRADED UNIVERSE: the portfolio can only express a view
    # on names it holds, so this is the ranking the Sharpe is built from.
    universe = grouped_rank_ic_by_label(
        dense_eval.ravel(), realized_eval[:, :, 0].ravel(),
        np.repeat(eval_decisions, len(assets)),
    )
    result = {
        "ic": float(all_rows.mean), "ic_se": float(all_rows.standard_error),
        "ic_cells": int(all_rows.observations), "ic_rows": int(ok.sum()),
        "ic_universe": float(universe.mean),
        "ic_universe_se": float(universe.standard_error),
    }

    beta = return_units(dense_train, ctx["realized_train"][:, :, slot])
    result["return_per_unit_forecast"] = float(beta)
    # A non-positive slope means the forecast did not point at returns even in
    # the month it was fit on. Flipping mu to "fix" that would trade a broken
    # signal backwards, so the horizon reports its IC and no portfolio.
    if not (np.isfinite(beta) and beta > 0):
        result["untradable"] = "forecast slope on returns is not positive"
        return result

    mu = demeaned(dense_eval) * beta
    forecast_h = ForwardReturns(mu[:, :, None], eval_decisions, assets, axis)

    # Sigma from the TRAINING month: 36 anchors a day against the eval month's
    # 8, which is the difference between a usable covariance on a few hundred
    # names and a rank-deficient one. It is also strictly out-of-sample for the
    # eval month's returns.
    allocator = MeanVarianceAllocator(
        risk_aversion=1.0, shrinkage=args.shrinkage, min_samples=20,
        max_iter=args.max_iter,
    ).fit(ForwardReturns(realized_train, train_decisions, assets, axis))
    lam = risk_aversion_for(mu, allocator.covariance_[0], args.target_gross)
    allocator.risk_aversion = lam
    weights = allocator.predict(forecast_h, cost=ctx["half_spread"])

    # THE SAME ALLOCATOR WITH THE COST TERM OFF, marked at mid. Without this
    # the frictionless MV number is not comparable with the rank control: the
    # traded weights are deliberately STALE (that is what the cost term buys),
    # and marking a sticky book at frictionless prices charges it for carrying
    # yesterday's view while the always-refreshed control pays nothing. This
    # isolates the covariance model's own contribution from that stickiness.
    frictionless = allocator.predict(forecast_h, cost=None)

    charged = np.nan_to_num(ctx["half_spread"], nan=0.0)
    periods = ctx["periods"]
    spread_charged = evaluate_weights(weights, outcome, half_spread=charged,
                                      periods_per_year=periods)[0]

    # The executable path: what the orders actually filled at, marked on the
    # position really held rather than on the target weights. An unquoted name
    # is left unfilled, so the realized path can lag.
    orders = orders_from_weights(weights)
    fills = simulate_fills(orders, np.nan_to_num(ctx["bid"], nan=0.0),
                           np.nan_to_num(ctx["ask"], nan=0.0))
    filled = evaluate_fills(fills, outcome, periods_per_year=periods)[0]
    held = positions_from_fills(fills).values[:, :, 0]

    control = PortfolioWeights(rank_weights(mu[:, :, None], args.target_gross),
                               eval_decisions, assets, axis)
    control_charged = evaluate_weights(control, outcome, half_spread=charged,
                                       periods_per_year=periods)[0]

    target = weights.values[:, :, 0]
    turnover = np.abs(np.diff(target, axis=0, prepend=0.0)).sum(axis=1)
    result.update({
        "sharpe_mid": float(mid_price_sharpe(weights, outcome,
                                             periods_per_year=periods)[0]),
        # THE SAME BOOK, CHARGED THE OBSERVABLE QUOTE. ``sharpe_mid`` marks
        # every trade at the midpoint; this one buys at the best ask and sells
        # at the best bid on the weight CHANGE, which is the pair the question
        # "does the signal survive the spread?" is asked in. It differs from
        # ``sharpe_net`` only in that it charges the TARGET weights rather
        # than simulating fills, so the two agree whenever every order fills
        # and the gap is a fill story, not a cost one.
        "sharpe_cross_spread": float(cross_spread_sharpe(
            weights, outcome, np.nan_to_num(ctx["bid"], nan=0.0),
            np.nan_to_num(ctx["ask"], nan=0.0), periods_per_year=periods)[0]),
        "sharpe_control_cross_spread": float(cross_spread_sharpe(
            control, outcome, np.nan_to_num(ctx["bid"], nan=0.0),
            np.nan_to_num(ctx["ask"], nan=0.0), periods_per_year=periods)[0]),
        "sharpe_mid_nocost": float(mid_price_sharpe(frictionless, outcome,
                                                    periods_per_year=periods)[0]),
        "mean_turnover_nocost": float(np.abs(np.diff(
            frictionless.values[:, :, 0], axis=0, prepend=0.0)).sum(axis=1).mean()),
        # The executable number. It equals the spread-charged Sharpe on the
        # target weights whenever every order fills, which is the identity
        # evaluate_fills promises; the two are kept apart here only so a month
        # where something stops quoting cannot hide inside one column.
        # `fill_gap` is that check, and is 0.0 when they agree.
        "sharpe_net": float(filled.sharpe),
        "fill_gap": float(filled.sharpe - spread_charged.sharpe),
        "sharpe_control_mid": float(mid_price_sharpe(control, outcome,
                                                     periods_per_year=periods)[0]),
        "sharpe_control_net": float(control_charged.sharpe),
        "cost_to_edge_ratio": float(
            np.nanmedian(ctx["half_spread"]) / np.nanmean(np.abs(mu))
        ),
        "risk_aversion": float(lam),
        "mean_gross_exposure": float(np.abs(target).sum(axis=1).mean()),
        "mean_turnover": float(turnover.mean()),
        "mean_gross_return_bps": float(np.nanmean(spread_charged.gross_returns) * 1e4),
        "mean_cost_bps": float(np.nanmean(spread_charged.costs) * 1e4),
        "unfilled_weight_fraction": float(
            np.abs(held - target).sum() / max(np.abs(target).sum(), 1e-12)
        ),
    })
    return result


def portfolio_one(path: Path, args) -> dict:
    panel = dict(np.load(path, allow_pickle=False))
    horizons = [int(h) for h in panel["horizons"]]
    train_ym, eval_ym = (str(m) for m in panel["months"])

    predictions, train_predictions = fit_probe(panel, horizons)
    assets = choose_universe(panel, args)
    if len(assets) < 20:
        raise RuntimeError(f"universe is {len(assets)} names")
    asset_of = {ticker: i for i, ticker in enumerate(assets.tolist())}

    grids = {}
    for tag in ("train", "eval"):
        cells = cell_ids(panel[f"{tag}_date"], panel[f"{tag}_anchor"])
        decisions = np.unique(cells)
        decision_of = {key: i for i, key in enumerate(decisions.tolist())}
        rows = np.array([decision_of.get(key, -1) for key in cells.tolist()])
        cols = np.array([asset_of.get(key, -1) for key in panel[f"{tag}_ticker"].tolist()])
        grids[tag] = {
            "cells": cells, "decisions": decisions, "rows": rows, "cols": cols,
            "realized": densify(rows, cols, panel[f"{tag}_raw"], len(decisions),
                                len(assets), len(horizons)),
        }
    eval_decisions = grids["eval"]["decisions"]
    train_decisions = grids["train"]["decisions"]

    def to_grid(tag, values, n_columns):
        return densify(grids[tag]["rows"], grids[tag]["cols"], values,
                       len(grids[tag]["decisions"]), len(assets), n_columns)

    quotes = to_grid("eval", panel["eval_quote"], 2)
    bid, ask = quotes[:, :, 0], quotes[:, :, 1]
    broken = ~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask < bid)
    bid, ask = np.where(broken, np.nan, bid), np.where(broken, np.nan, ask)
    with np.errstate(invalid="ignore"):
        half_spread = (ask - bid) / 2 / ((bid + ask) / 2)

    ctx = {
        "args": args, "assets": assets,
        "eval_decisions": eval_decisions, "train_decisions": train_decisions,
        "realized_eval": grids["eval"]["realized"],
        "realized_train": grids["train"]["realized"],
        "eval_uniform": panel["eval_uniform"], "cells": grids["eval"]["cells"],
        "bid": bid, "ask": ask, "half_spread": half_spread,
        "periods": args.anchors_per_day * 252,
    }

    probe_eval = to_grid("eval", predictions, len(horizons))
    probe_train = to_grid("train", train_predictions, len(horizons))
    pathways = {"probe": {
        str(horizon): score_pathway(
            ctx, predictions[:, slot].astype(np.float64),
            probe_eval[:, :, slot], probe_train[:, :, slot], slot, horizon,
        )
        for slot, horizon in enumerate(horizons)
    }}

    # THE HEAD'S OWN FORECAST, through the identical downstream pipeline. It
    # predicts one task at one horizon, so it contributes one entry rather than
    # the probe's six -- correlating a return_900 head against the other five
    # horizons would produce numbers that look like results and mean nothing.
    task = head_task_of(panel)
    head_horizon = int(task.rsplit("_", 1)[1]) if task else None
    if task and head_horizon in horizons:
        slot = horizons.index(head_horizon)
        flat = panel[f"eval_head:{task}"].astype(np.float64)
        pathways["head"] = {str(head_horizon): score_pathway(
            ctx, flat,
            to_grid("eval", flat[:, None], 1)[:, :, 0],
            to_grid("train", panel[f"train_head:{task}"].astype(np.float64)[:, None], 1)[:, :, 0],
            slot, head_horizon,
        )}

    return {
        "panel": path.name,
        "train_month": train_ym,
        "eval_month": eval_ym,
        "head_task": task,
        "n_assets": int(len(assets)),
        "n_eval_decisions": int(len(eval_decisions)),
        "n_train_decisions": int(len(train_decisions)),
        "n_eval_rows": int(len(panel["eval_X"])),
        "n_train_rows": int(len(panel["train_X"])),
        "median_half_spread_bps": float(np.nanmedian(half_spread) * 1e4),
        "periods_per_year": ctx["periods"],
        "target_gross": args.target_gross,
        "shrinkage": args.shrinkage,
        "min_presence": args.min_presence,
        "max_assets": args.max_assets,
        "pathways": pathways,
    }


def run_portfolio(args):
    panels = sorted(Path(args.panel_dir).glob("*.npz"))
    panels = [p for p in panels if not p.name.endswith(".partial.npz")]
    if args.limit:
        panels = panels[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> {len(panels)} panel(s)", flush=True)
    for index, path in enumerate(panels, start=1):
        started = time.time()
        try:
            result = portfolio_one(path, args)
        except Exception as error:  # noqa: BLE001
            print(f"[{index}/{len(panels)}] {path.stem}: FAILED {error!r}", flush=True)
            continue
        (out_dir / f"{path.stem}.json").write_text(json.dumps(result, indent=2))
        line = (f"[{index}/{len(panels)}] {result['train_month']}"
                f"->{result['eval_month']} ({result['n_assets']} names, "
                f"{time.time() - started:.0f}s)")
        for name, by_horizon in result["pathways"].items():
            headline = by_horizon.get(str(args.headline_horizon))
            if headline is None:
                continue
            line += (f"\n      {name:<6} IC {headline.get('ic', float('nan')):+.4f}"
                     f"  univIC {headline.get('ic_universe', float('nan')):+.4f}"
                     f"  mid {headline.get('sharpe_mid', float('nan')):+.2f}"
                     f"  net {headline.get('sharpe_net', float('nan')):+.2f}"
                     f"  ctrl {headline.get('sharpe_control_mid', float('nan')):+.2f}"
                     f"  turn {headline.get('mean_turnover', float('nan')):.2f}")
        print(line, flush=True)
    print("==> portfolio done", flush=True)


# ── entry point ─────────────────────────────────────────────────────────────

def resolve_dirs(args):
    dirs = [Path(d) for d in args.ckpt_dirs]
    if args.ckpt_glob:
        dirs += [Path(d) for d in sorted(glob.glob(args.ckpt_glob))]
    dirs = [d for d in dirs
            if (d / "backbone.pt").is_file() or (d / "model.pt").is_file()]
    return dirs[:args.limit] if args.limit else dirs


def parse_args():
    from market_jepa.schemas import BLL01MachineConfig

    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest="stage", required=True)

    embed = sub.add_parser("embed", help="GPU pass; writes one panel per checkpoint")
    embed.add_argument("--ckpt-glob", default=None, help="shell glob of run dirs (quote it)")
    embed.add_argument("--ckpt-dirs", nargs="*", default=[])
    embed.add_argument("--panel-dir", required=True)
    embed.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    embed.add_argument("--holiday-csv", default=machine.holiday_csv)
    embed.add_argument("--xs-anchor-stats-dir",
                       default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    embed.add_argument("--anchors-per-day", type=int, default=EVAL_ANCHORS_PER_DAY)
    embed.add_argument("--train-anchors-per-day", type=int, default=36)
    embed.add_argument("--batch-size", type=int, default=256)
    embed.add_argument("--smoke-shards", type=int, default=0,
                       help="use only 1/N of each month's shards; nothing is written")
    embed.add_argument("--limit", type=int, default=None)

    port = sub.add_parser("portfolio", help="CPU pass; probe, allocate, score")
    port.add_argument("--panel-dir", required=True)
    port.add_argument("--out-dir", required=True)
    port.add_argument("--anchors-per-day", type=int, default=EVAL_ANCHORS_PER_DAY,
                      help="only sets the Sharpe annualization")
    port.add_argument("--headline-horizon", type=int, default=900)
    port.add_argument("--max-assets", type=int, default=400,
                      help="cap on the traded universe. The covariance is "
                           "estimated from the training month's decisions, so "
                           "an unbounded universe would be rank-deficient in "
                           "the directions the allocator leans on hardest.")
    port.add_argument("--min-presence", type=float, default=0.6,
                      help="fraction of decisions an asset must be priced at, "
                           "in BOTH months, to enter the universe")
    port.add_argument("--shrinkage", type=float, default=0.3)
    port.add_argument("--target-gross", type=float, default=2.0,
                      help="mean gross exposure the cost-free solution is "
                           "scaled to; sets risk_aversion per horizon")
    port.add_argument("--max-iter", type=int, default=400,
                      help="FISTA iterations per rebalance")
    port.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "embed":
        run_embed(args)
    else:
        run_portfolio(args)


if __name__ == "__main__":
    main()
