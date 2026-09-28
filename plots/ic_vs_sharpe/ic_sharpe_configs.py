"""How much of a Sharpe ratio is the model, and how much is the harness?

The paper reports rank IC. A reviewer asks for a Sharpe ratio. Producing one
requires a chain of trading decisions -- tradable universe, how much of the
ranking to hold, risk model, allocator, rebalance frequency, transaction cost,
annualization -- that the model being evaluated has no opinion about. This
script measures the size of that chain's contribution by running the SAME
forecast through every configuration in the space and reporting the spread.

THE FORECAST IS SHARED AND THE PIPELINE IS SHARED. One ridge probe per
(arm, eval month) produces one forecast array; stable_finance.run_backtest
computes the IC and every configuration's Sharpe from that one array. A gap
between a checkpoint's IC and its Sharpe therefore cannot be a difference in
what was forecast, only in what was done with it.

NOTHING IS RE-EMBEDDED. The probe-breadth sweep already wrote embeddings for
every arm and month; quotes and realized returns are properties of a
(date, anchor, ticker) row and come from the cached panels, which are
model-neutral. This script is CPU-only and reads what is already on disk.

THE READOUT TAG IS PINNED PER ARM, never globbed. The same run_id has shards
under both ``pball`` (embedded at the model's training pool -- mean, for an SSL
arm) and ``pblast`` (re-embedded at last). Predictive evaluation is always read
at last, so a glob over ``pb*`` would silently concatenate two readouts and
violate the protocol in the quietest way available.

Usage:
    uv run plots/ic_vs_sharpe/ic_sharpe_configs.py \
        --arms pair_warp_6mo ts2vec_6mo --months 2008-08 2008-09 \
        --out-dir /data/lab/ic_sharpe_configs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# xs_ic_eval (ridge_alpha_for) and plots/core/readout.py both live outside this
# directory; this file moved here from scripts/eval and they did not.
sys.path.insert(0, str(ROOT / "scripts/eval"))

import stable_finance as sf  # noqa: E402

PANEL_CACHE = "/data/lab/market-jepa-mosaic/panel_cache"
STAGING = "/data/lab/probe_breadth/embeddings_staging"
# An "empty month" marker npz is a few hundred bytes; any real shard is MBs.
_EMPTY_MARKER_BYTES = 50_000
STATS_TAG = "xs_anchor_stats_fwdvwap60"
MANIFESTS = ("/data/lab/probe_breadth/manifest_all.index.json",
             "/data/lab/probe_breadth/manifest_floor.index.json")

HORIZON = 900
#: Column of ``return_900`` in the 18-column (3 target types x 6 horizons)
#: target block the sweep and the panel cache both store.
RETURN_COLUMN = 2

#: The return block occupies the first six columns, one per horizon, so
#: ``RETURN_COLUMN`` indexes HORIZONS as well. All six are fit in one streamed
#: pass: X'X dominates the cost and each extra target adds only an X'y, so a
#: horizon sweep costs a fraction of re-reading the pool six times. The sweep
#: itself still reads only ``RETURN_COLUMN`` and is unchanged by this.
HORIZONS = (300, 600, 900, 1800, 3600, 7200)

#: Sweeps that FORCED pool="last". For any arm whose training pool was mean,
#: these are the only staging tags whose embeddings satisfy the predictive
#: readout protocol.
REPOOLED_TAGS = ("pblast", "pblast2")


def last_by_construction() -> frozenset[str]:
    """Arms already at the predictive readout without a re-pooled sweep.

    Imported from ``plots/core/readout.py`` rather than restated here, because
    a second copy of this set is a second thing to get wrong, and getting it
    wrong means reading a mean-pooled embedding into a last-pooled study --
    a number that looks right and is not. That module verified the set by
    loading every arm and reading ``backbone.pool``.
    """
    sys.path.insert(0, str(ROOT / "plots/core"))
    from readout import LAST_BY_CONSTRUCTION
    return LAST_BY_CONSTRUCTION


def resolve_tag(arm: str, run_id: str, month: str) -> str:
    """The staging tag to read this (arm, month) from, or raise.

    THE TAG IS RESOLVED, NEVER GLOBBED. The same run_id has shards under both
    ``pball`` and ``pblast``, so a glob over ``pb*`` would concatenate two
    readouts into one matrix. An arm that trained at mean is accepted only from
    a re-pooled sweep; an arm that trained at last is accepted from any tag,
    because every tag read it at last.
    """
    found = set()
    for path in glob.glob(f"{STAGING}/*/pb*-part*/{run_id}/{month}_a8_*.npz"):
        tag = Path(path).parents[1].name.rsplit("-part", 1)[0]
        found.add(tag)
    usable = found if arm in last_by_construction() else found & set(REPOOLED_TAGS)
    if not usable:
        raise FileNotFoundError(
            f"{arm} {month}: no staged embeddings at the predictive readout "
            f"(found tags {sorted(found) or 'none'})"
        )
    # Prefer a re-pooled sweep when one exists: it is stamped, so the readout
    # is a recorded fact rather than an inference about how the arm trained.
    for tag in REPOOLED_TAGS:
        if tag in usable:
            return tag
    return sorted(usable)[0]


def manifest() -> list[dict]:
    rows = []
    for path in MANIFESTS:
        if Path(path).is_file():
            rows += json.loads(Path(path).read_text())
    return rows


def shard_paths(tag: str, run: str, month: str, anchors: int) -> list[str]:
    """One path per shard, deduplicated across nodes.

    THE SAME PART CAN SIT ON TWO NODES. The rescue waves that repopulated
    embeddings_staging copied some parts from more than one node, so a glob
    over ``<node>/<tag>-partNN/<run>/`` returns each shard twice. For a ridge
    that is not harmless: duplicating every row doubles both X'X and X'y, so
    the solution is the one a HALVED alpha would give. Arms duplicated in a
    month would then be regularized differently from arms that were not, and
    the comparison between them would be measuring the rescue, not the model.

    Shard names are unique within a (run, month), so the name is the identity.
    Where copies disagree in size the largest wins, on the grounds that a
    short copy is a truncated transfer rather than a different shard.
    """
    paths = sorted(glob.glob(
        f"{STAGING}/*/{tag}-part*/{run}/{month}_a{anchors}_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {tag} shards for {run} {month} a{anchors}")
    best: dict[str, tuple[int, str]] = {}
    for path in paths:
        name = Path(path).name
        size = Path(path).stat().st_size
        if name not in best or size > best[name][0]:
            best[name] = (size, path)
    deduped = [best[name][1] for name in sorted(best)]

    # A FULL-MONTH SHARD *IS* THE MONTH. Two staging generations share these
    # directories: an older one that wrote the whole month into shard 000 with
    # full_month=1, and a newer one that partitions the same rows across 32
    # shards. Where a re-stage left both behind, concatenating them reads every
    # row twice -- and because the two 000 files collide on name, the dedupe
    # above keeps the full-month copy and drops only ONE of the 32 partials, so
    # the pool comes out at 1.97x rather than a tell-tale 2x. Neither size nor
    # filename distinguishes the generations; only the flag does.
    full = [p for p in deduped if _is_full_month(p)]
    return full if full else deduped


def _is_full_month(path: str) -> bool:
    """Does this shard carry the whole month (staging generation A)?"""
    if Path(path).stat().st_size < _EMPTY_MARKER_BYTES:
        return False
    with np.load(path, allow_pickle=True) as handle:
        if "empty" in handle.files or "X" not in handle.files:
            return False
        return ("full_month" in handle.files
                and int(handle["full_month"][0]) == 1)


def iter_shards(tag: str, run: str, month: str, anchors: int):
    """Yield one shard at a time, never holding the pool.

    The six-month fit pool is ~2.8M rows x 384 features; materializing it costs
    about 30 GB once numpy promotes to float64, which is why it used to be
    subsampled. Streaming it keeps a job at a few hundred megabytes and lets the
    probe see every row -- and a frozen encoder plus ridge is a reservoir
    predictor whose IC climbs with fit size, so those rows are not a rounding
    detail.
    """
    for path in shard_paths(tag, run, month, anchors):
        f = np.load(path, allow_pickle=False)
        # A collapsed marker records a month that produced no usable rows. It
        # is a real state, not a corrupt file, and must be skipped.
        if "empty" in f.files or "X" not in f.files:
            continue
        yield {key: f[key] for key in ("X", "z", "date", "anchor", "ticker")}


def load_shards(tag: str, run: str, month: str, anchors: int) -> dict:
    """The whole month, concatenated. Used for the eval panel only."""
    out = {"X": [], "z": [], "date": [], "anchor": [], "ticker": []}
    for shard in iter_shards(tag, run, month, anchors):
        for key in out:
            out[key].append(shard[key])
    if not out["X"]:
        raise FileNotFoundError(f"every {tag} shard for {run} {month} was empty")
    return {k: np.concatenate(v) for k, v in out.items()}


def attach_market(rows: dict, month: str, anchors: int) -> dict:
    """Join quotes and realized returns onto embedding rows, and check the join."""
    market = sf.MarketPanel.from_cache(
        PANEL_CACHE, month, anchors_per_day=anchors, stats_tag=STATS_TAG
    )
    taken = market.take(market.align(rows["date"], rows["anchor"], rows["ticker"]))
    matched = float(taken["matched"].mean())
    if matched < 0.99:
        raise RuntimeError(
            f"{month} a{anchors}: only {matched:.1%} of embedding rows matched a "
            "cached market row; the quote table and the sweep disagree"
        )
    return taken


def fit_history(months: list[str], anchors: int = 36) -> dict:
    """Realized returns over the fit months, straight from the cached panels.

    THE COVARIANCE HISTORY MUST NOT COME FROM THE EMBEDDING SHARDS. The probe's
    fit pool is subsampled -- a 384-feature ridge needs far fewer rows than the
    pool holds -- but a covariance needs a dense (decision, asset) panel, and a
    subsample leaves most of its cells empty and some assets with no history at
    all. Returns are market data, so they are read in full from the cache and
    cost nothing extra: no embedding is touched here.
    """
    dates, anchor_ids, tickers, realized = [], [], [], []
    for month in months:
        market = sf.MarketPanel.from_cache(
            PANEL_CACHE, month, anchors_per_day=anchors, stats_tag=STATS_TAG
        )
        dates.append(market.dates)
        anchor_ids.append(market.anchors)
        tickers.append(market.tickers)
        realized.append(market.raw_targets[:, RETURN_COLUMN])
    return {"date": np.concatenate(dates), "anchor": np.concatenate(anchor_ids),
            "ticker": np.concatenate(tickers), "realized": np.concatenate(realized)}


def _write_pnl(scored, decision_axis, path, arm, eval_month) -> None:
    """Every configuration's per-decision profit path, in one array.

    Annualization is no longer swept -- it is pinned to ``daily`` -- so every
    scored configuration contributes one path and there is no half to drop.
    """
    kept = [r for r in scored if r.series is not None]
    if not kept:
        raise RuntimeError("no configuration produced a series")
    widths = {len(r.series.net) for r in kept}
    if len(widths) != 1:
        raise RuntimeError(f"ragged profit paths: {sorted(widths)}")
    axes = ("universe", "selection", "weighting", "risk_model", "efq",
            "rebalance")
    np.savez_compressed(
        path,
        arm=arm, eval_month=eval_month,
        decisions=np.asarray(kept[0].series.periods),
        gross=np.stack([r.series.gross for r in kept]).astype(np.float32),
        net=np.stack([r.series.net for r in kept]).astype(np.float32),
        cost=np.stack([r.series.cost for r in kept]).astype(np.float32),
        sharpe_mid=np.array([r.sharpe_mid for r in kept]),
        sharpe_net=np.array([r.sharpe_net for r in kept]),
        **{a: np.array([getattr(r.config, a) for r in kept]) for a in axes},
    )
    print(f"    wrote {len(kept)} profit path(s) x {len(kept[0].series.net)} "
          f"decisions to {path}", flush=True)


def score_one(arm: str, eval_month: str, row: dict, args) -> dict:
    """One (arm, eval month): fit the probe, attach the market, sweep the space."""
    started = time.time()
    run = row["run_id"]
    tag = resolve_tag(arm, run, eval_month)

    evaluation = load_shards(tag, run, eval_month, 8)

    from xs_ic_eval import ridge_alpha_for
    alphas = [ridge_alpha_for(f"return_{h}") for h in HORIZONS]
    alpha = alphas[RETURN_COLUMN]

    # PASS ONE: the probe, over every row of the six-month pool.
    probe = sf.StreamingRidge(alpha=alphas, min_samples=100)
    rows_seen = 0
    for month in row["fit_months"]:
        for shard in iter_shards(tag, run, month, 36):
            rows_seen += len(shard["X"])
            probe.partial_fit(shard["X"], shard["z"][:, :len(HORIZONS)])
    probe.finalize()
    forecast_eval = probe.predict(evaluation["X"])
    market_eval = attach_market(evaluation, eval_month, 8)

    # PASS TWO: the return-unit slope, which needs the fitted probe's own
    # predictions and so cannot share pass one. Only the five scalars the
    # regression needs are kept, so this pass is still O(1) in memory. Each
    # month's market panel is loaded once and joined shard by shard -- joining
    # a six-month pool against one month's panel would match a sixth of it.
    moments = np.zeros((5, len(HORIZONS)))  # n, sum f, sum r, sum ff, sum fr
    for month in row["fit_months"]:
        market = sf.MarketPanel.from_cache(
            PANEL_CACHE, month, anchors_per_day=36, stats_tag=STATS_TAG
        )
        for shard in iter_shards(tag, run, month, 36):
            taken = market.take(
                market.align(shard["date"], shard["anchor"], shard["ticker"])
            )
            fits = probe.predict(shard["X"])
            rs = taken["raw_targets"][:, :len(HORIZONS)]
            for h in range(len(HORIZONS)):
                f, r = fits[:, h], rs[:, h]
                ok = np.isfinite(f) & np.isfinite(r)
                if not ok.any():
                    continue
                f, r = f[ok], r[ok]
                moments[:, h] += [len(f), f.sum(), r.sum(),
                                  float(f @ f), float(f @ r)]
    n, sf_, sr, sff, sfr = moments
    if n[RETURN_COLUMN] < 2:
        raise RuntimeError("no fit rows carried both a forecast and a return")
    # WHY THE SLOPE EXISTS AT ALL. The probe predicts an empirical-uniform
    # rank, so its output spans ~[0, 1] whatever the month's returns did, while
    # the half-spread and the covariance are in return units. Handing a
    # rank-unit mu to a cost-aware allocator makes the cost term orders of
    # magnitude too small and silently disables the very path dependence the
    # allocator exists for. The scale cannot be recovered by tuning risk
    # aversion: the objective is positively homogeneous, so scaling lambda
    # rescales the whole solution and leaves its SHAPE -- including how far
    # cost damps each trade -- exactly where it was. Only the ratio moves it.
    # Slope through the origin on cross-sectionally demeaned pairs, assembled
    # from moments: identical to regressing the centred vectors directly.
    # One slope per horizon, from that horizon's own moments. A horizon whose
    # forecast has no spread carries no slope; only the swept horizon's is
    # fatal, because the others are carried for the panel and may legitimately
    # be absent at the session's end.
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = sff - sf_ * sf_ / np.where(n > 0, n, np.nan)
        betas = np.where(denominator > 0,
                         (sfr - sf_ * sr / np.where(n > 0, n, np.nan)) / denominator,
                         np.nan)
    beta = float(betas[RETURN_COLUMN])
    if not (np.isfinite(beta) and beta > 0):
        raise RuntimeError(
            f"forecast slope on returns is {beta:.3g}; the probe did not point "
            "at returns even on the months it was fit on"
        )
    # TWO DIFFERENT NUMBERS, both reported. ``seen`` is every row streamed and
    # is the quantity audit_probe_pools.py's invariant applies to: arms sharing
    # an eval month share their fit months, so it must be an IDENTICAL integer
    # across them, and any disagreement is lost shards rather than sampling.
    # ``used`` is the labelled subset the ridge actually fit, which is smaller
    # because some rows carry no return at this horizon. Reporting only the
    # second would make a complete pool look like a short one.
    n_fit_rows = int(probe.n_samples_[RETURN_COLUMN])
    n_fit_rows_seen = int(rows_seen)

    # ── the eval grid ───────────────────────────────────────────────────────
    decisions = sf.decision_labels(evaluation["date"], evaluation["anchor"])
    decision_axis = np.unique(decisions)
    seen = sf.to_grid(np.isfinite(market_eval["raw_targets"][:, RETURN_COLUMN]).astype(float),
                      decisions, evaluation["ticker"],
                      decision_axis=decision_axis,
                      asset_axis=np.unique(evaluation["ticker"]))[:, :, 0]
    keep = sf.presence_screen(np.isfinite(seen) & (seen > 0), minimum=args.min_presence)
    candidates = np.unique(evaluation["ticker"])[keep]

    # THE ASSET AXIS MUST SURVIVE BOTH PANELS. A name priced all through the
    # eval month but absent from the fit months has no covariance row, and the
    # risk model refuses to estimate one rather than inventing it.
    history_rows = fit_history(row["fit_months"])
    history_decisions = sf.decision_labels(history_rows["date"], history_rows["anchor"])
    history_axis = np.unique(history_decisions)
    history_grid = sf.to_grid(
        history_rows["realized"], history_decisions, history_rows["ticker"],
        decision_axis=history_axis, asset_axis=candidates,
    )[:, :, 0]
    observed = np.isfinite(history_grid).sum(axis=0)
    asset_axis = candidates[observed >= args.min_history]
    if len(asset_axis) < 20:
        raise RuntimeError(
            f"universe is {len(asset_axis)} names after requiring "
            f"{args.min_history} fit-month observations"
        )
    history_grid = history_grid[:, observed >= args.min_history]

    grid = lambda values: sf.to_grid(  # noqa: E731
        values, decisions, evaluation["ticker"],
        decision_axis=decision_axis, asset_axis=asset_axis)
    mu = grid(forecast_eval[:, RETURN_COLUMN])[:, :, 0] * beta
    realized = grid(market_eval["raw_targets"][:, RETURN_COLUMN])
    half_spread = grid(market_eval["half_spread"])[:, :, 0]
    days = np.array([label.split("|")[0] for label in decision_axis])

    axis = np.array([HORIZON])
    forecast_panel = sf.ForwardReturns(mu[:, :, None], decision_axis, asset_axis, axis)
    realized_panel = sf.ForwardReturns(realized, decision_axis, asset_axis, axis)

    # Sigma from the FIT months: 36 anchors a day against the eval month's 8,
    # which is the difference between a usable covariance on a few hundred
    # names and a rank-deficient one, and it is strictly out of sample.
    history = sf.ForwardReturns(
        history_grid[:, :, None], history_axis, asset_axis, axis
    )

    # THE PANEL IS THE EXPENSIVE PART. Staging, streaming the fit pool and
    # fitting the ridge cost ~6 minutes an (arm, month); sweeping a selection
    # rule over the result costs milliseconds. mu is dumped in RETURN UNITS --
    # already multiplied through by beta -- so a threshold rule can be tried
    # against the same half-spread the backtest charges, without refitting.
    if args.panel_out:
        panel_path = Path(args.panel_out) / f"{arm}-{eval_month}-panel.npz"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            panel_path, arm=arm, eval_month=eval_month, horizon=HORIZON,
            beta=beta, mu=mu.astype(np.float32),
            realized=realized[:, :, 0].astype(np.float32),
            # Every horizon, in return units, so a horizon sweep needs no
            # refit: mu_all[..., h] pairs with realized_all[..., h].
            horizons=np.array(HORIZONS),
            betas=betas.astype(np.float64),
            alphas=np.array(alphas, dtype=np.float64),
            fit_rows=probe.n_samples_.astype(np.int64),
            mu_all=np.stack(
                [grid(forecast_eval[:, h])[:, :, 0] * betas[h]
                 for h in range(len(HORIZONS))], axis=-1).astype(np.float32),
            realized_all=np.stack(
                [grid(market_eval["raw_targets"][:, h])[:, :, 0]
                 for h in range(len(HORIZONS))], axis=-1).astype(np.float32),
            half_spread=half_spread.astype(np.float32),
            history=history_grid.astype(np.float32),
            days=days, decisions=decision_axis, assets=asset_axis,
            history_decisions=history_axis,
        )
        print(f"    wrote panel {mu.shape} to {panel_path}", flush=True)

    space = sf.enumerate_configs(**json.loads(args.config_slice)) if args.config_slice \
        else sf.enumerate_configs()
    # THE EMBEDDING RISK MODEL NEEDS THE EMBEDDINGS, and a quarter of the
    # configuration space uses it. embedding_covariance reduces its input to
    # one mean vector per asset, so the asset means are passed directly as a
    # single-decision panel rather than materializing a
    # (decision, asset, feature) cube that would be ~300 MB per job and reduce
    # to the same numbers.
    asset_means = np.full((1, len(asset_axis), evaluation["X"].shape[1]), np.nan)
    index_of = {ticker: i for i, ticker in enumerate(asset_axis.tolist())}
    column = np.array([index_of.get(t, -1) for t in evaluation["ticker"].tolist()])
    for position in range(len(asset_axis)):
        rows = evaluation["X"][column == position]
        if len(rows):
            asset_means[0, position] = rows.mean(axis=0)
    usable = np.isfinite(asset_means[0]).all(axis=1)
    embeddings = None
    if usable.all():
        embeddings = sf.Embeddings(asset_means, np.array(["mean"]), asset_axis)

    # sweep_configs, not a loop over run_backtest: the covariance depends only
    # on the risk model and the book only on the weight-determining axes, so
    # the cost-aware solve -- essentially the whole runtime -- happens once per
    # distinct book instead of once per configuration.
    scored = list(sf.sweep_configs(
        forecast_panel, realized_panel, space, half_spread=half_spread,
        history=history, days=days, embeddings=embeddings, on_error="skip",
        with_series=bool(args.pnl_out),
    ))
    if args.pnl_out:
        _write_pnl(scored, decision_axis, args.pnl_out, arm, eval_month)
    reports = [report.as_dict() for report in scored]
    # A SHORTFALL IS REPORTED, NOT SWALLOWED. on_error="skip" keeps one bad
    # corner from losing the other four thousand, but a configuration that
    # never ran is a hole in the space, and a hole nobody counts is a hole
    # nobody notices.
    if len(reports) != len(space):
        scored = {tuple(r[a] for a in ("universe", "selection", "weighting",
                                       "risk_model", "efq", "rebalance"))
                  for r in reports}
        missing = [c for c in space
                   if (c.universe, c.selection, c.weighting, c.risk_model,
                       c.efq, c.rebalance) not in scored]
        by_risk = sorted({c.risk_model for c in missing})
        print(f"    WARNING {len(missing)}/{len(space)} configuration(s) did not "
              f"run; risk models affected: {by_risk}", flush=True)
    return {
        "arm": arm, "series_key": row["series_key"], "run_id": run,
        "readout_tag": tag, "eval_month": eval_month,
        "fit_months": row["fit_months"], "alpha": float(alpha),
        "return_per_unit_forecast": float(beta),
        "n_assets": int(len(asset_axis)),
        "n_configs_requested": int(len(space)),
        "n_configs_scored": int(len(reports)),
        "embedding_risk_available": bool(embeddings is not None),
        "n_history_decisions": int(len(history_axis)),
        "n_decisions": int(len(decision_axis)),
        "n_fit_rows": n_fit_rows, "n_fit_rows_seen": n_fit_rows_seen,
        "n_eval_rows": int(len(evaluation["X"])),
        "median_half_spread_bps": float(np.nanmedian(half_spread) * 1e4),
        "seconds": round(time.time() - started, 1),
        "configs": reports,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--arms", nargs="+", default=None,
                   help="series keys; default is every arm in the manifest")
    p.add_argument("--months", nargs="+", default=None,
                   help="eval months (default: every month the arm has)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fit-rows", type=int, default=0,
                   help="ignored; the probe streams the whole pool. Kept so "
                        "older command lines do not fail.")
    p.add_argument("--min-presence", type=float, default=0.6)
    p.add_argument("--min-history", type=int, default=60,
                   help="finite fit-month returns an asset needs to earn a "
                        "covariance row")
    p.add_argument("--pnl-out", default=None,
                   help="also write the per-decision profit path of every "
                        "configuration to this .npz. Only configurations that "
                        "annualize per decision are written, which loses "
                        "nothing: annualization rescales the Sharpe ratio and "
                        "never touches the path itself.")
    p.add_argument("--panel-out", default=None,
                   help="directory for the forecast panel (mu in return units, "
                        "realized, half_spread, history), one npz per (arm, "
                        "month), so selection rules can be swept without "
                        "refitting the probe.")
    p.add_argument("--config-slice", default=None,
                   help='JSON kwargs for enumerate_configs, e.g. \'{"efq": 0.7}\'')
    return p.parse_args()


def main():
    args = parse_args()
    index = manifest()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    wanted = set(args.arms) if args.arms else {r["series_key"] for r in index}
    for arm in sorted(wanted):
        for row in index:
            if row["series_key"] != arm:
                continue
            if args.months and row["eval_month"] not in args.months:
                continue
            jobs.append((arm, row["eval_month"], row))
    print(f"==> {len(jobs)} (arm, month) job(s)", flush=True)

    for number, (arm, month, row) in enumerate(jobs, start=1):
        target = out_dir / f"{arm}-{month}.json"
        # RESUME MUST COVER EVERY OUTPUT THIS RUN WAS ASKED FOR. Keying only on
        # the scored JSON silently skips a month whose panel was never written,
        # which is how a --panel-out pass over an already-scored sweep produces
        # nothing at all and reports success doing it.
        panel = (Path(args.panel_out) / f"{arm}-{month}-panel.npz"
                 if args.panel_out else None)
        if target.exists() and (panel is None or panel.exists()):
            print(f"[{number}/{len(jobs)}] {arm} {month}: already done", flush=True)
            continue
        try:
            result = score_one(arm, month, row, args)
        except Exception as error:  # noqa: BLE001
            print(f"[{number}/{len(jobs)}] {arm} {month}: FAILED {error!r}", flush=True)
            continue
        target.write_text(json.dumps(result, indent=2))
        ok = [c for c in result["configs"] if "failed" not in c]
        net = np.array([c["sharpe_net"] for c in ok], dtype=float)
        ic = ok[0]["information_coefficient"] if ok else float("nan")
        print(f"[{number}/{len(jobs)}] {arm} {month}: IC {ic:+.4f}  "
              f"net Sharpe p10 {np.nanpercentile(net, 10):+.2f} "
              f"median {np.nanmedian(net):+.2f} p90 {np.nanpercentile(net, 90):+.2f}  "
              f"({len(ok)}/{len(result['configs'])} configs, {result['seconds']:.0f}s)",
              flush=True)
    print("==> done", flush=True)


if __name__ == "__main__":
    main()
