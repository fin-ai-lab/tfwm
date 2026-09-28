"""Score the classical finance baselines by rank IC and write ``finance_panel_ic.json``.

Protocol, per fit month ``M`` — deliberately the same one every learned model
is held to (``xs_ic_series``):

  1. Build the panel table for the FIT POOL at 36 anchors/day and for ``M+1``
     at 8. The pool is the ``--fit-span`` months ending at ``M``.
  2. Fit each baseline on the pool, against its cross-sectional z-scores.
  3. Score its forecast on ``M+1`` with :func:`grouped_rank_ic` — Spearman
     within each ``(date, anchor)`` cell, averaged over cells.

THE FIT SPAN IS A FAIRNESS KNOB, AND ITS DEFAULT IS NOT THE REPORTED ONE.
``--fit-span 1`` is the original protocol and what ``finance_panel_ic.json``
holds. The learned arms it is plotted against do not get one month: a
supervised head trains on the six months ending at ``M``
(``build_probe_breadth_manifest.py``) and the probes in
``plots/core/probe_fit_table.py`` are fitted on that same six-month pool at
36 anchors/day, ~2.1M rows. A classical model fitted on 358k rows and drawn on
those axes is being compared on fit-set size as much as on model class, which
is the one thing this panel exists to hold fixed. ``--fit-span 6`` closes it.
Every record carries ``fit_months``/``n_fit_months``/``n_fit_rows`` so a pool
that came up short (the mosaic starts 2008-01) is visible in the artifact
rather than averaged in as if it were six.

Step 3 is the whole reason this is so much lighter than the AUC pipeline it
replaces. There is no discretizer, no logistic probe, and no calibration:
a rank correlation is invariant to any monotone transform of the forecast, so
a model's output never has to be pushed into label units and the entire
calibration-leakage surface disappears with it.

The output records carry the same schema as
``plots/metrics/mean_reversion_ic.json`` — ``{eval_month, target, predictor,
ic, se, n_cells}`` — plus a ``model`` key, so ``plots/metrics`` can consume a
series straight out of this file.

Usage::

    uv run plots/finance_baselines/calculate_panel_ic.py --months 2012-12
    uv run plots/finance_baselines/calculate_panel_ic.py --months-from-k2ind

    # the six-month pool at 15 minutes, the probe-table protocol
    uv run plots/finance_baselines/calculate_panel_ic.py --months-from-k2ind \
        --fit-span 6 --horizons 900 --view-learners --full-view \
        --json plots/metrics/finance_panel_ic_6mo.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "scripts" / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from market_jepa.eval.metrics import grouped_rank_ic  # noqa: E402
from models import default_baselines  # noqa: E402
from panel_tables import (  # noqa: E402
    build_month_table, build_pooled_table, fit_pool, next_month,
    restrict_targets,
)

# Written next to the other per-month result artifacts, which is where
# plots/metrics/metrics.py resolves a series' ``json_ic`` from.
OUT_PATH = _REPO / "plots" / "metrics" / "finance_panel_ic.json"

# Matches xs_ic_series.FIT_ANCHORS / EVAL_ANCHORS. A baseline fitted on a
# sparser grid than the encoders' probe would be handicapped by sample size
# rather than by its model, and one scored on a denser grid would not be
# averaging the same cross-sections.
FIT_ANCHORS = 36
EVAL_ANCHORS = 8


def k2ind_fit_months() -> list[str]:
    """Fit months of the standard LeJEPA series, so the lines share a panel.

    Deliberately NOT a glob written here. ``standard_k2ind_runs`` pins the
    generation hash and the lambda and refuses to return a short series; this
    function used to spell out ``k2ind-lamb-*``, which is the lambda sweep,
    and scored the whole experiment over 13 months instead of 32.
    """
    sys.path.insert(0, str(_REPO / "plots"))
    from style import standard_k2ind_months
    return standard_k2ind_months("train")


def score_month(fit_month: str, view_learners: bool, *, fit_span: int = 1,
                full_view: bool = False, horizons: list[int] | None = None,
                hgb_fit_rows: int | None = None, **tbl) -> list[dict]:
    """Every baseline's per-target IC for one ``(pool -> M+1)`` pair."""
    eval_month = next_month(fit_month)
    pool = fit_pool(fit_month, fit_span, tbl.get("mosaic_dir"))
    if len(pool) < fit_span:
        print(f"  pool is {len(pool)} months, not {fit_span}: the mosaic "
              f"starts 2008-01", flush=True)
    tr = build_pooled_table(pool, anchors_per_day=FIT_ANCHORS, **tbl)
    ev = build_month_table(eval_month, anchors_per_day=EVAL_ANCHORS, **tbl)
    if horizons:
        want = [t for t in tr["target_names"]
                if int(t.rsplit("_", 1)[1]) in horizons]
        tr, ev = restrict_targets(tr, want), restrict_targets(ev, want)
    # The full-view learners reopen the month as a view stream instead of being
    # handed 37 GB of tensors in the table, so they need the arguments that
    # built it. ``verbose`` is not one of them, and neither is ``cache_dir``:
    # a stream reads the mosaic, not the table cache, and iter_view_blocks
    # would reject the keyword.
    stream_kw = {k: v for k, v in tbl.items()
                 if k not in ("verbose", "cache_dir")}
    tr["stream_kw"] = ev["stream_kw"] = stream_kw

    models = list(default_baselines())
    views: list = []
    if view_learners or full_view:
        import view_models
        # THE GBM'S ROW CAP SCALES WITH THE POOL. Boosting cannot stream, so
        # it fits on a fixed-seed sample; leaving that sample at its
        # one-month size would hand the GBM a six-month SPREAD of the same
        # 60k rows while every other arm here got 2.1M, which is the
        # asymmetry --fit-span exists to remove.
        rows = (hgb_fit_rows if hgb_fit_rows is not None
                else view_models.HGB_FIT_ROWS * len(pool))
        if view_learners:
            views += view_models.default_view_learners(fit_rows=rows)
        if full_view:
            views += view_models.full_view_learners(fit_rows=rows)

    t0 = time.time()
    for model in models:
        model.fit(tr)
    print(f"  {len(models)} classical fits  {time.time() - t0:.0f}s", flush=True)

    # ONE STREAM FOR EVERY VIEW LEARNER, which is what fit_all/prime_all are
    # for and what this loop used to defeat: ``FullViewLearner.fit`` is the
    # single-learner convenience wrapper, so fitting the learners one at a
    # time in the loop below reopened the month once per learner, and then
    # ``predict`` reopened the EVAL month once per learner on top. At ~19
    # minutes per 36-anchor decode that is the difference between 25 minutes a
    # month and 95, and it got worse with every learner added -- the opposite
    # of the property the batching hooks were written to give. With a pool the
    # stream is one pass per POOLED MONTH, still one pass whatever the roster.
    if views:
        from view_models import fit_all, prime_all
        t0 = time.time()
        fit_all(views, tr)
        t1 = time.time()
        prime_all(views, ev)
        print(f"  {len(views)} view learners: fit {t1 - t0:.0f}s, "
              f"predict {time.time() - t1:.0f}s", flush=True)
        models += views

    stamp = {"fit_month": fit_month, "fit_months": pool,
             "n_fit_months": len(pool), "n_fit_rows": int(len(tr["Y"]))}
    records: list[dict] = []
    for model in models:
        for ti, target in enumerate(ev["target_names"]):
            pred = model.predict(ev, target)
            if pred is None:
                continue
            y = ev["Y"][:, ti]
            ok = np.isfinite(pred) & np.isfinite(y)
            if ok.sum() < 100:
                continue
            ic, se, n = grouped_rank_ic(pred[ok], y[ok], ev["cell"][ok])
            if not np.isfinite(ic):
                continue
            records.append({
                "eval_month": eval_month, **stamp,
                "target": target, "model": model.name,
                "predictor": model.label,
                "ic": float(ic), "se": float(se), "n_cells": int(n),
            })
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--months", nargs="+",
                   help="FIT months; each is scored on the following month")
    g.add_argument("--months-from-k2ind", action="store_true",
                   help="use exactly the months the standard LeJEPA series "
                        "covers, so every line averages the same panel")
    p.add_argument("--fit-span", type=int, default=1,
                   help="how many months the fit pool holds, ENDING at each "
                        "--months entry. 1 is the original protocol; 6 is the "
                        "span the learned arms train and probe on")
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   metavar="SECONDS",
                   help="score only these forward horizons (e.g. 900 for the "
                        "reported 15 minutes). Fewer targets also means fewer "
                        "censoring patterns, so the view learners' shell "
                        "partition is exact rather than projected")
    p.add_argument("--view-learners", action="store_true",
                   help="also fit the TAIL Ridge/GBM learners (8/24/64 tokens "
                        "back from the anchor). One extra decode pass per "
                        "pooled month; the Grams are megabytes")
    p.add_argument("--full-view", action="store_true",
                   help="also fit Ridge/GBM on the whole 2048x9 view. Rides "
                        "the same decode pass as --view-learners, but costs a "
                        "2.7 GB Gram per censoring shell and an 18,432-wide "
                        "GBM. Hours per eval month, not minutes")
    p.add_argument("--hgb-fit-rows", type=int, default=None,
                   help="row cap for the GBM sample. Default scales with the "
                        "pool (60k per month), so a 6-month pool draws 360k")
    p.add_argument("--mosaic-dir", default=None)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--cache-dir", default=None,
                   help="panel-table cache. A table's Z depends on the return "
                        "definition, so a new target needs its own directory.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--shard-stride", type=int, default=1,
                   help="1 = the whole month; N > 1 subsamples MDS shards. A "
                        "SMOKE TEST ONLY -- the panel is partial and nothing "
                        "is cached, so the ICs are not reportable")
    p.add_argument("--json", default=str(OUT_PATH))
    p.add_argument("--append", action="store_true",
                   help="merge into an existing json, replacing any record "
                        "with the same (eval_month, target, model)")
    return p.parse_args()


def main():
    args = parse_args()
    months = k2ind_fit_months() if args.months_from_k2ind else args.months
    print(f"{len(months)} fit months: {' '.join(months)}", flush=True)

    tbl = dict(mosaic_dir=args.mosaic_dir,
               xs_anchor_stats_dir=args.xs_anchor_stats_dir,
               **({} if args.cache_dir is None
                  else {"cache_dir": Path(args.cache_dir)}),
               batch_size=args.batch_size,
               shard_stride=args.shard_stride)

    out = Path(args.json)
    records: list[dict] = []
    if args.append and out.is_file():
        records = json.loads(out.read_text())
        # ONE FIT SPAN PER FILE. A record says which months it was fitted on,
        # but a reader averaging a model over its eval months does not look --
        # and nothing downstream does. Merging a six-month pass into the
        # one-month artifact would put both protocols under one series name
        # and make the difference between them invisible, which is the exact
        # comparison --fit-span was added to make.
        spans = {r.get("n_fit_months", 1) for r in records}
        if spans and spans != {args.fit_span}:
            raise SystemExit(
                f"{out.name} holds records at fit span(s) {sorted(spans)} and "
                f"this pass is --fit-span {args.fit_span}. Write a different "
                "--json; a file mixing spans cannot be read as one protocol.")

    for ym in months:
        print(f"\n{ym} -> {next_month(ym)}", flush=True)
        new = score_month(ym, args.view_learners, fit_span=args.fit_span,
                          full_view=args.full_view, horizons=args.horizons,
                          hgb_fit_rows=args.hgb_fit_rows, **tbl)
        key = {(r["eval_month"], r["target"], r["model"]) for r in new}
        records = [r for r in records
                   if (r["eval_month"], r["target"], r["model"]) not in key]
        records += new
        by_model: dict[str, list[float]] = {}
        for r in new:
            by_model.setdefault(r["model"], []).append(r["ic"])
        for name, ics in sorted(by_model.items()):
            print(f"  {name:16s} {len(ics):2d} targets  "
                  f"mean IC {np.mean(ics):+.4f}  max {np.max(ics):+.4f}", flush=True)
        # Written after every month: a 13-month pass is ~an hour and a crash
        # in month 12 must not discard the first eleven.
        out.write_text(json.dumps(records, indent=2))

    print(f"\nWrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
