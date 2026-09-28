"""Build every panel table a scoring pass needs, in parallel.

``calculate_panel_ic.py`` builds tables on demand, but it does so one month at
a time inside a single process, and the build is dominated by MDS decode — an
embarrassingly parallel cost. Running it here first turns a serial hour into a
few minutes and makes the scoring pass pure compute.

A pass needs the whole FIT POOL at 36 anchors/day — the ``--fit-span`` months
ending at each fit month — and the following month at 8. At ``--fit-span 6``
the pools of consecutive fit months overlap heavily, so the job list is
deduplicated: 32 fit months need 127 distinct 36-anchor tables, not 192.

Usage::

    uv run plots/finance_baselines/prebuild_tables.py --months 2012-12 --workers 8
    uv run plots/finance_baselines/prebuild_tables.py --months-from-k2ind \
        --fit-span 6 --workers 10
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "scripts" / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build(job):
    ym, anchors, kw = job
    from panel_tables import build_month_table, existing_table
    if existing_table(ym, anchors) is not None and not kw.get("force"):
        return f"{ym} a{anchors}: cached"
    t0 = time.time()
    try:
        tbl = build_month_table(ym, anchors_per_day=anchors,
                                verbose=False, **kw)
    except (Exception, SystemExit) as e:  # noqa: BLE001 — one bad month must not sink the pass
        # SystemExit, not just Exception: build_month_table raises it for an
        # empty panel and for a stale cache, and SystemExit is a
        # BaseException -- uncaught, it takes the worker down and the pool
        # hangs on a job that never returns.
        return f"{ym} a{anchors}: FAILED {type(e).__name__}: {e}"
    return (f"{ym} a{anchors}: {len(tbl['X'])} rows, "
            f"{len(set(tbl['cell'].tolist()))} cells, {time.time() - t0:.0f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--months", nargs="+", help="FIT months")
    g.add_argument("--months-from-k2ind", action="store_true")
    p.add_argument("--fit-span", type=int, default=1,
                   help="months in each fit pool, ending at the fit month. "
                        "Must match the --fit-span of the scoring pass")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--mosaic-dir", default=None)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--batch-size", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    from calculate_panel_ic import EVAL_ANCHORS, FIT_ANCHORS, k2ind_fit_months
    from panel_tables import fit_pool, next_month

    months = k2ind_fit_months() if args.months_from_k2ind else args.months
    kw = dict(mosaic_dir=args.mosaic_dir,
              xs_anchor_stats_dir=args.xs_anchor_stats_dir,
              batch_size=args.batch_size, force=args.force)

    # Deduplicated, because overlapping pools ask for the same table many
    # times and _build's "cached" check only fires once the first copy has
    # FINISHED -- two workers starting the same month together would both
    # decode it and then race on the rename.
    seen: set[tuple[str, int]] = set()
    jobs = []
    for ym in months:
        for fm in fit_pool(ym, args.fit_span, args.mosaic_dir):
            if (fm, FIT_ANCHORS) not in seen:
                seen.add((fm, FIT_ANCHORS))
                jobs.append((fm, FIT_ANCHORS, kw))
        ev = next_month(ym)
        if (ev, EVAL_ANCHORS) not in seen:
            seen.add((ev, EVAL_ANCHORS))
            jobs.append((ev, EVAL_ANCHORS, kw))
    # Largest first: the 36-anchor tables dominate, so starting them together
    # keeps the tail from being one lone fit month after every eval is done.
    jobs.sort(key=lambda j: -j[1])

    print(f"{len(jobs)} tables, {args.workers} workers")
    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(_build, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {msg}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
