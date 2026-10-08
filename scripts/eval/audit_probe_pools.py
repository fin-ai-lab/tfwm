"""Catch probe-breadth results that were fit on a short pool.

WHAT THIS DEFENDS AGAINST. probe_fit_size.cmd_reduce validates the shard count
of the EVAL panel and nothing else; the FIT pool is whatever _load_group finds
in the cache dir. Fit shards lost to ENOSPC therefore produce a
complete-looking result at a smaller ``n``, with nothing raising -- and because
a frozen encoder plus ridge is a reservoir predictor whose IC rises
monotonically with fit size, a short pool does not look broken. It looks like
a slightly worse model.

THE TEST IS EXACT, NOT STATISTICAL. Every arm sharing an eval month also
shares its six fit months (checked below), the same anchor grid and the same
panel, so their pool sizes must be IDENTICAL integers. Any disagreement inside
a month is lost shards, never sampling. That makes this a clean invariant
rather than a threshold someone has to tune.

Found on 2026-09-16: the pbfloor wave ran on the L40S partition, whose node-local scratch is
far smaller than the H100 partition's. Each job stages ~7 months of mosaic at
~75 GB/month, so the node filled and four eval months came back under-fit --
one of them at 85,257 rows against the month's true 2,424,861. That floor is
the bar every cell of probe_fit_table.tex is struck against, so a depressed
floor silently flatters every method in the paper.

    uv run python scripts/eval/audit_probe_pools.py
    uv run python scripts/eval/audit_probe_pools.py --quiet   # only problems
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("lab/probe_breadth")
MANIFESTS = ["manifest_all.tsv", "manifest_floor.tsv", "manifest_tsfm.tsv"]
INDEXES = ["manifest_all.index.json", "manifest_floor.index.json",
           "manifest_tsfm.index.json"]
MONTH = re.compile(r"(\d{4}-\d{2})(?=\.json$)")


def check_fit_months(root: Path, out) -> bool:
    """The premise: one eval month means one fit-month set, for every arm."""
    fits: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for mf in MANIFESTS:
        p = root / mf
        if not p.is_file():
            continue
        for ln in p.read_text().splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            ck, fit, ev = ln.split("\t")[:3]
            fits[ev][Path(ck).name].add(fit)
    bad = [ev for ev, d in fits.items() if len({frozenset(v) for v in d.values()}) != 1]
    if bad:
        print(f"  PREMISE BROKEN: {len(bad)} eval month(s) whose arms do NOT "
              f"share one fit-month set ({', '.join(sorted(bad)[:5])}). Pool "
              f"sizes are legitimately allowed to differ there; this audit "
              f"does not apply to them.", file=out)
    return not bad


def load_pools(root: Path, alpha: float):
    idx = {}
    for f in INDEXES:
        p = root / f
        if p.is_file():
            for r in json.loads(p.read_text()):
                idx[(r["run_id"], r["eval_month"])] = r["series_key"]
    # KEYED BY EVAL MONTH ALONE, NOT BY TAG. A re-run lands under a new tag
    # (pbfix-*, pbfl2-*), and probe_fit_table's load_results takes the largest
    # n per (checkpoint, eval month) across EVERY file -- so a good re-run
    # supersedes a bad original and the table is already correct. Keying this
    # audit by tag instead reported the superseded originals as live failures
    # forever, which trains you to ignore it. Merging also makes the check
    # STRONGER: the floor and the method arms of one month land in different
    # tags, and their pools must agree for the floor to be a like-for-like bar.
    pools: dict[str, dict[str, int]] = defaultdict(dict)
    waves: dict[str, set] = defaultdict(set)
    for f in sorted((root / "results").glob("*.json")):
        m = MONTH.search(f.name)
        if not m:
            continue
        ym = m.group(1)
        wave = f.name.split("-part")[0]
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:            # still being written
            continue
        for row in rows:
            if abs(float(row.get("alpha", -1)) - alpha) > 1e-9:
                continue
            s = idx.get((row["ckpt"], ym))
            if s is None:
                continue
            d = pools[ym]
            n = int(row["n"])
            if n >= d.get(s, 0):
                d[s] = n
            waves[ym].add(wave)
    return pools, waves


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--tol", type=float, default=0.98,
                   help="an arm is short below this fraction of the month's "
                        "modal pool; the invariant is exact equality, so the "
                        "default only absorbs a rounding-sized difference")
    p.add_argument("--quiet", action="store_true", help="print only problems")
    a = p.parse_args()

    root = Path(a.root)
    check_fit_months(root, sys.stdout)
    pools, waves = load_pools(root, a.alpha)
    if not pools:
        print("no results to audit")
        return 0

    bad = []
    if not a.quiet:
        print(f"{'month':9s} {'arms':>5s} {'modal pool':>12s} "
              f"{'min':>12s} {'max/min':>8s}  tags  short arms")
    for ym, d in sorted(pools.items()):
        ns = list(d.values())
        modal = Counter(ns).most_common(1)[0][0]
        short = sorted(s for s, n in d.items() if n < modal * a.tol)
        if short:
            bad.append((",".join(sorted(waves[ym])), ym, modal, short, d))
        if not a.quiet:
            tail = "  <-- " + ", ".join(short) if short else ""
            print(f"{ym:9s} {len(ns):>5d} {modal:>12,d} "
                  f"{min(ns):>12,d} {max(ns)/min(ns):>7.2f}x  "
                  f"{len(waves[ym])}{tail}")

    print()
    if not bad:
        print(f"CLEAN: every arm in all {len(pools)} eval month(s) was fit "
              f"on that month's full pool.")
        return 0
    print(f"*** {len(bad)} of {len(pools)} eval month(s) HAVE A SHORT ARM ***")
    for wave, ym, modal, short, d in bad:
        print(f"  {wave} {ym}: modal {modal:,d}")
        for s in short:
            print(f"      {s:16s} {d[s]:>12,d}  ({d[s]/modal:.1%} of the pool)")
    print("\nRe-run those eval months on nodes whose scratch can hold "
          "~7 staged months:")
    print("  PARTITION=<h100-partition> SBATCH_EXTRA='--exclude=<bad-nodes>' \\")
    print("    <your run_probe_breadth launcher> "
          "--manifest <subset>.tsv --tag pbfix")
    print("The table takes the largest n per (ckpt, eval month), so a good "
          "result supersedes a bad one and the stale json can be left alone.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
