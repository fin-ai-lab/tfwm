"""Trained-head IC for the supervised arms of the probe-breadth sweep.

WHY NOT supervised_head_ic.json. That artifact carries one record per (month,
target) and no arm, which was unambiguous while it held only the three
SPECIALISTS -- one head each, so the target named the arm. The MULTIHEAD has a
head on all three targets, so a consumer reading that file by target alone
hands the multihead the specialists' numbers: on 2013-01 the multihead's "head"
column read 0.0302/0.1027/0.1715, the specialists' values exactly. This writes
the arm on every record so that cannot happen.

THE CHECKPOINTS ARE THE SWEEP'S OWN. Run ids come from the probe-breadth
manifest index, so each head IC is read off THE SAME checkpoint whose probe IC
sits beside it in the table -- same wave, same commit, same eval month. A
second selection rule here (a glob, a commit pin, a recipe filter) could drift
from the sweep's and put two different models in one row under one name.

THE HEAD IS READ THROUGH ``ICRun.head``, not out of the raw ic dict. A
multihead checkpoint whose heads were written by a loader that never loaded the
head weights reports an UNTRAINED head -- that is what once put a multihead
below the zero line on the Return panel -- and ``ICRun.head`` refuses one that
lacks ``xs_head_schema`` rather than returning the noise. Reading
``run.ic["xs_ic/head:<task>"]`` directly would bypass exactly that guard.

Run::

    uv run python plots/metrics/build_probe_head_ic.py
    ... --index lab/probe_breadth/manifest_all.index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))

INDEX = Path("lab/probe_breadth/manifest_all.index.json")
OUT = Path(__file__).resolve().parent / "probe_head_ic.json"
CKPT_ROOT = Path("lab/market-jepa-checkpoints")

# The supervised arms and the targets each one's head was trained on. A
# specialist has one; the multihead has all three, which is the whole reason
# this file records the arm.
ARM_TASKS = {
    "sup_return_w8": ["return_900"],
    "sup_vol_w8": ["volatility_change_900"],
    "sup_spread_w8": ["spread_change_900"],
    "sup_multi_w8": ["return_900", "volatility_change_900",
                     "spread_change_900"],
}
# One glob per arm, so only the relevant tree is walked.
ARM_GLOB = {
    "sup_return_w8": "supervised-full-month-return-*",
    "sup_vol_w8": "supervised-full-month-vol-change-*",
    "sup_spread_w8": "supervised-full-month-spread-change-*",
    "sup_multi_w8": "supervised-full-month-multihead-*",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=str(INDEX))
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    p.add_argument("--xs-stats", default="xs_anchor_stats_fwdvwap60")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    from style import iter_ic_runs

    idx = json.loads(Path(a.index).read_text())
    want = {}                       # series_key -> {run_id: eval_month}
    for r in idx:
        if r["series_key"] in ARM_TASKS:
            want.setdefault(r["series_key"], {})[r["run_id"]] = r["eval_month"]
    if not want:
        raise SystemExit(f"no supervised arms in {a.index}")

    recs, missing = [], Counter()
    for series, runs in sorted(want.items()):
        found = 0
        for run in iter_ic_runs(ARM_GLOB[series], Path(a.ckpt_root),
                                xs_stats=a.xs_stats):
            if run.run_id not in runs:
                continue
            found += 1
            ev = runs[run.run_id]
            if run.eval_month != ev:
                # The index and the run disagree about which month this is.
                # Refuse rather than file the number under the wrong month.
                raise SystemExit(
                    f"{series} {run.run_id}: index says eval {ev}, run says "
                    f"{run.eval_month}")
            for task in ARM_TASKS[series]:
                v = run.head(task)
                if v is None:
                    missing[f"{series}:{task}"] += 1
                    continue
                recs.append({"series_key": series, "eval_month": ev,
                             "target": task, "ic": float(v),
                             "model": "probe_head", "run_id": run.run_id,
                             "xs_anchor_stats": a.xs_stats})
        print(f"  {series:16s} {found:3d}/{len(runs):3d} run(s) matched",
              file=sys.stderr)

    if missing:
        print(f"  no head IC for: {dict(missing)}", file=sys.stderr)
    if not recs:
        raise SystemExit("no head ICs found -- wrong ckpt-root or xs-stats?")

    Path(a.out).write_text(json.dumps(recs, indent=1))
    months = {r["eval_month"] for r in recs}
    print(f"wrote {len(recs)} record(s) over {len(months)} month(s) "
          f"to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
