"""RankMe for exactly the arms probe_fit_table.tex displays, mean +/- SE.

One row per table row, in the table's order and under its labels, so the two
can be read side by side. Aggregation follows the same two-stage rule
floor_row() uses in probe_fit_table: for the untrained floor, average the
THREE SEEDS WITHIN A MONTH first and only then across months -- a seed is a
draw from the same untrained architecture, so averaging seeds first keeps one
seed's noise out of the month-to-month spread the SE is computed from.

SE IS OVER MONTHS, which is the only axis that repeats here. A single-month
arm gets no SE, printed as "--", because one reading has no spread to report
and a zero would read as precision.

    uv run python plots/core/rankme_table.py --json <rankme.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "plots/core"))

FLOOR_LABEL = "Random ViT"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, help="output of scripts/eval/rankme.py")
    p.add_argument("--metric", default="rankme",
                   choices=("rankme", "cov_effrank"))
    a = p.parse_args()

    # ONE IMPLEMENTATION OF THE SEEDS-THEN-MONTHS RULE. probe_fit_table's
    # RankMe column reads the same json, and two copies of this aggregation
    # would be two things to keep in step; load_rankme() is the one.
    from probe_fit_table import load_rankme, rankme_for, roster

    by = load_rankme(a.json, metric=a.metric)

    def agg(key):
        got = by.get(key)
        # (mu, se, n, extra, widths) -- the extra slot is vestigial here and
        # kept so the printing below is untouched.
        return None if got is None else (got[0], got[1], got[2], None, got[3])

    out = []
    for key, label, fam, _native, _cite in roster():
        got = rankme_for(by, key)
        out.append((fam, label,
                    (got[0], got[1], got[2], None, got[3]) if got else None))
    out.append(("Untrained floor", FLOOR_LABEL, agg(FLOOR_LABEL)))

    # RANK ACROSS THE WHOLE TABLE, shown in ROSTER order. Printing the rows
    # sorted by the metric would break the correspondence with
    # probe_fit_table.tex, which is the entire point of this view; a rank
    # column carries the ordering without moving anything.
    ranked = sorted((r[0], lab) for _f, lab, r in out if r)
    place = {lab: i for i, (_v, lab) in enumerate(reversed(ranked), 1)}

    print(f"metric: {a.metric}   ({len(ranked)} arms ranked, 1 = widest "
          f"spectrum)\n")
    print(f"{'':32s} {'#':>3s} {'d':>4s} {'months':>6s} {'mean':>8s} "
          f"{'+/- SE':>8s}")

    fam_now = None
    for fam, label, r in out:
        if fam != fam_now:
            print(f"-- {fam}")
            fam_now = fam
        if r is None:
            print(f"   {label:29s} {'-':>3s} {'':>4} {0:>6d} {'--':>8s} "
                  f"{'--':>8s}")
            continue
        mu, se, n, _, dd = r
        d = str(dd[0]) if len(dd) == 1 else "*"
        se_s = f"{se:8.2f}" if se is not None else f"{'--':>8s}"
        print(f"   {label:29s} {place[label]:>3d} {d:>4s} {n:>6d} "
              f"{mu:>8.2f} {se_s}")

    miss = [l for f, l, r in out if r is None]
    if miss:
        print(f"\n{len(miss)} arm(s) with no staged embeddings: {', '.join(miss)}")
    thin = [(l, r[2]) for f, l, r in out if r and r[2] < 5]
    if thin:
        print("THIN COVERAGE (SE not meaningful): " +
              ", ".join(f"{l} n={n}" for l, n in thin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
