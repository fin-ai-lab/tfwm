# NO PEP 723 BLOCK, DELIBERATELY. An inline-metadata header makes `uv run
# holdout_months.py` build an ISOLATED environment from the declared
# dependencies -- and this module imports stable_finance.dataset, a project
# dependency that such a block cannot name (it is a local/git dep, not a PyPI
# one). With the header, every bare `uv run scripts/experiments/holdout_months.py`
# died on ModuleNotFoundError, which is how run_tsfm_layers.sh came to resolve
# its default month panel to the EMPTY LIST and submit nothing. Without it,
# `uv run` uses the project environment, where stable_finance is installed.
# sample_sweep_months.py survived the same header only because it is
# stdlib-only at module scope.
"""The 5-month HPO / optimization set — disjoint from the 32 reported months.

Any hyper-parameter chosen on the reported 32 is chosen on the panel the paper
scores, which is how a sweep quietly turns into a result. These five months
exist so that tuning happens somewhere else:

    2009-01  2011-03  2014-05  2020-05  2022-02

They are TRAINING months. Each run trains on the month and evaluates on the
NEXT one (2009-01 -> 2009-02, and so on), the same train/eval convention as
``sample_sweep_months.py``.

DISJOINT IN BOTH DIRECTIONS, which ``--check`` verifies: no holdout training
month is one of the 32, no holdout EVAL month is one of the 32, and neither
collides with the 32's own eval months. So nothing tuned here is tuned on a
panel the paper reports, in either role.

Mar-2020 is deliberately absent. The supervised head's per-seed bin collapse
shows up on that panel, and one such month in five would drive a ranking
rather than inform it.

FROZEN, NOT DERIVED. The list was drawn once (a fixed-seed sample over
2008-01..2023-12 with the COVID panel excluded) and the exact procedure was not
recorded in a form that reproduces it — re-running plausible reconstructions
does not return these five. So this file is the authority: the months are data,
not the output of the code beside them, and ``--check`` asserts the property
that matters rather than pretending to re-derive the draw. Do not "fix" this by
regenerating; that would silently move the optimization set out from under
every result tuned on it.

Consumers:
    scripts/pythia/specific/run_supervised_bins_penalty.sh
    plots/supervised_bins_penalty/bins_penalty.py  (via the checkpoint tree)

Usage:
    uv run scripts/experiments/holdout_months.py           # one YYYY-MM per line
    uv run scripts/experiments/holdout_months.py --eval    # the eval months
    uv run scripts/experiments/holdout_months.py --check   # assert disjointness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_finance.dataset import next_month

HOLDOUT_MONTHS: list[str] = [
    "2009-01", "2011-03", "2014-05", "2020-05", "2022-02",
]

# Set 2, added 2026-08-19 for the soft-label-temperature arm. Drawn the way
# set 1 should have been: `sample_set_2()` below reproduces it exactly, and
# --check asserts that it still does. Disjoint from the reported 32 AND from
# set 1, in both roles.
HOLDOUT_MONTHS_2: list[str] = [
    "2009-06", "2011-12", "2015-06", "2016-03", "2021-03",
]

SET_2_SEED = 7

# Both sets, for callers that want the whole off-panel pool.
ALL_HOLDOUT_MONTHS: list[str] = sorted(HOLDOUT_MONTHS + HOLDOUT_MONTHS_2)


def sample_set_2() -> list[str]:
    """Redraw set 2. Candidates are every month 2008-01..2023-12 whose own
    month AND eval month avoid the reported 32, set 1, and the COVID panel."""
    import random

    from sample_sweep_months import SWEEP_MONTHS

    taken: set[str] = set()
    for m in list(SWEEP_MONTHS) + list(HOLDOUT_MONTHS):
        taken |= {m, next_month(m)}
    covid = {"2020-02", "2020-03"}
    cands = [
        f"{y}-{m:02d}"
        for y in range(2008, 2024)
        for m in range(1, 13)
        if f"{y}-{m:02d}" not in taken
        and next_month(f"{y}-{m:02d}") not in taken
        and f"{y}-{m:02d}" not in covid
        and next_month(f"{y}-{m:02d}") not in covid
    ]
    return sorted(random.Random(SET_2_SEED).sample(cands, 5))


def eval_months(months: list[str] | None = None) -> list[str]:
    return [next_month(m) for m in (HOLDOUT_MONTHS if months is None else months)]


def _sweep_months() -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sample_sweep_months import SWEEP_MONTHS
    return list(SWEEP_MONTHS)


def check() -> int:
    sweep = _sweep_months()
    reported = set(sweep) | {next_month(m) for m in sweep}
    one = set(HOLDOUT_MONTHS) | set(eval_months(HOLDOUT_MONTHS))
    two = set(HOLDOUT_MONTHS_2) | set(eval_months(HOLDOUT_MONTHS_2))

    for name, ours in (("set 1", one), ("set 2", two)):
        clash = sorted(ours & reported)
        if clash:
            print(f"{name} OVERLAPS the reported 32 (train or eval): {clash}",
                  file=sys.stderr)
            return 1
        if "2020-03" in ours:
            print(f"Mar-2020 is in {name}; see the docstring.", file=sys.stderr)
            return 1
    cross = sorted(one & two)
    if cross:
        print(f"set 1 and set 2 OVERLAP (train or eval): {cross}", file=sys.stderr)
        return 1
    if len(set(ALL_HOLDOUT_MONTHS)) != len(ALL_HOLDOUT_MONTHS):
        print("duplicate months across the two sets", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    redrawn = sample_set_2()
    if redrawn != HOLDOUT_MONTHS_2:
        print("DRIFT: set 2 no longer reproduces from its seed.", file=sys.stderr)
        print(f"  redrawn: {redrawn}", file=sys.stderr)
        print(f"  frozen:  {HOLDOUT_MONTHS_2}", file=sys.stderr)
        return 1

    print(f"OK: set 1 ({len(HOLDOUT_MONTHS)}) and set 2 ({len(HOLDOUT_MONTHS_2)}) "
          f"holdout months, {len(one | two)} months once eval months are counted, "
          f"disjoint from each other and from the {len(reported)} the reported "
          f"32 occupy. Set 2 reproduces from seed {SET_2_SEED}.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Assert disjointness from the reported 32.")
    p.add_argument("--eval", action="store_true",
                   help="Print eval months (train + 1) instead of training months.")
    p.add_argument("--set", choices=("1", "2", "all"), default="1",
                   help="Which optimization set (default 1).")
    args = p.parse_args()

    if args.check:
        return check()
    months = {"1": HOLDOUT_MONTHS, "2": HOLDOUT_MONTHS_2,
              "all": ALL_HOLDOUT_MONTHS}[args.set]
    for m in ([next_month(x) for x in months] if args.eval else months):
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
