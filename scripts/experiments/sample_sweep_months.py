# /// script
# requires-python = ">=3.10"
# ///
"""Deterministically sample the 32 months used for the cross-month sweep.

We sweep each pipeline (scripts/pythia/sweeps/*.sh) across 32 randomly chosen
training months drawn from 2008-01 through 2023-12 inclusive. We stop at 2023-12
(not the end of available 2024 data) so there is a full year of forward data for
evaluation targets.

The seed is fixed (42). If this script is rerun and the printed months differ
from the list embedded at the top of this file (and exposed as SWEEP_MONTHS),
something drifted — either the Python `random` implementation changed or the
candidate range was modified. In either case, audit before updating consumers.

Last verified run (seed=42, range 2008-01..2023-12, n=32):

    2008-02  2008-07  2008-08  2008-09
    2009-11  2009-12  2010-03  2010-05
    2010-12  2011-05  2012-03  2012-08
    2012-09  2012-10  2012-12  2013-03
    2013-11  2013-12  2016-12  2017-01
    2017-07  2018-10  2019-08  2019-12
    2020-07  2020-08  2020-11  2021-08
    2021-11  2022-06  2023-03  2023-10

Usage:
    uv run scripts/experiments/sample_sweep_months.py           # prints one YYYY-MM per line
    uv run scripts/experiments/sample_sweep_months.py --check   # fail if drift from SWEEP_MONTHS
"""

from __future__ import annotations

import argparse
import random
import sys

SEED = 42
START_YEAR = 2008
END_YEAR = 2023  # inclusive; leaves 2024 for one-year-forward eval
N_MONTHS = 32

SWEEP_MONTHS: list[str] = [
    "2008-02", "2008-07", "2008-08", "2008-09",
    "2009-11", "2009-12", "2010-03", "2010-05",
    "2010-12", "2011-05", "2012-03", "2012-08",
    "2012-09", "2012-10", "2012-12", "2013-03",
    "2013-11", "2013-12", "2016-12", "2017-01",
    "2017-07", "2018-10", "2019-08", "2019-12",
    "2020-07", "2020-08", "2020-11", "2021-08",
    "2021-11", "2022-06", "2023-03", "2023-10",
]


def sample() -> list[str]:
    r = random.Random(SEED)
    candidates = [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]
    picks = sorted(r.sample(candidates, N_MONTHS))
    return [f"{y}-{m:02d}" for y, m in picks]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if the sampled list differs from SWEEP_MONTHS.")
    args = p.parse_args()

    sampled = sample()

    if args.check:
        if sampled != SWEEP_MONTHS:
            print("DRIFT: sampled months differ from SWEEP_MONTHS.", file=sys.stderr)
            print(f"  sampled: {sampled}", file=sys.stderr)
            print(f"  frozen:  {SWEEP_MONTHS}", file=sys.stderr)
            return 1
        print("OK: sampled months match SWEEP_MONTHS.")
        return 0

    for m in sampled:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
