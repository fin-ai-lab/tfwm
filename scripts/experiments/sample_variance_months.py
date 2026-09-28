# /// script
# requires-python = ">=3.10"
# ///
"""Deterministically sample the 10 months used for the variance-decomposition sweep.

The variance-decomposition experiment (scripts/sweeps/variance_decomp.sh) trains
10 seeds x {lejepa, supervised_*} on each of 10 training months. The months are
a seeded subsample of the frozen 32-month sweep list in sample_sweep_months.py,
so every month here already has per-month sweep results (lambda grid, finance
baselines, ...) to compare against.

TWO DRAWS, because the first one can name a month that cannot be trained. The
recipe trains TRAIN_SPAN_MONTHS of data ENDING at the training month, so a
month closer to the start of the panel than that has no span behind it: the
original draw picked 2008-02, whose six-month span reaches 2007-09, four months
before the mosaic begins. Its jobs staged, failed the rsync and died before
training (223084-86, 2026-09-13) and the launcher now drops it up front.

Dropping is the only safe response to that at LAUNCH time -- a shorter span for
one month would measure the seed spread of a model nothing else in the paper is
-- but it leaves the panel at nine months, which reads as a gap rather than as
a design. So the shortfall is made up HERE, by drawing a replacement from the
sweep months that are eligible and not already in the sample. The replacement
is as deterministic as the first draw and audited by the same --check.

THE SPAN IS FROZEN AS A NUMBER, not read from DatasetConfig: this is a PEP 723
script run in its own env, so it cannot import market_jepa. If the recipe's
span changes, eligibility changes with it and --check will say so -- which is
the intended alarm, not a bug.

Last verified run (seed=42, sample of 10 from SWEEP_MONTHS, 2008-02 replaced
by 2021-11):

    2008-09  2009-11  2010-05  2010-12  2018-10
    2019-12  2021-08  2021-11  2022-06  2023-10

Usage:
    uv run scripts/experiments/sample_variance_months.py           # one YYYY-MM per line
    uv run scripts/experiments/sample_variance_months.py --check   # fail on drift
"""

from __future__ import annotations

import argparse
import random
import sys

from sample_sweep_months import SWEEP_MONTHS

SEED = 42
N_MONTHS = 10

# The first month the mosaic holds, and the span the recipe trains. Together
# they say which months are trainable at all: a month before
# DATA_START + (TRAIN_SPAN_MONTHS - 1) has no span behind it. Mirrors
# DatasetConfig.train_span_months, which this script cannot import -- see the
# module docstring.
DATA_START_MONTH = "2008-01"
TRAIN_SPAN_MONTHS = 6

VD_MONTHS: list[str] = [
    "2008-09", "2009-11", "2010-05", "2010-12", "2018-10",
    "2019-12", "2021-08", "2021-11", "2022-06", "2023-10",
]


def span_floor() -> str:
    """The earliest month with a full training span behind it."""
    y, m = int(DATA_START_MONTH[:4]), int(DATA_START_MONTH[5:7])
    m += TRAIN_SPAN_MONTHS - 1
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def raw_sample() -> list[str]:
    """The original 10-month draw, before any month is replaced."""
    return sorted(random.Random(SEED).sample(SWEEP_MONTHS, N_MONTHS))


def sample() -> list[str]:
    """The draw with untrainable months replaced — what the sweep actually runs.

    The replacement pool is every OTHER eligible sweep month, so a replacement
    inherits the same per-month sweep results the first draw was chosen for.
    Drawn with the same seed off a pool that is itself a function of the first
    draw, which is enough to make the pair reproducible without the two draws
    being able to collide.
    """
    floor = span_floor()
    picked = raw_sample()
    keep = [m for m in picked if m >= floor]
    n_short = len(picked) - len(keep)
    if not n_short:
        return keep
    pool = [m for m in SWEEP_MONTHS if m >= floor and m not in keep]
    return sorted(keep + random.Random(SEED).sample(pool, n_short))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if the sampled list differs from VD_MONTHS.")
    args = p.parse_args()

    sampled = sample()

    if args.check:
        if sampled != VD_MONTHS:
            dropped = [m for m in raw_sample() if m < span_floor()]
            if dropped:
                print(f"  (note: {' '.join(dropped)} dropped as untrainable "
                      f"under a {TRAIN_SPAN_MONTHS}-month span; replacements "
                      f"redrawn)", file=sys.stderr)
            print("DRIFT: sampled months differ from VD_MONTHS.", file=sys.stderr)
            print(f"  sampled: {sampled}", file=sys.stderr)
            print(f"  frozen:  {VD_MONTHS}", file=sys.stderr)
            return 1
        print("OK: sampled months match VD_MONTHS.")
        return 0

    for m in sampled:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
