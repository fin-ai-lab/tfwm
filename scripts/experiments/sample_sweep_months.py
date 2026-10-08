# /// script
# requires-python = ">=3.10"
# ///
"""Deterministically sample the 32 months used for the cross-month sweep.

We sweep each pipeline (scripts/sweeps/*.sh) across 32 randomly chosen
training months drawn from 2008-01 through 2023-12 inclusive. We stop at 2023-12
(not the end of available 2024 data) so there is a full year of forward data for
evaluation targets.

The seed is fixed (42). If this script is rerun and the printed months differ
from the list embedded at the top of this file (and exposed as SWEEP_MONTHS),
something drifted — either the Python `random` implementation changed or the
candidate range was modified. In either case, audit before updating consumers.

TWO LISTS, because the first draw names a month that cannot be trained. The
recipe trains TRAIN_SPAN_MONTHS of data ENDING at the training month, and
2008-02's six-month span reaches 2007-09, before the mosaic begins. For two
weeks that month was simply dropped everywhere (metrics.SUP_SPAN_NO_SPAN), so
the paper reported 31 months while calling the panel 32.

  ORIGINAL_DRAW  the 32 months exactly as drawn. FROZEN FOREVER: two other
                 draws were made against it -- the variance-decomposition 10
                 (sample_variance_months.py samples FROM it) and holdout set 2
                 (holdout_months.py excludes it) -- and both reproduce only if
                 it stays the list they saw.
  SWEEP_MONTHS   the reported panel: ORIGINAL_DRAW with every month that has
                 no span behind it replaced by the NEXT month the same seeded
                 draw produces that is trainable and not already in the panel.
                 ``random.sample`` draws sequentially from the same stream, so
                 ``sample(candidates, 33)`` begins with ``sample(candidates, 32)``
                 and its 33rd element is literally the month this sampler would
                 have picked next. For 2008-02 that is 2022-12 (eval 2023-01).

Last verified run (seed=42, range 2008-01..2023-12, n=32, 2008-02 -> 2022-12):

    2008-07  2008-08  2008-09  2009-11
    2009-12  2010-03  2010-05  2010-12
    2011-05  2012-03  2012-08  2012-09
    2012-10  2012-12  2013-03  2013-11
    2013-12  2016-12  2017-01  2017-07
    2018-10  2019-08  2019-12  2020-07
    2020-08  2020-11  2021-08  2021-11
    2022-06  2022-12  2023-03  2023-10

THE SPAN IS FROZEN AS A NUMBER, not read from DatasetConfig: this is a PEP 723
script run in its own env, so it cannot import market_jepa. If the recipe's
span changes, eligibility changes with it and --check will say so.

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

# The first month the mosaic holds, and the span the recipe trains. Mirrors
# sample_variance_months.py and DatasetConfig.train_span_months.
DATA_START_MONTH = "2008-01"
TRAIN_SPAN_MONTHS = 6

ORIGINAL_DRAW: list[str] = [
    "2008-02", "2008-07", "2008-08", "2008-09",
    "2009-11", "2009-12", "2010-03", "2010-05",
    "2010-12", "2011-05", "2012-03", "2012-08",
    "2012-09", "2012-10", "2012-12", "2013-03",
    "2013-11", "2013-12", "2016-12", "2017-01",
    "2017-07", "2018-10", "2019-08", "2019-12",
    "2020-07", "2020-08", "2020-11", "2021-08",
    "2021-11", "2022-06", "2023-03", "2023-10",
]

SWEEP_MONTHS: list[str] = [
    "2008-07", "2008-08", "2008-09", "2009-11",
    "2009-12", "2010-03", "2010-05", "2010-12",
    "2011-05", "2012-03", "2012-08", "2012-09",
    "2012-10", "2012-12", "2013-03", "2013-11",
    "2013-12", "2016-12", "2017-01", "2017-07",
    "2018-10", "2019-08", "2019-12", "2020-07",
    "2020-08", "2020-11", "2021-08", "2021-11",
    "2022-06", "2022-12", "2023-03", "2023-10",
]


def span_floor() -> str:
    """The earliest month with a full training span behind it."""
    y, m = int(DATA_START_MONTH[:4]), int(DATA_START_MONTH[5:7])
    m += TRAIN_SPAN_MONTHS - 1
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def _candidates() -> list[tuple[int, int]]:
    return [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]


def sample_original() -> list[str]:
    """The first draw, before any month is replaced."""
    picks = sorted(random.Random(SEED).sample(_candidates(), N_MONTHS))
    return [f"{y}-{m:02d}" for y, m in picks]


def sample() -> list[str]:
    """The reported panel: the first draw with untrainable months replaced by
    the stream's next trainable draws."""
    floor = span_floor()
    cands = _candidates()
    stream = [f"{y}-{m:02d}" for y, m in random.Random(SEED).sample(cands, len(cands))]
    keep = [m for m in stream[:N_MONTHS] if m >= floor]
    for m in stream[N_MONTHS:]:
        if len(keep) == N_MONTHS:
            break
        if m >= floor:
            keep.append(m)
    return sorted(keep)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if the sampled list differs from SWEEP_MONTHS.")
    args = p.parse_args()

    sampled = sample()

    if args.check:
        ok = True
        if sample_original() != ORIGINAL_DRAW:
            print("DRIFT: first draw differs from ORIGINAL_DRAW.", file=sys.stderr)
            print(f"  sampled: {sample_original()}", file=sys.stderr)
            print(f"  frozen:  {ORIGINAL_DRAW}", file=sys.stderr)
            ok = False
        if sampled != SWEEP_MONTHS:
            print("DRIFT: sampled months differ from SWEEP_MONTHS.", file=sys.stderr)
            print(f"  sampled: {sampled}", file=sys.stderr)
            print(f"  frozen:  {SWEEP_MONTHS}", file=sys.stderr)
            ok = False
        if not ok:
            return 1
        print("OK: sampled months match SWEEP_MONTHS (and ORIGINAL_DRAW).")
        return 0

    for m in sampled:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
