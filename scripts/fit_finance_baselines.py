#!/usr/bin/env python
"""Fit the classical finance baselines' parameters, pooled across assets, per month.

Produces the JSON artifact that ``mode=finance_baseline`` loads via
``mode.params_path``. One run writes one file holding a ``(month, agg_factor) ->
(alpha, beta, phi_return, phi_spread)`` table plus a pooled global fallback; see
:mod:`market_jepa.modeling.modes.finance_baseline_params` for what the
coefficients mean and why they are keyed that way.

The augmentation settings **must match the run being baselined**. Parameters are
estimated per ``agg_factor``, and the aggregation scale a sample draws is a
function of ``global_scale_range`` and ``global_seq_len``; fitting under one
crop configuration and evaluating under another leaves every lookup falling back
to a neighbouring cell.

Usage::

    uv run scripts/fit_finance_baselines.py \
        --date-start 2023-01-01 --date-end 2023-12-31 \
        --out /data/lab/market-jepa-checkpoints/finance_baselines/2023.json

Then point a baseline run at it::

    uv run train.py mode=finance_baseline \
        mode.params_path=/data/lab/.../2023.json
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stable_finance.dataset import MarketSchedule
from market_jepa.modeling.modes.finance_baseline_params import BaselineFitter
from market_jepa.modeling.modes.finance_baselines import DEFAULT_HORIZONS as HORIZONS
from market_jepa.schemas import BLL01MachineConfig
from market_jepa.training.streaming_dataset import (
    StreamingMarketDataset,
    discover_streams,
)
from market_jepa.training.utils import collate_bucketed

logger = logging.getLogger("fit_finance_baselines")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--months", nargs="+", metavar="YYYY-MM",
        help="Explicit months to calibrate on. Use this for the sampled sweep "
             "months, which are deliberately discontiguous — a date range "
             "would drag in every month between them.",
    )
    src.add_argument("--date-start", help="Calibration start, YYYY-MM-DD")
    p.add_argument("--date-end", help="Calibration end, YYYY-MM-DD")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-batches", type=int, default=400,
        help="Stop after this many batches. The AR(1) statistics keep "
             "accumulating over the whole stream, so this mostly bounds wall "
             "clock, not accuracy.",
    )
    p.add_argument(
        "--max-series-per-cell", type=int, default=512,
        help="Return series retained per (month, agg) cell for the GARCH "
             "likelihood. Bounds memory; see BaselineFitter.",
    )

    aug = p.add_argument_group("augmentation (must match the run being baselined)")
    aug.add_argument("--n-global-views", type=int, default=2)
    aug.add_argument("--n-local-views", type=int, default=0)
    aug.add_argument("--global-seq-len", type=int, default=2048)
    aug.add_argument(
        "--global-scale-range", type=float, nargs=2, default=[0.5, 1.0],
        metavar=("LO", "HI"),
    )
    aug.add_argument(
        "--global-agg-range", type=int, nargs=2, default=None,
        metavar=("LO", "HI"),
        help="Absolute seconds-per-token band. Set this only if the run being "
             "baselined sets it (extended-hours training does).",
    )
    aug.add_argument(
        "--extended-hours", action="store_true",
        help="Build the 1 Hz grid over the extended session, as the run does.",
    )
    args = p.parse_args(argv)
    if args.date_start and not args.date_end:
        p.error("--date-start requires --date-end")
    return args


def month_bounds(month: str) -> tuple[str, str]:
    """``"2020-07"`` -> ``("2020-07-01", "2020-07-31")``."""
    year, mon = (int(x) for x in month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    return f"{month}-01", f"{month}-{last:02d}"


def build_loader(
    args: argparse.Namespace, date_start: str, date_end: str,
) -> DataLoader:
    machine = BLL01MachineConfig()
    schedule = MarketSchedule(machine.holiday_csv)
    streams = discover_streams(machine.mosaic_dir, date_start, date_end)

    augmentation = {
        "name": "random_resized_crop",
        "n_global_views": args.n_global_views,
        "n_local_views": args.n_local_views,
        "global_seq_len": args.global_seq_len,
        "global_scale_range": list(args.global_scale_range),
    }
    if args.global_agg_range is not None:
        augmentation["global_agg_range"] = list(args.global_agg_range)

    dataset = StreamingMarketDataset(
        augmentations=[augmentation],
        date_start=date_start,
        date_end=date_end,
        seed=args.seed,
        n_pairs_per_obs=1,
        # NINE CHANNELS, DECLARED. These read off the constructor default until
        # it flipped False -> True on 2026-09-13; pinning them keeps this fit
        # on the nine columns every published coefficient here was calibrated
        # on, and makes that a statement rather than an inheritance.
        #
        # THE CLASSICAL BASELINES DO GET THE INFORMATION TOKEN -- just not here.
        # It reaches them in panel_tables.iter_view_blocks, which requests it
        # from panel_source and then splits it off with _split_info: the nine
        # channels flatten over the tail and the eleven are appended ONCE, as
        # the per-window constants they are. This file is the other half, the
        # hand-built lag/vol/spread featurizers, and a constant column there is
        # a zero difference and a zero variance -- a wasted regressor, not a
        # free one. The AR(1)/GARCH fits take the view as a SERIES, and the
        # eleven are written to the final step only (training.utils
        # .append_view_info), so feeding them in would be eleven rows of zeros
        # with one live element, not the information the token carries.
        info_norm_stats=False,
        info_window=False,
        # The per-horizon AR(1) coefficients are fit as a predictive regression
        # of the next-h return / spread-change on the trailing-h predictor, so
        # calibration needs those targets alongside the input view. (GARCH is fit
        # on the view's own returns and needs no target.)
        targets={"horizons": list(HORIZONS), "types": ["return", "spread_change"]},
        schedule=schedule,
        streams=streams,
        shuffle=False,
        batch_size=args.batch_size,
        allow_unsafe_types=True,
        predownload=max(args.batch_size, 64),
        risk_factor_dir=machine.risk_factor_dir,
        risk_factor_tickers=[],
        risk_factor_columns=None,
        extended_hours=args.extended_hours,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_bucketed,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)

    if args.months:
        windows = [(m, *month_bounds(m)) for m in args.months]
    else:
        windows = [(f"{args.date_start}..{args.date_end}", args.date_start, args.date_end)]

    fitter = BaselineFitter(
        horizons=HORIZONS, max_series_per_cell=args.max_series_per_cell,
    )
    seen_cells: Counter = Counter()
    n_samples = 0

    # One loader per window. Each month is an independent cell anyway, so this
    # costs nothing over a single stream and keeps discontiguous months from
    # dragging in everything between them.
    with torch.no_grad():
        for label, start, end in windows:
            loader = build_loader(args, start, end)
            bar = tqdm(loader, total=args.max_batches, desc=f"calibrate {label}")
            for i, batch in enumerate(bar):
                if i >= args.max_batches:
                    break
                for bucket in batch["buckets"]:
                    fitter.add_bucket(bucket)
                    if "dates" in bucket and "agg_factors" in bucket:
                        for d, a in zip(bucket["dates"], bucket["agg_factors"].tolist()):
                            seen_cells[(d[:7], int(a))] += 1
                            n_samples += 1
            bar.close()

    if not n_samples:
        logger.error(
            "No samples with date/agg metadata were collected — nothing to fit. "
            "Check the date range and that the streams resolved."
        )
        return 1

    logger.info(
        "Collected %d samples across %d (month, agg) cells", n_samples, len(seen_cells),
    )
    thin = sorted(c for c, n in seen_cells.items() if n < 50)
    if thin:
        logger.warning(
            "%d cells have fewer than 50 samples and will be noisy: %s",
            len(thin), thin[:10],
        )

    params = fitter.fit()
    params.save(args.out)
    logger.info("Wrote %r to %s", params, args.out)

    # Print the table so a calibration run is self-documenting in the log. Show
    # the AR coefficients at the shortest and longest horizon to see the term
    # structure without dumping all six.
    h_lo, h_hi = HORIZONS[0], HORIZONS[-1]
    for (month, agg), p in sorted(params.cells.items()):
        logger.info(
            "  %s agg=%2d  alpha=%.4f beta=%.4f (persist=%.5f)  "
            "phi_ret[%d/%d]=%+.3f/%+.3f  phi_spr[%d/%d]=%+.3f/%+.3f  n=%d",
            month, agg, p.alpha, p.beta, p.persistence,
            h_lo, h_hi, p.phi_return(h_lo), p.phi_return(h_hi),
            h_lo, h_hi, p.phi_spread(h_lo), p.phi_spread(h_hi),
            p.n_series,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
