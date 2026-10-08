"""How much of the change targets is mechanical, and how much survives the view?

REWRITTEN FOR THE FORWARD-WINDOW TARGETS (2026-08-22). What this script used to
measure no longer exists. The old definitions each subtracted a quantity KNOWN
AT t:

    volatility_change = fwd_vol[t, t+h) - bwd_vol[t-h, t)      RETIRED
    spread_change     = spread(t+h)     - spread(t)            RETIRED

and the backward leg lay inside the crop, so ``-bwd_vol`` and ``-spread(t)``
forecast them with no market model at all -- rank IC ~0.32 and ~0.18 on the
view. Both targets are now differences of TWO FORWARD 60 s windows:

    volatility_change = rv[t+h, t+h+60) - rv[t, t+60)
    spread_change     = spr[t+h, t+h+60) - spr[t, t+60)

There is no backward leg left to read. The near leg is unknowable at t, so
nothing computed at t can be correlated with its measurement error, and this
baseline stops being mechanical and becomes an ordinary weak forecast. Measured
on the holdout months, mean reversion on volatility change fell +0.2243 ->
+0.0242 when the target changed. That 9x is the leak leaving.

WHAT IS COMPUTED NOW. The causal, window-MATCHED analogue of the target's near
leg: the same 60 s window shape, ending at t instead of starting there.

    volatility_change  ->  -rv over [t-60, t)
    spread_change      ->  -mean spread over [t-60, t)

Two consequences worth stating.

  THE PREDICTOR NO LONGER DEPENDS ON h. The target's near leg is the same 60 s
  window at every horizon; only the far leg moves. So one number per view
  serves all six horizons, and the IC's horizon profile now comes entirely
  from the target rather than half from the predictor's own widening window.

  IT IS READ OFF VWAP, NOT THE MIDPOINT. The midpoint at a single instant is
  the quantity the target was moved away from -- a quote at an extreme of its
  recent range reverts inside the spread, and that reversion used to read as a
  real return. Realized vol built from diff(mid) inherits the same quote
  flicker. The vwap channel is a traded price, so it is the honest input.
  (A volume-WEIGHTED average is not recoverable here: normalize_numpy z-scores
  the volume group, so the view's volume channel is signed and cannot serve as
  a weight. The vwap channel itself is unaffected -- only the weighting is
  lost, which is why this is a plain window statistic.)

TWO VERSIONS, AND THE GAP BETWEEN THEM IS THE POINT.

  raw       computed from the dense 1 Hz grid, in real units. This is what a
            practitioner with the order book would use.
  view      computed from the VIEW THE ENCODER IS FED -- the same array
            xs_ic_eval.iter_panel hands the model, after aggregation, vwap
            ffill and normalize_numpy. That last step standardizes the price
            group by EACH VIEW'S OWN mean and std, so the absolute spread level
            is gone and what remains is spread in units of that stock's own
            price dispersion.

The second is the fair ceiling for a model. A cross-sectional rank metric ranks
stocks against each other at one instant, and two stocks with identically
shaped views but different absolute spreads produce identical embeddings.

RESOLUTION CAVEAT. The target's 60 s window is measured on the 1 Hz grid, 60
samples. The view carries 6-11 s per token, so the same window is 5-10 tokens
and the trailing statistic is a coarse proxy for it. That is a genuine limit of
what the encoder is shown, not an artifact of this script -- but it is why the
view number should not be read as "the same statistic, slightly noisier".

RANK IC IS WHY THERE IS NO FIT MONTH. A monotone transform leaves a rank
correlation unchanged, so the reversion coefficient is irrelevant and only the
sign matters.

Usage:
    uv run plots/metrics/mechanical_baseline.py --months 2013-01
    uv run plots/metrics/mechanical_baseline.py --months-from-k2ind \
        --json plots/metrics/mean_reversion_ic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))

from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.eval.metrics import grouped_rank_ic  # noqa: E402
from market_jepa.eval.tasks import HORIZONS, TARGET_TYPES  # noqa: E402
from market_jepa.schemas import LocalMachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import cell_agg, day_anchors, iter_panel  # noqa: E402

# Column positions inside the stable-finance market schema.
BID, ASK, VWAP = 0, 4, 1

# The trailing window, imported rather than repeated: it must be the SAME shape
# as the window the target averages over, or this stops being the causal
# analogue of the target's near leg and becomes an unrelated statistic.
from stable_finance.dataset.anchors import RETURN_VWAP_WINDOW  # noqa: E402

# A std over fewer than this many tokens is noise, not a measurement. At 11 s
# per token a 60 s window is 5 tokens, so this bites at the coarse end.
MIN_TOKENS = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--months", nargs="+", default=["2013-01"])
    p.add_argument("--months-from-k2ind", action="store_true",
                   help="use exactly the eval months the STANDARD k2ind series "
                        "covers (all 32), "
                        "so this line and that one average the same panel")
    p.add_argument("--shard-stride", type=int, default=1,
                   help="1 = the whole month; N > 1 subsamples shards "
                        "(a diagnostic, NOT a reportable number)")
    p.add_argument("--mosaic-dir", default=LocalMachineConfig().mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="lab/market-jepa-mosaic/"
                           "xs_anchor_stats_fwdvwap60")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--json", default=None)
    return p.parse_args()


def readouts(view: np.ndarray, agg: int) -> tuple[float, np.ndarray]:
    """``(spread, vol)`` over the trailing 60 s, read off ONE view.

    Both are window-MATCHED to the target's near leg and strictly causal: the
    target averages over [t, t+60) and these average over [t-60, t).

    The vol is returned as a length-len(HORIZONS) array of the SAME value. The
    predictor does not depend on h any more -- the target's near leg is the
    same 60 s window at every horizon -- and the shape is kept so callers that
    index by horizon do not have to change.
    """
    n = int(round(RETURN_VWAP_WINDOW / agg))
    # Realized vol off the TRADED price, not the quote midpoint. See the
    # module docstring: the midpoint at an instant is the contaminated
    # quantity the target itself was moved away from.
    px = view[:, VWAP].astype(np.float64)
    d = np.diff(px)
    vol = (np.std(d[-n:]) if n >= MIN_TOKENS and len(d) >= n else np.nan)
    w = view[-n:] if n >= 1 and len(view) >= n else view
    spread = float(np.nanmean(w[:, ASK] - w[:, BID])) if len(w) else np.nan
    return spread, np.full(len(HORIZONS), vol)


def main():
    args = parse_args()
    months = args.months
    if args.months_from_k2ind:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from style import standard_k2ind_months
        # NOT a hand-written glob: see style.standard_k2ind_runs. This used
        # to read the lambda sweep and silently cover 13 months.
        months = standard_k2ind_months("eval")

    sched = MarketSchedule(LocalMachineConfig().holiday_csv)
    anchors = day_anchors(8)
    records = []

    for ym in months:
        stats = AnchorStats(Path(args.xs_anchor_stats_dir) / f"{ym}.npz")
        mdir = Path(args.mosaic_dir) / ym[:4] / ym[5:]
        Z, SP, VO, C = [], [], [], []
        for views, metas in iter_panel(
            mdir, stats, sched, anchors, args.batch_size,
            num_shards=max(args.shard_stride, 1),
        ):
            for view, (z, date, anchor, _tk, _raw, _quote) in zip(views, metas):
                agg = cell_agg(date, int(anchor), int(anchor))
                if agg is None:
                    continue
                spr, vols = readouts(view, agg)
                Z.append(z); SP.append(spr); VO.append(vols)
                C.append(f"{date}@{anchor}")
        if not Z:
            print(f"{ym}: empty panel, skipped")
            continue
        Z = np.asarray(Z, dtype=np.float64)
        SP = np.asarray(SP)[:, None].repeat(len(HORIZONS), axis=1)
        VO = np.asarray(VO)
        C = np.asarray(C)

        print(f"\n{ym}: {len(Z)} rows, {len(set(C))} cells"
              + (f" (1/{args.shard_stride} of shards — DIAGNOSTIC)"
                 if args.shard_stride > 1 else ""))
        print(f"  {'target':24s} {'predictor':16s} {'IC':>8s}")
        for ti, t in enumerate(TARGET_TYPES):
            if t == "return":
                continue
            pred, name = ((-VO, "-trailing_vwap_vol_60s (view)")
                          if t == "volatility_change"
                          else (-SP, "-trailing_spread_60s (view)"))
            for hi, h in enumerate(HORIZONS):
                col = ti * len(HORIZONS) + hi
                ok = np.isfinite(Z[:, col]) & np.isfinite(pred[:, hi])
                if ok.sum() < 100:
                    continue
                m, se, n = grouped_rank_ic(pred[ok, hi], Z[ok, col], C[ok])
                records.append({"eval_month": ym, "target": f"{t}_{h}",
                                "predictor": name, "ic": float(m),
                                "se": float(se), "n_cells": int(n)})
                print(f"  {t + chr(95) + str(h):24s} {name:18s} {m:+8.4f}  ({n} cells)")

    if args.json:
        if args.shard_stride > 1:
            raise SystemExit("refusing to write a subsampled panel as a result")
        Path(args.json).write_text(json.dumps(records, indent=2))
        print(f"\nWrote {args.json} ({len(records)} records)")


if __name__ == "__main__":
    main()
