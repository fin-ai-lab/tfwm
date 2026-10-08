"""Annualized Sharpe against EFQ, for two exit rules.

THE FIGURE. Two panels sharing an axis. Left: enter at one anchor a day and
exit 15 minutes later, both legs marketable. Right: the same entry, held to
the closing auction and exited in the cross. In each, x is EFQ and y is the
annualized Sharpe, one line per model.

EFQ, FOLLOWING LEVY (2022). The effective half-spread actually paid over the
half-spread quoted when the order was sent, in percent: 0% is a midpoint
fill and 100% is paying the whole quoted spread. That trial routed real
orders and measured it -- 52% on buys and 55% on sells through direct market
access, 79% at NASDAQ and 88% at NYSE -- so the axis is a measured quantity
rather than an assumption this study invents. It is swept past 100% because
size and crowding both put EFQ there: see ``effective_half_spread``.

WHY IT IS THE RIGHT AXIS. Which fraction of the spread an order pays depends
on the venue and the size far more than on the forecast. Reporting one
Sharpe at one cost assumption invites an argument about the assumption;
reporting the curve hands the reader the assumption as an axis, so they can
locate their own execution and read off their own answer.

WHAT THE TWO PANELS ARE FOR. Both exits pay for the same entry, so the panels
isolate the holding period. The 15-minute exit collects 15 minutes of edge
and pays two marketable legs. The auction exit collects hours of edge and
pays one marketable leg plus a cheap cross. A quoted spread is paid per
ROUND TRIP, not per unit of time, so holding longer buys more edge against
roughly the same charge -- the right panel should break even at a much worse
execution quality than the left, and the gap between them is the size of
that effect.

THE SCALE OF THE EDGE IS NOT THE SCALE OF THE FORECAST. The cached forecast
is already in return units -- the probe predicts an empirical-uniform rank
and the adapter rescales it by an out-of-sample beta fit at 900 seconds. That
beta is right for a 15-minute hold and wrong for a hold to the bell, which is
a longer move, so each panel applies its OWN correction: the slope of that
panel's realized return on the forecast, fit per anchor on cross-sectionally
demeaned pairs with the evaluated month LEFT OUT. Demeaned because a pooled
raw regression conflates the market's move with the cross-sectional signal
and comes out negative on a forecast whose rank IC is positive; left out
because a threshold fit on the month it is scored in carries a lookahead.
The 15-minute correction lands near 1 when the stored beta is well
calibrated, which makes it a check as well as a rescaling.

Usage:
    uv run plots/ic_vs_sharpe/eq_curve.py --anchor 15300 --k 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "plots"))

from stable_finance.costs import EFQ_INFORMED_RANGE                  # noqa: E402
from stable_finance.metrics import (                                 # noqa: E402
    sharpe_ratio,
    sharpe_vs_execution_quality,
)
from style import (COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL,     # noqa: E402
                   add_bottom_legend, apply_style, save_figure)

from close_strategy import MAX_ABS_RETURN, demeaned, load_arm        # noqa: E402

#: Supervised return, supervised multi-head, and the reported LeJEPA arm, in
#: the colors those families carry paper-wide (``style.SERIES_STYLES``), so a
#: reader holding this beside probe_fit_breadth sees the same model in the
#: same hue. ``pair_warp_6mo`` is the pairing the paper reports -- it is what
#: ``probe_fit_breadth.py`` draws as ``LEJEPA_KEY`` -- NOT ``pair_rrc``,
#: which is only the control in the augmentation ablation, and no longer
#: ``pair_k2ind``, which some older comments still call the standard arm.
ARMS = {
    "sup_return_w8": ("Supervised Specialist",
                      SERIES_STYLES["supervised_return"]["color"]),
    "sup_multi_w8": ("Multihead", SERIES_STYLES["multihead"]["color"]),
    "pair_warp_6mo": ("LeJEPA Time Warping", SERIES_STYLES["lejepa"]["color"]),
}
MIN_NAMES_PER_LEG = 3
#: The auction leg's cost relative to a marketable intraday leg at the same
#: EFQ. The closing cross carries on the order of a thousand times a median
#: second's volume at a single price, so it is cheap -- but a strategy that
#: unwinds the same direction every day is part of the imbalance, so it is
#: not free, and the discount is a flag rather than a constant.
AUCTION_LEG = 0.25


def _usable(data, key, cap):
    return (np.isfinite(data["mu"]) & np.isfinite(data["spread"])
            & (data["spread"] > 0) & np.isfinite(data[key])
            & (np.abs(data[key]) <= cap))


def leave_one_out_slopes(months, anchors, key, cap):
    """Per-month slopes fit on every OTHER month.

    Pooling all months and then evaluating on them lets the threshold peek at
    the month it is being scored on. The accumulators are additive, so
    leaving a month out is a subtraction rather than a refit.
    """
    per_month = {}
    total_n = {a: 0.0 for a in anchors}
    total_d = {a: 0.0 for a in anchors}
    for data in months:
        num = {a: 0.0 for a in anchors}
        den = {a: 0.0 for a in anchors}
        usable = _usable(data, key, cap)
        x = demeaned(data["mu"], usable)
        y = demeaned(data[key], usable)
        for row in range(len(data["anchors"])):
            mask = usable[row]
            if mask.sum() < 20:
                continue
            anchor = int(data["anchors"][row])
            num[anchor] += float(x[row, mask] @ y[row, mask])
            den[anchor] += float(x[row, mask] @ x[row, mask])
        per_month[data["month"]] = (num, den)
        for a in anchors:
            total_n[a] += num[a]
            total_d[a] += den[a]
    out = {}
    for month, (num, den) in per_month.items():
        out[month] = {a: ((total_n[a] - num[a]) / (total_d[a] - den[a])
                          if total_d[a] - den[a] > 0 else np.nan)
                      for a in anchors}
    return out


def book(months, slopes, *, anchor, k, key, cap):
    """Per-day gross return and per-day turnover priced at the QUOTED spread.

    The caller multiplies the turnover by the legs it pays and by EFQ, so one
    book can be charged under any execution assumption. Every decision day is
    returned, flat days included, so series taken at different thresholds stay
    comparable and annualization stays honest.
    """
    gross_all, cost_all, names_all = [], [], []
    for data in months:
        rows = data["anchors"] == anchor
        if not rows.any():
            continue
        slope = slopes[data["month"]][anchor] if isinstance(slopes, dict) \
            and data["month"] in slopes else slopes
        if not np.isfinite(slope):
            continue
        usable = _usable(data, key, cap)[rows]
        mu, spread = data["mu"][rows], data["spread"][rows]
        realized = data[key][rows]
        edge = demeaned(mu, usable) * slope

        clears = usable & (np.abs(edge) > k * spread)
        side = np.sign(edge) * clears
        longs, shorts = (side > 0).sum(axis=1), (side < 0).sum(axis=1)
        traded = (longs >= MIN_NAMES_PER_LEG) & (shorts >= MIN_NAMES_PER_LEG)
        per_long = np.divide(1.0, longs, out=np.zeros(len(longs)), where=longs > 0)
        per_short = np.divide(1.0, shorts, out=np.zeros(len(shorts)), where=shorts > 0)
        weight = (np.where(side > 0, per_long[:, None], 0.0)
                  - np.where(side < 0, per_short[:, None], 0.0))
        weight = np.where(traded[:, None], weight, 0.0)

        # A day the rule declines to trade is a FLAT day, not an absent one:
        # the capital earned zero and must stay in the series. Keeping only
        # traded days would annualize a selective strategy as though it were
        # invested all year, and this rule sits out more as E/Q rises, so
        # dropping them would reward tightening with a free Sharpe.
        turnover = np.where(traded, (np.abs(weight)
                                     * np.nan_to_num(spread)).sum(axis=1), 0.0)
        gross_all.append(np.where(traded, (weight * np.nan_to_num(realized))
                                  .sum(axis=1), 0.0))
        cost_all.append(turnover)
        names_all.append(np.where(traded, np.abs(side).sum(axis=1), 0.0))
    if not gross_all:
        return None
    # Per DAY, not averaged: the caller needs both the typical book size and
    # how often the rule traded at all.
    return (np.concatenate(gross_all), np.concatenate(cost_all),
            np.concatenate(names_all))


def adaptive_curve(months, slopes, *, anchor, grid, key, cap, legs, tau,
                   periods_per_year=252.0):
    """Sharpe across E/Q when the threshold is the cost actually paid.

    A fixed threshold ``|edge| > k * quoted`` is a free parameter, and the
    reported break-even moves a long way with it -- which makes choosing one
    an invitation to choose the flattering one. The threshold a trader would
    actually use is not fixed: it is the round trip they will actually be
    charged at their own execution quality,

        |edge| > tau * legs * EFQ * quoted half-spread

    so the rule tightens exactly as execution worsens and the parameter
    disappears into the axis. ``tau`` is what survives -- 1.0 is "trade when
    the edge covers the cost", and above 1 demands a margin on top.

    The selection now changes along the axis, so the curve is NOT linear in
    EFQ and cannot be produced from a single book: each point is re-run, and
    the break-even is found by interpolation rather than in closed form. At
    ``EFQ = 0`` the threshold is zero and everything eligible trades, which
    is the frictionless limit -- a midpoint fill, in Levy's units.
    """
    sharpe, names, active = [], [], []
    for ratio in grid:
        got = book(months, slopes, anchor=anchor, k=tau * legs * ratio,
                   key=key, cap=cap)
        if got is None:
            sharpe.append(np.nan)
            names.append(np.nan)
            active.append(np.nan)
            continue
        gross, turnover, held = got
        active.append(float((held > 0).mean()))
        held = float(held[held > 0].mean()) if (held > 0).any() else 0.0
        sharpe.append(sharpe_ratio(gross - legs * ratio * turnover,
                                   periods_per_year=periods_per_year))
        names.append(held)
    sharpe = np.asarray(sharpe, dtype=float)
    return (sharpe, np.asarray(names, dtype=float), np.asarray(active, dtype=float),
            _first_crossing(grid, sharpe))


def _first_crossing(grid, sharpe):
    """The EFQ where the curve first falls through zero, interpolated.

    Three outcomes, kept distinct because two of them are opposite findings.
    A finite value is the break-even. ``inf`` means the curve never falls
    through zero inside the swept range -- it is profitable at every
    execution quality measured, which is the STRONGEST result here and must
    not be reported as a missing one. ``nan`` means it started at or below
    zero and so has no break-even to find.
    """
    if not np.isfinite(sharpe[0]) or sharpe[0] <= 0:
        return float("nan")
    for index in range(1, len(sharpe)):
        left, right = sharpe[index - 1], sharpe[index]
        if np.isfinite(left) and np.isfinite(right) and left > 0 >= right:
            span = left - right
            if span <= 0:
                return float(grid[index])
            return float(grid[index - 1] + (grid[index] - grid[index - 1])
                         * left / span)
    return float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--anchor", type=int, default=15300)
    parser.add_argument("--threshold", choices=("adaptive", "fixed"),
                        default="adaptive",
                        help="adaptive: require the edge to cover the round "
                             "trip actually paid at that EFQ, which removes "
                             "the free parameter; fixed: a constant multiple "
                             "of the quoted spread")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="margin demanded over the cost, adaptive mode")
    parser.add_argument("--k", type=float, default=0.5,
                        help="threshold in quoted half-spreads, fixed mode")
    # Swept to 150%, past the 100% that is the whole quoted spread and the
    # right edge of Levy's own axis, because above 100% is a regime this
    # strategy would really be run in rather than an extrapolation: an order
    # bigger than the quoted depth walks the book, and a signal others are
    # also trading moves the quote away before the order completes. That is
    # the capacity question, entered from the cost side.
    parser.add_argument("--efq-max", type=float, default=1.5)
    # A fixed threshold keeps paying a cost it never agreed to and runs deep
    # negative; the adaptive rule stops trading instead, so it needs far less
    # room below zero.
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--out", default="plots/ic_vs_sharpe/efq_curve.png")
    parser.add_argument("--json-out", default="plots/metrics/efq_curve.json")
    args = parser.parse_args()
    if args.y_min is None:
        args.y_min = -2.5 if args.threshold == "adaptive" else -10.0
    if args.y_max is None:
        args.y_max = 4.0 if args.threshold == "adaptive" else 5.0

    # The adaptive rule re-selects at every point, so each one is its own
    # noisy estimate; a coarser grid is less jagged without hiding anything.
    grid = np.linspace(0.0, args.efq_max,
                       61 if args.threshold == "adaptive" else 121)
    # Left pays two marketable legs; right pays one plus the cheap cross.
    panels = [
        ("Exit After 15 Minutes", "realized_900", 0.05, 2.0),
        ("Exit in the Closing Auction", "realized", MAX_ABS_RETURN,
         1.0 + AUCTION_LEG),
    ]
    apply_style(extra=COMPACT_RC_PARAMS)
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.9), sharey=True)
    summary = {}

    for arm in args.arms:
        label, colour = ARMS.get(arm, (arm, None))
        months = list(load_arm(arm))
        if not months:
            print(f"!! {arm}: no months")
            continue
        anchors = sorted({int(a) for d in months for a in d["anchors"]})
        summary[arm] = {}
        for axis, (title, key, cap, legs) in zip(axes, panels):
            slopes = leave_one_out_slopes(months, anchors, key, cap)
            if args.threshold == "adaptive":
                curve_sharpe, held, traded_rate, breakeven = adaptive_curve(
                    months, slopes, anchor=args.anchor, grid=grid, key=key,
                    cap=cap, legs=legs, tau=args.tau)
                if not np.isfinite(curve_sharpe).any():
                    continue
                eq, standard_error = grid, None
                gross_sharpe, names = float(curve_sharpe[0]), float(held[0])
                active_at = lambda r: float(np.interp(r, grid, traded_rate))
            else:
                got = book(months, slopes, anchor=args.anchor, k=args.k,
                           key=key, cap=cap)
                if got is None:
                    continue
                gross, turnover, per_day = got
                names = float(per_day[per_day > 0].mean()) if (per_day > 0).any() else 0.0
                curve = sharpe_vs_execution_quality(
                    gross, legs * turnover, efq_grid=grid)
                eq, curve_sharpe = curve.efq, curve.sharpe
                standard_error = curve.standard_error
                gross_sharpe, breakeven = curve.gross_sharpe, curve.breakeven
                active_at = lambda r: float("nan")
            # EFQ is a fraction internally, because it multiplies a spread;
            # it is DISPLAYED as a percent, which is how Levy (2022) and the
            # microstructure literature quote it.
            axis.plot(100.0 * eq, curve_sharpe, color=colour, lw=1.4,
                      label=label)
            if standard_error is not None:
                axis.fill_between(100.0 * eq,
                                  curve_sharpe - 1.96 * standard_error,
                                  curve_sharpe + 1.96 * standard_error,
                                  color=colour, alpha=0.16, lw=0)
            axis.set_title(title)
            at = lambda ratio: float(np.interp(ratio, eq, curve_sharpe))
            summary[arm][title] = {
                "threshold": args.threshold, "tau": args.tau,
                "frictionless_sharpe": gross_sharpe, "breakeven": breakeven,
                "names_frictionless": names,
                "sharpe_at_1": at(1.0), "sharpe_at_075": at(0.75),
                "days_traded_at_075": active_at(0.75),
                "days_traded_at_1": active_at(1.0),
            }
            mark = ("never" if np.isnan(breakeven)
                    else ">swept" if np.isinf(breakeven) else f"{breakeven:.2f}")
            print(f"{arm:16s} {title:28s} S(0) {gross_sharpe:+6.2f}  "
                  f"EFQ* {mark:>6s}  S(0.75) {at(0.75):+6.2f}  "
                  f"S(1.0) {at(1.0):+6.2f}  "
                  f"active@0.75 {active_at(0.75):5.0%}")

    # The band an INFORMED order is routed in, from Levy (2022): direct
    # market access to lit exchange. Drawn, not stated in a title -- the
    # caption carries what it is.
    low, high = EFQ_INFORMED_RANGE
    for axis in axes:
        axis.axvspan(100.0 * low, 100.0 * high, color="#999999", alpha=0.13,
                     lw=0, zorder=0)
        axis.axhline(0, color="0.35", lw=0.8)
        axis.set_xlabel("EFQ (Percent)")
        axis.set_xlim(0, 100.0 * args.efq_max)
        axis.set_ylim(args.y_min, args.y_max)
    axes[0].set_ylabel("Annualized Sharpe")
    # Labelled once, on the first panel a reader meets: the band is the same
    # in both, and repeating it reads as two different things.
    axes[0].annotate("Typical EFQ", xy=(50.0 * (low + high), args.y_max),
                     xytext=(0, -4), textcoords="offset points", ha="center",
                     va="top", fontsize=6.5, color="0.55")
    figure.tight_layout()
    add_bottom_legend(figure, ncol=len(args.arms))
    save_figure(figure, args.out)
    print(f"\nwrote {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, default=float))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
