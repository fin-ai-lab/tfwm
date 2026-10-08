"""The hold-to-close strategy, measured against the real auction return.

THE SETUP, chosen to be defensible rather than flattering:

- One entry a day, at one anchor. Names whose forecast clears their own quoted
  cost by ``k`` go long or short by its sign; everything else is left alone.
- Equal weight within each leg, gross exposure 2.0 (1x long, 1x short), dollar
  neutral by construction.
- ENTRY crosses the spread and pays the name's own quoted half-spread.
- EXIT goes in the closing auction, the one moment of the day when a large
  order is cheap: a single cross at one price carrying on the order of a
  thousand times a median second's volume. It is cheap, NOT free -- the
  auction has impact, and a strategy that closes the same direction every day
  is part of the imbalance -- so the exit is charged ``--auction-cost`` times
  the half-spread, swept rather than assumed.
- Held intraday and flat overnight, so there is no borrow to pay on the short
  leg and no overnight gap risk. Stated rather than silently omitted.
- Annualized over 252 days from one non-overlapping observation per day, so
  the periods are independent and the annualization is honest.

THE FORECAST IS THE EXISTING 15-MINUTE MODEL. No retraining: mu is fit on the
900s target and scored against the return to the auction, which it predicts
nearly as well as a model fit directly on the long horizon. What it does NOT
carry is the right SCALE for a long hold, so the edge is rescaled by the slope
of the to-close return on the forecast, estimated per anchor on
cross-sectionally demeaned pairs pooled across months -- demeaned because a
pooled raw regression conflates the market's move with the cross-sectional
signal and comes out NEGATIVE, and per anchor because the holding period, and
so the size of the move being predicted, shortens through the day.

Usage:
    uv run plots/ic_vs_sharpe/close_strategy.py --arms sup_multi_w8
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANELS = "lab/ic_sharpe_configs/panels"
CLOSE = "lab/ic_sharpe_configs/close_returns"
TRADING_DAYS = 252.0
MIN_NAMES_PER_LEG = 3
#: A to-close move beyond this is a bad print, not a return. Dropped rather
#: than clipped: a clip still books a fictitious profit at the cap.
MAX_ABS_RETURN = 0.05


def load_arm(arm: str):
    """Forecast, quoted half-spread and REALIZED to-close return, aligned."""
    for path in sorted(glob.glob(f"{PANELS}/{arm}-*-panel.npz")):
        month = Path(path).name.rsplit("-", 3)[1] + "-" + Path(path).name.rsplit("-", 3)[2]
        close_path = Path(CLOSE) / f"{month}-close.npz"
        if not close_path.exists():
            continue
        panel = np.load(path, allow_pickle=True)
        close = np.load(close_path, allow_pickle=True)
        assets = panel["assets"].tolist()
        decisions = panel["decisions"]
        anchor_of = close["anchors"].tolist()
        index = {(d, t): i for i, (d, t)
                 in enumerate(zip(close["dates"].tolist(), close["tickers"].tolist()))}
        to_close = close["to_close"]
        realized = np.full((len(decisions), len(assets)), np.nan)
        for row, label in enumerate(decisions):
            day, anchor = label.split("|")
            column = anchor_of.index(int(anchor))
            for column_asset, ticker in enumerate(assets):
                found = index.get((day, ticker))
                if found is not None:
                    realized[row, column_asset] = to_close[found, column]
        yield {
            "month": month,
            "mu": panel["mu_all"][:, :, 2].astype(np.float64),
            "spread": panel["half_spread"].astype(np.float64),
            "realized": realized,
            # The 15-minute exit, for comparison against the same entry: the
            # realized 900s return and the out-of-sample beta that puts the
            # forecast into its units.
            "realized_900": panel["realized_all"][:, :, 2].astype(np.float64),
            "beta_900": float(panel["betas"][2]),
            "anchors": np.array([int(x.split("|")[1]) for x in decisions]),
            "days": np.array([x.split("|")[0] for x in decisions]),
        }


def demeaned(values, usable):
    """Cross-sectional deviation from the decision's own mean."""
    out = np.zeros_like(values)
    for row in range(values.shape[0]):
        mask = usable[row]
        if mask.sum() >= 2:
            out[row, mask] = values[row, mask] - values[row, mask].mean()
    return out


def fit_slopes(months, anchors):
    """Return-to-close per unit forecast, per anchor, on demeaned pairs.

    A slope through the origin on cross-sectionally demeaned pairs. Pooling the
    RAW pairs instead conflates the market's move with the cross-sectional
    signal and produces a confidently negative slope on a forecast whose rank
    IC is positive at every anchor.
    """
    numerator = {a: 0.0 for a in anchors}
    denominator = {a: 0.0 for a in anchors}
    for data in months:
        usable = _usable(data)
        x = demeaned(data["mu"], usable)
        y = demeaned(data["realized"], usable)
        for row in range(len(data["anchors"])):
            mask = usable[row]
            if mask.sum() < 20:
                continue
            anchor = int(data["anchors"][row])
            numerator[anchor] += float(x[row, mask] @ y[row, mask])
            denominator[anchor] += float(x[row, mask] @ x[row, mask])
    return {a: (numerator[a] / denominator[a] if denominator[a] > 0 else np.nan)
            for a in anchors}


def _usable(data):
    return (np.isfinite(data["mu"]) & np.isfinite(data["spread"])
            & (data["spread"] > 0) & np.isfinite(data["realized"])
            & (np.abs(data["realized"]) <= MAX_ABS_RETURN))


def run(months, slopes, *, anchor, multiple, auction_cost):
    """Per-day net returns for one (anchor, threshold, exit cost)."""
    gross_all, net_all, names_all = [], [], []
    for data in months:
        rows = data["anchors"] == anchor
        if not rows.any():
            continue
        usable = _usable(data)[rows]
        mu = data["mu"][rows]
        spread = data["spread"][rows]
        realized = data["realized"][rows]
        edge = demeaned(mu, usable) * slopes[anchor]

        clears = usable & (np.abs(edge) > multiple * spread)
        side = np.sign(edge) * clears
        longs, shorts = (side > 0).sum(axis=1), (side < 0).sum(axis=1)
        traded = (longs >= MIN_NAMES_PER_LEG) & (shorts >= MIN_NAMES_PER_LEG)
        per_long = np.divide(1.0, longs, out=np.zeros(len(longs)), where=longs > 0)
        per_short = np.divide(1.0, shorts, out=np.zeros(len(shorts)), where=shorts > 0)
        weight = (np.where(side > 0, per_long[:, None], 0.0)
                  - np.where(side < 0, per_short[:, None], 0.0))
        weight = np.where(traded[:, None], weight, 0.0)

        gross = (weight * np.nan_to_num(realized)).sum(axis=1)
        entry = (np.abs(weight) * np.nan_to_num(spread)).sum(axis=1)
        cost = entry * (1.0 + auction_cost)
        gross_all.append(gross[traded])
        net_all.append((gross - cost)[traded])
        names_all.append(np.abs(side).sum(axis=1)[traded])
    if not net_all:
        return None
    gross = np.concatenate(gross_all)
    net = np.concatenate(net_all)
    if len(net) < 20 or net.std(ddof=1) <= 0:
        return None
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    years = len(net) / TRADING_DAYS
    # Lo (2002): SE of an annualized Sharpe over this many years. At two or
    # three years anything below ~1 cannot be told from zero, which is the
    # honest frame for a result this size rather than a caveat on it.
    standard_error = float(np.sqrt((1 + sharpe ** 2 / 2) / years))
    return {
        "anchor": anchor, "multiple": multiple, "auction_cost": auction_cost,
        "days": int(len(net)), "names": float(np.concatenate(names_all).mean()),
        "gross_bp": float(gross.mean() * 1e4),
        "cost_bp": float((gross - net).mean() * 1e4),
        "sharpe_gross": float(gross.mean() / gross.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe, "se": standard_error,
        "ci_low": sharpe - 1.96 * standard_error,
        "ci_high": sharpe + 1.96 * standard_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["sup_multi_w8"])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--auction-cost", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5],
                        help="exit cost as a multiple of the entry half-spread")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    everything = {}
    for arm in args.arms:
        months = list(load_arm(arm))
        if not months:
            print(f"!! {arm}: no months with close returns yet")
            continue
        anchors = sorted({int(a) for d in months for a in d["anchors"]})
        slopes = fit_slopes(months, anchors)
        print(f"\n=== {arm}: {len(months)} months ===")
        print("  slope of the to-close return on the forecast, per anchor:")
        print("   " + "  ".join(f"{a}:{slopes[a]:.2f}" for a in anchors))
        rows = []
        for exit_cost in args.auction_cost:
            print(f"\n  exit charged at {exit_cost:.2f}x the entry half-spread")
            print(f"    {'anchor':>7} {'k':>5} {'days':>5} {'names':>6} "
                  f"{'gross':>9} {'cost':>8} {'S_gross':>8} {'Sharpe':>8} "
                  f"{'95% CI':>18}")
            for anchor in anchors:
                for multiple in args.thresholds:
                    got = run(months, slopes, anchor=anchor, multiple=multiple,
                              auction_cost=exit_cost)
                    if got is None:
                        continue
                    rows.append(got)
                    print(f"    {got['anchor']:7d} {got['multiple']:5.2f} "
                          f"{got['days']:5d} {got['names']:6.1f} "
                          f"{got['gross_bp']:+8.2f}bp {got['cost_bp']:7.2f}bp "
                          f"{got['sharpe_gross']:+8.2f} {got['sharpe']:+8.2f} "
                          f"[{got['ci_low']:+6.2f},{got['ci_high']:+6.2f}]")
        everything[arm] = {"slopes": slopes, "rows": rows}

    if args.out:
        Path(args.out).write_text(json.dumps(everything, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
