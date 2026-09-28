"""Correlate IC with Sharpe across the checkpoints scored by portfolio_ic_sharpe.

One row per checkpoint: the probe's rank IC on its eval month, and the Sharpe
of every portfolio built from that same forecast. The question is how tightly
the two move together, so the reported statistic is a correlation across
checkpoints -- Pearson for the linear reading and Spearman because neither
quantity has any reason to be linearly related to the other.

BOTH ARE ESTIMATES, and the correlation between two noisy measurements is
attenuated toward zero. Each IC carries the per-cell standard error the scorer
reports; a Sharpe over ~170 decisions carries roughly 1/sqrt(n) in annualized
units. The printed table shows both so a weak correlation can be read as
"weakly related" or "too noisy to tell" rather than conflated.

Usage:
    uv run scripts/eval/portfolio_ic_sharpe_summary.py \
        --results-dir /data/lab/portfolio_ic_sharpe/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stable_finance import paired_difference  # noqa: E402

# The spread-charged Sharpe on target weights and the filled Sharpe on the
# realized position path are the same number whenever every order fills, so
# only the executable one is a column; `fill_gap` carries the difference and is
# printed when it is ever non-zero.
SHARPE_COLUMNS = (
    ("sharpe_mid", "MV, mid"),
    # THE PAIR THE QUESTION IS ASKED IN: the same book marked at the midpoint,
    # and the same book after buying at the ask and selling at the bid on
    # every weight change (stable_finance.cross_spread_sharpe). ``sharpe_net``
    # is the same charge levied through simulated fills, and equals this one
    # whenever every order fills -- it is kept so a month where something
    # stops quoting cannot hide inside a single column.
    ("sharpe_cross_spread", "MV, cross spread"),
    ("sharpe_mid_nocost", "MV, mid nocost"),
    ("sharpe_net", "MV, net"),
    ("sharpe_control_mid", "rank, mid"),
    ("sharpe_control_cross_spread", "rank, cross spread"),
    ("sharpe_control_net", "rank, net"),
)
IC_COLUMNS = (("ic", "IC (all rows)"), ("ic_universe", "IC (universe)"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--horizon", type=int, default=900)
    p.add_argument("--pathway", default="probe", choices=("probe", "head"),
                   help="which entry point to report: the ridge probe on "
                        "embeddings, or the checkpoint's own trained head")
    p.add_argument("--json-out", default=None)
    p.add_argument("--compare", action="store_true",
                   help="also report head-minus-probe on the checkpoints that "
                        "have both, paired by checkpoint")
    return p.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = starts.cumsum()
    bounds = np.r_[np.flatnonzero(starts), len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = 0.5 * (bounds[dense] + bounds[dense - 1] + 1)
    return ranks


def correlate(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    """Pearson, Spearman, and Pearson's standard error under normality."""
    ok = np.isfinite(left) & np.isfinite(right)
    left, right = left[ok], right[ok]
    n = len(left)
    if n < 3:
        return np.nan, np.nan, np.nan

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan

    r = pearson(left, right)
    rho = pearson(rankdata(left), rankdata(right))
    se = np.sqrt((1 - r * r) / (n - 2)) if n > 2 and np.isfinite(r) else np.nan
    return r, rho, se


def main():
    args = parse_args()
    key = str(args.horizon)
    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        result = json.loads(path.read_text())
        horizon = result.get("pathways", {}).get(args.pathway, {}).get(key)
        if horizon is None:
            continue
        if "untradable" in horizon:
            print(f"skipping {result['train_month']}: {horizon['untradable']}")
            continue
        rows.append({**result, **horizon})
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} result(s) at horizon {key}")

    rows.sort(key=lambda r: r["train_month"])
    n_decisions = np.array([r["n_eval_decisions"] for r in rows], dtype=np.float64)

    print(f"\n=== {args.pathway} | horizon {args.horizon}s | "
          f"{len(rows)} checkpoints ===\n")
    header = (f"{'train':>8} {'eval':>8} {'IC':>8} {'±se':>7} {'univIC':>8} "
              f"{'MVmid':>7} {'MVxsp':>7} {'rkmid':>7} {'rkxsp':>7} "
              f"{'gross_bp':>9} {'cost_bp':>8} {'turn':>6} {'names':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['train_month']:>8} {r['eval_month']:>8} "
              f"{r['ic']:+8.4f} {r['ic_se']:7.4f} {r['ic_universe']:+8.4f} "
              f"{r['sharpe_mid']:+7.2f} {r['sharpe_cross_spread']:+7.2f} "
              f"{r['sharpe_control_mid']:+7.2f} "
              f"{r['sharpe_control_cross_spread']:+7.2f} "
              f"{r['mean_gross_return_bps']:+9.2f} {r['mean_cost_bps']:8.2f} "
              f"{r['mean_turnover']:6.2f} {r['n_assets']:6d}")

    gaps = np.array([abs(r.get("fill_gap", 0.0)) for r in rows])
    if np.nanmax(gaps) > 1e-9:
        print(f"\n  NOTE: {int((gaps > 1e-9).sum())} checkpoint(s) had unfilled "
              f"orders; max |filled - target| Sharpe gap {np.nanmax(gaps):.3f}")

    print("\nmedians:")
    for field, label in IC_COLUMNS + SHARPE_COLUMNS:
        values = np.array([r.get(field, np.nan) for r in rows], dtype=np.float64)
        print(f"  {label:<16} median {np.nanmedian(values):+7.3f}   "
              f"mean {np.nanmean(values):+7.3f}   sd {np.nanstd(values, ddof=1):6.3f}   "
              f"positive {int((values > 0).sum())}/{len(values)}")

    # A Sharpe estimated over n decisions carries about sqrt(periods/n) in
    # annualized units; printed so a near-zero correlation can be told apart
    # from one the measurements were never precise enough to see.
    periods = rows[0]["periods_per_year"]
    print(f"\n  Sharpe standard error ~ {np.mean(np.sqrt(periods / n_decisions)):.2f} "
          f"annualized (n ~ {int(np.mean(n_decisions))} decisions per checkpoint)")
    print(f"  IC standard error     ~ {np.mean([r['ic_se'] for r in rows]):.4f} per checkpoint")

    print("\n=== correlation across checkpoints ===\n")
    print(f"{'':<18}" + "".join(f"{label:>22}" for _, label in SHARPE_COLUMNS))
    summary = {}
    for ic_field, ic_label in IC_COLUMNS:
        ic_values = np.array([r[ic_field] for r in rows], dtype=np.float64)
        cells = []
        for sharpe_field, sharpe_label in SHARPE_COLUMNS:
            sharpe = np.array([r.get(sharpe_field, np.nan)
                               for r in rows], dtype=np.float64)
            r, rho, se = correlate(ic_values, sharpe)
            summary[f"{ic_field}~{sharpe_field}"] = {
                "pearson": r, "spearman": rho, "pearson_se": se, "n": len(rows),
            }
            cells.append(f"{r:+.2f}/{rho:+.2f} ±{se:.2f}".rjust(22))
        print(f"{ic_label:<18}" + "".join(cells))
    print("\n  cells are Pearson / Spearman ± Pearson standard error")

    if args.compare:
        compare_pathways(args, key)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"horizon": args.horizon, "pathway": args.pathway,
             "n_checkpoints": len(rows),
             "correlations": summary,
             "rows": [{k: r[k] for k in
                       ("train_month", "eval_month", "n_assets", "ic", "ic_se",
                        "ic_universe", *[f for f, _ in SHARPE_COLUMNS])}
                      for r in rows]},
            indent=2))
        print(f"\nwrote {args.json_out}")


def compare_pathways(args, key):
    """Head minus probe, paired by checkpoint.

    PAIRED, not two independent medians. The two pathways are scored on the
    same months, the same universe and the same decisions, so the month-to-
    month variation -- which dwarfs the difference between them -- cancels in
    the difference and would otherwise swamp any comparison of their levels.
    """
    pairs = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        result = json.loads(path.read_text())
        pathways = result.get("pathways", {})
        probe = pathways.get("probe", {}).get(key)
        head = pathways.get("head", {}).get(key)
        if not probe or not head:
            continue
        if "untradable" in probe or "untradable" in head:
            continue
        pairs.append((result["train_month"], probe, head))
    if len(pairs) < 3:
        print(f"\n=== head vs probe: only {len(pairs)} paired checkpoint(s) ===")
        return

    print(f"\n=== head minus probe | horizon {args.horizon}s | "
          f"{len(pairs)} paired checkpoints ===\n")
    print(f"{'metric':<18} {'probe':>9} {'head':>9} {'head-probe':>12} "
          f"{'±se':>8} {'t':>7} {'head wins':>10}")
    print("-" * 78)
    for field, label in IC_COLUMNS + SHARPE_COLUMNS:
        probe_values = np.array([p[field] for _, p, _ in pairs], dtype=np.float64)
        head_values = np.array([h[field] for _, _, h in pairs], dtype=np.float64)
        diff = paired_difference(head_values, probe_values)
        t = diff.mean / diff.standard_error if diff.standard_error else np.nan
        wins = int((head_values > probe_values).sum())
        print(f"{label:<18} {np.nanmean(probe_values):+9.3f} "
              f"{np.nanmean(head_values):+9.3f} {diff.mean:+12.3f} "
              f"{diff.standard_error:8.3f} {t:+7.2f} "
              f"{wins:>6}/{len(pairs)}")
    print("\n  columns are means across checkpoints; the difference is paired "
          "per checkpoint")


if __name__ == "__main__":
    main()
