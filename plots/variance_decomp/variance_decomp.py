"""How much of the month-to-month spread in probe IC is DATA, and how much is NOISE?

Train the same recipe on ten frozen months with ten seeds each and the ICs
scatter. Two things drive that scatter and they have opposite implications:

  month  the training month really was easier or harder. Real signal about
         regime dependence, and it does not shrink by running more seeds.
  seed   weight init and data order. Pure optimization noise. It shrinks as
         1/sqrt(n_seeds), and any single-seed comparison between two configs
         is competing against it.

A one-way random-effects ANOVA separates them. With ``k`` months and ``n``
seeds per month:

    MS_between = n * Var(month means)          df = k - 1
    MS_within  = mean of within-month variances df = k(n - 1)
    sigma^2_seed  = MS_within
    sigma^2_month = (MS_between - MS_within) / n     (clipped at 0)
    ICC           = sigma^2_month / (sigma^2_month + sigma^2_seed)

ICC is the share of variance that is real. It is also the number that says how
much a one-seed result can be trusted: at ICC 0.7 a single seed carries a
standard error of sigma_seed, and two configs differing by less than about
2*sigma_seed have not been distinguished.

BALANCE MATTERS. The estimator above assumes n seeds in every month; an
unbalanced panel makes MS_between confound cell size with month effect. So
this uses only months where EVERY series has all ``--seeds`` seeds scored, and
prints what it dropped.

Usage::

    uv run plots/variance_decomp/variance_decomp.py
    uv run plots/variance_decomp/variance_decomp.py --task volatility_change_900
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# The project suffix is the commit the sweep was submitted at, so each
# generation gets its own tree and they cannot be mixed by accident:
#   variance-decomp-c71f5d  AUC era
#   variance-decomp-a4e741  rank IC, mid-to-mid return / one-sided change
#                           targets, and both arms trained WITHOUT the anchor
#                           tables they were scored against
#   variance-decomp-558b89  forward-VWAP targets on the reported recipe
# Point --ckpt-root at an older one to re-read it; the numbers are not
# comparable across generations, so there is no default that spans them.
CKPT_ROOT = Path("lab/market-jepa-checkpoints/variance-decomp-558b89")
RUN_RE = re.compile(r"^(?P<series>.+)_(?P<month>\d{4}-\d{2})_seed-(?P<seed>\d+)$")

# Which target each series is REPORTED on. The supervised arms train one head
# per target, so a supervised_return model is only on-task for return; LeJEPA
# has no head and one encoder is probed for all three.
ON_TASK = {
    "vd_supervised_return": ["return_900"],
    "vd_supervised_vol_change": ["volatility_change_900"],
    "vd_supervised_spread_change": ["spread_change_900"],
    "vd_lejepa_k2ind": ["return_900", "volatility_change_900", "spread_change_900"],
}


def collect(ckpt_root: Path, metric: str = "ic") -> dict:
    """``{(series, task): {month: {seed: value}}}`` from the checkpoint tree."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for meta_f in sorted(ckpt_root.glob("*/train_meta.json")):
        ic_f = meta_f.with_name("xs_ic.json")
        if not ic_f.is_file():
            continue
        try:
            meta = json.loads(meta_f.read_text())
            ic = json.loads(ic_f.read_text())
        except json.JSONDecodeError:
            continue
        m = RUN_RE.match(meta.get("run_name") or "")
        if not m:
            continue
        series, month, seed = m["series"], m["month"], int(m["seed"])
        for task in ON_TASK.get(series, []):
            # BOTH READOUTS, under their own keys. A head-only run
            # (POST_TRAIN_PROBE=0) writes no xs_ic/<task> at all, so a
            # collector that knew only the probe key returned an empty panel
            # for a sweep that had in fact finished. Callers ask for the key
            # they want: "<task>" is the probe, "head:<task>" the head.
            for key in (task, f"head:{task}"):
                v = ic.get(f"xs_{metric}/{key}")
                if v is not None and np.isfinite(v):
                    out[(series, key)][month][seed] = float(v)
    return {k: dict(v) for k, v in out.items()}


def decompose(by_month: dict, n_seeds: int) -> dict | None:
    """One-way random-effects ANOVA over a BALANCED {month: {seed: value}}."""
    months = sorted(m for m, s in by_month.items() if len(s) >= n_seeds)
    if len(months) < 2:
        return None
    # Take the first n_seeds by seed index so every cell is the same size.
    M = np.array([[by_month[m][s] for s in sorted(by_month[m])[:n_seeds]]
                  for m in months], dtype=float)
    k, n = M.shape
    month_means = M.mean(axis=1)
    ms_between = n * month_means.var(ddof=1)
    ms_within = float(np.mean(M.var(axis=1, ddof=1)))
    var_month = max((ms_between - ms_within) / n, 0.0)
    var_seed = ms_within
    total = var_month + var_seed
    return {
        "months": months, "k": k, "n": n,
        "grand": float(M.mean()),
        "sd_month": float(np.sqrt(var_month)),
        "sd_seed": float(np.sqrt(var_seed)),
        "icc": float(var_month / total) if total > 0 else float("nan"),
        "month_means": {m: float(v) for m, v in zip(months, month_means)},
        "spread": float(month_means.max() - month_means.min()),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT),
                   help="checkpoint tree for ONE sweep generation; "
                        f"default {CKPT_ROOT.name}")
    p.add_argument("--seeds", type=int, default=10,
                   help="seeds required per cell; cells with fewer are dropped")
    p.add_argument("--metric", default="ic", choices=("ic", "auc"))
    p.add_argument("--common-months", action="store_true",
                   help="restrict every series to the months ALL of them have "
                        "complete. Without it a series whose extra month is "
                        "unusually easy or hard gets an ICC that is not "
                        "comparable with the others'.")
    p.add_argument("--json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data = collect(Path(args.ckpt_root), args.metric)
    if not data:
        raise SystemExit(f"no scored runs under {args.ckpt_root}")

    if args.common_months:
        per_series = [{m for m, sd in bm.items() if len(sd) >= args.seeds}
                      for bm in data.values()]
        common = set.intersection(*per_series) if per_series else set()
        data = {k: {m: v for m, v in bm.items() if m in common}
                for k, bm in data.items()}
        print(f"Restricted to the {len(common)} month(s) every series has "
              f"complete: {' '.join(sorted(common))}")

    print(f"VARIANCE DECOMPOSITION — probe rank IC, {args.seeds} seeds/cell\n")
    rows = []
    for (series, task), by_month in sorted(data.items()):
        full = [m for m, s in by_month.items() if len(s) >= args.seeds]
        partial = [m for m, s in by_month.items() if 0 < len(s) < args.seeds]
        res = decompose(by_month, args.seeds)
        if res is None:
            print(f"{series} / {task}: only {len(full)} complete month(s) — skipped")
            continue
        res.update(series=series, task=task, dropped=sorted(partial))
        rows.append(res)

    hdr = (f"{'series':30s} {'task':22s} {'k':>2s} {'grand IC':>9s} "
           f"{'sd_month':>9s} {'sd_seed':>8s} {'ICC':>6s} {'range':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['series']:30s} {r['task']:22s} {r['k']:2d} {r['grand']:+9.4f} "
              f"{r['sd_month']:9.4f} {r['sd_seed']:8.4f} {100*r['icc']:5.1f}% "
              f"{r['spread']:8.4f}")

    print("\nPer-month means (the data axis):")
    months = sorted({m for r in rows for m in r["months"]})
    print(f"{'series/task':46s}" + "".join(f"{m[2:]:>9s}" for m in months))
    for r in rows:
        lab = f"{r['series'].replace('vd_','')}/{r['task'].replace('_900','')}"
        print(f"{lab:46s}" + "".join(
            f"{r['month_means'].get(m, float('nan')):+9.4f}" for m in months))

    dropped = sorted({m for r in rows for m in r["dropped"]})
    if dropped:
        print(f"\nDropped (incomplete, < {args.seeds} seeds): {' '.join(dropped)}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
