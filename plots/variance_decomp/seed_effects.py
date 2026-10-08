"""Is a seed good, or is a seed only lucky? — the third axis of the VD sweep.

variance_decomp.py splits the IC spread into MONTH and SEED. That second term
is a bucket, and this script asks what is inside it: does seed 7 land high in
every month (a property of the seed -- an initialization the recipe happens to
like), or does it land high in one month and low in the next (a property of
nothing, which is what "noise" is supposed to mean)?

The distinction matters for how a single-seed comparison is read. Both cases
give the same sigma_seed, but only the second one shrinks as 1/sqrt(n): a
consistent seed ordering means the seeds are not exchangeable and pooling them
averages over a fixed effect rather than over noise.

THREE READINGS, because no single one of them is convincing alone:

  ANOVA        A one-way random-effects split (month vs seed) for the headline
               percentages, and a two-way additive fit (month + seed) that
               takes SEED IDENTITY out of the residual. The seed row's share
               of the within-month variance is the part of the noise that is
               repeatable.
  Friedman     Months as blocks, seeds as treatments, on RANKS. Distribution
               free, which matters here: the panel is 9 blocks deep and the
               per-month IC scales differ by 20x across tasks, so a test on
               raw values is a test on the biggest-scaled months.
  Mean ranks   The effect size, per seed, with the exchangeable null's own
               spread beside it. Under exchangeability a seed's mean rank has
               sd sqrt((n^2-1)/(12k)); the observed sd of the ten mean ranks
               divided by that is a ratio that reads as 1.0 for pure noise.

BALANCE. Friedman and the two-way fit need complete blocks, so a month missing
seeds is dropped from those two and named in the output. The one-way split
handles the unbalance directly (Satterthwaite's n0), so it keeps every month.

Usage::

    uv run plots/variance_decomp/seed_effects.py
    uv run plots/variance_decomp/seed_effects.py --readout probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from variance_decomp import collect  # noqa: E402
from variance_decomp_boxes import TASKS, TITLES, newest_tree, CKPT_PARENT  # noqa: E402


def one_way(by_month: dict) -> dict:
    """Month vs seed on the FULL panel, unbalanced-aware.

    Method of moments with Satterthwaite's effective cell size

        n0 = (N - sum(n_i^2)/N) / (k - 1)

    which reduces to n when the panel is balanced. Using a plain mean cell
    size instead would bias sigma^2_month by however much the cells differ --
    here one cell of 5 against eight of 10.
    """
    months = sorted(by_month)
    groups = [np.array([by_month[m][s] for s in sorted(by_month[m])], float)
              for m in months]
    ns = np.array([len(g) for g in groups], float)
    N, k = ns.sum(), len(groups)
    grand = float(np.concatenate(groups).mean())
    means = np.array([g.mean() for g in groups])
    ss_b = float((ns * (means - grand) ** 2).sum())
    ss_w = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    var_month = max((ms_b - ms_w) / n0, 0.0)
    var_seed = ms_w
    f = ms_b / ms_w
    return {
        "k": k, "N": int(N), "n0": float(n0),
        "grand": grand,
        "sd_month": float(np.sqrt(var_month)),
        "sd_seed": float(np.sqrt(var_seed)),
        "icc": float(var_month / (var_month + var_seed)),
        "F": float(f),
        "p": float(stats.f.sf(f, k - 1, N - k)),
        "range": float(means.max() - means.min()),
    }


def complete_matrix(by_month: dict) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    """(months x seeds) over the months that have every seed. Also the dropped."""
    seeds = sorted({s for sd in by_month.values() for s in sd})
    full = [m for m in sorted(by_month) if set(by_month[m]) == set(seeds)]
    dropped = [m for m in sorted(by_month) if m not in full]
    M = np.array([[by_month[m][s] for s in seeds] for m in full], float)
    return M, full, seeds, dropped


def two_way(M: np.ndarray) -> dict:
    """Additive month + seed fit on a complete block design, one obs per cell.

    With no replication the interaction is the residual, which is exactly the
    contrast wanted: ``var_seed_id`` is the part of the within-month scatter
    that repeats across months, ``var_resid`` the part that does not.
    """
    k, n = M.shape
    grand = M.mean()
    ss_month = n * ((M.mean(1) - grand) ** 2).sum()
    ss_seed = k * ((M.mean(0) - grand) ** 2).sum()
    ss_tot = ((M - grand) ** 2).sum()
    ss_resid = ss_tot - ss_month - ss_seed
    ms_seed = ss_seed / (n - 1)
    ms_resid = ss_resid / ((k - 1) * (n - 1))
    # EMS for a random seed effect in a two-way additive model: E[MS_seed] =
    # sigma^2_resid + k * sigma^2_seed.
    var_seed_id = max((ms_seed - ms_resid) / k, 0.0)
    f = ms_seed / ms_resid
    within = var_seed_id + ms_resid
    return {
        "sd_seed_id": float(np.sqrt(var_seed_id)),
        "sd_resid": float(np.sqrt(ms_resid)),
        "share_repeatable": float(var_seed_id / within) if within > 0 else 0.0,
        "F": float(f),
        "p": float(stats.f.sf(f, n - 1, (k - 1) * (n - 1))),
    }


def rank_stats(M: np.ndarray) -> dict:
    """Friedman + mean ranks per seed, against the exchangeable null."""
    k, n = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M)   # 1 = worst IC in its month
    mean_ranks = R.mean(0)
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(n)])
    null_sd = np.sqrt((n ** 2 - 1) / (12.0 * k))
    return {
        "mean_ranks": mean_ranks,
        "sd_mean_ranks": float(mean_ranks.std(ddof=1)),
        "null_sd": float(null_sd),
        "ratio": float(mean_ranks.std(ddof=1) / null_sd),
        "chi2": float(chi2), "p": float(p),
        # Kendall's W: 0 = the months disagree completely about seed order,
        # 1 = every month ranks the seeds identically.
        "W": float(chi2 / (k * (n - 1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--readout", choices=("head", "probe"), default="head")
    a = ap.parse_args()

    root = Path(a.ckpt_root) if a.ckpt_root else newest_tree(CKPT_PARENT)
    data = collect(root, metric="ic")
    print(f"reading {root}  (readout: {a.readout})\n")

    for series, task, style_key in TASKS:
        key = f"head:{task}" if a.readout == "head" else task
        by_month = data.get((series, key)) or {}
        if len(by_month) < 2:
            print(f"{TITLES.get(style_key, style_key)}: too few months\n")
            continue
        title = TITLES.get(style_key, style_key.replace("supervised_", "").title())
        print("=" * 72)
        print(f"{title}   ({len(by_month)} months, "
              f"{sum(len(v) for v in by_month.values())} runs)")
        print("=" * 72)

        ow = one_way(by_month)
        print(f"  MONTH vs SEED (one-way, all months, n0={ow['n0']:.2f})")
        print(f"    grand IC     {ow['grand']:+.4f}   "
              f"month range {ow['range']:.4f}")
        print(f"    sd_month     {ow['sd_month']:.4f}")
        print(f"    sd_seed      {ow['sd_seed']:.4f}")
        print(f"    share month  {100*ow['icc']:5.1f}%      "
              f"share seed {100*(1-ow['icc']):5.1f}%")
        print(f"    F({ow['k']-1},{ow['N']-ow['k']}) = {ow['F']:.1f}   "
              f"p = {ow['p']:.2e}")

        M, full, seeds, dropped = complete_matrix(by_month)
        print(f"\n  IS THE SEED NOISE REPEATABLE?  "
              f"({len(full)} complete months x {len(seeds)} seeds"
              + (f"; dropped {' '.join(dropped)}" if dropped else "") + ")")
        tw = two_way(M)
        print(f"    sd from seed identity  {tw['sd_seed_id']:.4f}")
        print(f"    sd residual            {tw['sd_resid']:.4f}")
        print(f"    repeatable share of the seed noise  "
              f"{100*tw['share_repeatable']:5.1f}%")
        print(f"    F({len(seeds)-1},{(len(full)-1)*(len(seeds)-1)}) = "
              f"{tw['F']:.2f}   p = {tw['p']:.4f}")

        rs = rank_stats(M)
        print(f"    Friedman chi2({len(seeds)-1}) = {rs['chi2']:.1f}   "
              f"p = {rs['p']:.4f}   Kendall W = {rs['W']:.3f}")
        print(f"    sd of mean ranks {rs['sd_mean_ranks']:.2f} vs "
              f"{rs['null_sd']:.2f} under exchangeability "
              f"(ratio {rs['ratio']:.2f})")
        order = np.argsort(-rs["mean_ranks"])
        print("    seed        " + "".join(f"{seeds[j]:>6d}" for j in order))
        print("    mean rank   " + "".join(
            f"{rs['mean_ranks'][j]:6.1f}" for j in order))
        print("    mean IC     " + "".join(
            f"{M[:, j].mean():+6.3f}" for j in order))
        print()


if __name__ == "__main__":
    main()
