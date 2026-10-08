"""The supervised cell recipe on the ten holdout months, every sweep, against the random-init floor.

Reads the scored arms straight from lab/market-jepa-checkpoints (the
xs_ic.json post_train_ic_eval writes beside each backbone) and the rebuilt
randinit-fwd3 floor. One figure, holdout_sweeps.png: per task, every arm's head and probe IC
MINUS the floor, averaged over the floored holdout months that arm has run,
with the s.e. across months and n. Months are never shown individually; the
month-to-month spread is 4-20x the seed spread, so only the paired mean over
months ranks two arms.

Re-run as arms land; nothing is cached.
    uv run python plots/metrics/holdout_sweeps.py
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plots.style import apply_style  # noqa: E402

CKPT = Path("lab/market-jepa-checkpoints")
FLOOR_GLOB = "lab/score_results/randinit-fwd3-part*.json"
# One local seed on 2011-12 -> 2012-01 (floor_2011.py, 2026-09-11); the
# cluster floor has not reached that month.
LOCAL_FLOOR = {("2011-12", "return_900"): [0.0152], ("2011-12", "volatility_change_900"): [0.0981],
               ("2011-12", "spread_change_900"): [0.1231]}
TASKS = [("return_900", "Return"), ("volatility_change_900", "Volatility Change"), ("spread_change_900", "Spread Change")]
HOLD = "2009-01 2009-06 2011-03 2011-12 2014-05 2015-06 2016-03 2020-05 2021-03 2022-02".split()
# project prefix -> (label prefix, arm parser)
SWEEPS = {
    "supervised-cells-months-1fbd8c": ("", lambda rn: rn + " 10ep"),
    "supervised-return-batch-07fbc1": ("", lambda rn: rn),
    "supervised-return-cellsize-75b4b5": ("K ", lambda rn: rn),
    "supervised-return-span-a56a1b": ("span ", lambda rn: rn),
    "supervised-span-tasks-310df4": ("span ", lambda rn: rn),
}
def load_floor():
    fl = collections.defaultdict(list)
    for f in sorted(glob.glob(FLOOR_GLOB)):
        for r in json.load(open(f)):
            for t, _ in TASKS:
                fl[(r["fit_month"], t)].append(r[t])
    for k, v in LOCAL_FLOOR.items():
        fl[k] += v
    return fl


def load_arms():
    """(task, arm) -> month -> list of (head, probe); seeds/replicates append."""
    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    for proj, (prefix, parse) in SWEEPS.items():
        for f in CKPT.glob(f"{proj}-*/*/train_meta.json"):
            icf = f.with_name("xs_ic.json")
            if not icf.exists():
                continue
            m = json.load(open(f)); ic = json.load(open(icf)); t = m["task"]
            rn = m["run_name"]; month = rn[:7]
            if month not in HOLD:
                continue
            arm = re.sub(r"^\d{4}-\d{2}_", "", rn)
            arm = re.sub(r"^(return|vol_change|spread_change)_", "", arm)
            arm = prefix + parse(arm)
            arms[(t, arm)][month].append((ic.get("xs_ic/head:" + t), ic.get("xs_ic/" + t)))
    return arms


def summary_rows(arms, floor, t):
    rows = []
    for (tt, arm), d in arms.items():
        if tt != t:
            continue
        dh, dp = [], []
        for mo, vals in d.items():
            if not floor[(mo, t)]:
                continue
            fl = np.mean(floor[(mo, t)])
            dh.append(np.mean([h for h, _ in vals]) - fl); dp.append(np.mean([p for _, p in vals]) - fl)
        if dh:
            rows.append((arm, np.mean(dh), np.std(dh, ddof=1) / np.sqrt(len(dh)) if len(dh) > 1 else 0.0,
                         np.mean(dp), len(dh), sum(v > 0 for v in dh)))
    rows.sort(key=lambda r: r[1])
    return rows


def summary_figure(arms, floor, out):
    """One panel per task: every arm, head and probe IC minus the floor, averaged over its floored months."""
    apply_style()
    panels = [(t, title, summary_rows(arms, floor, t)) for t, title in TASKS]
    heights = [max(len(r), 3) for _, _, r in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 0.42 * sum(heights) + 2.4),
                             gridspec_kw={"height_ratios": heights})
    for ax, (t, title, rows) in zip(axes, panels):
        y = np.arange(len(rows))
        ax.barh(y + 0.18, [r[1] for r in rows], height=0.36, xerr=[r[2] for r in rows], color="#4c72b0",
                label="head - floor", capsize=2)
        ax.barh(y - 0.18, [r[3] for r in rows], height=0.36, color="#a8c4e6", label="probe - floor")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r[0]}   (n={r[4]}, {r[5]}/{r[4]} months above floor)" for r in rows], fontsize=8)
        ax.axvline(0, color="k", lw=0.8); ax.grid(axis="x", alpha=0.3)
        ax.set_title(title, fontsize=10, loc="left")
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("rank IC minus random-init floor, mean over the floored holdout months (s.e. across months)", fontsize=9)
    fig.suptitle("Supervised cell recipe, ten holdout months: every arm, floor-subtracted", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


if __name__ == "__main__":
    floor = load_floor(); arms = load_arms()
    here = Path(__file__).resolve().parent
    summary_figure(arms, floor, here / "holdout_sweeps.png")
    n = sum(len(d) for d in arms.values())
    print(f"{n} (arm, month) cells over {len(arms)} arms -> {here/'holdout_sweeps.png'}")
