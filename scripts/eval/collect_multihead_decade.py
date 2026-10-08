"""The decade multihead runs, as metric rows for supervised_scaling_decade.py.

WHAT THESE RUNS ARE. One multihead trunk (return + volatility change + spread
change) per scale, trained on the DAY STORE over 2008-01..2017-12 and scored on
2018-01 -- ten years of training behind a single evaluation month. They are a
different experiment from supervised_scaling.json's 31 six-month specialists
and are deliberately kept in their own file and their own figure: same ladder,
same annealed rungs, different span and a different number of eval months.

WHY THE FLOPS CARRY A FACTOR OF THREE. MultiTaskSupervisedModel calls
torch.autograd.grad(objective, shared_parameters) once PER TASK to normalize
task gradients over the trunk, so a step costs one forward plus three task
backwards plus the real one -- nine forward-equivalents per view against the
three that training_flops_per_view counts for a single-task specialist. The
x-axis is compute actually spent, so the multiplier belongs on it; without it
every rung would sit at a third of its true cost.

NO STANDARD ERROR. Each point is ONE run scored on ONE month, so there is no
distribution over months to take an s.e. of.

    uv run scripts/eval/collect_multihead_decade.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from market_jepa.eval.flops import (  # noqa: E402
    shape_for_scale, training_flops_per_view)
from market_jepa.schemas import VIT_SCALES  # noqa: E402

RUNS = ROOT / "data/decade_runs"
OUT = ROOT / "plots/metrics/supervised_scaling_decade.json"
TASKS = ("return_900", "volatility_change_900", "spread_change_900")
MULTIHEAD_BACKWARDS = 3


def scale_of(hidden: int) -> str | None:
    for name, d in VIT_SCALES.items():
        if int(d["hidden_size"]) == hidden:
            return name
    return None


def collect(runs: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_f in sorted(runs.glob("*/[0-9]*/train_meta.json")):
        rung = meta_f.parent
        ic_f = rung / "xs_ic.json"
        if not ic_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        cfg = meta.get("config", {}) or {}
        bb = ((cfg.get("mode", {}) or {}).get("backbone", {}) or {})
        hidden = int((bb.get("config", {}) or {}).get("hidden_size") or 0)
        scale = scale_of(hidden)
        obs, step = meta.get("obs_seen"), meta.get("completed_steps")
        if scale is None or obs is None or step is None:
            continue
        ds = cfg.get("dataset", {}) or {}
        augs = ds.get("augmentations") or {}
        if isinstance(augs, dict):
            augs = list(augs.values())
        k = next((int(a.get("n_stocks") or 1) for a in augs
                  if a.get("name") == "cross_stock"), None)
        seq = next((int(a["global_seq_len"]) for a in augs
                    if a.get("global_seq_len")), None)
        if not k or not seq:
            continue
        views = int(obs) * k
        flops = (training_flops_per_view(shape_for_scale(scale, seq_len=seq))
                 * views * MULTIHEAD_BACKWARDS)
        ic = json.loads(ic_f.read_text())
        for task in TASKS:
            v = ic.get(f"xs_ic/head:{task}")
            if v is None:
                continue
            rows.append({
                "scale": scale, "task": task, "step": int(step),
                "flops": float(flops), "views": views, "ic_head": float(v),
                "train_end": str(ds.get("train_date_end", ""))[:7],
                "eval_month": str(ds.get("eval_date_start", ""))[:7],
                "run": rung.parent.name,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = collect(a.runs)
    if not rows:
        raise SystemExit(f"no scored rungs under {a.runs}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}  ({len(rows)} rows)")
    for scale in sorted({r["scale"] for r in rows}):
        steps = sorted({r["step"] for r in rows if r["scale"] == scale})
        print(f"  {scale:6s} {len(steps)} rung(s): {steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
