"""Read the per-scale LR grid (scripts/sweeps/supervised_scaling_lr.sh).

One table per scale: the month-mean head and probe rank IC at each learning
rate, per task, with the paired difference from the recipe's 2e-4 where that
rate ran on the same months. Read the way the recipe's own LR was read
(SupervisedModeConfig): the head is the score, the probe is the check that
the features survived -- a rate whose head rises while its probe falls to the
floor is erasing what it is scoring.

Writes plots/metrics/supervised_scaling_lr.json beside the printout, and the
winner goes into VIT_SCALE_BLR by hand, with the date and the margin.

    uv run scripts/eval/summarize_supervised_scaling_lr.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = Path("lab/market-jepa-checkpoints")
OUT = ROOT / "plots/metrics/supervised_scaling_lr.json"
PROJ_RE = re.compile(
    r"^supervised-scaling-lr-(?P<scale>[a-z]+)-(?P<commit>[0-9a-f]{6,7})"
    r"-(?P<start>\d{4}-\d{2}-\d{2})-(?P<end>\d{4}-\d{2}-\d{2})$")
RUN_RE = re.compile(r"_blr(?P<blr>[0-9.e+-]+)$")
REFERENCE_BLR = 2e-4
TASKS = ("return_900", "volatility_change_900", "spread_change_900")


def collect(ckpt_root: Path) -> list[dict]:
    rows = []
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        m = PROJ_RE.match(d.name)
        if not (m and d.is_dir()):
            continue
        for run in sorted(p for p in d.iterdir() if p.is_dir()):
            meta_f, ic_f = run / "train_meta.json", run / "xs_ic.json"
            if not (meta_f.exists() and ic_f.exists()):
                continue
            meta = json.loads(meta_f.read_text())
            task = meta.get("task")
            rm = RUN_RE.search(str(meta.get("run_name", "")))
            if not (task and rm):
                continue
            ic = json.loads(ic_f.read_text())
            rows.append({
                "scale": m.group("scale"), "commit": m.group("commit"),
                "train_end": m.group("end")[:7], "task": task,
                "blr": float(rm.group("blr")),
                "ic_head": ic.get(f"xs_ic/head:{task}"),
                "ic_probe": ic.get(f"xs_ic/{task}"),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    """{scale: {task: {blr: {readout: (mean, n, paired_delta_vs_ref, n_paired)}}}}"""
    out: dict = {}
    for scale in sorted({r["scale"] for r in rows}):
        out[scale] = {}
        for task in TASKS:
            sub = [r for r in rows if r["scale"] == scale and r["task"] == task]
            if not sub:
                continue
            by_lr: dict[float, dict[str, dict]] = {}
            for r in sub:
                by_lr.setdefault(r["blr"], {})[r["train_end"]] = r
            ref = by_lr.get(REFERENCE_BLR, {})
            out[scale][task] = {}
            for blr, months in sorted(by_lr.items()):
                cell = {}
                for readout in ("ic_head", "ic_probe"):
                    vals = [m[readout] for m in months.values() if m[readout] is not None]
                    paired = [months[k][readout] - ref[k][readout]
                              for k in months if k in ref
                              and months[k][readout] is not None
                              and ref[k][readout] is not None]
                    cell[readout] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "n": len(vals),
                        "delta_vs_ref": float(np.mean(paired)) if paired else None,
                        "se_delta": (float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
                                     if len(paired) > 1 else None),
                        "n_paired": len(paired),
                    }
                out[scale][task][f"{blr:g}"] = cell
    return out


def print_table(summary: dict) -> None:
    for scale, tasks in summary.items():
        print(f"\n== ViT-{scale} ==")
        for task, by_lr in tasks.items():
            print(f"  {task}")
            print(f"    {'blr':>8s}  {'head':>8s} {'n':>2s}  {'d(2e-4)':>9s}   "
                  f"{'probe':>8s} {'n':>2s}  {'d(2e-4)':>9s}")
            best_h = max(by_lr, key=lambda b: by_lr[b]['ic_head']['mean'] or -9)
            best_p = max(by_lr, key=lambda b: by_lr[b]['ic_probe']['mean'] or -9)
            for blr, cell in by_lr.items():
                h, p = cell["ic_head"], cell["ic_probe"]

                def fmt(c):
                    mean = "   --   " if c["mean"] is None else f"{c['mean']:+.4f}"
                    d = ("    --   " if c["delta_vs_ref"] is None
                         else f"{c['delta_vs_ref']:+.4f}"
                         + (f"±{c['se_delta']:.4f}" if c["se_delta"] is not None else ""))
                    return f"{mean:>8s} {c['n']:2d}  {d:>9s}"
                mark = ("*" if blr == best_h else " ") + ("+" if blr == best_p else " ")
                print(f"  {mark} {blr:>8s}  {fmt(h)}   {fmt(p)}")
            print("    (* best head mean, + best probe mean)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = collect(a.ckpt_root)
    if not rows:
        raise SystemExit(f"no supervised-scaling-lr runs under {a.ckpt_root}")
    commits = sorted({r["commit"] for r in rows})
    if len(commits) > 1:
        print(f"WARNING: {len(commits)} waves present ({', '.join(commits)}); "
              f"pooled as one grid", file=sys.stderr)
    summary = summarize(rows)
    print_table(summary)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
