"""Collect the SSL finetune breadth sweep into one json for the figure.

Reads ``xs_ic.json`` beside each finetune checkpoint -- the FILESYSTEM, not
W&B, whose project listings report run state wrongly in both directions -- and
writes one row per (eval month, task, label budget).

EVERY ROW CARRIES BOTH THE COUNT AND ITS DENOMINATOR. ``rows`` is the label
budget in a36 panel rows, which is what
plots/core/probe_fit_breadth.py puts on its x axis; ``n_rows_pool`` is the
full six-month pool the same month's ridge was fit on. So a row count converts
to a percentage of the span (and back) without re-deriving anything, and a
change in either one stays visible. ``frac`` is what the trainer was actually
given, kept as a cross-check: it should equal rows/n_rows_pool.

TWO AXES IN ONE FILE. ``rows`` is the label budget, and ``progress`` is how
far through its own training a point is (1.0 at the run root, step/max inside
it). probe_fit_breadth.py draws only ``progress == 1.0`` -- a mid-run
checkpoint has not annealed under the warmup+cosine schedule and is not a
converged fit at that budget. The rest is the within-arm training curve.

THE READOUT IS THE PROBE, NOT THE HEAD. Every arm in this project is reported
on a ridge probe over the frozen embeddings, and the finetune is no exception --
the ridge-initialized head is a training device, not a reporting one. The head
column is collected too, because a finetune whose head has drifted far from its
own probe is worth knowing about, but the curve is the probe.

    uv run scripts/eval/collect_ssl_finetune_breadth.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# THE NOMINAL LADDER, kept in step with SWEEP_TARGET_ROWS in
# scripts/sweeps/ssl_finetune/ssl_finetune_breadth.sh.
LADDER = (8192, 32768, 131072, 393216, 786432, 1152000,
          2097152, 4194304, 8388608)
# A checkpoint is taken at a STEP, so what it consumed is a whole number of
# batches and lands near its target rather than on it: 32,512 or 33,024 where
# 32,768 was asked for, which is 0.78% at worst over the whole wave. The rungs
# are 4x apart at the bottom and 1.46x at the tightest, so 3% cannot reach
# the wrong one.
RUNG_TOL = 0.03

CKPT_ROOT = Path("/data/lab/market-jepa-checkpoints")
RIDGE_ROOT = Path("/data/lab/market-jepa-checkpoints/_scratch/ridge_head_init")
OUT = Path(__file__).resolve().parents[2] / "plots/metrics/ssl_finetune_breadth.json"

# ft_<task>_rows-<n>_frac-<f>, as ssl_finetune_breadth.sh names its runs.
# ft_<task>_wsd, as ssl_finetune_breadth.sh names its runs. The older
# ft_<task>_rows-<n>_frac-<f> form was the pre-WSD wave, one run per budget.
# The trailing _blr<rate> is optional: it was added when the pilot needed
# several LRs in one project (skip_if_done keys on the run name), and
# wave 047986 and earlier have no suffix. The RATE IS NOT READ FROM HERE --
# `blr` comes from the run's own train_meta, which cannot drift from what
# actually ran.
RUN_RE = re.compile(r"^ft_(?P<task>[a-z_0-9]+)_wsd(?:_blr[0-9eE.+-]+)?$")
# The project is <sweep>-<commit>-<train_start>-<train_end>; the EVAL month is
# the one after train_end, and that is what everything else is keyed by.
PROJ_RE = re.compile(r"-(?P<start>\d{4}-\d{2}-\d{2})-(?P<end>\d{4}-\d{2}-\d{2})$")
# <sweep>-<commit6>-<start>-<end>: the commit is what separates one wave from
# another, and waves are NOT poolable -- the 2026-09-16 wave ran at the
# inherited blr 2e-4 and lost IC to it, the next at 1e-5. Pooling them would
# average a recipe change into the curve with nothing on the axis to say so.
COMMIT_RE = re.compile(r"-([0-9a-f]{6,7})-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}$")


def eval_month_of(project: str) -> str | None:
    m = PROJ_RE.search(project)
    if not m:
        return None
    y, mo = (int(x) for x in m.group("end").split("-")[:2])
    return f"{y + 1:04d}-01" if mo == 12 else f"{y:04d}-{mo + 1:02d}"


def _rows_per_obs(dm: dict) -> int:
    """How many labelled rows one unit of ``obs_seen`` stands for.

    1 on the mds backend, where a batch entry is a row. On the day-major
    backend a batch entry is a CELL holding ``n_stocks`` tickers at one
    anchor, so a run reports a sixteenth of the rows it actually trained on
    under the recipe's cross_stock(n_stocks=16).
    """
    cfg = dm.get("config", {}) or {}
    if (cfg.get("dataset", {}) or {}).get("backend") != "days":
        return 1
    augs = (cfg.get("dataset", {}) or {}).get("augmentations") or {}
    if isinstance(augs, dict):
        augs = list(augs.values())
    for a in augs or []:
        if isinstance(a, dict) and a.get("name") == "cross_stock":
            return int(a.get("n_stocks") or 1)
    raise SystemExit(
        "backend=days without a cross_stock augmentation: cannot tell how "
        "many rows an obs_seen unit stands for, and guessing would put every "
        "point on the wrong x.")


def rung(n: int | None, annealed: bool) -> int | None:
    """The nominal budget a ladder checkpoint stands for, or None.

    WHY THE FIGURE CANNOT USE `rows` DIRECTLY. probe_fit_breadth.py draws only
    the budgets EVERY month reached, so its x has to be shared across months.
    `rows` is what each run actually consumed and differs month to month by a
    batch or two, so keying on it leaves every rung with one month on it and
    the line disappears -- 8,192 is the only count 31 months happen to agree
    on, because it is the one rung small enough to be a whole number of steps
    everywhere.

    ANNEALED ROWS ARE NEVER SNAPPED, and that is not a detail: 12 of the 77
    run roots in the first wave sit within tolerance of a rung by coincidence.
    A run root is the decayed endpoint at that month's OWN full pool, which is
    a per-month quantity the figure draws as the hollow marker; folding one
    into a shared rung would put an annealed model on a curve of
    stable-phase ones.
    """
    if annealed or not n:
        return None
    best = min(LADDER, key=lambda t: abs(n - t))
    return best if abs(n - best) <= RUNG_TOL * best else None


def pool_sizes(series: str) -> dict[str, int]:
    idx = RIDGE_ROOT / series / "index.json"
    if not idx.exists():
        return {}
    return {r["eval_month"]: int(r["n_rows_pool"])
            for r in json.loads(idx.read_text())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="ssl-ft-breadth-pair-warp")
    ap.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT,
                    help="checkpoint tree to scan (override for tests)")
    ap.add_argument("--commit", default=None,
                    help="the wave to collect, as the 6-char commit in the "
                         "project name. Required when the tree holds more "
                         "than one: different waves are different recipes.")
    ap.add_argument("--series", default="pair_warp_6mo")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ckpt_root = args.ckpt_root
    pools = pool_sizes(args.series)

    # ONE WAVE AT A TIME, AND SAY SO RATHER THAN GUESS. Picking the newest
    # would silently drop a wave someone is still filling in; pooling them
    # would mix recipes. Both are worse than stopping.
    projects = sorted(ckpt_root.glob(f"{args.sweep}-*"))
    commits = sorted({m.group(1) for d in projects
                      if (m := COMMIT_RE.search(d.name))})
    if args.commit:
        if args.commit not in commits and commits:
            raise SystemExit(
                f"no {args.sweep} projects at commit {args.commit}; "
                f"found {', '.join(commits)}")
        projects = [d for d in projects
                    if (m := COMMIT_RE.search(d.name)) and m.group(1) == args.commit]
    elif len(commits) > 1:
        raise SystemExit(
            f"{len(commits)} waves of {args.sweep} are present: "
            f"{', '.join(commits)}.\nThese are different recipes and must not "
            f"be pooled — pass --commit <hash> to pick one, or delete the "
            f"wave you do not want."
        )
    # NAME THE WAVE ACTUALLY COLLECTED. This used to print commits[0] -- the
    # first of every commit PRESENT, not the one selected -- so collecting the
    # pilot while the old wave was still on disk announced the old wave's hash
    # over the new wave's numbers. The data was right and the label was wrong,
    # which is the worse of the two failures.
    chosen = args.commit or (commits[0] if commits else None)
    print(f"collecting {len(projects)} project(s)"
          + (f" at commit {chosen}" if chosen else ""))
    rows_out, skipped = [], []
    for proj in projects:
        ym = eval_month_of(proj.name)
        if ym is None:
            continue
        for run in sorted(p for p in proj.iterdir() if p.is_dir()):
            meta_f = run / "train_meta.json"
            if not ((run / "xs_ic.json").exists() and meta_f.exists()):
                skipped.append(f"{proj.name}/{run.name}")
                continue
            meta = json.loads(meta_f.read_text())
            m = RUN_RE.match(str(meta.get("run_name", "")))
            if not m:
                continue
            task = m.group("task")
            total = meta.get("max_train_steps")

            # The run root is the DECAYED endpoint; every <step>/ subdirectory
            # is a stable-phase checkpoint. They share a run and a task and
            # differ only in how many labels they have consumed.
            points = [(run, meta)]
            for sub in sorted((d for d in run.iterdir()
                               if d.is_dir() and d.name.isdigit()),
                              key=lambda d: int(d.name)):
                sm_f = sub / "train_meta.json"
                if not ((sub / "xs_ic.json").exists() and sm_f.exists()):
                    continue
                points.append((sub, json.loads(sm_f.read_text())))

            for d, dm in points:
                f = d / "xs_ic.json"
                if not f.exists():
                    continue
                ic = json.loads(f.read_text())
                # HEAD FIRST. This sweep scores head-only (POST_TRAIN_PROBE=0),
                # which is what makes it cheap: the a36 probe-fit month is never
                # embedded. The head IS the model's forecast here, and because
                # it was initialized from the ridge probe it starts exactly at
                # that probe's value. The probe key is read when present so a
                # probe-scored run still collects.
                head = ic.get(f"xs_ic/head:{task}")
                probe = next((k for k in ic
                              if k.startswith("xs_ic/") and k.endswith(task)
                              and not k.startswith("xs_ic/head:")), None)
                probe_ic = ic.get(probe) if probe else None
                steps = dm.get("completed_steps")
                # obs_seen COUNTS CELLS ON THE DAY BACKEND. It is
                # steps x batch x accum, and on backend=days a batch entry is
                # a CELL of n_stocks labelled tickers, so the run saw
                # n_stocks times as many ROWS as obs_seen says. On mds a batch
                # entry is one row and the factor is 1. The axis of this file
                # is ROWS -- the same quantity the frozen-probe curve puts on
                # its x -- so the factor is applied here rather than left for
                # the figure to guess.
                obs = dm.get("obs_seen")
                if obs is not None:
                    obs = int(obs) * _rows_per_obs(dm)
                annealed = (d is run)
                rows_out.append(dict(
                    eval_month=ym, task=task,
                    # THE BUDGET, counted by the trainer rather than derived:
                    # observations consumed by this checkpoint.
                    rows=obs,
                    # The nominal budget this rung stands for; None at the run
                    # root, whose budget is that month's own pool. The figure
                    # keys on this, `rows` stays the measured truth.
                    rung=rung(obs, annealed),
                    step=steps, max_train_steps=dm.get("max_train_steps") or total,
                    progress=(steps / total) if (steps and total) else None,
                    # The run root decayed; the ladder rungs did not.
                    annealed=annealed,
                    n_rows_pool=pools.get(ym),
                    ic=head if head is not None else probe_ic,
                    ic_head=head, ic_probe=probe_ic,
                    se=ic.get(f"xs_ic_se/head:{task}"),
                    n_cells=ic.get(f"xs_ic_cells/head:{task}"),
                    run=f"{proj.name}/{run.name}",
                    # THE LEARNING RATE, so two waves cannot be told apart
                    # only by their commit hash. The run NAME does not carry
                    # it (ft_<task>_wsd is the same at every LR), so a figure
                    # drawn from a pooled file would otherwise label an arm by
                    # a number nothing in the row states.
                    blr=(dm.get("config", {}).get("optimizer", {}) or {}).get("blr"),
                ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows_out, indent=1))
    months = {r["eval_month"] for r in rows_out}
    print(f"{args.out}: {len(rows_out)} runs over {len(months)} eval months")
    if skipped:
        print(f"{len(skipped)} run dirs had no xs_ic.json yet (unscored or "
              f"still training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
