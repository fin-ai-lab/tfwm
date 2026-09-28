"""What the reported supervised model cost to train, so the scaling figure
can mark it.

THE QUESTION the star on plots/scaling/supervised_scaling.png answers: where
on the compute axis does the paper's own supervised specialist sit? Its
recipe is not a FLOPs target -- it is "twelve passes over the trailing six
months" -- so its compute is a property of each month's data and has to be
MEASURED, one month at a time, and then averaged the way the figure averages
everything else.

FLOPS THE SAME WAY THE CURVE COUNTS THEM: 3 x an analytic forward per view,
times the views the optimizer saw (``obs_seen`` x the cells' stocks). Same
function, same architecture, so the star and the curve are on one ruler
(market_jepa/eval/flops.py, scripts/eval/collect_supervised_scaling.py).

WHERE THE STEP COUNT COMES FROM. The reported specialists
(``supervised-full-month-*``) were trained before ``train_meta.json`` carried
``obs_seen``, so their own metas cannot say. What can say is any OTHER run of
the SAME recipe on the same month: the step count of "12 passes over this
span at 256 cells a step" is a property of the DATA, not of the learning-rate
schedule, so the superseded stable-rung wave (supervised-scaling-small-
762334, 12 epochs, same span, same batch, same architecture) measures it
exactly. That wave must never be POOLED into the figure's curve
([[supervised-scaling-experiment-design]]) -- this is not pooling, it is
reading a step count off it.

So a run counts here if its CONFIG is the reported recipe -- supervised, day
store, six-month span, 12 epochs (no ``max_train_steps``, no
``anneal_steps``), ViT-Small, 256 cells x 16 stocks a step -- whoever ran it.
The measured head IC of the reported specialists themselves is carried along
for comparison; the figure does not draw it.

    uv run scripts/eval/default_model_flops.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "plots"))

from collect_supervised_scaling import (  # noqa: E402
    CKPT_ROOT, eval_month_of, rows_per_obs, seq_len_of)
from market_jepa.eval.flops import (  # noqa: E402
    shape_for_scale, training_flops_per_view)
from market_jepa.schemas import VIT_SCALES  # noqa: E402
from style import load_sweep_months  # noqa: E402

OUT = ROOT / "plots/metrics/supervised_default_model.json"
# The reported model is the recipe's own scale, and the figure's middle curve.
SCALE = "small"
TASKS = ("return_900", "volatility_change_900", "spread_change_900")
# supervised-full-month-<task>-<commit6>-<span>: the REPORTED specialists, and
# the only runs whose score is the reported model's. Other runs of the same
# recipe measure its step count (above) but not its IC -- they hold a
# different rate while spending the same FLOPs.
SPECIALIST_RE = re.compile(r"^supervised-full-month-[a-z-]+-(?P<commit>[0-9a-f]{6,7})-")


def is_the_recipe(cfg: dict, hidden: int) -> bool:
    """Is this run the reported supervised specialist's recipe?

    Everything here changes the step count or the cost of a step. The LR, the
    schedule and the task do not, and are deliberately not checked: a run that
    trains the same model on the same data for the same number of passes costs
    the same FLOPs whatever rate it holds while doing it.
    """
    ds = cfg.get("dataset", {}) or {}
    tr = cfg.get("training", {}) or {}
    bb = cfg.get("backbone", {}) or {}
    if ds.get("backend") != "days" or int(ds.get("train_span_months") or 0) != 6:
        return False
    if int(tr.get("num_epochs") or 0) != 12 or tr.get("max_train_steps"):
        return False
    if (cfg.get("checkpoint", {}) or {}).get("anneal_steps"):
        return False
    if int(bb.get("d_embedding") or 0) != hidden:
        return False
    ov = ((cfg.get("mode", {}) or {}).get("training_overrides", {}) or {})
    return int(ov.get("effective_batch_size") or 0) == 256


def best_ic(ic: dict[tuple[str, str], dict[str, float]]) -> dict:
    """The widest wave of specialists per task: months first, then the later
    commit, because a re-run wave supersedes the one it re-ran."""
    out: dict[str, dict] = {}
    for task in TASKS:
        waves = {c: v for (t, c), v in ic.items() if t == task and v}
        if not waves:
            continue
        commit = max(waves, key=lambda c: (len(waves[c]), c))
        out[task] = {"mean": statistics.fmean(waves[commit].values()),
                     "n_months": len(waves[commit]), "commit": commit}
    return out


def measure(ckpt_root: Path, months: list[str] | None) -> dict:
    hidden = VIT_SCALES[SCALE]["hidden_size"]
    per_month: dict[str, dict] = {}
    # (task, commit) -> {train_end: head IC}: one wave of specialists is the
    # reported one, and the others are its ancestors. Kept apart and picked
    # between at the end, never averaged together.
    ic: dict[tuple[str, str], dict[str, float]] = {}
    projects = set()
    for proj in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        if not proj.name.startswith("supervised-"):
            continue
        for run in sorted(p for p in proj.iterdir() if p.is_dir()):
            meta_f = run / "train_meta.json"
            if not meta_f.exists():
                continue
            meta = json.loads(meta_f.read_text())
            cfg = meta.get("config", {}) or {}
            if not is_the_recipe(cfg, hidden):
                continue
            # Run name = the YYYY-MM the training span ends on, as every
            # supervised sweep names it; the figure keys on the eval month.
            ym = str(meta.get("run_name", ""))[:7]
            if months is not None and ym not in months:
                continue
            obs, steps = meta.get("obs_seen"), meta.get("completed_steps")
            ic_f = run / "xs_ic.json"
            task = meta.get("task")
            spec = SPECIALIST_RE.match(proj.name)
            if spec and ic_f.exists() and task in TASKS:
                v = json.loads(ic_f.read_text()).get(f"xs_ic/head:{task}")
                if v is not None:
                    ic.setdefault((task, spec.group("commit")), {})[ym] = float(v)
            if obs is None or steps is None:
                continue
            views = int(obs) * rows_per_obs(cfg)
            flops = training_flops_per_view(
                shape_for_scale(SCALE, seq_len=seq_len_of(cfg))) * views
            prev = per_month.get(ym)
            if prev and prev["steps"] != int(steps):
                raise SystemExit(
                    f"two runs of the recipe disagree on {ym}: "
                    f"{prev['steps']} steps vs {int(steps)} -- the step count "
                    f"is the data's, so one of them is not this recipe")
            per_month[ym] = {"train_end": ym, "eval_month": eval_month_of(ym),
                             "steps": int(steps), "views": views,
                             "flops": int(flops)}
            projects.add(proj.name)
    if not per_month:
        raise SystemExit(
            f"no run of the reported recipe under {ckpt_root} carries a step "
            f"count; the star has nothing to stand on")
    xs = [m["flops"] for m in per_month.values()]
    out = {
        "scale": SCALE, "label": "Reported Model",
        "n_months": len(per_month),
        # The axis is log, so the months average in log: this is the x the
        # figure draws the star at.
        "flops": math.exp(statistics.fmean(math.log(x) for x in xs)),
        "flops_mean": statistics.fmean(xs),
        "flops_min": min(xs), "flops_max": max(xs),
        "steps_mean": statistics.fmean(m["steps"] for m in per_month.values()),
        "per_month": [per_month[k] for k in sorted(per_month)],
        # The reported score of the reported model, for comparison with the
        # curve the star sits on. The figure does not draw it.
        "ic_head": best_ic(ic),
        "flops_from": sorted(projects),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    ap.add_argument("--all-months", action="store_true",
                    help="measure every month found, not just the reported set")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    got = measure(a.ckpt_root, None if a.all_months else load_sweep_months())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(got, indent=1))
    print(f"wrote {a.out}")
    print(f"  {got['n_months']} month(s), {got['steps_mean']:.0f} steps mean")
    print(f"  {got['flops']:.3e} FLOPs "
          f"({got['flops_min']:.2e} - {got['flops_max']:.2e})")
    for t, v in got["ic_head"].items():
        print(f"  head IC {t:24s} {v['mean']:+.4f} over {v['n_months']} month(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
