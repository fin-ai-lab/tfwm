"""Collect the ViT-scale sweep into one json for the scaling figure.

Reads ``xs_ic.json`` beside each checkpoint of scripts/sweeps/
supervised_scaling.sh -- the run root and every ``<run>/<step>/`` the ladder
saved -- and writes one row per (eval month, task, scale, checkpoint) to
plots/metrics/supervised_scaling.json. plots/scaling/supervised_scaling.py
draws it.

THE X IS COUNTED, NOT TIMED. Every row carries ``flops``: the training FLOPs
that produced that checkpoint, from market_jepa/eval/flops.py applied to the
run's own state-dict shapes and the views the optimizer had seen
(``obs_seen`` x stocks per cell). A step costs the same in every month of a
scale, so a ladder rung is one x for all 31 months and the figure can average
across them. Under checkpoint.anneal_steps every step dir is an annealed
model and the root duplicates the last, so the root is dropped; under the
older save_steps ladder the root is the only annealed model and is flagged
``endpoint`` so the figure can draw it apart.

THE SHAPE IS READ OFF THE WEIGHTS. The saved config says ``n_info_channels:
0`` for every run (the dataset, not the config, decides that at runtime);
``backbone.pt`` carries the truth as the shape of ``info_proj.weight``. Only
the run root's weights travel (train_bundle_body.sh drops ``<step>/*.pt``),
and a run has one architecture, so the root's shapes bill every step.

THE READOUT IS THE HEAD. ``ic_head`` is the model's own forecast, and the
sweep scores nothing else (POST_TRAIN_PROBE=0). ``ic_probe`` is collected when
a checkpoint happens to carry it and is None otherwise.

ONE WAVE AT A TIME, like collect_ssl_finetune_breadth.py: the project name
carries the commit, waves at different commits are different recipes, and
the collector refuses to pool them unless told which one.

    uv run scripts/eval/collect_supervised_scaling.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from market_jepa.eval.flops import (  # noqa: E402
    count_params, shape_from_state_dicts, training_flops_per_view)

CKPT_ROOT = Path("/data/lab/market-jepa-checkpoints")
OUT = ROOT / "plots/metrics/supervised_scaling.json"

# supervised-scaling-<scale>-<commit6>-<train_start>-<train_end>, as the bundle
# names the project. The LR grid's projects are supervised-scaling-lr-<scale>
# and are excluded by the [a-z]+ on the scale.
PROJ_RE = re.compile(
    r"^supervised-scaling-(?P<scale>[a-z]+)-(?P<commit>[0-9a-f]{6,7})"
    r"-(?P<start>\d{4}-\d{2}-\d{2})-(?P<end>\d{4}-\d{2}-\d{2})$")


def eval_month_of(train_end: str) -> str:
    y, mo = (int(x) for x in train_end.split("-")[:2])
    return f"{y + 1:04d}-01" if mo == 12 else f"{y:04d}-{mo + 1:02d}"


def rows_per_obs(cfg: dict) -> int:
    """Views one unit of ``obs_seen`` stands for: a cell's stocks on the day
    backend, one row on mds. Mirrors collect_ssl_finetune_breadth."""
    ds = cfg.get("dataset", {}) or {}
    if ds.get("backend") != "days":
        return 1
    augs = ds.get("augmentations") or {}
    if isinstance(augs, dict):
        augs = list(augs.values())
    for a in augs or []:
        if isinstance(a, dict) and a.get("name") == "cross_stock":
            return int(a.get("n_stocks") or 1)
    raise SystemExit("backend=days without a cross_stock augmentation: "
                     "cannot tell how many views an obs_seen unit is")


def seq_len_of(cfg: dict) -> int:
    augs = (cfg.get("dataset", {}) or {}).get("augmentations") or {}
    if isinstance(augs, dict):
        augs = list(augs.values())
    for a in augs or []:
        if isinstance(a, dict) and a.get("global_seq_len"):
            return int(a["global_seq_len"])
    raise SystemExit("no global_seq_len in the run's augmentations")


def resolved_blr(cfg: dict) -> float | None:
    v = (cfg.get("optimizer", {}) or {}).get("blr")
    if v is None:
        v = ((cfg.get("mode", {}) or {}).get("training_overrides", {}) or {}).get("blr")
    return None if v is None else float(v)


def _ic(ic: dict, key: str) -> tuple[float | None, float | None]:
    v = ic.get(f"xs_ic/{key}")
    se = ic.get(f"xs_ic_se/{key}")
    return (None if v is None else float(v), None if se is None else float(se))


def collect(ckpt_root: Path, commit: str | None = None) -> list[dict]:
    import torch

    projects = []
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        m = PROJ_RE.match(d.name)
        if m and d.is_dir():
            projects.append((d, m))
    commits = sorted({m.group("commit") for _, m in projects})
    if commit:
        if commits and commit not in commits:
            raise SystemExit(f"no supervised-scaling projects at commit {commit}; "
                             f"found {', '.join(commits)}")
        projects = [(d, m) for d, m in projects if m.group("commit") == commit]
    elif len(commits) > 1:
        raise SystemExit(
            f"{len(commits)} waves of supervised-scaling are present: "
            f"{', '.join(commits)}.\nThese are different recipes and must not "
            f"be pooled -- pass --commit <hash> to pick one.")

    out, skipped = [], []
    for proj, m in projects:
        scale, wave = m.group("scale"), m.group("commit")
        ym = eval_month_of(m.group("end"))
        for run in sorted(p for p in proj.iterdir() if p.is_dir()):
            meta_f, bb_f, head_f = (run / "train_meta.json", run / "backbone.pt",
                                    run / "head.pt")
            if not (meta_f.exists() and bb_f.exists()):
                skipped.append(f"{proj.name}/{run.name}")
                continue
            meta = json.loads(meta_f.read_text())
            cfg = meta.get("config", {}) or {}
            task = meta.get("task")
            if not task:
                skipped.append(f"{proj.name}/{run.name} (no task)")
                continue
            bb_sd = torch.load(bb_f, map_location="cpu", weights_only=True)
            head_sd = (torch.load(head_f, map_location="cpu", weights_only=True)
                       if head_f.exists() else None)
            shape = shape_from_state_dicts(bb_sd, head_sd, seq_len=seq_len_of(cfg))
            per_view = training_flops_per_view(shape)
            k = rows_per_obs(cfg)
            # A sinusoidal table is a fixed buffer in the state dict, not a
            # parameter; the recipe's is (SupervisedModeConfig.backbone).
            pos = (((cfg.get("mode", {}) or {}).get("backbone", {}) or {})
                   .get("config", {}) or {}).get("pos_embed")
            fixed = ("position_embeddings",) if pos == "sinusoidal" else ()
            base = {
                "eval_month": ym, "train_end": m.group("end")[:7],
                "task": task, "scale": scale, "commit": wave,
                "run_id": run.name, "blr": resolved_blr(cfg),
                "hidden_size": shape.hidden_size,
                "params_backbone": count_params(bb_sd, exclude=fixed),
                "params_head": count_params(head_sd) if head_sd else 0,
                "flops_per_view": per_view,
                "warmup_steps": (cfg.get("optimizer", {}) or {}).get("warmup_steps"),
                "max_train_steps": meta.get("max_train_steps"),
            }

            # BRANCH-COOLDOWN RUNS (checkpoint.anneal_steps): every step dir
            # is an annealed model at its own count and the root is the last
            # of them again, so the root is dropped and nothing is an endpoint.
            # The older ladder (save_steps under WSD) is the reverse: raw
            # stable-phase rungs, and only the root annealed.
            branched = bool((cfg.get("checkpoint", {}) or {}).get("anneal_steps"))
            points = [] if branched else [(run, meta, True)]
            for sub in sorted((d for d in run.iterdir()
                               if d.is_dir() and d.name.isdigit()),
                              key=lambda d: int(d.name)):
                sm_f = sub / "train_meta.json"
                if sm_f.exists():
                    points.append((sub, json.loads(sm_f.read_text()), False))

            for d, dm, is_root in points:
                f = d / "xs_ic.json"
                if not f.exists():
                    skipped.append(str(d.relative_to(ckpt_root)))
                    continue
                ic = json.loads(f.read_text())
                step = dm.get("completed_steps")
                obs = dm.get("obs_seen")
                if step is None or obs is None:
                    skipped.append(f"{d.relative_to(ckpt_root)} (no step/obs_seen)")
                    continue
                views = int(obs) * k
                probe, se_p = _ic(ic, task)
                head, se_h = _ic(ic, f"head:{task}")
                out.append({
                    **base,
                    "step": int(step),
                    "annealed": bool(is_root or branched),
                    # The hollow marker: a run root that is the only annealed
                    # model of its run, drawn apart from its raw rungs.
                    "endpoint": bool(is_root and not branched),
                    "obs_seen": int(obs), "views": views,
                    "flops": per_view * views,
                    "ic_probe": probe, "se_probe": se_p,
                    "ic_head": head, "se_head": se_h,
                })
    if skipped:
        print(f"skipped {len(skipped)} unscored checkpoint(s), e.g. "
              f"{skipped[:3]}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    ap.add_argument("--commit", default=None,
                    help="the wave to collect; required when more than one is present")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = collect(a.ckpt_root, a.commit)
    if not rows:
        raise SystemExit(f"nothing collected under {a.ckpt_root}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    by = {}
    for r in rows:
        by.setdefault((r["scale"], r["task"]), set()).add(r["eval_month"])
    print(f"wrote {a.out}: {len(rows)} points")
    for (s, t), months in sorted(by.items()):
        print(f"  {s:6s} {t:24s} {len(months):2d} month(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
