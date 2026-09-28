"""Re-score every checkpoint that shares a (fit month, eval month) pair, on ONE
decode of the panel.

WHY THIS EXISTS. ``post_train_ic_eval.py`` scores a checkpoint at the tail of
the job that produced it, where one checkpoint is all there is. Re-scoring an
archived sweep is a different shape: the bins x penalty grid puts ~30
checkpoints on the same training month, and running the in-job hook 30 times
pays for 30 identical decodes of the same two months to change only which
encoder consumes them. The decode is the cost — a ViT-384 forward runs the GPU
at ~47% duty against it — so the batch is close to a 30x saving on the part
that dominates, and the marginal checkpoint costs about one forward pass.

It is also STRICTER than looping the hook. Every checkpoint here is scored on
one physically identical panel, so a difference between two cells of the figure
cannot be a difference in which views were drawn.

NOTHING ABOUT THE MEASUREMENT MOVES. The panel comes from xs_ic_eval.iter_panel
and the numbers from xs_ic_eval.score — the same generator and the same scorer
the in-job hook calls. The recomputed IC is compared against whatever the
checkpoint already has on disk and the drift is printed; it should be ~0, and a
non-zero one means the panel moved under the sweep, not that the AUC is new.

Writes ``xs_ic.json`` beside each checkpoint, replacing it: the IC keys are
recomputed identically and the AUC keys are added.

Usage:
    uv run scripts/eval/xs_score_many.py \
        --train-month 2009-06 --eval-month 2009-07 \
        --ckpt-glob '/data/lab/market-jepa-checkpoints/supervised-bins-penalty-*-2009-06-01-*/*/'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/generic"))

from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.eval.checkpoints import load_model  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
# The config-from-disk resolution and the AUC schema tag live with the in-job
# hook; importing them keeps one definition of "what this checkpoint is".
from post_train_ic_eval import AUC_SCHEMA, _load_cfg  # noqa: E402
from xs_ic_eval import panel_kwargs_for  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    AUC_FIT_ROWS, AUC_TASKS, AUC_TOL, TRAIN_ANCHORS_PER_DAY, day_anchors,
    embed_month_many, head_readout, score,
)


def parse_args():
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--train-month", required=True, help="YYYY-MM, fits the probe")
    p.add_argument("--eval-month", required=True, help="YYYY-MM, reports the metric")
    p.add_argument("--ckpt-glob", default=None,
                   help="shell glob of checkpoint dirs (quote it)")
    p.add_argument("--ckpt-dirs", nargs="*", default=[])
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--anchors-per-day", type=int, default=8)
    p.add_argument("--train-anchors-per-day", type=int,
                   default=TRAIN_ANCHORS_PER_DAY)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--auc-bins", type=int, default=5)
    p.add_argument("--auc-fit-rows", type=int, default=AUC_FIT_ROWS,
                   help="subsample the logistic's FIT rows to this many "
                        "(0 = all). The ridge always sees the whole month.")
    p.add_argument("--auc-tol", type=float, default=AUC_TOL,
                   help="lbfgs stopping tolerance for the logistic probe")
    p.add_argument("--auc-tasks", nargs="*", default=None,
                   help="targets to fit the logistic probe on. Default: the "
                        "checkpoint's OWN head task, and all of "
                        f"{list(AUC_TASKS)} for a checkpoint without a head. "
                        "The logistic is the expensive fit on this path — "
                        "~30 s per target per checkpoint against a ridge's "
                        "milliseconds — and a supervised arm's figure only "
                        "ever reads the task it was trained on.")
    p.add_argument("--skip-scored", action="store_true",
                   help="skip checkpoints whose xs_ic.json already carries "
                        "this AUC schema — makes the pass resumable")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N checkpoints (a smoke test)")
    p.add_argument("--smoke-shards", type=int, default=0,
                   help="use only the first N MDS shards of each month. The "
                        "panel is then PARTIAL and the numbers are not the "
                        "reported ones, so nothing is written to disk — this "
                        "exists to exercise the path in a minute instead of "
                        "twenty.")
    return p.parse_args()


def _stats_name(p) -> str:
    """Basename of an anchor-stat directory — the TARGET DEFINITION's name."""
    return str(p).rstrip("/").rsplit("/", 1)[-1]


def _needs_scoring(d: Path, skip: bool, stats_name: str) -> bool:
    if not skip:
        return True
    f = d / "xs_ic.json"
    if not f.is_file():
        return True
    try:
        prev = json.loads(f.read_text())
    except json.JSONDecodeError:
        return True
    # A score carries the target it was measured against. An unstamped file
    # predates the stamp, which means the retired mid-to-mid family -- so it
    # satisfies "already scored" only when that is what was asked for.
    if prev.get("xs_anchor_stats", "xs_anchor_stats") != stats_name:
        return True
    return prev.get("xs_auc_schema") != AUC_SCHEMA


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)

    dirs = [Path(d) for d in args.ckpt_dirs]
    if args.ckpt_glob:
        dirs += [Path(d) for d in sorted(glob.glob(args.ckpt_glob))]
    dirs = [d for d in dirs if (d / "backbone.pt").is_file()
            or (d / "model.pt").is_file()]
    stats_name = _stats_name(args.xs_anchor_stats_dir)
    dirs = [d for d in dirs if _needs_scoring(d, args.skip_scored, stats_name)]
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        print("nothing to score")
        return

    print(f"==> {len(dirs)} checkpoint(s), {args.train_month} -> "
          f"{args.eval_month}", flush=True)
    models = {}
    panel_by_dir = {}
    for d in dirs:
        cfg = _load_cfg(d, None)
        m = load_model(str(d), cfg, device)
        m.eval()
        models[str(d)] = m
        panel_by_dir[str(d)] = panel_kwargs_for(cfg)
    # One pass feeds every model the SAME views, so a checkpoint trained
    # without normalization -- or at a single resolution -- cannot ride along
    # with default ones: it would be scored on a tensor it never trained on.
    # Refuse the mixed batch instead of silently picking one.
    # Derived from the panel dict ITSELF rather than from a hand-written tuple
    # of its keys: this guard listed (nonorm, fixed_agg, norm stats) and so went
    # blind the moment seq_len was added to panel_kwargs_for -- a 256-token arm
    # would have batched with a 2048-token one and been scored on eight times
    # the context it trained on, which is precisely what the guard exists to
    # stop. Anything panel_kwargs_for returns now separates batches.
    def _kind(v):
        return tuple(sorted((k, tuple(x) if isinstance(x, list) else x)
                            for k, x in v.items()))
    kinds = {_kind(v) for v in panel_by_dir.values()}
    if len(kinds) > 1:
        raise SystemExit(
            "one batch mixes panels:\n  "
            + "\n  ".join(repr(dict(k)) for k in sorted(kinds))
            + "\nScore each arm separately -- they need different views.")
    panel_kw = next(iter(panel_by_dir.values()), {})
    tag = ""
    if panel_kw.get("norm_groups") == []:
        tag += "  [norm_mode=none]"
    if panel_kw.get("fixed_agg"):
        tag += f"  [fixed_agg={panel_kw['fixed_agg']}]"
    # The information token is on by default since 2026-08-25, so the tag marks
    # a checkpoint that LACKS it -- the unusual case now.
    if not panel_kw.get("info_norm_stats", True):
        tag += "  [no info norm stats]"
    if not panel_kw.get("info_window", True):
        tag += "  [no info window]"
    print(f"==> loaded {len(models)} model(s) onto {device}{tag}", flush=True)

    stats_dir = Path(args.xs_anchor_stats_dir)
    mosaic = Path(args.mosaic_dir)
    anchors = {args.train_month: day_anchors(args.train_anchors_per_day),
               args.eval_month: day_anchors(args.anchors_per_day)}

    caches, stats = {}, {}
    for ym in (args.train_month, args.eval_month):
        y, m = ym.split("-")
        print(f"==> embedding {ym} for all {len(models)} ...", flush=True)
        stats[ym] = AnchorStats(stats_dir / f"{ym}.npz")
        caches[ym] = embed_month_many(
            models, mosaic / y / m, ym, stats[ym], schedule, anchors[ym],
            device, args.batch_size,
            # Shard 0 of N is the first 1/N of the month's shards.
            num_shards=max(args.smoke_shards, 1),
            **panel_kw,
        )
        any_panel = next(iter(caches[ym].values()))
        print(f"    {len(any_panel['X'])} rows x {len(models)} encoders",
              flush=True)

    drifts = []
    retargeted = []          # scores preserved from a different target
    for key, model in models.items():
        d = Path(key)
        tr, ev = caches[args.train_month][key], caches[args.eval_month][key]
        head_reads = head_readout(model, ev["X"], device)
        tasks = (tuple(args.auc_tasks) if args.auc_tasks
                 else (tuple(head_reads) if head_reads else AUC_TASKS))
        res = score(tr, ev, head_readouts=head_reads,
                    auc_bins=args.auc_bins, auc_tasks=tasks,
                    auc_fit_rows=args.auc_fit_rows, auc_tol=args.auc_tol,
                    train_stats=stats[args.train_month],
                    eval_stats=stats[args.eval_month])

        metrics: dict[str, float] = {}
        for name, v in res.items():
            metrics[f"xs_ic/{name}"] = v["ic"]
            metrics[f"xs_ic_se/{name}"] = v["se"]
            metrics[f"xs_ic_cells/{name}"] = v["n_cells"]
            metrics[f"xs_ic_rows/{name}"] = v["n_rows"]
            if "auc" in v:
                metrics[f"xs_auc/{name}"] = v["auc"]
                metrics[f"xs_auc_bins/{name}"] = v["auc_bins"]
                metrics["xs_auc_schema"] = AUC_SCHEMA
                if "head_bins" in v:
                    metrics[f"xs_auc_head_bins/{name}"] = v["head_bins"]
            if "auc_native" in v:
                metrics[f"xs_auc_native/{name}"] = v["auc_native"]
                metrics[f"xs_auc_native_bins/{name}"] = v["auc_native_bins"]

        # The IC is a REPRODUCTION, not a new measurement: same panel, same
        # scorer. Drift here would mean the panel moved since the sweep ran,
        # which would make the AUC and the reported IC incomparable — so it is
        # checked on every checkpoint rather than spot-checked.
        # WHICH TARGET THIS SCORE IS OF. The anchor tables ARE the target --
        # all three became forward-window differences on 2026-08-22
        # (docs/return_bad_calculation.md) -- and xs_ic.json is otherwise
        # identical whichever family produced it, so a consumer reading a tree
        # scored across the change would average two different quantities.
        metrics["xs_anchor_stats"] = stats_name

        prev_path = d / "xs_ic.json"
        if args.smoke_shards:
            prev_path = None
        elif prev_path.is_file():
            try:
                prev = json.loads(prev_path.read_text())
            except json.JSONDecodeError:
                prev = {}
            prev_name = prev.get("xs_anchor_stats", "xs_anchor_stats")
            if prev_name == stats_name:
                # Same target: the IC is a REPRODUCTION and drift would mean
                # the panel moved since the sweep ran.
                deltas = [abs(metrics[k] - prev[k]) for k in metrics
                          if k.startswith("xs_ic/") and k in prev
                          and isinstance(prev[k], (int, float))]
                if deltas:
                    drifts.append(max(deltas))
            elif prev:
                # DIFFERENT target: drift is the point, not a warning, so it is
                # not pooled into the drift check. Keep the old score rather
                # than destroying it -- it is the only copy, re-deriving it
                # costs another full decode, and figures still cite it.
                keep = prev_path.with_suffix(f".{prev_name}.json")
                if not keep.exists():
                    keep.write_text(json.dumps(prev, indent=2))
                    retargeted.append(f"{d.name}: {prev_name} -> {keep.name}")
        if prev_path is not None:
            prev_path.write_text(json.dumps(metrics, indent=2))

        # The per-checkpoint summary line reports ONE task; a multihead has
        # several, so it reports the first and the json carries the rest.
        task = next(iter(head_reads), "return_900")
        probe_auc = metrics.get(f"xs_auc/{task}", float("nan"))
        head_auc = metrics.get(f"xs_auc/head:{task}", float("nan"))
        k_nat = metrics.get(f"xs_auc_native_bins/head:{task}", args.auc_bins)
        nat = ""
        if k_nat != args.auc_bins:
            nat = (f"  | k={k_nat:>2d} probe "
                   f"{metrics.get(f'xs_auc_native/{task}', float('nan')):.4f} "
                   f"head {metrics.get(f'xs_auc_native/head:{task}', float('nan')):.4f}")
        print(f"  {d.name}  {task:22s} "
              f"IC {metrics.get(f'xs_ic/{task}', float('nan')):+.4f}  "
              f"AUC@5 probe {probe_auc:.4f} head {head_auc:.4f}{nat}", flush=True)

    if retargeted:
        print(f"\n==> {len(retargeted)} checkpoint(s) were scored against a "
              f"DIFFERENT target; the previous xs_ic.json was kept alongside:")
        for line in retargeted[:5]:
            print(f"      {line}")
        if len(retargeted) > 5:
            print(f"      ... and {len(retargeted) - 5} more")
    if drifts:
        print(f"\n==> IC reproduction: max |delta| = {max(drifts):.2e} "
              f"over {len(drifts)} checkpoint(s)")
    if args.smoke_shards:
        print(f"==> SMOKE ({args.smoke_shards} shards): nothing written")
    else:
        print(f"==> wrote xs_ic.json for {len(models)} checkpoint(s)")


if __name__ == "__main__":
    main()
