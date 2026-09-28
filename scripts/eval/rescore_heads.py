"""Re-score ONLY the trained head(s) of already-scored checkpoints.

WHY THIS EXISTS. The 2026-08-29 multihead loader fix changed exactly one
number beside a checkpoint: ``xs_ic/head:<task>``. Every multihead checkpoint
had been rebuilt as a single-task SupervisedModel with a RANDOMLY INITIALIZED
head (it saves ``heads.pt``, the loader looked for ``head.pt``), so its head
keys are noise on one task. The ridge probe, its AUC and the target columns
were never touched -- the backbone loaded correctly all along.

So this does not re-run the pass. ``xs_score_many.py`` would, and measured 78
min/month here: it embeds the 487k-row PROBE-FIT panel that a head never sees
and refits the logistic AUC probes, which xs_ic_eval documents at ~20 min each
and which already sit on disk, correct. This embeds the eval panel only, runs
each head over it, and MERGES the head keys into the existing xs_ic.json.

Everything else in that file is left byte-for-byte alone. The metric is
``market_jepa.eval.metrics.grouped_rank_ic``, the same function xs_ic_eval's
``score`` computes every IC with, applied to the same (date, anchor) cells and
the same finite-label mask -- the head branch there depends on the probe-fit
month for nothing but a row-count gate.

A scalar head has no softmax, so no head AUC is written; nor was there one to
write before. Binned heads keep theirs from whatever wrote it.

Usage:
    uv run scripts/local/mass_eval/rescore_heads.py --eval-month 2008-03 \\
        --ckpt-glob '/data/lab/market-jepa-checkpoints/supervised-multihead-*-2008-02-01-*/*/'
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
from market_jepa.eval.metrics import grouped_rank_ic  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset.targets import AnchorTargetStats as AnchorStats  # noqa: E402
from post_train_ic_eval import _load_cfg  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    day_anchors, embed_month_many, head_readout, panel_kwargs_for,
)


# Mirrors merge_score_results.HEAD_SCHEMA -- see the note there.
HEAD_SCHEMA = 2


def parse_args():
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--eval-month", required=True, help="YYYY-MM, reports the metric")
    p.add_argument("--ckpt-glob", default=None, help="shell glob of ckpt dirs (quote it)")
    p.add_argument("--ckpt-dirs", nargs="*", default=[])
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--anchors-per-day", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dry-run", action="store_true",
                   help="print the head ICs, write nothing")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)

    dirs = [Path(d) for d in args.ckpt_dirs]
    if args.ckpt_glob:
        dirs += [Path(d) for d in sorted(glob.glob(args.ckpt_glob))]
    dirs = [d for d in dirs if (d / "xs_ic.json").is_file()]
    if not dirs:
        print("nothing to re-score")
        return

    models, panel_by_dir = {}, {}
    for d in dirs:
        cfg = _load_cfg(d, None)
        m = load_model(str(d), cfg, device)
        m.eval()
        models[str(d)] = m
        panel_by_dir[str(d)] = panel_kwargs_for(cfg)

    # Same guard as xs_score_many: one decode feeds every model, so models that
    # want different views cannot share it.
    def _kind(v):
        return tuple(sorted((k, tuple(x) if isinstance(x, list) else x)
                            for k, x in v.items()))
    kinds = {_kind(v) for v in panel_by_dir.values()}
    if len(kinds) > 1:
        raise SystemExit(
            "one batch mixes panels:\n  "
            + "\n  ".join(repr(dict(k)) for k in sorted(kinds))
            + "\nRe-score each arm separately -- they need different views.")
    panel_kw = next(iter(panel_by_dir.values()), {})

    ym = args.eval_month
    y, m = ym.split("-")
    print(f"==> {len(models)} checkpoint(s), head readout on {ym}", flush=True)
    caches = embed_month_many(
        models, Path(args.mosaic_dir) / y / m, ym,
        AnchorStats(Path(args.xs_anchor_stats_dir) / f"{ym}.npz"),
        schedule, day_anchors(args.anchors_per_day), device, args.batch_size,
        **panel_kw,
    )

    n_written = 0
    for key, model in models.items():
        ev = caches[key]
        names = ev["target_names"].tolist()
        # One cell id per (date, anchor) -- the unit a cross-sectional rank
        # correlation is defined over. Built exactly as xs_ic_eval.score does.
        cells = np.char.add(np.char.add(ev["date"], "@"),
                            ev["anchor"].astype(str))
        reads = head_readout(model, ev["X"], device)
        d = Path(key)
        prev = json.loads((d / "xs_ic.json").read_text())
        # DROP EVERY OLD HEAD KEY FIRST. A multihead's stale file names one
        # task; the fix produces three. Leaving the old set in place would mix
        # a re-scored head with an untrained one under two keys of one file.
        stale = [k for k in prev if ":" in k and k.split(":", 1)[0].endswith("head")]
        for k in stale:
            del prev[k]
        got = {}
        for task, (scores, _proba) in reads.items():
            if task not in names:
                print(f"  {d.name}: {task} not in the panel's targets — skipped")
                continue
            j = names.index(task)
            yev = ev["z"][:, j]
            ok = np.isfinite(yev)
            if ok.sum() < 100:
                continue
            ic, se, n_cells = grouped_rank_ic(scores[ok], yev[ok], cells[ok])
            prev[f"xs_ic/head:{task}"] = float(ic)
            prev[f"xs_ic_se/head:{task}"] = float(se)
            prev[f"xs_ic_cells/head:{task}"] = int(n_cells)
            prev[f"xs_ic_rows/head:{task}"] = int(ok.sum())
            got[task] = float(ic)
        if got:
            # Same stamp merge_score_results writes, for the same reason: a
            # reader cannot tell a re-scored head from the untrained one by
            # key presence alone. Both paths must set it or the plots would
            # silently drop whichever path skipped it.
            prev["xs_head_schema"] = HEAD_SCHEMA
        probe = {t: prev.get(f"xs_ic/{t}") for t in got}
        print(f"  {d.name}  " + "  ".join(
            f"{t.replace('_900','')}: head {v:+.4f} / probe {probe[t]:+.4f}"
            for t, v in sorted(got.items())), flush=True)
        if not args.dry_run and got:
            (d / "xs_ic.json").write_text(json.dumps(prev, indent=2))
            n_written += 1
        if stale and not got:
            print(f"  {d.name}: WARNING dropped {len(stale)} stale head key(s) "
                  f"and produced none")
    print(f"==> wrote {n_written} file(s)"
          + ("  [dry run: nothing written]" if args.dry_run else ""), flush=True)


if __name__ == "__main__":
    main()
