"""Score a series of SSL checkpoints on the synchronized panel, one per month.

The offline half of the arm comparison. Each checkpoint pretrained on month M
gets a ridge probe fit on M and reports its per-cell rank IC on M+1 -- exactly
the split the supervised arm uses, where the head trains on M and the hook
scores M+1. Nothing here reimplements the scoring: it calls xs_ic_eval.score,
the same function scripts/generic/post_train_ic_eval.py calls in-job, so the
ridge-vs-head substitution stays the only difference between the arms.

Embeddings come from ``probe_fit_size.py embed-many``, which already shards a
month across persistent workers. This module only reduces them.

Usage:
    # 1. build the task list and embed (see scripts under scratchpad/)
    # 2. reduce
    uv run scripts/eval/xs_ic_series.py reduce \
        --cache-dir /data/lab/xs_ic_series_cache \
        --manifest /data/lab/tmp/$USER/k2ind_manifest.tsv \
        --expect-shards 32 \
        --json plots/xs_ic_series/k2ind.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from stable_finance import pooled_estimates  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import AUC_TASKS, head_readout, score  # noqa: E402

# Anchor counts, mirroring xs_ic_eval: the fit month is dense, the eval month
# defines the reported cross-sections and must stay at 8.
FIT_ANCHORS = 36
EVAL_ANCHORS = 8


def _load_shards(cache_dir: Path, ckpt_name: str, month: str, anchors: int,
                 expect: int | None):
    """Concatenate one month's shard files into a single panel.

    Empty shards (a shard whose ticker-days were all unusable) are written as a
    one-key ``empty`` npz by the embedder and carry no rows; they still COUNT
    towards completeness, because a missing file and a legitimately empty shard
    are different failures and only the first one silently shrinks the panel.
    """
    d = cache_dir / ckpt_name
    files = sorted(d.glob(f"{month}_a{anchors}_*.npz"))
    if expect is not None and len(files) != expect:
        raise SystemExit(
            f"{ckpt_name} {month} a{anchors}: {len(files)} shards, expected "
            f"{expect}. A short panel scores fine and means nothing — refusing."
        )
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if "empty" in z.files:
            continue
        parts.append({k: z[k] for k in z.files})
    if not parts:
        raise SystemExit(f"{ckpt_name} {month} a{anchors}: every shard empty")
    return {
        "X": np.concatenate([p["X"] for p in parts]),
        "z": np.concatenate([p["z"] for p in parts]),
        "date": np.concatenate([p["date"] for p in parts]),
        "anchor": np.concatenate([p["anchor"] for p in parts]),
        "ticker": np.concatenate([p["ticker"] for p in parts]),
        "target_names": parts[0]["target_names"],
    }


def cmd_reduce(args):
    cache_dir = Path(args.cache_dir)
    rows = [ln.split() for ln in Path(args.manifest).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]

    # One table per month, loaded once and shared across every checkpoint that
    # scores that month. score() needs them to recover the raw target the AUC
    # bins live on; without them it emits IC only.
    stats_dir = Path(args.xs_anchor_stats_dir)
    stats: dict[str, AnchorStats] = {}

    # A supervised checkpoint's own head, scored alongside the ridge probe.
    # The embeddings are already cached, so this costs one MLP forward -- no
    # second pass over the panel. Off unless a checkpoint dir is given, since
    # the manifest's paths point at bll01 and the staged copies live elsewhere.
    #
    # Returns head_readout's MAPPING, ``{task: (scores, proba)}``, and hands it
    # to score() as ``head_readouts``. This used to unpack the return into a
    # (scores, proba, task) triple, which stopped being what head_readout
    # returns when it grew multihead support on 2026-08-29 -- every headed row
    # through this path then died on "not enough values to unpack". A
    # multihead checkpoint contributes one entry per task and each is scored
    # against its own target.
    def _heads(ckpt_name, X) -> dict:
        if not args.ckpt_dir:
            return {}
        d = Path(args.ckpt_dir) / ckpt_name
        if not (d / "head.pt").is_file() and not (d / "heads.pt").is_file():
            return {}
        import torch

        from market_jepa.eval.checkpoints import load_model
        sys.path.insert(0, str(ROOT / "scripts/generic"))
        from post_train_ic_eval import _load_cfg

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(str(d), _load_cfg(d, None), dev)
        model.eval()
        return head_readout(model, X, dev)

    def _stats(ym: str) -> AnchorStats:
        if ym not in stats:
            stats[ym] = AnchorStats(stats_dir / f"{ym}.npz")
        return stats[ym]

    out = []
    for ckpt_path, fit_month, eval_month in rows:
        name = Path(ckpt_path).name
        tr = _load_shards(cache_dir, name, fit_month, FIT_ANCHORS,
                          args.expect_shards)
        ev = _load_shards(cache_dir, name, eval_month, EVAL_ANCHORS,
                          args.expect_shards)
        heads = _heads(name, ev["X"])
        # THE LOGISTIC IS THE EXPENSIVE FIT ON THIS PATH -- 150k rows x 384
        # collinear features, minutes each, and --auc-extra-bins asks for two
        # more per task. A manifest with dozens of rows per job spends more
        # wall-clock on a retired metric than on the one being reported, so
        # --no-auc turns it off. Passing no auc_tasks leaves the ridge (which
        # covers every column) untouched.
        res = score(tr, ev, head_readouts=heads,
                    train_stats=None if args.no_auc else _stats(fit_month),
                    eval_stats=None if args.no_auc else _stats(eval_month),
                    auc_tasks=(tuple(heads) if heads else AUC_TASKS),
                    auc_extra_bins=() if args.no_auc
                    else tuple(args.auc_extra_bins))
        rec = {"ckpt": name, "fit_month": fit_month, "eval_month": eval_month,
               "fit_rows": int(len(tr["X"])), "eval_rows": int(len(ev["X"]))}
        for k, v in res.items():
            rec[k] = v["ic"]
            rec[f"{k}_se"] = v["se"]
            rec[f"{k}_cells"] = v["n_cells"]
            # The logistic-probe AUC rides along with the IC (xs_ic_eval.score),
            # and a random-init run through this path is exactly the ΔAUC
            # subtrahend, so it has to survive into the record.
            if "auc" in v:
                rec[f"{k}_auc"] = v["auc"]
                rec[f"{k}_auc_bins"] = v["auc_bins"]
                rec["auc_schema"] = 2
            if "auc_native" in v:
                rec[f"{k}_auc_native"] = v["auc_native"]
                rec[f"{k}_auc_native_bins"] = v["auc_native_bins"]
            for kk, vv in v.items():
                if kk.startswith("auc_k"):
                    rec[f"{k}_{kk}"] = vv
        out.append(rec)
        tasks = " ".join(
            f"{k.replace('_900', '')} {v['ic']:+.4f}+-{v['se']:.4f}"
            for k, v in sorted(res.items()) if k.endswith("_900"))
        print(f"{name} {fit_month}->{eval_month} "
              f"fit={len(tr['X']):,d} ev={len(ev['X']):,d}  {tasks}", flush=True)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {p}")

    # HEADLINE = the month-clustered pooled IC. A single month is not a
    # reportable number: cells stay independent only while anchors are spaced
    # a full horizon apart, which caps a month near 240 cells and SE ~ 0.005,
    # so a month clears 2 SE only above IC ~ 0.010. Pooling with the month as
    # the clustering unit is what makes the weaker targets reportable at all.
    # `infl` is how much a naive cell-pool would have understated the SE.
    print(f"\n{'task':26s} {'IC':>8s} {'SE':>8s} {'t':>7s} {'n':>3s} "
          f"{'infl':>5s}   per-month range")
    for task in sorted({k for r in out for k in r
                        if k.endswith("_900") and not k.endswith("_se")}):
        # NOT every record carries every key: a head column exists only for
        # checkpoints trained on that target, so a manifest mixing tasks (or
        # mixing headed and headless arms) has a ragged key set. Summing over
        # the records that HAVE the key is the only correct reading, and the
        # alternative was a KeyError that killed the job AFTER the results
        # json was written but BEFORE it was copied off the compute node.
        have = [r for r in out if task in r and f"{task}_se" in r]
        if not have:
            continue
        ics = [r[task] for r in have]
        p = pooled_estimates(ics)
        within = np.asarray([r[f"{task}_se"] for r in have], dtype=float)
        naive = float(np.sqrt(np.nansum(within ** 2)) / len(within))
        inflation = p.standard_error / naive if naive > 0 else float("nan")
        t_stat = p.mean / p.standard_error if p.standard_error else float("nan")
        print(f"  {task:24s} {p.mean:+8.4f} {p.standard_error:8.4f} {t_stat:7.1f} "
              f"{p.observations:3d} {inflation:5.2f}x  "
              f"[{min(ics):+.4f},{max(ics):+.4f}]  ({len(have)}/{len(out)} recs)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce")
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--manifest", required=True,
                   help="TSV: <ckpt_dir> <fit_month> <eval_month>")
    r.add_argument("--expect-shards", type=int, default=None)
    r.add_argument("--ckpt-dir", default=None,
                   help="directory holding the checkpoint dirs by name. Given, "
                        "a supervised checkpoint's own head is scored next to "
                        "the probe from the already-cached embeddings.")
    r.add_argument("--no-auc", action="store_true",
                   help="report rank IC only. The macro one-vs-rest AUC is "
                        "the retired metric and its logistic fit dominates "
                        "this path's wall-clock; the ridge probe covers every "
                        "column either way.")
    r.add_argument("--auc-extra-bins", type=int, nargs="*",
                   default=[11, 21],
                   help="additional bin counts to fit the logistic probe at. "
                        "A headless arm is the ΔAUC subtrahend for heads "
                        "trained at these k, and a subtrahend on the wrong "
                        "partition is not a baseline.")
    r.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60",
                   help="the (mu, sigma) tables the panel was standardized "
                        "against — the AUC needs them to undo the z-score")
    r.add_argument("--json", default=None)
    r.set_defaults(func=cmd_reduce)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
