"""Fold cluster scoring results back into the checkpoint tree.

``slurm_score_ckpts.sh`` writes one JSON per job to lab/score_results;
the plots read ``xs_ic.json`` beside each checkpoint. This copies the AUC keys
across and leaves the IC keys already on disk untouched.

IT ALSO CHECKS THEM. The cluster recomputes the IC on the way to the AUC, so
every record carries a second measurement of a number the checkpoint already
has. Agreement means the AUC was computed on the same panel the reported IC
came from. The tolerance is the checkpoint's OWN standard error, not an
absolute constant: a record whose IC moved by more than --max-drift-se
standard errors is REFUSED rather than merged.

WHY IT DOES NOT REPRODUCE TO THE BIT, AND WHY THAT IS FINE. The two paths
build the same panel — the cell counts match exactly — and the HEAD readout
reproduces to ~1e-5, which says the eval embeddings are the same. What moves
is the ridge: the cluster embeds a month in 32 shards and the in-job hook in
one pass, so the encoder's bf16 forward rounds differently, and a Gram matrix
with rcond ~6e-8 amplifies that. Measured across 75 checkpoints the probe IC
moved by a median of 0.05 SE and at most 0.76 SE. The stored IC is never
overwritten; only AUC keys are added.

Usage:
    uv run scripts/local/mass_eval/merge_score_results.py \
        --results 'lab/score_results/binpen-auc-part*.json' \
        --ckpt-root lab/market-jepa-checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

AUC_SCHEMA = 2
# Bumped by the 2026-08-29 multihead loader fix. A file WITHOUT this stamp may
# still carry xs_ic/head:* keys -- every checkpoint scored before that date
# does -- and those are an untrained head's readout. Readers that plot a head
# must require the stamp; key presence is not enough.
HEAD_SCHEMA = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--results", required=True, help="glob of result JSONs")
    ap.add_argument("--ckpt-root", default="lab/market-jepa-checkpoints")
    ap.add_argument("--max-drift-se", type=float, default=1.5,
                    help="largest |IC_new - IC_stored|, in units of the "
                         "checkpoint's own SE, that still merges")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    index = {d.name: d for d in root.glob("*/*/") if (d / "xs_ic.json").is_file()}

    merged = skipped = refused = 0
    worst = 0.0
    for f in sorted(glob.glob(args.results)):
        for rec in json.loads(Path(f).read_text()):
            d = index.get(str(rec.get("ckpt", "")))
            if d is None:
                skipped += 1
                continue
            cur = json.loads((d / "xs_ic.json").read_text())

            # In SE units, so a noisy month is not held to a quiet month's
            # tolerance. An SE of 0 (or missing) falls back to 1.0, which makes
            # the check absolute rather than silently vacuous.
            # HEAD KEYS ARE EXCLUDED FROM THE DRIFT CHECK. The check asks
            # whether the cluster rebuilt the same panel, and it answers that
            # from the probe ICs, which both paths compute the same way. A
            # head key is the one number this merge exists to CHANGE: every
            # pre-2026-08-29 multihead checkpoint stores an untrained head's
            # readout, so its "drift" is the size of the bug, not evidence of
            # a bad panel. Leaving them in refused every record on the return
            # task and merged nothing.
            drift = max(
                (abs(v - cur[f"xs_ic/{k}"]) / (cur.get(f"xs_ic_se/{k}") or 1.0)
                 for k, v in rec.items()
                 if f"xs_ic/{k}" in cur and isinstance(v, (int, float))
                 and not k.startswith("head:")),
                default=0.0,
            )
            worst = max(worst, drift)
            if drift > args.max_drift_se:
                print(f"REFUSED {d.name}: IC moved {drift:.2f} SE — that is "
                      f"more than resharding the encoder pass explains")
                refused += 1
                continue

            add = {}
            for k, v in rec.items():
                if k.startswith("head:"):
                    continue
                for suffix, out in (("_auc_native_bins", "xs_auc_native_bins"),
                                    ("_auc_native", "xs_auc_native"),
                                    ("_auc_bins", "xs_auc_bins"),
                                    ("_auc", "xs_auc")):
                    if k.endswith(suffix):
                        add[f"{out}/{k[: -len(suffix)]}"] = v
                        break
            if add:
                add["xs_auc_schema"] = AUC_SCHEMA

            # A --head run carries the checkpoint's OWN readout, and for a
            # multihead trunk that is one entry per task. It reaches the plots
            # the same way the AUC does, so fold it in here rather than give
            # the cluster path a second merge script.
            head = {}
            for k, v in rec.items():
                if not k.startswith("head:"):
                    continue
                for suffix, out in (("_cells", "xs_ic_cells"),
                                    ("_se", "xs_ic_se"),
                                    ("", "xs_ic")):
                    if suffix and not k.endswith(suffix):
                        continue
                    head[f"{out}/{k[: len(k) - len(suffix)]}"] = v
                    break
            if head:
                # DROP THE STALE SET FIRST. A pre-fix multihead file names one
                # task; the fix produces three. Updating in place would leave a
                # re-scored head and an untrained one side by side under two
                # keys of the same file, indistinguishable to every reader.
                # Same reasoning as rescore_heads.py, which does this locally.
                for k in [k for k in cur
                          if ":" in k and k.split(":", 1)[0].endswith("head")]:
                    del cur[k]
                head["xs_head_schema"] = HEAD_SCHEMA

            if not add and not head:
                skipped += 1
                continue
            cur.update(add)
            cur.update(head)
            if not args.dry_run:
                (d / "xs_ic.json").write_text(json.dumps(cur, indent=2))
            merged += 1

    print(f"merged {merged}  skipped {skipped}  refused {refused}  "
          f"max IC drift {worst:.2f} SE"
          + ("  (dry run — nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
