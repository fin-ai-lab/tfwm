"""Pin the finetune's base arm: one TSV row per eval month.

The compute nodes cannot read ``/data/lab``, so everything the sweep needs
about a month has to travel with the repo. That is this file: base checkpoint,
ridge head-init directory, and the size of the pool the ridge was fit on.

WHY THE POOL SIZE IS HERE. The sweep's axis is a NUMBER OF LABELLED ROWS, so
it can sit on the same x as plots/core/probe_fit_breadth.py, but the
trainer's knob is ``dataset.train_data_fraction`` -- a fraction of ticker-days.
The two convert exactly: the a36 panel has one row per (ticker, day, anchor) at
36 anchors/day (measured 35.83 after ragged days), so ``n / n_rows_pool`` is
the same fraction either way and the 36 cancels. Carrying the denominator means
a row count can always be read back as a percentage, and vice versa, without
re-deriving it from a tree the node cannot see.

THE ARM IS THIS FILE'S, NOT THE SWEEP'S. Regenerating against a different
series silently repoints every sweep that reads it, so the BASE_SERIES header
is what names the runs -- see ssl_base_lib.sh:ssl_base_arm_tag.

    uv run scripts/sweeps/ssl_finetune/make_ssl_base_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_6MO = Path(
    "/data/lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_6mo.json")
RIDGE_ROOT = Path("/data/lab/market-jepa-checkpoints/_scratch/ridge_head_init")
OUT = Path(__file__).resolve().parent / "ssl_base_manifest.tsv"

# The weights a head init actually loads, in the order the digest covers them.
HEAD_FILES = ("return_900.npz", "spread_change_900.npz",
              "volatility_change_900.npz")


def head_digest(d: Path) -> str:
    """A content digest of one month's head init.

    WHY THE MANIFEST CARRIES A DIGEST AT ALL. ssl_base_lib.sh validates a
    reused /hpc_temp staging directory against a `.source` stamp -- the path
    it was copied from. That catches a DIFFERENT checkpoint, which is the
    failure that cost a wave on 2026-09-16, but it cannot catch the SAME path
    whose contents have changed. On 2026-09-17 the head inits were refit in
    place when the readout moved from mean to last, so every node that had run
    the earlier wave went on serving the retired mean-pooled weights under a
    stamp that still matched: 18 stale directories across 5 nodes, and three
    jobs that died at the readout guard before it was noticed.

    The digest closes that: a reuse now has to match the bytes. It is computed
    over the .npz files only -- summary.json holds absolute source paths and
    is not what the model loads.

    Kept in sync with the `sha256sum` in ssl_base_lib.sh, which concatenates
    the same files in the same (lexicographic) order.
    """
    h = hashlib.sha256()
    for name in HEAD_FILES:
        h.update((d / name).read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="pair_warp_6mo",
                    help="the reported LeJEPA Time Warping row")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ckpts = [r for r in json.loads(MANIFEST_6MO.read_text())["ckpts"]
             if r["series_key"] == args.series]
    ridge_index = RIDGE_ROOT / args.series / "index.json"
    if not ridge_index.exists():
        raise SystemExit(
            f"no ridge head inits at {ridge_index} — run "
            f"scripts/eval/fit_ridge_head_init.py --series {args.series} first")
    ridge = {r["eval_month"]: r for r in json.loads(ridge_index.read_text())}

    lines = [
        f"# BASE_SERIES\t{args.series}",
        "# eval_month\tckpt_dir\thead_init_dir\tn_rows_pool"
        "\thead_init_digest",
    ]
    skipped = []
    for row in sorted(ckpts, key=lambda r: r["eval_month"]):
        ym = row["eval_month"]
        r = ridge.get(ym)
        if r is None:
            # No head init means no finetune: the arm's whole premise is that
            # the head starts at the probe. Dropping the month is correct and
            # must be loud -- a silently short manifest changes the month set
            # the curve averages over.
            skipped.append(ym)
            continue
        head_dir = RIDGE_ROOT / args.series / ym
        lines.append(
            f"{ym}\t{row['ckpt_dir']}\t{head_dir}"
            f"\t{r['n_rows_pool']}\t{head_digest(head_dir)}")
    args.out.write_text("\n".join(lines) + "\n")
    n = len(lines) - 2
    print(f"{args.out}: {n} months for {args.series}")
    if skipped:
        print(f"SKIPPED (no ridge head init): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
