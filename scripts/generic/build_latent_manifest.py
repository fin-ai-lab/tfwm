# /// script
# requires-python = ">=3.10"
# ///
"""Write a job-local noclamp-style manifest for just-trained checkpoints.

The latent-eval stages resolve trained encoders through the manifest by
(series_key, eval_month) — see fixed_panel_metrics._manifest_row, which
reads MJ_NOCLAMP_MANIFEST when set. A final32 bundle job trains one
checkpoint per method for one month and then runs those stages in-job, so
it needs a manifest whose ckpt_dir paths are the node-local ones it just
wrote. This emits exactly that.

series_key is the run_name with '-' -> '_' ("byol-final" -> "byol_final"),
matching the ssl_ic entries in industry_nn_sweep.MODEL_SPECS.

Usage (from sweep_post_bundle):
    uv run scripts/generic/build_latent_manifest.py \
        --eval-month 2016-04 --out /path/manifest.json <ckpt_dir> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-month", required=True, help="YYYY-MM")
    p.add_argument("--out", required=True)
    p.add_argument("ckpt_dirs", nargs="+")
    args = p.parse_args()

    rows = []
    for d in args.ckpt_dirs:
        d = Path(d)
        meta_f = d / "train_meta.json"
        if not meta_f.is_file():
            print(f"SKIP {d}: no train_meta.json", file=sys.stderr)
            continue
        meta = json.loads(meta_f.read_text())
        run_name = meta.get("run_name") or d.name
        rows.append({
            "series_key": run_name.replace("-", "_"),
            "project": d.parent.name,
            "run_id": d.name,
            "run_name": run_name,
            "train_end": meta.get("train_date_end", ""),
            "eval_month": args.eval_month,
            "ckpt_dir": str(d),
        })

    if not rows:
        print("no manifest rows built", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"fair_months": [args.eval_month],
         "series_keys": sorted({r["series_key"] for r in rows}),
         "ckpts": rows}, indent=1))
    print(f"wrote {out} ({len(rows)} rows: "
          f"{', '.join(sorted(r['series_key'] for r in rows))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
