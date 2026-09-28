"""Manifest rows for the six-month-span supervised specialists.

The latent suite resolves MODEL_ORDER keys through MJ_NOCLAMP_MANIFEST, keyed
by (series_key, EVAL month). The 581eb2 wave does not fit the resolvers that
already exist: sup_run_dir wants a FLAT supervised-full-month-<task>/ project
with month-prefixed run names (the archived layout), while this wave is one
project PER SPAN -- supervised-full-month-<task>-581eb2-<span> -- with the
training month in the run_name. So the mapping is built here, once, from
train_meta.json rather than from any path convention.

Same generation pin as metrics.SERIES_DEFS: SUP_SPAN_COMMIT and the
`<YYYY-MM>_pairwise_blr0.0002` run name, so the latent table and the IC figure
cannot end up describing different checkpoints under one name. To read a newer
wave, move SUP_SPAN_COMMIT in plots/metrics/metrics.py and re-run this; nothing
here names a hash of its own.

The output is MACHINE-LOCAL (it holds absolute ckpt_dir paths), which is why it
is written under the checkpoint root rather than committed -- run_eval.sh globs
`${TMPDIR}/manifest_*.json` and joins it onto the repo manifest.

Run::

    uv run python plots/latent_eval/build_supspan_manifest.py \\
        /data/lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_supspan.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "plots")
from style import load_sweep_months  # noqa: E402
sys.path.insert(0, "plots/metrics")
from metrics import SUP_SPAN_COMMIT, SUP_SPAN_NO_SPAN  # noqa: E402
from stable_finance.dataset import next_month  # noqa: E402

ROOT = Path("/data/lab/market-jepa-checkpoints")
RUN_RE = re.compile(r"^(\d{4}-\d{2})_pairwise_blr0\.0002$")
# latent series key -> project stem. The keys are the ones MODEL_SPECS already
# carries (manifest_series == the key), so no registry edit is needed.
SERIES = {
    "sup_return_w8": "supervised-full-month-return",
    "sup_vol_w8": "supervised-full-month-vol-change",
    "sup_spread_w8": "supervised-full-month-spread-change",
}

train_months = set(load_sweep_months()) - SUP_SPAN_NO_SPAN
want_eval = {next_month(m) for m in train_months}

rows, report = [], {}
for skey, stem in SERIES.items():
    found = {}
    for proj in sorted(ROOT.glob(f"{stem}-{SUP_SPAN_COMMIT}-*")):
        for run in sorted(p for p in proj.iterdir() if p.is_dir()):
            meta_f = run / "train_meta.json"
            if not meta_f.is_file():
                continue
            meta = json.loads(meta_f.read_text())
            m = RUN_RE.match(meta.get("run_name", ""))
            if not m:
                continue
            tm = m.group(1)
            if tm not in train_months:
                continue
            ev = next_month(tm)
            if ev in found:
                raise SystemExit(f"{skey} {ev}: two runs match "
                                 f"({found[ev]['ckpt_dir']} and {run})")
            found[ev] = {
                "series_key": skey, "project": proj.name,
                "run_id": run.name, "train_month": tm, "eval_month": ev,
                "ckpt_dir": str(run),
            }
    rows.extend(found.values())
    report[skey] = sorted(found)

out = Path(sys.argv[1])
out.write_text(json.dumps({"ckpts": rows}, indent=1))
print(f"wrote {out}  ({len(rows)} rows)")
for skey, evs in report.items():
    missing = sorted(want_eval - set(evs))
    print(f"  {skey}: {len(evs)} eval months" +
          (f"  MISSING {missing}" if missing else "  (complete)"))
shared = set.intersection(*(set(v) for v in report.values()))
print(f"shared eval months across the three tasks: {len(shared)}")
print(" ".join(sorted(shared)))
