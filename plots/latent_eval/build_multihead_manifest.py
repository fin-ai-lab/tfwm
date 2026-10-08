"""Manifest rows for the full-history supervised MULTIHEAD trunk, our panel only.

``sup_multi_w8`` has been a MODEL_ORDER key with no rows behind it: the wave is
one project PER BUNDLE of training months --
``supervised-full-month-multihead-<commit6>-<YYYY-MM>-<YYYY-MM>`` -- with the
training month in the run_name, which fits no resolver in the registry.

THE SELECTION IS THE FIGURE'S, NOT A SECOND ONE. plots/full_data_multihead is
what reports this campaign, and a latent row must describe the checkpoints that
figure draws or the two tables are about different models under one name. So
the filters are IMPORTED from it rather than restated here:

  * ``_BUNDLE_PROJECT_RE`` -- the commit-pinned bundle-project pattern, which
    structurally excludes the retired ``-ce`` binned generation;
  * ``keep_current_recipe`` -- the run's own training window must be the span
    schemas.DatasetConfig currently defines;
  * ``dedupe_by_train_month`` -- one run per training month, newest wins, since
    bundle projects overlap whenever a wave is resubmitted;
  * ``state == "finished"`` and the ``XS_STATS_DIR_REQUIRED`` target stamp --
    the two filters ``records_to_dataframe`` applies at plot time.

Runs come from the figure's OWN api_cache.json, so this reads the same snapshot
the checked-in figure was drawn from and needs no W&B call. A month whose run
is still training is therefore absent here exactly as it is absent there; to
pick up newly finished months, re-run the figure (which refreshes the cache)
and then re-run this.

The checkpoint tree is the second gate: a selected run must have a directory
with ``backbone.pt`` and ``heads.pt`` (a multihead saves the plural -- see
eval/checkpoints), and its train_meta window must be the one W&B reported.

Output is MACHINE-LOCAL (absolute ckpt_dir paths), written beside the caches
rather than committed, and joined onto the repo manifest by run_eval.sh.

Run::

    uv run python plots/latent_eval/build_multihead_manifest.py \\
        lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_multi.json
"""
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, "plots")
sys.path.insert(0, "plots/metrics")
sys.path.insert(0, "plots/full_data_multihead")
import full_data_multihead as F  # noqa: E402
from style import load_sweep_months  # noqa: E402
from metrics import SUP_SPAN_NO_SPAN  # noqa: E402
from stable_finance.dataset import next_month  # noqa: E402

ROOT = Path("lab/market-jepa-checkpoints")
SERIES_KEY = "sup_multi_w8"

# The same 31 eval months as the rest of the latent panel.
want_eval = sorted(next_month(m) for m in
                   set(load_sweep_months()) - SUP_SPAN_NO_SPAN)

cache = json.loads((F.HERE / "api_cache.json").read_text())
recs = [r for r in cache.values()
        if F._BUNDLE_PROJECT_RE.match(str(r.get("project", "")))]
print(f"{len(recs)} run(s) in {F.FULL_DATA_PROJECT}-* bundle projects")

recs, off = F.keep_current_recipe(recs)
print(f"  dropped {off} not on the {F.recipe_span_months()}-month span")
recs, dup = F.dedupe_by_train_month(recs)
print(f"  dropped {dup} duplicate month-run(s) (kept newest per train month)")
n = len(recs)
recs = [r for r in recs if r.get("state") == "finished"]
print(f"  dropped {n - len(recs)} not finished")
n = len(recs)
recs = [r for r in recs
        if PurePosixPath(str(r.get("xs_stats_dir"))).name
        == F.XS_STATS_DIR_REQUIRED]
print(f"  dropped {n - len(recs)} off-target (xs_stats_dir)")
print(f"  {len(recs)} run(s) the figure displays")

rows, unusable = [], []
for r in sorted(recs, key=lambda r: r["run_name"]):
    tm = F._train_month(r["run_name"])
    ev = next_month(tm)
    if ev not in want_eval:
        continue
    d = ROOT / r["project"] / r["run_id"]
    # A MULTIHEAD SAVES heads.pt, NOT head.pt. The loader keys Case 3 on
    # backbone.pt and then looks for the plural; a dir with only backbone.pt
    # would load a trunk with no heads and report nothing about it.
    if not ((d / "backbone.pt").is_file() and (d / "heads.pt").is_file()):
        unusable.append(f"{ev}: {d} has no backbone.pt+heads.pt")
        continue
    # The training window lives in config.dataset, not at the top level of
    # train_meta. Checked against W&B because the two are independent records
    # of the same run and a disagreement means the run_id is not the run the
    # figure thinks it is.
    ds = json.loads((d / "train_meta.json").read_text())["config"]["dataset"]
    for field in ("train_date_start", "train_date_end"):
        if str(ds.get(field, ""))[:10] != str(r.get(field, ""))[:10]:
            unusable.append(f"{ev}: {d} {field}={ds.get(field)} but W&B "
                            f"says {r.get(field)}")
            break
    else:
        rows.append({
            "series_key": SERIES_KEY, "project": r["project"],
            "run_id": r["run_id"], "train_month": tm, "eval_month": ev,
            "train_start": str(r["train_date_start"])[:10],
            "train_end": str(r["train_date_end"])[:10],
            "ckpt_dir": str(d),
        })

out = Path(sys.argv[1] if len(sys.argv) > 1
           else "lab/market-jepa-checkpoints/_scratch/latent_eval/"
                "manifest_multi.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"ckpts": rows}, indent=1))
have = sorted(r["eval_month"] for r in rows)
print(f"\nwrote {out}  ({len(rows)} rows)")
print(f"  {SERIES_KEY}: {len(have)}/{len(want_eval)} panel eval months")
missing = [m for m in want_eval if m not in have]
if missing:
    print(f"  MISSING {missing}")
    print("  (a month the figure does not display has no row here either --"
          " re-run the figure to refresh its cache, then re-run this)")
for u in unusable:
    print(f"  UNUSABLE {u}")
print(" ".join(have))
