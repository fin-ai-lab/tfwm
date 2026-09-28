"""Manifest rows for the six-month-span SSL and LeJEPA wave (2026-09-15).

The latent suite resolves MODEL_ORDER keys through MJ_NOCLAMP_MANIFEST, keyed
by (series_key, EVAL month). This wave has no row in the repo manifest and does
not fit any glob resolver in the registry: it is one project PER (arm, span) --
``{ssl,lejepa}-6mo-<arm>-<hash>-<start>-<end>`` -- with a single wandb run dir
inside, so the mapping is built here, once, from the PATH.

THE HASH IN THE PATH IS A WANDB PROJECT SUFFIX, NOT THE CODE THAT TRAINED THE
RUN. The repaired arms were deliberately pinned to the original namespace, so
ts2vec/timemae/cost/cpc/ijepa/mae checkpoints under ``-206b47-`` were trained on
later commits. Provenance belongs to train_meta.json; this file only resolves
which directory a (series, month) means.

The output is MACHINE-LOCAL (absolute ckpt_dir paths), which is why it is
written under the checkpoint root rather than committed -- run_eval.sh globs
``${TMPDIR}/manifest_*.json`` and joins it onto the repo manifest.

THE WAVE IS STILL FILLING. A missing (arm, month) is reported, not fatal: the
point of running before the last arm lands is to get the complete arms scored,
and the pooled table already carries each model's own n_months.

Run::

    uv run python plots/latent_eval/build_6mo_manifest.py \\
        /data/lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_6mo.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "plots")
from style import load_sweep_months  # noqa: E402
sys.path.insert(0, "plots/metrics")
from metrics import SUP_SPAN_NO_SPAN  # noqa: E402
from stable_finance.dataset import next_month  # noqa: E402

ROOT = Path("/data/lab/market-jepa-checkpoints")
PROJ_RE = re.compile(
    r"^(ssl|lejepa)-6mo-([a-z0-9]+)-([0-9a-f]{6})"
    r"-(\d{4}-\d{2})-\d{2}-(\d{4}-\d{2})-\d{2}$")

# ARMS RESUBMITTED UNDER A NEW PROJECT HASH, and which hash wins.
#
# When an arm is retrained wholesale the new wave lands in a DIFFERENT wandb
# namespace, so both waves sit in the checkpoint root claiming the same
# (arm, eval month) and the duplicate guard below -- rightly -- refuses to
# guess. Resolve it HERE, explicitly, rather than by "newest mtime wins":
# which of two waves is the real one is a judgement about the recipe, not a
# fact about a timestamp, and it belongs in a line someone can review.
#
#   ts2vec -> db85f5: commit 074a0ad, "ts2vec gets its own LR, and the whole
#   arm gets resubmitted at it". The 206b47 wave trained at the shared default
#   (blr override None) and covers 25 of the 31 panel months; db85f5 carries
#   the arm's own blr=1e-4 and covers all 31. The old wave is superseded in
#   full, not patched over, so nothing mixes two learning rates in one row.
#   timemae -> b808ed: same story one arm over. The 206b47 wave ran at an
#   effective optimizer.blr of 1e-3 and moved the backbone 1.71 on average
#   (max 2.43) -- above even the broken ts2vec wave's 1.04, and the only arm
#   to post a NEGATIVE probe IC. Rerun at 1e-4, drift lands at 0.23-0.27,
#   in line with ts2vec's corrected 0.25 and byol's 0.27.
#
# READ optimizer.blr, NOT mode.training_overrides.blr. Both exist in every
# config; the mode block is a default that the optimizer block overrides, and
# it still reads 1e-3 on a run that trained at 1e-4. backbone_drift is the
# measurement that actually distinguishes them.
PREFER_HASH = {"ts2vec": "db85f5", "timemae": "b808ed"}

# arm -> latent series key. The _6mo suffix keeps these apart from the archived
# one-month *_final and pair_*_final rows, which are a DIFFERENT SPAN and must
# never pool into the same cell (see plots/latent_eval/_archive).
ARM_KEY = {
    ("lejepa", a): f"pair_{a}_6mo" for a in
    ("rrc", "warp", "noise", "k2", "k2ind")
}
ARM_KEY.update({
    ("ssl", a): f"{a}_6mo" for a in
    ("byol", "cost", "cpc", "dino", "ijepa", "mae", "tfc", "timemae", "ts2vec")
})

# The panel: the sweep's train months (+1), minus the one with no six-month
# span. Identical to the supervised/TSFM run's, so the arms share a panel.
want_eval = sorted(next_month(m) for m in
                   set(load_sweep_months()) - SUP_SPAN_NO_SPAN)

rows, found = [], {k: {} for k in ARM_KEY.values()}
skipped = []
superseded = []
for proj in sorted(ROOT.glob("*-6mo-*")):
    m = PROJ_RE.match(proj.name)
    if not m:
        continue
    fam, arm, phash, start, end = m.groups()
    skey = ARM_KEY.get((fam, arm))
    if skey is None:
        skipped.append(f"{proj.name}: unknown arm {fam}/{arm}")
        continue
    if arm in PREFER_HASH and phash != PREFER_HASH[arm]:
        superseded.append(f"{proj.name}: superseded by the "
                          f"{PREFER_HASH[arm]} wave")
        continue
    ev = next_month(end)
    if ev not in want_eval:
        skipped.append(f"{proj.name}: eval month {ev} is off-panel")
        continue
    # A run dir counts only if it is LOADABLE. IJEPA writes no config.json
    # (it defines no save_pretrained), so model.pt is the one required file
    # and the loader rebuilds the rest from train_meta.json.
    runs = [d for d in sorted(proj.iterdir())
            if d.is_dir() and (d / "model.pt").is_file()]
    if not runs:
        skipped.append(f"{proj.name}: no run dir with model.pt")
        continue
    if len(runs) > 1:
        raise SystemExit(f"{skey} {ev}: {len(runs)} complete runs in "
                         f"{proj.name} -- {[d.name for d in runs]}")
    if ev in found[skey]:
        raise SystemExit(
            f"{skey} {ev}: two projects claim it "
            f"({found[skey][ev]['project']} and {proj.name}).\n"
            f"If one wave supersedes the other, say so in PREFER_HASH at the "
            f"top of this file -- do not let the newer one win by accident.")
    found[skey][ev] = {
        "series_key": skey, "project": proj.name, "run_id": runs[0].name,
        "train_start": start, "train_end": end, "eval_month": ev,
        "ckpt_dir": str(runs[0]),
    }

for skey in ARM_KEY.values():
    rows.extend(found[skey].values())

out = Path(sys.argv[1] if len(sys.argv) > 1
           else "/data/lab/market-jepa-checkpoints/_scratch/latent_eval/"
                "manifest_6mo.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"ckpts": rows}, indent=1))
print(f"wrote {out}  ({len(rows)} rows over {len(want_eval)} panel months)")
if superseded:
    print(f"  {len(superseded)} project(s) skipped as superseded "
          f"(PREFER_HASH): {', '.join(sorted({s.split('-')[2] for s in superseded}))}")
for skey in sorted(found):
    have = found[skey]
    missing = [m for m in want_eval if m not in have]
    print(f"  {skey:16s} {len(have):2d}/{len(want_eval)}" +
          ("  (complete)" if not missing else f"  MISSING {missing}"))
complete = [k for k, v in found.items() if len(v) == len(want_eval)]
print(f"{len(complete)}/{len(found)} arms cover the whole panel: "
      f"{' '.join(sorted(complete))}")
for s in skipped:
    print(f"  skipped {s}")
