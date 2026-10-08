#!/bin/bash
# run_6mo_eval.sh — the six-month-span SSL/LeJEPA wave through the latent suite.
#
# WHY THIS RUN EXISTS. run_meanpool_reeval.sh re-read every latent at the mean
# but could only cover the supervised specialists, the frozen TSFMs and the
# floor: the 18-method IC campaign's checkpoints are off this box, and its SSL
# and LeJEPA arms were trained on a ONE-MONTH span against six-month supervised
# specialists anyway. Those arms were retrained on the matched budget (6-month
# span x 12 passes); this scores them on the SAME 31-month
# panel and the same mean readout, so the table finally compares one budget.
#
# THE SPANS DO NOT POOL. Every key here carries the _6mo suffix and is a
# separate MODEL_ORDER row from its *_final twin. Never merge a _6mo tag with
# the archived one-month result files (plots/latent_eval/_archive).
#
# THE FLOOR AND THE TSFMs ARE NOT RE-RUN. `random` is in MODELS because it is
# seeded per eval month and costs one small forward, which keeps this run's own
# table self-contained; the TSFM layers stay in _meanpool and reach the paper
# table through `fixed_panel_table.py --tags _meanpool _6mo`, which merges the
# two files and takes the shared `random` row bit-identically from either.
#
# THE WAVE MAY STILL BE FILLING. A (arm, month) with no checkpoint is skipped
# loudly by stages 1 and 2 and pooled over the months it has, and each model
# carries its own n_months. Rebuild the manifest first -- it prints per-arm
# coverage -- and top an arm up later with fixed_panel_metrics.py's in-file
# merge (--out-suffix _6mo, no _m<month>), NOT by re-merging shards: the shard
# merge writes the pooled file from the shards present and nothing else.
#
# Usage (from the repo root):
#   uv run python plots/latent_eval/build_6mo_manifest.py \
#       lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_6mo.json
#   tmux new -s latent_6mo -d 'bash plots/latent_eval/run_6mo_eval.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

export TMPDIR=lab/market-jepa-checkpoints/_scratch/latent_eval
mkdir -p "${TMPDIR}"

# The wave has no row in the repo manifest: one project per (arm, span), which
# fits no glob resolver in the registry. build_6mo_manifest.py maps (series,
# eval month) -> ckpt_dir from the PATH. Named explicitly rather than left to
# run_eval.sh's ${TMPDIR}/manifest_*.json glob so this run does not depend on
# what else is sitting in the scratch dir (manifest_supspan.json is).
export MJ_NOCLAMP_MANIFEST="plots/metrics/noclamp_manifest.json:${TMPDIR}/manifest_6mo.json"
[ -f "${TMPDIR}/manifest_6mo.json" ] || {
    echo "no ${TMPDIR}/manifest_6mo.json — run build_6mo_manifest.py first" >&2
    exit 1; }

export TAG="${TAG:-_6mo}"
# Stage 2 must NOT build TSFM families here: a missing layer file sends the
# builder through the whole depth sweep (~12 GB/month) for a family this run
# does not evaluate. _meanpool already built them on these months.
export TSFM_FAMILIES=""

# Sized off the _meanpool run on the same box (64 cores / 188 GB / one A40): a
# month worker is ~4 GB and ~1.3 cores, and the binding constraint is the GPU
# sitting idle through each month's collection phase. Six months in flight,
# offset by about one collection phase so the first pass does not run every
# worker's CPU and GPU phases in lockstep. These arms are 15 small ViTs per
# month rather than _meanpool's 48 keys of TSFM depth sweep, so the card has
# room; raise JOBS if nvidia-smi still shows gaps.
export JOBS="${JOBS:-6}"
export STAGGER="${STAGGER:-90}"

# The SAME 31 eval months as _meanpool: the sweep's train months, minus the one
# with no six-month span (metrics.SUP_SPAN_NO_SPAN), +1. A rank is only
# meaningful if every arm earned it on one panel.
export MONTHS="$(uv run python -c "
import sys; sys.path.insert(0, 'plots'); sys.path.insert(0, 'plots/metrics')
from style import load_sweep_months
from metrics import SUP_SPAN_NO_SPAN
from stable_finance.dataset import next_month
print(' '.join(sorted(next_month(m) for m in
                      set(load_sweep_months()) - SUP_SPAN_NO_SPAN)))")"

# The 14 arms in registry order, plus the floor. Read off MODEL_ORDER rather
# than listed again here, so this driver cannot drift from the table's rows.
export MODELS="${MODELS:-$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import MODEL_ORDER
print(','.join([k for k in MODEL_ORDER if k.endswith('_6mo')] + ['random']))")}"

# Stage 2/4 series: the same 14 arms by emb key, plus the five random-init
# seeds. `random` is a stage-1-only key (built in-process, no checkpoint); the
# floor's EMBEDDINGS are randvit_s0..4, which run_eval.sh splits out of SERIES
# into --randvit-seeds. Their npz already exist on these months from _meanpool,
# so stage 2 skips them and stage 4 still gets its floor.
export SERIES="${SERIES:-$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import MODEL_ORDER
print(' '.join(k for k in MODEL_ORDER if k.endswith('_6mo')))") randvit_s0 randvit_s1 randvit_s2 randvit_s3 randvit_s4}"

echo "=== TAG=${TAG}  JOBS=${JOBS}"
echo "=== months (${MONTHS// /,})"
echo "=== $(echo ${MONTHS} | wc -w) months, $(echo ${MODELS//,/ } | wc -w) stage-1 models, $(echo ${SERIES} | wc -w) stage-4 series"
exec bash plots/latent_eval/run_eval.sh
