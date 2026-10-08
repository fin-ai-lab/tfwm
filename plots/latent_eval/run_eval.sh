#!/bin/bash
# run_eval.sh — the latent-structure evaluation pipeline, end to end.
#
# ONE driver, the same stage scripts for every model family, run MONTH BY
# MONTH (stages 1-3) and then once over the panel (the shard merge, 4-5).
# plots/tsfm_layers/run_latent.sh is this pipeline instantiated for the
# frozen-TSFM layer sweep on the optimization months; this file is the
# general form, defaulting to the reported panel: the 18 IC-era encoders of
# industry_nn_sweep.MODEL_ORDER over the 32 canonical sweep months.
#
#   1. fixed-panel latent structure T1-T4    fixed_panel/fixed_panel_metrics.py
#   2. full-day embeddings                   factors/build_fullday_embs.py
#   3. return panel + Pelger factors         factors/build_return_panel.py
#                                            factors/estimate_factors.py
#   4. the two factor analyses               factors/decode_loadings.py
#                                            factors/subspace_alignment.py
#   5. summaries                             factors/factor_summary.py
#                                            fixed_panel_table.py
#
# Every stage is idempotent: panels, factors, and embeddings skip what exists;
# the fixed-panel JSON merges per (model, metric); the analyses overwrite
# their JSON for exactly the series they were given.
#
# Parameters (env, all optional):
#   MODELS         comma list of fixed-panel registry keys for stage 1
#                  (default: industry_nn_sweep.MODEL_ORDER — the 18 reported
#                  IC-era encoders plus the random-init floor)
#   SERIES         space list of emb_<series> keys for stages 2 and 4
#                  (default: MODEL_ORDER's trained encoders + randvit_s0..4)
#   MONTHS         space list of EVAL months; unset derives them from the
#                  registry's training months (+1). THE LOOP IS MONTH-MAJOR,
#                  so stage 1 is now always given --months -- the retired
#                  glob-resolved keys (LEGACY_GLOB_KEYS), which resolve a
#                  checkpoint by TRAINING month, therefore raise rather than
#                  run. They are out of every default and their checkpoints
#                  are off this box; a run that needs them wants the pre-
#                  2026-09-14 stage-major driver (git history).
#   REDO           1 = re-run stage 1 for a month whose shard already
#                  exists (default: skip it, so a restart is cheap).
#   STAGGER        seconds between starting one month and the next while
#                  the pool is filling (default 0). Keeps the first pass from
#                  running every worker's CPU and GPU phases in lockstep.
#   JOBS           months in flight at once (default 1). Collection is CPU-
#                  bound and the forwards are GPU-bound, so >1 is what keeps
#                  the GPU busy; each month writes its own stage-1 shard, so
#                  they never contend for a file.
#   TAG            result-file suffix (default "" = the reported files;
#                  run_latent.sh uses _tsfmlayers)
#                  tsfm_* keys in SERIES are used by stage 4 only; stage 2
#                  builds those families through TSFM_FAMILIES.
#   TSFM_FAMILIES  TSFM families for stage 2 (default: NONE — deliberately.
#                  A missing layer file makes the builder redo the family's
#                  whole depth sweep, ~12 GB/month; that is run_latent.sh's
#                  job, not a side effect of a trained-model re-eval.)
#
# Usage (from the repo root, in tmux — stages 1-2 want the A40):
#   tmux new -s latent_eval -d 'bash plots/latent_eval/run_eval.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

# streaming decode stages through TMPDIR; /tmp on the data host is small and
# nearly full, and a zstd-only month expands to more than what is left.
export TMPDIR="${TMPDIR:-lab/market-jepa-checkpoints/_scratch/latent_eval}"
mkdir -p "${TMPDIR}"

# The 18 reported encoders resolve through the noclamp manifest, and their
# rows are split across files: the ssl *_final rows are in the repo manifest,
# the pair_*/sup_*_w8 rows in a machine-local one beside the caches. That
# split is why the reported panel was produced as two tagged runs. Naming
# both here (":"-separated, later wins) lets ONE bare run cover all 18.
# A manifest row whose checkpoint is missing is skipped loudly, so an
# unreadable extra file here costs a warning, not a wrong table.
if [ -z "${MJ_NOCLAMP_MANIFEST:-}" ]; then
    MJ_NOCLAMP_MANIFEST="plots/metrics/noclamp_manifest.json"
    for m in "${TMPDIR}"/manifest_*.json; do
        [ -f "${m}" ] && MJ_NOCLAMP_MANIFEST="${MJ_NOCLAMP_MANIFEST}:${m}"
    done
    export MJ_NOCLAMP_MANIFEST
fi
echo "=== manifests: ${MJ_NOCLAMP_MANIFEST}"

FP=plots/latent_eval/fixed_panel
FS=plots/latent_eval/factors
TAG="${TAG:-}"

MONTHS="${MONTHS:-$(uv run python -c "
import sys; sys.path.insert(0, '${FP}')
import panel_lib as eg
from industry_nn_sweep import MONTHS as TRAIN
print(' '.join(eg.eval_window_t_plus_n(ym, 1)[0] for ym in TRAIN))")}"

SERIES="${SERIES:-$(uv run python -c "
import sys; sys.path.insert(0, '${FP}')
from industry_nn_sweep import MODEL_ORDER, MODEL_SPECS
print(' '.join(k for k in MODEL_ORDER if k != 'random'
               and 'tsfm_family' not in MODEL_SPECS[k]
               and 'sup_family' not in MODEL_SPECS[k]))") randvit_s0 randvit_s1 randvit_s2 randvit_s3 randvit_s4}"
echo "=== months: ${MONTHS}"
echo "=== series: ${SERIES}"

# ── The month loop ───────────────────────────────────────────────────────────
#
# MONTH-MAJOR, NOT STAGE-MAJOR. Stages 1-3 each read the SAME month of raw
# data out of mosaic -- the panel windows, the full-day grid, the 5-min mid
# panel -- so walking the whole month list once per stage read every month
# three times, each read cold. One month's work is now done end to end before
# the next month starts.
#
# It also changes what a crash leaves behind: N complete months rather than
# one complete stage and nothing else.
#
# STAGES 4-5 STAY AT THE END because they are cross-month by construction --
# decode_loadings and subspace_alignment fit per month and then pool a t across
# months, and they read embeddings out of ff_fullday_cache rather than raw
# data, so there is nothing to co-locate and nothing to gain.
#
# JOBS MONTHS RUN AT ONCE. Panel collection is CPU/IO-bound and the forwards
# are GPU-bound, so a serial run leaves the GPU idle through every collection
# -- measured at 0% during stage 1's collect phase against 93% during its
# forwards. Overlapping months fills that: while one month is on the GPU the
# others are reading data. They compete for the GPU and that is the point;
# CUDA time-slices them and these models are small next to the card.
#
# Each month writes its OWN stage-1 shard (--out-suffix ${TAG}_m<month>), so
# parallel months never read-modify-write one json. merge_month_shards.py
# pools them afterwards, with the same arithmetic the in-file merge uses.
JOBS="${JOBS:-1}"
LOGS="${FP}/logs${TAG}"
mkdir -p "${LOGS}"
echo "=== ${JOBS} month(s) at a time; per-month logs in ${LOGS}"

run_month() {
    local M="$1" log="${LOGS}/${M}.log"
    {
        echo "=== ${M} stage 1: fixed-panel latent structure (T1-T4) ==="
        # STAGE 1 IS THE ONLY STAGE WITHOUT ITS OWN SKIP -- stages 2 and 3 both
        # check their caches -- and it is the expensive one (every model's
        # forward over the panel). Its shard existing means this month is
        # already done, so a restart after a failure costs the months that did
        # not finish rather than all of them. REDO=1 forces it.
        local shard="${FP}/fixed_panel_P3S2${TAG}_m${M}.json"
        if [ -f "${shard}" ] && [ "${REDO:-0}" != "1" ]; then
            echo "${M}: stage-1 shard exists — skip (REDO=1 to force)"
        else
            local a=(--out-suffix "${TAG}_m${M}" --months "${M}")
            [ -n "${MODELS:-}" ] && a+=(--models "${MODELS}")
            uv run python ${FP}/fixed_panel_metrics.py "${a[@]}"
        fi

        echo "=== ${M} stage 2: full-day embeddings ==="
        uv run python ${FS}/build_fullday_embs.py --months "${M}" \
            --families ${TSFM_FAMILIES:-} --series ${SERIES_ENC} \
            ${RANDVIT_SEEDS:+--randvit-seeds ${RANDVIT_SEEDS}}

        echo "=== ${M} stage 3: the return panel and its latent factors ==="
        uv run python ${FS}/build_return_panel.py "${M}"
        uv run python ${FS}/estimate_factors.py "${M}"
        echo "=== ${M} MONTH_DONE"
    } >"${log}" 2>&1
}

# The random-init floor is NOT a --series key: it has no registry entry and no
# checkpoint, so build_fullday_embs takes it through --randvit-seeds and builds
# the trunk itself. Passing it in --series only appeared to work because the
# reported months' emb_randvit_s*.npz already exist and the builder skips what
# it finds; on any NEW month set the same command dies with
# KeyError: 'randvit_s0'. Split the two here so the driver works off-panel.
RANDVIT_SEEDS="$(echo ${SERIES} | tr ' ' '\n' \
    | sed -n 's/^randvit_s\([0-9][0-9]*\)$/\1/p' | tr '\n' ' ')"
# TSFM KEYS COME OUT HERE TOO, for the same reason randvit does: a frozen TSFM
# has no checkpoint, so load_series_encoder raises on tsfm_* and the builder
# takes the whole family through --families instead. But stage 4 DOES want them
# by key (one ridge per layer), so only stage 2's list is narrowed -- SERIES
# itself keeps them. Without this split a run that evaluates both a trained
# arm and a TSFM arm under one tag is impossible: putting the layer keys in
# SERIES kills stage 2, leaving them out silently drops them from stage 4.
# `|| true`: a TSFM-only roster leaves nothing, grep exits 1, and under
# pipefail that silently ended the whole run before any month started.
SERIES_ENC="$(echo ${SERIES} | tr ' ' '\n' \
    | { grep -v '^randvit_s[0-9][0-9]*$' || true; } \
    | { grep -v '^tsfm_' || true; } | tr '\n' ' ')"

# A FAILING MONTH MUST NOT KILL THE OTHER THIRTY. Each month's status is
# recorded and the run fails at the END if any month did, so one bad month
# costs that month rather than the night.
declare -A MONTH_PID=()
FAILED=""
for M in ${MONTHS}; do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do
        wait -n || true
    done
    run_month "${M}" &
    MONTH_PID["${M}"]=$!
    echo "--- ${M} started (pid ${MONTH_PID[${M}]})"
    # STAGGER THE STARTS. A month alternates between collecting (CPU, GPU
    # idle) and forwarding (GPU busy). Workers launched together run those
    # phases in LOCKSTEP -- all collecting, then all forwarding -- which is
    # most of the idle the parallelism is meant to remove: measured 48% mean
    # GPU with four simultaneous starts against 0%/93% alternation serially.
    # Offsetting them puts some workers on the GPU while others read data.
    # They drift apart on their own as months finish at different rates; this
    # only stops the first pass from being synchronized.
    # `if` rather than an && chain purely for legibility -- the chain is safe
    # under `set -e` (a failing non-last command of an && list does not trip
    # it), which is worth knowing before "fixing" the other four in this file.
    if [ "${STAGGER:-0}" != "0" ] && [ "$(jobs -rp | wc -l)" -lt "${JOBS}" ]
    then
        sleep "${STAGGER}"
    fi
done
for M in ${MONTHS}; do
    if wait "${MONTH_PID[${M}]}"; then
        echo "--- ${M} ok"
    else
        echo "--- ${M} FAILED (see ${LOGS}/${M}.log)"
        FAILED="${FAILED} ${M}"
    fi
done
[ -n "${FAILED}" ] && echo "=== MONTHS FAILED:${FAILED}"

echo "=== merging the per-month fixed-panel shards ==="
uv run python ${FP}/merge_month_shards.py --tag "${TAG}" \
    2>&1 | tee "${FP}/fixed_panel${TAG}.out"

echo "=== stage 4: Pelger latent-factor analyses ==="
uv run python ${FS}/decode_loadings.py --months ${MONTHS} --series ${SERIES} \
    --tag "${TAG}" 2>&1 | tee "${FS}/decode_loadings${TAG}.out"
uv run python ${FS}/subspace_alignment.py --months ${MONTHS} --series ${SERIES} \
    --tag "${TAG}" 2>&1 | tee "${FS}/subspace_alignment${TAG}.out"

echo "=== stage 5: summaries ==="
uv run python ${FS}/factor_summary.py --tag "${TAG}" \
    2>&1 | tee "${FS}/factor_summary${TAG}.out"
# One level up from ${FP}: the table spans both stages (T1-T4 and the two
# factor columns), so it is not a fixed_panel/ script any more.
uv run python plots/core/fixed_panel_table.py --tags "${TAG}" \
    2>&1 | tee "${FP}/fixed_panel_table${TAG}.out"

[ -n "${FAILED}" ] && { echo "incomplete:${FAILED}"; exit 1; }

echo "LATENT_EVAL_PIPELINE_DONE"
