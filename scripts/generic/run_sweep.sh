#!/bin/bash
# run_sweep.sh — Submit a sweep to a generic cloud cluster, one sbatch
# job per sweep value, for a SINGLE train month (+ its next-month eval).
#
# Usage:  ./scripts/generic/run_sweep.sh <variant> <sweep_file> [YYYY-MM]
#         ./scripts/generic/run_sweep.sh h100 sweeps/lejepa_lamb.sh 2023-01
#
# Defaults to 2023-01 (train) + 2023-02 (eval), matching schema defaults.
# Contrast with run_all_months_sweep.sh, which fans one bundled job per
# sampled month (all values run sequentially on one GPU per month).

set -euo pipefail

CLUSTER_VARIANT="${1:?Usage: $0 <variant> <sweep_file> [YYYY-MM]}"
export CLUSTER_VARIANT
shift

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
source "${CLUSTER_DIR}/lib/setup.sh"
source "${CLUSTER_DIR}/lib/sync.sh"

SWEEP_FILE_ARG="${1:?Usage: $0 <variant> <sweep_file> [YYYY-MM]}"
TRAIN_YM="${2:-2023-01}"

SWEEP_FILE="${SWEEP_FILE_ARG}"
[[ "${SWEEP_FILE}" = /* ]] || SWEEP_FILE="${LOCAL_REPO}/scripts/${SWEEP_FILE}"
[ -f "${SWEEP_FILE}" ] || error "Sweep file not found: ${SWEEP_FILE}"

SWEEP_FILE_REL="$(realpath --relative-to="${LOCAL_REPO}" "${SWEEP_FILE}")"

( source "${SWEEP_FILE}" )
source "${SWEEP_FILE}"
: "${SWEEP_NAME:?Sweep file must define SWEEP_NAME}"
: "${SBATCH_EXTRA:=}"
[ ${#SWEEP_VALUES[@]} -gt 0 ] || error "Sweep file must define SWEEP_VALUES"
declare -F sweep_train_args >/dev/null || error "Sweep file must define sweep_train_args()"

# ── Dates ──────────────────────────────────────────────────────────────────

last_day_of_month() { date -d "${1}-01 +1 month -1 day" +%Y-%m-%d; }
next_month() { date -d "${1}-01 +1 month" +%Y-%m; }

TRAIN_START="${TRAIN_YM}-01"
TRAIN_END="$(last_day_of_month "${TRAIN_YM}")"
EVAL_YM="$(next_month "${TRAIN_YM}")"
EVAL_START="${EVAL_YM}-01"
EVAL_END="$(last_day_of_month "${EVAL_YM}")"

info "Sweep: ${SWEEP_NAME} (${#SWEEP_VALUES[@]} jobs)"
info "Train: ${TRAIN_START} .. ${TRAIN_END}    Eval: ${EVAL_START} .. ${EVAL_END}"
info "Grid cache: GRID_CACHE_GB=${GRID_CACHE_GB:-0} GiB/worker (0 = off)"

# ── Setup & sync ──────────────────────────────────────────────────────────

cluster_setup
cluster_sync

COMMIT_HASH=$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)

# ── Per-value submission ──────────────────────────────────────────────────

JOB_IDS=()
for VAL in "${SWEEP_VALUES[@]}"; do
    info "Submitting ${VAL} ..."
    SBATCH_OUTPUT=$(ssh "${CLUSTER_HOST}" \
        "cd ${CLUSTER_REPO} && export CLUSTER_VARIANT='${CLUSTER_VARIANT}' SWEEP_FILE_REL='${SWEEP_FILE_REL}' SWEEP_VALUE_ONLY='${VAL}' TRAIN_START='${TRAIN_START}' TRAIN_END='${TRAIN_END}' EVAL_START='${EVAL_START}' EVAL_END='${EVAL_END}' COMMIT_HASH='${COMMIT_HASH}' GRID_CACHE_GB='${GRID_CACHE_GB:-}' XS_STATS_DIR='${XS_STATS_DIR:-}' STAGE_ONLY='${STAGE_ONLY:-0}' POST_TRAIN_IC_EVAL='${POST_TRAIN_IC_EVAL:-1}' && sbatch \
            --export=ALL \
            --job-name=${SWEEP_NAME}-${TRAIN_YM}-${VAL//[^a-zA-Z0-9_-]/_} \
            --cpus-per-task=${SBATCH_CPUS_PER_TASK} \
            --mem=${SBATCH_MEM} \
            ${SBATCH_EXTRA} \
            scripts/generic/slurm_train_bundle.sh")
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${VAL}: ${SBATCH_OUTPUT}"
    JOB_IDS+=("${JOB_ID}")
    info "  Submitted job ${JOB_ID} (${VAL})"
done

echo ""
echo "  Sweep: ${SWEEP_NAME} (${TRAIN_YM})"
echo "  Jobs:  ${JOB_IDS[*]}"
echo "  Monitor:    ssh ${CLUSTER_HOST} 'squeue -u \$USER'"
echo "  Cancel all: ssh ${CLUSTER_HOST} 'scancel ${JOB_IDS[*]}'"
echo ""
