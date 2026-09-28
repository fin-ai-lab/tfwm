#!/bin/bash
# run_span_bundle.sh — one bundled job per (eval month, step budget).
#
# Submits sweeps/span_bundle.sh: every training span for one eval month runs
# sequentially inside a single SLURM job, so the longest span's data is staged
# once and every shorter span reads the same staged copy.
#
# Usage:
#   ./scripts/pythia/specific/run_span_bundle.sh 2009-03 2009-10 2015-09 2022-03
#   STEPS="10800 21600" ./scripts/pythia/specific/run_span_bundle.sh 2015-09

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

PARTITION="${PARTITION:-standard_l40s}"
# One job per (eval month, budget token). The token is exported as MAX_STEPS
# for sweeps that spend it as a step count; a sweep that fixes its own
# budget per arm (supervised_return_span.sh) takes BUDGETS=all for one job.
STEPS="${BUDGETS:-${STEPS:-10800 21600}}"
# The span is DatasetConfig.train_span_months; do not restate it here.
MAX_SPAN="${MAX_SPAN:-$(uv run python -c 'from market_jepa.schemas import DatasetConfig; print(DatasetConfig.train_span_months)')}"
SEED="${SEED:-42}"

[ $# -gt 0 ] || error "usage: $0 <eval-month YYYY-MM> [...]"
for E in "$@"; do
    [[ "${E}" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "bad eval month: ${E}"
done

# NO DEFAULT. It used to name scripts/sweeps/span_bundle.sh, which was
# deleted on 2026-09-10, so a bare invocation failed on a missing file
# rather than saying what it wanted. This launcher serves whichever sweep
# the caller names, so requiring the name is both honest and shorter to
# diagnose.
: "${SWEEP_FILE_REL:?set SWEEP_FILE_REL to the sweep this span should run, e.g. scripts/sweeps/supervised_specialists.sh}"
[ -f "${LOCAL_REPO}/${SWEEP_FILE_REL}" ] || error "missing ${SWEEP_FILE_REL}"

pythia_setup
pythia_sync
COMMIT_HASH="${COMMIT_HASH:-$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)}"

JOB_IDS=()
for E in "$@"; do
    EVAL_START="${E}-01"
    EVAL_END=$(date -d "${EVAL_START} +1 month -1 day" +%Y-%m-%d)
    TRAIN_END=$(date -d "${EVAL_START} -1 day" +%Y-%m-%d)
    # Stage from the LONGEST span's start: every shorter span nests inside it.
    TRAIN_START=$(date -d "$(date -d "${TRAIN_END}" +%Y-%m-01) -$((MAX_SPAN - 1)) months" +%Y-%m-01)

    for ST in ${STEPS}; do
        info "eval ${E} st${ST}: train ${TRAIN_START}..${TRAIN_END}, eval ${EVAL_START}..${EVAL_END}"
        OUT=$(ssh "${PYTHIA_HOST}" \
            "cd ${PYTHIA_REPO} && export \
                SWEEP_FILE_REL='${SWEEP_FILE_REL}' \
                TRAIN_START='${TRAIN_START}' TRAIN_END='${TRAIN_END}' \
                EVAL_START='${EVAL_START}' EVAL_END='${EVAL_END}' \
                MAX_STEPS='${ST}' SEED='${SEED}' \
                DAYSTORE='${DAYSTORE:-0}' \
                SWEEP_VALUES_OVERRIDE='${SWEEP_VALUES_OVERRIDE:-}' \
                COMMIT_HASH='${COMMIT_HASH}' && sbatch \
                    --export=ALL \
                    --job-name=span-bundle-${E}-st${ST} \
                    --partition=${PARTITION} \
                    ${SBATCH_EXTRA:-} \
                    scripts/pythia/slurm_train_bundle.sh")
        JOB_ID=$(echo "${OUT}" | awk '{print $4}')
        [ -n "${JOB_ID}" ] || error "sbatch failed for ${E}/${ST}: ${OUT}"
        JOB_IDS+=("${JOB_ID}")
        info "  job ${JOB_ID}"
    done
done

echo ""
echo "  Jobs:       ${#JOB_IDS[@]}  (${JOB_IDS[*]})"
echo "  Cancel all: ssh pythia 'scancel ${JOB_IDS[*]}'"
