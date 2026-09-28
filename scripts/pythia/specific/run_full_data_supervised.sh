#!/bin/bash
# run_full_data_supervised.sh — Submit the full-history supervised baselines.
#
# Enumerates every (year, month) under ${BLL01_DATA_DIR}/1Hz_mosaic_mnth from
# 2008-01 onwards, drops the very last month (it's needed as the next-month
# eval window for the previous one), and submits bundled SLURM jobs via
# slurm_train_bundle.sh. Each job stages mosaic data once per month and trains
# all selected tasks (return / vol_change / spread_change) sequentially on the
# same GPU.
#
# EMPTY-MONTH FILTER. Mosaic exposes directories that contain nothing: 2007/12
# and 2025/01..03 each hold a lone index.json declaring {"shards": []}. They
# are indistinguishable from real months by name and crash the loader on open.
# Left in, the tail of this sweep submitted training jobs for 2025-01 and
# 2025-02 that could not start, and 2024-12 trained fine but died in its
# post-training IC eval because its next-month eval window (2025-01) is a stub.
# Months are therefore screened on the shard list in index.json, not on the
# directory existing — so the real last trainable month is 2024-11, and the
# sweep spans 2008-01..2024-11 (203 months).
#
# MONTHS PER JOB. --months-per-job N puts N consecutive months in one bundle,
# for N x 3 sequential trainings. At the default 2 the 203 months become 102
# jobs, which clears the 200-job QOS submit cap in a single wave. A month costs
# roughly 3h wall (3 tasks x ~35min train + ~15min IC eval, staged once), so
# even N=4 sits far inside the 3-day limit; N is bounded by how much work one
# node failure takes down, not by time. Consecutive months also share staging:
# a 2-month bundle stages 3 distinct months rather than 2 x 2.
#
# Default partition is standard_l40s.
#
# Usage:
#   ./pythia/run_full_data_supervised.sh
#   ./pythia/run_full_data_supervised.sh --return                        # only the return task
#   ./pythia/run_full_data_supervised.sh --vol_change --spread_change    # subset
#   ./pythia/run_full_data_supervised.sh --partition bll
#   ./pythia/run_full_data_supervised.sh --months-per-job 1              # one month per job
#   ./pythia/run_full_data_supervised.sh --first-n-months 100            # cap submissions
#   ./pythia/run_full_data_supervised.sh --backward                      # newest months first
#   ./pythia/run_full_data_supervised.sh --commit-hash 7fad69            # resume prior wandb projects
#   ./pythia/run_full_data_supervised.sh --start-ym 2010-01              # later start
#
# --first-n-months counts MONTHS, not jobs, and is applied before bundling.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

# ── Parse options ───────────────────────────────────────────────────────────

PARTITION="standard_l40s"
COMMIT_HASH_OVERRIDE=""
BACKWARD=0
FIRST_N_MONTHS=""
MONTHS_PER_JOB=2
START_YM="2008-01"
TASK_FILTER=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition|-p)         PARTITION="${2:?--partition requires a value}"; shift 2 ;;
        --partition=*)          PARTITION="${1#*=}"; shift ;;
        --commit-hash|-c)       COMMIT_HASH_OVERRIDE="${2:?--commit-hash requires a value}"; shift 2 ;;
        --commit-hash=*)        COMMIT_HASH_OVERRIDE="${1#*=}"; shift ;;
        --backward)             BACKWARD=1; shift ;;
        --first-n-months)       FIRST_N_MONTHS="${2:?--first-n-months requires a positive integer}"; shift 2 ;;
        --first-n-months=*)     FIRST_N_MONTHS="${1#*=}"; shift ;;
        --months-per-job)       MONTHS_PER_JOB="${2:?--months-per-job requires a positive integer}"; shift 2 ;;
        --months-per-job=*)     MONTHS_PER_JOB="${1#*=}"; shift ;;
        --start-ym)             START_YM="${2:?--start-ym requires a YYYY-MM value}"; shift 2 ;;
        --start-ym=*)           START_YM="${1#*=}"; shift ;;
        --return)               TASK_FILTER+=("return"); shift ;;
        --vol_change)           TASK_FILTER+=("vol_change"); shift ;;
        --spread_change)        TASK_FILTER+=("spread_change"); shift ;;
        --return_reg)           TASK_FILTER+=("return_reg"); shift ;;
        --) shift; break ;;
        -*) error "Unknown flag: $1" ;;
        *)  break ;;
    esac
done

case "${PARTITION}" in
    standard_hopper|standard_l40s|bll) ;;
    *) error "Invalid --partition: ${PARTITION} (expected: standard_hopper, standard_l40s, bll)" ;;
esac

[[ "${START_YM}" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "--start-ym must be YYYY-MM, got: ${START_YM}"
[[ "${MONTHS_PER_JOB}" =~ ^[1-9][0-9]*$ ]] || error "--months-per-job must be positive, got: ${MONTHS_PER_JOB}"

# ── Sweep file ──────────────────────────────────────────────────────────────

# OVERRIDABLE so the same month enumeration can drive a different recipe.
# Everything below (the empty-month screen, the bundling, the QOS-aware wave)
# is about WHICH MONTHS EXIST and is recipe-independent, so forking this
# launcher to change the sweep file would duplicate the one part that is
# genuinely hard to get right.
#
# THE DEFAULT IS THE MULTIHEAD, and it has to be a full_history sweep: the
# month-set check below refuses anything else, so a default that declares
# sampled32 makes a bare invocation error every time. It defaulted to
# supervised_specialists.sh (sampled32) between 62bffe2 and 2026-09-13 --
# submitted before the check existed, that default is what trained 198 months
# by mistake, and after the check it could not run at all.
SWEEP_FILE_REL="${SWEEP_FILE_REL:-scripts/sweeps/full_data_multihead.sh}"
SWEEP_FILE="${LOCAL_REPO}/${SWEEP_FILE_REL}"
[ -f "${SWEEP_FILE}" ] || error "Sweep file not found: ${SWEEP_FILE}"

# Source locally to validate + pick up SWEEP_NAME and (filtered) SWEEP_VALUES
# for the info banner. TASK_FILTER must be exported so the bundle script's
# source of the same file at job time produces the same filtered list.
TASK_FILTER_STR="${TASK_FILTER[*]:-}"
export TASK_FILTER="${TASK_FILTER_STR}"

( source "${SWEEP_FILE}" )
source "${SWEEP_FILE}"
: "${SWEEP_NAME:?Sweep file must define SWEEP_NAME}"

# WHICH MONTHS THIS SWEEP IS FOR. Declared by the sweep, checked here: this
# launcher enumerates EVERY month the mosaic has, and handing it a sweep built
# for the 32 reported months runs a different, much larger experiment without
# erroring. That happened on 2026-09-13 with supervised_specialists.sh.
case "${SWEEP_MONTH_SET:-any}" in
    any|full_history) ;;
    sampled32) error "${SWEEP_FILE_REL} declares SWEEP_MONTH_SET=sampled32 — submit it with scripts/pythia/run_all_months_sweep.sh, not this launcher." ;;
    *) error "${SWEEP_FILE_REL} declares an unknown SWEEP_MONTH_SET='${SWEEP_MONTH_SET}'." ;;
esac
[ ${#SWEEP_VALUES[@]} -gt 0 ] || error "Sweep file produced empty SWEEP_VALUES"

# ── Enumerate available months ──────────────────────────────────────────────

MOSAIC_ROOT="${BLL01_DATA_DIR}/1Hz_mosaic_mnth"
[ -d "${MOSAIC_ROOT}" ] || error "Mosaic root not found locally: ${MOSAIC_ROOT}"

# A month counts only if its index.json lists at least one shard. Mosaic
# carries empty placeholder dirs (2007/12, 2025/01..03) holding nothing but
# {"shards": [], "version": 2}; they look like real months to `ls` and crash
# the loader on open. Screening here rather than by name keeps this correct
# when mosaic is extended.
mapfile -t ALL_MONTHS < <(
    cd "${MOSAIC_ROOT}"
    for y in $(ls -1 | grep -E '^[0-9]{4}$' | sort); do
        for m in $(ls -1 "${y}" | grep -E '^(0[1-9]|1[0-2])$' | sort); do
            IDX="${y}/${m}/index.json"
            [ -f "${IDX}" ] || continue
            grep -q '"shards"[[:space:]]*:[[:space:]]*\[[[:space:]]*\]' "${IDX}" && continue
            echo "${y}-${m}"
        done
    done
)
[ ${#ALL_MONTHS[@]} -gt 0 ] || error "No months discovered under ${MOSAIC_ROOT}"

# THE FIRST FEW MONTHS HAVE NO SPAN BEHIND THEM. A run trains the
# DatasetConfig.train_span_months ending at its month, so the earliest
# trainable month is (first month of data + span - 1); asking for anything
# earlier stages a window that does not exist and the job skips the month
# after paying for the attempt. The mosaic and the day store both begin at
# ${ALL_MONTHS[0]}.
TRAIN_SPAN_MONTHS="${TRAIN_SPAN_MONTHS:-$(cd "${LOCAL_REPO}" && uv run python -c 'from market_jepa.schemas import DatasetConfig; print(DatasetConfig.train_span_months)')}"
[[ "${TRAIN_SPAN_MONTHS}" =~ ^[1-9][0-9]*$ ]] || error "could not resolve DatasetConfig.train_span_months"
SPAN_FLOOR=$(date -d "${ALL_MONTHS[0]}-01 +$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m)
if [[ "${START_YM}" < "${SPAN_FLOOR}" ]]; then
    info "Span is ${TRAIN_SPAN_MONTHS} months and data begins ${ALL_MONTHS[0]}: starting at ${SPAN_FLOOR}, not ${START_YM}."
    START_YM="${SPAN_FLOOR}"
fi

# Filter to >= START_YM
MONTHS=()
for YM in "${ALL_MONTHS[@]}"; do
    [[ "${YM}" < "${START_YM}" ]] && continue
    MONTHS+=("${YM}")
done
[ ${#MONTHS[@]} -ge 2 ] || error "Need at least 2 months at/after ${START_YM} (found ${#MONTHS[@]})"

# Drop the last month — it's reserved as the next-month eval for the previous one.
LAST_MONTH="${MONTHS[-1]}"
unset 'MONTHS[-1]'
MONTHS=("${MONTHS[@]}")
info "Reserving ${LAST_MONTH} as forward-eval window (not trained on)."

if [ "${BACKWARD}" -eq 1 ]; then
    REVERSED=()
    for (( i=${#MONTHS[@]}-1; i>=0; i-- )); do REVERSED+=("${MONTHS[i]}"); done
    MONTHS=("${REVERSED[@]}")
    info "Iterating months newest → oldest."
fi
if [ -n "${FIRST_N_MONTHS}" ]; then
    [[ "${FIRST_N_MONTHS}" =~ ^[1-9][0-9]*$ ]] || error "--first-n-months must be positive, got: ${FIRST_N_MONTHS}"
    (( FIRST_N_MONTHS <= ${#MONTHS[@]} )) || error "--first-n-months=${FIRST_N_MONTHS} exceeds ${#MONTHS[@]} available months"
    MONTHS=("${MONTHS[@]:0:${FIRST_N_MONTHS}}")
    info "Limiting to first ${FIRST_N_MONTHS} months."
fi

# Bundle consecutive months so the job count clears the 200-job QOS cap.
CHUNKS=()
for (( i=0; i<${#MONTHS[@]}; i+=MONTHS_PER_JOB )); do
    CHUNKS+=("${MONTHS[*]:i:MONTHS_PER_JOB}")
done

# WHICH TARGET DEFINITION THE SWEEP IS TRAINED AND SCORED ON. The anchor tables
# are the (mu, sigma) OF a particular target, and all three targets became
# forward-window differences on 2026-08-22, so every old table is stale. Both
# the training config and the in-job scorer read this -- a run that trained
# under one and scored under the other measures a distribution shift.
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
XS_MIN_TABLES="${XS_MIN_TABLES:-200}"

info "Sweep: ${SWEEP_NAME}  partition=${PARTITION}"
info "Tasks per month (${#SWEEP_VALUES[@]}): ${SWEEP_VALUES[*]}"
info "Months (${#MONTHS[@]}): ${MONTHS[0]} … ${MONTHS[-1]}"
info "Bundled jobs: ${#CHUNKS[@]} (${MONTHS_PER_JOB} month(s)/job, $(( MONTHS_PER_JOB * ${#SWEEP_VALUES[@]} )) trainings/job)"
info "Anchor tables: ${XS_STATS_DIR} (min ${XS_MIN_TABLES})"

# ── Setup & sync ────────────────────────────────────────────────────────────

pythia_setup
pythia_sync

if [ -n "${COMMIT_HASH_OVERRIDE}" ]; then
    COMMIT_HASH="${COMMIT_HASH_OVERRIDE}"
    info "Using overridden COMMIT_HASH=${COMMIT_HASH}."
else
    COMMIT_HASH=$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)
fi

SBATCH_EXTRA="--partition=${PARTITION} ${SBATCH_EXTRA:-}"

# ── Submit one bundled job per month ────────────────────────────────────────

# Each month's train/eval range is derived inside slurm_train_bundle.sh's
# multi-month mode: train = the DatasetConfig.train_span_months span ENDING at
# the month, eval = the next calendar month. Deriving it here too would be a
# second copy of that rule to keep in sync.
JOB_IDS=()
for CHUNK in "${CHUNKS[@]}"; do
    read -ra CHUNK_MONTHS <<< "${CHUNK}"
    FIRST_YM="${CHUNK_MONTHS[0]}"
    LAST_YM="${CHUNK_MONTHS[-1]}"
    if [ "${FIRST_YM}" = "${LAST_YM}" ]; then
        JOB_TAG="${FIRST_YM}"
    else
        JOB_TAG="${FIRST_YM}_${LAST_YM}"
    fi

    info "Submitting ${JOB_TAG}  (${#CHUNK_MONTHS[@]} month(s): ${CHUNK}) ..."
    SBATCH_OUTPUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export \
            SWEEP_FILE_REL='${SWEEP_FILE_REL}' \
            MONTHS='${CHUNK}' \
            COMMIT_HASH='${COMMIT_HASH}' \
            TASK_FILTER='${TASK_FILTER_STR}' \
            XS_STATS_DIR='${XS_STATS_DIR}' XS_MIN_TABLES='${XS_MIN_TABLES}' \
            SKIP_CKPT_SYNC='${SKIP_CKPT_SYNC:-0}' \
            DAYSTORE='${DAYSTORE:-0}' \
            POST_TRAIN_PROBE='${POST_TRAIN_PROBE:-1}' \
            PANEL_CACHE_REQUIRED='${PANEL_CACHE_REQUIRED:-1}' \
            WANDB_NO_SUFFIX=1 && sbatch \
                --export=ALL \
                --job-name=${SWEEP_NAME}-${JOB_TAG} \
                ${SBATCH_EXTRA} \
                scripts/pythia/slurm_train_bundle.sh")
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${JOB_TAG}: ${SBATCH_OUTPUT}"

    JOB_IDS+=("${JOB_ID}")
    info "  Submitted job ${JOB_ID} (${JOB_TAG})"
done

echo ""
echo "  Sweep:        ${SWEEP_NAME}"
echo "  Partition:    ${PARTITION}"
echo "  Tasks/month:  ${SWEEP_VALUES[*]}"
echo "  Months:       ${#MONTHS[@]}  (${MONTHS_PER_JOB}/job)"
echo "  Jobs:         ${#JOB_IDS[@]}  (${JOB_IDS[*]})"
echo "  Monitor:      ssh pythia 'squeue -u \$USER'"
echo "  Cancel all:   ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
