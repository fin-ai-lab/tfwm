#!/bin/bash
# run_variance_decomp.sh — Fan out the variance-decomposition sweep: one
# bundled SLURM job per (training month x series); each job loops over
# seeds 0..9 — 10 sequential trainings/job, ~6-10 h at the observed
# ~35-60 min per single-month run.
#
# Months are the frozen 10-month subsample (seed 42) of the 32 sweep months —
# see scripts/experiments/sample_variance_months.py.
#
# Usage (from bll01):
#   ./scripts/pythia/specific/run_variance_decomp.sh
#   ./scripts/pythia/specific/run_variance_decomp.sh --partition standard_l40s
#   ./scripts/pythia/specific/run_variance_decomp.sh --months 2008-02,2019-12   # subset / resubmit
#   ./scripts/pythia/specific/run_variance_decomp.sh --series lejepa
#   ./scripts/pythia/specific/run_variance_decomp.sh --seeds "0 1 2"
#   ./scripts/pythia/specific/run_variance_decomp.sh --commit-hash 7fad69       # resume prior wandb project
#   XS_STATS_DIR=xs_anchor_stats_fwdvwap60 ./scripts/pythia/specific/run_variance_decomp.sh
#
# XS_STATS_DIR NAMES A TARGET DEFINITION, not a path. The anchor tables are the
# (mu, sigma) OF a particular target, and all three targets became
# forward-window differences on 2026-08-22 (docs/return_bad_calculation.md).
# It is checked here rather than on the node because a missing table surfaces
# ~40 minutes into a job, after the stage and the venv build.
#
# Checkpoints land at /data/lab/market-jepa-checkpoints/variance-decomp-<hash>/
# (rsync'd back from pythia by slurm_variance_decomp.sh); all wandb runs go to
# the single project variance-decomp-<hash>, run names vd_<series>_<YM>_seed-<N>.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

# ── Parse options ───────────────────────────────────────────────────────────

PARTITION="standard_hopper"
COMMIT_HASH_OVERRIDE=""
MONTH_FILTER=""
SERIES_FILTER=""
VD_SEEDS_OVERRIDE=""
VD_EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition|-p)   PARTITION="${2:?--partition requires a value}"; shift 2 ;;
        --partition=*)    PARTITION="${1#*=}"; shift ;;
        --commit-hash|-c) COMMIT_HASH_OVERRIDE="${2:?--commit-hash requires a value}"; shift 2 ;;
        --commit-hash=*)  COMMIT_HASH_OVERRIDE="${1#*=}"; shift ;;
        --months)         MONTH_FILTER="${2:?--months requires a value}"; shift 2 ;;
        --months=*)       MONTH_FILTER="${1#*=}"; shift ;;
        --series)         SERIES_FILTER="${2:?--series requires a value}"; shift 2 ;;
        --series=*)       SERIES_FILTER="${1#*=}"; shift ;;
        --seeds)          VD_SEEDS_OVERRIDE="${2:?--seeds requires a value}"; shift 2 ;;
        --seeds=*)        VD_SEEDS_OVERRIDE="${1#*=}"; shift ;;
        # Extra Hydra overrides appended to every run's TRAIN_ARGS, e.g.
        # --extra-args skip_if_done=false to retrain ghost-"finished" W&B runs.
        --extra-args)     VD_EXTRA_ARGS="${2:?--extra-args requires a value}"; shift 2 ;;
        --extra-args=*)   VD_EXTRA_ARGS="${1#*=}"; shift ;;
        --) shift; break ;;
        -*) error "Unknown flag: $1" ;;
        *)  break ;;
    esac
done

case "${PARTITION}" in
    standard_hopper|standard_l40s|long_hopper) ;;
    *) error "Invalid --partition: ${PARTITION}" ;;
esac

# ── Sweep file ──────────────────────────────────────────────────────────────

SWEEP_FILE_REL="scripts/sweeps/variance_decomp.sh"
SWEEP_FILE="${LOCAL_REPO}/${SWEEP_FILE_REL}"
[ -f "${SWEEP_FILE}" ] || error "Sweep file not found: ${SWEEP_FILE}"

( source "${SWEEP_FILE}" )
source "${SWEEP_FILE}"
: "${SWEEP_NAME:?Sweep file must define SWEEP_NAME}"
[ ${#SWEEP_VALUES[@]} -gt 0 ] || error "Sweep file must define SWEEP_VALUES"

VD_SERIES="${SWEEP_VALUES[*]}"
if [ -n "${SERIES_FILTER}" ]; then
    IFS=',' read -ra _KEEP <<<"${SERIES_FILTER}"
    NEW_SERIES=()
    for K in "${_KEEP[@]}"; do
        FOUND=0
        for V in "${SWEEP_VALUES[@]}"; do
            [ "${V}" = "${K}" ] && FOUND=1 && break
        done
        [ "${FOUND}" = "1" ] || error "Unknown series '${K}' (valid: ${SWEEP_VALUES[*]})"
        NEW_SERIES+=("${K}")
    done
    VD_SERIES="${NEW_SERIES[*]}"
fi

VD_SEEDS="${VD_SEEDS_OVERRIDE:-${VD_SEEDS_DEFAULT:?Sweep file must define VD_SEEDS_DEFAULT}}"

# ── Months ──────────────────────────────────────────────────────────────────

uv run "${LOCAL_REPO}/scripts/experiments/sample_variance_months.py" --check >/dev/null \
    || error "sample_variance_months.py drifted from its frozen VD_MONTHS list."
mapfile -t VD_MONTHS < <( cd "${LOCAL_REPO}" && uv run scripts/experiments/sample_variance_months.py )
[ ${#VD_MONTHS[@]} -gt 0 ] || error "No months returned by sample_variance_months.py"

if [ -n "${MONTH_FILTER}" ]; then
    IFS=',' read -ra _KEEP <<<"${MONTH_FILTER}"
    NEW_MONTHS=()
    for K in "${_KEEP[@]}"; do
        FOUND=0
        for M in "${VD_MONTHS[@]}"; do
            [ "${M}" = "${K}" ] && FOUND=1 && break
        done
        [ "${FOUND}" = "1" ] || error "Month '${K}' is not one of the frozen VD months: ${VD_MONTHS[*]}"
        NEW_MONTHS+=("${K}")
    done
    VD_MONTHS=("${NEW_MONTHS[@]}")
fi

XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"

# Every training month AND the calendar month after it: the probe fits on the
# train month and reports on the next one, so a job needs both tables.
XS_TABLE_DIR="${BLL01_DATA_DIR}/${XS_STATS_DIR}"
XS_MISSING=()
for M in "${VD_MONTHS[@]}"; do
    for T in "${M}" "$(date -d "${M}-01 +1 month" +%Y-%m)"; do
        [ -f "${XS_TABLE_DIR}/${T}.npz" ] || XS_MISSING+=("${T}")
    done
done
[ ${#XS_MISSING[@]} -eq 0 ] \
    || error "${#XS_MISSING[@]} anchor-stat table(s) missing from ${XS_STATS_DIR}: ${XS_MISSING[*]}"

info "Months (${#VD_MONTHS[@]}): ${VD_MONTHS[*]}"
info "Series: ${VD_SERIES}"
info "Seeds:  ${VD_SEEDS}"
info "Anchor tables: ${XS_STATS_DIR} (all train+eval months present)"

# ── One-time setup & sync ───────────────────────────────────────────────────

pythia_setup
pythia_sync

if [ -n "${COMMIT_HASH_OVERRIDE}" ]; then
    COMMIT_HASH="${COMMIT_HASH_OVERRIDE}"
    info "Using overridden COMMIT_HASH=${COMMIT_HASH}."
else
    COMMIT_HASH=$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)
fi

# ── Submit one bundled job per (month x series) ─────────────────────────────

TRAIN_SPAN_MONTHS="${TRAIN_SPAN_MONTHS:-$(uv run python -c 'from market_jepa.schemas import DatasetConfig; print(DatasetConfig.train_span_months)')}"
[[ "${TRAIN_SPAN_MONTHS}" =~ ^[1-9][0-9]*$ ]] || error "could not resolve DatasetConfig.train_span_months"
info "Training span: ${TRAIN_SPAN_MONTHS} month(s) ending at each month"

# A MONTH WITH NO SPAN BEHIND IT CANNOT RUN, and must be dropped HERE rather
# than discovered by a job. The frozen VD sample starts at 2008-02, whose
# six-month span reaches 2007-09 -- four months before the mosaic begins -- so
# its jobs staged, failed the rsync and died before training (223084-86,
# 2026-09-13). Skipping is the only option that keeps the recipe: a shorter
# span for one month would measure the seed spread of a model nothing else in
# the paper is, which is precisely what this sweep must not do.
DATA_FLOOR=$(ls -1 "${MOSAIC_ROOT:-/data/lab/market-jepa-mosaic/1Hz_mosaic_mnth}" 2>/dev/null \
    | grep -E '^[0-9]{4}$' | sort | head -1)
if [ -n "${DATA_FLOOR}" ]; then
    SPAN_FLOOR=$(date -d "${DATA_FLOOR}-01-01 +$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m)
    KEPT=() DROPPED=()
    for M in "${VD_MONTHS[@]}"; do
        if [[ "${M}" < "${SPAN_FLOOR}" ]]; then DROPPED+=("${M}"); else KEPT+=("${M}"); fi
    done
    if [ ${#DROPPED[@]} -gt 0 ]; then
        info "Dropping ${#DROPPED[@]} month(s) with no ${TRAIN_SPAN_MONTHS}-month span behind them (data starts ${DATA_FLOOR}-01): ${DROPPED[*]}"
        VD_MONTHS=("${KEPT[@]}")
        [ ${#VD_MONTHS[@]} -gt 0 ] || error "every requested month is before ${SPAN_FLOOR}"
    fi
fi

JOB_IDS=()
for YM in "${VD_MONTHS[@]}"; do
    # THE SPAN ENDING AT ${YM}, not the single calendar month: the recipe is
    # DatasetConfig.train_span_months of data before the eval month (settled
    # 2026-09-13). This sweep measures the seed spread OF THE REPORTED MODEL,
    # so its window has to be the reported one; a single month is a different
    # and measurably worse recipe, and its seed spread would answer a question
    # nobody is asking.
    TRAIN_END=$(date -d "${YM}-01 +1 month -1 day" +%Y-%m-%d)
    TRAIN_START=$(date -d "${YM}-01 -$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m-01)
    EVAL_START=$(date -d "${YM}-01 +1 month" +%Y-%m-%d)
    EVAL_END=$(date -d "${EVAL_START} +1 month -1 day" +%Y-%m-%d)

    for SERIES in ${VD_SERIES}; do
        info "Submitting ${YM} / ${SERIES} (train ${TRAIN_START}..${TRAIN_END}, eval ${EVAL_START}..${EVAL_END}) ..."
        SBATCH_OUTPUT=$(ssh "${PYTHIA_HOST}" \
            "cd ${PYTHIA_REPO} && export \
                YM='${YM}' \
                TRAIN_START='${TRAIN_START}' \
                TRAIN_END='${TRAIN_END}' \
                EVAL_START='${EVAL_START}' \
                EVAL_END='${EVAL_END}' \
                VD_SEEDS='${VD_SEEDS}' \
                VD_SERIES='${SERIES}' \
                VD_EXTRA_ARGS='${VD_EXTRA_ARGS}' \
                POST_TRAIN_PROBE='${POST_TRAIN_PROBE:-1}' \
                DAYSTORE='${DAYSTORE:-0}' \
                PANEL_CACHE_REQUIRED='${PANEL_CACHE_REQUIRED:-1}' \
                XS_STATS_DIR='${XS_STATS_DIR}' \
                SWEEP_FILE_REL='${SWEEP_FILE_REL}' \
                COMMIT_HASH='${COMMIT_HASH}' && sbatch \
                    --export=ALL \
                    --job-name=${SWEEP_NAME}-${YM}-${SERIES} \
                    --partition=${PARTITION} \
                    scripts/pythia/specific/slurm_variance_decomp.sh")
        JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
        [ -n "${JOB_ID}" ] || error "sbatch failed for ${YM}/${SERIES}: ${SBATCH_OUTPUT}"
        JOB_IDS+=("${JOB_ID}")
        info "  Submitted job ${JOB_ID} (${YM} / ${SERIES})"
    done
done

echo ""
echo "  Sweep:        ${SWEEP_NAME}"
echo "  Jobs:         ${#JOB_IDS[@]} = ${#VD_MONTHS[@]} months x $(echo ${VD_SERIES} | wc -w) series"
echo "  IDs:          ${JOB_IDS[*]}"
echo "  Runs/job:     $(echo ${VD_SEEDS} | wc -w) seeds"
echo "  wandb:        ${WANDB_ENTITY:-boothai}/${SWEEP_NAME}-${COMMIT_HASH}"
echo "  Targets:      ${XS_STATS_DIR}"
echo "  Monitor:      ssh pythia 'squeue -u \$USER'"
echo "  Cancel all:   ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
