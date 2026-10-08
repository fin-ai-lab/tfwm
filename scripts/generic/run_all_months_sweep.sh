#!/bin/bash
# run_all_months_sweep.sh — Fan out one or more sweeps across all 32 sampled
# months on a generic cloud cluster, ONE JOB PER MONTH.
#
# Usage:  ./scripts/generic/run_all_months_sweep.sh <variant> [flags] <sweep_file>...
#         ./scripts/generic/run_all_months_sweep.sh h100 sweeps/lejepa_lamb.sh
#         ./scripts/generic/run_all_months_sweep.sh h100 --backward sweeps/foo.sh
#         ./scripts/generic/run_all_months_sweep.sh 40 --commit-hash cf3df6 sweeps/mae_patch_mask.sh
#         ./scripts/generic/run_all_months_sweep.sh 80 sweeps/ssl_ic/cost_r1.sh sweeps/ssl_ic/ts2vec_r1.sh
#         STAGE_ONLY=1 ./scripts/generic/run_all_months_sweep.sh h100 sweeps/foo.sh   # pre-stage data only
#
# First arg selects the cluster variant ('h100', '40', '80'); the matching
# scripts/generic/.env.<variant> is sourced for host + sbatch resources.
#
# Submits ONE bundled slurm job per month (slurm_train_bundle.sh), which runs
# every sweep value of every sweep file sequentially on the same GPU. So 32
# months → 32 jobs, no matter how many sweep values or how many SWEEP FILES
# are given.
#
# SEVERAL SWEEP FILES IS HOW MONTH-MAJOR WORKS, and it is why they go in one
# job rather than one job each: a month's mosaic rsync, shard materialization,
# anchor tables and ~26 GB eval panel are staged NODE-LOCALLY, and only jobs
# that are the same job are guaranteed the same node — which on these boxes
# also means the same GPU, so a grouped job's wall clock is the sum of the
# sweeps. See scripts/submit_month_major.sh.

set -euo pipefail

CLUSTER_VARIANT="${1:?Usage: $0 <variant> [--commit-hash <hash>] [--backward] [--skip-month YYYY-MM] [--first-n-months N] <sweep_file>...}"
export CLUSTER_VARIANT
shift

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
source "${CLUSTER_DIR}/lib/setup.sh"
source "${CLUSTER_DIR}/lib/sync.sh"
source "${CLUSTER_DIR}/lib/sweep_list.sh"

# ── Parse options ───────────────────────────────────────────────────────────

COMMIT_HASH_OVERRIDE=""
SKIP_MONTHS=()
BACKWARD=0
FIRST_N_MONTHS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit-hash|-c)
            COMMIT_HASH_OVERRIDE="${2:?--commit-hash requires a value}"
            shift 2 ;;
        --commit-hash=*)
            COMMIT_HASH_OVERRIDE="${1#*=}"
            shift ;;
        --skip-month)
            SKIP_MONTHS+=("${2:?--skip-month requires a YYYY-MM value}")
            shift 2 ;;
        --skip-month=*)
            SKIP_MONTHS+=("${1#*=}")
            shift ;;
        --backward)
            BACKWARD=1
            shift ;;
        --first-n-months)
            FIRST_N_MONTHS="${2:?--first-n-months requires a positive integer}"
            shift 2 ;;
        --first-n-months=*)
            FIRST_N_MONTHS="${1#*=}"
            shift ;;
        --) shift; break ;;
        -*) error "Unknown flag: $1" ;;
        *)  break ;;
    esac
done

[ $# -gt 0 ] || error "Usage: $0 <variant> [flags] <sweep_file>..."

# Resolve + validate every sweep file locally (each is sourced in a subshell to
# surface errors early and to pick up its SWEEP_NAME / SBATCH_EXTRA / value
# count). Sets SWEEP_FILES_REL, SWEEP_NAMES, SWEEP_TOTAL, SWEEP_JOB_LABEL and
# folds the sweeps' SBATCH_EXTRA in front of the caller's.
resolve_sweep_list "$@"

# ── Load months (drift-checked) ─────────────────────────────────────────────

MONTHS_SCRIPT="${LOCAL_REPO}/scripts/experiments/sample_sweep_months.py"
[ -f "${MONTHS_SCRIPT}" ] || error "Months script not found: ${MONTHS_SCRIPT}"

# SET-BUT-EMPTY IS AN ERROR, NOT A FALLBACK. The override is nearly always a
# command substitution -- `MONTHS_OVERRIDE="$(... holdout_months.py --set 2)"`
# -- and when that command fails it expands to "", which under a plain -n test
# silently means "use the reported 32". On 2026-09-09 that submitted an
# off-panel HPO sweep onto the panel the paper reports, 32 jobs, caught only
# because the traceback happened to be visible. Refuse instead.
if [ "${MONTHS_OVERRIDE+set}" = "set" ] && [ -z "${MONTHS_OVERRIDE// /}" ]; then
    error "MONTHS_OVERRIDE is set but empty -- the command that produced it failed. Refusing to fall back to the sampled months."
fi

if [ -n "${MONTHS_OVERRIDE:-}" ]; then
    info "Using MONTHS_OVERRIDE='${MONTHS_OVERRIDE}'"
    read -ra MONTHS <<< "${MONTHS_OVERRIDE}"
else
    info "Verifying sampled months have not drifted ..."
    ( cd "${LOCAL_REPO}" && uv run "${MONTHS_SCRIPT}" --check )
    mapfile -t MONTHS < <( cd "${LOCAL_REPO}" && uv run "${MONTHS_SCRIPT}" )
fi
if [ "${BACKWARD}" -eq 1 ]; then
    REVERSED=()
    for (( i=${#MONTHS[@]}-1; i>=0; i-- )); do REVERSED+=("${MONTHS[i]}"); done
    MONTHS=("${REVERSED[@]}")
    info "Iterating months in BACKWARD order."
fi
if [ -n "${FIRST_N_MONTHS}" ]; then
    [[ "${FIRST_N_MONTHS}" =~ ^[1-9][0-9]*$ ]] || error "--first-n-months must be a positive integer, got: ${FIRST_N_MONTHS}"
    (( FIRST_N_MONTHS <= ${#MONTHS[@]} )) || error "--first-n-months=${FIRST_N_MONTHS} exceeds available months (${#MONTHS[@]})"
    MONTHS=("${MONTHS[@]:0:${FIRST_N_MONTHS}}")
    info "Limiting to first ${FIRST_N_MONTHS} months: ${MONTHS[*]}"
fi
info "Loaded ${#MONTHS[@]} months. ${#SWEEP_NAMES[@]} sweep(s), ${SWEEP_TOTAL} values per month → ${#MONTHS[@]} bundled jobs."

# GRID_CACHE_GB comes from the variant .env (sized to the box's RAM), a
# caller's env override wins. Unlike our cluster there is no --mem juggling here:
# SBATCH_MEM in the .env is already budgeted for the cache.
info "Grid cache: GRID_CACHE_GB=${GRID_CACHE_GB:-0} GiB/worker (0 = off)"

# ── One-time setup & sync ───────────────────────────────────────────────────

cluster_setup
cluster_sync

if [ -n "${COMMIT_HASH_OVERRIDE}" ]; then
    COMMIT_HASH="${COMMIT_HASH_OVERRIDE}"
    info "Using overridden COMMIT_HASH=${COMMIT_HASH} (wandb project suffix will match prior runs)."
else
    COMMIT_HASH=$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)
fi

# ── Per-month bundled submission ────────────────────────────────────────────

last_day_of_month() { date -d "${1}-01 +1 month -1 day" +%Y-%m-%d; }
next_month() { date -d "${1}-01 +1 month" +%Y-%m; }

JOB_IDS=()
for YM in "${MONTHS[@]}"; do
    for SKIP in "${SKIP_MONTHS[@]:-}"; do
        if [ "${YM}" = "${SKIP}" ]; then
            info "Skipping ${YM} (per --skip-month)"
            continue 2
        fi
    done
    TRAIN_START="${YM}-01"
    TRAIN_END="$(last_day_of_month "${YM}")"
    EVAL_YM="$(next_month "${YM}")"
    EVAL_START="${EVAL_YM}-01"
    EVAL_END="$(last_day_of_month "${EVAL_YM}")"

    info "Submitting bundled job for ${YM} (train ${TRAIN_START}..${TRAIN_END}, eval ${EVAL_START}..${EVAL_END}) ..."

    SBATCH_OUTPUT=$(ssh "${CLUSTER_HOST}" \
        "cd ${CLUSTER_REPO} && export CLUSTER_VARIANT='${CLUSTER_VARIANT}' SWEEP_FILES_REL='${SWEEP_FILES_REL}' TRAIN_START='${TRAIN_START}' TRAIN_END='${TRAIN_END}' EVAL_START='${EVAL_START}' EVAL_END='${EVAL_END}' COMMIT_HASH='${COMMIT_HASH}' GRID_CACHE_GB='${GRID_CACHE_GB:-}' XS_STATS_DIR='${XS_STATS_DIR:-}' STAGE_ONLY='${STAGE_ONLY:-0}' POST_TRAIN_IC_EVAL='${POST_TRAIN_IC_EVAL:-1}' SWEEP_VALUES_OVERRIDE='${SWEEP_VALUES_OVERRIDE:-}' DAYSTORE='${DAYSTORE:-}' POST_TRAIN_PROBE='${POST_TRAIN_PROBE:-}' PANEL_CACHE_REQUIRED='${PANEL_CACHE_REQUIRED:-}' FT_BLR='${FT_BLR:-}' SWEEP_TASKS='${SWEEP_TASKS:-}' SWEEP_TARGET_ROWS='${SWEEP_TARGET_ROWS:-}' PROBE_TASK='${PROBE_TASK:-}' PROBE_SAVE_STEPS='${PROBE_SAVE_STEPS:-}' EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}' FT_ROWS_PER_OBS='${FT_ROWS_PER_OBS:-}' SCALES='${SCALES:-}' SCALING_LADDER_STEPS='${SCALING_LADDER_STEPS:-}' SCALING_FLOPS_TARGETS='${SCALING_FLOPS_TARGETS:-}' SCALING_ANNEAL_FRAC='${SCALING_ANNEAL_FRAC:-}' SCALING_VIEWS_PER_STEP='${SCALING_VIEWS_PER_STEP:-}' SCALING_WARMUP_STEPS='${SCALING_WARMUP_STEPS:-}' SCALING_BLR='${SCALING_BLR:-}' SCALING_LRS='${SCALING_LRS:-}' && sbatch \
            --export=ALL \
            --job-name=${SWEEP_JOB_LABEL}-${YM} \
            --cpus-per-task=${SBATCH_CPUS_PER_TASK} \
            --mem=${SBATCH_MEM} \
            ${SBATCH_EXTRA} \
            scripts/generic/slurm_train_bundle.sh")
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${YM}: ${SBATCH_OUTPUT}"

    JOB_IDS+=("${JOB_ID}")
    info "  Submitted job ${JOB_ID} (${YM})"
done

echo ""
echo "  Sweeps: ${SWEEP_NAMES[*]}  (bundled across ${#MONTHS[@]} months, one job per month)"
echo "  Jobs:  ${JOB_IDS[*]}"
echo "  Monitor:    ssh ${CLUSTER_HOST} 'squeue -u \$USER'"
echo "  Cancel all: ssh ${CLUSTER_HOST} 'scancel ${JOB_IDS[*]}'"
echo ""
