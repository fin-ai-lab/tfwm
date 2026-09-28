#!/bin/bash
# run_all_months_sweep.sh — Fan out one or more sweeps across all 32 sampled
# months, ONE JOB PER MONTH.
#
# Usage:  ./pythia/run_all_months_sweep.sh [--partition standard_hopper|standard_l40s|bll] [--commit-hash <hash>] [--backward] <sweep_file>...
#         ./pythia/run_all_months_sweep.sh sweeps/ijepa_lr.sh
#         ./pythia/run_all_months_sweep.sh --partition bll sweeps/ijepa_lr.sh
#         ./pythia/run_all_months_sweep.sh --commit-hash 383b34 sweeps/ijepa_lr.sh  # resume prior wandb project
#         ./pythia/run_all_months_sweep.sh --backward sweeps/ijepa_lr.sh             # iterate months newest→oldest
#         ./pythia/run_all_months_sweep.sh --first-n-months 2 sweeps/ijepa_lr.sh     # only first 2 months (after --backward is applied)
#         ./pythia/run_all_months_sweep.sh sweeps/ssl_ic/cost_r1.sh sweeps/ssl_ic/ts2vec_r1.sh
#
# Submits ONE bundled slurm job per month (slurm_train_bundle.sh), which runs
# every sweep value of every sweep file sequentially on the same GPU. So 32
# months → 32 jobs, no matter how many sweep values or how many SWEEP FILES
# are given.
#
# SEVERAL SWEEP FILES IS HOW MONTH-MAJOR WORKS, and it is why they go in one
# job rather than one job each: a month's mosaic rsync, shard materialization,
# anchor tables and ~26 GB eval panel are staged NODE-LOCALLY, and only jobs
# that are the same job are guaranteed the same node. The wall clock per job
# becomes the sum of the sweeps — see scripts/submit_month_major.sh.
#
# SINGLE-MONTH RUNS come through here too, with MONTHS_OVERRIDE='<YYYY-MM>'.
# run_sweep.sh + slurm_train.sh were the one-job-per-sweep-value path and were
# retired on 2026-09-13: every sweep file they documented had been deleted, and
# the bundle gives the same thing while sharing a month's staging.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"
source "${PYTHIA_DIR}/lib/sweep_list.sh"

# ── Parse options ───────────────────────────────────────────────────────────

PARTITION=""
COMMIT_HASH_OVERRIDE=""
SKIP_MONTHS=()
BACKWARD=0
FIRST_N_MONTHS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition|-p)
            PARTITION="${2:?--partition requires a value}"
            shift 2 ;;
        --partition=*)
            PARTITION="${1#*=}"
            shift ;;
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

case "${PARTITION}" in
    ""|standard_hopper|standard_l40s|bll) ;;
    *) error "Invalid --partition: ${PARTITION} (expected: standard_hopper, standard_l40s, bll)" ;;
esac

[ $# -gt 0 ] || error "Usage: $0 [--partition standard_hopper|standard_l40s|bll] <sweep_file>..."

# Resolve + validate every sweep file locally (each is sourced in a subshell to
# surface errors early and to pick up its SWEEP_NAME / SBATCH_EXTRA / value
# count). Sets SWEEP_FILES_REL, SWEEP_NAMES, SWEEP_TOTAL, SWEEP_JOB_LABEL and
# folds the sweeps' SBATCH_EXTRA in front of the caller's.
resolve_sweep_list "$@"

# This launcher serves the 32 sampled months and nothing else; a sweep that
# says it belongs elsewhere is refused rather than silently run on them.
require_month_set "sampled32"

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
    # NEWLINES ARE SEPARATORS TOO. `read -ra` consumes ONE LINE, and the
    # command substitution this option documents -- holdout_months.py, which
    # prints one YYYY-MM per line -- therefore resolved to its FIRST month and
    # submitted 1 job where 5 were asked for (2026-09-15, the ts2vec/timemae
    # blr sweep). It reported "bundled across 1 months" and was otherwise
    # silent. The set-but-empty guard above cannot see this: the variable is
    # neither empty nor wrong, just truncated at the first newline.
    read -ra MONTHS <<< "${MONTHS_OVERRIDE//$'\n'/ }"
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
info "Loaded ${#MONTHS[@]} months. ${#SWEEP_NAMES[@]} sweep(s), ${SWEEP_TOTAL} values per month → ${#MONTHS[@]} bundled jobs.  partition=${PARTITION:-<default>}"

_PARTITION_SBATCH=""
[ -n "${PARTITION}" ] && _PARTITION_SBATCH="--partition=${PARTITION}"
SBATCH_EXTRA="${_PARTITION_SBATCH} ${SBATCH_EXTRA}"

# ── Dense-grid cache: OFF by default since the dense mosaic ─────────────────
#
# GRID_CACHE_GB is GiB per dataloader worker of cached 1 Hz grids (see
# DatasetConfig.grid_cache_gb). It existed to amortize the sparse->dense grid
# rebuild across repeat visits to a ticker-day, and that rebuild was 71% of a
# worker's per-sample CPU. THE DATASET NOW STORES THE RECONSTRUCTED SESSION,
# so there is nothing left to amortize: the cache would only take memory from
# the reader, num_workers x this in RSS.
#
#   GRID_CACHE_GB=0   cache off, --mem is the PARTITION's share (below)
#   GRID_CACHE_GB=20  cache on,  --mem=240G — only worth setting against a
#                                             SPARSE dataset. Results are
#                                             IDENTICAL either way; the cache
#                                             is bitwise exact.
#
# THE TWO BRANCHES ARE NOT THE SAME KIND OF NUMBER. With the cache ON,
# GRID_CACHE_GB x num_workers is ANONYMOUS RSS and the request has to cover
# it (86G did not: see the 2026-08-21 OOMs in
# specific/slurm_variance_decomp.sh). With it OFF the resident set is a
# MAPPED day store -- ~7 GB of anon, nearly all /dev/shm, under reclaimable
# page cache -- so --mem is a CACHE CEILING.
#
# A CEILING IS NOT SPARE CAPACITY. It decides how much of the span survives
# between visits, and DayStoreCellDataset permutes the WHOLE span every
# epoch, so a ceiling under the span re-reads nearly all of it every pass.
# 64G against an 87 GB six-month span, six jobs to a node: GPU at 0%, md0 at
# 965 MB/s, 16-35 s/step where the same recipe gets 0.7 (2026-09-15, the
# cancelled variance-decomp wave). Size this against ``du -sh`` on the staged
# day store, never against MaxRSS -- sacct reports the ceiling, so the same
# sweep "peaks" at 15 GB when given 16G and 66 GB when given 86G, and any
# request sized off that number is circular.
#
# SO THE NUMBER IS THE PARTITION'S MEMORY PER GPU, LESS HEADROOM (user,
# 2026-09-15). standard_hopper is 2063255 MB over 8 H100s = 251.9 GiB each,
# so 240G, and 8 x 240G = 1920 GiB leaves ~95 GiB for the OS. Billing is
# MAX_TRES and one GPU already bills 1000, so that costs no priority.
# standard_l40s nodes have 773 GB over 8 cards = 96 GiB each, so 64G there
# and a wider span wants the hopper partition rather than a bigger request.
# PARTITION unset means the cluster default, which is an l40s queue.
# An explicit --mem in SBATCH_EXTRA always wins.
: "${GRID_CACHE_GB:=0}"
case "${PARTITION}" in
    *hopper*) _MEM_CACHE_OFF="240G" ;;
    *)        _MEM_CACHE_OFF="64G"  ;;
esac
case "${SBATCH_EXTRA}" in
    *--mem=*) : ;;
    *) if [ "${GRID_CACHE_GB%%.*}" = "0" ]; then
           SBATCH_EXTRA="--mem=${_MEM_CACHE_OFF} ${SBATCH_EXTRA}"
       else
           SBATCH_EXTRA="--mem=240G ${SBATCH_EXTRA}"
       fi ;;
esac
info "Grid cache: GRID_CACHE_GB=${GRID_CACHE_GB} (0 = off)"

# ── One-time setup & sync ───────────────────────────────────────────────────

pythia_setup
pythia_sync

if [ -n "${COMMIT_HASH_OVERRIDE}" ]; then
    COMMIT_HASH="${COMMIT_HASH_OVERRIDE}"
    info "Using overridden COMMIT_HASH=${COMMIT_HASH} (wandb project suffix will match prior runs)."
else
    COMMIT_HASH=$(git -C "${LOCAL_REPO}" rev-parse --short=6 HEAD)
fi

# ── Per-month bundled submission ────────────────────────────────────────────

last_day_of_month() { date -d "${1}-01 +1 month -1 day" +%Y-%m-%d; }
next_month() { date -d "${1}-01 +1 month" +%Y-%m; }

# THE SPAN, AND THE MONTHS THAT HAVE ONE. A run trains the
# DatasetConfig.train_span_months ending at its month (settled 2026-09-13), so
# TRAIN_START moves back with the span and a month whose span predates the data
# cannot run at all -- its staging fails and the job dies before training. Drop
# those here, naming them, rather than leaving a job to discover it.
TRAIN_SPAN_MONTHS="${TRAIN_SPAN_MONTHS:-$(cd "${LOCAL_REPO}" && uv run python -c 'from market_jepa.schemas import DatasetConfig; print(DatasetConfig.train_span_months)')}"
[[ "${TRAIN_SPAN_MONTHS}" =~ ^[1-9][0-9]*$ ]] || error "could not resolve DatasetConfig.train_span_months"
DATA_FLOOR=$(ls -1 "${BLL01_DATA_DIR}/1Hz_mosaic_mnth" 2>/dev/null | grep -E '^[0-9]{4}$' | sort | head -1)
if [ -n "${DATA_FLOOR}" ]; then
    SPAN_FLOOR=$(date -d "${DATA_FLOOR}-01-01 +$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m)
    KEPT=() DROPPED=()
    for M in "${MONTHS[@]}"; do
        if [[ "${M}" < "${SPAN_FLOOR}" ]]; then DROPPED+=("${M}"); else KEPT+=("${M}"); fi
    done
    if [ ${#DROPPED[@]} -gt 0 ]; then
        info "Dropping ${#DROPPED[@]} month(s) with no ${TRAIN_SPAN_MONTHS}-month span behind them (data starts ${DATA_FLOOR}-01): ${DROPPED[*]}"
        MONTHS=("${KEPT[@]}")
        [ ${#MONTHS[@]} -gt 0 ] || error "every month is before ${SPAN_FLOOR}"
    fi
fi
info "Training span: ${TRAIN_SPAN_MONTHS} month(s) ending at each month."

JOB_IDS=()
for YM in "${MONTHS[@]}"; do
    for SKIP in "${SKIP_MONTHS[@]}"; do
        if [ "${YM}" = "${SKIP}" ]; then
            info "Skipping ${YM} (per --skip-month)"
            continue 2
        fi
    done
    TRAIN_END="$(last_day_of_month "${YM}")"
    TRAIN_START=$(date -d "${YM}-01 -$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m-01)
    EVAL_YM="$(next_month "${YM}")"
    EVAL_START="${EVAL_YM}-01"
    EVAL_END="$(last_day_of_month "${EVAL_YM}")"

    info "Submitting bundled job for ${YM} (train ${TRAIN_START}..${TRAIN_END}, eval ${EVAL_START}..${EVAL_END}) ..."

    SBATCH_OUTPUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export GRID_CACHE_GB='${GRID_CACHE_GB}' SWEEP_FILES_REL='${SWEEP_FILES_REL}' TRAIN_START='${TRAIN_START}' TRAIN_END='${TRAIN_END}' EVAL_START='${EVAL_START}' EVAL_END='${EVAL_END}' COMMIT_HASH='${COMMIT_HASH}' STAGE_ONLY='${STAGE_ONLY:-0}' POST_TRAIN_IC_EVAL='${POST_TRAIN_IC_EVAL:-1}' SWEEP_VALUES_OVERRIDE='${SWEEP_VALUES_OVERRIDE:-}' DAYSTORE='${DAYSTORE:-0}' POST_TRAIN_PROBE='${POST_TRAIN_PROBE:-1}' PANEL_CACHE_REQUIRED='${PANEL_CACHE_REQUIRED:-1}' FT_BLR='${FT_BLR:-}' SWEEP_TASKS='${SWEEP_TASKS:-}' SWEEP_TARGET_ROWS='${SWEEP_TARGET_ROWS:-}' PROBE_TASK='${PROBE_TASK:-}' PROBE_SAVE_STEPS='${PROBE_SAVE_STEPS:-}' EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}' FT_ROWS_PER_OBS='${FT_ROWS_PER_OBS:-}' SCALES='${SCALES:-}' SCALING_LADDER_STEPS='${SCALING_LADDER_STEPS:-}' SCALING_FLOPS_TARGETS='${SCALING_FLOPS_TARGETS:-}' SCALING_ANNEAL_FRAC='${SCALING_ANNEAL_FRAC:-}' SCALING_VIEWS_PER_STEP='${SCALING_VIEWS_PER_STEP:-}' SCALING_WARMUP_STEPS='${SCALING_WARMUP_STEPS:-}' SCALING_BLR='${SCALING_BLR:-}' SCALING_LRS='${SCALING_LRS:-}' && sbatch \
            --export=ALL \
            --job-name=${SWEEP_JOB_LABEL}-${YM} \
            ${SBATCH_EXTRA} \
            scripts/pythia/slurm_train_bundle.sh")
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${YM}: ${SBATCH_OUTPUT}"

    JOB_IDS+=("${JOB_ID}")
    info "  Submitted job ${JOB_ID} (${YM})"
done

echo ""
echo "  Sweeps: ${SWEEP_NAMES[*]}  (bundled across ${#MONTHS[@]} months, one job per month)"
echo "  Jobs:  ${JOB_IDS[*]}"
echo "  Monitor:    ssh pythia 'squeue -u \$USER'"
echo "  Cancel all: ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
