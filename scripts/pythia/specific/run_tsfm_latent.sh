#!/bin/bash
# run_tsfm_latent.sh — the LATENT half of the frozen-TSFM layer sweep, one job
# per eval month. The IC half is specific/run_tsfm_layers.sh.
#
# TWO PANELS, AND THEY ARE NOT INTERCHANGEABLE (same rule as the IC half):
#
#   MONTHS_FROM=holdout          the 5-month OPTIMIZATION set's eval months.
#       Where a layer gets CHOSEN. Already done serially on bll01 by
#       plots/tsfm_layers/run_latent.sh; this is the parallel equivalent.
#   MONTHS_FROM=sweep32 (default) the 32 REPORTED months' eval months. Where a
#       chosen layer gets its number. An argmax read off this panel is
#       selection on the test set.
#
# Give them separate TSFM_RESULT_DEST directories.
#
# ONE JOB PER MONTH. Every latent statistic is pooled ACROSS months at the very
# end and never within one, so splitting by month is arithmetically free --
# merge_latent.py reassembles the per-month jsons and redoes the pooling the
# way the serial run does it. ~1.5 h per month, so 32 months is ~49 h serial
# and ~1.5 h wall when they run together.
#
# Usage:
#   ./scripts/pythia/specific/run_tsfm_latent.sh
#   ./scripts/pythia/specific/run_tsfm_latent.sh 2010-01 2013-04
#   MONTHS_FROM=holdout TSFM_RESULT_DEST=/data/lab/tsfm_latent_holdout \
#       ./scripts/pythia/specific/run_tsfm_latent.sh
#
# Args are EVAL months (YYYY-MM) — unlike the IC half, which takes TRAINING
# months and derives the eval month. Results land on bll01 at
# ${TSFM_RESULT_DEST}/<analysis>_tsfml_<eval month>.json.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

PARTITION="${PARTITION:-standard_l40s}"
FAMILIES="${FAMILIES:-chronos2 timesfm sundial kronos}"
# ~1.5 h per month measured serially (phase 1 is ~64% of it). 8 h is 5x the
# worst case and short enough to backfill; do NOT raise it to a day, since
# Slurm plans around the DECLARED limit and a 24 h wall on a 2 h job pushes
# every backfill start a day out.
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
TSFM_RESULT_DEST="${TSFM_RESULT_DEST:-/data/lab/tsfm_latent_32}"
MONTHS_FROM="${MONTHS_FROM:-sweep32}"
DRY_RUN="${DRY_RUN:-0}"

# Both halves are defined by TRAINING months; the latent analyses run on the
# EVAL month, so map train -> train+1 exactly as the IC half does.
next_month() { date -d "${1}-01 +1 month" +%Y-%m; }

case "${MONTHS_FROM}" in
    holdout) SRC="holdout_months.py" ;;
    sweep32) SRC="sample_sweep_months.py" ;;
    *) error "MONTHS_FROM must be holdout or sweep32, got '${MONTHS_FROM}'" ;;
esac

if [ $# -gt 0 ]; then
    MONTHS=( "$@" )
else
    MONTHS=()
    for TM in $( cd "${LOCAL_REPO}" && uv run "scripts/experiments/${SRC}" ); do
        MONTHS+=( "$(next_month "${TM}")" )
    done
fi
for M in "${MONTHS[@]}"; do
    [[ "${M}" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "bad eval month: ${M}"
done
for F in ${FAMILIES}; do
    case "${F}" in
        chronos2|timesfm|timesfm3|sundial|kronos) ;;
        sup_return|sup_vol|sup_spread) ;;
        *) error "unknown family: ${F}" ;;
    esac
done

N_JOBS=${#MONTHS[@]}
info "families=[${FAMILIES}]  eval months=[${MONTHS[*]}]"
info "panel=${MONTHS_FROM}  ->  ${N_JOBS} jobs  ->  ${TSFM_RESULT_DEST}"
[ "${N_JOBS}" -le 180 ] || error "${N_JOBS} jobs is too close to the 200-job QOS cap"

if [ "${DRY_RUN}" = 1 ]; then
    for YM in "${MONTHS[@]}"; do echo "  ${YM}   [${FAMILIES}]"; done
    exit 0
fi

# PREFLIGHT. pythia_sync rsyncs the WORKING TREE, not a commit, so a half-edited
# file ships exactly as it is; on 2026-08-20 that cost 44 IC jobs. Import every
# entry point this job runs before syncing.
info "preflight: do the entry points import?"
( cd "${LOCAL_REPO}" && uv run python -c "
import sys
sys.path.insert(0, 'plots/latent_eval/fixed_panel')
sys.path.insert(0, 'plots/latent_eval/factors')
import fixed_panel_metrics, build_fullday_embs, build_return_panel
import estimate_factors, decode_loadings, subspace_alignment
from industry_nn_sweep import TSFM_KEYS, SUP_KEYS
assert sum(len(v) for f, v in TSFM_KEYS.items() if not f.endswith('_cmean')) == 81, TSFM_KEYS.keys()  # 5 concat families; *_cmean are the channel-averaged twins
assert sum(len(v) for v in SUP_KEYS.values()) == 36, SUP_KEYS.keys()
print('imports ok')
" ) || error "latent entry points do not import — refusing to sync a broken tree"

pythia_setup
pythia_sync

# POST-SYNC, ON THE REMOTE. The local preflight above proves the code RUNS; it
# cannot prove the code ARRIVED, because it runs where plots/ obviously exists.
# pythia_sync excluded plots/ wholesale until 2026-08-20 and all 32 jobs of the
# first latent submission died on "No module named 'industry_nn_sweep'" — a
# local import check passed cleanly while every compute node was missing the
# entire pipeline. Verify on the far side, where the job will actually run.
info "preflight: did the entry points arrive on ${PYTHIA_HOST}?"
REMOTE_FILES="plots/style.py \
plots/latent_eval/fixed_panel/fixed_panel_metrics.py \
plots/latent_eval/fixed_panel/industry_nn_sweep.py \
plots/latent_eval/fixed_panel/panel_lib.py \
plots/latent_eval/factors/build_fullday_embs.py \
plots/latent_eval/factors/build_return_panel.py \
plots/latent_eval/factors/estimate_factors.py \
plots/latent_eval/factors/decode_loadings.py \
plots/latent_eval/factors/subspace_alignment.py \
backtesting/build_cache.py \
data/industry_map.parquet"
MISSING=$(ssh "${PYTHIA_HOST}" "cd ${PYTHIA_REPO} && for f in ${REMOTE_FILES}; do [ -f \"\$f\" ] || echo \"\$f\"; done")
[ -z "${MISSING}" ] || error "these did not reach ${PYTHIA_HOST} (check pythia_sync excludes):
${MISSING}"
info "  all entry points present on ${PYTHIA_HOST}"

# Supervised run ids resolved HERE, where /data/lab is mounted. The latent
# analyses run on an EVAL month, and its encoder is the one trained the month
# BEFORE -- the same pairing the IC sweep uses.
sup_runs_for_eval() {
    ( cd "${LOCAL_REPO}" && uv run python -c "
import sys
sys.path.insert(0, 'scripts/eval')
from market_jepa.eval.checkpoints import sup_run_dir, SUP_PROJECTS, prev_month
fams = [f for f in '${FAMILIES}'.split() if f in SUP_PROJECTS]
tm = prev_month('$1')
print(' '.join(f'{SUP_PROJECTS[f]}/{sup_run_dir(f, tm).name}' for f in fams))
" )
}

JOB_IDS=()
for YM in "${MONTHS[@]}"; do
    SUP_RUNS=""
    case "${FAMILIES}" in
        *sup_*) SUP_RUNS="$(sup_runs_for_eval "${YM}")"
                [ -n "${SUP_RUNS}" ] || error "no supervised runs for eval ${YM}" ;;
    esac
    OUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export \
            TSFM_FAMILIES='${FAMILIES}' \
            TSFM_SERIES_FAMILIES='${SERIES_FAMILIES:-${FAMILIES}}' \
            TSFM_SUP_RUNS='${SUP_RUNS}' \
            TSFM_EVAL_MONTH='${YM}' \
            TSFM_RESULT_DEST='${TSFM_RESULT_DEST}' && sbatch \
                --export=ALL \
                --job-name=tsfmlat-${YM} \
                --partition=${PARTITION} \
                --time=${TIME_LIMIT} \
                ${SBATCH_EXTRA:-} \
                scripts/pythia/slurm_tsfm_latent.sh")
    JOB_ID=$(echo "${OUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${YM}: ${OUT}"
    JOB_IDS+=("${JOB_ID}")
    info "  ${YM}: job ${JOB_ID}"
done

echo ""
echo "  Sweep:      tsfm-latent"
echo "  Jobs:       ${#JOB_IDS[@]}  (${JOB_IDS[*]})"
echo "  Results:    ${TSFM_RESULT_DEST}/<analysis>_tsfml_<eval month>.json"
echo "  Merge:      uv run python plots/tsfm_layers/merge_latent.py \\"
echo "                  --results-dir ${TSFM_RESULT_DEST}"
echo "  Cancel all: ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
