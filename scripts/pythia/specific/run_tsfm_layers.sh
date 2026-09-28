#!/bin/bash
# run_tsfm_layers.sh — the frozen-TSFM layer sweep, one job per (family, month).
#
# Every hidden state of every family, scored on the synchronized rank-IC panel
# at h=900 for all three targets.
#
# TWO PANELS, AND THEY ARE NOT INTERCHANGEABLE:
#
#   MONTHS_FROM=holdout (default)  the 5-month OPTIMIZATION set. Where a layer
#       and a ridge alpha get CHOSEN — the panel is disjoint from the reported
#       32 in both the training and the eval role, so a hyper-parameter picked
#       here is not picked on the panel the paper scores.
#   MONTHS_FROM=sweep32            the 32 REPORTED months. Where a chosen layer
#       gets its number. Reading an argmax off this panel is selecting on the
#       test set; the profile is a result, the argmax is not a choice.
#
# Give them separate TSFM_RESULT_DEST directories. A reducer pointed at a
# directory holding both would silently pool 37 month pairs of two different
# kinds.
#
# ONE JOB PER MONTH PAIR, RUNNING EVERY FAMILY. Two things are free and one is
# not. Layers are free: compute_features_multi captures every depth from one
# forward, so a 21-layer TimesFM sweep costs what layer 20 alone would. The
# panel DECODE is not free but it is SHARED — MDS read, dense grid, crop,
# normalize is a quarter of a Chronos-2 pass and nearly half a Kronos one, and
# byte-identical for every family. Grouping the families into one process pays
# it once instead of four times: ~29 of the ~94 GPU-hours on the 32-month
# panel, and each month gets staged once rather than four times.
#
# Total wall clock is unchanged either way (the GPU work is fixed and the
# constraint is how many slots the queue gives you), so the grouping is pure
# saving. 5 pairs = 5 jobs of ~3 h; 32 pairs = 32 jobs of ~2 h.
#
# Usage:
#   ./scripts/pythia/specific/run_tsfm_layers.sh
#   ./scripts/pythia/specific/run_tsfm_layers.sh 2009-01 2011-03
#   FAMILIES="chronos2 timesfm" ./scripts/pythia/specific/run_tsfm_layers.sh
#   TSFM_EXTRA="--alphas 10 100 1000" ./scripts/pythia/specific/run_tsfm_layers.sh
#
#   # TimesFM 3.0 — a family of its own job, not a fifth passenger. See the
#   # FAMILIES default below for why, and --tsfm-batch-size for the 288.
#   FAMILIES=timesfm3 TSFM_EXTRA="--tsfm-batch-size 288" \
#       ./scripts/pythia/specific/run_tsfm_layers.sh
#
#   MONTHS_FROM=sweep32 TSFM_RESULT_DEST=/data/lab/tsfm_layer_ic_32 \
#       ./scripts/pythia/specific/run_tsfm_layers.sh
#
#   XS_STATS_DIR=xs_anchor_stats_fwdvwap60 TSFM_EXTRA="--alphas 10" \
#       ./scripts/pythia/specific/run_tsfm_layers.sh
#
#   # the mean-pool readout instead of the 9x concat (optimization set only)
#   FAMILIES="chronos2 timesfm sundial" TSFM_EXTRA="--channel-pool mean" \
#       TSFM_RESULT_DEST=/data/lab/tsfm_layer_ic_meanpool \
#       ./scripts/pythia/specific/run_tsfm_layers.sh
#
# Args are TRAINING months (YYYY-MM); each job evaluates on the next month.
# Results land on bll01 at ${TSFM_RESULT_DEST}/<family>_<train month>.json.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

PARTITION="${PARTITION:-standard_l40s}"
# THE DEFAULT IS THE FOUR ALREADY-REPORTED FAMILIES, NOT EVERY FAMILY.
# timesfm3 (added 2026-09-02) is the fifth and it is opted into by name, for
# two reasons that both point the same way. Its concat width is TimesFM 2.5's
# 11,520, so it brings another 21 x 1.06 GB of float64 Grams: a five-family job
# holds ~53 GB of accumulators against the four-family ~31 GB the 96 GB
# --mem line was sized for. And it decodes at roughly 0.6x TimesFM 2.5's rate
# (measured on an A40: 50 vs 85 views/s with every layer captured), which on
# the largest month pair pushes a combined job from ~5.5 h toward the 12 h
# wall. Alone it is a ~3 h job with the memory of a TimesFM run. Running it
# separately also leaves the reported four reproducible by the plain default.
FAMILIES="${FAMILIES:-chronos2 timesfm sundial kronos}"
# The panel grows with the calendar: a 2009 fit month streams 166k views, a 2022
# one 642k, and at the 4-family rate of ~48 views/s the largest pair is ~5.5 h.
# 12 h is 2x the worst case. A day of headroom is NOT free -- Slurm plans around
# the DECLARED limit, so a 24 h wall on a 2 h job tells the scheduler the node
# is busy until tomorrow and pushes every backfill start a day out.
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
TSFM_EXTRA="${TSFM_EXTRA:-}"
# Repo-relative path to the frozen alpha table (data/..., since
# pythia_sync excludes plots/). Set it and the ladder is NOT swept on this
# panel — each layer solves only the alpha chosen for it on the optimization
# set. Leave it empty to sweep, which is only correct off-panel.
TSFM_ALPHA_TABLE="${TSFM_ALPHA_TABLE:-}"
# Where the result jsons land on bll01. Override to keep a different month
# panel out of the optimization set's directory — the two are NOT
# interchangeable and a reducer pointed at a mixed directory would pool them.
TSFM_RESULT_DEST="${TSFM_RESULT_DEST:-/data/lab/tsfm_layer_ic}"
# MONTHS_FROM=holdout (default) | sweep32 — which month panel to sweep.
MONTHS_FROM="${MONTHS_FROM:-holdout}"
# WHICH ANCHOR-STAT TABLES, WHICH IS TO SAY WHICH TARGET. The tables hold the
# (mu, sigma) OF a specific return/spread_change/volatility_change definition,
# so this selects the metric the sweep measures, not where a file lives. All
# three were redefined on 2026-08-22 (fb362fc, 17160d1) and the new tables live
# in xs_anchor_stats_fwdvwap60; the default is still the midpoint-target set so
# an old rerun reproduces. Checked below against the months this run needs.
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
DRY_RUN="${DRY_RUN:-0}"

case "${MONTHS_FROM}" in
    holdout) DEFAULT_MONTHS="$( cd "${LOCAL_REPO}" && uv run scripts/experiments/holdout_months.py | tr "\n" " " )" ;;
    sweep32) DEFAULT_MONTHS="$( cd "${LOCAL_REPO}" && uv run scripts/experiments/sample_sweep_months.py | tr "\n" " " )" ;;
    *) error "MONTHS_FROM must be holdout or sweep32, got '${MONTHS_FROM}'" ;;
esac
if [ $# -gt 0 ]; then MONTHS=( "$@" ); else read -ra MONTHS <<< "${DEFAULT_MONTHS}"; fi
for M in "${MONTHS[@]}"; do
    [[ "${M}" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "bad training month: ${M}"
done
for F in ${FAMILIES}; do
    case "${F}" in
        chronos2|timesfm|timesfm3|sundial|kronos) ;;
        # The full-history supervised specialists: one trained ViT per task per
        # month, swept for depth exactly as the frozen TSFMs are.
        sup_return|sup_vol|sup_spread) ;;
        *) error "unknown family: ${F}" ;;
    esac
done

# CHANNEL POOL IS PART OF THE ARM, AND THE FILENAME DOES NOT SAY SO. Results
# are written as <family>_<train month>.json with no pooling in the name, so a
# `--channel-pool mean` run aimed at the concat directory overwrites the
# reported numbers with a different measurement. Same mixed-directory hazard as
# the two month panels, one step more silent.
case "${TSFM_EXTRA}" in
    *--channel-pool*mean*)
        [ "${TSFM_RESULT_DEST}" != "/data/lab/tsfm_layer_ic" ] \
            && [ "${TSFM_RESULT_DEST}" != "/data/lab/tsfm_layer_ic_32" ] \
            || error "channel-pool=mean writing to the concat directory ${TSFM_RESULT_DEST}; set TSFM_RESULT_DEST elsewhere"
        case "${FAMILIES}" in
            *kronos*|*sup_*) error "kronos / sup_* have no per-channel axis to pool; drop them from FAMILIES" ;;
        esac
        ;;
esac

# THE READOUT TOKEN IS PART OF THE ARM, AND THE FILENAME DOES NOT SAY SO
# EITHER. Everything under the two reported directories was produced at
# time_pool=mean, before --time-pool existed. A `--time-pool last` run aimed at
# one of them overwrites the reported numbers with a measurement taken at a
# different token -- the same hazard as channel-pool, and the payload's
# time_pool field only helps AFTER the overwrite.
case "${TSFM_EXTRA}" in
    *--time-pool*last*|*--time-pool*reg*)
        [ "${TSFM_RESULT_DEST}" != "/data/lab/tsfm_layer_ic" ] \
            && [ "${TSFM_RESULT_DEST}" != "/data/lab/tsfm_layer_ic_32" ] \
            || error "--time-pool writing to the mean-pool directory ${TSFM_RESULT_DEST}; set TSFM_RESULT_DEST elsewhere"
        case "${FAMILIES}" in
            *sup_*) error "sup_* load at the pool they trained with (already PREDICT_POOL); drop them from FAMILIES" ;;
        esac
        ;;
esac

N_JOBS=${#MONTHS[@]}
info "families=[${FAMILIES}]  months=[${MONTHS[*]}]"
info "panel=${MONTHS_FROM}  ->  ${N_JOBS} jobs  ->  ${TSFM_RESULT_DEST}"
info "anchor tables=${XS_STATS_DIR}"
# QOSMaxSubmitJobPerUserLimit is 200 per user; refuse rather than half-submit.
[ "${N_JOBS}" -le 180 ] || error "${N_JOBS} jobs is too close to the 200-job QOS cap"

next_month() {  # 2009-01 -> 2009-02
    date -d "${1}-01 +1 month" +%Y-%m
}

# EVERY month pair needs a table BEFORE anything is submitted. A missing one is
# not a degraded run, it is a job that stages ~13 GB and then dies on "Missing
# anchor-stat tables" — and while a table set is still being built (a full
# history is ~20 h) that is the normal state of the later months. Check here,
# where the directory is mounted, rather than 32 times on the compute nodes.
XS_TABLE_DIR="${BLL01_DATA_DIR}/${XS_STATS_DIR}"
[ -d "${XS_TABLE_DIR}" ] || error "no such anchor-stat table set: ${XS_TABLE_DIR}"
MISSING=()
for YM in "${MONTHS[@]}"; do
    for M in "${YM}" "$(next_month "${YM}")"; do
        [ -f "${XS_TABLE_DIR}/${M}.npz" ] || MISSING+=("${M}")
    done
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    # Under --dry-run this is the question being asked ("is the table set ready
    # for this panel yet?"), so report it and still print the plan.
    if [ "${DRY_RUN}" = 1 ]; then
        echo "  WARNING: ${#MISSING[@]} table(s) missing from ${XS_STATS_DIR}: ${MISSING[*]}" >&2
    else
        error "${#MISSING[@]} anchor-stat table(s) missing from ${XS_STATS_DIR}: ${MISSING[*]}"
    fi
else
    info "anchor tables: all $((N_JOBS * 2)) months present in ${XS_STATS_DIR}"
fi

if [ "${DRY_RUN}" = 1 ]; then
    for YM in "${MONTHS[@]}"; do
        echo "  ${YM} -> $(next_month "${YM}")   [${FAMILIES}]"
    done
    exit 0
fi

# PREFLIGHT. pythia_sync rsyncs the WORKING TREE, not a commit, so submitting
# while a file is half-edited ships exactly that. On 2026-08-20 it shipped a
# tsfm_layer_ic.py whose main() still called the old fit_pass signature, and 44
# jobs staged their months and died on a TypeError. --help exercises the
# imports and the argument parser, which is enough to catch it.
info "preflight: does the entry point run?"
( cd "${LOCAL_REPO}" && uv run scripts/eval/tsfm_layer_ic.py --help >/dev/null ) \
    || error "tsfm_layer_ic.py --help fails — refusing to sync a broken tree"
[ -z "${TSFM_ALPHA_TABLE:-}" ] || [ -f "${LOCAL_REPO}/${TSFM_ALPHA_TABLE}" ] \
    || error "alpha table not found: ${TSFM_ALPHA_TABLE} (note pythia_sync excludes plots/)"

pythia_setup
pythia_sync

# Resolve the supervised run ids HERE, where /data/lab is mounted and Python
# can match the month prefix against train_meta's run_name. The compute node
# only gets the answer.
sup_runs_for() {
    ( cd "${LOCAL_REPO}" && uv run python -c "
import sys
sys.path.insert(0, 'scripts/eval')
from tsfm_layer_ic import sup_run_dir, SUP_PROJECTS
fams = [f for f in '${FAMILIES}'.split() if f in SUP_PROJECTS]
out = []
for f in fams:
    d = sup_run_dir(f, '$1')
    out.append(f'{SUP_PROJECTS[f]}/{d.name}')
print(' '.join(out))
" )
}

JOB_IDS=()
for YM in "${MONTHS[@]}"; do
    EV="$(next_month "${YM}")"
    SUP_RUNS=""
    case "${FAMILIES}" in
        *sup_*) SUP_RUNS="$(sup_runs_for "${YM}")"
                [ -n "${SUP_RUNS}" ] || error "no supervised runs resolved for ${YM}" ;;
    esac
    OUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export \
            TSFM_FAMILIES='${FAMILIES}' \
            TSFM_TRAIN_MONTH='${YM}' TSFM_EVAL_MONTH='${EV}' \
            TSFM_EXTRA='${TSFM_EXTRA}' \
            TSFM_ALPHA_TABLE='${TSFM_ALPHA_TABLE}' \
            XS_STATS_DIR='${XS_STATS_DIR}' \
            TSFM_SUP_RUNS='${SUP_RUNS}' \
            TSFM_RESULT_DEST='${TSFM_RESULT_DEST}' \
            STAGE_PANEL_CACHE='${STAGE_PANEL_CACHE:-1}' && sbatch \
                --export=ALL \
                --job-name=tsfml-${YM} \
                --partition=${PARTITION} \
                --time=${TIME_LIMIT} \
                ${SBATCH_EXTRA:-} \
                scripts/pythia/slurm_tsfm_layers.sh")
    JOB_ID=$(echo "${OUT}" | awk '{print $4}')
    [ -n "${JOB_ID}" ] || error "sbatch failed for ${YM}: ${OUT}"
    JOB_IDS+=("${JOB_ID}")
    info "  ${YM} -> ${EV}: job ${JOB_ID}"
done

echo ""
echo "  Sweep:      tsfm-layer-ic"
echo "  Jobs:       ${#JOB_IDS[@]}  (${JOB_IDS[*]})"
echo "  Results:    ${TSFM_RESULT_DEST}/<family>_<train month>.json"
echo "  Monitor:    ssh pythia 'squeue -u \$USER'"
echo "  Cancel all: ssh pythia 'scancel ${JOB_IDS[*]}'"
echo ""
