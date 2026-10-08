#!/bin/bash
# supervised_scaling_lr.sh — the learning rate for each NEW ViT scale, on
# holdout set 2.
#
# blr 2e-4 was measured on ViT-Small (SupervisedModeConfig, the 2026-09-13
# passes x LR grid) and there is no reason it is right for a model a quarter
# or four times the width. A scaling curve whose rungs train at a rate tuned
# for one of them measures the tuning as much as the scale. So each other
# rung gets its own grid here, and the winner goes into VIT_SCALE_BLR
# (schemas.py), which is where supervised_scaling.sh reads it from.
#
# HOLDOUT SET 2 (scripts/experiments/holdout_months.py --set 2), five months
# disjoint from the 31 reported ones in both training and eval roles, so the
# rate is not chosen on a panel the figure reports. Small is NOT swept: its
# LR is the recipe's and was chosen on holdout months already.
#
# THE SAME RUN AS THE LADDER, minus the ladder: WSD with the pinned warmup and
# the 10% decay, the recipe's batch, span and passes, on the day store -- so
# the annealed endpoint scored here is the same object the scaling sweep's
# run root is. No checkpoint.save_steps, and head-only scoring like the
# ladder (POST_TRAIN_PROBE=0), so scoring is one head per run.
#
# THE GRID is per scale and geometric in 2x: a smaller model usually wants a
# hotter rate and a larger one a cooler rate, so each grid is centred one
# step off the recipe's 2e-4 in that direction, and both include 2e-4 so the
# comparison to the reported model's rate is paired. SCALING_LRS overrides
# both with one list.
#
# READ IT with scripts/eval/summarize_supervised_scaling_lr.py, which prints
# the month-mean head IC per (scale, task, LR) and the paired difference
# from 2e-4 (the probe column stays empty under head-only scoring). The
# figure reads the head, so the head picks the rate.
#
# ── LAUNCH ─────────────────────────────────────────────────────────────────
#
#   DAYSTORE=1 POST_TRAIN_PROBE=0 MONTHS_OVERRIDE="$(uv run scripts/experiments/holdout_months.py --set 2)" \
#     SCALES="tiny" ./scripts/generic/run_all_months_sweep.sh <variant> --partition <l40s-partition> \
#     sweeps/supervised_scaling_lr.sh
#   DAYSTORE=1 POST_TRAIN_PROBE=0 MONTHS_OVERRIDE="$(uv run scripts/experiments/holdout_months.py --set 2)" \
#     SCALES="base" ./scripts/generic/run_all_months_sweep.sh <variant> --partition <h100-partition> \
#     sweeps/supervised_scaling_lr.sh
#
# 4 rates x 3 tasks = 12 runs a job, 5 jobs a scale.

source "$(dirname "${BASH_SOURCE[0]}")/scaling_lib.sh"

SWEEP_NAME="supervised-scaling-lr"
# Any month set: this sweep is launched on holdout set 2 via MONTHS_OVERRIDE.

SCALES="${SCALES:-tiny base}"
SWEEP_TASKS="${SWEEP_TASKS:-return_900 volatility_change_900 spread_change_900}"
SCALING_WARMUP_STEPS="${SCALING_WARMUP_STEPS:-128}"

# Per-scale grids, see THE GRID above. One override list serves every scale.
_lr_grid() {
    if [ -n "${SCALING_LRS:-}" ]; then
        echo "${SCALING_LRS}"
        return
    fi
    case "$1" in
        tiny)  echo "1e-4 2e-4 4e-4 8e-4" ;;
        base)  echo "5e-5 1e-4 2e-4 4e-4" ;;
        *)     echo "1e-4 2e-4 4e-4" ;;
    esac
}

SWEEP_VALUES=()
for _S in ${SCALES}; do
    for _LR in $(_lr_grid "${_S}"); do
        for _T in ${SWEEP_TASKS}; do
            SWEEP_VALUES+=("${_S}:${_LR}:${_T}")
        done
    done
done

sweep_train_args() {
    local SCALE LR TASK
    IFS=':' read -r SCALE LR TASK <<< "$1"
    scaling_require_daystore || return 1
    scaling_require_head_only || return 1
    local BB
    BB=$(scaling_backbone_args "${SCALE}") || return 1
    local YM="${TRAIN_END:0:7}"
    # optimizer.blr is the swept axis, so it is passed as a variable and never
    # trips the restated-default check. THIS IS THE LR ITSELF, not a bs128
    # quote: SupervisedModeConfig carries default_batch_size=None.
    echo "mode=supervised \
        mode.task=${TASK} \
        backbone=transformer \
        ${BB} \
        optimizer.blr=${LR} \
        optimizer.lr_schedule=wsd \
        optimizer.decay_frac=0.1 \
        optimizer.warmup_steps=${SCALING_WARMUP_STEPS} \
        ${EXTRA_TRAIN_ARGS:-} \
        wandb.project=supervised-scaling-lr-${SCALE} \
        wandb.group=supervised-scaling-lr \
        wandb.run_name=${YM}_${TASK}_vit-${SCALE}_blr${LR}"
}
