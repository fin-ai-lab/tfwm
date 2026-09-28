#!/bin/bash
# supervised_scaling.sh — rank IC against training compute, one ViT size at a
# time, on the 31 reported months.
#
# THE FIGURE THIS FEEDS is plots/scaling/supervised_scaling.png: three panels
# (one per target), x = training FLOPs, one curve per scale in VIT_SCALES
# (schemas.py). Collected by scripts/eval/collect_supervised_scaling.py.
#
# ── WHAT A RUN IS ─────────────────────────────────────────────────────────
#
# The supervised specialist recipe, unchanged in everything but three knobs:
#
#   1. THE SCALE. mode.backbone.* from VIT_SCALES: width, heads and MLP width.
#      Depth, patch, sequence, readout, drop-path, the objective, the batch
#      (256 cells x K=16 a step), the span (six months) and the passes (12)
#      are the recipe's and are NOT passed here (scripts/sweeps/README.md).
#      "small" IS the reported model, so that curve's rungs are the paper's
#      own arm under a WSD schedule.
#
#   2. A STABLE SCHEDULE WITH BRANCH COOLDOWNS, the WSD scaling method.
#      Under cosine the LR differs at every step and a mid-run checkpoint is
#      nobody's intended endpoint. Here the run warms up (128 steps, pinned)
#      and holds the peak rate to its last step; at each target the run
#      LEAVES the stable trajectory, decays linearly to 0.1x over the last
#      10% of that target's steps, saves, and is put back (weights and
#      optimizer state) to continue -- checkpoint.anneal_steps, run by
#      pretrain.py. Every dot is therefore a finished model at its own
#      compute, from one run per (month, task, scale). The last target ends
#      the run, so there is no separate endpoint.
#
#      The warmup is pinned in steps (optimizer.warmup_steps) so a target is
#      the same point of the same schedule in every month: 5% of the run
#      would end at step 119 in one month and 268 in another.
#
#   3. THE LEARNING RATE, per scale, from VIT_SCALE_BLR -- measured on holdout
#      set 2 by supervised_scaling_lr.sh. Small inherits the recipe's 2e-4.
#      A rung whose LR has not been measured REFUSES TO RUN (scaling_lib.sh).
#
# ── THE TARGETS ────────────────────────────────────────────────────────────
#
# Half-decades of training FLOPs, 1e16 to a per-scale top (user, 2026-09-20):
# tiny to 3.33e17, small to 1e18, base to 3.33e18. A target is a STEP COUNT
# per scale (market_jepa/eval/flops.py:steps_for_flops), because a step costs
# the same FLOPs in every month -- same batch, same view length -- so a
# target is one x for all 31 months and the figure can average them. The
# runs are the target's length, NOT the recipe's 12 passes: past 3.33e17 the
# run is longer than the recipe and the data does not grow, so the high end
# measures more passes over the same six months. Every target the run
# reaches is saved; each is scored in-job, head only.
#
# A target whose branch point sits inside the 128-step warmup is dropped for
# that scale (steps_for_flops): base starts at 1e17, small and tiny at 1e16.
#
#   tiny  [243 810 2433 8103]        (3.33e17 = 8,103 steps)
#   small [223 670 2231 6700]        (1e18    = 6,700 steps)
#   base  [176 588 1764 5875]        (3.33e18 = 5,875 steps, 1e16/3.33e16 in warmup)
#
# ── COST ───────────────────────────────────────────────────────────────────
#
# Per view forward (market_jepa/eval/flops.py): tiny 3.3 GF, small 12.1 GF,
# base 46.1 GF; x3 for training, x4,096 views a step. The cooldowns add 10%
# of every target on top of the last one (~15%). Measured rates: tiny ~0.85
# s/step on an L40S, small ~1.2 s/step there, base ~2.8 s/step on an H100.
# A month (three tasks, one job) is ~20 h tiny, ~8 h small, ~16 h base,
# inside the bundle's 3-day limit.
#
# Every target is scored in-job (train_bundle_body.sh scores <run>/<step>/),
# HEAD ONLY (POST_TRAIN_PROBE=0, enforced by scaling_lib.sh): the eval
# month's a8 panel and nothing else. The run root is the last target's model
# again; the collector keeps the step dir and drops the root.
#
# ── LAUNCH ─────────────────────────────────────────────────────────────────
#
#   DAYSTORE=1 POST_TRAIN_PROBE=0 SCALES="tiny" ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_l40s sweeps/supervised_scaling.sh
#   DAYSTORE=1 POST_TRAIN_PROBE=0 SCALES="small" \
#       SBATCH_EXTRA="--partition=standard_hopper,standard_l40s --mem=96G" \
#       ./scripts/pythia/run_all_months_sweep.sh sweeps/supervised_scaling.sh
#
# Small goes to BOTH queues (user, 2026-09-19): tiny is loader-bound and
# wastes an H100, base needs one, and small fills whichever drains first.
# --mem=96G because the 87 GB span otherwise crawls on hopper's default.
#   DAYSTORE=1 POST_TRAIN_PROBE=0 SCALES="base" ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/supervised_scaling.sh
#
# Then, on bll01, once jobs land:
#   uv run scripts/eval/collect_supervised_scaling.py
#   uv run plots/scaling/supervised_scaling.py

source "$(dirname "${BASH_SOURCE[0]}")/scaling_lib.sh"

SWEEP_NAME="supervised-scaling"
SWEEP_MONTH_SET="sampled32"

SCALES="${SCALES:-tiny small base}"
SWEEP_TASKS="${SWEEP_TASKS:-return_900 volatility_change_900 spread_change_900}"
# In steps, pinned: see (2) above. 128 is ~5% of the shortest month's recipe
# run, the recipe's warmup fraction applied to the month that set the old
# ladder; kept so the Small LR-grid and ladder waves share it.
SCALING_WARMUP_STEPS="${SCALING_WARMUP_STEPS:-128}"
SCALING_ANNEAL_FRAC="${SCALING_ANNEAL_FRAC:-0.1}"
# The recipe's batch: 256 cells x K=16 stocks, the only thing about a step
# that the FLOPs-to-steps conversion needs beyond the model. Pinned to the
# schema by tests/test_supervised_scaling.py.
SCALING_VIEWS_PER_STEP="${SCALING_VIEWS_PER_STEP:-4096}"
# Training-FLOPs targets, per scale; SCALING_FLOPS_TARGETS overrides all.
_flops_targets() {
    if [ -n "${SCALING_FLOPS_TARGETS:-}" ]; then
        echo "${SCALING_FLOPS_TARGETS}"
        return
    fi
    case "$1" in
        tiny)  echo "1e16 3.33e16 1e17 3.33e17" ;;
        small) echo "1e16 3.33e16 1e17 3.33e17 1e18" ;;
        base)  echo "1e16 3.33e16 1e17 3.33e17 1e18 3.33e18" ;;
        *)     echo "1e16 3.33e16 1e17 3.33e17 1e18" ;;
    esac
}
# checkpoint.anneal_steps for a scale: the targets as total step counts,
# warmup-bound ones dropped (flops.py:steps_for_flops).
scaling_anneal_steps() {
    uv run python -m market_jepa.eval.flops --scale "$1" \
        --targets $(_flops_targets "$1") \
        --views-per-step "${SCALING_VIEWS_PER_STEP}" \
        --warmup-steps "${SCALING_WARMUP_STEPS}" \
        --anneal-frac "${SCALING_ANNEAL_FRAC}"
}

SWEEP_VALUES=()
for _S in ${SCALES}; do
    for _T in ${SWEEP_TASKS}; do
        SWEEP_VALUES+=("${_S}:${_T}")
    done
done

sweep_train_args() {
    local SCALE="${1%%:*}" TASK="${1#*:}"
    scaling_require_daystore || return 1
    scaling_require_head_only || return 1
    local BB BLR STEPS LAST
    BB=$(scaling_backbone_args "${SCALE}") || return 1
    BLR=$(scaling_blr_arg "${SCALE}") || return 1
    STEPS=$(scaling_anneal_steps "${SCALE}") || return 1
    [ -n "${STEPS}" ] || { echo "ERROR: no FLOPs target of '${SCALE}' clears the warmup" >&2; return 1; }
    LAST="${STEPS##* }"
    STEPS="[$(echo ${STEPS} | tr ' ' ',')]"
    # Run name = the YYYY-MM the training span ENDS on, like the specialists.
    local YM="${TRAIN_END:0:7}"
    # ONE PROJECT PER SCALE: the collector keys the curve on the project name
    # and the bundle suffixes it with the commit and the span, so a wave at
    # a different recipe cannot be pooled with this one by accident.
    echo "mode=supervised \
        mode.task=${TASK} \
        backbone=transformer \
        ${BB} \
        ${BLR} \
        optimizer.lr_schedule=stable \
        optimizer.anneal_frac=${SCALING_ANNEAL_FRAC} \
        optimizer.warmup_steps=${SCALING_WARMUP_STEPS} \
        training.num_epochs=null \
        training.max_train_steps=${LAST} \
        checkpoint.anneal_steps=${STEPS} \
        ${EXTRA_TRAIN_ARGS:-} \
        wandb.project=supervised-scaling-${SCALE} \
        wandb.group=supervised-scaling \
        wandb.run_name=${YM}_${TASK}_vit-${SCALE}_wsd"
}
