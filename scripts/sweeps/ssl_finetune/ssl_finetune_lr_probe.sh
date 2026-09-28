#!/bin/bash
# ssl_finetune_lr_probe.sh — how fast does the finetune lose the probe it
# started from, as a function of the learning rate?
#
# ── WHY THIS EXISTS ────────────────────────────────────────────────────────
#
# In the 515b8a wave the head init reproduces the frozen probe EXACTLY
# (2008-08 volatility: step-0 head 0.0984, frozen probe 0.0986, and every
# month's head init is fit on exactly the rows of the probe curve's top point,
# at readout=last). The first point the figure draws is nevertheless far below
# it: 0.0627 at step 32, about seven standard errors down.
#
# That is not explained by how far the weights moved. At step 32 the WSD
# schedule is at 1.77e-6 and the cumulative sum of the LR over those steps is
# 2.99e-5, which bounds an Adam update per parameter at ~3e-5 -- 0.15% of a
# backbone weight. Over the WHOLE run backbone_drift reaches only 0.0202.
#
# So either a 0.15% perturbation really does cost a third of a low-SNR
# readout, or something other than training moved between step 0 and step 32.
# THE blr=0 ARM SEPARATES THOSE and is the point of this file: with no
# learning there is nothing to degrade, so a step-32 IC below the init would
# be a property of the checkpoint/scoring path, not of the optimizer.
#
# ── THE ARMS ───────────────────────────────────────────────────────────────
#
# 0 is spelled 1e-12 rather than 0 because the LR is a divisor in the scaling
# report; at 1e-12 a 3,700-step run moves a weight by under 4e-9, which is
# zero for every purpose here.
#
# ── THE CHECKPOINTS ARE STEPS, NOT FRACTIONS ───────────────────────────────
#
# The breadth sweep asks for label budgets and gets whatever step they land
# on. Here the question is about the first few dozen steps specifically, so
# checkpoint.save_steps names them: the collapse in the wave happened entirely
# before step 32 and no checkpoint in it looked earlier.

source "$(dirname "${BASH_SOURCE[0]}")/ssl_base_lib.sh"
SWEEP_NAME="ssl-ft-lrprobe-$(ssl_base_arm_tag)"

# One task. volatility_change_900 has the largest init IC of the three and so
# the clearest signal; return_900 is the task the wave hurt most, and is the
# obvious follow-up once the mechanism is known.
PROBE_TASK="${PROBE_TASK:-volatility_change_900}"

# blr. 1e-5 is the wave's value and is included so this is a superset of it.
SWEEP_VALUES=(1e-12 1e-6 3e-6 1e-5)

# Dense where the wave collapsed, thinning out afterwards.
PROBE_SAVE_STEPS="${PROBE_SAVE_STEPS:-[1,2,4,8,16,32,64,128,256,512,1024]}"

sweep_train_args() {
    local BLR="$1"
    local YM="${EVAL_START:0:7}"
    local RUN_ID
    RUN_ID=$(ssl_base_run_id "${YM}")
    if [ -z "${RUN_ID}" ]; then
        echo "ERROR: no run id for eval month ${YM} in the base manifest" >&2
        return 1
    fi
    # THE LR IS IN THE RUN NAME. Four arms share a project, and the breadth
    # sweep's name (ft_<task>_wsd) does not mention the LR -- reusing it here
    # would put four different runs under one name and leave whichever
    # finished last.
    echo "mode=supervised \
        mode.task=${PROBE_TASK} \
        mode.init_backbone_from=${SSL_BASE_DIR}/${RUN_ID} \
        mode.init_head_from=${SSL_HEAD_DIR}/${RUN_ID} \
        optimizer.blr=${BLR} \
        optimizer.lr_schedule=wsd \
        optimizer.decay_frac=0.1 \
        checkpoint.save_steps=${PROBE_SAVE_STEPS} \
        backbone=transformer \
        ${EXTRA_TRAIN_ARGS:-} \
        wandb.project=${SWEEP_NAME} \
        wandb.group=${SWEEP_NAME} \
        wandb.run_name=ft_${PROBE_TASK}_blr${BLR}"
}
