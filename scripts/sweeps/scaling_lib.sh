#!/bin/bash
# scaling_lib.sh — what the two ViT-scale sweeps share: how a scale name
# becomes backbone overrides, and how it becomes a learning rate.
#
# The table is market_jepa/schemas.py:VIT_SCALES; nothing numeric lives here.
# Sourced by supervised_scaling.sh (the ladder) and supervised_scaling_lr.sh
# (the holdout-2 LR grid that fills in VIT_SCALE_BLR).

# Hydra overrides that make mode=supervised's backbone the named scale.
# WIDTH, HEADS AND MLP ONLY (see VIT_SCALES); d_embedding follows the width so
# the output projection stays square, as it is in the reported model.
scaling_backbone_args() {
    local SCALE="$1" DIMS H A M
    DIMS=$(uv run python -c "
from market_jepa.schemas import VIT_SCALES as S
d = S['${SCALE}']
print(d['hidden_size'], d['num_attention_heads'], d['intermediate_size'])") \
        || { echo "ERROR: '${SCALE}' is not a VIT_SCALES rung" >&2; return 1; }
    read -r H A M <<< "${DIMS}"
    echo "mode.backbone.d_embedding=${H} \
        mode.backbone.config.hidden_size=${H} \
        mode.backbone.config.num_attention_heads=${A} \
        mode.backbone.config.intermediate_size=${M}"
}

# The ``optimizer.blr=...`` override for a scale, or NOTHING for a scale that
# trains the recipe's own LR. Resolution, in order:
#
#   SCALING_BLR (env)     > an explicit pin for THIS launch, every scale
#   VIT_SCALE_BLR[scale]  > the measured value (supervised_scaling_lr.sh)
#   absent from the table > the recipe's LR, inherited from the mode
#   present but None      > HARD ERROR: the LR sweep has not landed
#
# No numeric fallback for tiny/base, deliberately: 2e-4 was measured on Small
# and is not known to be right anywhere else, so running a rung at it by
# default would put an untuned point on a scaling curve with nothing on the
# axis to say so.
scaling_blr_arg() {
    local SCALE="$1" V
    if [ -n "${SCALING_BLR:-}" ]; then
        echo "optimizer.blr=${SCALING_BLR}"
        return 0
    fi
    V=$(uv run python -c "
from market_jepa.schemas import VIT_SCALE_BLR as B
print('inherit' if '${SCALE}' not in B else ('unset' if B['${SCALE}'] is None else B['${SCALE}']))") \
        || return 1
    case "${V}" in
        inherit) echo "" ;;
        unset)
            echo "ERROR: VIT_SCALE_BLR['${SCALE}'] is None -- run" >&2
            echo "       scripts/sweeps/supervised_scaling_lr.sh on holdout set 2 first," >&2
            echo "       record the winner in schemas.py, or pin SCALING_BLR=... for this launch." >&2
            return 1 ;;
        *) echo "optimizer.blr=${V}" ;;
    esac
}

# Every scale sweep trains on the day store: the recipe's batch is 256 cells
# x 16 stocks = 4,096 rows a step THERE and a sixteenth of that on mds (see
# ssl_finetune_breadth.sh for the wave that cost). Refuse rather than train
# a different recipe quietly.
scaling_require_daystore() {
    if [ "${DAYSTORE:-0}" != "1" ]; then
        echo "ERROR: the scale sweeps run on the day-major store; launch with DAYSTORE=1" >&2
        return 1
    fi
}

# HEAD ONLY. The scaling figure reads the model's own forecast (user,
# 2026-09-19): the ridge probe's a36 embedding of the train month is the
# bulk of scoring a checkpoint, and a 12-rung ladder pays it twelve times a
# run for a number the figure does not draw. POST_TRAIN_PROBE=0 also drops
# the probe panel and the train-end mosaic from staging. Refuse rather than
# quietly pay for probes: the flag travels from the launcher's environment,
# so a launch that forgets it would otherwise run the whole wave the slow
# way.
scaling_require_head_only() {
    if [ "${POST_TRAIN_PROBE:-1}" != "0" ]; then
        echo "ERROR: the scale sweeps score the head only; launch with POST_TRAIN_PROBE=0" >&2
        return 1
    fi
}
