#!/bin/bash
# sweeps/ts2vec_blr.sh — a learning rate for ts2vec that does not collapse,
# on the OFF-PANEL optimization set.
#
# TIMEMAE IS NOT HERE, AND THAT IS THE FINDING. It shared ts2vec's outage and
# was swept alongside it for one revision of this file, then dropped: since
# the 2026-09-15 fix it has trained 8 of 8 spans without collapsing, so its
# inherited 1e-3 is not costing anything and the campaign trains it at the
# DEFAULT. Re-adding a timemae rung needs a reason beyond symmetry with
# ts2vec -- the two arms failed together for one shared cause and only one of
# them still fails.
#
# WHY TS2VEC, AND WHY NOW. Both arms were dead from 2026-09-13 to 2026-09-15:
# the information token moved the per-window columns out of the patch
# embedding and neither arm's path was taught to split them off (ts2vec via
# forward_patches, timemae via a direct backbone.patch_embed call for its
# tokenizer). They crashed at step 0 on every month, so neither has ever been
# tuned against the 6-month / 12-pass recipe -- both still carry the original
# papers' 1e-3, which is a one-month, 300-pass number wearing a 6-month hat.
# For timemae that turned out to be survivable; for ts2vec it is not.
#
# ts2vec THEN FAILED A SECOND WAY once it could run at all. It trains four
# epochs of twelve and diverges:
#
#   WARNING - Non-finite loss at step 2796, skipping optimizer step
#   RuntimeError: 50 consecutive steps had no trainable batch
#
# on 3 of 15 attempts (2026-09-15, spans 2008-03, 2010-03 and one other).
# Diverge-then-abort partway through a run, on some months and not others, at
# an LR inherited from a different budget, is what an LR that is too high
# looks like. The loss is the official port's: hierarchical contrast over RAW
# DOT PRODUCTS of unnormalised representations, with no temperature and no
# normalisation anywhere -- so the logits are free to grow with the
# representation norm, and nothing in the objective pushes back.
#
# NOT A PRECISION BUG, CHECKED FIRST. Training is fp32 (no autocast anywhere
# in pretrain.py) and TrainingConfig.max_grad_norm=1.0 is applied, so neither
# fp16 overflow nor an unclipped gradient step is available as the mechanism.
#
# NOT THE POOL CHANGE EITHER, ALSO CHECKED. TS2VecModeConfig moved pool
# "max" -> "mean" and that is a natural suspect. It is provably not this: the
# loss reaches the backbone through forward_patches, which returns per-patch
# rows and never pools, and max vs mean changes no module construction (only
# pool="cls" does, which the mode rejects). Same seed, same batch, both
# settings: bit-identical loss, 9.454476356506348. Do not spend a wave on it.
#
# ── WHAT IS SWEPT, AND NOTHING ELSE ────────────────────────────────────────
#
# optimizer.blr, which both mode configs already name as their primary knob
# ("Primary sweep knob: optimizer.blr (original lr: 1e-3)"). The ladder keeps
# 1e-3 as the CONTROL -- it is the current default and the arm that diverges,
# so a wave without it measures the fix but not the thing it fixes.
#
# Per scripts/sweeps/README.md nothing else is restated here. Both arms carry
# their own dataset_overrides (views, augmentation) and the rest of the recipe
# -- 12 passes over DatasetConfig.train_span_months=6, batch size, targets,
# readout -- is schemas.py's. timemae's num_epochs comes from the MODE and
# ts2vec's from TrainingConfig.fallback_num_epochs; both are 12 as of
# 2026-09-13 and they move by different routes, so check both before pooling
# this sweep across a commit that touches either.
#
# ── THE OFF-PANEL SET, AND WHY THE LAUNCHER HAS TO BE TOLD ─────────────────
#
# Holdout set 2 (2009-06 2011-12 2015-06 2016-03 2021-03), disjoint from the
# reported 32 in BOTH roles -- no holdout training month and no holdout eval
# month collides with the 32 or with their eval months. An LR chosen on the
# reported panel is a hyper-parameter chosen on the panel the paper scores.
#
# SWEEP_MONTH_SET="any" because run_all_months_sweep.sh declares "sampled32"
# and would otherwise refuse the file. That is the guard doing its job: this
# is the rare sweep that legitimately runs off-panel, and it says so.
#
#   POST_TRAIN_IC_EVAL=1 \
#   MONTHS_OVERRIDE="$(uv run scripts/experiments/holdout_months.py --set 2)" \
#       ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/ts2vec_blr.sh
#
# MONTHS_OVERRIDE MUST BE A COMMAND SUBSTITUTION THAT SUCCEEDED. The launcher
# refuses an empty one rather than falling back to the sampled months, which
# is the failure run_tsfm_layers.sh hit when holdout_months.py died on its
# PEP 723 header and resolved to the empty list.
#
# ── SCORED, UNLIKE THE WAVE THIS REPAIRS ───────────────────────────────────
#
# ssl_lejepa_all.sh is train-only: the probe-fit protocol is unsettled, so
# scoring it now would buy numbers a protocol change invalidates. An HPO sweep
# cannot do that -- "it stopped diverging" ranks 1e-5 first, and an LR that is
# stable because it learns nothing is the wrong answer. So this one evaluates,
# and the panel cache for all five months and their eval months is present
# (verified 2026-09-15, keys 6ea86b6bdc61 probe / 36be77c236ec eval).

SWEEP_MONTH_SET="any"

SWEEP_NAME="ts2vec-blr"

WANDB_GROUP="${WANDB_GROUP:-ssl-blr-holdout2}"

# 1e-3 is the CONTROL, not a candidate on equal footing: it is what both arms
# run today and what ts2vec diverges at. Downward only -- the failure is
# divergence, so there is no reading of it under which a higher LR is the fix.
#
# DELIBERATELY SHORT. The question is "does it stop collapsing", not "which LR
# is best", and those want the budget spent differently: collapse is
# stochastic (ts2vec diverged on 3 of 15 spans, not on all of them), so the
# signal is the RATE, and a rate needs months per rung rather than rungs. Five
# months on three ts2vec rungs distinguishes "never collapsed in 5" from "1e-3
# collapsed"; the same budget spread over eight rungs would give one sample
# each and could not. 1e-4 is off the ladder for the same reason -- an LR that
# is stable because it learns nothing is not an answer, and 2e-4 already
# brackets it if 5e-4 is not enough.
#
SWEEP_VALUES=(
    "ts2vec_blr1e-3"
    "ts2vec_blr5e-4"
    "ts2vec_blr2e-4"
)

if [ -n "${ARM_FILTER:-}" ]; then
    _FILTERED=()
    for V in "${SWEEP_VALUES[@]}"; do
        for K in ${ARM_FILTER}; do
            [ "${V}" = "${K}" ] && { _FILTERED+=("${V}"); break; }
        done
    done
    if [ ${#_FILTERED[@]} -eq 0 ]; then
        echo "ERROR: ARM_FILTER='${ARM_FILTER}' matched no SWEEP_VALUES" >&2
        return 1 2>/dev/null || exit 1
    fi
    SWEEP_VALUES=("${_FILTERED[@]}")
fi

# Same refusal as ssl_lejepa_all.sh and for the same reason: dataset.backend=
# days serves supervised cross_stock cells only, and pretrain.py raises for
# both of these modes within seconds of the first batch.
if [ "${DAYSTORE:-0}" = "1" ]; then
    echo "ERROR: DAYSTORE=1 cannot run this sweep -- ts2vec is not a" >&2
    echo "       supervised cross_stock arm; it ValueErrors immediately." >&2
    return 1 2>/dev/null || exit 1
fi

# The inverse of ssl_lejepa_all.sh's guard. That wave refuses to be scored;
# this one refuses NOT to be, because the whole output is a ranking.
if [ "${POST_TRAIN_IC_EVAL:-1}" = "0" ]; then
    echo "ERROR: this sweep exists to RANK learning rates and cannot do that" >&2
    echo "       without a score. Submit with POST_TRAIN_IC_EVAL=1." >&2
    return 1 2>/dev/null || exit 1
fi

sweep_train_args() {
    local ARM="$1"
    local YM="${TRAIN_END:0:7}"
    local MODE BLR

    case "${ARM}" in
        ts2vec_blr*)  MODE="ts2vec" ;;
        *) echo "ERROR: unknown arm '${ARM}'" >&2; return 1 ;;
    esac
    BLR="${ARM#*_blr}"

    # One project per (mode, blr) so a month is unique inside it, matching the
    # wave's convention -- and so a later `ls` over the checkpoint tree reads
    # the ladder off the directory names without opening a train_meta.json.
    echo "mode=${MODE} \
        backbone=transformer \
        optimizer.blr=${BLR} \
        wandb.project=ssl-blr-${MODE}-${BLR} \
        wandb.group=${WANDB_GROUP} \
        wandb.run_name=${YM}_${MODE}_blr${BLR}"
}
