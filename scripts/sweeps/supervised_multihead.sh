#!/bin/bash
# sweeps/supervised_multihead.sh — the multihead supervised generalist:
# ONE backbone, THREE heads, on all 32 sampled months.
#
# The question this exists to answer is what sharing a trunk costs. That only
# has an answer if the multihead differs from the three single-task
# specialists in the shared backbone and IN NOTHING ELSE, so every objective
# knob below is copied from sweeps/full_data_supervised.sh:
#
#     cross-entropy over 11 bins of the RAW target
#     soft labels tau=0.5
#     expected-bin MSE penalty 0.025
#     blr 1e-4 @ bs128
#
# ── THE BINNED MULTIHEAD ARM IS NEW (2026-08-21) ───────────────────────────
#
# MultiTaskSupervisedModel used to REFUSE binned losses outright. The reason
# was real: bin edges are equal-count quantiles of a specific target's
# empirical distribution, so the three heads need three Discretizers — a
# return's quantiles say nothing about a spread change's, and on this data the
# edge ranges differ by three orders of magnitude. `calibrate_discretizers`
# now fits one per task in a single pass over the calibration batches, and
# `_per_task_loss` reads the entry for the task it is scoring.
#
# ── ONE LEARNING RATE, NOT THREE ───────────────────────────────────────────
#
# A multihead trains ONE backbone, so there is one blr to set regardless. And
# under cross-entropy there is no per-task blr to inherit even in principle:
# that split belonged to the retired regression arm, where MSE on a z-score
# gave each target its own gradient scale. A binned objective's scale no
# longer moves with the task, which is why full_data_supervised.sh runs all
# three specialists at a single 1e-4 — see its header.
#
# Gradient balancing across the three heads is the mode's own business
# (task_weights default to equal, normalized over the shared-backbone gradient
# norm), and is left at its defaults deliberately: tuning it would be the
# optimization this sweep was asked not to do.
#
# ── ONLY WHAT THE CALLER SET ───────────────────────────────────────────────
#
# See scripts/sweeps/README.md. This header used to read "EVERY KNOB IS
# PINNED, none inherited from schemas.py", citing the bins x penalty sweep,
# which lost arms when soft_label_temperature's default moved 0.0 -> 0.5 under
# running jobs. Pinning does not fix that: it makes the sweep go on training
# the OLD recipe after schemas.py moves, silently.
#
# So this builder emits an override for a knob ONLY when the caller actually
# set it (the _*_SET capture below, taken before the ':-' defaults apply).
# Left alone, loss_fn, blr, batch, epochs, seed and the pool=last/pos_embed=rope
# encoder all come from mode=supervised_multitask, and the run name still
# records whatever did get set.
#
# Submit:
#   ./scripts/pythia/run_all_months_sweep.sh -p standard_hopper \
#       sweeps/supervised_multihead.sh

# The 32 reported sweep months (run_all_months_sweep.sh). The full-history
# variant is full_data_multihead.sh, which sources this file and overrides
# both this and SWEEP_NAME.
SWEEP_MONTH_SET="sampled32"

SWEEP_NAME="supervised-multihead"

# ── THE OBJECTIVE IS A KNOB NOW (2026-08-25) ───────────────────────────────
#
# LOSS defaults to cross_entropy, which is what every multihead run before
# today used and what the header above describes. Setting it to a SCALAR loss
# (pairwise/corr/mse/smooth_l1) drops the whole binned block -- no n_bins, no
# tau, no expected-bin penalty -- and switches the target to rank, because the
# bins were the only reason xs_target was raw.
#
# The reason to want that: the single-task specialists moved to pairwise on
# 2026-08-25, and "what does sharing a trunk cost" only has an answer if the
# multihead differs from them in the trunk and in NOTHING ELSE. A binned
# multihead against pairwise specialists measures the objective instead.
# WAS IT THE CALLER, OR IS IT JUST THE DEFAULT? Captured BEFORE the ":-"
# below fills it in, because the two need different answers: the run NAME
# wants a concrete value either way, while training must be passed the knob
# ONLY when a caller actually moved it. Emitting it unconditionally would put
# a second copy of the recipe in this file -- the thing README.md forbids --
# and comparing against a literal "pairwise" here would be that same copy
# wearing a condition. See scripts/sweeps/README.md.
_LOSS_SET="${LOSS+x}"
_SEED_SET="${SEED+x}"
LOSS="${LOSS:-pairwise}"
SEED="${SEED:-42}"
# Appended to the run name, so two arms differing only in EXTRA_TRAIN_ARGS are
# not name-identical and pooled by everything that reads a run name.
SWEEP_TAG="${SWEEP_TAG:-}"

# ONE VALUE, and it has to stay one rather than none: the submitter loops
# months over SWEEP_VALUES, so an empty list trains nothing. The trunk has no
# grid axis of its own left -- everything that varies (loss, bins, tau,
# penalty, tag) is an environment knob, and months are fanned out by the
# caller.
SWEEP_VALUES=("base")

TASKS="${TASKS:-[return_900,volatility_change_900,spread_change_900]}"
# The budget lives in MultiTaskSupervisedModeConfig.training_overrides and is
# NOT restated here. The ":-" values below feed the run name only; the same
# _SET capture as above decides what training actually receives.
_BLR_SET="${BLR+x}"
_BATCH_SET="${BATCH_SIZE+x}"
_EPOCHS_SET="${NUM_EPOCHS+x}"
BLR="${BLR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"

sweep_train_args() {

    # Run name = YYYY-MM of the month the TRAINING SPAN ENDS ON, from
    # TRAIN_END. It was TRAIN_START, which named the same month only while a
    # run trained one calendar month; the recipe is now
    # DatasetConfig.train_span_months of data BEFORE the eval month, so
    # TRAIN_START moves with the span and TRAIN_END does not. Keying the name
    # to TRAIN_END keeps a run's name pointing at the month it is evaluated
    # against, and keeps it stable if the span length ever changes again.
    local YM="${TRAIN_END:0:7}"

    local TAG=""
    [ -n "${SWEEP_TAG}" ] && TAG="_${SWEEP_TAG}"

    # BINNED: xs_target=raw, because the bins ARE quantiles of the raw target.
    # SCALAR: xs_target=rank, the value the supervised HPO selected, and NONE
    # of the bin knobs -- MultiTaskSupervisedModel refuses a binned knob on a
    # scalar loss, and it is right to.
    #
    # xs_eval_target stays zscore either way, so the probe is scored on exactly
    # what every other arm's probe is scored on.
    #
    # n_global_views / n_local_views are NOT pinned here: they are not fields
    # of DatasetConfig at all, they live in the mode's own dataset_overrides,
    # which already sets 1 and 0 (every head reads views[0], so extra crops
    # were pure dataloader waste). Overriding dataset.n_global_views fails
    # config composition outright.
    local TARGET="uniform" LOSS_TAG="${LOSS}"

    # Only what a caller actually moved. Everything absent here -- loss_fn,
    # blr, batch, epochs, seed, the dataset targets, live_eval, pool/pos_embed
    # -- is mode=multi_supervised's own default and is deliberately NOT
    # restated. mode.tasks stays because the schema declares it MISSING.
    local OV=""
    [ -n "${_LOSS_SET}" ]   && OV="${OV} mode.loss_fn=${LOSS}"
    [ -n "${_BLR_SET}" ]    && OV="${OV} optimizer.blr=${BLR}"
    [ -n "${_BATCH_SET}" ]  && OV="${OV} training.per_device_train_batch_size=${BATCH_SIZE}"
    [ -n "${_EPOCHS_SET}" ] && OV="${OV} training.num_epochs=${NUM_EPOCHS}"
    [ -n "${_SEED_SET}" ]   && OV="${OV} training.seed=${SEED}"

    echo "mode=multi_supervised \
        mode.tasks=${TASKS} \
        backbone=transformer \
       ${OV} \
        ${EXTRA_TRAIN_ARGS:-} \
        wandb.project=${SWEEP_NAME} \
        wandb.group=${SWEEP_NAME}${TAG} \
        wandb.run_name=${YM}_multi_${LOSS_TAG}${TAG}_s${SEED}"
}
