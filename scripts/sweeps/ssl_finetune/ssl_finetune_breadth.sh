#!/bin/bash
# ssl_finetune_breadth.sh — what a label budget buys when it is spent on the
# ENCODER rather than on a probe.
#
# For each month of the base arm, the month's SSL checkpoint is finetuned
# END-TO-END on N labelled rows, with the head STARTING AT the ridge probe
# (scripts/eval/fit_ridge_head_init.py). One run per (task, N).
#
# ── WHAT THIS REPLACES ─────────────────────────────────────────────────────
#
# The old ssl_finetune.sh swept `head` and `e2e` styles over seven label
# fractions. Both of the things it did differently are gone:
#
#   1. THE HEAD-ONLY ARM. A frozen encoder with a head trained by SGD is a
#      probe, fit badly. The probe is already measured properly, on the full
#      pool, by plots/core/probe_fit_breadth.py -- so the arm spent GPU
#      time to get a worse estimate of a number already in hand.
#   2. THE RANDOM HEAD. Starting the head at random made every small-budget
#      run spend its labels learning to read an embedding it was handed. That
#      is the same probe again, now billed to the finetune. Here the head
#      begins AS the probe and every label goes into the encoder.
#
# The multihead finetune arm is gone too; it was not being run, and
# MultiTaskSupervisedModeConfig no longer has the knobs for it.
#
# ── THE AXIS IS ROWS, AND WHY ──────────────────────────────────────────────
#
# SWEEP_ROWS is a count of a36 panel rows, matching FIT_SIZES in
# scripts/eval/probe_fit_size.py exactly, so a point here lands on the same x
# as a point there and the two curves can be read on one pair of axes. The
# trainer's knob is a FRACTION of ticker-days; the conversion is n /
# n_rows_pool from the manifest, and it is exact because the panel has a fixed
# 36 anchors per ticker-day, which cancels.
#
# ── THE CAVEAT, STATED WHERE IT CANNOT BE MISSED ───────────────────────────
#
# THE HEAD INIT ALWAYS SEES THE FULL SIX-MONTH SPAN. It does not shrink with
# N. So this is NOT a total-label-budget curve: at small N the model has
# already been handed a head fit on ~2.7M rows, and the curve's left end
# asymptotes to the FULL-POOL PROBE IC rather than to zero. What the axis
# measures is how many labels the ENCODER was adapted on, holding the readout
# start fixed. Anyone reading the left half of this curve as "SSL + N labels"
# is reading it wrong.
#
# Below ~131072 rows there is also barely any training to speak of: 32768 rows
# is ~915 ticker-days, which at effective batch 256 over 12 passes is ~36
# optimizer steps. Those points are kept because they anchor the curve at its
# init, not because they measure finetuning.
#
# ── THE BATCH IS THE RECIPE'S, AND THAT IS THE WHOLE POINT ────────────────
#
# THIS SWEEP RUNS ON THE DAY-MAJOR BACKEND. The comparison it exists to make
# is "same recipe, different initialization", and for a long time it was not:
# the supervised specialist trains on backend=days at 256 cells/step x K=16 =
# 4,096 rows per optimizer step, while this sweep ran on mds, where the same
# nominal batch gives 16 cells x K=16 = 256 rows (measured on job 230857:
# `train/rows_per_step_avg 256.0`). A SIXTEENTH of the batch, and therefore a
# different recipe in the one place a finetune is most sensitive.
#
# That cost two full waves, though the LR was never the reason -- see the
# learning-rate section below, which has been rewritten now that the cause is
# known.
#
# WHY MDS WAS USED AT ALL, AND WHY THAT REASON IS GONE. backend=days raises
# when train_data_fraction != 1.0 (pretrain.py), and the label budget used to
# BE train_data_fraction. The WSD rewrite moved the budget axis to the
# checkpoint ladder and stopped passing it, so the constraint lapsed -- but
# the sweep kept the backend, and a "DO NOT SET DAYSTORE=1" comment kept
# saying the restriction still applied. It does not. Nothing here passes
# train_data_fraction.
#
# ── THE LEARNING RATE WAS NEVER THE PROBLEM ───────────────────────────────
#
# Three waves were spent on this axis -- 2e-4, 1e-5, 1e-6 -- and the axis was
# the wrong one. MEASURED on wave 047986 (days, 4,096 rows/step, blr 1e-5, all
# 93 runs), against each run's own ridge init:
#
#   cos(w_init, w_final) = 1.000000      the head direction never re-aimed
#   ||w||  0.99837 observed              vs 0.998362 predicted by WEIGHT DECAY
#                                        ALONE -- agreeing to 5 s.f.
#
# The head's entire net movement was decay. The gradient did nothing, and the
# MLP branch reached 1e-4 of the skip, so it could not compensate either.
#
# THE CAUSE WAS head_init_scale='unit' FOLDING THE RESCALE INTO THE WEIGHT.
# That rescale is 1/pred_std (20-95x here), which put the skip's RMS at 3.78
# against the backbone's 0.136. Adam steps a parameter by ~lr regardless of
# its magnitude, so the head's RELATIVE step was 28x smaller than the
# encoder's -- against a supervised RegressionHead that sits at 0.43x its own
# backbone and therefore adapts slightly FASTER than it. The two recipes were
# ~65x apart in the one ratio that decides whether a readout can follow its
# encoder, which dwarfed the 20x LR gap everyone was staring at.
#
# So the encoder drifted 0.0044 out from under a frozen readout -> the IC
# valley; then the encoder slowly learned to serve that fixed linear
# functional -> the recovery. And BOTH LR directions were doomed, which is why
# neither worked: lowering it scales head and encoder down together and leaves
# the ratio untouched, so the valley MOVES (exactly what 1e-6 did); raising it
# to 2e-4 drives the encoder 20x harder against the same frozen head (exactly
# the feature erasure that was seen).
#
# THE VALLEY IS IN THE HEAD, NOT THE ENCODER, and the pilot (commit 653736,
# POST_TRAIN_PROBE=1) is the first wave able to tell them apart. A fresh ridge
# on the finetuned encoder is flat across the ladder and then rises, ending
# ABOVE the frozen full-pool probe at 5e-5 and 2e-4 on all three tasks; the
# worst rung-over-rung step is -0.0039, mostly inside 1 SE. The same
# checkpoints' HEAD readout still falls 0.0151 -> 0.0012 (return, 2e-4).
# Finetuning is not damaging the representation and never was -- head-only
# scoring simply could not distinguish "encoder broke" from "readout fell out
# of alignment", which is why three waves were read the wrong way.
# DO NOT RUN THIS SWEEP WITH POST_TRAIN_PROBE=0.
#
# The fix is in market_jepa/eval/heads.py: `gain` is a NON-TRAINABLE buffer,
# the weights stay at the ridge's own magnitude (now 0.6-3.1x the backbone),
# and tests/test_ssl_finetune.py pins that so it cannot regress. Only with
# that in place is an LR comparison meaningful at all.
#
# ── NOTHING ELSE IS PINNED THAT IS A DEFAULT ───────────────────────────────
#
# Per scripts/sweeps/README.md. The objective, batch size and epoch count are
# INHERITED from SupervisedModeConfig, because the point of the comparison is
# that the finetune and the from-scratch supervised arm differ in the
# INITIALIZATION and in nothing else. Verified 2026-09-16 as the live defaults
# rather than assumed:
#
#   loss_fn=pairwise                  SupervisedModeConfig.loss_fn
#   12 epochs, eff. bs 256            SupervisedModeConfig.training_overrides
#   (backend is NOT inherited: DAYSTORE=1 selects days, see above)
#   xs_target / xs_eval_target=uniform DatasetConfig (both)
#   train_span_months=6               DatasetConfig
#   training.live_eval=false          TrainingConfig.live_eval
#   training.seed=42                  TrainingConfig.seed
#
# live_eval IS worth a word: the retired ssl_finetune.sh pinned it false, and
# the README's worked example is about it being pinned in one sweep and
# inherited in another. It has since become the default, so pinning it now is
# the restatement the rule forbids -- tests/test_default_recipe.py enforces
# that, and caught this file doing it.
#
# SET DAYSTORE=1 ON THIS SWEEP -- see the batch section above. Without it the
# run silently uses a sixteenth of the base recipe's batch.
#
# Launch (the manifest's months, as TRAIN_END months):
#   DAYSTORE=1 MONTHS_OVERRIDE="$(scripts/sweeps/ssl_finetune/ssl_base_lib.sh --train-months)" \
#     ./scripts/pythia/run_all_months_sweep.sh sweeps/ssl_finetune/ssl_finetune_breadth.sh

source "$(dirname "${BASH_SOURCE[0]}")/ssl_base_lib.sh"

SWEEP_NAME="ssl-ft-breadth-$(ssl_base_arm_tag)"

SWEEP_TASKS="${SWEEP_TASKS:-return_900 volatility_change_900 spread_change_900}"

# ── ONE RUN PER (MONTH, TASK). THE LADDER IS CHECKPOINTS ───────────────────
#
# Under a cosine schedule the LR differs at every step, so a checkpoint taken
# part-way through is not a model the schedule ever intended to stop at, and a
# budget curve needs one RUN per budget: 6 budgets x 3 tasks x 31 months = 558
# runs, each scored 5 times = 2,790 evals. That wave was CPU-bound in scoring
# with the GPUs at 0-55% and an ETA of 17-19 hours.
#
# optimizer.lr_schedule=wsd holds the LR CONSTANT through the stable phase, so
# every checkpoint inside it was trained under the same optimizer state and
# they are comparable to each other. One run's checkpoint ladder is therefore
# the budget ladder: 93 runs, 465 evals.
#
# THE FRACTIONS ARE OF THE FULL-POOL RUN, so they are label budgets directly:
# the run sees the whole six-month pool over 12 passes, and a checkpoint at
# fraction f has been trained on f of that. Chosen to line up with FIT_SIZES
# in scripts/eval/probe_fit_size.py once converted through n_rows_pool.
#
# 0.9 IS THE LAST LADDER RUNG, NOT 1.0: with decay_frac=0.1 (set below) the
# stable phase ends at 0.9, and a checkpoint after that is mid-decay and not
# comparable to the rest of the ladder -- which is the `stable` cutoff the awk
# below applies. The run root is the decayed endpoint and is the one
# properly-annealed point on the curve.
# TARGET ROW COUNTS, not fractions. FIT_SIZES of scripts/eval/probe_fit_size.py,
# so a checkpoint lands on the same x as a point of the frozen-probe curve.
# The fraction each one needs is computed PER MONTH in sweep_train_args,
# because a run's length moves with its month's pool.
# 524288 IS NOT A FIT_SIZE, and is here because the others run out. A run
# consumes ~12 x its month's ticker-days, and the smallest month affords only
# 546,953 observations inside the stable phase -- so 786,432 is reachable by
# 13 of 31 months and the figure, which draws only rungs EVERY month reached,
# would stop at 393,216 while the probe curve runs to 1,152,000. 2^19 is the
# largest round number the shortest month clears. 786,432 is kept for the
# months that can: it costs them nothing and the collector records what each
# checkpoint actually consumed.
# THE LADDER MUST SPAN THE RUN, NOT ITS FIRST FIVE PERCENT. On the day
# backend a step is 4,096 rows, so the old mds ladder (top rung 786,432)
# ended at step 191 of 3,696 -- five of its six rungs inside the 184-step
# warmup, where the LR is still ramping and a checkpoint is the init barely
# perturbed. The whole stable phase, which is what WSD exists to make
# comparable, went unsampled.
#
# These are probe_fit_size.py's FIT_SIZES (so a rung lands on the frozen
# curve's own x) extended by powers of two to the end of the stable phase.
# 2048 is dropped: it is half a step here and would silently report 4,096.
# 8,388,608 is the largest that the SHORTEST month still reaches inside its
# stable phase (pool 1,814,567 -> f=0.86 < 0.9).
SWEEP_TARGET_ROWS="${SWEEP_TARGET_ROWS:-8192 32768 131072 393216 786432 1152000 2097152 4194304 8388608}"

# See the LR block above: NOT the mode default, deliberately.
FT_BLR="${FT_BLR:-1e-5}"

# THE LR IS IN THE RUN NAME BECAUSE skip_if_done KEYS ON IT. That probe asks
# W&B whether <project>/<run_name> already finished, and the project name
# carries only the sweep and the COMMIT -- so two launches that differ only in
# FT_BLR used to collide, and the second one would be skipped as "already
# done" while looking like a successful submission. A wave comparing learning
# rates is exactly the case that breaks.
SWEEP_VALUES=()
for _T in ${SWEEP_TASKS}; do
    SWEEP_VALUES+=("${_T}")
done

sweep_train_args() {
    local TASK="$1"

    # The manifest is keyed by EVAL month; the bundle's YM is the last TRAINING
    # month, six months after TRAIN_START. Using the wrong one here would load
    # a checkpoint from a different span and say nothing about it.
    local YM="${EVAL_START:0:7}"
    local RUN_ID
    # The staged tree is keyed by RUN ID, not by month -- see ssl_base_lib.sh
    # for the wave that cost. Resolved from the same manifest column
    # sweep_stage_extra stages from, so the path trained on is the path staged.
    RUN_ID=$(ssl_base_run_id "${YM}")
    if [ -z "${RUN_ID}" ]; then
        echo "ERROR: no run id for eval month ${YM} in the base manifest" >&2
        return 1
    fi

    # THE LADDER, in this month's own units. A full run consumes
    # max_train_steps x effective_batch observations; the trainer records the
    # count it actually reached as obs_seen, so the collector never re-derives
    # this. Here we only need the FRACTION at which each target lands.
    local POOL ROWS_TOTAL FRACS
    POOL=$(ssl_base_pool_rows "${YM}")
    if [ -z "${POOL}" ]; then
        echo "ERROR: no pool size for eval month ${YM} in the base manifest" >&2
        return 1
    fi
    # obs per run = 12 passes x ticker-days, and a ticker-day is 35.83 a36
    # rows (measured). Approximate ON PURPOSE: it only has to place the
    # checkpoints, and obs_seen records where they actually landed.
    # obs_seen COUNTS CELLS ON THE DAY BACKEND AND ROWS ON MDS.
    # obs_seen = steps x batch x accum, and on days a batch entry is a CELL of
    # n_stocks=16 labelled tickers (schemas.py: "per_device_train_batch_size
    # is therefore CELLS per step"), so one optimizer step is 256 cells = 4,096
    # rows. On mds the same nominal batch is 16 cells = 256 rows. The sweep's
    # axis is ROWS, so a target has to be divided by that factor before it can
    # be expressed as a fraction of the run. Getting this wrong puts every rung
    # 16x too far right and the figure draws nothing, because no rung then
    # snaps to a nominal budget.
    local K=1
    [ "${DAYSTORE:-0}" = "1" ] && K=${FT_ROWS_PER_OBS:-16}
    FRACS=$(awk -v pool="${POOL}" -v targets="${SWEEP_TARGET_ROWS}" -v stable="0.9" -v k="${K}" '''BEGIN{
        total = 12.0 * (pool / 35.83)
        n = split(targets, t, " "); out = ""
        for (i = 1; i <= n; i++) {
            f = (t[i] / k) / total
            if (f > stable) continue          # past the stable phase: not comparable
            # NO FLOOR. It used to clamp at 0.001, which on mds was below one
            # step anyway but on the day backend is FOUR steps: a step there is
            # 4,096 rows, so the 8,192 rung is step 2 and the floor pushed it
            # to 16,384. resolve_save_steps already rounds and clamps into
            # [1, max_train_steps], so a fraction under one step lands on
            # step 1 rather than vanishing.
            out = out (out == "" ? "" : ",") sprintf("%.4g", f)
        }
        print "[" out "]"
    }''')
    if [ "${FRACS}" = "[]" ]; then
        echo "ERROR: no target rows fit inside the stable phase for ${YM}" >&2
        return 1
    fi

    # NO train_data_fraction: the budget axis is the checkpoint ladder now, so
    # the run trains on the whole pool exactly once.
    echo "mode=supervised \
        mode.task=${TASK} \
        mode.init_backbone_from=${SSL_BASE_DIR}/${RUN_ID} \
        mode.init_head_from=${SSL_HEAD_DIR}/${RUN_ID} \
        optimizer.blr=${FT_BLR} \
        optimizer.lr_schedule=wsd \
        checkpoint.save_fractions=${FRACS} \
        optimizer.decay_frac=0.1 \
        backbone=transformer \
        ${EXTRA_TRAIN_ARGS:-} \
        wandb.project=${SWEEP_NAME} \
        wandb.group=${SWEEP_NAME} \
        wandb.run_name=ft_${TASK}_wsd_blr${FT_BLR}"
}
