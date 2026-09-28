#!/bin/bash
# sweeps/ssl_6mo_ts2vec_lr.sh — the ts2vec arm of the ssl-6mo wave, at an LR
# that does not diverge.
#
# WHY THIS EXISTS RATHER THAN A LINE IN ssl_lejepa_all.sh. That file emits
# every SSL arm at its mode default, and TS2VecModeConfig's default is
# training_overrides.blr=1e-3 -- the paper's LR, a one-month/300-pass number
# worn over a six-month/12-pass recipe. It is the only arm of the nine that
# needs its LR named, and naming it there would put one arm's tuning inside a
# file whose whole point is that the mode IS the arm.
#
# WHAT THE HOLDOUT-2 LADDER SHOWED (sweeps/ts2vec_blr.sh, 2026-09-15). On the
# five off-panel months, at 1e-3 and at 5e-4, ts2vec trains four epochs of
# twelve and then dies:
#
#   WARNING - Non-finite loss at step 2796, skipping optimizer step
#   RuntimeError: 50 consecutive steps had no trainable batch
#
# Both rungs cleared 2009-06, 2011-12, 2015-06 and 2016-03 and both died on
# 2020-10..2021-03. 2e-4 cleared that month with real IC (spread +0.0754,
# vol +0.0163, return +0.0093). The mechanism is the official loss port:
# hierarchical contrast over RAW DOT PRODUCTS of unnormalised
# representations, no temperature, no normalisation -- the logits grow with
# the representation norm and nothing in the objective pushes back. Not
# precision (fp32 throughout, max_grad_norm=1.0) and not the pool change
# (bit-identical loss), both checked first.
#
# 1e-4, NOT THE 2e-4 THAT WAS MEASURED (user, 2026-09-15). 2e-4 is the
# lowest rung the ladder actually reached and it survived exactly one month;
# 1e-4 buys margin on a ladder whose failures are all at the top, at the cost
# of being extrapolated rather than measured. The call was to stop tuning and
# take the margin.
#
# A NEW NAMESPACE, DELIBERATELY. The 25 ts2vec cells already on disk under
# ssl-6mo-ts2vec-206b47-* trained at 1e-3. Filling only the six holes at 1e-4
# would leave the arm a mixture of two LRs, with the lower one on exactly the
# months that are hardest -- a systematic difference where it does the most
# damage, and the same shape of error as the architecture-mixed VD tree. So
# this resubmits ALL of ts2vec and lets the commit hash in the project prefix
# separate the generations. Do NOT pass --commit-hash 206b47.
#
# Usage (train-only, like the rest of the ssl-6mo wave, and parked behind
# other work):
#
#   POST_TRAIN_IC_EVAL=0 SBATCH_EXTRA="--nice=10000" \
#       ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/ssl_6mo_ts2vec_lr.sh
#
# THE PATH IS RELATIVE TO scripts/ (resolve_sweep_list prepends it).

SWEEP_MONTH_SET="sampled32"

SWEEP_NAME="ssl-6mo-ts2vec"

WANDB_GROUP="${WANDB_GROUP:-ssl-lejepa-6mo}"

SWEEP_VALUES=(
    "ts2vec"
)

# ts2vec is not a cells mode: it reads views, not cross-sections, so the
# day-major store has nothing to serve it. Same guard, same reason, as the
# wave this belongs to.
if [ "${DAYSTORE:-0}" = "1" ]; then
    echo "ERROR: DAYSTORE=1 cannot run this sweep -- ts2vec is not a" >&2
    echo "       cells mode and the day store has nothing to serve it." >&2
    return 1 2>/dev/null || exit 1
fi

sweep_train_args() {
    local ARM="$1"
    # Run name = the YYYY-MM the TRAINING SPAN ENDS ON, matching
    # ssl_lejepa_all.sh so a month is addressed the same way in both.
    local YM="${TRAIN_END:0:7}"
    case "${ARM}" in
        ts2vec) : ;;
        *) echo "ERROR: unknown arm '${ARM}'" >&2; return 1 ;;
    esac
    # optimizer.blr beats the mode's training_overrides.blr (pretrain.py:927
    # precedence: cfg.optimizer.blr > backbone blr > overrides.blr), which is
    # what makes this a one-line override rather than a schema edit.
    echo "mode=ts2vec \
        backbone=transformer \
        optimizer.blr=1e-4 \
        wandb.project=ssl-6mo-ts2vec \
        wandb.group=${WANDB_GROUP} \
        wandb.run_name=${YM}_ts2vec"
}
