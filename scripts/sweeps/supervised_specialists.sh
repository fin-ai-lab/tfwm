#!/bin/bash
# sweeps/supervised_specialists.sh — Full-history supervised baselines, one
# specialist per task (renamed from full_data_supervised.sh).
#
# ViT trained on the DatasetConfig.train_span_months span ENDING at each
# bundle month and scored by rank IC on the synchronized cross-section of the
# month AFTER it. One bundled SLURM job per month; all selected tasks run
# sequentially on that job's GPU so the mosaic is staged once.
#
#   return_900 / volatility_change_900 / spread_change_900
#       pairwise (LTR) on the uniform target, blr 2e-4, 256 CELLS per
#       optimizer step (16-cell micro-batches x 16 grad-accum, 16 stocks a
#       cell), 12 passes over the six-month span
#
# THOSE NUMBERS ARE schemas.py's AND ARE NOT PASSED FROM HERE --
# SupervisedModeConfig.training_overrides (blr 2e-4,
# per_device_train_batch_size=16, effective_batch_size=256, num_epochs=12),
# SupervisedModeConfig.dataset_overrides (cross_stock, n_stocks=16) and
# DatasetConfig.train_span_months=6. They are restated only so the header says
# what this sweep trains; if they ever disagree with schemas.py, schemas.py is
# the recipe and this comment is the bug. The measurements behind them
# (2026-09-13, ten holdout months) are in SupervisedModeConfig.
#
# This header said "one calendar month at a time, blr 1e-5 @ bs256, 100
# epochs" until 2026-09-13 -- the pre-cell recipe, from before the span and
# before the within-cell loss ran at all.
#
# PAIRWISE ON THE UNIFORM SCORE. The head is a scalar RegressionHead trained
# with a listwise-ranking loss against the cross-sectional uniform target — the
# same quantity the probe is scored on. Selected by
# sweeps/supervised_loss_ablation.sh, where the two pointwise losses (mse,
# smooth_l1) are minimized by a near-constant prediction on a standardized
# cross-section and are in that table as the thing to beat, not as candidates.
#
# THE BINNED FAMILY IS GONE (2026-09-07). This sweep used to train a k-way head
# over equal-count quantile bins with soft labels and an expected-bin penalty,
# at blr 1e-4 quoted at bs128. cross_entropy, expected_bin_mse,
# expected_bin_mae and the hybrid were all removed, and the knobs they needed
# (n_bins, tau and its anneal, the penalty weight) went with them.
#
# WHY ONE RECIPE FOR ALL THREE TASKS, where the retired regression arm had
# three. The per-task blr split existed because MSE on a z-score has a target
# scale that differs by task. A ranking loss on a uniform score removes that:
# the label is a quantile in [0,1] whatever the target is, so the objective's
# scale no longer moves with the task. If a per-task blr turns out to matter it
# should be measured, not inherited from the regression HPO.
#
# BOTH READOUTS, ALL THREE TASKS. slurm_train_bundle.sh runs
# scripts/generic/post_train_ic_eval.py after each training, and that scores
#   xs_ic/<task>        ridge probe — fit for ALL 18 target columns, so a
#                       month's return-trained run also reports a probe IC on
#                       volatility_change and spread_change
#   xs_ic/head:<task>   the head's own scalar prediction, for the task it
#                       trained on. It read "expected bin" until 2026-09-13,
#                       left over from the retired binned family; the head is
#                       a scalar RegressionHead ranked by the pairwise loss.
#   xs_auc/<task>, xs_auc/head:<task>   the same pair as macro one-vs-rest AUC
# So across the three runs of one month, every task has both a probe and a
# head number, and no extra scoring pass is needed to get them.
#
# n_pairs_per_obs is left at 1. It was set to 2 for dataloader throughput, but
# the two crops come from the same ticker-day and the same session, so their
# gradients are correlated and a batch of 128 carried only 64 distinct
# observations — halved diversity against a CROSS-SECTIONAL target, for 1.6x
# wall-clock. Not a trade worth making.
#
# NO LIVE EVAL -- now schemas.py's default (TrainingConfig.live_eval=False as
# of 2026-09-09), not a line this file passes. The in-training probe is CPU-bound
# subprocess competing with the dataloader for 7 cores, and neither of its
# numbers is what gets reported — the probe fits on 4096 rows and under-reports
# IC by ~2.4x, and the pooled eval IC carries a 0.018 sampling SD. Instead
# slurm_train_bundle.sh runs scripts/generic/post_train_ic_eval.py at the end
# of the job, on the same node, and logs synchronized cross-section IC (with a
# per-cell SE) to the same wandb run under xs_ic/. The freed core goes back to
# the dataloader (machine=<cluster> num_workers 5 -> 6).
#
# THE DAY STORE IS OPT-IN AT THE LAUNCHER, and nothing here or in schemas.py
# turns it on. The settled recipe trains from the day-major store, but
# DatasetConfig.backend defaults to "mds" and both launchers forward
# DAYSTORE='${DAYSTORE:-0}', so a plain invocation trains off MDS; pass
# DAYSTORE=1 to get dataset.backend=days. It is a THROUGHPUT choice and not an
# IC one -- the two loaders differ by -0.002 +- 0.001 over nine months with no
# consistent sign, the views being equal element by element
# (tests/test_cell_dataset.py) -- so an MDS run is a valid arm, just ~1.65x
# slower (~1,900 views/s at ~60% GPU util against ~3,150 at 95%).
#
# Sourced by our cluster's run_full_data_supervised launcher (not included).
#
# TASK_FILTER (space-separated keys from {return, vol_change, spread_change})
# subsets SWEEP_VALUES — the launcher uses this for --return / --vol_change /
# --spread_change.
#
# Each task gets its own wandb project so resubmissions slot in cleanly.
#
# CAVEAT on spread_change, remeasured 2026-09-09. THIS BLOCK USED TO DESCRIBE
# THE RETIRED TARGETS (volatility_change = fwd_vol - bwd_vol, spread_change =
# spread(t+h) - spread(t)) and had the two the wrong way round for the current
# ones: it indicted volatility and exonerated spread, where the measurement
# says the reverse.
#
# All three targets became differences of two FORWARD 60 s windows on
# 2026-08-22. A leak then needs the near leg to be BOTH observable at t AND
# mean-reverting:
#
#   spread_change      near leg IS observable -- spread is a step function, so
#                      mean spread over [t, t+60) is essentially spread(t),
#                      which is in the crop -- and spread reverts to the
#                      stock's own norm. So F - B ~ (mu - B)(1 - phi), a
#                      negative multiple of a known number. A trivial
#                      -spread(t) predictor (raw dollars, off the quote) scores
#                      +0.2199 / +0.1690 / +0.1325 rank IC on 2012-12 /
#                      2018-10 / 2021-08, against this sweep's probe at
#                      +0.0799 / +0.2210 / +0.2559. It is ABOVE the model in
#                      the early sample. It survives within-ticker demeaning,
#                      so it is per-stock level reversion, not stock identity.
#
#   volatility_change  near leg is NOT observable: 60 s realized vol is mostly
#                      estimation noise. Trivial predictor +0.012..+0.034
#                      against this sweep's probe at +0.0655. Clean.
#
#   return             base leg is observable but price is a martingale (no
#                      reversion), and it is a RATIO, so the base divides out
#                      rather than subtracting. Clean.
#
# THE DIAGNOSTIC SIGNATURE is the horizon profile: -spread(t) rises 300s
# +0.1636 -> 900s +0.2199 -> 7200s +0.3914, while vol_change stays flat and
# return is noise. Real skill DECAYS with horizon; a subtracted known leg
# STRENGTHENS with it, as the far leg decorrelates and -B takes over the
# target's variance. So the contamination is worst exactly at the long horizons.
#
# NOT FIXED BY log(F/B) -- same structure, log B still known and log-spread
# still reverts -- nor by predicting F directly, which is more persistent still.
# The real fix is to residualize F on information at t before the uniform
# transform. Until then, quote spread_change against the -spread(t) baseline,
# NOT against the random-init floor: a random encoder cannot read absolute
# spread off a per-view z-scored array, so its floor sits near zero and
# launders the leak. plots/metrics/mechanical_baseline.py measures on the view
# and so understates this ~4x (+0.0644 vs +0.2199 at h=900).

# THE 32 REPORTED SWEEP MONTHS, and the launcher is checked against it:
# run_all_months_sweep.sh. Submitted through the full-history launcher by
# mistake on 2026-09-13, this trained 198 months before anyone noticed --
# a wrong month set is not an error, just a much larger experiment.
SWEEP_MONTH_SET="sampled32"

SWEEP_NAME="full-data-supervised"

# LOW PRIORITY. 203 months x 3 tasks is a large and entirely non-urgent sweep;
# at default priority it would sit in front of every one-off job submitted
# after it. --nice is subtracted straight from the multifactor priority, so a --nice
# larger than any job's priority score parks this sweep behind everything
# else of ours permanently, however long it ages. On our cluster a
# low-priority QOS would have done nothing; nice was the only lever.
#
# Backfill still runs them. sched/backfill starts a niced job whenever it fits
# a hole without delaying a higher-priority one, which is exactly the wanted
# behaviour: the sweep soaks up idle GPUs and never blocks anything.
#
# It does NOT make the sweep free. Nice reorders the queue, it does not
# discount usage — these jobs charge fairshare like any other (14-day decay
# half-life), which depresses the priority of everything submitted afterwards.
#
# Global to the wave, as SBATCH_EXTRA always is. The launcher appends the
# caller's SBATCH_EXTRA after this one and sbatch takes the last occurrence of
# a repeated flag, so SBATCH_EXTRA="--nice=0" restores normal priority.
SBATCH_EXTRA="--nice=10000 ${SBATCH_EXTRA:-}"

SWEEP_VALUES=(
    "return"
    "vol_change"
    "spread_change"
)

if [ -n "${TASK_FILTER:-}" ]; then
    _FILTERED=()
    for V in "${SWEEP_VALUES[@]}"; do
        for K in ${TASK_FILTER}; do
            if [ "${V}" = "${K}" ]; then
                _FILTERED+=("${V}")
                break
            fi
        done
    done
    if [ ${#_FILTERED[@]} -eq 0 ]; then
        echo "ERROR: TASK_FILTER='${TASK_FILTER}' matched no SWEEP_VALUES" >&2
        return 1 2>/dev/null || exit 1
    fi
    SWEEP_VALUES=("${_FILTERED[@]}")
fi

# NOTHING IS PINNED HERE -- see scripts/sweeps/README.md. This block used to
# open "EVERY KNOB IS PINNED, none inherited from schemas.py", citing the bins
# x penalty sweep, which lost a set of arms when soft_label_temperature's
# default moved 0.0 -> 0.5 mid-flight and already-running jobs picked up the
# new value silently.
#
# That is a real failure but pinning is the wrong remedy: it does not prevent
# drift, it makes drift undetectable, because the sweep goes on training the
# OLD recipe after schemas.py has moved and nothing errors. One place defines
# the recipe and it is schemas.py.
#
# The precedence that motivated the old pin still holds -- cfg.optimizer.blr
# WINS over mode.training_overrides.blr (pretrain.py:448) -- but it now argues
# the other way: leaving optimizer.blr unset is exactly what lets the mode's
# blr and batch size move together, instead of a pinned LR outliving a batch
# size that moved without it.
#
# NOT PASSED TO TRAINING. The run name is the only thing that reads BLR, and it
# is RESOLVED FROM schemas.py rather than written here: a literal went stale
# the moment the recipe moved (it said 1e-5 long after 2e-4 was measured and
# locked), which is worse than no LR in the name at all -- the name is what
# gets read years later and believed. Set BLR=... to label a deliberate
# one-off; the sweep still does not pass it to training.
BLR="${BLR:-$(uv run python -c 'from market_jepa.schemas import SupervisedModeConfig; print(f"{SupervisedModeConfig().training_overrides.blr:g}")' 2>/dev/null || echo unknown)}"

WANDB_GROUP="${WANDB_GROUP:-full-data-supervised}"

sweep_train_args() {
    local TASK_KEY="$1"
    local TASK PROJECT
    case "${TASK_KEY}" in
        return)
            TASK="return_900"
            PROJECT="supervised-full-month-return"
            ;;
        vol_change)
            TASK="volatility_change_900"
            PROJECT="supervised-full-month-vol-change"
            ;;
        spread_change)
            TASK="spread_change_900"
            PROJECT="supervised-full-month-spread-change"
            ;;
        *)
            echo "ERROR: unknown task key '${TASK_KEY}'" >&2
            return 1
            ;;
    esac

    # Run name = YYYY-MM of the month the TRAINING SPAN ENDS ON, from
    # TRAIN_END. It was TRAIN_START, which named the same month only while a
    # run trained one calendar month; the recipe is now
    # DatasetConfig.train_span_months of data BEFORE the eval month, so
    # TRAIN_START moves with the span and TRAIN_END does not. Each task has its
    # own project, so YYYY-MM still uniquely identifies a run within it.
    local YM="${TRAIN_END:0:7}"

    # TRAIN AND SCORE ON THE SAME QUANTITY. xs_target=uniform is the score
    # the pairwise loss ranks; xs_eval_target=uniform is what the probe is
    # scored against, here and in every other arm, so the specialists and the
    # SSL encoders stay comparable.
    # THE METHOD AND THE MONTH, and nothing else -- see scripts/sweeps/README.md.
    # loss_fn, blr, batch, epochs, the dataset targets and live_eval are all
    # mode=supervised's own defaults now, so restating them here would only
    # create a second place for them to disagree.
    echo "mode=supervised \
        mode.task=${TASK} \
        backbone=transformer \
        wandb.project=${PROJECT} \
        wandb.group=${WANDB_GROUP} \
        wandb.run_name=${YM}_pairwise_blr${BLR}"
}
