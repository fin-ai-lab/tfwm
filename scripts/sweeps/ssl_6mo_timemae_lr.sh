# TimeMAE at a TENTH of its default LR, over the whole six-month panel.
#
# WHY. TimeMAE is the worst row in probe_fit_table.tex by a wide margin and the
# only one NEGATIVE on return (-0.0012 / 0.0044 / 0.0081, struck through
# against the untrained floor on all three targets). A trained encoder that
# lands below a randomly initialised one is not a weak method, it is a broken
# run, and the signature -- worse than random on the hardest target, near-zero
# on the others -- is what a diverged optimiser leaves behind. ts2vec showed
# exactly this and the cause was the LR: its 1e-3 rung produced non-finite
# losses for hundreds of consecutive steps and 1e-4 trains clean.
#
# THE ARITHMETIC. TimeMAEModeConfig.training_overrides pins blr=1e-3 (the
# paper's AdamW value, market_jepa/schemas.py:1517). A tenth of that is 1e-4,
# which is what this sets -- not because 1e-4 was measured for TimeMAE, but
# because the user asked for a tenth (2026-09-16). It is the same value ts2vec
# settled on, which is weak corroboration and nothing more.
#
# optimizer.blr beats the mode's training_overrides.blr (pretrain.py:927
# precedence: cfg.optimizer.blr > backbone blr > overrides.blr), which is what
# makes this a one-line override rather than a schema edit. Recipe defaults
# live in schemas.py and sweeps do not restate them; this is a deliberate
# departure from one, so it belongs here.
#
# WHY A FRESH NAMESPACE, AND WHY THE OLD CHECKPOINTS HAD TO GO. The launcher
# suffixes wandb.project with -<commit>-<start>-<end>, so these land under
# ssl-6mo-timemae-<hash>-... while the 1e-3 wave sat under -206b47-. Two
# projects claiming one (arm, eval month) is a HARD FAILURE in
# plots/latent_eval/build_6mo_manifest.py -- it raises SystemExit and takes
# the manifest down for all eighteen arms, not just this one. That is exactly
# what the ts2vec resubmit did. So the 1e-3 checkpoints were retired to
# _superseded/ before this was submitted. Do NOT pass --commit-hash 206b47:
# pinning the old namespace would mix two learning rates inside one arm.
#
# ALL 31 MONTHS, not just the ones that look bad. An arm whose months carry
# different LRs is not one arm, and the decay and probe-fit tables both
# average within a row.  sampled32 is 32 months; 2008-02 has no six-month
# span, so 31 jobs run.
#
# Submit with (TRAIN ONLY -- this wave scores nothing, same as its siblings):
#   POST_TRAIN_IC_EVAL=0 ./pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/ssl_6mo_timemae_lr.sh
#
# THE PATH IS RELATIVE TO scripts/ (resolve_sweep_list prepends it).

SWEEP_MONTH_SET="sampled32"

SWEEP_NAME="ssl-6mo-timemae"

WANDB_GROUP="${WANDB_GROUP:-ssl-lejepa-6mo}"

SWEEP_VALUES=(
    "timemae"
)

# timemae is not a cells mode: it reads views, not cross-sections, so the
# day-major store has nothing to serve it. Same guard, same reason, as the
# wave this belongs to.
if [ "${DAYSTORE:-0}" = "1" ]; then
    echo "ERROR: DAYSTORE=1 cannot run this sweep -- timemae is not a" >&2
    echo "       cells mode and the day store has nothing to serve it." >&2
    return 1 2>/dev/null || exit 1
fi

sweep_train_args() {
    local ARM="$1"
    # Run name = the YYYY-MM the TRAINING SPAN ENDS ON, matching
    # ssl_lejepa_all.sh so a month is addressed the same way in both.
    local YM="${TRAIN_END:0:7}"
    case "${ARM}" in
        timemae) : ;;
        *) echo "ERROR: unknown arm '${ARM}'" >&2; return 1 ;;
    esac
    echo "mode=timemae \
        backbone=transformer \
        optimizer.blr=1e-4 \
        wandb.project=ssl-6mo-timemae \
        wandb.group=${WANDB_GROUP} \
        wandb.run_name=${YM}_timemae"
}
