#!/bin/bash
# sweeps/ssl_lejepa_all.sh — EVERY unsupervised arm of the paper, retrained on
# the current recipe. Five LeJEPA pairings + nine SSL baselines = 14 arms, one
# bundled SLURM job per month.
#
# WHY THIS EXISTS. The reported unsupervised campaign
# (`ssl-ic-final-222d99-*`, `lejepa-*-lambda-*`) trained on ONE CALENDAR MONTH
# -- 300 passes over it -- while the supervised specialists now train 12 passes
# over a SIX-MONTH trailing span. Re-scoring the old checkpoints cannot fix
# that: it would still put a 1-month encoder beside a 6-month one and call the
# gap a method difference. They are archived
# (plots/latent_eval/_archive/pre_meanpool_readout/README.md) and this sweep
# replaces them.
#
# ── THE BUDGET IS IDENTICAL TO THE SUPERVISED ARMS ──────────────────────────
#
# 12 passes over DatasetConfig.train_span_months=6, for all 14 arms, and NONE
# of it is passed from here. Verified 2026-09-14 against the trainer's own
# precedence (cfg.training.num_epochs -> mode.training_overrides.num_epochs ->
# TrainingConfig.fallback_num_epochs), because the two halves get there by
# DIFFERENT ROUTES and only one of them is visible in a mode config:
#
#   num_epochs=12 from the MODE      cpc, ijepa, lejepa, mae   (+ supervised)
#   num_epochs=None -> FALLBACK=12   byol, cost, dino, tfc, ts2vec, timemae
#
# So six arms have no epoch setting of their own and inherit
# `TrainingConfig.fallback_num_epochs = 12` -- whose comment states the intent
# exactly: "TWELVE PASSES OVER THE SPAN, for every method (2026-09-13). The
# budget is a property of the COMPARISON, not of a method." If that fallback
# ever moves, those six move with it and these four do not; check both before
# pooling a wave that straddles such a commit.
#
# BATCH SIZE IS NOT EQUALIZED AND SHOULD NOT BE. The arms see the same data the
# same number of times; how many optimizer steps that takes is a method
# property (lejepa 256, cpc/ijepa/mae 1024 micro / 2048 effective, the other
# six at fallback_train_batch_size=128). Equal passes, not equal steps, is what
# makes an IC difference a representation difference.
#
# ── NOTHING ELSE IS PINNED ──────────────────────────────────────────────────
#
# Per scripts/sweeps/README.md, and the deleted sweeps this replaces
# (lejepa_samestock_lambda.sh, lejepa_k2_lambda.sh, lejepa_noisefix_lambda.sh)
# pinned a great deal that is now schemas.py's. Each was VERIFIED as the
# current default before being dropped, not assumed -- that is the trap the
# README names:
#
#   backbone.state_token=false        TransformerBackboneConfig.state_token
#   dataset.info_norm_stats=true      DatasetConfig.info_norm_stats
#   dataset.info_window=true          DatasetConfig.info_window
#   dataset.n_pairs_per_obs=1         DatasetConfig.n_pairs_per_obs
#   dataset.risk_factor_tickers=[]    DatasetConfig.risk_factor_tickers
#   dataset.xs_eval_target=uniform    DatasetConfig.xs_eval_target
#   optimizer.blr=6e-5                LeJEPAModeConfig.training_overrides.blr
#   noise_sigma / warp_strength 0.75  AugmentationConfig defaults
#
# ONE OF THEM CHANGED MEANING AND THAT IS THE POINT. The old LeJEPA sweeps
# pinned `dataset.xs_target=rank`; the default is now `uniform`, which is what
# the supervised specialists train and score on. Dropping the pin moves these
# arms onto the same quantity -- deliberately. An arm carried over verbatim
# would have trained against a different target than the models it is compared
# to.
#
# ── THE LAMBDAS ARE INHERITED AND THAT IS A KNOWN WEAKNESS ──────────────────
#
# Each LeJEPA arm keeps the lambda that won its holdout-2 sweep (2026-08-27),
# named below. THOSE WINNERS WERE SELECTED UNDER THE OLD RECIPE -- one month,
# 300 passes, cls pooling, xs_target=rank -- and none of those four things is
# true here. The lambda balances the isotropy term against the invariance term,
# and the invariance term's scale moves with the pairing AND with how long you
# train, so there is no reason to expect the same optimum on a 6-month, 12-pass,
# uniform-target run.
#
# Carried anyway because re-running a 5-pairing x 7-lambda holdout-2 sweep is a
# larger job than this one, and a stale-but-named lambda is recoverable while an
# unnamed guess is not. If a LeJEPA arm underperforms its archived self, THIS IS
# THE FIRST THING TO SUSPECT, not the readout or the span.
#
# ── PROJECT NAMES SAY `6mo` ON PURPOSE ──────────────────────────────────────
#
# The archived generation is `ssl-ic-final-*` / `lejepa-*-lambda-*`. These are
# `ssl-6mo-*` / `lejepa-6mo-*` so no glob, manifest or figure can ever resolve
# one generation while claiming the other -- which is exactly the confusion
# that cost a day on 2026-09-14.
#
# ── TRAIN ONLY. NO EVAL, NO PROBE. ─────────────────────────────────────────
#
# This wave produces CHECKPOINTS and nothing else: the probe-fit protocol is
# still being decided (probe_fit_breadth.png says the ridge is sample-starved
# at the current one-month / 36-anchor pool, and the fix is a cache rebuild
# either way), so scoring now would spend hours per month producing numbers
# under a protocol that is about to change. Submit with POST_TRAIN_IC_EVAL=0;
# the guard below refuses otherwise. Score the checkpoints later, once.
#
# Nothing here needs the panel cache, for the same reason.
#
# ── THESE ARMS CANNOT USE THE DAY STORE. THIS IS NOT A SPEED CHOICE. ────────
#
# DAYSTORE=1 makes train_bundle_body.sh pass `dataset.backend=days`
# UNCONDITIONALLY, and pretrain.py then raises for every arm in this file:
#
#     if not (is_supervised or is_multi_supervised) or not (
#             augmentations and augmentations[0].get("name") == "cross_stock"):
#         raise ValueError(
#             "dataset.backend=days serves supervised cross_stock cells only")
#
# The day store holds SUPERVISED CELLS with precomputed targets. It does not
# emit augmented multi-crop views, which is the entire input to a LeJEPA or an
# SSL objective -- so there is no day-store path for these fourteen arms to be
# slow or fast on. All fourteen would ValueError in the first seconds.
#
# The "~1.65x slower on MDS" figure belongs to the SUPERVISED sweep, where both
# backends exist and one is chosen. Here MDS is the only backend that serves
# the data at all, so its throughput is the cost of running these arms, not a
# setting anyone picked. Making DAYSTORE=1 a repo-wide default would turn every
# non-supervised run in the tree into that ValueError.
#
# The guard below therefore REFUSES DAYSTORE=1 rather than requiring it.
#
# Usage -- TWO WAVES, LeJEPA FIRST. All 14 arms in one job needs ~14 arm-runs
# inside a 3-day TimeLimit, which is the standard_hopper maximum and cannot be
# extended; splitting keeps each job well inside it and puts the arms that
# matter most at the front of the queue. The sweep file names the job after the
# subset, so the two waves are told apart in squeue.
#
#   POST_TRAIN_IC_EVAL=0 SWEEP_VALUES_OVERRIDE="rrc,warp,noise,k2,k2ind" \
#       ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/ssl_lejepa_all.sh
#
#   POST_TRAIN_IC_EVAL=0 \
#   SWEEP_VALUES_OVERRIDE="byol,cost,cpc,dino,ijepa,mae,tfc,timemae,ts2vec" \
#       ./scripts/pythia/run_all_months_sweep.sh \
#       --partition standard_hopper sweeps/ssl_lejepa_all.sh
#
# THE PATH IS RELATIVE TO scripts/ (resolve_sweep_list prepends it), so
# "sweeps/..." and not "scripts/sweeps/...".
#
# POST_TRAIN_IC_EVAL=0 IS NOW ENOUGH TO STAGE NO PANELS. It used to leave the
# ~22 GB probe-fit panel being rsynced per job and left require_panel_cache
# able to exit 1 over a panel nothing would read; stage_data.sh gates both on
# it since 2026-09-14 (tests/test_train_only_stages_no_panels.py). Passing
# POST_TRAIN_PROBE=0 as well is redundant, not wrong.
#
#   SWEEP_VALUES_OVERRIDE='rrc,k2ind'   subset the arms ON THE NODE -- this is
#                                       the one run_all_months_sweep.sh
#                                       forwards through ssh.
#   ARM_FILTER='rrc k2ind byol'         LOCAL ONLY: it never crosses the ssh,
#                                       so a pythia job launched with it still
#                                       runs all 14. Use it for local dry runs.

SWEEP_MONTH_SET="sampled32"

SWEEP_NAME="ssl-lejepa-all"

# LeJEPA arms vs SSL-baseline arms, named so a WAVE can be one or the other.
# 14 arms run sequentially in ONE 3-day job (the partition maximum, so it
# cannot be extended), and a timeout loses every arm that has not started, so
# splitting the wave is how the tail arms get run at all. Each arm's
# checkpoint rsyncs to bll01 inside the loop, so a timeout only costs the
# in-flight arm and the un-started ones.
LEJEPA_ARMS="rrc warp noise k2 k2ind"

# SWEEP_VALUES_OVERRIDE is the knob run_all_months_sweep.sh actually forwards
# to the node (ARM_FILTER below is LOCAL ONLY -- it never crosses the ssh, so
# a job launched with it would still run all 14). When a wave carries a pure
# subset, rename the job so two concurrent waves are told apart in squeue.
# COSMETIC: wandb projects come from sweep_train_args, not from SWEEP_NAME.
if [ -n "${SWEEP_VALUES_OVERRIDE:-}" ]; then
    _N_LJ=0 _N_SSL=0
    for _V in ${SWEEP_VALUES_OVERRIDE//,/ }; do
        case " ${LEJEPA_ARMS} " in
            *" ${_V} "*) _N_LJ=$(( _N_LJ + 1 )) ;;
            *)           _N_SSL=$(( _N_SSL + 1 )) ;;
        esac
    done
    if   [ "${_N_SSL}" = "0" ] && [ "${_N_LJ}" -gt 0 ]; then SWEEP_NAME="lejepa-6mo"
    elif [ "${_N_LJ}"  = "0" ] && [ "${_N_SSL}" -gt 0 ]; then SWEEP_NAME="ssl-6mo"
    fi
    unset _N_LJ _N_SSL _V
fi

# 14 arms x 32 months. Left at DEFAULT PRIORITY, unlike
# supervised_specialists.sh's --nice=10000: that sweep was explicitly
# non-urgent, this one is the blocking dependency for every unsupervised row of
# the paper. Pass SBATCH_EXTRA="--nice=10000" to park it behind other work.

SWEEP_VALUES=(
    # LeJEPA pairings — the augmentation IS the arm; lambda is its holdout-2 winner
    "rrc"
    "warp"
    "noise"
    "k2"
    "k2ind"
    # SSL baselines — mode name only; every knob is the mode's own default
    "byol"
    "cost"
    "cpc"
    "dino"
    "ijepa"
    "mae"
    "tfc"
    "timemae"
    "ts2vec"
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

# The gaussian_noise arm is only meaningful with the information-token mask in
# place; without it the noise draw covers the info channels and the arm
# reproduces the OLD behaviour under a name that claims otherwise. Guard
# inherited from the deleted lejepa_noisefix_lambda.sh.
_NOISEFIX_SRC="${LOCAL_REPO:-$(git rev-parse --show-toplevel)}/market_jepa/training/streaming_dataset.py"
if ! grep -q 'noise_rows\[i0 : i0 + self._n_info_features\] = False' "${_NOISEFIX_SRC}"; then
    echo "ERROR: the gaussian_noise information-token fix is NOT in this worktree." >&2
    echo "       Verify: uv run pytest tests/test_time_warp_noise.py -k information" >&2
    return 1 2>/dev/null || exit 1
fi
unset _NOISEFIX_SRC

# THE GUARDS. Both conditions are read by the launcher from the SUBMITTING
# shell, and lib/sweep_list.sh sources this file (in a subshell) at resolve
# time -- so these fire before a single job is submitted rather than 32 jobs
# later, which is the whole point.
if [ "${DAYSTORE:-0}" = "1" ]; then
    echo "ERROR: DAYSTORE=1 cannot run this sweep. dataset.backend=days serves" >&2
    echo "       supervised cross_stock cells only; every LeJEPA and SSL arm" >&2
    echo "       here raises ValueError in pretrain.py within seconds." >&2
    echo "       Submit WITHOUT DAYSTORE (MDS is the only backend for these)." >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${POST_TRAIN_IC_EVAL:-1}" != "0" ]; then
    echo "ERROR: this wave is TRAIN ONLY -- the probe-fit protocol is still" >&2
    echo "       undecided, so scoring now burns hours per month on numbers" >&2
    echo "       that a protocol change invalidates. Submit with:" >&2
    echo "  POST_TRAIN_IC_EVAL=0 ./scripts/pythia/run_all_months_sweep.sh \\" >&2
    echo "      --partition standard_hopper scripts/sweeps/ssl_lejepa_all.sh" >&2
    return 1 2>/dev/null || exit 1
fi

WANDB_GROUP="${WANDB_GROUP:-ssl-lejepa-6mo}"

sweep_train_args() {
    local ARM="$1"
    # Run name = YYYY-MM the TRAINING SPAN ENDS ON (TRAIN_END), matching
    # supervised_specialists.sh. TRAIN_START moves with the span; TRAIN_END
    # does not, and each arm has its own project so YYYY-MM is unique in it.
    local YM="${TRAIN_END:0:7}"
    local PROJECT RUN AUG=""

    case "${ARM}" in
        # ── LeJEPA: mode.lamb has no default (LeJEPAModeConfig.lamb=None), so
        #    it must be passed; the augmentation name defines the arm.
        rrc)
            PROJECT="lejepa-6mo-rrc";   RUN="${YM}_rrc_lamb0.05"
            AUG="mode.lamb=0.05 dataset.augmentations.0.name=random_resized_crop" ;;
        warp)
            PROJECT="lejepa-6mo-warp";  RUN="${YM}_warp_lamb0.001"
            AUG="mode.lamb=0.001 dataset.augmentations.0.name=time_warp" ;;
        noise)
            PROJECT="lejepa-6mo-noise"; RUN="${YM}_noise_lamb0.3"
            AUG="mode.lamb=0.3 dataset.augmentations.0.name=gaussian_noise" ;;
        k2)
            PROJECT="lejepa-6mo-k2";    RUN="${YM}_k2_lamb0.2"
            # industry_table MUST be cleared: DatasetConfig.augmentations'
            # default INSTANCE is cross_stock(industry_table=STANDARD), so
            # naming only the pairing leaves this arm industry-restricted and
            # identical to k2ind but for lamb.
            AUG="mode.lamb=0.2 dataset.augmentations.0.name=cross_stock \
                 dataset.augmentations.0.n_stocks=2 \
                 dataset.augmentations.0.industry_table=null" ;;
        k2ind)
            # The ONLY difference from k2: pairs are drawn within an FF49
            # industry. n_stocks=2 is already the AugmentationConfig default
            # but is named in both arms so the pairing reads off the file.
            PROJECT="lejepa-6mo-k2ind"; RUN="${YM}_k2ind_lamb0.1"
            AUG="mode.lamb=0.1 dataset.augmentations.0.name=cross_stock \
                 dataset.augmentations.0.n_stocks=2 \
                 dataset.augmentations.0.industry_table=data/industry_map.parquet" ;;
        # ── SSL baselines: the mode is the whole arm. Each carries its own
        #    dataset_overrides (views / augmentation) and training_overrides,
        #    so naming anything here would only create a second place to drift.
        #    DINO and BYOL NOW DECLARE dataset_overrides.name="time_warp"
        #    (set 2026-09-14), so they pair the way the warp arm above does and
        #    the three joint-embedding objectives differ only in the objective.
        #    Until then they INHERITED the k2ind default on purpose -- that was
        #    the repo rule for every uses_multi_view mode. Neither pairing gives
        #    them local crops; only random_resized_crop would.
        byol|cost|cpc|dino|ijepa|mae|tfc|timemae|ts2vec)
            PROJECT="ssl-6mo-${ARM}";   RUN="${YM}_${ARM}" ;;
        *)
            echo "ERROR: unknown arm '${ARM}'" >&2
            return 1 ;;
    esac

    local MODE="${ARM}"
    case "${ARM}" in rrc|warp|noise|k2|k2ind) MODE="lejepa" ;; esac

    echo "mode=${MODE} \
        backbone=transformer \
        ${AUG} \
        wandb.project=${PROJECT} \
        wandb.group=${WANDB_GROUP} \
        wandb.run_name=${RUN}"
}
