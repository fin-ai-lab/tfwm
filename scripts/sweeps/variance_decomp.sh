#!/bin/bash
# sweeps/variance_decomp.sh — model configs for the variance-decomposition
# experiment: how much of the month-to-month spread in probe IC is data
# (training month) vs. optimization noise (weight init + data order)?
#
# Sourced by our cluster's run_variance_decomp launcher (not included), which submits one
# bundled SLURM job per (training month x series); each job loops over
# (seed 0..9) so a wall-time truncation still leaves a seed-balanced sample.
#
# training.seed drives weight init, augmentation crops, AND (as of the
# shuffle_seed wiring in pretrain.py) the streaming shuffle order. It is
# appended by slurm_variance_decomp.sh, NOT set here, and the same script
# suffixes wandb.run_name with _<YM>_seed-<N>.
#
# Defines:
#   SWEEP_NAME       — wandb project prefix / SLURM job-name prefix
#   SWEEP_VALUES     — series keys
#   VD_SEEDS_DEFAULT — seed list (10 seeds = the project's SE-band standard)
#   sweep_train_args — maps a series key to TRAIN_ARGS for `uv run train.py`
#
# ── THE METHOD AND WHAT IS SWEPT, AND NOTHING ELSE ─────────────────────────
#
# See scripts/sweeps/README.md. This file used to carry the banner "EVERY KNOB
# IS PINNED, none inherited from schemas.py", on the reasoning that a sweep
# measuring run-to-run variation must not have a config drift into the number
# being estimated. That reasoning was right about the risk and wrong about the
# remedy: pinning the recipe in a second place does not stop drift, it makes
# drift SILENT, because the sweep keeps training the old recipe after
# schemas.py has moved and nothing errors. The 2026-09-08 wave found this the
# hard way -- the pinned blr 1e-4 @ bs128 below is the RETIRED binned recipe,
# and this file would have measured the seed spread of a model the paper no
# longer reports.
#
# So: the supervised arms name their task and nothing else. blr (1e-5),
# batch (256), epochs (100), loss_fn (pairwise), the uniform train/eval
# targets, n_pairs_per_obs, risk_factor_tickers, live_eval and the
# pool=last/pos_embed=rope encoder are all mode=supervised's own defaults, and
# restating them here would only create a second place for them to disagree.
#
# WHAT THAT COSTS, stated plainly: this sweep's meaning now moves when
# schemas.py moves. That is the intended behaviour -- it measures the seed
# spread OF THE REPORTED MODEL, whatever the reported model currently is --
# but it means a defaults change mid-wave splits the sample. Every run records
# its full resolved config in train_meta.json, so check those before pooling
# seeds across a wave that straddles a schemas.py commit.
#
# ── ALL THREE TASKS FOR SUPERVISED, ONE ARM FOR LEJEPA ─────────────────────
#
# The supervised arm trains one head per target, so seed noise can differ by
# target -- return has an IC ~13x smaller than spread_change, and there is no
# reason to assume its seed spread scales the same way. LeJEPA has no head:
# one pretrained encoder is probed for every target, so a second or third
# LeJEPA series would be the same trainings scored twice. Four series, not six.
#
# XS_STATS_DIR NAMES A TARGET DEFINITION. All three targets became
# forward-window differences on 2026-08-22 (docs/return_bad_calculation.md);
# the launcher defaults to xs_anchor_stats_fwdvwap60 and preflights that every
# train+eval month's table exists. Seed spread is not target-invariant: a
# target with a mechanical component has a floor every seed hits, which
# shrinks exactly the spread this sweep exists to measure. spread_change_900
# is known to carry such a component (a scalar known at t scores +0.13..+0.22
# on it), so read its seed spread against that, not against zero.

SWEEP_NAME="variance-decomp"

SWEEP_VALUES=(
    "lejepa"
    "supervised_return"
    "supervised_vol_change"
    "supervised_spread_change"
)

VD_SEEDS_DEFAULT="0 1 2 3 4 5 6 7 8 9"

# LeJEPA's arm is NOT a defaults question: cross-stock K=2 drawn from the focal
# stock's FF49 industry, lamb 0.01 off the k2ind lambda sweep (the code default
# 0.02 is past the knee). These are the swept method, so they are named here.
VD_LAMB="${VD_LAMB:-0.01}"

sweep_train_args() {
    local SERIES="$1"
    local MODE_CFG RUN_NAME
    case "${SERIES}" in
        lejepa)
            MODE_CFG="mode=lejepa \
                mode.lamb=${VD_LAMB} \
                dataset.augmentations.0.name=cross_stock \
                dataset.augmentations.0.n_stocks=2 \
                dataset.augmentations.0.industry_table=data/industry_map.parquet"
            RUN_NAME="vd_lejepa_k2ind"
            ;;
        supervised_return)
            MODE_CFG="mode=supervised mode.task=return_900"
            RUN_NAME="vd_supervised_return"
            ;;
        supervised_vol_change)
            MODE_CFG="mode=supervised mode.task=volatility_change_900"
            RUN_NAME="vd_supervised_vol_change"
            ;;
        supervised_spread_change)
            MODE_CFG="mode=supervised mode.task=spread_change_900"
            RUN_NAME="vd_supervised_spread_change"
            ;;
        *)
            echo "ERROR: unknown variance-decomp series '${SERIES}'" >&2
            return 1
            ;;
    esac

    echo "${MODE_CFG} \
        backbone=transformer \
        wandb.project=${SWEEP_NAME} \
        wandb.group=${SWEEP_NAME} \
        wandb.run_name=${RUN_NAME}"
}
