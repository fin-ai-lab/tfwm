#!/bin/bash
# sweeps/full_data_multihead.sh — Full-history supervised baseline, ONE MODEL.
#
# The full_data_supervised.sh recipe with a single multihead trunk in place of
# three separate specialists. 2008-01..2024-11, one calendar month at a time.
#
# WHY ONE MODEL AND NOT THREE. The 32-month W sweep measured what sharing a
# trunk costs, and the answer was nothing or better: multihead minus specialist
# was +0.0050 return at W=32 (t=+4.01), +0.0028 vol at W=8 (t=+2.51), and
# nothing significantly negative on any task at any window. Three specialists
# per month is therefore three times the GPU for a result the shared trunk
# already matches -- on a 203-month sweep that is the difference between one
# overnight wave and three.
#
#
# NOTE THE MULTIHEAD'S OWN W CURVE RAN THE OTHER WAY on the 32 reported months:
# its return head preferred the WIDEST window (+0.0126 at W=32 against +0.0112
# at W=8). W=8 is chosen here for comparability with the specialists this
# replaces, not because the multihead asked for it. It is a real, small cost on
# return and it is deliberate.
#
# A THIN WRAPPER over supervised_multihead.sh, so there is one multihead arg
# builder rather than two that drift. SWEEP_NAME is overridden AFTER the source
# because sweep_train_args reads it at call time -- the full-history series
# needs its own wandb project so it never pools with the 32-month W sweep.
#
# THE TASK LIST AND NOTHING ELSE -- see scripts/sweeps/README.md. This block
# used to read "EVERY KNOB IS PINNED, none inherited", and set LOSS, SEED, BLR,
# BATCH_SIZE and NUM_EPOCHS just below. Those are all
# mode=supervised_multitask's own defaults now (pairwise, seed 42, blr 1e-5,
# bs 256, 100 epochs, plus the pool=last/pos_embed=rope encoder), so restating
# them created a second place for them to disagree.
#
# WORSE THAN REDUNDANT, because supervised_multihead.sh emits an override ONLY
# for a variable the caller actually set: assigning them here made every one of
# them look caller-set, so the arg builder emitted all of them unconditionally
# and the conditional path could never fire. Setting LOSS=... on the command
# line still works and still lands in the run name.
#
# The TAU_FINAL note that used to live here is gone with the binned family
# (2026-09-07): soft_label_temperature is a cross_entropy knob and this series
# is pairwise, so there is no anneal endpoint left to pin.
#
# Submit: SWEEP_FILE_REL=scripts/sweeps/full_data_multihead.sh \
#           our cluster's run_full_data_supervised launcher (not included) \
#           --partition <h100-partition> --months-per-job 4

TASKS="${TASKS:-[return_900,volatility_change_900,spread_change_900]}"

# LOW PRIORITY, same rationale and same escape hatch as full_data_supervised.sh:
# 203 months is large and entirely non-urgent, --nice=10000 parks it behind
# everything else of ours permanently, and backfill still soaks it into idle
# GPUs. SBATCH_EXTRA="--nice=0" from the caller restores normal priority.
#
# It matters more than usual right now: the reported-month LeJEPA sweep is on
# the same queue and is what the SSL finetuning decision is waiting on. This
# must not get in front of it.
SBATCH_EXTRA="--nice=10000 ${SBATCH_EXTRA:-}"

source "$(dirname "${BASH_SOURCE[0]}")/supervised_multihead.sh"

# NO "-ce" SUFFIX. This series trained cross_entropy until the binned family
# was retired on 2026-09-07; it is pairwise (LTR) now, and a project name that
# says otherwise is the kind of thing that gets read years later and believed.
# The old name's checkpoints are archived, so nothing is being renamed under a
# live result.
SWEEP_NAME="supervised-full-month-multihead"

# EVERY MONTH THE MOSAIC HAS, not the 32 reported ones. Overridden AFTER
# the source for the same reason SWEEP_NAME is: supervised_multihead.sh
# declares sampled32, and this file is the full-history variant of it.
# specific/run_full_data_supervised.sh is the only launcher that serves it.
SWEEP_MONTH_SET="full_history"
