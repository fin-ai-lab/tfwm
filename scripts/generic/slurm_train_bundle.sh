#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
# --cpus-per-task / --mem are passed by the submitter from the variant .env
# (SBATCH_CPUS_PER_TASK / SBATCH_MEM) so one script serves every cloud box.

# slurm_train_bundle.sh — Run sweep values for one or more months in a single
# job, on a generic single-node cloud cluster (8×H100/A100 + local SLURM).
# Thin wrapper: the month × value loop, staging, scoring, and checkpoint
# sync live in scripts/pythia/lib/train_bundle_body.sh, shared with the
# pythia tree so the two flows cannot drift apart again.
#
# Differences from pythia, all expressed as knobs below:
#   - No /hpc_temp scratch — venv/wandb/data live under $HOME (per .env).
#   - machine=${MACHINE_NAME} from the variant .env (h100_cluster, a100_*).
#   - Single node: a broken CUDA driver fails the job instead of requeueing.
#
# Required env (set by the submitter): CLUSTER_VARIANT, SWEEP_FILE_REL,
# COMMIT_HASH, and either MONTHS or TRAIN_*/EVAL_* — see the body.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/generic/lib/common.sh"

BLL01_HOST="bll01"
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="${CLUSTER_DATA}"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"

source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

SCRATCH_BASE="${HOME}"
: "${MACHINE_NAME:?ERROR: MACHINE_NAME not set (should come from the variant .env).}"
LOCAL_CKPT_DIR="${CLUSTER_CKPT}"
BUNDLE_SCRIPT_REL="scripts/generic/slurm_train_bundle.sh"
RESUBMIT_ON_NO_GPU=0   # single node — resubmitting lands on the same GPU
# One venv per uv.lock hash, shared by all 8 concurrent jobs (see the body).
# Safe here and not on pythia: $HOME is persistent, not modtime-reaped scratch.
SHARED_VENV=1
# Pre-built eval panels (scripts/eval/panel_cache.py). On here
# because these boxes have 5.6-19 TB of persistent local disk; pythia leaves
# it off (/hpc_temp is modtime-reaped). Months with no panel built simply
# fall through to a live build.
STAGE_PANEL_CACHE=1

source "${REPO_DIR}/scripts/pythia/lib/train_bundle_body.sh"
