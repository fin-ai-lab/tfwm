#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-reclaim-%j.out
# reclaim_scratch.sh — free a node's /hpc_temp by hand, CPU only.
#
# The bundle does this itself at job start when the node is short
# (stage_data.sh:reclaim_scratch_if_low); this is the same routine for the
# case where a node is already full and every GPU job it receives dies in
# staging before that check can help. No GPU is requested, so it starts at
# once, and a node can only be reached from a job on it (pam_slurm_adopt).
#
#   sbatch --partition=standard_l40s --nodelist=pgpu014 scripts/pythia/reclaim_scratch.sh
#   SCRATCH_MIN_FREE_GB=100000 sbatch ... # force: reclaim everything not in use
set -uo pipefail
REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"
SCRATCH_BASE="/hpc_temp/${USER}"
source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"
echo "node $(hostname -s)"; df -h /hpc_temp | tail -1
SCRATCH_MIN_FREE_GB="${SCRATCH_MIN_FREE_GB:-100000}" reclaim_scratch_if_low
du -sh "${DATA_DIR}"/* 2>/dev/null | sort -h | tail -6
