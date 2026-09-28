#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
# 64G, MEASURED (2026-09-14), not budgeted. Read the cgroup of a running
# bundle and the request is almost entirely page cache:
#
#     total_rss            6.4 GB   <- anonymous; and it is nearly all
#     total_shmem          6.5 GB   <- /dev/shm for dataloader tensors
#     total_active_anon   77 KB
#     total_cache        107.9 GB   <- of which 100.2 GB inactive_file
#
# The day store is opened with mmap_mode="r" (stable_finance DayRecord:
# "nothing is read until sliced") and DayStoreCellDataset keeps an LRU of
# max_open_days=48 MAPPED days per worker, so a bundle's resident set is
# whatever of the store it has touched and not yet had reclaimed. Four
# concurrent jobs sampled at 5.5-6.4 GB anon against 19-108 GB of cache.
#
# WHICH MEANS --mem IS A CACHE CEILING HERE, NOT A REQUIREMENT, and sacct
# MaxRSS measures the ceiling rather than the job: the same supervised
# sweep "peaks" at 15 GB when given 16G and at 66 GB when given 86G. The
# old 240G reading of that number as demand was circular.
#
# AND THE CEILING IS NOT FREE, which is why this moved. Billing is MAX_TRES
# and one GPU already bills 1000, so 240G costs no priority on
# standard_hopper (2 TB nodes, 8 x 240G = 1920 GiB, 8 GPUs -- memory and
# GPUs run out together). standard_l40s nodes have 773 GB, so the SAME
# request packs 3 jobs onto an 8-GPU node and strands five L40S cards for a
# job needing 7 GB. 64G is 9x the measured anon and packs 8 per node on
# either partition.
#
# A LOWER CEILING CANNOT OOM THIS -- the 100 GB is clean inactive_file and
# is reclaimed, only the ~7 GB of shmem is unreclaimable -- AND THAT IS NOT
# THE SAME AS BEING FREE. The price is "re-reading a local NVMe scratch",
# which this block once wrote off in half a sentence and which turned out to
# be the whole cost: DayStoreCellDataset permutes the WHOLE span every epoch,
# so a ceiling under the span re-reads nearly all of it every pass with no
# locality to recover. 64G against an 87 GB six-month span, six jobs to a
# node: GPU at 0%, md0 at 965 MB/s, 16-35 s/step where this same recipe gets
# 0.7 (2026-09-15, the cancelled variance-decomp wave -- that script's --mem
# block has the measurement). SIZE THE CEILING AGAINST THE SPAN, via ``du
# -sh`` on the staged day store, never against MaxRSS.
#
# 64G IS THE l40s NUMBER AND THIS HEADER HAS NO PARTITION, so it inherits the
# cluster default, which is an l40s queue: 773 GB over 8 cards is 96 GiB a
# GPU, and 240G there packs 3 jobs onto 8 cards and strands five. On hopper
# the share is 251.9 GiB (2063255 MB / 8) and the number is 240G --
# run_all_months_sweep.sh branches on PARTITION and injects it, and an
# explicit --mem in SBATCH_EXTRA beats this line. A hopper run that reaches
# sbatch with 64G came past both and is about to crawl.
#
# Raise it further if the dense-grid cache is ever turned back on --
# GRID_CACHE_GB is per WORKER of anonymous RSS, which is real memory, and
# run_all_months_sweep.sh sizes that case itself.
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=pgpu015,pgpu018

# slurm_train_bundle.sh — Run sweep values for one or more months in a single
# job, on pythia. Thin wrapper: everything but the SBATCH headers and the
# pythia-specific paths lives in lib/train_bundle_body.sh, shared with
# scripts/generic/slurm_train_bundle.sh (the cloud single-node clusters).
# See the body for modes and the env contract.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"

source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

SCRATCH_BASE="/hpc_temp/${USER}"
MACHINE_NAME="pythia"
LOCAL_CKPT_DIR="/hpc_temp/${USER}/market-jepa-checkpoints"
BUNDLE_SCRIPT_REL="scripts/pythia/slurm_train_bundle.sh"
RESUBMIT_ON_NO_GPU=1   # many nodes — requeue past a broken driver
# Pre-built eval panels (scripts/eval/panel_cache.py). ON here
# as of 2026-08-27: /hpc_temp measured 56 T with 55 T free, so the ~26 GB
# (info-off) / ~57 GB (info-on) per month a job stages is not a threat to
# it. Modtime eviction is still real, which is why stage_panel_month counts
# .ready markers and re-stages rather than trusting its own done-marker --
# an evicted panel degrades to a live rebuild, never to a wrong score.
STAGE_PANEL_CACHE=1

source "${REPO_DIR}/scripts/pythia/lib/train_bundle_body.sh"
