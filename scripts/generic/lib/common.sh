#!/bin/bash
# common.sh — Shared helpers and configuration for the generic cloud-cluster
# scripts. The heavy lifting (staging, node helpers, the bundle body, repo
# sync) lives in the other lib/ files — this file only resolves paths
# and loads the per-cluster .env.
#
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#   (from a script in generic/)
#
# Provides:
#   - info(), warn(), error() logging helpers
#   - prepare_shm_limits(), require_gpu_or_resubmit(), require_gpu_or_fail()
#     (from lib/slurm_node.sh)
#   - CLUSTER_DIR, LOCAL_REPO, CLUSTER_HOST, CLUSTER_REPO
#   - variant .env loaded and validated

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==>\033[0m $*"; }
error() { echo -e "\033[1;31m==>\033[0m $*" >&2; exit 1; }

# ── Paths ────────────────────────────────────────────────────────────────────

# CLUSTER_DIR = the generic/ directory (parent of lib/)
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Repo root — prefer git resolution, but fall back to path math because the
# rsync to cluster excludes .git/, so `git rev-parse` fails on compute nodes.
# CLUSTER_DIR is scripts/generic, so repo root is two levels up.
LOCAL_REPO="$(git -C "${CLUSTER_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${CLUSTER_DIR}/../.." && pwd))"

# Compute-node helpers.
source "${CLUSTER_DIR}/lib/slurm_node.sh"

# ── Load .env ────────────────────────────────────────────────────────────────

# CLUSTER_VARIANT selects which .env to source ('h100', '40', '80', ...).
# The submit scripts set it from their first CLI arg. A variant X resolves to
# .env.X, or the legacy .env.XGB spelling (the A100 boxes: 40 → .env.40GB).
: "${CLUSTER_VARIANT:?CLUSTER_VARIANT not set — pass the variant ('h100', '40', '80') as the first arg}"

if [ -f "${CLUSTER_DIR}/.env.${CLUSTER_VARIANT}" ]; then
    ENV_FILE="${CLUSTER_DIR}/.env.${CLUSTER_VARIANT}"
elif [ -f "${CLUSTER_DIR}/.env.${CLUSTER_VARIANT}GB" ]; then
    ENV_FILE="${CLUSTER_DIR}/.env.${CLUSTER_VARIANT}GB"
else
    error "No .env for variant '${CLUSTER_VARIANT}' (looked for .env.${CLUSTER_VARIANT} and .env.${CLUSTER_VARIANT}GB in ${CLUSTER_DIR}). Copy .env.example and fill in your values."
fi
source "${ENV_FILE}"

# ── Derived variables ────────────────────────────────────────────────────────

CLUSTER_HOST="${CLUSTER_HOST:-cluster}"
CLUSTER_REPO="${CLUSTER_HOME}/market-jepa"

# Per-job sbatch resources (differ per cluster).
: "${SBATCH_CPUS_PER_TASK:?SBATCH_CPUS_PER_TASK not set in ${ENV_FILE}}"
: "${SBATCH_MEM:?SBATCH_MEM not set in ${ENV_FILE}}"

# Data host (data source) — required for cluster_setup() to authorize SSH and
# for compute jobs to rsync month data on demand.
: "${DATA_HOST_USER:?DATA_HOST_USER not set in ${ENV_FILE}}"
: "${DATA_HOST_IP:?DATA_HOST_IP not set in ${ENV_FILE}}"
: "${DATA_HOST_DIR:?DATA_HOST_DIR not set in ${ENV_FILE}}"
: "${DATA_HOST_METADATA:?DATA_HOST_METADATA not set in ${ENV_FILE}}"
