#!/bin/bash
# common.sh — Shared helpers and configuration for Pythia scripts.
#
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#   (from a script in pythia/)
#
# Provides:
#   - info(), warn(), error() logging helpers
#   - PYTHIA_DIR, LOCAL_REPO, PYTHIA_HOST, PYTHIA_REPO
#   - .env loaded and validated

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==>\033[0m $*"; }
error() { echo -e "\033[1;31m==>\033[0m $*" >&2; exit 1; }

# ── Paths ────────────────────────────────────────────────────────────────────

# PYTHIA_DIR = the pythia/ directory (parent of lib/)
PYTHIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Repo root — prefer git resolution, but fall back to path math because the
# rsync to pythia excludes .git/, so `git rev-parse` fails on compute nodes.
# PYTHIA_DIR is scripts/pythia, so repo root is two levels up.
LOCAL_REPO="$(git -C "${PYTHIA_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${PYTHIA_DIR}/../.." && pwd))"

# ── Load .env ────────────────────────────────────────────────────────────────

ENV_FILE="${PYTHIA_DIR}/.env"
[ -f "${ENV_FILE}" ] || error "${ENV_FILE} not found. Copy .env.example to .env and fill in your values."
source "${ENV_FILE}"

# ── Derived variables ────────────────────────────────────────────────────────

PYTHIA_HOST="pythia"
PYTHIA_REPO="${PYTHIA_HOME}/market-jepa"

# ── Compute-node helpers (shared with scripts/generic/) ──────────────────────
# prepare_shm_limits, require_gpu_or_resubmit, require_gpu_or_fail moved to
# slurm_node.sh so the generic cloud-cluster tree can source them too.
source "$(dirname "${BASH_SOURCE[0]}")/slurm_node.sh"
