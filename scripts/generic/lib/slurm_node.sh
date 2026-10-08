#!/bin/bash
# slurm_node.sh — Compute-node helpers for the SLURM launchers. Sourced by
# lib/common.sh.
#
# Everything here runs ON a compute node inside an sbatch job; nothing is
# cluster-specific beyond what the caller passes in.

# ── Shared-memory / file-descriptor headroom ─────────────────────────────────
#
# mosaicml streaming allocates a batch of POSIX shm segments per
# StreamingDataset, keyed by a 6-digit prefix, and unlinks them only on a
# graceful exit. A scancel, an OOM, or any crash orphans the whole set.
#
# The cost is not the leaked memory (segments are small) — it is that
# streaming's prefix scan OPENS every existing segment on the node to find a
# free prefix. A node carrying a few hundred orphans therefore needs several
# hundred fds before the first dataset even exists, and the SLURM default soft
# limit is 1024. This killed 67 of the 135 batch-size-stability jobs on
# 2026-08-15 with `OSError: [Errno 24] Too many open files` at dataset
# construction — the four worst-hit nodes had walked up to prefix 000454,
# orphaned by the 90 seed-variance runs a `scancel -u` had dropped moments
# earlier. Failures then cascaded: each crash orphaned another set.
#
# Raise the soft limit (the hard limit on our cluster was 131072) and sweep segments that
# no live process holds open. Both are best-effort; neither is worth failing on.
prepare_shm_limits() {
    ulimit -n 65536 2>/dev/null || warn "could not raise fd limit (now $(ulimit -n))"

    # ONE fuser EXEC PER SEGMENT. A busy node carries hundreds (326 on one node
    # while a job sat there), and the sweep is only worth its minutes to a
    # job that will itself allocate streaming shm -- a scoring pass never does.
    # SHM_SWEEP=0 keeps the ulimit above, which every job does want, and skips
    # the reclaim.
    [ "${SHM_SWEEP:-1}" = 1 ] || return 0
    command -v fuser >/dev/null 2>&1 || return 0
    local f n=0
    for f in /dev/shm/[0-9][0-9][0-9][0-9][0-9][0-9]_*; do
        [ -e "${f}" ] || continue
        # Only our own, and only with no live opener — a concurrent job of
        # ours on this node is still holding its own segments open.
        [ -O "${f}" ] || continue
        fuser -s "${f}" 2>/dev/null && continue
        rm -f "${f}" 2>/dev/null && n=$((n + 1))
    done
    [ "${n}" -gt 0 ] && echo "==> swept ${n} orphaned streaming shm segments"
    return 0
}

# ── GPU safeguard ────────────────────────────────────────────────────────────
#
# Compute nodes occasionally boot with a broken CUDA driver state (e.g. the
# "CUDA unknown error" seen on one job) and torch.cuda.is_available() returns
# False. Training on CPU would silently waste the allocation, so detect this
# and resubmit the job to the back of the queue instead.
#
# Requires: running inside an SBATCH script (uses $SLURM_JOB_ID, sbatch).
# Argument: path to the SBATCH script that should be resubmitted.
require_gpu_or_resubmit() {
    local script_path="$1"
    if uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        return 0
    fi
    warn "No CUDA device detected on $(hostname) — resubmitting job ${SLURM_JOB_ID} to the back of the queue on partition ${SLURM_JOB_PARTITION}."
    # Preserve the env vars the sweep scripts pass in, and resubmit to the same partition (SBATCH header default may differ).
    # SWEEP_FILES_REL is the multi-sweep spelling a bundled job arrives with;
    # SWEEP_FILE_REL the single-file one run_sweep.sh and the specific/
    # launchers use. ALL already carries both, but a requeue that dropped
    # either would run a different sweep than the one that was submitted.
    local export_vars="TRAIN_ARGS,TRAIN_START,TRAIN_END,EVAL_START,EVAL_END,SWEEP_FILES_REL,SWEEP_FILE_REL,SWEEP_VALUES,SWEEP_VALUES_OVERRIDE,WANDB_SUFFIX,COMMIT_HASH"
    sbatch --export="ALL,${export_vars}" --job-name="${SLURM_JOB_NAME}" --partition="${SLURM_JOB_PARTITION}" "${script_path}" || error "Resubmission failed."
    exit 0
}

# On a SINGLE-NODE cluster resubmission is the wrong move — the resubmitted
# job lands on the same broken driver and loops forever. Fail loudly instead
# so the box gets looked at.
require_gpu_or_fail() {
    if uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        return 0
    fi
    error "No CUDA device detected on $(hostname). Single-node cluster — not resubmitting. Check nvidia-smi / driver state."
}
