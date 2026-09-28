#!/bin/bash
# submit_month_major.sh — submit several sweeps MONTH-MAJOR: one job per
# month, carrying every sweep, so everything that needs a month's data is in
# the job that staged it.
#
# WHY THIS IS THE DEFAULT WAY TO SUBMIT MULTIPLE SWEEPS.
#
# The per-job cost that is NOT the training is all keyed on the month: the
# mosaic rsync, the shard materialization, the anchor tables and -- since
# 2026-08-27 -- the pre-built eval panel, which is ~26 GB for a month's
# probe+eval pair. All of it is staged NODE-LOCALLY. Submitting sweep-major
# (all of sweep A's 32 months, then all of sweep B's) spreads each month
# across the whole queue, so a node stages a month, finishes, and by the time
# another job wants that month the staged copy has aged out -- on pythia
# literally, since /hpc_temp evicts by modtime.
#
# THE FIX IS THE JOB, NOT THE ORDER. This script used to submit one job per
# (month, sweep) and rely on adjacent job IDs to make them run together. That
# only biases the ORDER slurm starts them in: with many GPUs the jobs for one
# month land on different nodes, and each stages the month again. Sharing
# staging requires sharing a node, and the only thing that guarantees a shared
# node is being the SAME JOB. So the sweeps are handed to the submitter as one
# list and slurm_train_bundle.sh runs them month × sweep × value, staging each
# month once.
#
# THE COST: the sweeps run back to back on the job's single GPU, so a month's
# wall clock is now the SUM of the sweeps. Raise the limit if the default is
# short:  SBATCH_EXTRA="--time=2-00:00:00" ./scripts/submit_month_major.sh ...
#
# Usage:
#   ./scripts/submit_month_major.sh <submitter> <variant-or-flags...> -- <sweep>...
#
#   # A100 80GB box, three sweeps, month-major:
#   ./scripts/submit_month_major.sh generic 80 -- \
#       sweeps/ssl_ic/cost_r1.sh sweeps/ssl_ic/ts2vec_r1.sh sweeps/ssl_ic/dino_r1.sh
#
#   # pythia, explicit months:
#   MONTHS_OVERRIDE="2009-06 2011-12" ./scripts/submit_month_major.sh pythia \
#       -p standard_hopper -- sweeps/lejepa_k2_lambda.sh sweeps/k2ind_lamb.sh
#
# MONTHS_OVERRIDE selects the month list; unset, the reported 32 are used.
# Both are passed straight through to the submitter, which is what this script
# now is: a thin, self-documenting front end for
# `run_all_months_sweep.sh <flags> <sweep>...`. One sweep is a legitimate
# degenerate case (it is just that submitter's normal behavior).
#
# Per-sweep gap fills still work, positionally: SWEEP_VALUES_OVERRIDE takes one
# '|'-separated entry per sweep file, an empty entry running that sweep whole
# (e.g. SWEEP_VALUES_OVERRIDE="blr-3e-4|" for two sweeps).
set -euo pipefail

TREE="${1:?usage: $0 <generic|pythia> <submitter flags...> -- <sweep>...}"; shift
case "${TREE}" in
    generic) SUBMIT="./scripts/generic/run_all_months_sweep.sh" ;;
    pythia)  SUBMIT="./scripts/pythia/run_all_months_sweep.sh" ;;
    *) echo "ERROR: tree must be 'generic' or 'pythia', got '${TREE}'" >&2; exit 1 ;;
esac

FLAGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do FLAGS+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "ERROR: expected '--' before the sweep list" >&2; exit 1; }
shift
SWEEPS=("$@")
[ ${#SWEEPS[@]} -gt 0 ] || { echo "ERROR: no sweep files given" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

echo "Month-major: ${#SWEEPS[@]} sweep(s) in ONE job per month, over the staged month they share."
exec "${SUBMIT}" "${FLAGS[@]}" "${SWEEPS[@]}"
