#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_l40s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
# 128G, down from 400G. A scoring job is the same mmap-backed read as a
# training bundle -- see slurm_train_bundle.sh's --mem block -- so its resident
# set is mostly reclaimable page cache and 400G was buying cache, not memory.
# NOT 64G like the trainers, deliberately: this runs N_WORKERS embed processes
# (14 on the shape the forward-decay campaign submits) against a trainer's 7,
# each with its own CUDA context and shard buffers, and that part IS anonymous.
# 128G is the conservative number until one of these is read off the cgroup the
# way the trainers were.
#SBATCH --mem=128G
# Overridden per-submission by run_score_ckpts.sh's TIME_LIMIT (default 20h);
# this is the floor for a bare sbatch of this file. Raised from 8h after
# fwdv2pairs-581eb2-part05 timed out at row 120 of 135 and lost the lot --
# results are written once at the end, so a TIMEOUT costs the whole job.
#SBATCH --time=0-20:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=score-ckpts
#SBATCH --exclude=pgpu015

# slurm_score_ckpts.sh — score ALREADY-TRAINED checkpoints on the synchronized
# rank-IC panel, on a compute node instead of on bll01.
#
# Every other pythia entry point trains first and scores at the tail
# (slurm_train_bundle.sh runs post_train_ic_eval.py after
# the last optimizer step). There was no path for "here are N checkpoints that
# already exist, score them" — so those passes ran on bll01, one box, and the
# arithmetic is bad: the archive comparison was 3968 shards at ~27 shards/min,
# ~2.5 h wall-clock, against a cluster with ~128 GPUs sitting next to it. A
# 203-month full-history pass would be ~40 h locally. This closes that gap.
#
# WHAT IT DOES NOT CHANGE: nothing about the measurement. It calls the same
# probe_fit_size.py embed-many and the same xs_ic_series.py reduce that the
# local runs call, which in turn call xs_ic_eval.score — the identical function
# the in-job hook uses. Moving the work to a compute node must not move the
# number, so the scoring path is imported, never reimplemented.
#
# CONTRACT (env vars):
#   SCORE_MANIFEST  path ON PYTHIA to a TSV of
#                     <ckpt_dir_on_bll01>\t<fit_month>\t<eval_month>
#                   — exactly the manifest xs_ic_series.py reduce consumes, so
#                   the same file drives a local and a cluster run.
#   SCORE_TAG       names the cache dir and the result json.
#   SCORE_PROBE     1 (default) = delete head.pt from the STAGED copy so
#                   load_model builds no head and scoring takes the ridge path.
#                   0 = keep the head and score with it.
#   N_WORKERS       default 32, one per CPU.
#   NUM_SHARDS      default 32. MUST match --expect-shards in the reduce, and
#                   changing it invalidates every cached shard.
#   XS_STATS_DIR    which anchor-stat table set to stage and score against.
#                   THE TABLE SET IS THE TARGET DEFINITION, not a path
#                   preference: it holds the (mu, sigma) OF a particular
#                   return/spread_change/volatility_change, and the reduce
#                   also uses it to UNDO the z-score. Scoring one method
#                   against a different set than its comparators is a
#                   different metric, not a different file layout. Default
#                   xs_anchor_stats (midpoint targets); the two-forward-window
#                   definition of 2026-08-22 is xs_anchor_stats_fwdvwap60.
#
# Submitted by specific/run_score_ckpts.sh, which chunks a big manifest into
# per-job pieces so each job stages only the months its own rows need.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"
source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

: "${SCORE_MANIFEST:?ERROR: SCORE_MANIFEST not set (path on pythia to the ckpt TSV).}"
: "${SCORE_TAG:?ERROR: SCORE_TAG not set.}"
SCORE_PROBE="${SCORE_PROBE:-1}"
# 1 (default) keeps the retired macro one-vs-rest AUC alongside the rank IC.
# 0 drops it. The AUC's logistic probe is the expensive fit in the reduce --
# 150k rows x 384 collinear features, and --auc-extra-bins asks for two more
# per task -- so a manifest with dozens of rows per job spends most of its
# wall-clock on a number no current figure reads. The ridge probe that
# produces the IC is unaffected.
SCORE_AUC="${SCORE_AUC:-1}"
N_WORKERS="${N_WORKERS:-32}"
# stage_xs_anchor_stats keys its rsync source, lock and marker off this, so
# naming it stages the right tables; the embed and reduce below have to be
# pointed at the same subdirectory by hand.
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
export XS_STATS_DIR
export XS_PROBE_FIT_TARGET="${XS_PROBE_FIT_TARGET:-zscore}"
NUM_SHARDS="${NUM_SHARDS:-32}"
N_GPUS="${SLURM_GPUS_ON_NODE:-4}"

# Anchor counts are the reported protocol, mirrored from xs_ic_series.py: the
# fit month is dense, the eval month defines the reported cross-sections and
# must stay at 8. Not knobs — a different pair is a different measurement.
FIT_ANCHORS=36
EVAL_ANCHORS=8

CACHE_DIR="/hpc_temp/${USER}/score-ckpts/${SCORE_TAG}"
CKPT_STAGE="/hpc_temp/${USER}/score-ckpts/.ckpts"
RESULT_JSON="${CACHE_DIR}/${SCORE_TAG}.json"
# Results must be pushed back: /hpc_temp is mounted on compute nodes ONLY, so
# anything left here is unreachable from the login node and from bll01.
RESULT_DEST="${SCORE_RESULT_DEST:-/data/lab/score_results}"

[ -f "${SCORE_MANIFEST}" ] || error "manifest not found on pythia: ${SCORE_MANIFEST}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job ${SLURM_JOB_ID} on $(hostname)"
echo "  CPUs ${SLURM_CPUS_PER_TASK}  GPUs ${N_GPUS}  workers ${N_WORKERS}"
echo "  manifest : ${SCORE_MANIFEST} ($( { grep -cve '^[[:space:]]*$' "${SCORE_MANIFEST}" || true; } ) rows)"
echo "  targets  : ${XS_STATS_DIR}"
echo "  mode     : $([ "${SCORE_PROBE}" = 1 ] && echo 'PROBE (ridge, head stripped)' || echo 'HEAD')"
echo "  auc      : $([ "${SCORE_AUC}" = 1 ] && echo 'on' || echo 'off (rank IC only)')"
echo "  cache    : ${CACHE_DIR}"
echo "════════════════════════════════════════════════════════════════"

export PATH="${HOME}/.local/bin:${PATH}"
VENV_DIR="/hpc_temp/${USER}/market-jepa-venv-${SLURM_JOB_ID}"
export UV_CACHE_DIR="/hpc_temp/${USER}/.cache/uv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE=copy
trap 'rm -rf "${VENV_DIR}" "${UV_CACHE_DIR}"' EXIT
cd "${REPO_DIR}"
uv sync --frozen --quiet
require_gpu_or_resubmit "${REPO_DIR}/scripts/pythia/slurm_score_ckpts.sh"

# ── Stage only the months this manifest actually touches ──────────────────────
# A scoring manifest is sparse in time (the archive panel jumps 2008-02 ->
# 2008-07 -> ... -> 2023-10), so stage_all's contiguous range would pull ~190
# months to get 32. stage_month_list takes the explicit set.
MONTHS=$(awk 'NF && $0 !~ /^#/ {print $2; print $3}' "${SCORE_MANIFEST}" | sort -u)
stage_month_list ${MONTHS}
prepare_shm_limits

# USE THE CACHE. stage_month_list has already staged each month's pre-built
# panel, but until 2026-09-11 this script never told the scorer where they
# were, so embed_month_many's panel_source found no cache_root and re-decoded
# every month from the mosaic -- the staged panels sat on the node unread.
#
# Safe unconditionally: panel_source falls through to a live iter_panel decode
# for any (month, key) it cannot match, so an absent panel or a key mismatch
# behaves exactly as before. The info-token flags are deliberately NOT part of
# the panel key -- one panel holds the 9 real channels plus the token's 11
# constants and the reader broadcasts whichever blocks a checkpoint was trained
# with -- so the SAME cached panel serves the 20-channel floor and a 9-channel
# model alike.
if [ -d "${DATA_DIR}/panel_cache" ]; then
    export MJ_PANEL_CACHE="${DATA_DIR}/panel_cache"
    echo "==> panel cache: ${MJ_PANEL_CACHE}"
else
    echo "==> panel cache: none staged (STAGE_PANEL_CACHE=${STAGE_PANEL_CACHE:-1}); decoding live"
fi

# Single-threaded per worker: 32 processes over 32 cores, so BLAS threads would
# oversubscribe and make the CPU-bound per-ticker-day decode slower, not faster.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# xs_ic_eval stages each month's decode in tempfile.TemporaryDirectory, which
# honours TMPDIR and otherwise lands in /tmp. /tmp is a small node-local volume
# (9.8 G on bll01) and 32 workers each staging ~128 MB fill it; the run then
# dies with ENOSPC that looks like a data-disk problem and is not. A killed
# worker also leaks its directory, so the volume degrades across retries. Point
# it at the big scratch instead.
export TMPDIR="/hpc_temp/${USER}/score-ckpts/.tmp"
mkdir -p "${TMPDIR}"

# ── Stage the checkpoints ─────────────────────────────────────────────────────
# -L dereferences symlinks: backbone-only shadow dirs built on bll01 are
# symlinks into models-archive, and a plain rsync would copy a dangling link.
mkdir -p "${CKPT_STAGE}" "${CACHE_DIR}"
declare -A LOCAL_OF
while read -r CKPT FIT EV <&3; do
    [ -z "${CKPT:-}" ] && continue
    case "${CKPT}" in \#*) continue ;; esac
    NAME="$(basename "${CKPT}")"
    LOCAL="${CKPT_STAGE}/${NAME}"
    if [ -z "${LOCAL_OF[${NAME}]:-}" ]; then
        mkdir -p "${LOCAL}"
        rsync -azL --no-motd "${BLL01_HOST}:${CKPT}/" "${LOCAL}/" </dev/null \
            || error "checkpoint rsync failed: ${CKPT}"
        # Strip the head on the STAGED copy only; bll01's checkpoint is
        # untouched. load_model builds a head from config and a missing head.pt
        # leaves it randomly initialised, so scoring falls to the ridge probe.
        [ "${SCORE_PROBE}" = 1 ] && rm -f "${LOCAL}/head.pt"
        LOCAL_OF[${NAME}]="${LOCAL}"
    fi
done 3< "${SCORE_MANIFEST}"
echo "==> staged ${#LOCAL_OF[@]} checkpoint(s)"

# ── Build the shard task list ─────────────────────────────────────────────────
TASKS="${CACHE_DIR}/tasks.txt"
: > "${TASKS}"
while read -r CKPT FIT EV <&3; do
    [ -z "${CKPT:-}" ] && continue
    case "${CKPT}" in \#*) continue ;; esac
    LOCAL="${CKPT_STAGE}/$(basename "${CKPT}")"
    for ((s = 0; s < NUM_SHARDS; s++)); do
        echo "${LOCAL} ${FIT} ${s} ${FIT_ANCHORS}" >> "${TASKS}"
        echo "${LOCAL} ${EV} ${s} ${EVAL_ANCHORS}" >> "${TASKS}"
    done
done 3< "${SCORE_MANIFEST}"
echo "==> $(wc -l < "${TASKS}") shard tasks"

# ── Embed ─────────────────────────────────────────────────────────────────────
# Two passes. embed-many is resumable (an existing shard is skipped), and a
# transient failure — a full /hpc_temp, an evicted month — otherwise silently
# leaves gaps that only surface at reduce, after the whole embed is paid for.
# Writes are atomic (probe_fit_size._atomic_savez), so a shard's existence
# means it is complete and pass 2 can trust the skip.
for PASS in 1 2; do
    echo "==> embed pass ${PASS} / 2"
    for ((i = 0; i < N_WORKERS; i++)); do
        CUDA_VISIBLE_DEVICES=$((i % N_GPUS)) \
        uv run scripts/eval/probe_fit_size.py embed-many \
            --tasks "${TASKS}" \
            --worker-index "${i}" --num-workers "${N_WORKERS}" \
            --num-shards "${NUM_SHARDS}" \
            --out-dir "${CACHE_DIR}" \
            --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
            --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR}" \
            > "${CACHE_DIR}/embed_p${PASS}_w${i}.log" 2>&1 &
    done
    # embed-many exits non-zero on ANY failed shard, and `wait` under `set -e`
    # would abort the job — but pass 2 exists precisely to repair pass 1.
    wait || true
    NFAIL=$( { grep -hc "FAIL" "${CACHE_DIR}"/embed_p${PASS}_w*.log 2>/dev/null || true; } \
            | awk '{s+=$1} END{print s+0}')
    echo "    shards on disk: $(find "${CACHE_DIR}" -name '*.npz' | wc -l)"
    echo "    failures this pass: ${NFAIL}"
    [ "${NFAIL}" -eq 0 ] && break
    { grep -h "FAIL" "${CACHE_DIR}"/embed_p${PASS}_w*.log 2>/dev/null || true; } \
        | sed 's/.*: //' | sort | uniq -c | head
done

# ── Reduce ────────────────────────────────────────────────────────────────────
# --expect-shards makes a short panel a hard error rather than a number: a
# missing shard scores fine and means nothing.
echo "==> reducing ..."
# The anchor tables again, this time to UNDO the z-score: the AUC's bins live
# on the raw target. The default path is bll01's, which does not exist here.
# With the head kept (SCORE_PROBE=0) the reduce also scores it, from the
# embeddings already cached — one MLP forward, no second pass over the panel.
REDUCE_HEAD=()
[ "${SCORE_PROBE}" = 0 ] && REDUCE_HEAD=(--ckpt-dir "${CKPT_STAGE}")
REDUCE_AUC=()
[ "${SCORE_AUC}" = 0 ] && REDUCE_AUC=(--no-auc)
REDUCE_RC=0
uv run scripts/eval/xs_ic_series.py reduce \
    --cache-dir "${CACHE_DIR}" \
    --manifest "${SCORE_MANIFEST}" \
    --expect-shards "${NUM_SHARDS}" \
    --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR}" \
    "${REDUCE_HEAD[@]}" "${REDUCE_AUC[@]}" \
    --json "${RESULT_JSON}" || REDUCE_RC=$?
# The json is written BEFORE the closing summary, so a reduce that died late
# may still have produced a complete result. Under set -e that once threw away
# a finished 40-minute embed because a print statement raised.
if [ "${REDUCE_RC}" -ne 0 ] && [ ! -s "${RESULT_JSON}" ]; then
    error "reduce failed (rc=${REDUCE_RC}) and wrote no result"
fi
[ "${REDUCE_RC}" -ne 0 ] && echo "==> WARNING: reduce exited ${REDUCE_RC} but "\
    "the result json exists — pushing it"

# ── Push results back ─────────────────────────────────────────────────────────
# Unconditional: the reduce writes its json before it prints its summary, so a
# formatting bug in the summary must not strand a finished result on a compute
# node whose /hpc_temp the login node cannot even see.
ssh "${BLL01_HOST}" "mkdir -p ${RESULT_DEST}"
rsync -az "${RESULT_JSON}" "${BLL01_HOST}:${RESULT_DEST}/" \
    || error "result push failed; json is at ${RESULT_JSON} on $(hostname)"
echo "==> results at ${BLL01_HOST}:${RESULT_DEST}/$(basename "${RESULT_JSON}")"

echo "════════════════════════════════════════════════════════════════"
echo "  Done: $(date)"
echo "════════════════════════════════════════════════════════════════"
