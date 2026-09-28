#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_hopper
# ONE GPU, EIGHT CORES -- sized for the QUEUE, not for the node. On hopper a
# job bills max(cpus*142.857, mem_gb*4.069, gpus*1000): this shape bills 1143,
# so the 31 jobs of a full panel fit inside qos_bll's 36000 and run AT ONCE.
# The 8-GPU/32-CPU shape bills 8000 and would run four at a time -- eight times
# the hardware per job and a quarter of the throughput. Memory is not what
# bills here (it only enters above 281 GB at this shape), so 128G is chosen to
# pack nodes, not to buy priority. See run_score_ckpts.sh's billing block.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# ~3 h expected (24 fit-month forwards + 4 eval panels at ~4 min each, plus
# staging); 12 h because a TIMEOUT loses the whole embed -- the results are
# written at the end -- and an over-long wall costs only queue priority.
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=probe-breadth
#SBATCH --exclude=pgpu015

# slurm_probe_breadth.sh — the probe-fit-size curve for the reported arms.
#
# plots/core/probe_fit_breadth.png asked "one month deep or six months
# broad"; this asks the question the latent table raises instead: at the SAME
# number of fitting rows, how does a ridge on each arm's frozen features
# compare, and where does it stand against the supervised head that trained on
# those very rows. Four arms (three supervised specialists + LeJEPA +time warp)
# over the 31-month panel.
#
# WHY THE CLUSTER. The pool is the head's own six-month span at 36 anchors/day
# -- ~358k rows a month, ~2.1M in a pool -- and there are 744 (checkpoint, fit
# month) forwards. Measured on bll01's A40 at 8 workers: ~3.9 min each, so
# ~40-50 h on one GPU. Here it is 8 GPUs a node across as many nodes as the
# chunker asks for.
#
# THE MEASUREMENT IS UNCHANGED. Same probe_fit_size.py embed-many the local
# runs call, same reduce, same anchor protocol -- the fit month dense at 36,
# the eval month at the reported 8. Moving the work must not move the number.
#
# ONE JOB HOLDS WHOLE (arm, eval month) GROUPS. The reduce pools every non-eval
# month it finds in a checkpoint's cache dir, so a group whose six fit months
# were split across two jobs would be reduced on a PARTIAL pool and report a
# plausible, wrong, smaller-n curve. run_probe_breadth.sh chunks by eval month
# for that reason; this script re-checks it rather than trusting the caller.
#
# CONTRACT (env):
#   PB_MANIFEST  path ON PYTHIA to the TSV <ckpt_dir>\t<fit_month>\t<eval_month>
#                (scripts/eval/build_probe_breadth_manifest.py), six fit rows
#                per (arm, eval month).
#   PB_TAG       names the cache dir and the result jsons.
#   N_WORKERS    default 32, one per CPU.
#   NUM_SHARDS   default 32; changing it invalidates every cached shard.
#   XS_STATS_DIR the anchor-stat table set = THE TARGET DEFINITION.
#   PB_POOL      the READOUT to embed at, e.g. "last". Empty (default) means
#                each checkpoint's own training pool, which is what every run
#                before 2026-09-17 did -- and why the SSL arms were mean-pooled
#                while the supervised arms and the random-init floor they are
#                struck against were read at last. Prediction is scored at the
#                LAST token for every arm; the latent suite reads everything at
#                the mean (panel_lib.LATENT_POOL). It is stamped into every
#                shard and every result row, so a cache dir or a results
#                directory cannot silently mix two readouts.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
# stage_data.sh reads BLL01_DATA when a month is NOT already staged; without it
# the rsync path dies on "unbound variable" under set -u, and only for months
# this node has never seen -- so it hides on a warm node and fails on a cold
# one. It fails QUIETLY here besides: the staging fan-out runs each month in a
# background subshell, so its `exit 1` kills that subshell and the job walks on
# to embed a month whose mosaic never arrived. Same line and same reason as
# slurm_tsfm_latent.sh and slurm_score_ckpts.sh.
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"
source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

: "${PB_MANIFEST:?ERROR: PB_MANIFEST not set.}"
: "${PB_TAG:?ERROR: PB_TAG not set.}"
N_WORKERS="${N_WORKERS:-8}"
NUM_SHARDS="${NUM_SHARDS:-32}"
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
PB_POOL="${PB_POOL:-}"
[ -n "${PB_POOL}" ] && POOL_ARG=(--pool "${PB_POOL}") || POOL_ARG=()
export XS_STATS_DIR
export XS_PROBE_FIT_TARGET="${XS_PROBE_FIT_TARGET:-zscore}"
# This job never allocates streaming shm, so prepare_shm_limits has nothing of
# ours to reclaim and its per-segment fuser sweep is pure preamble. The fd
# limit it raises is still wanted. Likewise the du: see stage_data.sh.
export SHM_SWEEP="${SHM_SWEEP:-0}"
export STAGE_REPORT_SIZE="${STAGE_REPORT_SIZE:-0}"
N_GPUS="${SLURM_GPUS_ON_NODE:-1}"

# The reported protocol, mirrored from xs_ic_series.py: dense fit month, and
# the eval month at 8 because that is what defines the reported cross-sections.
# Not knobs -- a different pair is a different measurement.
FIT_ANCHORS=36
EVAL_ANCHORS=8

CACHE_DIR="/hpc_temp/${USER}/probe-breadth/${PB_TAG}"
CKPT_STAGE="/hpc_temp/${USER}/probe-breadth/.ckpts"
RESULT_DEST="${PB_RESULT_DEST:-/data/lab/probe_breadth/results}"

[ -f "${PB_MANIFEST}" ] || error "manifest not found on pythia: ${PB_MANIFEST}"

# EVERY GROUP WHOLE, checked here. A (ckpt, eval month) group must bring all
# six of its fit months into this job; anything less silently reduces a short
# pool. Refuse rather than produce a number.
BAD=$(awk 'NF && $0 !~ /^#/ {n[$1"\t"$3]++} END {for (k in n) if (n[k] != 6) print k, n[k]}' \
        "${PB_MANIFEST}")
[ -z "${BAD}" ] || error "manifest has partial groups (want 6 fit months each):
${BAD}"
# AND EVERY CHECKPOINT DIR SERVES ONE EVAL MONTH. The cache dir is keyed by
# the checkpoint's basename and the reduce pools every non-eval month in it,
# so a dir shared by two eval months would reduce a 7-12 month pool as if it
# were the six-month one. Trained arms satisfy this by construction; the
# frozen TSFMs do because build_probe_breadth_manifest.py --tsfm writes one
# dir per eval month. Checked here so nothing else has to be trusted.
SHARED=$(awk 'NF && $0 !~ /^#/ {n=split($1,p,"/"); k=p[n]; if (!(k SUBSEP $3 in s)) {s[k SUBSEP $3]=1; c[k]++}} END {for (k in c) if (c[k] > 1) print k, c[k]}' \
        "${PB_MANIFEST}")
[ -z "${SHARED}" ] || error "checkpoint dirs shared across eval months (the reduce would pool them):
${SHARED}"

# ── Instrumentation: the whole allocation, not just the forwards ─────────────
# A job holds its GPU from the moment SLURM allocates it, and everything before
# the first forward -- venv sync, mosaic staging, checkpoint rsync -- is GPU
# time bought and not used. Sizing the next wave needs to know WHICH phase ate
# the wall-clock, so the job samples itself from here (the first line after the
# allocation) until it exits, and stamps each phase transition. nvidia-smi's
# own -l loop: no ssh round trip, no python, ~1 row per 2 s.
mkdir -p "${CACHE_DIR}"
PHASE_FILE="${CACHE_DIR}/phases.tsv"
GPU_FILE="${CACHE_DIR}/gpu.csv"
: > "${PHASE_FILE}"
phase() {
    printf '%s\t%s\t%s\n' "$(date +%s)" "$(date +%H:%M:%S)" "$1" >> "${PHASE_FILE}"
    echo "==> PHASE ${1} at $(date +%H:%M:%S)"
}
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader -l 2 > "${GPU_FILE}" 2>/dev/null &
GPU_SAMPLER=$!
phase alloc

echo "════════════════════════════════════════════════════════════════"
echo "  Job ${SLURM_JOB_ID} on $(hostname)"
echo "  CPUs ${SLURM_CPUS_PER_TASK}  GPUs ${N_GPUS}  workers ${N_WORKERS}"
echo "  manifest : ${PB_MANIFEST} ($( { grep -cve '^[[:space:]]*$' "${PB_MANIFEST}" || true; } ) rows)"
echo "  targets  : ${XS_STATS_DIR}"
echo "  cache    : ${CACHE_DIR}"
echo "════════════════════════════════════════════════════════════════"

export PATH="${HOME}/.local/bin:${PATH}"
VENV_DIR="/hpc_temp/${USER}/market-jepa-venv-${SLURM_JOB_ID}"
export UV_CACHE_DIR="/hpc_temp/${USER}/.cache/uv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE=copy
trap 'kill "${GPU_SAMPLER}" 2>/dev/null || true; rm -rf "${VENV_DIR}" "${UV_CACHE_DIR}"' EXIT
phase venv_sync
cd "${REPO_DIR}"
uv sync --frozen --quiet
require_gpu_or_resubmit "${REPO_DIR}/scripts/pythia/slurm_probe_breadth.sh"

phase stage_months
MONTHS=$(awk 'NF && $0 !~ /^#/ {print $2; print $3}' "${PB_MANIFEST}" | sort -u)
stage_month_list ${MONTHS}
phase verify_months

# VERIFY WHAT WAS STAGED. stage_month_list fans the months out into background
# subshells, so a month whose rsync fails exits ITS subshell and the job walks
# on -- embedding a month with no mosaic, which produces empty shards, a short
# fit pool, and a curve at the wrong n with nothing raising. Check the thing
# itself rather than trusting the fan-out's exit status.
MISSING=""
for M in ${MONTHS}; do
    Y="${M%%-*}"; MM="${M##*-}"
    D="${DATA_DIR}/1Hz_mosaic_mnth/${Y}/${MM}"
    [ -d "${D}" ] && [ -n "$(ls -A "${D}" 2>/dev/null)" ] || MISSING="${MISSING} ${M}"
done
[ -z "${MISSING}" ] || error "mosaic missing after staging:${MISSING}"
echo "==> verified ${DATA_DIR}/1Hz_mosaic_mnth for $(echo ${MONTHS} | wc -w) month(s)"

phase shm_limits
prepare_shm_limits
phase panel_check

# The staged panels are model-neutral and all four arms resolve to ONE panel
# key, so a month decoded for the first arm is read from memmap by the other
# three. Without this the decode is paid per (checkpoint, month) instead of
# per month -- 4x on the fit pool.
if [ -d "${DATA_DIR}/panel_cache" ]; then
    export MJ_PANEL_CACHE="${DATA_DIR}/panel_cache"
    echo "==> panel cache: ${MJ_PANEL_CACHE}"
else
    echo "==> panel cache: none staged; decoding live"
fi

# ONE BLAS THREAD FOR THE EMBED, where N_WORKERS processes each own one core
# and a threaded BLAS in each would oversubscribe the allocation N_WORKERS-fold.
# The REDUCE re-raises this (see below) -- it is a single process.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Unbuffered, or nothing this job prints from python appears until it exits:
# the reduce runs for hours between its per-checkpoint lines, and a silent log
# is indistinguishable from a hang. Cost is nil; every print here is a line.
export PYTHONUNBUFFERED=1
export TMPDIR="/hpc_temp/${USER}/probe-breadth/.tmp"
mkdir -p "${TMPDIR}" "${CKPT_STAGE}" "${CACHE_DIR}"

# ── Stage the checkpoints ─────────────────────────────────────────────────────
phase stage_ckpts
declare -A LOCAL_OF
while read -r CKPT FIT EV <&3; do
    [ -z "${CKPT:-}" ] && continue
    case "${CKPT}" in \#*) continue ;; esac
    NAME="$(basename "${CKPT}")"
    [ -n "${LOCAL_OF[${NAME}]:-}" ] && continue
    LOCAL="${CKPT_STAGE}/${NAME}"
    mkdir -p "${LOCAL}"
    rsync -azL --no-motd "${BLL01_HOST}:${CKPT}/" "${LOCAL}/" </dev/null \
        || error "checkpoint rsync failed: ${CKPT}"
    # This sweep is ridge-only: the head is never scored here (its IC is
    # already in the checkpoint's own xs_ic.json and needs no forward), and a
    # stripped head is the proven embed path. bll01's copy is untouched.
    rm -f "${LOCAL}/head.pt" "${LOCAL}/heads.pt"
    LOCAL_OF[${NAME}]="${LOCAL}"
done 3< "${PB_MANIFEST}"
echo "==> staged ${#LOCAL_OF[@]} checkpoint(s)"

# ── Frozen-TSFM weights, only if a staged checkpoint is one ──────────────────
# A PretrainedTSFM checkpoint is a config.json naming a Hub repo; the weights
# are not in it. Mirrored from bll01's cache, as slurm_tsfm_layers.sh does, so
# the job reads the revisions bll01 pinned -- but ONLY the three reported
# families (Kronos needs its tokenizer too), under its OWN marker: that
# script's list still names Sundial, which is no longer in bll01's cache, so
# sharing its marker would either fail here or mark a set complete that is
# not. Same lock, since both write the same hub dir. Offline, so a miss fails
# loudly instead of downloading a different revision.
if grep -lq '"PretrainedTSFM"' "${CKPT_STAGE}"/*/config.json 2>/dev/null; then
    phase stage_hf
    export HF_HOME="/hpc_temp/${USER}/hf"
    HF_HUB="${HF_HOME}/hub"
    HF_MARKER="${MARKER_DIR}/hf-tsfm-probe.${MARKER_VERSION}.v1.done"
    mkdir -p "${HF_HUB}" "${LOCK_DIR}" "${MARKER_DIR}"
    (
        flock -x 9
        if [ -f "${HF_MARKER}" ]; then
            echo "==> TSFM weights already staged"
        else
            echo "==> Staging TSFM weights from ${BLL01_HOST}"
            for REPO in models--amazon--chronos-2 \
                        models--google--timesfm-3.0-pytorch \
                        models--NeoQuasar--Kronos-base \
                        models--NeoQuasar--Kronos-Tokenizer-base; do
                rsync -az --no-motd \
                    "${BLL01_HOST}:~/.cache/huggingface/hub/${REPO}" "${HF_HUB}/" \
                    </dev/null || error "TSFM weight rsync failed: ${REPO}"
            done
            touch "${HF_MARKER}"
        fi
        refresh_mtimes "${HF_HUB}"
    ) 9> "${LOCK_DIR}/hf-tsfm.lock"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

# ── Shard task list ───────────────────────────────────────────────────────────
# The eval month repeats once per fit row; `sort -u` keeps it from being
# embedded six times (embed-many would skip the repeats, but they would still
# cost a task slot in every worker's stride).
phase build_tasks
TASKS="${CACHE_DIR}/tasks.txt"
while read -r CKPT FIT EV <&3; do
    [ -z "${CKPT:-}" ] && continue
    case "${CKPT}" in \#*) continue ;; esac
    LOCAL="${CKPT_STAGE}/$(basename "${CKPT}")"
    for ((s = 0; s < NUM_SHARDS; s++)); do
        echo "${LOCAL} ${FIT} ${s} ${FIT_ANCHORS}"
        echo "${LOCAL} ${EV} ${s} ${EVAL_ANCHORS}"
    done
done 3< "${PB_MANIFEST}" | sort -u > "${TASKS}"
echo "==> $(wc -l < "${TASKS}") shard tasks"

# ── Embed ─────────────────────────────────────────────────────────────────────
for PASS in 1 2; do
    phase "embed_pass${PASS}"
    echo "==> embed pass ${PASS} / 2"
    WPIDS=()
    for ((i = 0; i < N_WORKERS; i++)); do
        CUDA_VISIBLE_DEVICES=$((i % N_GPUS)) \
        uv run scripts/eval/probe_fit_size.py embed-many \
            --tasks "${TASKS}" \
            --worker-index "${i}" --num-workers "${N_WORKERS}" \
            --num-shards "${NUM_SHARDS}" \
            --out-dir "${CACHE_DIR}" \
            --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
            --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR}" \
            "${POOL_ARG[@]+"${POOL_ARG[@]}"}" \
            > "${CACHE_DIR}/embed_p${PASS}_w${i}.log" 2>&1 &
        WPIDS+=($!)
    done
    # WAIT ON THE WORKERS BY PID, NEVER BARE. The GPU sampler above is a
    # background child of THIS shell and `nvidia-smi -l 2` never exits, so a
    # bare `wait` blocks on it forever: the embed finishes, the shell sits in
    # wait4 holding an idle GPU, and reduce/push never run. That cost the first
    # wave its whole 12 h wall with every shard already on disk.
    wait "${WPIDS[@]}" || true
    NFAIL=$( { grep -hc "FAIL" "${CACHE_DIR}"/embed_p${PASS}_w*.log 2>/dev/null || true; } \
            | awk '{s+=$1} END{print s+0}')
    echo "    shards on disk: $(find "${CACHE_DIR}" -name '*.npz' | wc -l)"
    echo "    failures this pass: ${NFAIL}"
    [ "${NFAIL}" -eq 0 ] && break
    { grep -h "FAIL" "${CACHE_DIR}"/embed_p${PASS}_w*.log 2>/dev/null || true; } \
        | sed 's/.*: //' | sort | uniq -c | head
done

# ── Reduce, one eval month at a time ──────────────────────────────────────────
# cmd_reduce scores ONE eval panel per call and skips checkpoints with no
# shards for it, so the four arms of an eval month reduce together and the
# other groups in this job are simply not participants.
phase reduce
# GIVE THE REDUCE THE WHOLE ALLOCATION. It is ONE process -- 18 checkpoints x 8
# fit sizes x 3 alphas x 3 tasks of ridge, and the Gram matrix at n=1,152,000
# dominates -- so the single-thread setting the embed needs leaves 7 of 8 cores
# idle and costs ~4-5 h a job. gemm scales close to linearly here.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RC=0
for EV in $(awk 'NF && $0 !~ /^#/ {print $3}' "${PB_MANIFEST}" | sort -u); do
    echo "==> reduce ${EV}"
    uv run scripts/eval/probe_fit_size.py reduce \
        --out-dir "${CACHE_DIR}" \
        --eval-month-glob "${EV}" \
        --expect-shards "${NUM_SHARDS}" \
        --alphas 1 10 100 \
        --json "${CACHE_DIR}/${PB_TAG}-${EV}.json" || RC=$?
done
[ "${RC}" -ne 0 ] && echo "==> WARNING: at least one reduce exited non-zero"

phase push
# Stop sampling before the report reads the csv, so its last row is complete
# and the push itself is not measured as idle GPU (it is, but it is also the
# one phase that cannot be made to use one).
kill "${GPU_SAMPLER}" 2>/dev/null || true
wait "${GPU_SAMPLER}" 2>/dev/null || true

echo "════════════════════════════════════════════════════════════════"
uv run python scripts/eval/gpu_phase_report.py \
    --phases "${PHASE_FILE}" --gpu "${GPU_FILE}" \
    | tee "${CACHE_DIR}/${PB_TAG}-gpu_report.txt" || true
echo "════════════════════════════════════════════════════════════════"

ssh "${BLL01_HOST}" "mkdir -p ${RESULT_DEST}/${PB_TAG}"
# The instrumentation travels with the results: a wave that has to be resized
# is resized off these, and they are on a /hpc_temp the login node cannot see.
rsync -az "${CACHE_DIR}"/${PB_TAG}-*.json "${BLL01_HOST}:${RESULT_DEST}/" \
    || error "result push failed; jsons are at ${CACHE_DIR} on $(hostname)"
rsync -az "${PHASE_FILE}" "${GPU_FILE}" "${CACHE_DIR}/${PB_TAG}-gpu_report.txt" \
    "${BLL01_HOST}:${RESULT_DEST}/${PB_TAG}/" || true
echo "==> results at ${BLL01_HOST}:${RESULT_DEST}/"
echo "  Done: $(date)"
