#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
# 240G = THIS NODE'S MEMORY DIVIDED BY ITS GPUs, less headroom (user,
# 2026-09-15). standard_hopper is RealMemory=2063255 MB over 8 H100s, so a
# GPU's fair share is 251.9 GiB; 8 x 240G = 1920 GiB leaves ~95 GiB for the
# OS and slurmd, and memory and GPUs still run out together. Billing is
# MAX_TRES and one GPU already bills 1000, so the ceiling is free here.
#
# IT WAS 64G FOR ONE DAY AND THAT COST THE WHOLE WAVE. The reasoning was
# that only ~7 GB is anonymous (/dev/shm dataloader tensors) and everything
# above it is reclaimable page cache off the mmap'd day store -- true, and
# beside the point. --mem IS the cache ceiling, and the SPAN IS 87 GB: the
# six 2021 day-store months measured 17+16+15+14+15+13 GB, and
# DayStoreCellDataset draws a fresh permutation over the WHOLE span every
# epoch, so access is uniformly random across all of it and a partial cache
# has no locality to exploit. Under a 60 GiB ceiling essentially every cell
# is re-read: measured 2026-09-15 on pgpu009 with six of these resident --
# GPU utilization 0%, load 33 of 64 cores, md0 reading 965 MB/s, and
# memory.failcnt at 7,079,147 on one job. 16-35 s/step against the 0.7 s/step
# the same recipe gets on the campaign, so a 1-hour job was pacing to 22-46
# hours and the wave was cancelled.
#
# SO THE RULE IS: THE CEILING MUST CLEAR THE TRAINING SPAN, not the anon
# footprint. A 12-month span would want more than this; check the span
# against ``du -sh`` on the staged day store before trusting 240G either.
#
# THE 2026-08-21 OOMs THIS USED TO CITE WERE THE GRID CACHE, and it is off.
# Four jobs died with OUT_OF_MEMORY and eight more wedged (a worker is
# OOM-killed, the main process waits on it forever, and the job burns its
# wall clock having trained for ~100 seconds) because the per-worker grid
# cache reached 10-13.6 GiB and 7 workers put an 86G job at the limit. That
# cache is ANONYMOUS memory and it is real -- but GRID_CACHE_GB=0 has been
# the default since the dense mosaic, so nothing allocates it. Raise this
# further, by workers x GRID_CACHE_GB, on any run that turns it on.
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=pgpu015,pgpu018

# TMPDIR MUST BE SET *AFTER* THE #SBATCH BLOCK. sbatch stops reading
# directives at the first line that is neither blank nor a comment, so an
# executable line above them silently discards EVERY ONE -- no --gres, no
# --cpus-per-task, no --time, no --exclude. That is what this preamble did
# when it sat under the shebang: jobs ran with cpu=1, mem from the command
# line, and NO GPU, and the only symptom was torch reporting
# "CUDA: False" on every node of every partition.
if [[ -z "${TMPDIR:-}" && -d /data/lab/tmp ]]; then
    export TMPDIR="/data/lab/tmp/${USER:-market-jepa}"
fi
mkdir -p "${TMPDIR:-/tmp}"

# slurm_variance_decomp.sh — train every (seed, series) combo for ONE training
# month inside a single allocation, for the variance-decomposition experiment.
# The launcher submits one job per (month, series), so VD_SERIES is normally a
# single value and the job runs 10 seeds of one objective (~6-10 h).
#
# Loops seed-major (seed 0: all series, seed 1: all series, ...) so that a
# wall-time truncation leaves a seed-balanced sample across series. Each run:
#   1. Trains a fresh ckpt (training.seed=$SEED — init + augs + shuffle order)
#      on the train month, probe-evaled on the next calendar month.
#   2. Rsyncs the ckpt dir back to bll01.
#
# Required env vars (set by run_variance_decomp.sh):
#   YM             — training month (YYYY-MM)
#   TRAIN_START    — YYYY-MM-01
#   TRAIN_END      — last day of YM
#   EVAL_START     — first day of the next calendar month
#   EVAL_END       — last day of the next calendar month
#   VD_SEEDS       — space-separated seed list (e.g. "0 1 2 3 4 5 6 7 8 9")
#   VD_SERIES      — space-separated series subset (default: all SWEEP_VALUES)
#   SWEEP_FILE_REL — relative path to scripts/sweeps/variance_decomp.sh
#   COMMIT_HASH    — short git hash for the wandb project suffix
#
# Optional:
#   XS_STATS_DIR   — which anchor-stat tables to stage, TRAIN against and
#                    SCORE against. This names a target definition, not a path
#                    preference: the tables are the (mu, sigma) OF a particular
#                    target, and all three targets became forward-window
#                    differences on 2026-08-22. Defaults to xs_anchor_stats
#                    (the retired mid-to-mid family) so old resubmissions keep
#                    reproducing themselves.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"

source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

: "${YM:?ERROR: YM not set.}"
: "${TRAIN_START:?ERROR: TRAIN_START not set.}"
: "${TRAIN_END:?ERROR: TRAIN_END not set.}"
: "${EVAL_START:?ERROR: EVAL_START not set.}"
: "${EVAL_END:?ERROR: EVAL_END not set.}"
: "${VD_SEEDS:?ERROR: VD_SEEDS not set.}"
: "${SWEEP_FILE_REL:?ERROR: SWEEP_FILE_REL not set.}"
: "${COMMIT_HASH:?ERROR: COMMIT_HASH not set.}"

SWEEP_FILE="${REPO_DIR}/${SWEEP_FILE_REL}"
[ -f "${SWEEP_FILE}" ] || { echo "ERROR: sweep file not found: ${SWEEP_FILE}" >&2; exit 1; }
source "${SWEEP_FILE}"
declare -F sweep_train_args >/dev/null || { echo "ERROR: sweep_train_args() missing" >&2; exit 1; }

VD_SERIES="${VD_SERIES:-${SWEEP_VALUES[*]}}"

# Read by stage_data.sh (which tables to rsync from bll01), by train.py, and by
# post_train_ic_eval.py below. All three MUST agree: a run that trains under
# one target family and is scored under another measures a distribution shift
# on top of the seed noise this sweep exists to isolate.
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
export XS_STATS_DIR

# One stable wandb project across all 10 months (analysis collects the full
# month x series x seed grid from a single project).
WANDB_SUFFIX="-${COMMIT_HASH}"
WANDB_PROJECT="${SWEEP_NAME}${WANDB_SUFFIX}"
WANDB_ENTITY="${WANDB_ENTITY:-boothai}"

# ── Job info ────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  variance-decomp : ${YM}"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Node            : $(hostname)"
echo "  Train month     : ${TRAIN_START} .. ${TRAIN_END}"
echo "  Eval  month     : ${EVAL_START} .. ${EVAL_END}"
echo "  Seeds           : ${VD_SEEDS}"
echo "  Series          : ${VD_SERIES}"
echo "  Commit          : ${COMMIT_HASH}"
echo "  Anchor tables   : ${XS_STATS_DIR}"
echo "════════════════════════════════════════════════════════════════"

# ── Install deps once ────────────────────────────────────────────────────────
# MUST precede stage_all: re-materialization runs `uv run` for the venv's
# zstandard decompressor (see stage_data.sh materialize_month), and without
# UV_PROJECT_ENVIRONMENT that would implicitly sync ${REPO_DIR}/.venv on
# shared storage — six ways in parallel under STAGE_MAX_PARALLEL.

echo ""
echo "==> Installing dependencies with uv ..."
export PATH="${HOME}/.local/bin:${PATH}"
export WANDB_DIR="/hpc_temp/${USER}/wandb"
mkdir -p "${WANDB_DIR}"
export MARKET_JEPA_TEMP_PROBE_EVAL_DIR="/hpc_temp/${USER}/market-jepa-temp-probe-eval"
mkdir -p "${MARKET_JEPA_TEMP_PROBE_EVAL_DIR}"
VENV_DIR="/hpc_temp/${USER}/market-jepa-venv-${SLURM_JOB_ID}"
export UV_CACHE_DIR="/hpc_temp/${USER}/.cache/uv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE=copy
trap 'rm -rf "${VENV_DIR}" "${UV_CACHE_DIR}"' EXIT
cd "${REPO_DIR}"
uv sync --frozen --quiet

echo "==> Python: $(uv run python --version)"
echo "==> torch:  $(uv run python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# ── Stage data (train month + eval month) ────────────────────────────────────

# DAYSTORE=1: training reads the day-major store and only the scorer's months
# come from the mosaic, exactly as the campaign bundles do. Same config, same
# staging cost -- a six-month span is ~180 GB of mosaic and ~10 GB of daystore,
# and the views the loader emits are equal element for element
# (tests/test_cell_dataset.py); only the cell sampling differs, measured at
# -0.002 +- 0.001 over nine months with no consistent sign.
#
# LEJEPA CANNOT USE IT. dataset.backend=days serves supervised cross_stock
# cells only, so that series stays on the mosaic and says so.
VD_DAYSTORE="${DAYSTORE:-0}"
if [ "${VD_DAYSTORE}" = "1" ] && [ "${VD_SERIES:-}" = "lejepa" ]; then
    echo "==> lejepa: daystore serves supervised cells only; staging the mosaic instead."
    VD_DAYSTORE=0
fi
if [ "${VD_DAYSTORE}" = "1" ]; then
    stage_all_daystore "${TRAIN_START}" "${TRAIN_END}" "${EVAL_END}"
else
    stage_all "${TRAIN_START}" "${EVAL_END}"
fi

# LEJEPA HAS NO HEAD, so the probe is the only thing there is to score and its
# panel is genuinely required. POST_TRAIN_PROBE=0 is about dropping a redundant
# probe next to a trained head; it cannot apply here.
if [ "${VD_SERIES:-}" = "lejepa" ]; then
    POST_TRAIN_PROBE=1
    export POST_TRAIN_PROBE
fi

# ── The eval panel is an INPUT, not an optimization ──────────────────────────
# Fail now rather than train ten seeds and then re-decode the panel the cache
# was supposed to hold. See require_panel_cache in lib/stage_data.sh.
if ! require_panel_cache "${TRAIN_END:0:7}" "${EVAL_START:0:7}"; then
    echo "==> ABORTING: panel cache missing for ${TRAIN_END:0:7} / ${EVAL_START:0:7}." >&2
    echo "    MJ_PANEL_CACHE=<root> uv run scripts/eval/build_panel_cache.py \\" >&2
    echo "        --months ${TRAIN_END:0:7} --roles probe eval --jobs 40" >&2
    exit 1
fi

require_gpu_or_resubmit "${REPO_DIR}/scripts/pythia/specific/slurm_variance_decomp.sh"

# ── Loop seed-major over (seed x series) ─────────────────────────────────────
# Don't let one run's failure kill the whole bundle.

set +e
SUCCEEDED=()
FAILED=()

for SEED in ${VD_SEEDS}; do
  for SERIES in ${VD_SERIES}; do
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Run: ${YM} / ${SERIES} / seed=${SEED}"
    echo "────────────────────────────────────────────────────────────────"

    TRAIN_ARGS=$(sweep_train_args "${SERIES}")
    if [ -z "${TRAIN_ARGS}" ]; then
        echo "==> sweep_train_args failed for '${SERIES}'; skipping."
        FAILED+=("${SERIES}/seed=${SEED} (args)")
        continue
    fi
    TRAIN_ARGS="${TRAIN_ARGS} dataset.train_date_start=${TRAIN_START} dataset.train_date_end=${TRAIN_END}"
    TRAIN_ARGS="${TRAIN_ARGS} dataset.eval_train_date_start=${TRAIN_START} dataset.eval_train_date_end=${TRAIN_END}"
    TRAIN_ARGS="${TRAIN_ARGS} dataset.eval_date_start=${EVAL_START} dataset.eval_date_end=${EVAL_END}"
    TRAIN_ARGS="${TRAIN_ARGS} training.seed=${SEED}"
    if [ "${VD_DAYSTORE}" = "1" ]; then
        TRAIN_ARGS="${TRAIN_ARGS} dataset.backend=days machine.daystore_dir=${DATA_DIR}/1Hz_daystore"
    fi
    [ -n "${VD_EXTRA_ARGS:-}" ] && TRAIN_ARGS="${TRAIN_ARGS} ${VD_EXTRA_ARGS}"
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed "s|wandb\.project=\([^ ]*\)|wandb.project=\1${WANDB_SUFFIX}|")
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed "s|wandb\.run_name=\([^ ]*\)|wandb.run_name=\1_${YM}_seed-${SEED}|")
    RUN_NAME=$(echo "${TRAIN_ARGS}" | grep -oP '(?<=wandb\.run_name=)[^ ]+' | head -1)

    # ── Skip-if-done (resubmission aid) ─────────────────────────────────────
    # Skip only when a *finished* wandb run with this name exists AND its ckpt
    # dir is already on bll01. (Listing state can be stale — worst case here
    # is a wasted retrain, never a silent gap.)
    EXISTING_RUN_ID=$(uv run python - <<EOF 2>/dev/null
import wandb
try:
    api = wandb.Api()
    runs = list(api.runs(
        "${WANDB_ENTITY}/${WANDB_PROJECT}",
        filters={"display_name": "${RUN_NAME}", "state": "finished"},
    ))
    print(runs[0].id if runs else "")
except Exception:
    print("")
EOF
)
    if [ -n "${EXISTING_RUN_ID}" ]; then
        REMOTE_CKPT="/data/lab/market-jepa-checkpoints/${WANDB_PROJECT}/${EXISTING_RUN_ID}"
        HAS_CKPT=$(ssh "${BLL01_HOST}" "test -d ${REMOTE_CKPT} && ls -A ${REMOTE_CKPT} | head -1" 2>/dev/null)
        if [ -n "${HAS_CKPT}" ]; then
            echo "==> ${RUN_NAME} already done (run ${EXISTING_RUN_ID}, ckpt on bll01); skipping."
            SUCCEEDED+=("${SERIES}/seed=${SEED} (cached)")
            continue
        fi
    fi

    TRAIN_LOG=$(mktemp)

    # shellcheck disable=SC2086
    uv run train.py \
        machine=pythia \
        machine.mosaic_dir="${DATA_DIR}/1Hz_mosaic_mnth" \
        machine.risk_factor_dir="${DATA_DIR}/1Hz_risk_factors" \
        machine.metadata_path="${DATA_DIR}/metadata.parquet" \
        dataset.xs_anchor_stats_dir="${DATA_DIR}/${XS_STATS_DIR}" \
        checkpoint.chkpt_dir="/hpc_temp/${USER}/market-jepa-checkpoints" \
        checkpoint.remote_dir="/data/lab/market-jepa-checkpoints" \
        probe_eval.targets.horizons=[900] \
        probe_eval.targets.types=[return] \
        probe_eval.num_threads=1 \
        ${TRAIN_ARGS} 2>&1 | tee "${TRAIN_LOG}"
    RC=${PIPESTATUS[0]}

    if [ "${RC}" -ne 0 ]; then
        echo "==> Train FAILED for ${SERIES}/seed=${SEED} (exit ${RC})."
        FAILED+=("${SERIES}/seed=${SEED}")
        rm -f "${TRAIN_LOG}"
        continue
    fi

    CKPT_PATH=$(grep -oP '(?<=Saved checkpoint to ).*' "${TRAIN_LOG}" | tail -1)
    rm -f "${TRAIN_LOG}"

    if [ -z "${CKPT_PATH}" ] || [ ! -d "${CKPT_PATH}" ]; then
        echo "==> Could not resolve checkpoint path from train log for ${SERIES}/seed=${SEED}."
        FAILED+=("${SERIES}/seed=${SEED} (no ckpt)")
        continue
    fi

    RUN_ID=$(basename "${CKPT_PATH}")
    CKPT_PROJECT=$(basename "$(dirname "${CKPT_PATH}")")

    # Score BEFORE the sync, so xs_ic.json travels with the checkpoint it
    # describes. This bundle used to train and sync only, which left 400
    # checkpoints on bll01 with backbone.pt, head.pt, train_meta.json and no
    # metric of any kind -- recoverable, but only by re-decoding all ten
    # months a second time. A run that is not scored is a run that has to be
    # visited twice.
    #
    # Same invocation as lib/train_bundle_body.sh: the probe fits on the LAST month of
    # the training span (identical to the first for this single-month sweep)
    # and reports on the eval month. Non-fatal on failure -- a checkpoint
    # without a score is still worth syncing, which is exactly the situation
    # xs_score_many.py exists to repair.
    if [ "${POST_TRAIN_IC_EVAL:-1}" = "1" ]; then
        echo "==> Post-training IC eval for ${SERIES}/seed=${SEED} ..."
        # HEAD ONLY where there IS a head. POST_TRAIN_PROBE=0 drops the ridge
        # and with it the probe-fit month's panel, which is 36 anchors/day
        # against the eval month's 8. The lejepa series has no head at all, so
        # it keeps the probe whatever the flag says -- it is the only thing
        # there is to score.
        VD_HEAD_ONLY=""
        if [ "${POST_TRAIN_PROBE:-1}" = "0" ] && [ "${SERIES}" != "lejepa" ]; then
            VD_HEAD_ONLY="--head-only"
        fi
        uv run scripts/generic/post_train_ic_eval.py \
            ${VD_HEAD_ONLY} \
            --ckpt-dir "${CKPT_PATH}" \
            --train-month "${TRAIN_END:0:7}" \
            --eval-month "${EVAL_START:0:7}" \
            --wandb-run-id "${RUN_ID}" \
            --wandb-project "${WANDB_PROJECT}" \
            --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
            --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR}" \
            --risk-factor-dir "${DATA_DIR}/1Hz_risk_factors" \
            || { echo "==> post-training IC eval FAILED for ${SERIES}/seed=${SEED}"; \
                 SCORE_FAILED=1; }
    fi

    # Sync the ckpt dir back to bll01.
    CKPT_DST="${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${CKPT_PROJECT}/${RUN_ID}/"
    echo "==> Syncing ${CKPT_PATH} -> ${CKPT_DST} ..."
    ssh "${BLL01_HOST}" "mkdir -p /data/lab/market-jepa-checkpoints/${CKPT_PROJECT}"
    if rsync -az --info=progress2 "${CKPT_PATH}/" "${CKPT_DST}"; then
        SUCCEEDED+=("${SERIES}/seed=${SEED}")
    else
        echo "==> WARNING: ckpt rsync FAILED for ${SERIES}/seed=${SEED}."
        FAILED+=("${SERIES}/seed=${SEED} (ckpt-sync)")
    fi
  done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done: $(date)"
echo "  Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
echo "  Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "════════════════════════════════════════════════════════════════"

# A SCORING FAILURE MUST FAIL THE JOB. It used to be swallowed with a WARNING
# so the checkpoint would still sync -- which it does, above, before this line.
# But the job then exited 0, sacct said COMPLETED, and the only evidence was a
# line in one slurm log. On 2026-08-23 that hid 51 checkpoints that had trained
# for hours and scored nothing (a keyword argument added to panel_kwargs_for
# and not to embed_month); it surfaced only when someone listed checkpoint dirs
# looking for xs_ic.json. The checkpoint is already safe; what is left is to
# say so out loud.
if [ "${SCORE_FAILED:-0}" = "1" ]; then
    echo "==> FAILING THE BUNDLE: at least one run synced but scored nothing."
    exit 1
fi

# Exit non-zero only if every run failed; partial success is normal when one
# of 20 runs crashes mid-bundle.
[ ${#SUCCEEDED[@]} -gt 0 ]
