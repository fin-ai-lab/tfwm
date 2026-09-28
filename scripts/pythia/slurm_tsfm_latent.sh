#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=tsfm-latent
#SBATCH --exclude=pgpu015

# slurm_tsfm_latent.sh — the LATENT half of the frozen-TSFM layer sweep, for
# ONE eval month. The IC half is slurm_tsfm_layers.sh.
#
# Four analyses, every layer of every family:
#   phase 1  fixed-panel latent structure T1-T4   (~64% of the cost)
#   phase 2  full-day per-layer embeddings + the untrained ViT floor
#   phase 3  the return panel and its Pelger latent factors
#   phase 4  loading decode / subspace alignment
#
# ONE MONTH PER JOB. Serially this is ~1.5 h per month, so the 32-month panel
# is ~49 h on one GPU; per-month it is ~1.5 h wall. The months are genuinely
# independent -- every analysis takes --months and every statistic is pooled
# ACROSS months at the end, never within -- so the split changes nothing but
# the wall clock. merge_latent.py reassembles the per-month jsons and redoes
# the pooling with the same arithmetic the serial run uses.
#
# MEMORY IS MEASURED, NOT GUESSED: MaxRSS is 28.2 GB for a full four-family
# month (canary 195846), so 64 GB is 2.3x peak. It was 120G on the first
# submission, which was a guess -- and on a 755 GB L40S node the request,
# not the usage, is what decides how many jobs fit. See the billing note.
#
# CACHES ARE NODE-LOCAL. The pipeline is built around two caches that live on
# bll01 (ff_fullday_cache, factor_structure_cache) and compute nodes do not
# mount /data/lab. Both are env-overridable, so each job builds its own month
# under /hpc_temp and only the (small) result jsons are pushed back. The
# embeddings alone are ~38 GB per month -- pushing those back would be 1.2 TB
# for no reason, since nothing downstream reads them again.
#
# CONTRACT (env vars):
#   TSFM_EVAL_MONTH    YYYY-MM, the month to analyse
#   TSFM_FAMILIES      subset of chronos2 timesfm timesfm3 sundial kronos
#                      (default: the four reported ones; timesfm3 by name)
#   TSFM_SERIES_FAMILIES  families phase 4 writes (default: TSFM_FAMILIES).
#                      Set to the FULL list when adding one family, or the
#                      others' Pelger results for this month are dropped.
#   TSFM_RESULT_DEST   bll01 directory for the per-month result jsons
#
# Submitted by specific/run_tsfm_latent.sh.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
# stage_data.sh reads BLL01_DATA when a month is NOT already staged; without it
# the rsync path dies on "unbound variable" under set -u, and only for months
# this node has never seen -- so it hides on a warm node and fails on a cold one.
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"
source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

: "${TSFM_EVAL_MONTH:?ERROR: TSFM_EVAL_MONTH not set.}"
TSFM_FAMILIES="${TSFM_FAMILIES:-chronos2 timesfm sundial kronos}"
RESULT_DEST="${TSFM_RESULT_DEST:-/data/lab/tsfm_latent_32}"
export TSFM_EVAL_MONTH TSFM_FAMILIES TSFM_SUP_RUNS

YM="${TSFM_EVAL_MONTH}"
# DISTINCT TAG PER FAMILY KIND. fixed_panel_metrics MERGES over an existing
# output file (so a --models subset run refreshes only those entries), and the
# file lives in the repo tree on the node. With one shared tag, a supervised
# job landing where the TSFM job ran for the same month merged its 36 keys into
# that file's 60 and pushed a 97-model result. Same month, same filename,
# different panel.
case "${TSFM_FAMILIES}" in
    *sup_*) TAG="_supl_${YM}" ;;
    *)      TAG="_tsfml_${YM}" ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo "  Job ${SLURM_JOB_ID} on $(hostname)"
echo "  latent sweep, eval month ${YM}"
echo "  families ${TSFM_FAMILIES}   ->  ${RESULT_DEST}"
echo "════════════════════════════════════════════════════════════════"

export PATH="${HOME}/.local/bin:${PATH}"
VENV_DIR="/hpc_temp/${USER}/market-jepa-venv-${SLURM_JOB_ID}"
export UV_CACHE_DIR="/hpc_temp/${USER}/.cache/uv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE=copy
trap 'rm -rf "${VENV_DIR}" "${UV_CACHE_DIR}"' EXIT
cd "${REPO_DIR}"
# uv sync, WITH RETRIES. The project has a git dependency and UV_CACHE_DIR is
# per-job (above), so every job clones from github.com at startup, and pythia's
# compute nodes drop outbound connections often enough to lose real work. Job
# 207733 died here in 2 minutes with "Failed to connect to github.com port 443"
# — a burned queue slot before a single view was decoded. The same loop is in
# lib/train_bundle_body.sh; this file carries its own venv
# block and was never patched when those were.
_n=0
until uv sync --frozen --quiet; do
    _n=$((_n + 1))
    if [ "${_n}" -ge "${UV_SYNC_RETRIES:-4}" ]; then
        echo "ERROR: uv sync failed ${_n}x (network?); giving up." >&2
        exit 1
    fi
    echo "==> uv sync failed (attempt ${_n}); retrying in $((_n * 30))s ..." >&2
    sleep $((_n * 30))
done
require_gpu_or_resubmit "${REPO_DIR}/scripts/pythia/slurm_tsfm_latent.sh"

# The analyses read the eval month itself; build_return_panel needs it whole.
stage_month_list "${YM}"
prepare_shm_limits

# ── TSFM weights from bll01, never the Hub (same contract as the IC half) ─────
export HF_HOME="/hpc_temp/${USER}/hf"
HF_HUB="${HF_HOME}/hub"
# See slurm_tsfm_layers.sh: the repo list carries its own version so adding a
# family re-stages the weights without re-verifying every staged month.
HF_TSFM_SET="v2"
HF_MARKER="${MARKER_DIR}/hf-tsfm.${MARKER_VERSION}.${HF_TSFM_SET}.done"
mkdir -p "${HF_HUB}" "${LOCK_DIR}" "${MARKER_DIR}"
(
    flock -x 9
    if [ -f "${HF_MARKER}" ]; then
        echo "==> TSFM weights already staged"
    else
        echo "==> Staging TSFM weights from ${BLL01_HOST}"
        for REPO in models--amazon--chronos-2 \
                    models--google--timesfm-2.5-200m-pytorch \
                    models--google--timesfm-3.0-pytorch \
                    models--thuml--sundial-base-128m \
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

# gvkey_ticker_history.parquet (1.3 MB) lives on bll01 and phase 4 reads it
# for firm identity. One flock-guarded copy per node, like the HF weights.
GVKEY_LOCAL="/hpc_temp/${USER}/tsfm-latent/gvkey_ticker_history.parquet"
mkdir -p "$(dirname "${GVKEY_LOCAL}")" "${LOCK_DIR}"
(
    flock -x 9
    [ -s "${GVKEY_LOCAL}" ] || rsync -az --no-motd \
        "${BLL01_HOST}:/data/lab/market-text-data/gvkey_ticker_history.parquet" \
        "${GVKEY_LOCAL}" </dev/null || error "gvkey rsync failed"
) 9> "${LOCK_DIR}/gvkey.lock"
[ -s "${GVKEY_LOCAL}" ] || error "gvkey parquet missing after staging"
export MJ_GVKEY_PATH="${GVKEY_LOCAL}"

# Supervised trunks, when the family list asks for them. TSFM_SUP_RUNS is
# resolved at SUBMIT time (where /data/lab is mounted) as "project/run_id".
if [ -n "${TSFM_SUP_RUNS:-}" ]; then
    SUP_ROOT="/hpc_temp/${USER}/tsfm-latent/sup_ckpts"
    for SPEC in ${TSFM_SUP_RUNS}; do
        mkdir -p "${SUP_ROOT}/${SPEC}"
        rsync -az --no-motd \
            "${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${SPEC}/backbone.pt" \
            "${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${SPEC}/train_meta.json" \
            "${SUP_ROOT}/${SPEC}/" </dev/null \
            || error "supervised ckpt rsync failed: ${SPEC}"
        [ -s "${SUP_ROOT}/${SPEC}/backbone.pt" ] || error "empty backbone: ${SPEC}"
        echo "==> staged ${SPEC}"
    done
    export MJ_SUP_CKPT_ROOT="${SUP_ROOT}"
fi

export TMPDIR="/hpc_temp/${USER}/tsfm-latent/.tmp"
mkdir -p "${TMPDIR}"
# The plots/ analyses hardcode BLL01MachineConfig, whose paths are bll01's.
# machine_from_env applies these; without them the job reads /data/lab, which
# a compute node does not mount, and dies looking for the mosaic index.
export MJ_MOSAIC_DIR="${DATA_DIR}/1Hz_mosaic_mnth"
export MJ_RISK_FACTOR_DIR="${DATA_DIR}/1Hz_risk_factors"
export MJ_METADATA_PATH="${DATA_DIR}/metadata.parquet"
export FF_FULLDAY_CACHE="/hpc_temp/${USER}/tsfm-latent/ff_fullday_cache"
export FACTOR_STRUCTURE_CACHE="/hpc_temp/${USER}/tsfm-latent/factor_structure_cache"
mkdir -p "${FF_FULLDAY_CACHE}" "${FACTOR_STRUCTURE_CACHE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
       OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

FS=plots/latent_eval/factors
# A family is either a frozen TSFM (TSFM_KEYS, depths 0..N) or a supervised
# specialist (SUP_KEYS, depths 1..12 -- depth 0 is the CLS token before any
# attention and is input-independent).
keys_for() {
    uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import TSFM_KEYS, SUP_KEYS
reg = {**TSFM_KEYS, **SUP_KEYS}
print(','.join(reg['$1']))"
}

echo "=== phase 1: fixed-panel latent structure (T1-T4) ==="
for FAM in ${TSFM_FAMILIES}; do
    echo "--- ${FAM}"
    uv run python plots/latent_eval/fixed_panel/fixed_panel_metrics.py \
        --months "${YM}" --models "$(keys_for "${FAM}")" --out-suffix "${TAG}"
done
# The floor. Not optional: several of these tasks put an untrained ViT well
# above chance, so a ratio-over-chance alone does not say whether a layer
# learned anything.
uv run python plots/latent_eval/fixed_panel/fixed_panel_metrics.py \
    --months "${YM}" --models random --out-suffix "${TAG}"

echo "=== phase 2: full-day embeddings + the untrained floor ==="
TSFM_ONLY=""; SUP_ONLY=""
for F in ${TSFM_FAMILIES}; do
    case "${F}" in sup_*) SUP_ONLY="${SUP_ONLY} ${F}" ;;
                   *)     TSFM_ONLY="${TSFM_ONLY} ${F}" ;; esac
done
uv run python $FS/build_fullday_embs.py --months "${YM}" \
    --families ${TSFM_ONLY:-} --sup-families ${SUP_ONLY:-}

echo "=== phase 3: the return panel and its latent factors ==="
uv run python $FS/build_return_panel.py "${YM}"
uv run python $FS/estimate_factors.py "${YM}"

# PHASE 4 IS NOT INCREMENTAL, PHASES 1-2 ARE. decode_loadings.py and
# subspace_alignment.py OVERWRITE their json with exactly the --series handed
# to them, while fixed_panel_metrics merges and build_fullday_embs skips what
# is cached. So a run for one NEW family would leave phases 1-2 right and
# silently drop every other family's Pelger results for this month. Phase 4
# reads TSFM_SERIES_FAMILIES, defaulting to TSFM_FAMILIES (the historical
# behaviour, correct when the run covers every family) — set it to the full
# list when adding a family incrementally. The re-run costs no forward pass:
# phase 4 reads the cached full-day embeddings.
TSFM_SERIES_FAMILIES="${TSFM_SERIES_FAMILIES:-${TSFM_FAMILIES}}"
SERIES="$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import TSFM_KEYS, SUP_KEYS
reg = {**TSFM_KEYS, **SUP_KEYS}
fams = '${TSFM_SERIES_FAMILIES}'.split()
print(' '.join(k for f in fams for k in reg[f]))") randvit_s0 randvit_s1 randvit_s2 randvit_s3 randvit_s4"

echo "=== phase 4: Pelger latent-factor analyses ==="
uv run python $FS/decode_loadings.py     --months "${YM}" --series ${SERIES} --tag "${TAG}"
uv run python $FS/subspace_alignment.py  --months "${YM}" --series ${SERIES} --tag "${TAG}"

# ── Push back the jsons only ─────────────────────────────────────────────────
# /hpc_temp is compute-node-only; anything left here is unreachable from bll01.
# The embeddings stay behind on purpose (~38 GB/month, nothing reads them again).
ssh "${BLL01_HOST}" "mkdir -p ${RESULT_DEST}"
N=0
for F in "plots/latent_eval/fixed_panel/fixed_panel_P3S2${TAG}.json" \
         "$FS/decode_loadings${TAG}.json" \
         "$FS/subspace_alignment${TAG}.json"; do
    [ -f "${F}" ] || error "expected output missing: ${F}"
    rsync -az "${F}" "${BLL01_HOST}:${RESULT_DEST}/" || error "push failed: ${F}"
    N=$((N + 1))
done
echo "==> ${N} jsons at ${BLL01_HOST}:${RESULT_DEST}/"

echo "════════════════════════════════════════════════════════════════"
echo "  Done: $(date)"
echo "════════════════════════════════════════════════════════════════"
