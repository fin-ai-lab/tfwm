#!/bin/bash
#SBATCH --account=pi-bll
#SBATCH --partition=standard_l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=tsfm-layers
#SBATCH --exclude=pgpu015

# slurm_tsfm_layers.sh — every layer of EVERY frozen TSFM family, scored on the
# synchronized rank-IC panel for one (train, eval) month pair.
#
# ONE GPU, MANY CPUS. The usual scoring job (slurm_score_ckpts.sh) asks for 4
# GPUs and 32 workers because a ViT forward is trivial next to the per-ticker-
# day decode. Here it is the other way round: the decode runs at ~21
# ticker-days/s (~15 min for a 36-anchor month) while the TSFM forward runs at
# 50-150 views/s, so the GPU is the wall and a second one on the same job would
# idle. Each job asks for the CPU/memory a single GPU bills for free (7 CPUs,
# 240 GB; see the MAX_TRES billing note in pythia_docs).
#
# ONE JOB PER MONTH PAIR, ALL FOUR FAMILIES. The decode is a quarter of a
# Chronos-2 pass and nearly half a Kronos one, and it is identical for every
# family — so the families share it rather than each paying it. On the
# 32-month panel that is ~29 of the ~94 GPU-hours the sweep would otherwise
# cost. It also means each month is STAGED ONCE instead of four times.
#
# ASK FOR THE MEMORY IT USES, NOT FOR THE MEMORY THAT IS FREE. TimesFM's
# per-channel concat is d = 11,520 and the sweep holds one (d x d) float64 Gram
# per layer: 21 x 1.06 GB = 22 GB for TimesFM alone, ~31 GB for all four
# families. Measured peak RSS is 45 GB mid-fit and ~48 GB through the
# eigendecompositions, and it is MONTH-INDEPENDENT -- the Grams are d x d, only
# the number of rows streamed through them changes. 96 GB is 2x that.
#
# This asked 240 GB until 2026-08-20, reasoning that one GPU bills for it free
# (MAX_TRES; see the pythia billing note). Billing-free is not scheduling-free:
# at 240 GB only TWO jobs fit on a 755 GB node, so the 32-month sweep ran 6-wide
# behind 18 idle L40S GPUs. The billing is unchanged at 96 GB -- max(7 CPUs x 40,
# 96/1.3, 1 GPU x 320) is still the GPU's 320 -- and the packing is 3-4x better.
#
# CONTRACT (env vars):
#   TSFM_FAMILIES     space-separated subset of chronos2 timesfm timesfm3
#                     sundial kronos (default: the four REPORTED families, not
#                     all five — timesfm3 is opted into by name because it adds
#                     a second 22 GB of TimesFM-width Grams and decodes at 0.6x
#                     TimesFM 2.5's rate; see run_tsfm_layers.sh). They run in
#                     ONE process and SHARE the panel decode — see below.
#   TSFM_TRAIN_MONTH  YYYY-MM, fits the ridge (36 anchors/day)
#   TSFM_EVAL_MONTH   YYYY-MM, reports the IC (8 anchors/day)
#   TSFM_RESULT_DEST  bll01 directory for the result jsons (one per family)
#   TSFM_ALPHA_TABLE  repo-relative JSON of frozen alphas; when set the ridge
#                     ladder is NOT swept (see freeze_alphas.py)
#   XS_STATS_DIR      which anchor-stat table set to stage and score against.
#                     THE TABLE SET IS THE TARGET DEFINITION, not a path
#                     preference -- the (mu, sigma) it holds are OF a
#                     particular return/spread/vol definition, so a sweep run
#                     against the wrong one is measuring a different metric.
#                     Default xs_anchor_stats (the midpoint-target tables);
#                     xs_anchor_stats_fwdvwap60 is the two-forward-window
#                     definition of 2026-08-22.
#   TSFM_EXTRA        extra flags forwarded to tsfm_layer_ic.py
#
# Submitted by specific/run_tsfm_layers.sh.

set -euo pipefail

REPO_DIR="${HOME}/market-jepa"
source "${REPO_DIR}/scripts/pythia/lib/common.sh"

BLL01_HOST="bll01"
BLL01_DATA="${BLL01_DATA_DIR}"
DATA_DIR="/hpc_temp/${USER}/market-jepa-data"
LOCK_DIR="${DATA_DIR}/.locks"
MARKER_DIR="${DATA_DIR}/.markers"
source "${REPO_DIR}/scripts/pythia/lib/stage_data.sh"

: "${TSFM_TRAIN_MONTH:?ERROR: TSFM_TRAIN_MONTH not set.}"
: "${TSFM_EVAL_MONTH:?ERROR: TSFM_EVAL_MONTH not set.}"
TSFM_FAMILIES="${TSFM_FAMILIES:-chronos2 timesfm sundial kronos}"
TSFM_EXTRA="${TSFM_EXTRA:-}"
TSFM_ALPHA_TABLE="${TSFM_ALPHA_TABLE:-}"
[ -z "${TSFM_ALPHA_TABLE}" ] || TSFM_EXTRA="${TSFM_EXTRA} --alpha-table ${TSFM_ALPHA_TABLE}"
# stage_xs_anchor_stats reads this to pick BOTH the rsync source and the marker,
# so naming it here is enough to stage the right tables; the scorer below has
# to be pointed at the same subdirectory by hand.
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
# require_gpu_or_resubmit re-sbatches with --export=ALL and a hardcoded list of
# the TRAINING sweep's variables; these are not on it, so they only survive a
# no-GPU resubmit because they are in the environment. Make that explicit.
export TSFM_FAMILIES TSFM_TRAIN_MONTH TSFM_EVAL_MONTH TSFM_EXTRA TSFM_ALPHA_TABLE TSFM_SUP_RUNS XS_STATS_DIR

RESULT_DEST="${TSFM_RESULT_DEST:-/data/lab/tsfm_layer_ic}"
OUT_LOCAL="/hpc_temp/${USER}/tsfm-layers/${TSFM_TRAIN_MONTH}"
mkdir -p "${OUT_LOCAL}"
# Clear this job's own outputs up front so a rerun cannot mistake a previous
# attempt's json for one it wrote.
for FAM in ${TSFM_FAMILIES}; do
    rm -f "${OUT_LOCAL}/${FAM}_${TSFM_TRAIN_MONTH}.json"
done

echo "════════════════════════════════════════════════════════════════"
echo "  Job ${SLURM_JOB_ID} on $(hostname)"
echo "  families ${TSFM_FAMILIES}"
echo "  ${TSFM_TRAIN_MONTH} -> ${TSFM_EVAL_MONTH}  ->  ${RESULT_DEST}"
echo "  anchor tables ${XS_STATS_DIR}"
echo "  CPUs ${SLURM_CPUS_PER_TASK}  GPUs ${SLURM_GPUS_ON_NODE:-1}"
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
require_gpu_or_resubmit "${REPO_DIR}/scripts/pythia/slurm_tsfm_layers.sh"

# ── Stage the two months ──────────────────────────────────────────────────────
stage_month_list "${TSFM_TRAIN_MONTH}" "${TSFM_EVAL_MONTH}"
prepare_shm_limits

# PRE-BUILT PANELS. stage_month_list already calls stage_panel_month for both
# months, which is ON BY DEFAULT as of 2026-09-11 -- every sweep uses the cache
# (set STAGE_PANEL_CACHE=0 only to force a live decode). Until 2026-09-11 this export was missing and
# tsfm_layer_ic.py called iter_panel directly, so BOTH halves of the cache path
# were dead here: a staged panel sat on the node unread and every job re-decoded
# the month from the mosaic (~3028 s each, per stage_panel_month's note).
#
# Safe to set unconditionally. panel_source falls through to a live iter_panel
# decode for any (month, key) it cannot match -- identical contract, identical
# numbers -- so a node with no panel, a month never built, or a key mismatch
# all behave exactly as before.
if [ -d "${DATA_DIR}/panel_cache" ]; then
    export MJ_PANEL_CACHE="${DATA_DIR}/panel_cache"
    echo "==> panel cache: ${MJ_PANEL_CACHE}"
else
    echo "==> panel cache: none staged (STAGE_PANEL_CACHE=${STAGE_PANEL_CACHE:-1}); decoding live"
fi

# ── Stage the TSFM weights from bll01, do not download them ───────────────────
#
# The four families live on the Hub, and every job pulling them independently
# makes the run's success a function of Hub availability and of whether this
# node has egress at all. bll01 already holds the exact revisions the reported
# numbers were produced with, so mirror those instead: one flock-guarded copy
# per node, shared by every job that lands on it.
export HF_HOME="/hpc_temp/${USER}/hf"
HF_HUB="${HF_HOME}/hub"
# The staged REPO LIST has its own version, bumped whenever a family is added
# — a node holding the previous set would otherwise see the marker and skip the
# rsync, and the new family's weights would never arrive. Kept separate from
# MARKER_VERSION so adding a model does not re-verify every staged month on
# every node. v2: timesfm-3.0.
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
# Offline so a stale-cache miss is a loud failure rather than a silent download
# of a different revision than the one bll01 pinned.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ── Supervised specialists: stage the month's three backbones ────────────────
# sup_* families read a trained ViT trunk from the checkpoint tree, which lives
# on bll01 and is not mounted here. TSFM_SUP_RUNS is resolved AT SUBMIT TIME by
# run_tsfm_layers.sh (which runs where the tree is mounted) as a space-
# separated "project/run_id" list -- resolving it here meant a remote shell
# with three levels of quoting inside a set -e script, which is what killed
# jobs 196492-6 with no error line at all.
if [ -n "${TSFM_SUP_RUNS:-}" ]; then
    SUP_ROOT="/hpc_temp/${USER}/tsfm-layers/sup_ckpts"
    for SPEC in ${TSFM_SUP_RUNS}; do
        mkdir -p "${SUP_ROOT}/${SPEC}"
        rsync -az --no-motd \
            "${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${SPEC}/backbone.pt" \
            "${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${SPEC}/train_meta.json" \
            "${SUP_ROOT}/${SPEC}/" </dev/null \
            || error "supervised ckpt rsync failed: ${SPEC}"
        [ -s "${SUP_ROOT}/${SPEC}/backbone.pt" ] \
            || error "backbone.pt empty after staging: ${SPEC}"
        echo "==> staged ${SPEC}"
    done
    export MJ_SUP_CKPT_ROOT="${SUP_ROOT}"
fi

# xs_ic_eval decodes each month through tempfile.TemporaryDirectory, which
# honours TMPDIR and otherwise fills the small node-local /tmp.
export TMPDIR="/hpc_temp/${USER}/tsfm-layers/.tmp"
mkdir -p "${TMPDIR}"

# One process, so BLAS gets the whole allocation: the fit pass ends in one
# (d x d) float64 eigendecomposition per layer -- ~46 s each at TimesFM's
# width -- which is the only genuinely multi-core step in the job.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
       OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ── Sweep ─────────────────────────────────────────────────────────────────────
uv run scripts/eval/tsfm_layer_ic.py \
    --families ${TSFM_FAMILIES} \
    --train-month "${TSFM_TRAIN_MONTH}" \
    --eval-month "${TSFM_EVAL_MONTH}" \
    --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
    --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR}" \
    --out-dir "${OUT_LOCAL}" --overwrite ${TSFM_EXTRA}

# ── Push the result back ──────────────────────────────────────────────────────
# /hpc_temp is mounted on compute nodes ONLY; anything left here is unreachable
# from the login node and from bll01.
ssh "${BLL01_HOST}" "mkdir -p ${RESULT_DEST}"
# PUSH ONLY WHAT THIS JOB PRODUCED. OUT_LOCAL is keyed by train month alone, so
# a node that ran a DIFFERENT family set for the same month still has those
# jsons sitting here -- and `rsync OUT_LOCAL/*.json` swept them into this run's
# results directory. On 2026-08-21 that put 36 stale TSFM files into the
# supervised results, which is exactly the mixed-directory hazard the header of
# specific/run_tsfm_layers.sh warns about.
N=0
for FAM in ${TSFM_FAMILIES}; do
    F="${OUT_LOCAL}/${FAM}_${TSFM_TRAIN_MONTH}.json"
    [ -f "${F}" ] || error "expected output missing: ${F}"
    rsync -az "${F}" "${BLL01_HOST}:${RESULT_DEST}/" \
        || error "result push failed for ${FAM}; jsons are in ${OUT_LOCAL}"
    N=$((N + 1))
done
echo "==> ${N} results at ${BLL01_HOST}:${RESULT_DEST}/"

echo "════════════════════════════════════════════════════════════════"
echo "  Done: $(date)"
echo "════════════════════════════════════════════════════════════════"
