#!/bin/bash
if [[ -z "${TMPDIR:-}" && -d /data/lab/tmp ]]; then
    export TMPDIR="/data/lab/tmp/${USER:-market-jepa}"
fi
mkdir -p "${TMPDIR:-/tmp}"
# run_score_ckpts.sh — score existing checkpoints on pythia instead of bll01.
#
# Takes ONE manifest — the same TSV xs_ic_series.py reduce consumes:
#
#     <ckpt_dir>\t<fit_month>\t<eval_month>
#
# and splits it into SLURM jobs, each staging only the months its own rows
# need. So the same file scores a 4-checkpoint spot check locally and a
# 200-checkpoint panel on the cluster; nothing about the measurement changes.
#
# WHY CHUNK BY MONTHS AND NOT BY CHECKPOINT. Staging dominates a scoring job:
# a mosaic month is ~6-7 GB and every job pays for its own copy on the node it
# lands on, while the GPU pass over one (checkpoint, month) is minutes. So the
# unit that must be packed is the MONTH, not the checkpoint. Rows are grouped
# by their (fit, eval) pair and pairs are accumulated until a job holds
# MONTHS_PER_JOB distinct months; every checkpoint sharing those months rides
# along for free. For a per-month series (checkpoint trained on M, scored on
# M+1) consecutive pairs overlap by one month, so the marginal cost of a second
# pair in a job is one month, not two.
#
# CKPT PATHS ARE RESOLVED ON BLL01 — compute nodes rsync them from there — so
# run this on bll01 and give ordinary local paths.
#
# Usage:
#   ./scripts/pythia/specific/run_score_ckpts.sh --manifest ar_spec.tsv --tag spec
#   MONTHS_PER_JOB=8 ./scripts/pythia/specific/run_score_ckpts.sh \
#       --manifest m.tsv --tag full-month --head
#
# Results land on bll01 at /data/lab/score_results/<tag>-partNN.json (one per
# job); concatenate them, or point plots at the directory.
#
# XS_STATS_DIR PICKS THE TARGET DEFINITION, not a path. It names the anchor-stat
# table set the panel is z-scored against and the reduce un-z-scores with, so
# results carrying different values of it are NOT comparable and must not share
# a --tag. Give a new definition a new tag:
#
#   XS_STATS_DIR=xs_anchor_stats_fwdvwap60 \
#       ./scripts/pythia/specific/run_score_ckpts.sh --manifest m.tsv --tag randinit-fwd3

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

PARTITION="${PARTITION:-standard_l40s}"
# CROSS-EVALUATION ONLY, forwarded to probe_fit_size._panel_for. Setting it
# scores every checkpoint in the manifest on a view length that is NOT its own,
# which is a different measurement -- give such a run its own --tag so it can
# never be read as the checkpoint's reported score.
MJ_FORCE_SEQ_LEN="${MJ_FORCE_SEQ_LEN:-}"
MONTHS_PER_JOB="${MONTHS_PER_JOB:-8}"
# SCORE_AUC=0 reports rank IC only -- see slurm_score_ckpts.sh. Forwarded here
# so a caller can set it once beside the other env knobs.
SCORE_AUC="${SCORE_AUC:-1}"
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
N_WORKERS="${N_WORKERS:-32}"
# SBATCH_EXTRA passes through to sbatch, which is how the 4-GPU/32-CPU/128G
# shape in slurm_score_ckpts.sh gets trimmed. That shape is sized for
# standard_l40s, where it bills 1280 (CPU and GPU tie). The SAME job on
# standard_hopper bills max(32*142.857, 400*4.069, 4*1000) = 4571, because
# hopper weights a core at 142.857 against l40s's 40 — so five of them eat
# 22855 of the 36000 the qos_bll group has, and pend behind any training.
# On hopper prefer: SBATCH_EXTRA="--gres=gpu:1 --cpus-per-task=8 --mem=128G"
# N_WORKERS=8, which bills 1143.
#
# MEMORY IS NOT WHAT BILLS THESE, so do not trim it hoping for priority. At
# 8 CPUs / 1 GPU the bill is max(1143, mem*4.069, 1000) and memory only enters
# above 281 GB; at the 14-CPU/2-GPU shape the forward-decay campaign uses, CPU
# and GPU tie at 2000 and memory is irrelevant below 491 GB. Trim --mem to pack
# more jobs onto a node (it is mostly page cache -- see slurm_score_ckpts.sh),
# and trim CPUs and GPUs to pay less for the queue. They are different levers.
# 8h WAS TOO TIGHT AND FAILED SILENTLY-ISH. The fwdv2pairs-581eb2 wave ran
# 6h12 / 6h+ / 8h+ per part against an 8h wall, and part05 hit it at row 120
# of 135. A TIMEOUT loses EVERYTHING: results are written once at the end, so
# eight hours of GPU produced no file and the wave sat one part short with
# nothing to show which part or why. Sized at ~2x the slowest observed part
# rather than at the mean; the cost of an over-long wall is queue priority,
# the cost of a short one is the whole job.
TIME_LIMIT="${TIME_LIMIT:-0-20:00:00}"
SCORE_PROBE=1
MANIFEST=""
TAG=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --probe)    SCORE_PROBE=1; shift ;;
        --head)     SCORE_PROBE=0; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) error "unknown argument: $1" ;;
    esac
done

[ -n "${MANIFEST}" ] || error "usage: $0 --manifest <tsv> --tag <name> [--head] [--dry-run]"
[ -n "${TAG}" ] || error "--tag is required (it names the cache dir and result json)"
[ -f "${MANIFEST}" ] || error "manifest not found: ${MANIFEST}"
[[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || error "--tag must be filename-safe: ${TAG}"

# ── Validate before submitting anything ───────────────────────────────────────
# A bad row costs an 8-hour queue slot and dies after staging, so check the
# whole manifest up front: every checkpoint must exist and carry a backbone.
NROW=0
while read -r CKPT FIT EV <&3; do
    [ -z "${CKPT:-}" ] && continue
    case "${CKPT}" in \#*) continue ;; esac
    NROW=$((NROW + 1))
    [ -n "${EV:-}" ] || error "row ${NROW}: expected 3 columns <ckpt> <fit> <eval>"
    # A checkpoint is loadable if load_model can build it, and that is three
    # different layouts: SSL/JEPA is config.json + model.pt, I-JEPA is model.pt
    # alone, supervised is backbone.pt (+ head.pt). Demanding backbone.pt
    # rejected every LeJEPA run outright -- the cluster scoring path could not
    # score the SSL arm at all, which is most of what this repo trains.
    [ -e "${CKPT}/backbone.pt" ] || [ -e "${CKPT}/model.pt" ] \
        || error "row ${NROW}: no backbone.pt or model.pt in ${CKPT}"
    [[ "${FIT}" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "row ${NROW}: bad fit month '${FIT}'"
    [[ "${EV}"  =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || error "row ${NROW}: bad eval month '${EV}'"
done 3< "${MANIFEST}"
[ "${NROW}" -gt 0 ] || error "manifest has no rows"
info "${MANIFEST}: ${NROW} checkpoints, $(awk 'NF && $0!~/^#/{print $2"\n"$3}' "${MANIFEST}" | sort -u | wc -l) distinct months"

# The anchor tables have to exist for every month in the manifest BEFORE
# anything is submitted: a job that stages 8 months and then dies on "Missing
# anchor-stat tables" has burned a queue slot for nothing, and a table set that
# is still being built is missing exactly its most recent months.
XS_TABLE_DIR="${BLL01_DATA_DIR}/${XS_STATS_DIR}"
[ -d "${XS_TABLE_DIR}" ] || error "no such anchor-stat table set: ${XS_TABLE_DIR}"
XS_MISSING=()
while read -r M; do
    [ -f "${XS_TABLE_DIR}/${M}.npz" ] || XS_MISSING+=("${M}")
done < <(awk 'NF && $0!~/^#/{print $2"\n"$3}' "${MANIFEST}" | sort -u)
[ "${#XS_MISSING[@]}" -eq 0 ] \
    || error "${#XS_MISSING[@]} anchor-stat table(s) missing from ${XS_STATS_DIR}: ${XS_MISSING[*]}"
info "targets: ${XS_STATS_DIR} (all months present)"

# ── Chunk into jobs ───────────────────────────────────────────────────────────
CHUNK_DIR="$(mktemp -d)"
trap 'rm -rf "${CHUNK_DIR}"' EXIT
uv run python - "${MANIFEST}" "${CHUNK_DIR}" "${MONTHS_PER_JOB}" "${TAG}" <<'PY'
import sys, collections, pathlib
manifest, outdir, per_job, tag = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

rows = []
for ln in open(manifest):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    ck, fit, ev = ln.split()
    rows.append((ck, fit, ev))

# Group by month-pair first: every checkpoint sharing a pair MUST land in the
# same job or its months get staged twice.
by_pair = collections.OrderedDict()
for ck, fit, ev in sorted(rows, key=lambda r: (r[1], r[2], r[0])):
    by_pair.setdefault((fit, ev), []).append(ck)

chunks, cur, cur_months = [], [], set()
for (fit, ev), cks in by_pair.items():
    want = cur_months | {fit, ev}
    if cur and len(want) > per_job:
        chunks.append(cur)
        cur, cur_months = [], set()
        want = {fit, ev}
    cur.extend((ck, fit, ev) for ck in cks)
    cur_months = want
if cur:
    chunks.append(cur)

for i, ch in enumerate(chunks):
    months = sorted({m for _, f, e in ch for m in (f, e)})
    p = outdir / f"{tag}-part{i:02d}.tsv"
    p.write_text("".join(f"{ck}\t{f}\t{e}\n" for ck, f, e in ch))
    print(f"{p.name}\t{len(ch)}\t{len(months)}\t{months[0]}..{months[-1]}")
PY

mapfile -t CHUNKS < <(ls "${CHUNK_DIR}"/*.tsv | sort)
info "split into ${#CHUNKS[@]} job(s) at <=${MONTHS_PER_JOB} months each"

if [ "${DRY_RUN}" = 1 ]; then
    for C in "${CHUNKS[@]}"; do
        echo "  $(basename "${C}"): $(wc -l < "${C}") ckpts, months $(awk '{print $2"\n"$3}' "${C}" | sort -u | tr '\n' ' ')"
    done
    info "dry run — nothing submitted"
    exit 0
fi

# ── Sync and submit ───────────────────────────────────────────────────────────
pythia_setup
pythia_sync

REMOTE_MANIFEST_DIR="${PYTHIA_HOME}/score-manifests"
ssh "${PYTHIA_HOST}" "mkdir -p ${REMOTE_MANIFEST_DIR}"
# The manifests are NOT part of the repo rsync (they are generated per run and
# would pollute the tree), so push them explicitly.
rsync -az "${CHUNK_DIR}/" "${PYTHIA_HOST}:${REMOTE_MANIFEST_DIR}/"

JOB_IDS=()
for C in "${CHUNKS[@]}"; do
    NAME="$(basename "${C}" .tsv)"
    OUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export \
            SCORE_MANIFEST='${REMOTE_MANIFEST_DIR}/$(basename "${C}")' \
            SCORE_TAG='${NAME}' SCORE_PROBE='${SCORE_PROBE}' \
            SCORE_AUC='${SCORE_AUC:-1}' \
            N_WORKERS='${N_WORKERS}' XS_STATS_DIR='${XS_STATS_DIR}' \
            MJ_FORCE_SEQ_LEN='${MJ_FORCE_SEQ_LEN:-}' \
            STAGE_PANEL_CACHE='${STAGE_PANEL_CACHE:-1}' && \
         sbatch --export=ALL --partition='${PARTITION}' \
            --job-name='${NAME}' --time='${TIME_LIMIT}' \
            ${SBATCH_EXTRA:-} \
            scripts/pythia/slurm_score_ckpts.sh") \
        || error "sbatch failed for ${NAME}: ${OUT}"
    JID="${OUT##* }"
    JOB_IDS+=("${JID}")
    info "${NAME}: job ${JID} ($(wc -l < "${C}") ckpts)"
done

info "submitted ${#JOB_IDS[@]} job(s): ${JOB_IDS[*]}"
info "results will land on bll01 at /data/lab/score_results/"
