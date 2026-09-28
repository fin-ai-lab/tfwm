#!/bin/bash
if [[ -z "${TMPDIR:-}" && -d /data/lab/tmp ]]; then
    export TMPDIR="/data/lab/tmp/${USER:-market-jepa}"
fi
mkdir -p "${TMPDIR:-/tmp}"
# run_probe_breadth.sh — submit the probe-fit-size sweep to pythia.
#
# Takes the TSV build_probe_breadth_manifest.py writes:
#
#     <ckpt_dir>\t<fit_month>\t<eval_month>
#
# six rows per (arm, eval month), and splits it into SLURM jobs.
#
# CHUNKED BY EVAL MONTH, NOT BY MONTH PAIR. run_score_ckpts.sh groups rows by
# their (fit, eval) pair, which is right when each row is an independent
# score -- but here the six fit months of one group are POOLED into a single
# ridge, and the reduce pools whatever it finds in the cache dir. A group split
# across two jobs would therefore reduce twice, each on a third of its pool,
# and report a plausible curve at the wrong n with nothing raising. So the
# atom here is the eval month: all four arms and all six of their fit months
# ride together. slurm_probe_breadth.sh re-checks the invariant.
#
# Packing is still by MONTH, because staging dominates: adjacent eval months
# share five of their six fit months, so a job holding 2008-08 and 2008-09
# stages seven months rather than fourteen.
#
# CKPT PATHS ARE RESOLVED ON BLL01 -- compute nodes rsync them from there -- so
# run this on bll01 and give ordinary local paths.
#
# Usage:
#   uv run python scripts/eval/build_probe_breadth_manifest.py \
#       /data/lab/probe_breadth/manifest.tsv
#   ./scripts/pythia/specific/run_probe_breadth.sh \
#       --manifest /data/lab/probe_breadth/manifest.tsv --tag pb1
#   ... --dry-run          # show the chunking, submit nothing
#
# Results land on bll01 at /data/lab/probe_breadth/results/<tag>-partNN-<eval>.json

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "${PYTHIA_DIR}/lib/setup.sh"
source "${PYTHIA_DIR}/lib/sync.sh"

PARTITION="${PARTITION:-standard_hopper}"
# Eval months per job. One job = 1 GPU / 8 CPUs (see slurm_probe_breadth.sh's
# billing note), and one eval month is ~24 fit-month forwards ~= 2 h. Two per
# job halves the job count and shares five staged months; the default is 1 so
# the whole panel runs at once inside the qos_bll budget.
EVAL_MONTHS_PER_JOB="${EVAL_MONTHS_PER_JOB:-1}"
XS_STATS_DIR="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
N_WORKERS="${N_WORKERS:-8}"
TIME_LIMIT="${TIME_LIMIT:-0-12:00:00}"
MANIFEST=""
TAG=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) error "unknown argument: $1" ;;
    esac
done
[ -n "${MANIFEST}" ] || error "--manifest is required"
[ -n "${TAG}" ] || error "--tag is required"
[ -f "${MANIFEST}" ] || error "manifest not found: ${MANIFEST}"

CHUNK_DIR="$(mktemp -d)"
trap 'rm -rf "${CHUNK_DIR}"' EXIT
uv run python - "${MANIFEST}" "${CHUNK_DIR}" "${EVAL_MONTHS_PER_JOB}" "${TAG}" <<'PY'
import collections, pathlib, sys
manifest, outdir, per_job, tag = (
    sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

rows = []
for ln in open(manifest):
    ln = ln.strip()
    if ln and not ln.startswith("#"):
        rows.append(tuple(ln.split()))

# EVAL MONTH IS THE ATOM. Every row sharing one goes to one job, so no
# (ckpt, eval) group can be split and reduced on a partial pool.
by_eval = collections.OrderedDict()
for ck, fit, ev in sorted(rows, key=lambda r: (r[2], r[0], r[1])):
    by_eval.setdefault(ev, []).append((ck, fit, ev))

groups = list(by_eval.items())
chunks = [groups[i:i + per_job] for i in range(0, len(groups), per_job)]
for i, ch in enumerate(chunks):
    out = [r for _, rs in ch for r in rs]
    months = sorted({m for _, f, e in out for m in (f, e)})
    evs = sorted(e for e, _ in ch)
    p = outdir / f"{tag}-part{i:02d}.tsv"
    p.write_text("".join(f"{c}\t{f}\t{e}\n" for c, f, e in out))
    print(f"{p.name}\t{len(out)} rows\teval {','.join(evs)}\t"
          f"{len(months)} months {months[0]}..{months[-1]}")
PY

mapfile -t CHUNKS < <(ls "${CHUNK_DIR}"/*.tsv | sort)
info "split into ${#CHUNKS[@]} job(s) at <=${EVAL_MONTHS_PER_JOB} eval month(s) each"

if [ "${DRY_RUN}" = 1 ]; then
    for C in "${CHUNKS[@]}"; do
        echo "  $(basename "${C}"): $(wc -l < "${C}") rows, months $(awk '{print $2"\n"$3}' "${C}" | sort -u | tr '\n' ' ')"
    done
    info "dry run — nothing submitted"
    exit 0
fi

pythia_setup
pythia_sync

REMOTE_MANIFEST_DIR="${PYTHIA_HOME}/probe-breadth-manifests"
ssh "${PYTHIA_HOST}" "mkdir -p ${REMOTE_MANIFEST_DIR}"
rsync -az "${CHUNK_DIR}/" "${PYTHIA_HOST}:${REMOTE_MANIFEST_DIR}/"

JOB_IDS=()
for C in "${CHUNKS[@]}"; do
    NAME="$(basename "${C}" .tsv)"
    OUT=$(ssh "${PYTHIA_HOST}" \
        "cd ${PYTHIA_REPO} && export \
            PB_MANIFEST='${REMOTE_MANIFEST_DIR}/$(basename "${C}")' \
            PB_TAG='${NAME}' \
            N_WORKERS='${N_WORKERS}' XS_STATS_DIR='${XS_STATS_DIR}' \
            PB_POOL='${PB_POOL:-}' \
            STAGE_PANEL_CACHE='${STAGE_PANEL_CACHE:-1}' && \
         sbatch --export=ALL --partition='${PARTITION}' \
            --job-name='${NAME}' --time='${TIME_LIMIT}' \
            ${SBATCH_EXTRA:-} \
            scripts/pythia/slurm_probe_breadth.sh") \
        || error "sbatch failed for ${NAME}: ${OUT}"
    JID="${OUT##* }"
    JOB_IDS+=("${JID}")
    info "${NAME}: job ${JID} ($(wc -l < "${C}") rows)"
done

info "submitted ${#JOB_IDS[@]} job(s): ${JOB_IDS[*]}"
info "results will land on bll01 at /data/lab/probe_breadth/results/"
