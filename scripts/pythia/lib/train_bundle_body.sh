#!/bin/bash
if [[ -z "${TMPDIR:-}" && -d /data/lab/tmp ]]; then
    export TMPDIR="/data/lab/tmp/${USER:-market-jepa}"
fi
mkdir -p "${TMPDIR:-/tmp}"
# train_bundle_body.sh — The shared body of slurm_train_bundle.sh, sourced by
# both scripts/pythia/slurm_train_bundle.sh and
# scripts/generic/slurm_train_bundle.sh after they set the cluster knobs.
#
# The two trees drifted for three months and the generic one silently lost the
# xs-anchor-stat staging, shard materialization, the stage-failure guard, the
# hydra date-override guard, and post-training IC scoring. One body, two thin
# SBATCH wrappers, so a fix lands on both clusters or neither.
#
# The wrapper must, BEFORE sourcing this file:
#   - set -euo pipefail
#   - source its lib/common.sh  (logging, prepare_shm_limits, require_gpu_*)
#   - set: REPO_DIR, DATA_DIR, LOCK_DIR, MARKER_DIR, BLL01_HOST, BLL01_DATA
#   - source scripts/pythia/lib/stage_data.sh
#   - set: SCRATCH_BASE       — per-job scratch root (/hpc_temp/$USER or $HOME)
#          MACHINE_NAME       — hydra machine= group (pythia, h100_cluster, …)
#          LOCAL_CKPT_DIR     — node-local checkpoint dir
#          BUNDLE_SCRIPT_REL  — wrapper path relative to repo root
#          RESUBMIT_ON_NO_GPU — 1 on multi-node clusters (requeue on a broken
#                               driver), 0 on single-node boxes (fail loudly)
#
# Two modes (MONTHS takes precedence over TRAIN_*/EVAL_* if both set):
#   Multi-month:  MONTHS="YYYY-MM YYYY-MM ..." — loop over each month, derive
#                 train range = whole month, eval range = next calendar month,
#                 stage that month's data, run every sweep sequentially.
#   Single-month: TRAIN_START/TRAIN_END/EVAL_START/EVAL_END — one stage, one
#                 sweep loop (legacy behavior).
#
# The loop is month × SWEEP FILE × sweep value: a job may carry SEVERAL sweep
# files, and they run back to back over ONE staging of each month. That is how
# month-major submission works (scripts/submit_month_major.sh) -- see
# scripts/pythia/lib/sweep_list.sh for why it has to be one job.
#
# Required env:
#   SWEEP_FILES_REL — one or more sweep file paths relative to repo root,
#                     space-separated. SWEEP_FILE_REL (a single path) is the
#                     older spelling and still works; run_sweep.sh and the
#                     scripts/pythia/specific/ launchers still use it.
#   COMMIT_HASH     — short git hash for wandb project suffix
#   Either MONTHS, or all of TRAIN_START / TRAIN_END / EVAL_START / EVAL_END.
#
# Optional env:
#   SWEEP_VALUES_OVERRIDE — comma-separated subset of SWEEP_VALUES (gap
#                           fills). With several sweep files it is
#                           '|'-separated and POSITIONAL: one entry per file,
#                           an empty entry running that sweep whole.
#   SWEEP_VALUE_ONLY      — exactly one value (run_sweep.sh per-value jobs);
#                           single-sweep jobs only
#   STAGE_ONLY=1          — stage data (incl. materialization), skip training
#   DAYSTORE=1            — train from the day-major store (dataset.backend=days);
#                           stages daystore months for the span and mosaic
#                           months only for the scorer
#   GRID_CACHE_GB         — GiB of dense-grid cache per dataloader worker
#   PROBE_NUM_THREADS     — live probe-eval threads (default 1)
#   POST_TRAIN_IC_EVAL    — default 1; the synchronized cross-section scoring
#   XS_STATS_DIR          — anchor-stat table set (defines the return target)
#
# Optional sweep-defined hooks (declare -F guarded, absent = no-op). Both are
# per (month, SWEEP FILE): a grouped job runs each sweep's own hooks, and one
# sweep's hooks never run for another (see load_sweep).
#   sweep_stage_extra     — after stage_all, before training (extra staging)
#   sweep_post_bundle YM CKPT...
#                         — after a sweep's value loop for one month, with
#                           every checkpoint dir it trained+synced. For
#                           work that needs ALL the month's models at once
#                           (e.g. the ssl_ic final32 latent evals). A nonzero
#                           return marks the month FAILED but does not stop
#                           later months; the checkpoints are already synced
#                           by the time it runs.

: "${COMMIT_HASH:?ERROR: COMMIT_HASH not set.}"
: "${SCRATCH_BASE:?ERROR: SCRATCH_BASE not set (wrapper bug).}"
: "${MACHINE_NAME:?ERROR: MACHINE_NAME not set (wrapper bug).}"
: "${LOCAL_CKPT_DIR:?ERROR: LOCAL_CKPT_DIR not set (wrapper bug).}"
: "${BUNDLE_SCRIPT_REL:?ERROR: BUNDLE_SCRIPT_REL not set (wrapper bug).}"
: "${RESUBMIT_ON_NO_GPU:?ERROR: RESUBMIT_ON_NO_GPU not set (wrapper bug).}"

if [ -n "${MONTHS:-}" ]; then
    BUNDLE_MONTHS=( ${MONTHS} )
    [ ${#BUNDLE_MONTHS[@]} -gt 0 ] || { echo "ERROR: MONTHS is empty" >&2; exit 1; }
    BUNDLE_MODE="multi"
else
    : "${TRAIN_START:?ERROR: TRAIN_START not set (and MONTHS unset).}"
    : "${TRAIN_END:?ERROR: TRAIN_END not set (and MONTHS unset).}"
    : "${EVAL_START:?ERROR: EVAL_START not set (and MONTHS unset).}"
    : "${EVAL_END:?ERROR: EVAL_END not set (and MONTHS unset).}"
    BUNDLE_MONTHS=( "${TRAIN_START:0:7}" )
    BUNDLE_MODE="single"
fi

# ── The sweep files this job carries ─────────────────────────────────────────
#
# A JOB is the unit that stages a month, so a job is the only unit that can
# GUARANTEE several sweeps share that staging: same job -> same node -> one
# mosaic rsync, one materialization, one panel build, one set of anchor
# tables. Submitting the sweeps as separate jobs with adjacent job IDs biases
# only the ORDER slurm starts them in; with many GPUs they start on different
# nodes and each stages the month again. See scripts/pythia/lib/sweep_list.sh.
if [ -n "${SWEEP_FILES_REL:-}" ]; then
    BUNDLE_SWEEPS=( ${SWEEP_FILES_REL} )
else
    : "${SWEEP_FILE_REL:?ERROR: neither SWEEP_FILES_REL nor SWEEP_FILE_REL set.}"
    BUNDLE_SWEEPS=( "${SWEEP_FILE_REL}" )
fi
N_SWEEPS=${#BUNDLE_SWEEPS[@]}

# Optional gap-fill override: a comma-separated subset of SWEEP_VALUES to run
# instead of the full sweep, for resubmitting only missing (month, value)
# pairs. With several sweep files it is '|'-separated and POSITIONAL — one
# entry per file, an empty entry running that sweep whole — which for a single
# file is exactly the old comma-separated form.
SWEEP_VALUE_OVERRIDES=()
if [ -n "${SWEEP_VALUES_OVERRIDE:-}" ]; then
    IFS='|' read -ra SWEEP_VALUE_OVERRIDES <<< "${SWEEP_VALUES_OVERRIDE}"
    # `read` DROPS TRAILING EMPTY FIELDS: "a2,a3|" splits to one entry, not
    # two, so "override sweep 1, run sweep 2 whole" would look like a count
    # mismatch and abort. Count the separators instead and pad.
    N_OVERRIDES=$(( $(tr -cd '|' <<< "${SWEEP_VALUES_OVERRIDE}" | wc -c) + 1 ))
    while [ ${#SWEEP_VALUE_OVERRIDES[@]} -lt "${N_OVERRIDES}" ]; do
        SWEEP_VALUE_OVERRIDES+=("")
    done
    if [ "${N_OVERRIDES}" -ne "${N_SWEEPS}" ]; then
        echo "ERROR: SWEEP_VALUES_OVERRIDE has ${N_OVERRIDES} '|'-separated entr(ies) but this job carries ${N_SWEEPS} sweep file(s)." >&2
        exit 1
    fi
fi
if [ -n "${SWEEP_VALUE_ONLY:-}" ] && [ "${N_SWEEPS}" -gt 1 ]; then
    echo "ERROR: SWEEP_VALUE_ONLY is a single-sweep knob; this job carries ${N_SWEEPS} sweep files." >&2
    exit 1
fi

# load_sweep <repo-relative path> <index>
#   Source one sweep file into this shell and apply that sweep's value
#   override, leaving SWEEP_NAME / SWEEP_VALUES / the hooks / WANDB_SUFFIX set
#   for it.
#
#   THE UNSETS ARE THE LOAD-BEARING PART. SWEEP_VALUES is an ARRAY, so a
#   4-value sweep loaded after a 9-value one would otherwise inherit five
#   stale values and train them; the three hooks are OPTIONAL, so sweep A's
#   sweep_post_bundle would otherwise run over sweep B's checkpoints. Every
#   name in the sweep-file contract is cleared before the source.
# THE RECIPE KNOBS A SWEEP TESTS WITH ${VAR+x}. A sweep emits an override only
# when the CALLER set one of these -- otherwise the value is schemas.py's and
# restating it would put a second copy of the recipe in the sweep file. The
# test is "is it set", so it is only correct the FIRST time the file is
# sourced: sourcing assigns BLR/BATCH_SIZE/NUM_EPOCHS/LOSS/SEED as ordinary
# shell variables, and load_sweep runs once per (month x sweep), so from the
# SECOND month on every one of them looked caller-set and was pinned.
#
# That is not cosmetic. It pinned the RETIRED recipe -- blr 1e-5, 100 epochs,
# and batch 256, which under the cells recipe means 256 CELLS of 16 stocks,
# 4,096 views a micro-batch -- and the full-history multihead wave died of CUDA
# OOM on every month after the first (223176-223225, 2026-09-13).
#
# So the caller's state is captured ONCE, here, and restored before each
# source. A knob genuinely exported by the submitter keeps working; one left
# behind by the previous month does not.
SWEEP_KNOBS=(BLR BATCH_SIZE NUM_EPOCHS LOSS SEED)
declare -A _KNOB_WAS_SET=() _KNOB_VALUE=()
for _k in "${SWEEP_KNOBS[@]}"; do
    if [ -n "${!_k+x}" ]; then
        _KNOB_WAS_SET["${_k}"]=1
        _KNOB_VALUE["${_k}"]="${!_k}"
    fi
done

restore_sweep_knobs() {
    local k
    for k in "${SWEEP_KNOBS[@]}"; do
        if [ -n "${_KNOB_WAS_SET[${k}]:-}" ]; then
            printf -v "${k}" '%s' "${_KNOB_VALUE[${k}]}"
        else
            unset "${k}"
        fi
    done
}

load_sweep() {
    local rel="$1" idx="$2"
    local path="${REPO_DIR}/${rel}"
    [ -f "${path}" ] || { echo "ERROR: sweep file not found: ${path}" >&2; return 1; }

    unset SWEEP_NAME SWEEP_VALUES WANDB_NO_SUFFIX
    unset -f sweep_train_args sweep_stage_extra sweep_post_bundle
    restore_sweep_knobs

    # shellcheck disable=SC1090
    source "${path}" || { echo "ERROR: failed to source ${rel}" >&2; return 1; }

    [ -n "${SWEEP_NAME:-}" ] || { echo "ERROR: ${rel}: SWEEP_NAME not set" >&2; return 1; }
    [ -n "${SWEEP_VALUES+x}" ] || { echo "ERROR: ${rel}: SWEEP_VALUES missing or empty" >&2; return 1; }
    declare -F sweep_train_args >/dev/null || { echo "ERROR: ${rel}: sweep_train_args() missing" >&2; return 1; }

    local ov="${SWEEP_VALUE_OVERRIDES[idx]:-}"
    if [ -n "${ov}" ]; then
        IFS=',' read -ra SWEEP_VALUES <<< "${ov}"
        echo "==> ${rel}: SWEEP_VALUES overridden: ${SWEEP_VALUES[*]}"
    fi
    # Restrict to a single value when launched by run_sweep.sh (one sbatch job
    # per sweep value instead of one bundle per month).
    if [ -n "${SWEEP_VALUE_ONLY:-}" ]; then
        SWEEP_VALUES=("${SWEEP_VALUE_ONLY}")
    fi

    # By default, append a -{commit}-{train_start}-{train_end} suffix to
    # wandb.project so each (commit, month) bundle lands in a fresh project.
    # Sweeps that want a stable, single project across months (e.g. full-data
    # supervised) opt out via WANDB_NO_SUFFIX=1 — a PER-SWEEP flag, which is
    # why the suffix is computed here and not once per job.
    # In multi-month mode TRAIN_START/TRAIN_END are not set yet (they are
    # derived per month inside the loop), so a suffixed multi-month bundle
    # would die on `set -u`. Suffix off the first and last month instead.
    if [ "${WANDB_NO_SUFFIX:-0}" = "1" ]; then
        WANDB_SUFFIX=""
    elif [ "${BUNDLE_MODE}" = "multi" ]; then
        WANDB_SUFFIX="-${COMMIT_HASH}-${BUNDLE_MONTHS[0]}-${BUNDLE_MONTHS[-1]}"
    else
        WANDB_SUFFIX="-${COMMIT_HASH}-${TRAIN_START}-${TRAIN_END}"
    fi
}

# Fail fast: every sweep file must load NOW — before uv sync, before staging,
# before hours of training. A typo in the third sweep of a grouped job must not
# surface six hours in, with two sweeps already trained behind it.
SWEEP_PLAN=()
for SI in "${!BUNDLE_SWEEPS[@]}"; do
    load_sweep "${BUNDLE_SWEEPS[SI]}" "${SI}" || exit 1
    SWEEP_PLAN+=("${SWEEP_NAME}(${#SWEEP_VALUES[@]})")
done

# ── Job info ──────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Bundled job   : ${N_SWEEPS} sweep(s), ${#BUNDLE_MONTHS[@]} month(s)"
for SI in "${!BUNDLE_SWEEPS[@]}"; do
    echo "    ${SWEEP_PLAN[SI]}  ${BUNDLE_SWEEPS[SI]}"
done
echo "  Job ID        : ${SLURM_JOB_ID}"
echo "  Node          : $(hostname)  GPU=${CUDA_VISIBLE_DEVICES:-?}"
echo "  Machine       : ${MACHINE_NAME}"
echo "  Mode          : ${BUNDLE_MODE} (${#BUNDLE_MONTHS[@]} month(s): ${BUNDLE_MONTHS[*]})"
echo "  Commit        : ${COMMIT_HASH}"
echo "════════════════════════════════════════════════════════════════"

# ── Install deps once ────────────────────────────────────────────────────────
# Even STAGE_ONLY needs the venv: stage_all's shard materialization runs the
# venv's zstandard (compute nodes have no zstd CLI).

echo ""
echo "==> Installing dependencies with uv ..."
export PATH="${HOME}/.local/bin:${PATH}"
export WANDB_DIR="${SCRATCH_BASE}/wandb"
mkdir -p "${WANDB_DIR}"
export MARKET_JEPA_TEMP_PROBE_EVAL_DIR="${SCRATCH_BASE}/market-jepa-temp-probe-eval"
mkdir -p "${MARKET_JEPA_TEMP_PROBE_EVAL_DIR}"
export UV_LINK_MODE=copy
cd "${REPO_DIR}"

# ── uv sync, with retries ───────────────────────────────────────────────────
# The project has a GIT dependency, and on pythia UV_CACHE_DIR is per-job (see
# the SHARED_VENV comment below: /hpc_temp is evicted by modtime, so a shared
# cache is not safe there). Every job therefore clones from github.com at
# startup, and a sweep whose jobs launch together times out often enough to
# lose real work -- a 33-job wave on 2026-08-29 lost 4 of the first 5 starters
# to "Failed to connect to github.com port 443: Connection timed out", with
# nothing written and a queue slot burned on a two-minute failure.
#
# The retry is the whole fix: the error is transient and uncorrelated, so a
# second attempt a minute later almost always lands. Behaviour on success is
# unchanged.
_uv_sync_retry() {
    local n=0
    until uv sync --frozen --quiet; do
        n=$((n + 1))
        if [ "${n}" -ge "${UV_SYNC_RETRIES:-4}" ]; then
            echo "ERROR: uv sync failed ${n}x (network?); giving up." >&2
            return 1
        fi
        echo "==> uv sync failed (attempt ${n}); retrying in $((n * 30))s ..." >&2
        sleep $((n * 30))
    done
}

if [ "${SHARED_VENV:-0}" = "1" ]; then
    # ── One venv per DEPENDENCY SET, shared by every job on the node ────────
    #
    # The per-job alternative below rebuilds a 7.7G venv AND a fresh
    # UV_CACHE_DIR for every job, so each one re-downloads torch from PyPI and
    # then deletes it on exit: measured 122G of venvs+caches live on the
    # 8-job A100 box, for one identical dependency set.
    #
    # CONTENT-ADDRESSED ON uv.lock, which is the property that makes sharing
    # safe. A running job's venv must never be mutated underneath it, so the
    # venv path carries the lock's hash: change a dependency and NEW jobs
    # build a new directory while in-flight jobs keep the one they started
    # with. Same lock -> same path -> built once, reused.
    #
    # flock + .ready marker, the same idiom stage_data.sh uses: eight jobs
    # starting together must not run `uv sync` into one directory at once.
    # The marker is written INSIDE the lock and after a successful sync, so a
    # sync that dies half way leaves no marker and the next job retries
    # rather than importing a half-populated venv.
    #
    # Single-node cloud boxes only (set by scripts/generic/slurm_train_bundle.sh).
    # Pythia keeps the per-job venv: /hpc_temp is evicted by MODTIME, and a
    # long-lived shared venv there can be reaped mid-run, while a per-job one
    # is touched constantly and survives.
    LOCK_HASH=$(sha256sum "${REPO_DIR}/uv.lock" 2>/dev/null | cut -c1-12)
    : "${LOCK_HASH:=nolock}"
    VENV_DIR="${SCRATCH_BASE}/market-jepa-venv-${LOCK_HASH}"
    export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
    export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
    mkdir -p "${LOCK_DIR}" "${UV_CACHE_DIR}"
    (
        flock -x 9
        if [ -f "${VENV_DIR}/.ready" ]; then
            echo "==> Reusing shared venv ${VENV_DIR}"
        else
            echo "==> Building shared venv ${VENV_DIR} (uv.lock ${LOCK_HASH}) ..."
            rm -rf "${VENV_DIR}"
            _uv_sync_retry && touch "${VENV_DIR}/.ready"
        fi
    ) 9> "${LOCK_DIR}/venv-${LOCK_HASH}.lock"
    [ -f "${VENV_DIR}/.ready" ] || { echo "ERROR: shared venv build failed" >&2; exit 1; }
else
    VENV_DIR="${SCRATCH_BASE}/market-jepa-venv-${SLURM_JOB_ID}"
    export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv-${SLURM_JOB_ID}"
    export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
    trap 'rm -rf "${VENV_DIR}" "${UV_CACHE_DIR}"' EXIT
    _uv_sync_retry
fi

echo "==> Python: $(uv run python --version)"
# REPORT WHY, not just that. A bare "CUDA: False" says nothing about whether
# the driver is broken, the allocation is missing, or the venv is incomplete,
# and the answer decides whether requeueing to another node can possibly help.
echo "==> torch:  $(uv run python -c '
import os, torch
ok = torch.cuda.is_available()
msg = f"{torch.__version__} CUDA: {ok}"
if not ok:
    msg += f" | compiled={torch.cuda._is_compiled()}"
    msg += f" CUDA_VISIBLE_DEVICES={os.environ.get(chr(67)+chr(85)+chr(68)+chr(65)+chr(95)+chr(86)+chr(73)+chr(83)+chr(73)+chr(66)+chr(76)+chr(69)+chr(95)+chr(68)+chr(69)+chr(86)+chr(73)+chr(67)+chr(69)+chr(83))!r}"
    try:
        torch.cuda.init()
    except Exception as e:
        msg += f" | {type(e).__name__}: {str(e)[:200]}"
print(msg)
')"

if [ "${STAGE_ONLY:-0}" != "1" ]; then
    if [ "${RESUBMIT_ON_NO_GPU}" = "1" ]; then
        require_gpu_or_resubmit "${REPO_DIR}/${BUNDLE_SCRIPT_REL}"
    else
        require_gpu_or_fail
    fi
fi

prepare_shm_limits

# ── The training span, from schemas and nowhere else ─────────────────────────
TRAIN_SPAN_MONTHS="${TRAIN_SPAN_MONTHS:-$(uv run python -c 'from market_jepa.schemas import DatasetConfig; print(DatasetConfig.train_span_months)' 2>/dev/null)}"
[[ "${TRAIN_SPAN_MONTHS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: could not resolve DatasetConfig.train_span_months (got '${TRAIN_SPAN_MONTHS}')" >&2
    exit 1
}
echo "==> training span: ${TRAIN_SPAN_MONTHS} month(s) ending at each bundle month"

# POST_TRAIN_PROBE=0 scores the trained HEAD only. The probe-fit month is then
# never embedded, staged or required -- it is 36 anchors/day against the eval
# month's 8, so it is the bulk of scoring. Right for a sweep that reports the
# head; wrong wherever a supervised arm has to stay comparable to an SSL one.
HEAD_ONLY_FLAG=""
if [ "${POST_TRAIN_PROBE:-1}" = "0" ]; then
    HEAD_ONLY_FLAG="--head-only"
    echo "==> scoring: HEAD ONLY (no ridge probe, no probe-fit panel)"
fi

# ── Loop over months × sweep files × sweep values ────────────────────────────

# Don't let a single run failure kill the whole bundle.
set +e
FAILED=()
SUCCESS=()

for YM in "${BUNDLE_MONTHS[@]}"; do
    if [ "${BUNDLE_MODE}" = "multi" ]; then
        # A "month" IS THE SPAN ENDING AT THAT MONTH. The recipe is
        # DatasetConfig.train_span_months of data before the eval month
        # (settled 2026-09-13), so a bundle month names the span's LAST month
        # and training starts SPAN-1 months earlier. It used to be the single
        # calendar month, which is a different and measurably worse recipe --
        # single-month return sits at the random-init floor on 2020-05 and
        # 2022-02, six months lifts both.
        #
        # Sweeps name their runs from TRAIN_END for this reason: TRAIN_START
        # now moves with the span while TRAIN_END does not.
        TRAIN_END=$(date -d "${YM}-01 +1 month -1 day" +%Y-%m-%d)
        TRAIN_START=$(date -d "${YM}-01 -$(( TRAIN_SPAN_MONTHS - 1 )) months" +%Y-%m-01)
        NEXT_YM_START=$(date -d "${YM}-01 +1 month" +%Y-%m-%d)
        EVAL_START="${NEXT_YM_START}"
        EVAL_END=$(date -d "${NEXT_YM_START} +1 month -1 day" +%Y-%m-%d)
        export TRAIN_START TRAIN_END EVAL_START EVAL_END
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Month ${YM}: train ${TRAIN_START}..${TRAIN_END}, eval ${EVAL_START}..${EVAL_END}"
    echo "════════════════════════════════════════════════════════════════"

    # Make room BEFORE staging, and only if the node is short: a full
    # /hpc_temp fails every job SLURM hands the node (see
    # reclaim_scratch_if_low in stage_data.sh). Never fails the job.
    reclaim_scratch_if_low || true

    # Under the loop's `set +e`, stage_all's abort-before-training return
    # value must be checked explicitly or a partial staging trains anyway.
    #
    # DAYSTORE=1: the training span comes from the day-major store and only
    # the scorer's months from the mosaic (see stage_all_daystore); train.py
    # is then pointed at it with dataset.backend=days.
    if [ "${DAYSTORE:-0}" = "1" ]; then
        if ! stage_all_daystore "${TRAIN_START}" "${TRAIN_END}" "${EVAL_END}"; then
            echo "==> stage_all_daystore FAILED for ${YM}; skipping month."
            FAILED+=("${YM}/staging")
            continue
        fi
    elif ! stage_all "${TRAIN_START}" "${EVAL_END}"; then
        echo "==> stage_all FAILED for ${YM}; skipping month."
        FAILED+=("${YM}/staging")
        continue
    fi

    # ── The eval panel is an INPUT, not an optimization ────────────────────
    #
    # Fail the JOB, now, rather than train for hours and then re-decode the
    # month-panel the cache was supposed to hold. See require_panel_cache.
    if ! require_panel_cache "${TRAIN_END:0:7}" "${EVAL_START:0:7}"; then
        echo "==> ABORTING: the panel cache for ${TRAIN_END:0:7} (probe-fit) and/or" >&2
        echo "    ${EVAL_START:0:7} (eval) is not built. Build it on the mosaic host:" >&2
        echo "      MJ_PANEL_CACHE=<root> uv run scripts/eval/build_panel_cache.py \\" >&2
        echo "          --months ${TRAIN_END:0:7} --roles probe eval --jobs 40" >&2
        echo "    then relaunch. PANEL_CACHE_REQUIRED=0 overrides (non-default panel" >&2
        echo "    geometry only); full_data_multihead is exempt by design." >&2
        exit 1
    fi

    # ── Each sweep file in turn, over the month that is now staged ──────────
    #
    # Same job, same node, ONE staging — the reason a job carries a list of
    # sweeps rather than the queue carrying a job per (month, sweep). Each
    # iteration re-sources its own sweep file, so SWEEP_NAME, SWEEP_VALUES,
    # WANDB_SUFFIX and the hooks below all belong to THIS sweep.
    for SI in "${!BUNDLE_SWEEPS[@]}"; do
        if ! load_sweep "${BUNDLE_SWEEPS[SI]}" "${SI}"; then
            echo "==> could not load ${BUNDLE_SWEEPS[SI]} for ${YM}; skipping it."
            FAILED+=("${YM}/${BUNDLE_SWEEPS[SI]}/load")
            continue
        fi

        if [ "${N_SWEEPS}" -gt 1 ]; then
            echo ""
            echo "  ──── sweep $((SI + 1))/${N_SWEEPS}: ${SWEEP_NAME} (${#SWEEP_VALUES[@]} values) ────"
        fi

        # Optional sweep-defined staging hook (e.g. ssl_finetune sweeps pull the
        # month's base SSL checkpoint from bll01 — see sweeps/ssl_finetune/ssl_base_lib.sh).
        if declare -F sweep_stage_extra >/dev/null; then
            if ! sweep_stage_extra; then
                echo "==> sweep_stage_extra FAILED for ${YM}/${SWEEP_NAME}; skipping sweep."
                FAILED+=("${YM}/${SWEEP_NAME}/stage_extra")
                continue
            fi
        fi

        if [ "${STAGE_ONLY:-0}" = "1" ]; then
            echo "==> STAGE_ONLY=1 — data staged for ${YM}/${SWEEP_NAME}, skipping training."
            SUCCESS+=("${YM}/${SWEEP_NAME}/stage")
            continue
        fi

        BUNDLE_CKPT_PATHS=()
        for VAL in "${SWEEP_VALUES[@]}"; do
            echo ""
            echo "────────────────────────────────────────────────────────────────"
            echo "  Run: ${VAL}   (${TRAIN_START} .. ${TRAIN_END})"
            echo "────────────────────────────────────────────────────────────────"

            TRAIN_ARGS=$(sweep_train_args "${VAL}")
            # A sweep that varies the TRAINING SPAN across its own values (several
            # spans sharing one eval month, in one job) emits its own
            # train_date_start. Appending ours after it would silently win under
            # hydra last-wins, so only supply the dates the sweep left alone.
            # TRAIN_START then means "the earliest date this job needs staged",
            # which is what stage_all wants anyway.
            case "${TRAIN_ARGS}" in
                *dataset.train_date_start=*) ;;
                *) TRAIN_ARGS="${TRAIN_ARGS} dataset.train_date_start=${TRAIN_START} dataset.train_date_end=${TRAIN_END} dataset.eval_train_date_start=${TRAIN_START} dataset.eval_train_date_end=${TRAIN_END}" ;;
            esac
            case "${TRAIN_ARGS}" in
                *dataset.eval_date_start=*) ;;
                *) TRAIN_ARGS="${TRAIN_ARGS} dataset.eval_date_start=${EVAL_START} dataset.eval_date_end=${EVAL_END}" ;;
            esac
            TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed "s|wandb\.project=\([^ ]*\)|wandb.project=\1${WANDB_SUFFIX}|")

            # Dense-grid cache, GiB per dataloader worker (0 = off). A sweep file
            # that sets it per-arm wins; otherwise the job-level GRID_CACHE_GB does.
            case "${TRAIN_ARGS}" in
                *dataset.grid_cache_gb=*) : ;;
                *) TRAIN_ARGS="${TRAIN_ARGS} dataset.grid_cache_gb=${GRID_CACHE_GB:-0}" ;;
            esac

            TRAIN_LOG=$(mktemp)

            if [ "${DAYSTORE:-0}" = "1" ]; then
                TRAIN_ARGS="${TRAIN_ARGS} dataset.backend=days machine.daystore_dir=${DATA_DIR}/1Hz_daystore"
            fi

            # shellcheck disable=SC2086
            uv run train.py \
                machine="${MACHINE_NAME}" \
                machine.mosaic_dir="${DATA_DIR}/1Hz_mosaic_mnth" \
                machine.risk_factor_dir="${DATA_DIR}/1Hz_risk_factors" \
                machine.metadata_path="${DATA_DIR}/metadata.parquet" \
                dataset.xs_anchor_stats_dir="${DATA_DIR}/${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}" \
                checkpoint.chkpt_dir="${LOCAL_CKPT_DIR}" \
                checkpoint.remote_dir="/data/lab/market-jepa-checkpoints" \
                probe_eval.targets.horizons=[300,600,900,1800,3600,7200] \
                probe_eval.targets.types=[return,volatility_change,spread_change] \
                probe_eval.num_threads="${PROBE_NUM_THREADS:-1}" \
                ${TRAIN_ARGS} 2>&1 | tee "${TRAIN_LOG}"
            RC=${PIPESTATUS[0]}

            if [ "${RC}" -ne 0 ]; then
                echo "==> Run ${YM}/${SWEEP_NAME}/${VAL} FAILED (exit ${RC}). Continuing with next value."
                FAILED+=("${YM}/${SWEEP_NAME}/${VAL}")
                rm -f "${TRAIN_LOG}"
                continue
            fi
            SUCCESS+=("${YM}/${SWEEP_NAME}/${VAL}")

            # Sync checkpoint back
            # || true — see slurm_train.sh: grep exits 1 when train.py skipped an
            # already-finished run, and pipefail + set -e would kill the job.
            CKPT_PATH=$( { grep -oP '(?<=Saved checkpoint to ).*' "${TRAIN_LOG}" || true; } | tail -1)
            rm -f "${TRAIN_LOG}"

            if [ -n "${CKPT_PATH}" ] && [ -d "${CKPT_PATH}" ]; then
                RUN_ID=$(basename "${CKPT_PATH}")
                WANDB_PROJECT=$(basename "$(dirname "${CKPT_PATH}")")

                # Synchronized cross-section IC, on this node, before the staged
                # mosaic and the GPU go away. With training.live_eval=false these
                # are the only metrics the run logs -- a full month of cross
                # sections with a per-cell SE, instead of 4096 pooled rows.
                if [ "${POST_TRAIN_IC_EVAL:-1}" = "1" ]; then
                    # Pre-built panels, when this node has them staged. The scorer
                    # falls through to a live build for any (month, geometry) it
                    # cannot match, so this is safe to set unconditionally --
                    # see scripts/eval/panel_cache.py for the key.
                    if [ -d "${DATA_DIR}/panel_cache" ]; then
                        export MJ_PANEL_CACHE="${DATA_DIR}/panel_cache"
                    fi
                    echo "==> Post-training IC eval for ${YM}/${SWEEP_NAME}/${VAL} ..."
                    # The sharded decode (post_train_ic_eval --decode-procs)
                    # decompresses several MDS shards into $TMPDIR at once;
                    # node /tmp can be too small for that (pythia), so default
                    # it onto the job's scratch.
                    EVAL_TMP="${TMPDIR:-${SCRATCH_BASE}/pt_eval_tmp}"
                    mkdir -p "${EVAL_TMP}"
                    TMPDIR="${EVAL_TMP}" uv run scripts/generic/post_train_ic_eval.py \
                        --ckpt-dir "${CKPT_PATH}" \
                        ${HEAD_ONLY_FLAG} \
                        --train-month "${TRAIN_END:0:7}" \
                        --eval-month "${EVAL_START:0:7}" \
                        --wandb-run-id "${RUN_ID}" \
                        --wandb-project "${WANDB_PROJECT}" \
                        --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
                        --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}" \
                        --risk-factor-dir "${DATA_DIR}/1Hz_risk_factors" \
                        || { echo "==> post-training IC eval FAILED for ${YM}/${SWEEP_NAME}/${VAL}"; \
                             SCORE_FAILED=1; }

                    # INTERMEDIATE CHECKPOINTS, scored HERE and for the same
                    # reason the final one is: the staged mosaic, the panel
                    # cache and the GPU are all on this node and all go away
                    # when the job ends. checkpoint.save_fractions writes them
                    # to <run>/<step>/; no other sweep sets it, so this loop
                    # finds nothing and costs nothing for them.
                    #
                    # --no-wandb: five checkpoints share ONE wandb run, and
                    # logging each under that run id would leave only the last
                    # one's xs_ic/* visible, silently. The scientific record is
                    # the xs_ic.json each writes into its own directory, which
                    # is what collect_ssl_finetune_breadth.py reads.
                    for STEP_DIR in "${CKPT_PATH}"/[0-9]*; do
                        [ -d "${STEP_DIR}" ] || continue
                        echo "==> IC eval for intermediate step $(basename "${STEP_DIR}") ..."
                        TMPDIR="${EVAL_TMP}" uv run scripts/generic/post_train_ic_eval.py \
                            --ckpt-dir "${STEP_DIR}" \
                            ${HEAD_ONLY_FLAG} \
                            --train-month "${TRAIN_END:0:7}" \
                            --eval-month "${EVAL_START:0:7}" \
                            --no-wandb \
                            --mosaic-dir "${DATA_DIR}/1Hz_mosaic_mnth" \
                            --xs-anchor-stats-dir "${DATA_DIR}/${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}" \
                            --risk-factor-dir "${DATA_DIR}/1Hz_risk_factors" \
                            || echo "==> IC eval FAILED for ${STEP_DIR} (continuing)"
                    done
                fi

                # SKIP_CKPT_SYNC=1 keeps the weights on the node and never
                # copies them to bll01. For a sweep whose checkpoints we do not
                # intend to keep, the sync is pure cost: the scientific result
                # -- xs_ic -- is already in W&B by this line, and the checkpoint
                # still lands in /hpc_temp, which is not purged between jobs.
                #
                # It is also a circuit breaker. On 2026-09-08 a wedged ext4
                # journal on bll01 made /data unwritable, and the ssh below --
                # which has no timeout and cannot be interrupted once the kernel
                # puts it in uninterruptible sleep -- held 32 jobs on their GPUs
                # until the 12-hour reservations expired. Every one of them had
                # finished training and scoring. Setting this to 1 lets a sweep
                # run to completion while bll01's storage is down.
                if [ "${SKIP_CKPT_SYNC:-0}" = "1" ]; then
                    echo "==> SKIP_CKPT_SYNC=1: leaving ${CKPT_PATH} on node scratch, not syncing to bll01."
                else
                    CKPT_DST="${BLL01_HOST}:/data/lab/market-jepa-checkpoints/${WANDB_PROJECT}/${RUN_ID}/"
                    echo "==> Syncing checkpoint ${CKPT_PATH} to ${CKPT_DST} ..."
                    ssh "${BLL01_HOST}" "mkdir -p /data/lab/market-jepa-checkpoints/${WANDB_PROJECT}"
                    # THE INTERMEDIATE WEIGHTS DO NOT TRAVEL. They were scored
                    # above, so what they are worth -- xs_ic.json and
                    # train_meta.json -- is already in <run>/<step>/ and does
                    # travel. The backbone is 86 MB, and a label-budget sweep
                    # at five fractions would put ~240 GB of snapshots onto a
                    # /data that is already 94% full. The "*/" prefix means a
                    # SUBDIRECTORY, so the run's own final backbone.pt at the
                    # top level is unaffected.
                    #
                    # SYNC_STEP_WEIGHTS=1 keeps them, for a sweep that really
                    # does want to reload a mid-training model.
                    STEP_EXCLUDE=(--exclude "*/*.pt")
                    [ "${SYNC_STEP_WEIGHTS:-0}" = "1" ] && STEP_EXCLUDE=()
                    if rsync -az --info=progress2 "${STEP_EXCLUDE[@]}" "${CKPT_PATH}/" "${CKPT_DST}"; then
                        echo "==> Checkpoint synced."
                    else
                        echo "==> WARNING: checkpoint rsync to ${CKPT_DST} FAILED."
                        FAILED+=("${YM}/${SWEEP_NAME}/${VAL}/ckpt-sync")
                    fi
                fi
                BUNDLE_CKPT_PATHS+=("${CKPT_PATH}")
            fi
        done

        # Month-level hook over everything THIS sweep trained this month (header).
        if declare -F sweep_post_bundle >/dev/null && [ ${#BUNDLE_CKPT_PATHS[@]} -gt 0 ]; then
            echo ""
            echo "==> sweep_post_bundle for ${YM}/${SWEEP_NAME} (${#BUNDLE_CKPT_PATHS[@]} checkpoints) ..."
            if sweep_post_bundle "${YM}" "${BUNDLE_CKPT_PATHS[@]}"; then
                SUCCESS+=("${YM}/${SWEEP_NAME}/post_bundle")
            else
                echo "==> sweep_post_bundle FAILED for ${YM}/${SWEEP_NAME}."
                FAILED+=("${YM}/${SWEEP_NAME}/post_bundle")
            fi
        fi
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Bundle done: $(date)"
echo "  Sweeps    : ${SWEEP_PLAN[*]}"
echo "  Succeeded (${#SUCCESS[@]}): ${SUCCESS[*]:-none}"
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

[ ${#FAILED[@]} -eq 0 ]
