#!/bin/bash
# stage_data.sh — Per-month data staging helpers for SLURM jobs. Shared by
# every launcher -- a second copy once drifted for three months (no xs-anchor
# staging, no materialization, no rsync-failure guard) before it was deleted.
#
# Sourced by the slurm_train*.sh entrypoints (which run on compute nodes). Expects these to be set by the caller:
#   DATA_HOST, DATA_HOST_DATA, DATA_HOST_METADATA
#   DATA_DIR, LOCK_DIR, MARKER_DIR
#
# Locking: each month has its own flock, and shared resources (risk factors,
# metadata) each have one global flock. A ".done" marker lets concurrent jobs
# on the same node skip resources that are already staged.
#
# MARKER_VERSION invalidates every marker written by an earlier revision of
# this file. Bumped to v3 because markers written before the refresh_mtimes
# fix may describe months whose shards node-local scratch already evicted (or is about
# to). Bumping re-verifies once (rsync --size-only is a near-no-op for months
# that are genuinely complete) while still letting concurrent jobs on the
# node share the result.
# v4: staged months are MATERIALIZED — any .zstd-only shard is decompressed
# under the month lock, index.json is rewritten to declare no compression, and
# the .zstd copies are removed. Without this, several concurrent jobs iterating
# the same months (the 5yr event-conditioning sweep) race inside mosaicml
# streaming's lazy decompression (shard.X.mds.tmp -> shard.X.mds renames in the
# SHARED staged dir) and die with FileNotFoundError — streaming's shard prep is
# only safe across ranks of one job, not across independent jobs.
MARKER_VERSION="v7"
# v6: anchor tables rebuilt with per-cell empirical QUANTILES (the rank
# target). They are a strict superset — mu/sigma/count are unchanged — but
# each month grew 210 KB -> ~4.9 MB, so every staged copy must be refreshed.
# v7: the mosaic rsync ships .mds.zstd only, and the anchor tables were
# rebuilt for the uniform target (they carry per-cell order statistics now, so
# they are larger again). A v6 month on a node holds raw shards that came over
# the wire rather than out of a local decompression — harmless, but the marker
# has to move for the table refresh regardless.

# node-local scratch evicts by MODTIME, and rsync -a preserves source mtimes — so a
# freshly staged month whose shards were built long ago on the data host looks
# months old and can be evicted the very night it was staged (this killed
# 5 of the 10 variance-decomp months mid-job on 2026-07-24 at ~02:00).
# Touch everything after staging AND on every already-staged skip, so data
# in active use always looks fresh to the cleaner.
refresh_mtimes() {
    local path="$1"
    [ -e "${path}" ] && find "${path}" -exec touch {} + 2>/dev/null
    return 0
}

# Enumerate YYYY-MM strings covered by [TRAIN_START, TRAIN_END] inclusive.
months_in_range() {
    local start="$1" end="$2"
    local y m end_y end_m cur stop cy cm
    y=$(date -d "${start}" +%Y)
    m=$(date -d "${start}" +%m)
    end_y=$(date -d "${end}" +%Y)
    end_m=$(date -d "${end}" +%m)
    cur=$((10#${y} * 12 + 10#${m}))
    stop=$((10#${end_y} * 12 + 10#${end_m}))
    while [ "${cur}" -le "${stop}" ]; do
        cy=$(( (cur - 1) / 12 ))
        cm=$(( (cur - 1) % 12 + 1 ))
        printf "%04d-%02d\n" "${cy}" "${cm}"
        cur=$((cur + 1))
    done
}

# Decompress any shard present only as .zstd, size-verify every raw shard
# against index.json, rewrite the index to declare no compression, and delete
# the .zstd copies. Runs under the caller's month flock, AFTER uv sync (the
# decompressor is the venv's zstandard — compute nodes have no zstd CLI).
# After this, streaming reads raw .mds only — nothing left to race on
# between jobs. See lib/materialize_mds.py.
materialize_month() {
    local target="$1" ym="$2"
    rm -f "${target}"/*.mds.tmp* 2>/dev/null   # orphans from crashed jobs
    (cd "${REPO_DIR}" && uv run python scripts/generic/lib/materialize_mds.py \
        "${target}") || return 1
}

stage_mosaic_month() {
    local ym="$1"
    local year="${ym%%-*}"
    local mm="${ym##*-}"
    local lock="${LOCK_DIR}/mosaic-${ym}.lock"
    local marker="${MARKER_DIR}/mosaic-${ym}.${MARKER_VERSION}.done"
    local target="${DATA_DIR}/1Hz_mosaic_mnth/${year}/${mm}"
    (
        flock -x 9
        if [ -f "${marker}" ] && [ -f "${target}/index.json" ]; then
            refresh_mtimes "${target}"
            touch "${marker}"
            echo "==> mosaic ${ym}: already staged, skipping (mtimes refreshed)."
            exit 0
        fi
        if [ -f "${marker}" ]; then
            echo "==> mosaic ${ym}: stale marker (index.json missing), re-staging."
            rm -f "${marker}"
            rm -rf "${target}"
        fi
        mkdir -p "${target}"
        echo "==> mosaic ${ym}: rsyncing from ${DATA_HOST} ..."
        # Only mark done when rsync actually succeeded. Marking regardless lets
        # a refused SSH connection masquerade as a staged month, and the job
        # then dies much later on a missing index.json.
        # --size-only: shards are immutable once built, and local mtimes are
        # deliberately touched (see refresh_mtimes) — mtime comparison would
        # re-copy every file on each marker-version re-verify.
        # SHIP THE COMPRESSED SHARDS ONLY, and nothing else changes: the
        # exclude drops shard.N.mds and keeps shard.N.mds.zstd (the pattern
        # ends the name, so the .zstd twin does not match), along with
        # index.json and ticker_date_map.json, which cross_stock needs.
        #
        # the data host holds most shards twice. Unfiltered this moved 9.15 GB per
        # month where 1.88 GB suffices -- a 4.9x tax on the step that gates
        # every job on a cold node -- and the raw copy was transferred only
        # for materialize_month to overwrite it seconds later by decompressing
        # the file sitting next to it.
        #
        # SAFE BECAUSE EVERY RAW SHARD HAS A .zstd TWIN: audited by name over
        # all 208 month dirs, zero orphans. 49 months hold FEWER raw than
        # compressed (a partial materialization someone left behind), never
        # the reverse. If that ever inverts, this filter silently ships an
        # incomplete month, so re-run that audit before trusting it again.
        #
        # --compress-level=1 stays for the json sidecars; the shards are
        # already zstd and rsync's own pass does nothing for them.
        if ! rsync -az --size-only --info=progress2 --compress-level=1 \
            --exclude='*.mds' \
            "${DATA_HOST}:${DATA_HOST_DATA}/1Hz_mosaic_mnth/${year}/${mm}/" \
            "${target}/"; then
            echo "==> mosaic ${ym}: rsync FAILED" >&2
            exit 1
        fi
        if ! materialize_month "${target}" "${ym}"; then
            echo "==> mosaic ${ym}: materialization FAILED" >&2
            exit 1
        fi
        refresh_mtimes "${target}"
        touch "${marker}"
        echo "==> mosaic ${ym}: staged (materialized)."
    ) 9>"${lock}"
}

stage_daystore_month() {
    # ONE MONTH OF THE DAY-MAJOR STORE (stable_finance.dataset.daystore): a
    # record per trading day holding the whole cross-section and its targets.
    # The training loader for dataset.backend=days reads this and nothing
    # else -- no mosaic, no anchor tables -- so a training span stages only
    # these months; the mosaic is still staged for the scorer's months.
    #
    # THE WIRE CARRIES features.npy.zst ONLY (~9x smaller than the raw
    # array); the raw features.npy is excluded on the way over and rebuilt
    # here under the month lock, once, and the .zst is then deleted -- the
    # same materialize-once idiom as the mosaic. Every other file in a day
    # (meta.json, first_row, the target arrays, count) is small and ships as
    # is. The size check inside decompress_day is what makes a partially
    # copied or half-decompressed day re-do itself rather than train on a
    # truncated array.
    local ym="$1"
    local year="${ym%%-*}"
    local mm="${ym##*-}"
    local lock="${LOCK_DIR}/daystore-${ym}.lock"
    local marker="${MARKER_DIR}/daystore-${ym}.${MARKER_VERSION}.done"
    local target="${DATA_DIR}/1Hz_daystore/${year}/${mm}"
    (
        flock -x 9
        if [ -f "${marker}" ] && [ -f "${target}/index.json" ]; then
            refresh_mtimes "${target}"
            touch "${marker}"
            echo "==> daystore ${ym}: already staged, skipping (mtimes refreshed)."
            exit 0
        fi
        if [ -f "${marker}" ]; then
            echo "==> daystore ${ym}: stale marker (index.json missing), re-staging."
            rm -f "${marker}"
        fi
        mkdir -p "${target}"
        echo "==> daystore ${ym}: rsyncing from ${DATA_HOST} ..."
        if ! rsync -az --size-only --info=progress2 --compress-level=1 \
            --exclude='features.npy' --exclude='*.partial' \
            "${DATA_HOST}:${DATA_HOST_DATA}/1Hz_daystore/${year}/${mm}/" \
            "${target}/"; then
            echo "==> daystore ${ym}: rsync FAILED" >&2
            exit 1
        fi
        [ -f "${target}/index.json" ] || {
            echo "==> daystore ${ym}: no index.json on the data host -- month not converted yet" >&2
            exit 1
        }
        if ! (cd "${REPO_DIR}" && uv run python - "${target}" <<'PY'
import json, sys
from pathlib import Path
from stable_finance.dataset.daystore import decompress_day
month = Path(sys.argv[1])
days = [d["date"] for d in json.loads((month / "index.json").read_text())["days"]]
for date in days:
    decompress_day(month / date)
print(f"    {len(days)} days materialized")
PY
        ); then
            echo "==> daystore ${ym}: materialization FAILED" >&2
            exit 1
        fi
        refresh_mtimes "${target}"
        touch "${marker}"
        echo "==> daystore ${ym}: staged (materialized)."
    ) 9>"${lock}"
}

stage_risk_factors() {
    local lock="${LOCK_DIR}/risk_factors.lock"
    local marker="${MARKER_DIR}/risk_factors.${MARKER_VERSION}.done"
    (
        flock -x 9
        if [ -f "${marker}" ]; then
            refresh_mtimes "${DATA_DIR}/1Hz_risk_factors"
            touch "${marker}"
            echo "==> risk_factors: already staged, skipping (mtimes refreshed)."
            exit 0
        fi
        mkdir -p "${DATA_DIR}/1Hz_risk_factors"
        rsync -az --size-only --info=progress2 --compress-level=1 \
            "${DATA_HOST}:${DATA_HOST_DATA}/1Hz_risk_factors/" \
            "${DATA_DIR}/1Hz_risk_factors/"
        refresh_mtimes "${DATA_DIR}/1Hz_risk_factors"
        touch "${marker}"
        echo "==> risk_factors: staged."
    ) 9>"${lock}"
}

stage_xs_anchor_stats() {
    # The cross-sectional (mu, sigma) tables behind the z-score target. One
    # ~200 KB npz per month, so the whole history is a few tens of MB — staged
    # wholesale like risk_factors rather than per month.
    #
    # XS_STATS_DIR IS THE TARGET DEFINITION, not a path preference. The tables
    # are (mu, sigma) OF a particular return; when the return changed to two
    # forward VWAP windows (2026-08-22) every table became stale for `return`.
    # A new directory rather than an in-place rebuild, so every result already
    # measured against the midpoint target stays reproducible -- and so a run
    # cannot silently mix the two, since the name travels into the lock, the
    # marker and the rsync source together.
    local sub="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}"
    local lock="${LOCK_DIR}/${sub}.lock"
    local marker="${MARKER_DIR}/${sub}.${MARKER_VERSION}.done"
    (
        flock -x 9
        # Count the tables, don't just trust the marker. node-local scratch evicts by
        # modtime and the marker is tiny and freshly touched, so it outlives
        # the data it vouches for; a marker alone let 372 of 600 runs die on
        # "Missing anchor-stat tables" after the directory was reaped. The
        # mosaic staging above has always re-checked index.json — this is the
        # same guard. rsync --size-only makes a genuine re-verify near-free.
        # The `|| true` is load-bearing. find exits 1 when the directory does
        # not exist, pipefail propagates that through wc, and `set -e` then
        # kills this staging subshell before a single echo — the job dies with
        # only the generic "one or more staging tasks failed", pointing at
        # nothing. Invisible on any node that has staged before, so it lay
        # dormant until the bs128 jobs moved to the L40S partition and met nodes
        # with an empty node-local scratch (6 jobs, 2026-08-15).
        _n_tables=$( { find "${DATA_DIR}/${sub}" -name '*.npz' 2>/dev/null || true; } | wc -l )
        if [ -f "${marker}" ] && [ "${_n_tables}" -ge "${XS_MIN_TABLES:-200}" ]; then
            refresh_mtimes "${DATA_DIR}/${sub}"
            touch "${marker}"
            echo "==> ${sub}: already staged (${_n_tables} tables), skipping."
            exit 0
        fi
        if [ -f "${marker}" ]; then
            echo "==> ${sub}: marker present but only ${_n_tables} tables — re-staging."
            rm -f "${marker}"
        fi
        mkdir -p "${DATA_DIR}/${sub}"
        if ! rsync -az --size-only --compress-level=1 \
            "${DATA_HOST}:${DATA_HOST_DATA}/${sub}/" \
            "${DATA_DIR}/${sub}/"; then
            echo "==> ${sub}: rsync FAILED" >&2
            exit 1
        fi
        refresh_mtimes "${DATA_DIR}/${sub}"
        touch "${marker}"
        echo "==> ${sub}: staged (${_n_tables} tables -> $(find "${DATA_DIR}/${sub}" -name '*.npz' | wc -l))."
    ) 9>"${lock}"
}

stage_metadata() {
    local lock="${LOCK_DIR}/metadata.lock"
    local marker="${MARKER_DIR}/metadata.${MARKER_VERSION}.done"
    (
        flock -x 9
        if [ -f "${marker}" ]; then
            touch "${DATA_DIR}/metadata.parquet" "${marker}" 2>/dev/null
            echo "==> metadata: already staged, skipping (mtime refreshed)."
            exit 0
        fi
        rsync -az "${DATA_HOST}:${DATA_HOST_METADATA}" "${DATA_DIR}/metadata.parquet"
        touch "${DATA_DIR}/metadata.parquet" "${marker}"
        echo "==> metadata: staged."
    ) 9>"${lock}"
}

# Stage an EXPLICIT, possibly non-contiguous list of months (plus the shared
# resources). stage_all covers a contiguous span, which is right for training
# but wrong for scoring: a checkpoint manifest can touch 2008-02 and 2023-11
# and nothing in between, and staging the range would pull 190 months to get 2.
stage_month_list() {
    mkdir -p "${DATA_DIR}" "${LOCK_DIR}" "${MARKER_DIR}"
    local months="$*"
    echo "==> Staging ${#} month(s): ${months}"
    local max_par="${STAGE_MAX_PARALLEL:-6}"
    local failed=0
    local pids=() next=0

    stage_risk_factors & pids+=($!)
    stage_metadata & pids+=($!)
    stage_xs_anchor_stats & pids+=($!)
    local ym
    for ym in ${months}; do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_mosaic_month "${ym}" & pids+=($!)
        stage_panel_month "${ym}" & pids+=($!)
    done
    while [ "${next}" -lt "${#pids[@]}" ]; do
        wait "${pids[next]}" || failed=1
        next=$((next + 1))
    done

    if [ "${failed}" -ne 0 ]; then
        echo "==> ERROR: one or more staging tasks failed." >&2
        return 1
    fi
    # A du -sh OVER THE WHOLE STAGED TREE, for a log line. ${DATA_DIR} is 3.9 TB
    # of mosaic on a warm node and the recursive stat runs for minutes -- inside
    # a GPU allocation, every job, to print a number nothing reads. Measured on
    # one job: staging 14.5 min of which this was a visible slice, at 0% GPU
    # throughout. Off unless asked for.
    if [ "${STAGE_REPORT_SIZE:-0}" = 1 ]; then
        echo "==> Data ready. Total size: $(du -sh "${DATA_DIR}" | cut -f1)"
    else
        echo "==> Data ready (set STAGE_REPORT_SIZE=1 for the du)."
    fi
}

# Stage a training span from the day-major store and only the SCORER'S months
# from the mosaic: [train_start, train_end] as daystore months, and
# [train_end, eval_end] as mosaic months + panels (the post-train IC eval fits
# its probe on the last training month and scores the eval month, and reads
# both from the mosaic when the panel cache misses). The shared resources
# (risk factors, metadata, anchor tables) are staged as before: the eval
# dataset the trainer still constructs reads the tables.
stage_all_daystore() {
    local train_start="$1" train_end="$2" eval_end="$3"
    mkdir -p "${DATA_DIR}" "${LOCK_DIR}" "${MARKER_DIR}"
    local train_months mosaic_months
    train_months=$(months_in_range "${train_start}" "${train_end}")
    mosaic_months=$(months_in_range "${train_end}" "${eval_end}")
    echo "==> Staging daystore months [${train_start} .. ${train_end}]: $(echo ${train_months} | tr '\n' ' ')"
    echo "==> Staging mosaic months [${train_end} .. ${eval_end}]: $(echo ${mosaic_months} | tr '\n' ' ')"
    local max_par="${STAGE_MAX_PARALLEL:-6}"
    local failed=0
    local pids=() next=0
    stage_risk_factors & pids+=($!)
    stage_metadata & pids+=($!)
    stage_xs_anchor_stats & pids+=($!)
    local ym
    for ym in ${train_months}; do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_daystore_month "${ym}" & pids+=($!)
    done
    for ym in ${mosaic_months}; do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_mosaic_month "${ym}" & pids+=($!)
    done
    for ym in $(panel_months_needed); do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_panel_month "${ym}" & pids+=($!)
    done
    while [ "${next}" -lt "${#pids[@]}" ]; do
        wait "${pids[next]}" || failed=1
        next=$((next + 1))
    done
    if [ "${failed}" -ne 0 ]; then
        echo "==> ERROR: one or more staging tasks failed; aborting before training." >&2
        return 1
    fi
    # A du -sh OVER THE WHOLE STAGED TREE, for a log line. ${DATA_DIR} is 3.9 TB
    # of mosaic on a warm node and the recursive stat runs for minutes -- inside
    # a GPU allocation, every job, to print a number nothing reads. Measured on
    # one job: staging 14.5 min of which this was a visible slice, at 0% GPU
    # throughout. Off unless asked for.
    if [ "${STAGE_REPORT_SIZE:-0}" = 1 ]; then
        echo "==> Data ready. Total size: $(du -sh "${DATA_DIR}" | cut -f1)"
    else
        echo "==> Data ready (set STAGE_REPORT_SIZE=1 for the du)."
    fi
}

# Stage everything needed for [TRAIN_START, TRAIN_END]. Parallel but each
# flock serializes within the resource.
stage_panel_month() {
    # ONE MONTH'S PRE-BUILT EVAL PANEL (scripts/eval/panel_cache.py).
    #
    # Per-month, like the mosaic and unlike the anchor tables: a probe-fit
    # panel is ~22 GB and an eval panel ~4 GB, so wholesale staging is out.
    #
    # OPTIONAL BY CONSTRUCTION. A month with no panel built simply does not
    # rsync and the scorer falls through to a live build -- the same result,
    # just ~3028 s slower. So this can be rolled out month by month, and a
    # cluster that has never seen a panel behaves exactly as before.
    #
    # ON BY DEFAULT, EVERYWHERE, AND THAT IS THE RULE: every sweep uses the
    # cache. It was off by default until 2026-09-11 on the theory that
    # node-local scratch's modtime eviction made a ~26 GB/month payload risky to leave
    # there. node-local scratch has ample space, the eviction is not a problem in
    # practice, and the cost of the old default was paid on every single sweep
    # -- the frozen-TSFM layer sweep re-decoded all 32 reported months from the
    # mosaic while a built panel sat on the data host beside it. A staging miss is
    # still free (falls through to a live decode), so defaulting ON can only
    # lose when a panel is absent, which costs nothing. Set
    # STAGE_PANEL_CACHE=0 to force a live decode, e.g. to verify that a cached
    # and an uncached run agree.
    local ym="$1"
    [ "${STAGE_PANEL_CACHE:-1}" = "1" ] || return 0
    local src="${DATA_HOST_PANEL_DIR:-lab/market-jepa-mosaic/panel_cache}"
    local dst="${DATA_DIR}/panel_cache"
    local lock="${LOCK_DIR}/panel-${ym}.lock"
    local marker="${MARKER_DIR}/panel-${ym}.${MARKER_VERSION}.done"
    mkdir -p "${dst}" "${LOCK_DIR}" "${MARKER_DIR}"
    (
        flock -x 9
        # Count .ready markers rather than trusting the done-marker: node-local scratch
        # evicts by modtime and a tiny marker outlives the tens of GB it
        # vouches for. Same guard stage_xs_anchor_stats learned the hard way.
        #
        # AND RSYNC EVEN SO. A month is staged under one directory PER CACHE
        # KEY (one per panel geometry: the 8-anchor eval key, the 36-anchor
        # probe-fit key, the risk-factor variants), and a .ready under ANY of
        # them used to end this function. On 2026-09-20 one node held 2008-10's
        # probe-fit panel complete and its eval panel as an EMPTY directory
        # (the leftover of a job that died on ENOSPC mid-copy): one .ready,
        # "already staged", and require_panel_cache failed the job on a panel
        # that sat built on the data host. Another node did the same with 2015-07 under two
        # keys nothing needed. The marker cannot know which keys a run wants,
        # so it decides only the message; the rsync is incremental -- a
        # complete key is a file-list compare, seconds -- and a missing or
        # partial key is pulled.
        _had=$( { find "${dst}" -path "*/${ym}/.ready" 2>/dev/null || true; } | wc -l )
        _marked=0; [ -f "${marker}" ] && _marked=1
        # --prune-empty-dirs so a key-variant with no panel for this month
        # does not leave an empty directory that looks like a staged one.
        # --size-only because refresh_mtimes touches the staged copy (the
        # mosaic learned this): under -a's mtime compare every re-verify
        # would ship the whole panel again. A panel is immutable once built.
        if rsync -az --size-only --prune-empty-dirs \
                --include='*/' --include="*/${ym}/***" --exclude='*' \
                "${DATA_HOST}:${src}/" "${dst}/" 2>/dev/null; then
            _n=$( { find "${dst}" -path "*/${ym}/.ready" 2>/dev/null || true; } | wc -l )
            if [ "${_n}" -gt 0 ]; then
                refresh_mtimes "${dst}"
                touch "${marker}"
                if [ "${_marked}" = 1 ] && [ "${_n}" -eq "${_had}" ]; then
                    echo "==> panel ${ym}: already staged (${_n} panel(s)), refreshed."
                else
                    echo "==> panel ${ym}: staged (${_n} panel(s), ${_had} were here)."
                fi
            else
                echo "==> panel ${ym}: none built upstream — scorer will build live."
            fi
        elif [ "${_had}" -gt 0 ]; then
            # Keep what is here; require_panel_cache decides if it is enough.
            refresh_mtimes "${dst}"
            echo "==> panel ${ym}: rsync failed — keeping the ${_had} staged panel(s)."
        else
            # NOT a job failure: the panel is an optimization, not an input.
            echo "==> panel ${ym}: rsync failed — scorer will build live."
        fi
    ) 9> "${lock}"
}

panel_months_needed() {
    # THE MONTHS THE SCORER ACTUALLY READS, not every month in the span. A
    # panel is ~22 GB in the probe-fit role and ~4 GB in the eval role, and the
    # span is DatasetConfig.train_span_months long, so staging one per training
    # month moves >150 GB per job to read two of them. The scorer fits its
    # probe on TRAIN_END's month and scores EVAL_START's; under
    # POST_TRAIN_PROBE=0 it reads only the latter. (stage_month_list is
    # deliberately NOT routed through this: a re-scoring run is handed the
    # exact months it needs and every one of them is scored.)
    #
    # POST_TRAIN_IC_EVAL=0 READS NEITHER. A train-only wave scores nothing, so
    # every panel it stages is dead weight -- 31 jobs x ~22 GB of probe-fit
    # panel on 2026-09-14, staged and never opened, because this function
    # honoured POST_TRAIN_PROBE but not the flag that turns scoring off
    # outright. Checked here so a caller cannot forget: POST_TRAIN_PROBE=0 is
    # then an optimization, not a requirement.
    if [ "${POST_TRAIN_IC_EVAL:-1}" = "0" ]; then
        echo ""
        return 0
    fi
    local out=""
    [ "${POST_TRAIN_PROBE:-1}" = "0" ] || out="${TRAIN_END:0:7}"
    local m
    for m in $(months_in_range "${EVAL_START}" "${EVAL_END}"); do
        case " ${out} " in *" ${m} "*) ;; *) out="${out} ${m}" ;; esac
    done
    echo "${out}"
}

stage_all() {
    local train_start="$1" train_end="$2"
    mkdir -p "${DATA_DIR}" "${LOCK_DIR}" "${MARKER_DIR}"
    local months
    months=$(months_in_range "${train_start}" "${train_end}")
    echo "==> Staging months for [${train_start} .. ${train_end}]: $(echo ${months} | tr '\n' ' ')"
    # Each task opens its own SSH connection to the data host. Unbounded fan-out is
    # fine for a one-month span (3 tasks) but a multi-year span opens dozens at
    # once and sshd starts refusing them (MaxStartups), which shows up as
    # rsync exit 255 / "kex_exchange_identification: Connection closed".
    local max_par="${STAGE_MAX_PARALLEL:-6}"
    local failed=0
    # Reap by PID, oldest outstanding first. `wait <pid>` returns the child's
    # status whether or not it has already exited, so no completion can slip
    # by uncollected — a `jobs -rp` count sees only *running* jobs and can
    # declare success while a failed task sits Done-but-unwaited (guaranteed
    # on short spans, where every task finishes before the drain starts).
    local pids=() next=0

    stage_risk_factors & pids+=($!)
    stage_metadata & pids+=($!)
    stage_xs_anchor_stats & pids+=($!)
    local ym
    for ym in ${months}; do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_mosaic_month "${ym}" & pids+=($!)
    done
    for ym in $(panel_months_needed); do
        if [ $(( ${#pids[@]} - next )) -ge "${max_par}" ]; then
            wait "${pids[next]}" || failed=1
            next=$((next + 1))
        fi
        stage_panel_month "${ym}" & pids+=($!)
    done
    while [ "${next}" -lt "${#pids[@]}" ]; do
        wait "${pids[next]}" || failed=1
        next=$((next + 1))
    done

    if [ "${failed}" -ne 0 ]; then
        echo "==> ERROR: one or more staging tasks failed; aborting before training." >&2
        return 1
    fi
    # A du -sh OVER THE WHOLE STAGED TREE, for a log line. ${DATA_DIR} is 3.9 TB
    # of mosaic on a warm node and the recursive stat runs for minutes -- inside
    # a GPU allocation, every job, to print a number nothing reads. Measured on
    # one job: staging 14.5 min of which this was a visible slice, at 0% GPU
    # throughout. Off unless asked for.
    if [ "${STAGE_REPORT_SIZE:-0}" = 1 ]; then
        echo "==> Data ready. Total size: $(du -sh "${DATA_DIR}" | cut -f1)"
    else
        echo "==> Data ready (set STAGE_REPORT_SIZE=1 for the du)."
    fi
}

require_panel_cache() {
    # THE EVAL PANEL MUST ALREADY EXIST. A sweep that starts without it spends
    # its GPU hours and then re-decodes the same month-panel from the mosaic at
    # the end -- ~3028 s per run, for a panel that is a function of (month,
    # anchors, geometry) and NOT of the checkpoint, so every run of every sweep
    # pays it again for a byte-identical result. It is also silent: the scorer
    # falls through to a live build and the numbers come out right, just hours
    # late, which is why it went unnoticed across whole campaigns.
    #
    # So the rule is fail-fast (2026-09-13): no panel, no job. Build it first
    #     MJ_PANEL_CACHE=<root> uv run scripts/eval/build_panel_cache.py \
    #         --months <train-month> --roles probe eval --jobs 40
    # on the machine that holds the mosaic, then relaunch. That build is the
    # SAME work the job would have done, done once and reused by every arm.
    #
    # TWO ROLES PER MONTH, and both are required: the training month is the
    # probe-fit panel at TRAIN_ANCHORS_PER_DAY anchors, and the month after it
    # is the eval panel at 8. The anchor count is part of the cache key, so a
    # month built in one role does not satisfy the other.
    #
    # EXEMPT: full_data_multihead, whose 203 months are scored once each -- a
    # cache there is pure cost (see scripts/eval/build_panel_cache.py). Also
    # skipped when STAGE_PANEL_CACHE=0, which asks for a live decode on
    # purpose (verifying that a cached and an uncached run agree), and when
    # PANEL_CACHE_REQUIRED=0, which a run on a NON-DEFAULT panel geometry
    # needs: risk-factor channels, norm_mode=none and a fixed aggregation each
    # key differently, and this checks the default geometry only.
    local train_ym="$1" eval_ym="$2"
    # NOTHING SCORED, NOTHING REQUIRED. This function exit 1's the whole job,
    # and it used to do so for a panel a POST_TRAIN_IC_EVAL=0 wave would never
    # open -- the same hole panel_months_needed had. Train-only waves must not
    # be gated on scoring inputs.
    [ "${POST_TRAIN_IC_EVAL:-1}" = "0" ] && return 0
    [ "${PANEL_CACHE_REQUIRED:-1}" = "1" ] || return 0
    # HEAD-ONLY SCORING NEEDS ONLY THE EVAL PANEL. With POST_TRAIN_PROBE=0 the
    # probe-fit month is never embedded, so requiring its 36-anchor panel would
    # block jobs on a file nothing reads.
    local roles_probe=1
    [ "${POST_TRAIN_PROBE:-1}" = "0" ] && roles_probe=0
    [ "${STAGE_PANEL_CACHE:-1}" = "1" ] || return 0
    # EVERY sweep in the bundle must be exempt for the bundle to be, and a
    # bundle with no sweeps at all is NOT exempt. BUNDLE_SWEEPS is set only by
    # train_bundle_body.sh; slurm_variance_decomp.sh calls this without it, and
    # with exempt starting at 1 the loop body never ran, so the check returned
    # early and variance decomp kept doing the silent live decode this function
    # exists to stop.
    # EVERY sweep in the bundle must be exempt for the bundle to be, and a
    # bundle with no sweeps at all is NOT exempt. exempt starts at 0 and only a
    # multihead entry raises it, so an unset BUNDLE_SWEEPS falls through to the
    # check. It started at 1, and BUNDLE_SWEEPS is set only by
    # train_bundle_body.sh -- slurm_variance_decomp.sh calls this without it, so
    # the loop body never ran, exempt stayed 1, and variance decomp kept doing
    # the silent live decode this function exists to stop.
    #
    # Counting with ${#BUNDLE_SWEEPS[@]} instead would be an unbound-variable
    # error under set -u when it is unset; "${BUNDLE_SWEEPS[@]}" in a for is
    # special-cased and expands to nothing, which is why this counts in-loop.
    local exempt=0 s
    for s in "${BUNDLE_SWEEPS[@]}"; do
        case "${s}" in
            *full_data_multihead*) exempt=1 ;;
            *)                     exempt=0; break ;;
        esac
    done
    [ "${exempt}" = "1" ] && return 0

    local anchors
    anchors=$(grep -oE '^TRAIN_ANCHORS_PER_DAY[[:space:]]*=[[:space:]]*[0-9]+' \
        "${LOCAL_REPO:-.}/scripts/eval/xs_ic_eval.py" 2>/dev/null \
        | grep -oE '[0-9]+$')
    anchors="${anchors:-36}"
    MJ_PANEL_CACHE="${DATA_DIR}/panel_cache" \
    PC_TRAIN="${train_ym}" PC_EVAL="${eval_ym}" PC_ANCHORS="${anchors}" \
    PC_STATS="${XS_STATS_DIR:-xs_anchor_stats_fwdvwap60}" PC_PROBE="${roles_probe}" \
        uv run python - <<'PY'
import os, sys
sys.path.insert(0, "scripts/eval")
import panel_cache as pc

root = pc.cache_root()
want = [(os.environ["PC_EVAL"], 8, "eval")]
if os.environ.get("PC_PROBE", "1") == "1":
    want.insert(0, (os.environ["PC_TRAIN"], int(os.environ["PC_ANCHORS"]), "probe-fit"))
missing = []
for ym, n, role in want:
    key = pc.panel_key(anchors_per_day=n, stats_tag=os.environ["PC_STATS"],
                       has_rf=False, norm_groups=None, fixed_agg=None,
                       seq_len=2048)
    if root is None or not pc.is_built(root, ym, key):
        missing.append(f"{ym} ({role}, {n} anchors, key {key['hash']})")
for m in missing:
    print(f"==> MISSING PANEL: {m}", file=sys.stderr)
sys.exit(1 if missing else 0)
PY
}

# ── Reclaim node scratch when it is short ────────────────────────────────────
#
# Node-local scratch is NEVER FREED by anything above. The modtime
# eviction this file's header relies on is slow: one node filled up with
# months from sweeps weeks old, hit 100%, and eleven jobs died in
# staging on ENOSPC (rsync broken pipes, uv install failures) while SLURM
# kept handing the node new jobs to die the same way.
#
# NOT A DEFAULT SWEEP. A staged month is reused by the next job that lands on
# the node, which saves it the resync, so nothing is touched while there is
# room. Only when free space is under SCRATCH_MIN_FREE_GB does this delete --
# and then only months no running job of ours on this node can be using.
#
# HOW "IN USE" IS DECIDED, with no bookkeeping: every stager touches a month's
# marker when it stages it AND when it finds it already staged, so a running
# job has touched every month it needs no earlier than its own start. A
# marker last touched BEFORE THE OLDEST RUNNING JOB OF OURS ON THIS NODE
# STARTED therefore belongs to no running job, and its month can go. The
# calling job is itself running and has staged nothing yet, so its months
# are not at risk. Data directories with no marker at all (a crashed job's
# partial copy) are deleted on the same age rule. Per-job venvs and uv caches
# of job ids no longer on the node go too.
reclaim_scratch_if_low() {
    local min_free="${SCRATCH_MIN_FREE_GB:-400}"
    [ "${min_free}" != "0" ] || return 0
    local free
    free=$( { df -BG --output=avail "${DATA_DIR}" 2>/dev/null || true; } | tail -1 | tr -dc '0-9')
    if [ -z "${free}" ]; then
        echo "==> scratch: cannot read free space under ${DATA_DIR}; not reclaiming."
        return 0
    fi
    if [ "${free}" -ge "${min_free}" ]; then
        echo "==> scratch: ${free}G free (>= ${min_free}G), nothing reclaimed."
        return 0
    fi
    local host oldest cutoff
    host="$(hostname -s)"
    oldest=$( { squeue -h -u "${USER}" -w "${host}" -t R -o %S 2>/dev/null || true; } | sort | head -1)
    if [ -z "${oldest}" ]; then
        # Not inside a job, or squeue is unreachable: anything older than an
        # hour is fair game, and a fresh copy is by definition being made now.
        cutoff=$(( $(date +%s) - 3600 ))
    else
        cutoff=$(date -d "${oldest}" +%s)
    fi
    echo "==> scratch: ${free}G free (< ${min_free}G); reclaiming months untouched since $(date -d "@${cutoff}" +%FT%T) ..."
    local marker kind ym year mm target n=0
    for marker in "${MARKER_DIR}"/daystore-*.done "${MARKER_DIR}"/mosaic-*.done "${MARKER_DIR}"/panel-*.done; do
        [ -f "${marker}" ] || continue
        [ "$(stat -c %Y "${marker}")" -lt "${cutoff}" ] || continue
        kind="$(basename "${marker}")"; kind="${kind%%-*}"
        ym="$(basename "${marker}")"; ym="${ym#*-}"; ym="${ym%%.*}"
        year="${ym%%-*}"; mm="${ym##*-}"
        case "${kind}" in
            daystore) target="${DATA_DIR}/1Hz_daystore/${year}/${mm}" ;;
            mosaic)   target="${DATA_DIR}/1Hz_mosaic_mnth/${year}/${mm}" ;;
            panel)    target="" ;;
        esac
        if [ "${kind}" = "panel" ]; then
            rm -rf "${DATA_DIR}"/panel_cache/*/"${ym}"
        else
            rm -rf "${target}"
        fi
        rm -f "${MARKER_DIR}/${kind}-${ym}".*
        n=$(( n + 1 ))
        echo "    reclaimed ${kind} ${ym}"
    done
    # Unmarked month directories: a crashed job's partial copy. Same age rule,
    # on the directory itself.
    local d
    for d in "${DATA_DIR}"/1Hz_daystore/*/*/ "${DATA_DIR}"/1Hz_mosaic_mnth/*/*/ "${DATA_DIR}"/panel_cache/*/*/; do
        [ -d "${d}" ] || continue
        ym="$(basename "$(dirname "${d}")")-$(basename "${d}")"
        case "${d}" in
            */1Hz_daystore/*) kind=daystore ;;
            */1Hz_mosaic_mnth/*) kind=mosaic ;;
            *) kind=panel; ym="$(basename "${d}")" ;;
        esac
        [ -f "${MARKER_DIR}/${kind}-${ym}.${MARKER_VERSION}.done" ] && continue
        [ "$(stat -c %Y "${d}")" -lt "${cutoff}" ] || continue
        rm -rf "${d}"
        n=$(( n + 1 ))
        echo "    reclaimed unmarked ${kind} ${ym}"
    done
    # Per-job venvs and uv caches of jobs no longer on this node.
    local live v id
    live=" $( { squeue -h -u "${USER}" -w "${host}" -o %i 2>/dev/null || true; } | tr '\n' ' ') "
    for v in "${SCRATCH_BASE:-$(dirname "${DATA_DIR}")}"/market-jepa-venv-[0-9]* \
             "${SCRATCH_BASE:-$(dirname "${DATA_DIR}")}"/.cache/uv-[0-9]*; do
        [ -e "${v}" ] || continue
        id="${v##*-}"
        case "${live}" in *" ${id} "*) continue ;; esac
        rm -rf "${v}"
        echo "    reclaimed dead ${v}"
    done
    free=$( { df -BG --output=avail "${DATA_DIR}" 2>/dev/null || true; } | tail -1 | tr -dc '0-9')
    echo "==> scratch: reclaimed ${n} month(s); ${free:-?}G free."
    return 0
}
