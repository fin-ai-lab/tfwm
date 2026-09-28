#!/usr/bin/env bash
# Run the IC-vs-Sharpe configuration sweep across every (arm, eval month) on
# this machine. CPU only, nothing staged, nothing re-embedded: the sweep reads
# the probe-breadth embeddings and the cached market panels that are already on
# local disk.
#
# ONE PROCESS PER (arm, month), because the work is embarrassingly parallel and
# a crashed job must not take its neighbours with it. Each worker pins its BLAS
# thread count: without that, every worker tries to use the whole machine and
# they spend their time descheduling each other rather than working.
#
# Jobs whose output json already exists are skipped, so an interrupted sweep is
# resumed by re-running this script.
#
#   WORKERS=8 BLAS_THREADS=8 plots/ic_vs_sharpe/run_ic_sharpe_configs.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-/data/lab/ic_sharpe_configs/results}"
LOG_DIR="${LOG_DIR:-/data/lab/ic_sharpe_configs/logs}"
FIT_ROWS="${FIT_ROWS:-200000}"
CONFIG_SLICE="${CONFIG_SLICE:-}"

# 64 logical cores on this box. A single job turns out to use only ~1.2 cores
# -- the cost-aware solve is a sequential walk over decisions, and BLAS cannot
# thread across it -- so parallelism belongs at the JOB level, not inside one.
# Many workers on few threads each beats the reverse by a wide margin here.
# Measured: one full-space job is ~18 min and ~1.2 effective cores, holding
# ~3 GB. 30 workers therefore fit both the 64 logical cores and the 188 GB of
# RAM with headroom, and put the 615-job sweep at roughly six hours.
WORKERS="${WORKERS:-30}"
BLAS_THREADS="${BLAS_THREADS:-2}"

# Every arm with staged embeddings at the PREDICTIVE readout: the 13 arms
# re-pooled to last by the pblast/pblast2 sweeps, plus the supervised arms,
# CoST and the random-init floor, which trained or are built at last already
# (plots/core/readout.py::LAST_BY_CONSTRUCTION). The sweep script resolves and
# enforces that per (arm, month); an arm with no protocol-valid tag fails loudly
# rather than being read at the wrong pool.
ARMS="${ARMS:-byol_6mo cpc_6mo dino_6mo ijepa_6mo mae_6mo pair_k2_6mo \
pair_k2ind_6mo pair_noise_6mo pair_rrc_6mo pair_warp_6mo \
tfc_6mo timemae_6mo ts2vec_6mo cost_6mo \
sup_return_w8 sup_vol_w8 sup_spread_w8 sup_multi_w8 \
randinit_s42 randinit_s43 randinit_s44}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$REPO"

# The manifest is the source of truth for which months an arm actually has.
mapfile -t JOBS < <(uv run python - "$ARMS" <<'PY'
import json, sys
arms = set(sys.argv[1].split())
seen = set()
for path in ("/data/lab/probe_breadth/manifest_all.index.json",
             "/data/lab/probe_breadth/manifest_floor.index.json"):
    try:
        rows = json.load(open(path))
    except OSError:
        continue
    for row in rows:
        key = (row["series_key"], row["eval_month"])
        if row["series_key"] in arms and key not in seen:
            seen.add(key)
# ORDERED BY MONTH, NOT BY ARM. The across-methods correlation needs several
# arms on the same months, so a sweep grouped by arm produces nothing usable
# until it is most of the way done. Sorting by month means every completed wave
# is a set of arms that can be compared, and an interrupted sweep still answers
# the question on the months it reached.
for month, arm in sorted((m, a) for a, m in seen):
    print(f"{arm} {month}")
PY
)

TODO=()
for job in "${JOBS[@]}"; do
    read -r arm month <<<"$job"
    [ -f "${OUT_DIR}/${arm}-${month}.json" ] || TODO+=("$job")
done

echo "==> ${#JOBS[@]} (arm, month) pair(s); ${#TODO[@]} still to run"
echo "==> ${WORKERS} workers x ${BLAS_THREADS} BLAS threads, out=${OUT_DIR}"
[ ${#TODO[@]} -eq 0 ] && { echo "==> nothing to do"; exit 0; }

printf '%s\n' "${TODO[@]}" | xargs -P "$WORKERS" -I{} bash -c '
    read -r arm month <<<"{}"
    export OMP_NUM_THREADS='"$BLAS_THREADS"' \
           OPENBLAS_NUM_THREADS='"$BLAS_THREADS"' \
           MKL_NUM_THREADS='"$BLAS_THREADS"' \
           NUMEXPR_NUM_THREADS='"$BLAS_THREADS"'
    slice_args=()
    [ -n "'"$CONFIG_SLICE"'" ] && slice_args=(--config-slice "'"$CONFIG_SLICE"'")
    uv run plots/ic_vs_sharpe/ic_sharpe_configs.py \
        --arms "$arm" --months "$month" \
        --out-dir "'"$OUT_DIR"'" --fit-rows '"$FIT_ROWS"' \
        "${slice_args[@]}" \
        > "'"$LOG_DIR"'/${arm}-${month}.log" 2>&1 \
        && echo "  done ${arm} ${month}" \
        || echo "  FAILED ${arm} ${month} (see '"$LOG_DIR"'/${arm}-${month}.log)"
'

DONE=$(find "$OUT_DIR" -name '*.json' | wc -l)
echo "==> sweep finished: ${DONE}/${#JOBS[@]} result file(s) in ${OUT_DIR}"
