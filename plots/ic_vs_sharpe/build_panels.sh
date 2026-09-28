#!/usr/bin/env bash
# Build the forecast-panel cache: one npz per (arm, eval month) holding mu and
# the realized return at EVERY horizon, in return units, plus the quoted
# half-spread and the covariance history.
#
# WHY THIS EXISTS. Staging, streaming the six-month fit pool and fitting the
# ridge is ~80 s an (arm, month); reading a panel back is milliseconds. Every
# question about horizon, selection, holding period or cost that does not
# change the probe is a read of this cache rather than a re-fit. The probe is
# fit for all six horizons in ONE pass -- X'X dominates and each extra horizon
# adds only an X'y -- so the cache costs about what a single horizon used to.
#
# The config sweep is pinned to one configuration on purpose: this run exists
# for the panels, not for the Sharpe numbers, and the full 4,080-config sweep
# would multiply the cost by five for output this job discards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${RESULTS:-/data/lab/ic_sharpe_configs/results}"
PANELS="${PANELS:-/data/lab/ic_sharpe_configs/panels}"
LOGS="${LOGS:-/data/lab/ic_sharpe_configs/panel_logs}"
SCRATCH="${SCRATCH:-/data/lab/ic_sharpe_configs/panel_run}"
# Each worker's BLAS gets its own threads; the product must stay under the
# machine's core count or the workers fight each other for the gram.
WORKERS="${WORKERS:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

ONE_CONFIG='{"cost_model":"cross","weighting":"equal","risk_model":"diagonal","rebalance":"every_decision","annualization":"decisions","universe":"all","selection":"all"}'

mkdir -p "$PANELS" "$LOGS" "$SCRATCH"

# The (arm, month) pairs that actually exist are the ones the third sweep
# scored; deriving them from its results avoids re-deriving the manifest logic
# and avoids asking for the 36 (arm, month) jobs that were never embedded.
cd "$ROOT"
uv run python - "$RESULTS" > "$LOGS/jobs.txt" <<'PY'
import glob, json, sys
pairs = set()
for path in glob.glob(f"{sys.argv[1]}/*.json"):
    row = json.load(open(path))
    pairs.add((row["arm"], row["eval_month"]))
for arm, month in sorted(pairs):
    print(arm, month)
PY

total=$(wc -l < "$LOGS/jobs.txt")
echo "==> $total (arm, month) panels, $WORKERS workers x $OMP_NUM_THREADS threads"
date

run_one() {
    arm="$1"; month="$2"
    out="$PANELS/$arm-$month-panel.npz"
    if [ -f "$out" ]; then echo "    skip $arm $month (cached)"; return 0; fi
    uv run python plots/ic_vs_sharpe/ic_sharpe_configs.py \
        --arms "$arm" --months "$month" \
        --out-dir "$SCRATCH" --panel-out "$PANELS" \
        --config-slice "$ONE_CONFIG" \
        > "$LOGS/$arm-$month.log" 2>&1 \
        && echo "    ok   $arm $month" \
        || echo "    FAIL $arm $month (see $LOGS/$arm-$month.log)"
}
export -f run_one
export PANELS LOGS SCRATCH ONE_CONFIG

xargs -a "$LOGS/jobs.txt" -n 2 -P "$WORKERS" bash -c 'run_one "$0" "$1"'

echo "==> done"
date
ls -1 "$PANELS"/*-panel.npz 2>/dev/null | wc -l | xargs echo "panels written:"
grep -l . "$LOGS"/*.log 2>/dev/null | xargs grep -l "Traceback" 2>/dev/null | head -20 || true
