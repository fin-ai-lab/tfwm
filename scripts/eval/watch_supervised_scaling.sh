#!/bin/bash
# watch_supervised_scaling.sh — redraw plots/scaling/supervised_scaling.png as
# the scaling wave lands on bll01.
#
#   tmux new-session -d -s scaling-watch \
#       'scripts/eval/watch_supervised_scaling.sh 2f4703 120'
#
# Every INTERVAL seconds: collect the wave at COMMIT (the collector refuses to
# pool waves, so the hash is required while the superseded 2026-09-19 tree is
# still on disk), then draw. A collect with nothing landed yet, or a draw with
# nothing to draw, is not an error here -- the previous figure stays.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
COMMIT="${1:?usage: watch_supervised_scaling.sh <commit> [interval-seconds]}"
INTERVAL="${2:-120}"
LOG="${HOME}/.cache/market-jepa/scaling-watch.log"
mkdir -p "$(dirname "${LOG}")"
while true; do
    {
        echo "== $(date '+%F %T') collect --commit ${COMMIT}"
        if uv run scripts/eval/collect_supervised_scaling.py --commit "${COMMIT}" 2>&1 | tail -4; then
            uv run plots/scaling/supervised_scaling.py 2>&1 | tail -12
        fi
    } >> "${LOG}" 2>&1
    tail -3 "${LOG}"
    sleep "${INTERVAL}"
done
