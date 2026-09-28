#!/usr/bin/env bash
# Rebuild all three core results. Run from the repo root:  plots/core/regen.sh
#
# WHY THIS EXISTS. Each generator has at least one flag that is easy to omit
# and that fails QUIETLY when omitted -- the table still renders, just wrong or
# bare. Those flags live here so they are not retyped from memory:
#
#   probe_fit_table.py and fixed_panel_table.py defaults are already correct (n column auto,
#   caption built by default_caption()), so they take no flags. RankMe is NOT in the
#   latent table any more: its appendix table lives outside core, in
#   plots/latent_eval/rankme/rankme_table.py.
#
#   probe_fit_breadth.py must be run TWICE: once bare, once --with-floor. The
#   two figures are different files and the second is not a superset.
#
# NEVER pipe these to `head`. SIGPIPE kills the writer mid-write and the shell
# still reports exit 0, which has already produced one stale table in this repo.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CORE=plots/core
LOG=$(mktemp -d)
echo "logs: $LOG"

echo "[1/3] probe_fit_table.tex"
uv run python $CORE/probe_fit_table.py --latex --out $CORE/probe_fit_table.tex \
    >"$LOG/probe_fit_table.log" 2>&1
grep -E "CAPTION|WARN|warn" "$LOG/probe_fit_table.log" || true

echo "[2/3] fixed_panel_table.tex"
uv run python $CORE/fixed_panel_table.py --latex --out $CORE/fixed_panel_table.tex \
    >"$LOG/fixed_panel_table.log" 2>&1

echo "[3/3] probe_fit_breadth.png / probe_fit_breadth_floor.png"
uv run python $CORE/probe_fit_breadth.py             >"$LOG/breadth.log" 2>&1
uv run python $CORE/probe_fit_breadth.py --with-floor >"$LOG/breadth_floor.log" 2>&1
tail -1 "$LOG/breadth.log"; tail -1 "$LOG/breadth_floor.log"

echo "OK -- all three rebuilt"
