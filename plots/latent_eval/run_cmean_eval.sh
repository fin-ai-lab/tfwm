#!/bin/bash
# run_cmean_eval.sh — the frozen TSFMs' latent suite with CHANNELS AVERAGED.
#
# WHY. The prediction evals read each TSFM with its nine per-channel states
# averaged into one d_model vector (channel_pool="mean"); the _meanpool latent
# wave (run_meanpool_reeval.sh) concatenated them, 9 x d_model wide. This puts
# the latent half on the prediction half's readout. Everything else is that
# wave's protocol unchanged: the same run_eval.sh stages, the same mean over
# valid patches, every layer.
#
# THE SERIES are tsfm_<fam>_cmean_l<L> (industry_nn_sweep.TSFM_FAMILIES,
# resolve_family in pretrained_tsfm): new keys and new cache files, so nothing
# concatenated is overwritten. No floor and no trained arms here -- those are
# already in the _meanpool / _6mo / _multi tags, which the table merges.
#
# TWO PANELS, two tags:
#   _cmean      the 31 reported eval months (run_meanpool_reeval.sh's list)
#   _cmean_opt  the 5-month Optimization Set (holdout_months.py) -- the layer
#               picks (the stars) are re-made here, never on the 31
#
# Usage (from the repo root):
#   tmux new -s latent_cmean -d 'bash plots/latent_eval/run_cmean_eval.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

export TMPDIR=lab/market-jepa-checkpoints/_scratch/latent_eval
mkdir -p "${TMPDIR}"
export TSFM_FAMILIES="kronos_cmean timesfm3_cmean chronos2_cmean"
# Same concurrency as the _meanpool wave, sized off the same box.
export JOBS="${JOBS:-6}"
export STAGGER="${STAGGER:-90}"

KEYS="$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import TSFM_KEYS
print(' '.join(k for f in '${TSFM_FAMILIES}'.split() for k in TSFM_KEYS[f]))")"
# Stage 1: each family's layers consecutive (fixed_panel_metrics holds one
# family's multi-layer forward at a time).
export MODELS="${KEYS// /,}"
export SERIES="${KEYS}"

REPORTED="$(uv run python -c "
import sys; sys.path.insert(0, 'plots'); sys.path.insert(0, 'plots/metrics')
from style import load_sweep_months
from metrics import SUP_SPAN_NO_SPAN
from stable_finance.dataset import next_month
print(' '.join(sorted(next_month(m) for m in
                      set(load_sweep_months()) - SUP_SPAN_NO_SPAN)))")"
OPT="$(uv run python -c "
import sys; sys.path.insert(0, 'scripts/experiments')
from holdout_months import HOLDOUT_MONTHS
from stable_finance.dataset import next_month
print(' '.join(sorted(next_month(m) for m in HOLDOUT_MONTHS)))")"

echo "=== $(echo ${KEYS} | wc -w) series; reported $(echo ${REPORTED} | wc -w) months, optimization $(echo ${OPT} | wc -w)"

# The optimization set first: it is 5 months, and its picks gate the figures.
# Each panel's exit status is RECORDED, not swallowed: run_eval.sh's stage 5
# summary table needs trained arms and exits non-zero on a TSFM-only tag
# after every result is already written, which must not stop the next panel.
# Read the per-month logs and the stage-4 .out files, not just this line.
for spec in "_cmean_opt:${OPT}" "_cmean:${REPORTED}"; do
    TAG="${spec%%:*}" MONTHS="${spec#*:}" bash plots/latent_eval/run_eval.sh \
        && echo "=== ${spec%%:*}: run_eval.sh exit 0" \
        || echo "=== ${spec%%:*}: run_eval.sh exit $? (check logs${spec%%:*})"
done
echo "CMEAN_WAVE_DONE"
