#!/bin/bash
# run_meanpool_reeval.sh — the 2026-09-14 latent re-evaluation.
#
# WHY THIS RUN EXISTS. Every latent result before it was read at the readout
# each checkpoint TRAINED with: the supervised specialists at pool="last" (one
# patch of the day), the LeJEPA arms at "cls" (their OBJECTIVE is computed on
# the mean, their backbone readout is not -- see the archive README), the SSL
# baselines spread across cls/mean/last/max, and the floor at "cls".
# 7ee9cdf (2026-09-08) pinned the latent suite to the mean but only reached
# _load_backbone call sites, while every MODEL_ORDER key is manifest-resolved
# and goes through load_encoder, which had no pool parameter at all. Fixed
# 2026-09-14; everything produced before that is in
# plots/latent_eval/_archive/pre_meanpool_readout/ and must not be quoted.
#
# THE ROSTER. What this box can still run:
#   * the three supervised span specialists (581eb2 wave, the checkpoints
#     plots/metrics/all_metrics_supervised_ic.png draws), resolved per eval
#     month through manifest_supspan.json;
#   * the frozen TSFMs kronos / timesfm3 / chronos2, every layer. These are
#     already mean-pooled -- PretrainedTSFM defaults time_pool="mean" and every
#     captured layer goes through _pool_time -- so the encoder side of their
#     numbers is unchanged; it is their FLOOR that was wrong, and the floor is
#     rebuilt here with them.
#   * the random-init floor: `random` in stage 1, randvit_s0..4 in stage 2.
# The 18-method campaign is NOT here: ssl-ic-final-222d99-* and the pair_*
# projects are gone from this box, and ff_fullday_cache no longer holds the
# reported months, so those arms need retraining, not re-scoring.
#
# THE PANEL IS 31 EVAL MONTHS, not 32. A six-month span ending 2008-02 starts
# before the data, so the launcher never produced that specialist and no rerun
# will (metrics.SUP_SPAN_NO_SPAN). Dropping its eval month 2008-03 from the
# WHOLE run keeps every arm, TSFMs included, on one panel -- a rank is only
# meaningful if every model earned it on the same months.
#
# Usage (from the repo root):
#   tmux new -s latent_meanpool -d 'bash plots/latent_eval/run_meanpool_reeval.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

export TMPDIR=/data/lab/market-jepa-checkpoints/_scratch/latent_eval
mkdir -p "${TMPDIR}"

# The supervised rows live in a machine-local manifest built from the 581eb2
# checkpoint tree (plots/latent_eval/build_supspan_manifest.py); the repo manifest
# carries nothing for this wave. run_eval.sh would glob this in anyway -- named
# explicitly so the run does not depend on what else is in TMPDIR.
export MJ_NOCLAMP_MANIFEST="plots/metrics/noclamp_manifest.json:${TMPDIR}/manifest_supspan.json"

export TAG="${TAG:-_meanpool}"
export TSFM_FAMILIES="${TSFM_FAMILIES:-kronos timesfm3 chronos2}"

# FOUR MONTHS IN FLIGHT. run_eval.sh is month-major, so one month's raw data is
# read once and stages 1-3 all consume it -- but a single month alternates
# between collecting (CPU/IO, GPU at 0%) and forwarding (GPU at 93%), so serial
# months leave the card idle for a large fraction of the run. Four overlapping
# months keep something on the GPU at all times.
#
# Sized off the box, not guessed: a month worker measured ~4 GB RSS and ~1.3
# cores against 64 cores / 188 GB, so CPU and memory are not the binding
# constraint; the GPU is, and each worker holds at most ONE TSFM family at a
# time (the bank frees the previous before loading the next) on a 46 GB card.
# Raise it if `nvidia-smi` still shows idle gaps, lower it if chronos2's
# forwards start colliding on memory.
export JOBS="${JOBS:-6}"
# Offset the first six so they do not collect and forward in lockstep -- four
# simultaneous starts measured 48% mean GPU because all four sat in the same
# phase. 90 s is about a collection phase's head start.
export STAGGER="${STAGGER:-90}"

# 31 eval months: the sweep's train months, minus the one with no span, +1.
export MONTHS="$(uv run python -c "
import sys; sys.path.insert(0, 'plots'); sys.path.insert(0, 'plots/metrics')
from style import load_sweep_months
from metrics import SUP_SPAN_NO_SPAN
from stable_finance.dataset import next_month
print(' '.join(sorted(next_month(m) for m in
                      set(load_sweep_months()) - SUP_SPAN_NO_SPAN)))")"

# Stage 1 models: the three specialists, then each TSFM family's layers in
# depth order (fixed_panel_metrics keeps ONE family's multi-layer forward in
# memory, so a family's keys must be consecutive), then the floor.
export MODELS="$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import TSFM_KEYS
fams = '${TSFM_FAMILIES}'.split()
sup = ['sup_return_w8', 'sup_vol_w8', 'sup_spread_w8']
print(','.join(sup + [k for f in fams for k in TSFM_KEYS[f]] + ['random']))")"

# Stage 4 series: the same roster by emb key. tsfm_* are stripped from stage
# 2's list inside run_eval.sh (they build through --families) and kept here.
export SERIES="$(uv run python -c "
import sys; sys.path.insert(0, 'plots/latent_eval/fixed_panel')
from industry_nn_sweep import TSFM_KEYS
fams = '${TSFM_FAMILIES}'.split()
sup = ['sup_return_w8', 'sup_vol_w8', 'sup_spread_w8']
print(' '.join(sup + [k for f in fams for k in TSFM_KEYS[f]]))") randvit_s0 randvit_s1 randvit_s2 randvit_s3 randvit_s4"

echo "=== TAG=${TAG}  JOBS=${JOBS}"
echo "=== months (${MONTHS// /,}) "
echo "=== $(echo ${MONTHS} | wc -w) months, $(echo ${MODELS//,/ } | wc -w) stage-1 models, $(echo ${SERIES} | wc -w) stage-4 series"
exec bash plots/latent_eval/run_eval.sh
