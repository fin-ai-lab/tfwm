#!/bin/bash
# run_multihead_eval.sh — the full-history supervised MULTIHEAD trunk on our panel.
#
# The fourth supervised row. sup_return/vol/spread_w8 are three specialists at
# one task each; this is the single trunk with three heads that replaced them
# for the full-history sweep, and it has been a MODEL_ORDER key with no rows
# behind it because its wave is one project PER BUNDLE of training months.
#
# THE CHECKPOINTS ARE THE ONES plots/full_data_multihead DRAWS, selected by
# that figure's own filters (build_multihead_manifest.py imports them), so the
# latent row and the IC figure cannot describe different models under one name.
#
# NOT 31 MONTHS. The sweep is still running: 25 of the panel's 31 eval months
# have a finished run with a checkpoint, and the other six have nothing on disk
# at all. MONTHS comes from the manifest, so this run covers exactly what
# exists and the table's n column reports 25 against the other rows' 31. Re-run
# the builder and this script when more land -- stage 1's shard skip makes the
# repeat cost only the new months.
#
# NEITHER THE FLOOR NOR THE TSFMs ARE RE-RUN: they are in _meanpool, and
# fixed_panel_table.py --tags merges the files. Stage 5's own _multi table has
# no floor row and therefore no strike marks; the merged one does.
#
# Usage (from the repo root):
#   uv run python plots/latent_eval/build_multihead_manifest.py \
#       lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_multi.json
#   tmux new -s latent_multi -d 'bash plots/latent_eval/run_multihead_eval.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

export TMPDIR=lab/market-jepa-checkpoints/_scratch/latent_eval
mkdir -p "${TMPDIR}"
MANIFEST="${TMPDIR}/manifest_multi.json"
[ -f "${MANIFEST}" ] || {
    echo "no ${MANIFEST} — run build_multihead_manifest.py first" >&2; exit 1; }

export MJ_NOCLAMP_MANIFEST="plots/metrics/noclamp_manifest.json:${MANIFEST}"
export TAG="${TAG:-_multi}"
export TSFM_FAMILIES=""
export JOBS="${JOBS:-6}"
export STAGGER="${STAGGER:-90}"

# STRAIGHT OFF THE MANIFEST, not off the panel: a month with no checkpoint
# would otherwise cost a full stage-1 collection to score nothing.
export MONTHS="$(uv run python -c "
import json
print(' '.join(sorted(r['eval_month'] for r in
      json.load(open('${MANIFEST}'))['ckpts'])))")"
export MODELS="sup_multi_w8"
export SERIES="sup_multi_w8"

echo "=== TAG=${TAG}  JOBS=${JOBS}"
echo "=== $(echo ${MONTHS} | wc -w) months: ${MONTHS}"
exec bash plots/latent_eval/run_eval.sh
