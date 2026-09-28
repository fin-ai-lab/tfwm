#!/bin/bash
# sync.sh — Rsync repo code to a cluster (pythia or a generic cloud box).
#
# Requires: common.sh sourced first (provides LOCAL_REPO, and for the
# pythia_sync wrapper PYTHIA_HOST / PYTHIA_REPO).
#
# Usage:
#   source "${PYTHIA_DIR}/lib/common.sh"
#   source "${PYTHIA_DIR}/lib/sync.sh"
#   pythia_sync                      # pythia tree
#   repo_sync "${HOST}" "${DEST}"    # any other cluster (scripts/generic/)

repo_sync() {
    local host="$1" dest="$2"
    info "Syncing repo code to ${host}:${dest} ..."

    # plots/ is 2.5 GB, but 2.1 GB of that is .pkl caches and the rest is
    # figures and result jsons — none of which a compute node wants. It used to
    # be excluded WHOLESALE, which silently broke any job whose entry point
    # lives there: on 2026-08-20 all 32 latent-sweep jobs died on
    # "No module named 'industry_nn_sweep'" because the entire pipeline is
    # under plots/. Take the SOURCE and leave the artifacts. The manifest is
    # 277 KB and is read lazily by fixed_panel_metrics for checkpoint series.
    # data/ IS SHIPPED, and used to be excluded. It held a local dataset when
    # that rule was written; the stable-finance migration moved the reference
    # tables into it -- market_holidays*.csv, industry_map.parquet,
    # char_table, sup_alpha_table -- which are exactly what a compute node
    # needs. Excluding them left pythia with no data/ at all, and every job
    # died in MarketSchedule with FileNotFoundError AFTER the GPU had been
    # allocated and the month staged. It is 6.6 MB.
    rsync -az --delete \
        --include='plots/' \
        --include='plots/**/' \
        --include='plots/*.py' \
        --include='plots/**/*.py' \
        --include='plots/metrics/noclamp_manifest.json' \
        --exclude='plots/**' \
        --exclude='wandb/' \
        --exclude='checkpoints/' \
        --exclude='multirun/' \
        --exclude='outputs/' \
        --exclude='.git/' \
        --exclude='temp_probe_eval/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.venv/' \
        --exclude='ijepa/' \
        --exclude='slurm-*.out' \
        "${LOCAL_REPO}/" "${host}:${dest}/"

    info "Repo synced."
}

pythia_sync() {
    repo_sync "${PYTHIA_HOST}" "${PYTHIA_REPO}"
}
