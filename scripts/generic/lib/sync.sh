#!/bin/bash
# sync.sh — Rsync repo code to the cloud cluster. The rsync itself (and its
# hard-won include/exclude list) is repo_sync() in pythia/lib/sync.sh, shared
# so the two trees can't drift on what a compute node needs.
#
# Requires: common.sh sourced first (provides LOCAL_REPO, CLUSTER_HOST,
# CLUSTER_REPO, CLUSTER_DIR).

source "${CLUSTER_DIR}/../pythia/lib/sync.sh"

cluster_sync() {
    repo_sync "${CLUSTER_HOST}" "${CLUSTER_REPO}"
}
