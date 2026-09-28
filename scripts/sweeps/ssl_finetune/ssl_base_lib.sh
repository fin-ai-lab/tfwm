#!/bin/bash
# ssl_base_lib.sh — staging for the SSL finetune sweep.
#
# Sourced by ssl_finetune_breadth.sh. Defines:
#   SSL_BASE_DIR       where a month's base checkpoint lands on /hpc_temp
#   SSL_HEAD_DIR       where its ridge head init lands
#   ssl_base_arm_tag   the arm name, read from the manifest header
#   ssl_base_pool_rows the a36 row count the ridge was fit on, for a month
#   sweep_stage_extra  per-month staging hook, called by
#                      scripts/pythia/lib/train_bundle_body.sh on the compute
#                      node after the regular data staging
#
# Regenerate the manifest with make_ssl_base_manifest.py alongside it.
#
# THE ARM IS THE MANIFEST'S, NOT THE SWEEP FILE'S. One manifest, one base arm:
# regenerating it for a different series silently repoints every sweep that
# reads it, and if the sweep NAME were hardcoded that rename would not follow —
# runs land in the project of an arm they are no longer initialized from, with
# nothing on disk to say otherwise. So the name is DERIVED from the header.

# KEYED BY RUN ID, NOT BY MONTH, AND NOT THE OLD PATH.
#
# This cost a whole wave on 2026-09-16. The first version staged to
# .../market-jepa-ssl-base/<eval_month>/ and skipped the rsync when a model.pt
# was already there. That is the path the RETIRED 2026-08 ssl_finetune sweep
# used, with the same month keys and a different generation of base
# checkpoints -- so on any node that had run the old sweep, "already staged"
# silently handed this one a 2026-08 model. Jobs 230860 and 230868 died on
# `Unexpected key(s): cls_token, recency_slope_raw`, which is the GOOD case:
# that generation is architecturally incompatible and strict loading caught
# it. A stale checkpoint of the CURRENT architecture but the old ONE-MONTH
# training span would have loaded without complaint and trained the wrong
# model, and nothing downstream would have said so.
#
# A run id names exactly one checkpoint, so "already staged" can now only mean
# the right one. The directory name also no longer collides with the old
# sweep's tree at all, and /hpc_temp evicts the orphans by modtime.
SSL_BASE_DIR="${SSL_BASE_DIR:-/hpc_temp/${USER}/market-jepa-ft-base-by-run}"
SSL_HEAD_DIR="${SSL_HEAD_DIR:-/hpc_temp/${USER}/market-jepa-ft-head-by-run}"
SSL_BASE_MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ssl_base_manifest.tsv"

ssl_base_arm_tag() {
    local SERIES
    SERIES=$(sed -n 's/^# BASE_SERIES[[:space:]]*//p' "${SSL_BASE_MANIFEST}")
    if [ -z "${SERIES}" ]; then
        echo "ERROR: ${SSL_BASE_MANIFEST} carries no '# BASE_SERIES' header." \
             "Regenerate it with make_ssl_base_manifest.py." >&2
        return 1
    fi
    # pair_warp_6mo -> pair-warp
    SERIES="${SERIES%_6mo}"
    echo "${SERIES//_/-}"
}

# The a36 row count the month's ridge was fit on — the denominator that turns a
# target row count into dataset.train_data_fraction. See the manifest generator.
ssl_base_pool_rows() {
    awk -F'\t' -v ym="$1" '$1==ym {print $4}' "${SSL_BASE_MANIFEST}"
}

ssl_base_months() {
    awk -F'\t' '!/^#/ && NF {print $1}' "${SSL_BASE_MANIFEST}"
}

# Requires (set by train_bundle_body.sh before the call):
#   TRAIN_START, BLL01_HOST, LOCK_DIR
#
# NOTE ON THE MONTH KEY. The manifest is keyed by EVAL month; TRAIN_START is
# six months earlier. The bundle exports EVAL_START alongside it, and that is
# what this looks up — keying off TRAIN_START would silently fetch the
# checkpoint of a span ending six months before the one being trained.
sweep_stage_extra() {
    local YM="${EVAL_START:0:7}"
    local ROW SRC_CKPT SRC_HEAD
    ROW=$(awk -F'\t' -v ym="${YM}" '$1==ym {print; exit}' "${SSL_BASE_MANIFEST}")
    if [ -z "${ROW}" ]; then
        echo "ERROR: eval month ${YM} not in ${SSL_BASE_MANIFEST}" >&2
        return 1
    fi
    SRC_CKPT=$(echo "${ROW}" | cut -f2)
    SRC_HEAD=$(echo "${ROW}" | cut -f3)
    # The run id IS the checkpoint dir's basename, so it cannot disagree with
    # the path it was taken from.
    local RUN_ID
    RUN_ID=$(basename "${SRC_CKPT}")

    local DEST="${SSL_BASE_DIR}/${RUN_ID}" HDEST="${SSL_HEAD_DIR}/${RUN_ID}"
    mkdir -p "${SSL_BASE_DIR}" "${SSL_HEAD_DIR}" "${LOCK_DIR}"
    (
        flock -x 9
        # The stamp is belt and braces on top of the run-id key: it records
        # WHICH source produced this directory, so a reuse is only accepted
        # when that source is the one being asked for now. Anything else --
        # including a directory left by some earlier version of this script --
        # is re-fetched rather than trusted.
        if [ -f "${DEST}/model.pt" ] && [ -f "${DEST}/config.json" ] \
           && [ "$(cat "${DEST}/.source" 2>/dev/null)" = "${SRC_CKPT}" ]; then
            echo "==> ssl-base ${YM} (${RUN_ID}): already staged."
        else
            echo "==> ssl-base ${YM} (${RUN_ID}): rsyncing base checkpoint from ${BLL01_HOST} ..."
            rm -rf "${DEST}"
            mkdir -p "${DEST}"
            # model.pt, NOT backbone.pt — the SSL and LeJEPA arms save the whole
            # model, and load_pretrained_backbone looks for exactly this name.
            rsync -az \
                "${BLL01_HOST}:${SRC_CKPT}/config.json" \
                "${BLL01_HOST}:${SRC_CKPT}/model.pt" \
                "${DEST}/"
            echo "${SRC_CKPT}" > "${DEST}/.source"
        fi
        # A `.source` MATCH IS NOT ENOUGH FOR THE HEAD INIT, because the head
        # inits get REFIT IN PLACE. On 2026-09-17 they were refit from
        # mean-pooled to last-pooled embeddings at the same paths, so every
        # node that had run the earlier wave kept serving the retired weights
        # under a stamp that still matched -- 18 stale directories on 5 nodes.
        # Only the readout guard in eval/heads.py stopped those runs, and it
        # only stops them because the stale artifacts happen to be unstamped;
        # a refit that changed the WEIGHTS but not the readout would have
        # trained silently on the wrong probe.
        #
        # So the reuse is validated on CONTENT. The digest is the manifest's
        # 5th column, over the .npz files in lexicographic order -- the same
        # bytes in the same order as make_ssl_base_manifest.py:head_digest.
        local WANT_DIGEST GOT_DIGEST
        WANT_DIGEST=$(echo "${ROW}" | cut -f5)
        GOT_DIGEST=$(cat "${HDEST}"/return_900.npz "${HDEST}"/spread_change_900.npz \
                         "${HDEST}"/volatility_change_900.npz 2>/dev/null \
                     | sha256sum | cut -c1-16)
        if [ -f "${HDEST}/summary.json" ] \
           && [ "$(cat "${HDEST}/.source" 2>/dev/null)" = "${SRC_HEAD}" ] \
           && [ -n "${WANT_DIGEST}" ] && [ "${GOT_DIGEST}" = "${WANT_DIGEST}" ]; then
            echo "==> ridge-head ${YM} (${RUN_ID}): already staged (${GOT_DIGEST})."
        else
            echo "==> ridge-head ${YM} (${RUN_ID}): rsyncing ridge head init ..."
            rm -rf "${HDEST}"
            mkdir -p "${HDEST}"
            rsync -az "${BLL01_HOST}:${SRC_HEAD}/" "${HDEST}/"
            echo "${SRC_HEAD}" > "${HDEST}/.source"
            GOT_DIGEST=$(cat "${HDEST}"/return_900.npz "${HDEST}"/spread_change_900.npz \
                             "${HDEST}"/volatility_change_900.npz 2>/dev/null \
                         | sha256sum | cut -c1-16)
            # A fetch that does not produce the manifest's bytes means the
            # manifest and /data/lab have diverged, which is worth stopping
            # for rather than training on whichever one arrived.
            if [ -n "${WANT_DIGEST}" ] && [ "${GOT_DIGEST}" != "${WANT_DIGEST}" ]; then
                echo "ERROR: ridge-head ${YM} (${RUN_ID}) fetched digest" \
                     "${GOT_DIGEST}, manifest says ${WANT_DIGEST}." \
                     "Regenerate ssl_base_manifest.tsv." >&2
                return 1
            fi
        fi
        # /hpc_temp evicts by modtime — refresh even when already staged.
        touch "${DEST}" "${DEST}"/* "${HDEST}" "${HDEST}"/*
    ) 9>"${LOCK_DIR}/ssl-base-${RUN_ID}.lock"
}

# The staged path for a month, which the sweep needs to pass to hydra. Derived
# from the manifest the same way sweep_stage_extra derives it, so the two
# cannot drift apart.
ssl_base_run_id() {
    awk -F'\t' -v ym="$1" '$1==ym {n=split($2,a,"/"); print a[n]}' "${SSL_BASE_MANIFEST}"
}

# ── Executed directly: emit the month list the launcher wants ──────────────
#
# run_all_months_sweep.sh iterates TRAIN_END months ("YM"), while the manifest
# is keyed by EVAL month one later. Getting this backwards is silent: every job
# would stage the checkpoint of the span before the one it trains.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    case "${1:-}" in
        --train-months)
            ssl_base_months | while read -r YM; do
                date -d "${YM}-01 -1 month" +%Y-%m
            done | tr '\n' ' '
            echo ;;
        --eval-months) ssl_base_months | tr '\n' ' '; echo ;;
        --arm) ssl_base_arm_tag ;;
        *)
            echo "usage: ssl_base_lib.sh [--train-months|--eval-months|--arm]" >&2
            exit 2 ;;
    esac
fi
