#!/bin/bash
# setup.sh — Idempotent one-time setup for a generic single-node cluster.
#
# Designed to be machine-agnostic: anyone with SSH access to ${CLUSTER_HOST}
# can submit sweeps. The data-host-side authorize step (adding the cluster's
# pubkey to the data host's ~/.ssh/authorized_keys) runs **only when the
# submitter is the data host itself** (its short hostname equals
# DATA_HOST_NAME). From any other submitter, we just verify the
# cluster→data host hop already works; if it doesn't, we error with a hint
# to run it once from the data host.
#
# Requires (from common.sh + .env): CLUSTER_HOST, DATA_HOST_USER, DATA_HOST_IP,
#   CLUSTER_DATA, CLUSTER_CKPT, CLUSTER_UV_CACHE. Optional: DATA_HOST_NAME.

cluster_setup() {
    info "Checking one-time setup on ${CLUSTER_HOST} ..."

    # 1. Ensure cluster has an ed25519 key it can use to reach the data host.
    if ! ssh "${CLUSTER_HOST}" "test -f ~/.ssh/id_ed25519"; then
        info "Generating ed25519 key on ${CLUSTER_HOST} ..."
        ssh "${CLUSTER_HOST}" "ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -C ${CLUSTER_USER}@${CLUSTER_HOST}"
    fi

    # 2. SSH config on the cluster so compute jobs can `ssh datahost` / rsync.
    #    Per-cluster state, not per-submitter — fine to write from anywhere.
    if ! ssh "${CLUSTER_HOST}" "grep -q 'Host datahost' ~/.ssh/config 2>/dev/null"; then
        info "Adding datahost SSH alias on ${CLUSTER_HOST} ..."
        ssh "${CLUSTER_HOST}" "cat >> ~/.ssh/config && chmod 600 ~/.ssh/config" <<EOF

Host datahost
  HostName ${DATA_HOST_IP}
  User ${DATA_HOST_USER}
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
EOF
    fi

    # 3. Verify cluster → data host SSH works. If yes, we're done with the SSH
    #    plumbing. If no, fix it from the data host only (others can't).
    if ssh "${CLUSTER_HOST}" "ssh -o BatchMode=yes -o ConnectTimeout=5 datahost true" 2>/dev/null; then
        info "${CLUSTER_HOST} → data host SSH already works."
    else
        warn "${CLUSTER_HOST} cannot SSH to the data host."
        if [ -n "${DATA_HOST_NAME:-}" ] && [ "$(hostname -s)" = "${DATA_HOST_NAME}" ]; then
            info "Submitter is the data host — authorizing ${CLUSTER_HOST} pubkey here ..."
            local cluster_pubkey
            cluster_pubkey=$(ssh "${CLUSTER_HOST}" "cat ~/.ssh/id_ed25519.pub") \
                || error "Cannot read ${CLUSTER_HOST}:~/.ssh/id_ed25519.pub"
            if ! grep -qF "${cluster_pubkey}" ~/.ssh/authorized_keys 2>/dev/null; then
                echo "${cluster_pubkey}" >> ~/.ssh/authorized_keys
                chmod 600 ~/.ssh/authorized_keys
                info "${CLUSTER_HOST} pubkey added to the data host's authorized_keys."
            fi
            ssh "${CLUSTER_HOST}" "ssh -o BatchMode=yes -o ConnectTimeout=5 datahost true" \
                || error "Even after adding the pubkey, ${CLUSTER_HOST} → data host still fails. Check DATA_HOST_IP is reachable from the cluster."
            info "${CLUSTER_HOST} → data host SSH now works."
        else
            error "Cluster→data host SSH not yet authorized. Run \`cluster_setup\` once from the data host, with DATA_HOST_NAME set to its short hostname (it's idempotent)."
        fi
    fi

    # 4. uv
    if ! ssh "${CLUSTER_HOST}" "test -f ~/.local/bin/uv"; then
        info "Installing uv on ${CLUSTER_HOST} ..."
        ssh "${CLUSTER_HOST}" "curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        info "uv already installed on ${CLUSTER_HOST}."
    fi

    # 5. Wandb credentials. Per-cluster state — first submitter wins. If
    #    you want your own wandb account, edit ~/.netrc on the cluster
    #    directly. Skip silently if the local submitter has no key.
    if ! ssh "${CLUSTER_HOST}" "grep -q 'api.wandb.ai' ~/.netrc 2>/dev/null"; then
        local wandb_key
        wandb_key=$(grep -A2 'api.wandb.ai' ~/.netrc 2>/dev/null | grep password | head -1 | awk '{print $2}')
        if [ -n "${wandb_key}" ]; then
            info "Pushing local wandb credentials to ${CLUSTER_HOST} ..."
            ssh "${CLUSTER_HOST}" "cat >> ~/.netrc && chmod 600 ~/.netrc" <<EOF
machine api.wandb.ai
  login user
  password ${wandb_key}
EOF
        else
            warn "No wandb key in local ~/.netrc and ${CLUSTER_HOST} has none either — wandb runs will fail. Add credentials to ${CLUSTER_HOST}:~/.netrc manually."
        fi
    else
        info "Wandb credentials already present on ${CLUSTER_HOST}."
    fi

    # 6. Working dirs under $HOME
    ssh "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_DATA} ${CLUSTER_CKPT} ${CLUSTER_UV_CACHE}"
}
