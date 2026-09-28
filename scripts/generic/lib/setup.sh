#!/bin/bash
# setup.sh — Idempotent one-time setup for the generic 8×A100 cluster.
#
# Designed to be machine-agnostic: anyone with SSH access to ${CLUSTER_HOST}
# can submit sweeps. The bll01-side authorize step (adding the cluster's
# pubkey to bll01:~/.ssh/authorized_keys) is run **only when the submitter
# is bll01**. From any other submitter, we just verify the cluster→bll01
# hop already works; if it doesn't, we error with a hint to run it once
# from bll01.
#
# Requires (from common.sh + .env): CLUSTER_HOST, BLL01_USER, BLL01_IP,
#   CLUSTER_DATA, CLUSTER_CKPT, CLUSTER_UV_CACHE.

cluster_setup() {
    info "Checking one-time setup on ${CLUSTER_HOST} ..."

    # 1. Ensure cluster has an ed25519 key it can use to reach bll01.
    if ! ssh "${CLUSTER_HOST}" "test -f ~/.ssh/id_ed25519"; then
        info "Generating ed25519 key on ${CLUSTER_HOST} ..."
        ssh "${CLUSTER_HOST}" "ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -C ${CLUSTER_USER}@${CLUSTER_HOST}"
    fi

    # 2. SSH config on the cluster so compute jobs can `ssh bll01` / rsync.
    #    Per-cluster state, not per-submitter — fine to write from anywhere.
    if ! ssh "${CLUSTER_HOST}" "grep -q 'Host bll01' ~/.ssh/config 2>/dev/null"; then
        info "Adding bll01 SSH alias on ${CLUSTER_HOST} ..."
        ssh "${CLUSTER_HOST}" "cat >> ~/.ssh/config && chmod 600 ~/.ssh/config" <<EOF

Host bll01
  HostName ${BLL01_IP}
  User ${BLL01_USER}
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
EOF
    fi

    # 3. Verify cluster → bll01 SSH works. If yes, we're done with the SSH
    #    plumbing. If no, fix it from bll01 only (other submitters can't).
    if ssh "${CLUSTER_HOST}" "ssh -o BatchMode=yes -o ConnectTimeout=5 bll01 true" 2>/dev/null; then
        info "${CLUSTER_HOST} → bll01 SSH already works."
    else
        warn "${CLUSTER_HOST} cannot SSH to bll01."
        if [ "$(hostname -s)" = "bll01" ]; then
            info "Submitter is bll01 — authorizing ${CLUSTER_HOST} pubkey on bll01 ..."
            local cluster_pubkey
            cluster_pubkey=$(ssh "${CLUSTER_HOST}" "cat ~/.ssh/id_ed25519.pub") \
                || error "Cannot read ${CLUSTER_HOST}:~/.ssh/id_ed25519.pub"
            if ! grep -qF "${cluster_pubkey}" ~/.ssh/authorized_keys 2>/dev/null; then
                echo "${cluster_pubkey}" >> ~/.ssh/authorized_keys
                chmod 600 ~/.ssh/authorized_keys
                info "${CLUSTER_HOST} pubkey added to bll01 authorized_keys."
            fi
            ssh "${CLUSTER_HOST}" "ssh -o BatchMode=yes -o ConnectTimeout=5 bll01 true" \
                || error "Even after adding the pubkey, ${CLUSTER_HOST} → bll01 still fails. Check Tailscale / BLL01_IP."
            info "${CLUSTER_HOST} → bll01 SSH now works."
        else
            error "Cluster→bll01 SSH not yet authorized. Run \`cluster_setup\` once from bll01 (it's idempotent)."
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
