#!/bin/bash
# setup.sh — Idempotent one-time setup for Pythia cluster access.
#
# Requires: common.sh sourced first (provides PYTHIA_HOST, BLL01_IP, etc.)
#
# Usage:
#   source "${PYTHIA_DIR}/lib/common.sh"
#   source "${PYTHIA_DIR}/lib/setup.sh"
#   pythia_setup

pythia_setup() {
    info "Checking one-time setup..."

    # 1. Authorize pythia's SSH key on bll01 (so compute nodes can rsync back)
    local pythia_pubkey
    pythia_pubkey=$(ssh "${PYTHIA_HOST}" "cat ~/.ssh/id_ecdsa_booth.pub" 2>/dev/null) \
        || error "Cannot read pythia SSH public key. Is ~/.ssh/id_ecdsa_booth.pub present?"

    if ! grep -qF "${pythia_pubkey}" ~/.ssh/authorized_keys 2>/dev/null; then
        info "Adding pythia's SSH key to bll01 authorized_keys..."
        echo "${pythia_pubkey}" >> ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys
    else
        info "Pythia SSH key already authorized on bll01."
    fi

    # 2. SSH config on pythia so compute nodes can reach bll01
    if ! ssh "${PYTHIA_HOST}" "grep -q 'Host bll01' ~/.ssh/config 2>/dev/null"; then
        info "Creating SSH config entry for bll01 on pythia..."
        ssh "${PYTHIA_HOST}" "cat >> ~/.ssh/config && chmod 600 ~/.ssh/config" <<EOF

Host bll01
  HostName ${BLL01_IP}
  User ${BLL01_USER}
  IdentityFile ~/.ssh/id_ecdsa_booth
  StrictHostKeyChecking no
EOF
    else
        info "SSH config for bll01 already exists on pythia."
    fi

    # 3. Install uv on pythia if missing
    if ! ssh "${PYTHIA_HOST}" "test -f ~/.local/bin/uv"; then
        info "Installing uv on pythia..."
        ssh "${PYTHIA_HOST}" "curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        info "uv already installed on pythia."
    fi

    # 4. WANDB credentials on pythia
    if ! ssh "${PYTHIA_HOST}" "grep -q 'api.wandb.ai' ~/.netrc 2>/dev/null"; then
        info "Setting up wandb credentials on pythia..."
        local wandb_key
        wandb_key=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | head -1 | awk '{print $2}')
        [ -n "${wandb_key}" ] || error "No wandb API key found in local ~/.netrc"
        ssh "${PYTHIA_HOST}" "cat >> ~/.netrc && chmod 600 ~/.netrc" <<EOF
machine api.wandb.ai
  login user
  password ${wandb_key}
EOF
    else
        info "Wandb credentials already on pythia."
    fi
}
