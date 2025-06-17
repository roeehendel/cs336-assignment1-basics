#!/bin/bash

# GitHub SSH Setup Script for CS336 VMs
# Run this locally to set up SSH access to GitHub on your VMs

set -euo pipefail

# Configuration
SSH_KEY_NAME="cs336-github"
SSH_KEY_PATH="$HOME/.ssh/$SSH_KEY_NAME"
GITHUB_USERNAME="roeehendel"  # Your GitHub username
PROJECT_ID=$(gcloud config get-value project)
ZONE="me-west1-b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  generate    Generate SSH key locally (run once)"
    echo "  deploy      Deploy SSH key to all running VMs"
    echo "  setup       Generate key + deploy to VMs (full setup)"
    echo "  status      Check SSH key status on VMs"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Full setup (generate + deploy)"
    echo "  $0 deploy    # Deploy existing key to VMs"
}

generate_ssh_key() {
    if [[ -f "$SSH_KEY_PATH" ]]; then
        warn "SSH key already exists at $SSH_KEY_PATH"
        read -p "Overwrite existing key? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Using existing SSH key"
            return 0
        fi
        rm -f "$SSH_KEY_PATH" "${SSH_KEY_PATH}.pub"
    fi

    log "Generating new SSH key for GitHub access..."
    ssh-keygen -t ed25519 -C "${GITHUB_USERNAME}@cs336-vms" -f "$SSH_KEY_PATH" -N ""
    
    log "SSH key generated successfully!"
    echo ""
    echo "📋 Add this public key to your GitHub account:"
    echo "   https://github.com/settings/ssh/new"
    echo ""
    cat "${SSH_KEY_PATH}.pub"
    echo ""
    read -p "Press Enter after adding the key to GitHub..."
}

get_running_vms() {
    gcloud compute instances list \
        --filter="name~'^cs336-' AND status=RUNNING" \
        --format="value(name)" \
        --zones="$ZONE" 2>/dev/null || echo ""
}

deploy_key_to_vm() {
    local vm_name="$1"
    
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        error "SSH key not found at $SSH_KEY_PATH"
        echo "Run '$0 generate' first to create the key"
        return 1
    fi
    
    log "Deploying SSH key to $vm_name..."
    
    # Copy private key
    gcloud compute scp "$SSH_KEY_PATH" "$vm_name:~/.ssh/id_ed25519" --zone="$ZONE" --quiet
    
    # Copy public key
    gcloud compute scp "${SSH_KEY_PATH}.pub" "$vm_name:~/.ssh/id_ed25519.pub" --zone="$ZONE" --quiet
    
    # Set proper permissions and add GitHub to known_hosts
    gcloud compute ssh "$vm_name" --zone="$ZONE" --command="
        chmod 600 ~/.ssh/id_ed25519
        chmod 644 ~/.ssh/id_ed25519.pub
        ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
        echo 'SSH key deployed and GitHub added to known_hosts'
    " --quiet
    
    log "✅ SSH key deployed to $vm_name"
}

deploy_to_all_vms() {
    local running_vms
    running_vms=$(get_running_vms)
    
    if [[ -z "$running_vms" ]]; then
        warn "No running VMs found"
        echo "Start a VM first with: ./vm start <type>"
        return 1
    fi
    
    log "Found running VMs: $running_vms"
    
    for vm in $running_vms; do
        deploy_key_to_vm "$vm"
    done
    
    log "🎉 SSH keys deployed to all running VMs!"
    echo ""
    echo "Now you can clone/push to GitHub using SSH on your VMs:"
    echo "  git clone git@github.com:${GITHUB_USERNAME}/assignment1-basics.git"
}

check_vm_ssh_status() {
    local vm_name="$1"
    
    log "Checking SSH setup on $vm_name..."
    
    gcloud compute ssh "$vm_name" --zone="$ZONE" --command="
        echo 'Checking SSH key files:'
        ls -la ~/.ssh/id_ed25519* 2>/dev/null || echo 'No SSH keys found'
        echo ''
        echo 'Testing GitHub SSH connection:'
        ssh -T git@github.com 2>&1 || true
    " --quiet
}

check_all_vm_status() {
    local running_vms
    running_vms=$(get_running_vms)
    
    if [[ -z "$running_vms" ]]; then
        warn "No running VMs found"
        return 1
    fi
    
    for vm in $running_vms; do
        check_vm_ssh_status "$vm"
        echo "----------------------------------------"
    done
}

main() {
    local command="${1:-}"
    
    if [[ -z "$command" ]]; then
        usage
        exit 1
    fi
    
    case "$command" in
        "generate")
            generate_ssh_key
            ;;
        "deploy")
            deploy_to_all_vms
            ;;
        "setup")
            generate_ssh_key
            deploy_to_all_vms
            ;;
        "status")
            check_all_vm_status
            ;;
        *)
            error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@" 