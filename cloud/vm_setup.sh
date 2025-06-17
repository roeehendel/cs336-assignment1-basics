#!/bin/bash

# CS336 VM One-Time Setup Script
# Only runs on the first boot of a new disk/VM

set -e

SETUP_MARKER="/var/lib/cs336-setup-complete"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check if setup has already been completed
if [[ -f "$SETUP_MARKER" ]]; then
    log "CS336 setup already completed, skipping..."
    exit 0
fi

log "Starting CS336 one-time VM setup..."

# Auto-shutdown after 8 hours to save costs (runs every boot)
SHUTDOWN_MINUTES=${SHUTDOWN_MINUTES:-480}
log "Setting up auto-shutdown in $SHUTDOWN_MINUTES minutes"
echo "sudo shutdown -h +$SHUTDOWN_MINUTES" | sudo at now 2>/dev/null || true

# Install uv package manager
log "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Install useful tools
log "Installing useful tools..."
sudo apt-get update -qq
sudo apt-get install -y htop tmux zsh tree jq

# Clone project repository
PROJECT_DIR="/home/$(whoami)/assignment1-basics"
if [[ ! -d "$PROJECT_DIR" ]]; then
    log "Cloning project repository..."
    
    # Try SSH first if key exists, fallback to HTTPS
    if [[ -f ~/.ssh/id_ed25519 ]]; then
        REPO_URL="git@github.com:roeehendel/assignment1-basics.git"
        log "Using SSH clone (key found)"
    else
        REPO_URL="https://github.com/roeehendel/assignment1-basics.git"
        log "Using HTTPS clone (no SSH key found)"
    fi
    
    git clone "$REPO_URL" "$PROJECT_DIR" || {
        log "Failed to clone $REPO_URL"
        mkdir -p "$PROJECT_DIR"
    }
fi

# Install project dependencies if pyproject.toml exists
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
    log "Installing project dependencies..."
    cd "$PROJECT_DIR"
    export PATH="$HOME/.local/bin:$PATH"
    uv pip install --system -e .
    log "Dependencies installed successfully!"
else
    log "pyproject.toml not found, skipping dependency installation"
    log "After cloning your repo, install with: cd ~/assignment1-basics && uv pip install -e ."
fi

# Mark setup as complete
log "Creating setup completion marker..."
sudo touch "$SETUP_MARKER"

log "CS336 setup complete! Your environment is ready."
log "Project location: $PROJECT_DIR"
log "This setup will not run again on future boots." 