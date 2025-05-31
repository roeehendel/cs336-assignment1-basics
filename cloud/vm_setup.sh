#!/usr/bin/env bash
set -eux

# -------------------------------------------------------------
# 1. Auto‐shutdown after N minutes
# -------------------------------------------------------------
SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-480}"
sudo shutdown -h +"$SHUTDOWN_MINUTES" &

# -------------------------------------------------------------
# 2. Mount & format /dev/sdb → /mnt/data, then chown it
# -------------------------------------------------------------
DEVICE=/dev/sdb
MOUNT=/mnt/data

if ! mount | grep -q "$MOUNT"; then
  sudo mkdir -p "$MOUNT"
  if ! sudo blkid "$DEVICE" >/dev/null 2>&1; then
    sudo mkfs.ext4 -m 0 -F "$DEVICE"
  fi
  sudo mount "$DEVICE" "$MOUNT"
fi

# Make /mnt/data owned by your SSH user
sudo chown -R "$(whoami)":"$(whoami)" "$MOUNT"

# Setup persistent SSH
PERSISTENT_SSH="/mnt/data/.ssh"
mkdir -p "$PERSISTENT_SSH"
chmod 700 "$PERSISTENT_SSH"
ln -sf "$PERSISTENT_SSH" "$HOME/.ssh"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup dotfiles
PERSISTENT_DOTFILES="/mnt/data/dotfiles"
if [ ! -d "$PERSISTENT_DOTFILES/.git" ]; then
  git clone git@github.com:roeehendel/dotfiles.git "$PERSISTENT_DOTFILES"
fi

if [ -d "$PERSISTENT_DOTFILES" ]; then
  ln -sf "$PERSISTENT_DOTFILES" "$HOME/dotfiles"
  cd "$HOME/dotfiles" && ./.scripts/install.sh
fi

echo "✅ VM setup complete: /mnt/data mounted, auto‐shutdown in $SHUTDOWN_MINUTES minutes."