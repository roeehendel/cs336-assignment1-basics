#!/usr/bin/env bash
# One-time setup for any new VM: Docker + NVIDIA container support

set -e

echo "🧱 Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

echo "🎮 Installing NVIDIA container toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/libnvidia-container.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

echo "✅ Done. You may need to log out and log back in or run: newgrp docker"