#!/usr/bin/env bash
#
# Installs Google Cloud SDK, logs you in, selects project,
# and sets a default zone.
#
# Supports macOS (Homebrew) and Debian/Ubuntu (apt).
# Other distros can just copy the gcloud install line they like.

set -euo pipefail

# -------- Detect OS and install SDK ------------------------------------------------
if [[ "$(uname)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null; then
    echo "❌ Homebrew not found. Install it first: https://brew.sh" ; exit 1
  fi
  brew install --quiet google-cloud-sdk
else
  # Debian/Ubuntu
  if ! command -v apt-get >/dev/null; then
    echo "❌ Unsupported OS. Install gcloud manually: https://cloud.google.com/sdk/docs/install" ; exit 1
  fi
  sudo apt-get update -y
  sudo apt-get install -y apt-transport-https ca-certificates gnupg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo tee /usr/share/keyrings/cloud.google.gpg >/dev/null
  sudo apt-get update -y && sudo apt-get install -y google-cloud-sdk
fi

# -------- Login / project / zone ---------------------------------------------------
echo -e "\n🚪  Opening browser for Google login…"
gcloud auth login --brief

echo -e "\n📂  Listing projects (pick the one with the $300 credit):"
gcloud projects list

read -rp $'\n🔤  Enter your PROJECT_ID: ' PROJECT_ID
gcloud config set project "$PROJECT_ID"

read -rp $'\n🌍  Default zone (e.g. me-west1-b): ' ZONE
gcloud config set compute/zone "$ZONE"

echo -e "\n🔧  Enabling Compute Engine API…"
gcloud services enable compute.googleapis.com

echo -e "\n✅  Local bootstrap complete."