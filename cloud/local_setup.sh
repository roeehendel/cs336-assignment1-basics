#!/bin/bash
# Simple local Google Cloud setup

set -e

echo "🔧 Installing Google Cloud SDK..."

# Install gcloud SDK
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS
  if ! command -v brew >/dev/null; then
    echo "❌ Install Homebrew first: https://brew.sh"
    exit 1
  fi
  brew install google-cloud-sdk
else
  # Linux - use the simple install script
  curl https://sdk.cloud.google.com | bash
  exec -l $SHELL  # Restart shell to pick up gcloud
fi

echo "🚪 Logging in to Google Cloud..."
gcloud auth login

echo "📂 Setting up project..."
echo "Available projects:"
gcloud projects list

read -p "Enter your PROJECT_ID: " PROJECT_ID
gcloud config set project "$PROJECT_ID"
gcloud config set compute/zone "me-west1-b"

echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com

echo "✅ Setup complete! You can now use ./cloud/vm start t4"