#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

gcloud compute instances create cs336-dev \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --maintenance-policy=TERMINATE --preemptible \
  "${BOOT_FLAGS[@]}"

gcloud compute config-ssh --project="$(gcloud config get-value project)"