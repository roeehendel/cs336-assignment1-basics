#!/usr/bin/env bash
set -e
source "$(dirname "$0")/common.sh"

gcloud compute instances create cs336-train \
  --machine-type=a2-highgpu-1g \
  --accelerator=count=1,type=nvidia-tesla-a100-40gb \
  --preemptible \
  "${BOOT_FLAGS[@]}"