#!/usr/bin/env bash
# Creates the shared data disk

set -e
source "$(dirname "$0")/env.sh"

# ---------- Create data disk (idempotent) ---------------------------
gcloud compute disks create "$DATA_DISK" \
  --size="$DATA_DISK_SIZE" --type=pd-ssd --zone="$ZONE" 2>/dev/null || true