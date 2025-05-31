#!/usr/bin/env bash
# Creates the shared data disk and reminds you to request GPU quota.

set -e
source "$(dirname "$0")/../env.sh"

# ---------- Create data disk (idempotent) ---------------------------
gcloud compute disks create "$DATA_DISK" \
  --size="$DATA_DISK_SIZE" --type=pd-ssd --zone="$ZONE" 2>/dev/null || true

# ---------- Open quota request page --------------------------------
echo -e "\n🚀  Opening browser to request 1 T4 and 1 A100 in zone $ZONE."
open "https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}&service=compute.googleapis.com&metrics=GPUs%20(Tesla)" \
     || xdg-open "https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}&service=compute.googleapis.com&metrics=GPUs%20(Tesla)" \
     || true

read -p $'\nPress Enter after you have filed the quota request…'

echo "✅  Cloud bootstrap done."