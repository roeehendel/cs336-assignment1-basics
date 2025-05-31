#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
for vm in cs336-dev cs336-train; do
  gcloud compute instances stop "$vm" --zone="$ZONE" --quiet || true
done