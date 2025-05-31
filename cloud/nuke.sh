#!/usr/bin/env bash
source "$(dirname "$0")/stop.sh"
source "$(dirname "$0")/env.sh"
gcloud compute instances delete cs336-dev cs336-train --zone="$ZONE" --quiet || true
gcloud compute disks delete "$DATA_DISK" --zone="$ZONE" --quiet || true