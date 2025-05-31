#!/usr/bin/env bash
# -------------------------------------------------------------------
# Common flags for any VM (dev or train) when using Google's DLVM.
# -------------------------------------------------------------------

source "$(dirname "$0")/env.sh"

BOOT_FLAGS=(
  # 1. Use the PyTorch DLVM (GPU-ready) as the base image:
  --image-family=pytorch-latest-gpu
  --image-project=deeplearning-platform-release

  # 2. Attach/format/mount the data disk via our setup script:
  --metadata-from-file=startup-script=cloud/vm_setup.sh
  --metadata=SHUTDOWN_MINUTES=$((SHUTDOWN_HOURS * 60))

  # 3. Standard VM boot-disk + persistent-disk configuration:
  --boot-disk-size="$BOOT_DISK_SIZE"
  --boot-disk-type=pd-ssd
  --disk=name="$DATA_DISK",mode=rw,device-name=data

  --tags=ssh
  --zone="$ZONE"
)