# cloud/common.sh
source "$(dirname "$0")/env.sh"

BOOT_FLAGS=(
  --boot-disk-size="$BOOT_DISK_SIZE"
  --image-family=ubuntu-2204-lts
  --image-project=ubuntu-os-cloud        # <-- add this line
  --metadata="install-nvidia-driver=True,startup-script=sudo shutdown -h +$((SHUTDOWN_HOURS*60))"
  --disk=name="$DATA_DISK",mode=rw,device-name=data
  --tags=ssh
  --zone="$ZONE"
)