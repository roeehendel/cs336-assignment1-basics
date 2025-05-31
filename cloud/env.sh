#!/usr/bin/env bash
export PROJECT_ID=$(gcloud config get-value project)
export ZONE="me-west1-b"          # Tel-Aviv
export REGION="me-west1"

export DATA_DISK="data-disk"
export DATA_DISK_SIZE="200GB"
export BOOT_DISK_SIZE="50GB"

export SHUTDOWN_HOURS=8           # auto-halt time