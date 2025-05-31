# CS336 – One-repo Cloud GPU Workflow

## One-time setup (local machine)

Run:

    ./bootstrap_local.sh          # installs gcloud, chooses project & zone, adds budget alert
    ./cloud/init.sh               # creates shared data disk, opens GPU-quota request page

Then request **1 T4** and **1 A100-40GB** quota in your zone (`me-west1-b` recommended).  
Wait for the approval email (usually under 24 hours).

---

## Daily workflow

    # 🛠️  interactive development
    ./cloud/dev_up.sh             # start T4 spot VM
    code .                        # VS Code Remote-SSH → “Reopen in Container”
    ./cloud/stop.sh               # stop VM when done

    # 🚂  heavy training
    ./cloud/train_up.sh           # start A100 spot VM
    python train.py …             # checkpoints live in /mnt/data
    ./cloud/stop.sh

---

## What lives where

| Path / File                | Purpose |
|---------------------------|---------|
| `cloud/env.sh`            | all tunables (zone, disk sizes, shutdown hours) |
| `cloud/*.sh`              | scripted VM lifecycle (create / stop / delete) |
| `docker/Dockerfile`       | builds CUDA + Python image; installs `uv` and project deps |
| `pyproject.toml`, `uv.lock` | define exact Python environment via `uv` |
| `.devcontainer.json`      | tells VS Code to use that same Docker image on remote |
| `bootstrap_local.sh`      | local gcloud install + login + billing budget alert |

---

## Docker, uv, and dev-container: how they work together

- **Docker** ensures every VM has the same base OS + CUDA + Python tooling.
- **uv** (inside Docker) reads `pyproject.toml` + `uv.lock` to install the exact Python packages.
- **.devcontainer.json** tells VS Code: when I SSH to this box, reopen the repo *inside the Docker container*, so all terminals and kernels see the right env and the GPU.

Build & run the Docker image (once per image update) inside the VM:

    sudo docker build -t cs336 docker/
    sudo docker run -it --gpus all -v /mnt/data:/workspace cs336

All state (code, env, checkpoints) lives on `/mnt/data`, the shared persistent disk.  
If a spot VM is preempted, you just rerun `*_up.sh` and continue.

---

## Optional: Budget alert (auto-handled by `bootstrap_local.sh`)

This will email you when you’ve used 250 USD of your $300 free credit.

    gcloud billing budgets create \
      --billing-account=YOUR_ACCOUNT_ID \
      --display-name="trial-credit-guard" \
      --budget-amount=250USD \
      --threshold-rule="percent=1"

To find your billing account ID:

    gcloud beta billing projects describe YOUR_PROJECT_ID --format='value(billingAccountName)'