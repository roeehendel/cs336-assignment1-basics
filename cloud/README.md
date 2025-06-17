# Cloud VM Management

This directory contains scripts for managing Google Cloud VMs with shared storage for your CS336 assignment. The setup allows you to easily switch between T4 and A100 GPUs while maintaining persistent data.

## 🔐 Setup

**One-time setup required before using VMs:**

1. **Setup Google Cloud authentication**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

**Optional: Setup budget alerts** (recommended):
   ```bash
   ./cloud/create_budget_alert.sh
   ```



2. **Start your first VM**:
   ```bash
   ./cloud/vm start t4
   ```

## 🚀 Quick Start

The `vm` script is your single interface for all VM operations:

```bash
# Start a T4 VM for development
./cloud/vm start t4

# Switch to A100 for training (automatically stops T4)
./cloud/vm switch a100

# Check status of all VMs
./cloud/vm status

# SSH into running VM
./cloud/vm ssh a100

# Stop a VM
./cloud/vm stop a100

# Clean up (stop/delete all VMs)
./cloud/vm cleanup
```

## 🖥️ VM Types

| Type | Machine | GPU | Use Case |
|------|---------|-----|----------|
| `t4` | n1-standard-4 | NVIDIA T4 | Development, inference, debugging |
| `a100` | a2-highgpu-1g | NVIDIA A100 | Large-scale training |
| `cpu` | n1-standard-4 | None | Testing, data preprocessing |

## 📁 Shared Storage

All VMs share the same boot disk (`cs336-shared-boot`), which means:
- **Your data persists** when switching between VMs
- **Code, models, datasets** are accessible from any VM type
- **Docker images and containers** persist across VM switches

Your home directory on the shared boot disk persists across VM switches

### Environment Setup

**Option 1: One-Time Auto Setup (Recommended)**
```bash
# First time: installs everything automatically (runs once)
./cloud/vm start t4

# Wait ~3-5 minutes for one-time setup, then SSH in
./cloud/vm ssh t4

# Your project is ready at: ~/assignment1-basics

# Future starts are instant (setup already done)
./cloud/vm stop t4
./cloud/vm start t4  # <- Fast! No reinstallation
```

**Option 2: Docker Workflow**
```bash
# VMs boot with Docker + GPU drivers pre-installed
./cloud/vm start t4

# Build and run your containerized environment
docker build -t cs336 .
docker run -it --gpus all -v $(pwd):/workspace cs336
```

## 📋 Commands

### Basic Operations
- `./cloud/vm start <type>` - Start a VM (creates if doesn't exist)
- `./cloud/vm stop <type>` - Stop a running VM
- `./cloud/vm switch <type>` - Switch to different VM type
- `./cloud/vm status` - Show status of all VMs and shared storage
- `./cloud/vm ssh <type>` - SSH into a running VM
- `./cloud/vm cleanup` - Interactive cleanup (stop/delete VMs)



## 🔧 Configuration

Configuration is at the top of the `vm` script:
- **Project**: Automatically detected from `gcloud config`
- **Zone**: `me-west1-b` (Tel Aviv)
- **Shared Disk**: `cs336-shared-boot` (400GB)
- **Auto-shutdown**: 8 hours

## 💡 Tips

- **Preemptible VMs**: All VMs are preemptible to save costs
- **Auto-shutdown**: VMs automatically shutdown after 8 hours
- **Shared State**: All your work persists when switching VM types
- **SSH Configuration**: Automatically updated when starting VMs

## 🧹 Cleanup

The `cleanup` command helps you manage costs:
1. **Stop VMs**: Keeps instances but stops billing for compute
2. **Delete VMs**: Removes instances entirely (shared disk remains)
3. **Manual cleanup**: Use Google Cloud Console for shared disk deletion

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   T4 Instance   │    │  A100 Instance  │    │  CPU Instance   │
│   (cs336-t4)    │    │  (cs336-a100)   │    │  (cs336-cpu)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │   Shared Boot Disk        │
                   │   (cs336-shared-boot)     │
                   │   • 400GB SSD             │
                   │   • Persistent storage    │
                   │   • PyTorch DLVM image    │
                   └───────────────────────────┘
```

Only one VM can use the shared disk at a time, ensuring data consistency and cost efficiency.

## 📁 File Structure

The cloud directory contains only essential files:
- `vm` - Main VM management script (handles everything!)
- `vm_setup.sh` - One-time environment setup script
- `create_budget_alert.sh` - Budget monitoring setup
- `local_setup.sh` - Local development environment setup