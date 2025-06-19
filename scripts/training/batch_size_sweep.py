"""Simple batch size sweep script."""

import copy
import math
import sys
from pathlib import Path

import wandb
from configs.tinystories_config import config as base_config
from cs336_basics.training.train import train

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def create_batch_size_sweep_config():
    """Create a simple batch size sweep configuration."""
    sweep_config = {
        "method": "grid",
        "metric": {"name": "validation/loss", "goal": "minimize"},
        "parameters": {
            "batch_size": {"values": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        },
    }
    return sweep_config


def train_batch_size_sweep():
    """Training function for batch size sweep."""
    wandb.init()
    sweep_config = wandb.config

    experiment_config = copy.deepcopy(base_config)

    lr_scaling_factor = math.sqrt(sweep_config.batch_size / experiment_config.training.batch_size)

    # Update batch size
    experiment_config.training.batch_size = sweep_config.batch_size
    experiment_config.training.device_batch_size = min(64, sweep_config.batch_size)

    # set learning rate based on batch size
    experiment_config.lr_scheduler.max_lr = experiment_config.lr_scheduler.max_lr * lr_scaling_factor
    experiment_config.lr_scheduler.min_lr = experiment_config.lr_scheduler.min_lr * lr_scaling_factor

    # early stop
    experiment_config.training.early_stop_fraction = 0.02

    # Ensure wandb logging
    experiment_config.logging.wandb = True

    # Unique checkpoint dir
    run_id = wandb.run.id
    experiment_config.checkpointing.dir = f"output/checkpoints/lr_sweep_{run_id}"

    train(experiment_config)


if __name__ == "__main__":
    sweep_config = create_batch_size_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="cs336-assignment-1")
    print(f"Batch size sweep ID: {sweep_id}")
    wandb.agent(sweep_id, train_batch_size_sweep, count=len(sweep_config["parameters"]["batch_size"]["values"]))
