"""Simple learning rate sweep script."""

import copy
import sys
from pathlib import Path

import wandb
from configs.tinystories_config import config as base_config
from cs336_basics.training.train import train

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def create_lr_sweep_config():
    """Create a simple learning rate sweep configuration focused on max_lr."""
    sweep_config = {
        "method": "grid",
        "metric": {"name": "validation/loss", "goal": "minimize"},
        "parameters": {
            "max_lr": {"values": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]},
            # We'll set min_lr = 0.01 * max_lr programmatically
        },
    }
    return sweep_config


def train_lr_sweep():
    """Training function for learning rate sweep."""
    wandb.init()
    sweep_config = wandb.config

    experiment_config = copy.deepcopy(base_config)

    # Update learning rate parameters
    max_lr = sweep_config.max_lr
    min_lr = 0.01 * max_lr  # Set min_lr as 1% of max_lr

    experiment_config.lr_scheduler.max_lr = max_lr
    experiment_config.lr_scheduler.min_lr = min_lr

    # early stop
    experiment_config.training.early_stop_fraction = 0.2

    # Log the computed min_lr for visibility
    wandb.log({"computed_min_lr": min_lr}, step=0)

    # Ensure wandb logging
    experiment_config.logging.wandb = True

    # Unique checkpoint dir
    run_id = wandb.run.id
    experiment_config.checkpointing.dir = f"output/checkpoints/lr_sweep_{run_id}"

    train(experiment_config)


if __name__ == "__main__":
    sweep_config = create_lr_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="cs336-assignment-1")
    print(f"Learning rate sweep ID: {sweep_id}")
    print("Sweeping max_lr with min_lr = 0.1 * max_lr")
    wandb.agent(sweep_id, train_lr_sweep, count=7)  # Updated count to match number of max_lr values
