"""Script to find maximum batch size using exponential binary search."""

import copy
import sys
from pathlib import Path

import torch

from configs.tinystories_config import config as base_config
from cs336_basics.training.train import train

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def find_max_batch_size(start_batch_size: int = 32):
    """
    Find the maximum batch size using exponential binary search.
    We test batch sizes where batch_size = device_batch_size (no gradient accumulation).

    Args:
        start_batch_size: Initial batch size to start the search

    Returns:
        Maximum batch size that fits in memory
    """
    print(f"Starting batch size search from {start_batch_size}")

    # Phase 1: Exponential search to find upper bound
    current_batch_size = start_batch_size
    last_successful_size = 0

    print("Phase 1: Exponential search for upper bound...")
    while True:
        if try_batch_size(current_batch_size):
            print(f"✓ Batch size {current_batch_size} succeeded")
            last_successful_size = current_batch_size
            current_batch_size *= 2
        else:
            print(f"✗ Batch size {current_batch_size} failed (OOM)")
            break

    # Phase 2: Binary search between last successful and current (failed) size
    low = last_successful_size
    high = current_batch_size

    print(f"Phase 2: Binary search between {low} and {high}...")
    while low < high - 1:
        mid = (low + high) // 2

        if try_batch_size(mid):
            print(f"✓ Batch size {mid} succeeded")
            low = mid
        else:
            print(f"✗ Batch size {mid} failed (OOM)")
            high = mid

    max_batch_size = low
    print(f"\n🎯 Maximum batch size found: {max_batch_size}")

    return max_batch_size


def try_batch_size(batch_size: int) -> bool:
    """
    Try training with a specific batch size for one step.

    Args:
        batch_size: Batch size to test (will be used for both batch_size and device_batch_size)

    Returns:
        True if successful, False if OOM
    """
    try:
        # Create a copy of the base config
        experiment_config = copy.deepcopy(base_config)

        # Modify config for batch size testing
        experiment_config.training.batch_size = batch_size
        experiment_config.training.device_batch_size = batch_size

        # Set to single step by using minimal total_tokens
        experiment_config.training.total_tokens = batch_size * experiment_config.training.context_length

        # Disable logging and checkpointing for speed
        experiment_config.logging.wandb = False
        experiment_config.logging.console = False
        experiment_config.checkpointing.every_n_tokens = None
        experiment_config.validation.every_n_tokens = None

        # Clear GPU cache before each attempt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run training for one step
        train(experiment_config)

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            # Clear GPU cache after OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        else:
            # Re-raise non-OOM errors
            raise e
    except Exception as e:
        print(f"Unexpected error with batch size {batch_size}: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find maximum batch size")
    parser.add_argument("--start-batch-size", type=int, default=32, help="Initial batch size to start search from")

    args = parser.parse_args()

    max_batch_size = find_max_batch_size(start_batch_size=args.start_batch_size)

    print("\nRecommended configuration:")
    print(f"training.batch_size = {max_batch_size}")
    print(f"training.device_batch_size = {max_batch_size}")
