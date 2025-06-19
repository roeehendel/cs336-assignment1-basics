"""Configuration for training a TinyStories model."""

from configs.tinystories_data_config import TINYSTORIES_DATA_CONFIG
from cs336_basics.models.transformer_lm_config import TransformerLMConfig
from cs336_basics.training.device_utils import find_best_device
from cs336_basics.training.train_config import (
    AdamWConfig,
    CheckpointingConfig,
    CosineAnnealingConfig,
    ExperimentConfig,
    LoggingConfig,
    TrainingConfig,
    ValidationConfig,
)

config = ExperimentConfig(
    data=TINYSTORIES_DATA_CONFIG,
    training=TrainingConfig(
        seed=42,
        context_length=256,
        total_tokens=327_680_000,
        # total_tokens=40_000_000,
        # early_stop_fraction=0.2,
        batch_size=32,
        device_batch_size=32,
        # device="cpu",
        device=find_best_device(),
        # single_batch_for_debug=True,
    ),
    validation=ValidationConfig(
        every_n_tokens=200 * 8192,
        iterations=16,
        batch_size=32,
    ),
    model=TransformerLMConfig(
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
    ),
    optimizer=AdamWConfig(
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ),
    lr_scheduler=CosineAnnealingConfig(
        max_lr=2e-3,
        min_lr=2e-5,
        warmup_fraction=0.05,
        annealing_fraction=1.0,
    ),
    checkpointing=CheckpointingConfig(
        dir="output/checkpoints/tinystories",
        every_n_tokens=1000 * 8192,
    ),
    logging=LoggingConfig(
        wandb=True,
        console=True,
    ),
)
