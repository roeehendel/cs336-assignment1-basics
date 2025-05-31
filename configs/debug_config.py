"""Tiny model configuration for fast experimentation."""

from configs.tinystories_data_config import TINYSTORIES_DATA_CONFIG
from cs336_basics.models.transformer_lm_config import TransformerLMConfig
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
        context_length=32,
        total_tokens=32 * 8 * 1000,
        batch_size=8,
        device_batch_size=8,
        # device="cpu",
        device="mps",
        # single_batch_for_debug=True,
    ),
    validation=ValidationConfig(
        every_n_steps=200,
        iterations=4,
        batch_size=8,
    ),
    model=TransformerLMConfig(
        vocab_size=10_000,
        # vocab_size=32_000,
        context_length=32,
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        rope_theta=10000,
    ),
    optimizer=AdamWConfig(
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ),
    lr_scheduler=CosineAnnealingConfig(
        max_lr=0.003,
        min_lr=0.0003,
        warmup_fraction=0.1,
        annealing_fraction=1.0,
    ),
    checkpointing=CheckpointingConfig(
        dir="output/checkpoints/debug",
        every_n_steps=1000,
    ),
    logging=LoggingConfig(
        wandb=True,
    ),
)
