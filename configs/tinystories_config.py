"""Configuration for training a TinyStories model."""

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
        context_length=256,
        # total_tokens=327_680_000,
        total_tokens=40_000_000,
        batch_size=32,
        device_batch_size=8,
        # device="cpu",
        device="mps",
        # single_batch_for_debug=True,
    ),
    validation=ValidationConfig(
        every_n_steps=200,
        iterations=32,
        batch_size=8,
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
        min_lr=5e-5,
        warmup_fraction=0.05,
        annealing_fraction=1.0,
    ),
    checkpointing=CheckpointingConfig(
        dir="output/checkpoints/tinystories",
        every_n_steps=1000,
    ),
    logging=LoggingConfig(
        wandb=True,
        console=True,
    ),
)
