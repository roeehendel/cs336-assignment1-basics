"""Tiny model configuration for fast experimentation."""

from cs336_basics.models.transformer_lm_config import TransformerLMConfig
from cs336_basics.training.train_config import (
    CheckpointingConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainingConfig,
    ValidationConfig,
)

config = ExperimentConfig(
    training=TrainingConfig(
        context_length=32,
        num_iterations=200,
        dataset_path="output/token_ids/TinyStoriesV2-GPT4-train.txt.npy",
        # dataset_path="output/token_ids/owt_train.txt.npy",
        batch_size=4,
        # device="cpu",
        device="mps",
        # single_batch_for_debug=True,
    ),
    validation=ValidationConfig(
        dataset_path="output/token_ids/TinyStoriesV2-GPT4-valid.txt.npy",
        # dataset_path="output/token_ids/owt_valid.txt.npy",
        run_every_n_train_iterations=20,
        # run_every_n_train_iterations=None,
        iterations=4,
        batch_size=8,
    ),
    model=TransformerLMConfig(
        vocab_size=10_000,
        # vocab_size=32_000,
        context_length=256,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000,
    ),
    optimizer=OptimizerConfig(
        lr=0.03,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    ),
    checkpointing=CheckpointingConfig(
        dir="output/checkpoints/debug",
        every_n_iterations=10,
    ),
)
