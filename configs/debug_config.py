"""Tiny model configuration for fast experimentation."""

from cs336_basics.models.transformer_lm_config import TransformerLMConfig
from cs336_basics.training.train_config import (
    AdamWConfig,
    CheckpointingConfig,
    CosineAnnealingConfig,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    OptimizationConfig,
    TrainingConfig,
    ValidationConfig,
)
from scripts.data.paths import TS_TOKENZIER_DIR

config = ExperimentConfig(
    data=DataConfig(
        train_path="output/token_ids/TinyStoriesV2-GPT4-train.txt.npy",
        valid_path="output/token_ids/TinyStoriesV2-GPT4-valid.txt.npy",
        tokenizer_path=TS_TOKENZIER_DIR,
        end_of_text_token="<|endoftext|>",
    ),
    training=TrainingConfig(
        context_length=32,
        num_steps=10_000,
        batch_size=8,
        # device="cpu",
        device="mps",
        single_batch_for_debug=False,
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
    optimization=OptimizationConfig(
        optimizer=AdamWConfig(
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        ),
        scheduler=CosineAnnealingConfig(
            max_lr=0.003,
            min_lr=0.0003,
            warmup_fraction=0.1,
            annealing_fraction=1.0,
        ),
    ),
    checkpointing=CheckpointingConfig(
        dir="output/checkpoints/debug",
        every_n_steps=1000,
    ),
    logging=LoggingConfig(
        wandb=True,
    ),
)
