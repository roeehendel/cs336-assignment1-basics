from pydantic import BaseModel

from cs336_basics.bpe_tokenizer.types import FilePath
from cs336_basics.models.transformer_lm import TransformerLMConfig


class ExperimentConfig(BaseModel):
    training: "TrainingConfig"
    validation: "ValidationConfig"
    model: TransformerLMConfig
    optimizer: "OptimizerConfig"
    checkpointing: "CheckpointingConfig"


class TrainingConfig(BaseModel):
    context_length: int

    num_iterations: int

    batch_size: int

    dataset_path: FilePath

    device: str

    single_batch_for_debug: bool = False


class ValidationConfig(BaseModel):
    dataset_path: FilePath
    run_every_n_train_iterations: int | None = None
    iterations: int
    batch_size: int


class CheckpointingConfig(BaseModel):
    dir: FilePath
    every_n_iterations: int


class OptimizerConfig(BaseModel):
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
