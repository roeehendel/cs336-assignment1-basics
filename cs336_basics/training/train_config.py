from pydantic import BaseModel, Field

from cs336_basics.bpe_tokenizer.types import FilePath
from cs336_basics.models.transformer_lm import TransformerLMConfig


class DataConfig(BaseModel):
    train_path: FilePath
    valid_path: FilePath
    tokenizer_path: FilePath
    end_of_text_token: str


class TrainingConfig(BaseModel):
    context_length: int

    num_steps: int

    batch_size: int

    device: str

    single_batch_for_debug: bool = False


class ValidationConfig(BaseModel):
    every_n_steps: int | None = None
    iterations: int
    batch_size: int


class CheckpointingConfig(BaseModel):
    dir: FilePath
    every_n_steps: int | None = None


class AdamWConfig(BaseModel):
    betas: tuple[float, float]
    eps: float
    weight_decay: float


class CosineAnnealingConfig(BaseModel):
    max_lr: float
    min_lr: float
    warmup_fraction: float
    annealing_fraction: float


class OptimizationConfig(BaseModel):
    optimizer: AdamWConfig
    scheduler: CosineAnnealingConfig


class LoggingConfig(BaseModel):
    wandb: bool = False
    console: bool = True


class ExperimentConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model: TransformerLMConfig = Field(default_factory=TransformerLMConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
