from pydantic import BaseModel, Field, computed_field

from cs336_basics.bpe_tokenizer.types import FilePath
from cs336_basics.models.transformer_lm import TransformerLMConfig


class DataConfig(BaseModel):
    train_path: FilePath
    valid_path: FilePath
    tokenizer_path: FilePath
    end_of_text_token: str


class TrainingConfig(BaseModel):
    seed: int | None = None

    context_length: int

    total_tokens: int

    early_stop_fraction: float | None = None

    batch_size: int
    device_batch_size: int

    device: str

    single_batch_for_debug: bool = False

    @computed_field
    @property
    def num_steps(self) -> int:
        total_steps = self.total_tokens // (self.batch_size * self.context_length)
        if self.early_stop_fraction is not None:
            return int(self.early_stop_fraction * total_steps)
        return total_steps


class ValidationConfig(BaseModel):
    every_n_tokens: int | None = None
    iterations: int
    batch_size: int


class CheckpointingConfig(BaseModel):
    dir: FilePath
    every_n_tokens: int | None = None


class AdamWConfig(BaseModel):
    betas: tuple[float, float]
    eps: float
    weight_decay: float


class CosineAnnealingConfig(BaseModel):
    max_lr: float
    min_lr: float
    warmup_fraction: float
    annealing_fraction: float


class LoggingConfig(BaseModel):
    wandb: bool = False
    console: bool = True


class ExperimentConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model: TransformerLMConfig = Field(default_factory=TransformerLMConfig)
    optimizer: AdamWConfig
    lr_scheduler: CosineAnnealingConfig
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
