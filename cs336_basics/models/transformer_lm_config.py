from pydantic import BaseModel, Field


class TransformerLMConfig(BaseModel):
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = Field(default=10000)
