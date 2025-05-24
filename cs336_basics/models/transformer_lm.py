import math
from collections.abc import Sequence
from typing import Union

import torch
from einops import einsum, parse_shape, rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        rope = RoPE(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: Int[Tensor, "..."],
    ) -> Float[Tensor, "... vocab_size"]:
        embeddings = self.token_embeddings(token_ids)
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: Union["RoPE", None] = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            rope=rope,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln1, self.ln2 = (
            RMSNorm(
                d_model=d_model,
                device=device,
                dtype=dtype,
            )
            for _ in range(2)
        )

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... sequence_length d_model"]:
        in_features = in_features + self.attn(self.ln1(in_features))
        in_features = in_features + self.ffn(self.ln2(in_features))
        return in_features


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Union["RoPE", None] = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not d_model % num_heads == 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d = d_model // num_heads

        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

        self.device = device

        self.rope = rope

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... sequence_length d_model"]:
        input_shape = parse_shape(in_features, "batch_size ... sequence_length d_model")
        batch_size = input_shape["batch_size"]
        sequence_length = input_shape["sequence_length"]

        queries = self.q_proj(in_features)
        keys = self.k_proj(in_features)
        values = self.v_proj(in_features)

        queries_per_head = self.rearrange_to_heads(queries)
        keys_per_head = self.rearrange_to_heads(keys)
        values_per_head = self.rearrange_to_heads(values)

        if self.rope:
            # TODO: does repeat actually copy and require more memory? if so - can we avoid this?
            token_positions = repeat(
                torch.arange(0, sequence_length, device=self.device),
                "position -> batch_size position",
                batch_size=batch_size,
            )
            queries_per_head = self.rope.forward(queries_per_head, token_positions=token_positions)
            keys_per_head = self.rope.forward(keys_per_head, token_positions=token_positions)

        mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=self.device))

        attention_output_per_head = scaled_dot_product_attention(queries_per_head, keys_per_head, values_per_head, mask)

        attention_output: Float[Tensor, "... sequence_length d_model"] = self.rearrange_from_heads(
            attention_output_per_head
        )

        return self.output_proj(attention_output)

    def rearrange_to_heads(
        self,
        X: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... num_heads sequence_length d_head"]:
        return rearrange(
            X,
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
        )

    def rearrange_from_heads(self, X: Float[Tensor, "... num_heads sequence_length d_head"]):
        return rearrange(
            X,
            "... num_heads sequence_length d_head -> ... sequence_length (num_heads d_head)",
            num_heads=self.num_heads,
        )


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            data=get_truncated_normal_tensor(
                size=(out_features, in_features),
                std=math.sqrt(2 / (in_features + out_features)),
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[Tensor, " num_embeddings embedding_dim"] = nn.Parameter(
            data=get_truncated_normal_tensor(
                size=(num_embeddings, embedding_dim),
                std=1,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(
            data=get_truncated_normal_tensor(
                size=(d_model,),
                std=1,
                device=device,
                dtype=dtype,
            ),
        )
        self.eps = eps

    def forward(self, x: Float[Tensor, "... dim_model"]) -> Float[Tensor, "... dim_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(swish(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: str | None = None,
    ):
        super().__init__()

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.rotation_group_size = 2

        sequence_position, feature_group_position = torch.meshgrid(
            torch.arange(0, max_seq_len, device=device),
            torch.arange(0, d_k // self.rotation_group_size, device=device),
            indexing="ij",
        )
        angle = sequence_position / (theta ** ((self.rotation_group_size * feature_group_position) / d_k))
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, "... seq_len d_kq"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_kq"]:
        cos = self.get_buffer("cos")
        sin = self.get_buffer("sin")

        all_rotations = torch.stack((torch.stack((cos, -sin), dim=-1), torch.stack((sin, cos), dim=-1)), dim=-2)
        rotations = all_rotations[token_positions]

        in_query_or_key_grouped = rearrange(
            in_query_or_key,
            "... (rotation_groups rotation_group_size) -> ... rotation_groups rotation_group_size",
            rotation_group_size=self.rotation_group_size,
        )
        in_query_or_key_grouped_rotated = einsum(
            rotations, in_query_or_key_grouped, "... group_out group_in, ... group_in -> ... group_out"
        )
        in_query_or_key_rotated = rearrange(
            in_query_or_key_grouped_rotated,
            "... rotation_groups rotation_group_size -> ... (rotation_groups rotation_group_size)",
        )

        return in_query_or_key_rotated


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Float[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    logits: Float[Tensor, "... queries keys"] = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    logits = torch.where(mask, logits, -torch.inf)

    d_k = parse_shape(Q, "... queries d_k")["d_k"]
    attention: Float[Tensor, "... queries keys"] = softmax(logits / math.sqrt(d_k), dim=-1)

    return einsum(attention, V, "... queries keys, ... keys d_v ->... queries d_v")


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x = x - x.max(dim=dim, keepdim=True)[0]  # for numerical stability, subtract the max value along dim
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def swish(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


def get_truncated_normal_tensor(
    size: Sequence[int],
    mean: float = 0.0,
    std: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return nn.init.trunc_normal_(
        torch.empty(size, device=device, dtype=dtype),
        mean=mean,
        std=std,
        a=min_value or -3 * std,
        b=max_value or 3 * std,
    )
