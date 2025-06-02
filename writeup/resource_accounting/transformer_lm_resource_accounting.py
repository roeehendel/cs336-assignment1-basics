from cs336_basics.models.transformer_lm_config import TransformerLMConfig


def count_transformer_lm_params(cfg: TransformerLMConfig):
    token_embeddings = cfg.vocab_size * cfg.d_model
    glu = 3 * cfg.d_model * cfg.d_ff
    qkvo_proj = 4 * cfg.d_model * cfg.d_model
    lm_head = cfg.vocab_size * cfg.d_model

    total = token_embeddings + cfg.num_layers * (glu + qkvo_proj) + lm_head

    return {
        "token_embeddings": token_embeddings,
        "glu": glu,
        "qkvo_proj": qkvo_proj,
        "lm_head": lm_head,
        "non_embeddings": total - token_embeddings,
        "total": total,
    }


def count_transformer_lm_activations_per_batch_element(cfg: TransformerLMConfig) -> int:
    layer_rms_norm = cfg.context_length * cfg.d_model

    attn_kqv = cfg.context_length * 3 * cfg.d_model
    attn_qk = cfg.num_heads * cfg.context_length**2
    attn_softmax = attn_qk
    attn_values_weighted_sum = cfg.context_length * cfg.d_model
    attn_output_projection = cfg.context_length * cfg.d_model
    attn = attn_kqv + attn_qk + attn_softmax + attn_values_weighted_sum + attn_output_projection

    ffn_w1_mm = cfg.context_length * cfg.d_ff
    ffn_silu = ffn_w1_mm
    ffn_w3_mm = ffn_w1_mm
    ffn_w2_mm = cfg.context_length * cfg.d_model
    ffn = ffn_w1_mm + ffn_silu + ffn_w3_mm + ffn_w2_mm

    layer = 2 * layer_rms_norm + attn + ffn

    final_rms_norm = cfg.context_length * cfg.d_model

    output_embedding = cfg.context_length * cfg.vocab_size  # (logits)

    total = cfg.num_layers * layer + final_rms_norm + output_embedding

    return {
        "layer": layer,
        "layers": cfg.num_layers * layer,
        "final_rms_norm": final_rms_norm,
        "output_embedding": output_embedding,
        "total": total,
    }


def count_lm_transformer_adamw_memory_usage(cfg: TransformerLMConfig, batch_size: int, bytes_per_float: int = 4) -> int:
    params = count_transformer_lm_params(cfg)["total"] * bytes_per_float
    activations = count_transformer_lm_activations_per_batch_element(cfg)["total"] * bytes_per_float
    gradients = params
    optimizer_state = params * 2

    return {
        "params": params,
        "activations": activations * batch_size,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "activations_per_batch_element": activations,
        "non_activations": params + gradients + optimizer_state,
        "total": params + activations + gradients + optimizer_state,
    }


def calc_transformer_lm_flops(cfg: TransformerLMConfig) -> dict[str, int]:
    rope_group_size = 2
    d_head = cfg.d_model // cfg.num_heads

    glu_flops = 2 * matmul_flops(cfg.context_length, cfg.d_model, cfg.d_ff) + matmul_flops(
        cfg.context_length, cfg.d_ff, cfg.d_model
    )

    qkv_proj_flops = 3 * matmul_flops(cfg.context_length, cfg.d_model, cfg.d_model)
    rope_flops = 2 * matmul_flops(
        cfg.context_length * cfg.num_heads, rope_group_size, rope_group_size
    )  # 2 * for queries and keys
    attn_flops = matmul_flops(cfg.num_heads * cfg.context_length, d_head, cfg.context_length) + matmul_flops(
        cfg.num_heads * cfg.context_length, cfg.context_length, d_head
    )
    output_proj_flops = matmul_flops(cfg.context_length, cfg.d_model, cfg.d_model)
    mha_flops = qkv_proj_flops + rope_flops + attn_flops + output_proj_flops

    layer_flops = mha_flops + glu_flops

    lm_head_flops = matmul_flops(cfg.context_length, cfg.d_model, cfg.vocab_size)

    total_flops = cfg.num_layers * layer_flops + lm_head_flops

    return {
        "total": total_flops,
        "glu": glu_flops * cfg.num_layers,
        "mha": {
            "total": mha_flops * cfg.num_layers,
            "qkv_proj": qkv_proj_flops * cfg.num_layers,
            "rope": rope_flops * cfg.num_layers,
            "attn": attn_flops * cfg.num_layers,
            "output_proj": output_proj_flops * cfg.num_layers,
        },
        "lm_head": lm_head_flops,
    }


def matmul_flops(m: int, n: int, p: int) -> int:
    return 2 * m * n * p


def print_teraflops(name: str, num: int, total: int, indent: int):
    TERA = 10**12
    print(" " * indent + f"- {name}={num / TERA:.2f} ({num / total:.2%})")


def print_flops_dict(d: dict):
    total = d["total"]
    _print_dict(d, total)


def _print_dict(d: dict, total: int, indent: int = 0):
    for k, v in d.items():
        if k == "total":
            continue
        if isinstance(v, dict):
            print_teraflops(name=k, num=v["total"], total=total, indent=indent)
            _print_dict(v, total=total, indent=indent + 1)
        else:
            print_teraflops(name=k, num=v, total=total, indent=indent)


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if num_bytes < 0:
        raise ValueError("Number of bytes cannot be negative")

    if num_bytes == 0:
        return "0 B"

    units = [
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024, "KB"),
    ]

    for threshold, unit in units:
        if num_bytes >= threshold:
            value = num_bytes / threshold
            return f"{value:.2f} {unit}" if value != int(value) else f"{int(value)} {unit}"

    return f"{num_bytes} B"


def format_params_count(num_params: int) -> str:
    """Format parameter count into human-readable string."""
    if num_params < 0:
        raise ValueError("Number of parameters cannot be negative")

    if num_params == 0:
        return "0"

    units = [
        (1000**4, "T"),
        (1000**3, "B"),
        (1000**2, "M"),
        (1000, "K"),
    ]

    for threshold, unit in units:
        if num_params >= threshold:
            value = num_params / threshold
            return f"{value:.2f}{unit}" if value != int(value) else f"{int(value)}{unit}"

    return str(num_params)


shared = {
    "vocab_size": 50_257,
    "context_length": 1_024,
}

GPT_2_CONFIGS: dict[str, TransformerLMConfig] = {
    "small": TransformerLMConfig(
        **shared,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=4 * 768,
    ),
    "medium": TransformerLMConfig(
        **shared,
        num_layers=24,
        d_model=1_024,
        num_heads=16,
        d_ff=4 * 1024,
    ),
    "large": TransformerLMConfig(
        **shared,
        num_layers=36,
        d_model=1_280,
        num_heads=20,
        d_ff=4 * 1280,
    ),
    "xl": TransformerLMConfig(
        **shared,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=4 * 1600,
    ),
}
