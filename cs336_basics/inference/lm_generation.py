import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from cs336_basics.models.transformer_lm import TransformerLM, softmax


def lm_generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    end_of_text_token: str,
    max_tokens: int,
    temperature: float,
    device: str,
    prompt: str | None = None,
    top_p: float | None = None,
) -> str:
    # TODO: enable batch generation

    if prompt:
        prompt_token_ids = tokenizer.encode(prompt)
    else:
        prompt_token_ids = tokenizer.encode(end_of_text_token)

    generated_token_ids = []

    model = model.to(device)

    while len(generated_token_ids) < max_tokens:
        context_token_ids = prompt_token_ids + generated_token_ids
        model_input = rearrange(
            torch.tensor(context_token_ids, dtype=torch.long, device=device),
            "seq_len -> 1 seq_len",
        )

        # TODO: implement kv cache
        logits = model.forward(token_ids=model_input)
        last_token_logits = logits[0, -1, :]
        if temperature > 0:
            probs = softmax(last_token_logits / temperature, dim=-1)
            if top_p:
                probs = top_p_filtering(probs, top_p)
            pred_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            new_token_id = pred_token_ids.item()
        else:
            new_token_id = last_token_logits.argmax(dim=-1).item()

        generated_token_ids.append(new_token_id)

    if end_of_text_token in generated_token_ids:
        generated_token_ids = generated_token_ids[: generated_token_ids.index(end_of_text_token)]

    return tokenizer.decode(generated_token_ids)


def top_p_filtering(probs: Float[Tensor, " vocab_size"], top_p: float) -> Float[Tensor, " vocab_size"]:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: keep tokens until cumulative probability exceeds top_p
    # But always keep at least the first token
    cutoff = torch.searchsorted(cumulative_probs, top_p, right=True) + 1
    cutoff = torch.clamp(cutoff, min=1)  # Ensure we keep at least one token

    # Zero out probabilities beyond cutoff
    sorted_probs[cutoff:] = 0.0

    # Create output tensor and scatter the filtered probabilities back to original positions
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(0, sorted_indices, sorted_probs)

    # Renormalize
    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
