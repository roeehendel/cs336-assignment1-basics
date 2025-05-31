import torch
from jaxtyping import Int
from numpy.typing import NDArray
from torch import Tensor

BatchTokenIds = Int[Tensor, " batch_size context_length"]


def get_batch(
    dataset: NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    single_batch_for_debug: bool = False,
    random_seed: int | None = None,
) -> tuple[BatchTokenIds, BatchTokenIds]:
    if single_batch_for_debug:
        input_start_idx = torch.zeros(batch_size, dtype=torch.long)
    else:
        max_start_idx = dataset.shape[0] - context_length
        if random_seed:
            torch.manual_seed(random_seed)
        input_start_idx = torch.randint(low=0, high=max_start_idx, size=(batch_size,))

    input_indices = input_start_idx.unsqueeze(1) + torch.arange(context_length)
    output_indices = input_indices + 1

    return (
        torch.from_numpy(dataset[input_indices]).to(device=device).long(),
        torch.from_numpy(dataset[output_indices]).to(device=device).long(),
    )
