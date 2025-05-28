import torch
from numpy.typing import NDArray
from torch import Tensor


def get_batch(dataset: NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    max_start_idx = dataset.shape[0] - context_length
    input_start_idx = torch.randint(low=0, high=max_start_idx, size=(batch_size,))

    input_indices = input_start_idx.unsqueeze(1) + torch.arange(context_length)
    output_indices = input_indices + 1

    return (
        torch.from_numpy(dataset[input_indices]).to(device),
        torch.from_numpy(dataset[output_indices]).to(device),
    )
