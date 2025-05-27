import torch
from torch import nn


def gradient_clipping(parameters: list[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    params_with_grad = [param for param in parameters if param.grad is not None]

    with torch.no_grad():
        grad_l2_norm = torch.sqrt(sum(torch.norm(param.grad, p=2) ** 2 for param in params_with_grad))

        if grad_l2_norm > max_l2_norm:
            for param in params_with_grad:
                param.grad *= max_l2_norm / (grad_l2_norm + eps)
