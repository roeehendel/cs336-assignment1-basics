import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            base_lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                first_moment = state.get("first_moment", torch.zeros_like(p))
                second_moment = state.get("second_moment", torch.zeros_like(p))

                grad = p.grad.data

                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad**2

                lr = base_lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data -= lr * first_moment / (torch.sqrt(second_moment) + eps)
                p.data -= (base_lr * weight_decay) * p.data

                state["t"] = t + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

        return loss
