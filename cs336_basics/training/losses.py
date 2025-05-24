import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... logits"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    logits = logits - logits.max(dim=-1, keepdim=True)[0]  # for numerical stability

    target_logits = torch.gather(input=logits, dim=-1, index=targets.unsqueeze(-1)).squeeze()

    normalizer = torch.log(logits.exp().sum(dim=-1))

    loss = -(target_logits - normalizer).mean()

    return loss
