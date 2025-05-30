import torch
from torch import nn, optim
from torch.serialization import FILE_LIKE


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: FILE_LIKE):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj=checkpoint, f=out)


def load_checkpoint(src: FILE_LIKE, model: nn.Module, optimizer: optim.Optimizer) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
