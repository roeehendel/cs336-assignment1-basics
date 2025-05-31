import numpy as np
from tqdm import tqdm

from cs336_basics.models.transformer_lm import TransformerLM
from cs336_basics.training.dataloading import get_batch
from cs336_basics.training.losses import cross_entropy
from cs336_basics.training.optimizers.adamw import AdamW
from cs336_basics.training.train_config import ExperimentConfig


def run_lr_finding(
    cfg: ExperimentConfig,
    num_steps: int = 100,
    start_lr: float = 1e-6,
    stop_lr: float = 1e0,
    warmup_lr: float = 1e-3,
    warmup_steps: int = 10,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    model = TransformerLM(cfg.model)
    model.to(cfg.training.device)

    dataset = np.load(cfg.data.train_path, mmap_mode="r")

    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())

    lrs = np.geomspace(start=start_lr, stop=stop_lr, num=num_steps)

    losses = []

    def step(lr: float) -> float:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_token_ids, target_token_ids = get_batch(
            dataset,
            batch_size=batch_size,
            context_length=cfg.training.context_length,
            device=cfg.training.device,
            single_batch_for_debug=cfg.training.single_batch_for_debug,
        )
        logits = model.forward(token_ids=input_token_ids)
        loss = cross_entropy(logits=logits, targets=target_token_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    for i in range(warmup_steps):
        step(lr=i / warmup_steps * warmup_lr)

    for lr in tqdm(lrs):
        loss = step(lr=lr)
        losses.append(loss)

    return lrs, losses
