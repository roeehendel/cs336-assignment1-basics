import importlib
import os
import sys
from functools import partial

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

import wandb
from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from cs336_basics.inference.lm_generation import lm_generate
from cs336_basics.models.transformer_lm import TransformerLM
from cs336_basics.training.checkpointing import save_checkpoint
from cs336_basics.training.dataloading import get_batch
from cs336_basics.training.losses import cross_entropy
from cs336_basics.training.lr_schedulers import lr_cosine_schedule
from cs336_basics.training.optimizers.adamw import AdamW
from cs336_basics.training.train_config import ExperimentConfig
from scripts.bpe_tokenizer.tokenizer_training_utils import load_bpe_tokenizer


def train(cfg: ExperimentConfig):
    # TODO: make logging more modular
    _init_logging(cfg)

    model = TransformerLM(cfg=cfg.model)
    optimizer = AdamW(model.parameters(), **cfg.optimization.optimizer.model_dump())
    lr_scheduler = partial(
        lr_cosine_schedule, **cfg.optimization.scheduler.model_dump(), num_steps=cfg.training.num_steps
    )

    model = model.to(cfg.training.device)

    train_dataset = np.load(cfg.data.train_path, mmap_mode="r")
    validation_dataset = np.load(cfg.data.valid_path, mmap_mode="r")

    tokenizer = load_bpe_tokenizer(cfg.data.tokenizer_path)

    for step in range(cfg.training.num_steps):
        optimizer.zero_grad()

        lr = lr_scheduler(step=step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_token_ids, output_token_ids = get_batch(
            dataset=train_dataset,
            batch_size=cfg.training.batch_size,
            context_length=cfg.training.context_length,
            device=cfg.training.device,
            single_batch_for_debug=cfg.training.single_batch_for_debug,
        )

        logits = model.forward(token_ids=input_token_ids)

        loss = cross_entropy(logits=logits, targets=output_token_ids)
        perplexity = torch.exp(loss)

        loss.backward()
        optimizer.step()

        _write_log(
            cfg=cfg,
            step=step,
            data={
                "train/loss": loss.item(),
                "train/perplexity": perplexity.item(),
                "train/lr": lr,
            },
        )

        if cfg.checkpointing.every_n_steps and step % cfg.checkpointing.every_n_steps == 0:
            os.makedirs(cfg.checkpointing.dir, exist_ok=True)
            checkpoint_path = os.path.join(cfg.checkpointing.dir, f"checkpoint_{step}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_path,
            )

        if cfg.validation.every_n_steps and step % cfg.validation.every_n_steps == 0:
            _validate(model=model, dataset=validation_dataset, cfg=cfg, step=step)
            # TODO: consider separating generation from validation
            _generate(cfg=cfg, step=step, model=model, tokenizer=tokenizer, input_token_ids=input_token_ids)


def _init_logging(cfg: ExperimentConfig):
    if cfg.logging.wandb:
        wandb.login()
        wandb.init(
            project="cs336-assignment-1",
            config=cfg.model_dump(),
        )


def _write_log(cfg: ExperimentConfig, step: int, data: dict):
    if cfg.logging.wandb:
        wandb.log(data, step=step)
    if cfg.logging.console:
        print(f"Step {step} {data}")


def _generate(
    cfg: ExperimentConfig,
    step: int,
    model: TransformerLM,
    tokenizer: BPETokenizer,
    input_token_ids: Int[Tensor, " batch_size context_length"],
):
    prompt = tokenizer.decode(input_token_ids[0].tolist()[:1]) if cfg.training.single_batch_for_debug else None
    generated_text = lm_generate(
        model=model,
        tokenizer=tokenizer,
        end_of_text_token=cfg.data.end_of_text_token,
        prompt=prompt,
        max_tokens=cfg.training.context_length,
        temperature=0.1,
        top_p=0.1,
        device=cfg.training.device,
    )
    _write_log(cfg=cfg, step=step, data={"train/generated_text": generated_text})


def _validate(model: TransformerLM, dataset: np.ndarray, cfg: ExperimentConfig, step: int):
    losses = []
    for iteration in range(cfg.validation.iterations):
        input_token_ids, output_token_ids = get_batch(
            dataset=dataset,
            batch_size=cfg.validation.batch_size,
            context_length=cfg.training.context_length,
            device=cfg.training.device,
        )

        logits = model.forward(token_ids=input_token_ids)

        loss = cross_entropy(logits=logits, targets=output_token_ids)
        losses.append(loss.item())

    loss = torch.tensor(losses).mean()
    perplexity = torch.exp(loss)

    _write_log(cfg=cfg, step=step, data={"validation/loss": loss.item(), "validation/perplexity": perplexity.item()})


if __name__ == "__main__":
    from jsonargparse import CLI

    # Check if user wants to use a Python config module
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        # Assume it's a Python module (e.g., "configs.base_config")
        config_module = importlib.import_module(sys.argv[1])
        train(config_module.config)
    else:
        # Use jsonargparse for CLI/YAML configs
        CLI(train)
