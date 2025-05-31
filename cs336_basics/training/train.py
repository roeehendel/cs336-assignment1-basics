import importlib
import os
import sys

import numpy as np
import torch

import wandb
from cs336_basics.models.transformer_lm import TransformerLM
from cs336_basics.training.checkpointing import save_checkpoint
from cs336_basics.training.dataloading import get_batch
from cs336_basics.training.losses import cross_entropy
from cs336_basics.training.optimizers.adamw import AdamW
from cs336_basics.training.train_config import ExperimentConfig
from scripts.bpe_tokenizer.tinystories.train_bpe_tinystories import load_tinystories_tokenizer


def train(cfg: ExperimentConfig):
    # TODO: create an abstract logger class
    wandb.login()
    run = wandb.init(
        project="cs336-assignment-1",
        config=cfg.model_dump(),
    )

    model = TransformerLM(cfg=cfg.model)
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())

    model = model.to(cfg.training.device)

    train_dataset = np.load(cfg.training.dataset_path, mmap_mode="r")
    validation_dataset = np.load(cfg.validation.dataset_path, mmap_mode="r")

    tokenizer = load_tinystories_tokenizer()

    for iteration in range(cfg.training.num_iterations):
        optimizer.zero_grad()

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

        print(f"Iteration {iteration} loss: {loss.item()} perplexity: {perplexity.item()}")
        wandb.log({"train/loss": loss, "train/perplexity": perplexity})

        if cfg.checkpointing and iteration % cfg.checkpointing.every_n_iterations == 0:
            os.makedirs(cfg.checkpointing.dir, exist_ok=True)
            checkpoint_path = os.path.join(cfg.checkpointing.dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                out=checkpoint_path,
            )

        if cfg.validation.run_every_n_train_iterations and iteration % cfg.validation.run_every_n_train_iterations == 0:
            validate(model=model, dataset=validation_dataset, cfg=cfg)


def validate(model: TransformerLM, dataset: np.ndarray, cfg: ExperimentConfig):
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

    print(f"Validation loss: {loss.item()} perplexity: {perplexity.item()}")
    wandb.log({"validation/loss": loss, "validation/perplexity": perplexity})


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
