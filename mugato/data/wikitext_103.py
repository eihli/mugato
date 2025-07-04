import random
from functools import partial
from typing import Any

import torch
from datasets import load_dataset  # type: ignore
from torch.utils.data import DataLoader

from mugato.data.utils import infinite_dataloader
from mugato.utils import Timesteps, TransformDataset, generic_collate_fn


def initialize() -> dict[str, Any]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return {
        "train": dataset["train"],
        "val": dataset["validation"],
        "test": dataset["test"],
    }


def tokenize(
    tokenizer: Any, sample: Any, block_size: int = 1024
) -> tuple[Timesteps, Timesteps]:
    eot = torch.tensor([[tokenizer.eot_token_id]], dtype=torch.long)
    text = sample["text"]
    if len(text) > block_size:
        start = random.randint(0, len(text) - (block_size + 1))
        end = min(start + block_size, len(text))
    else:
        start = 0
        end = len(text)
    text = text[start:end]
    tokens = tokenizer.encode_text(text)
    if start == 0 and len(tokens) < block_size:
        tokens = torch.cat([eot, tokens])
    if end >= len(text) and len(tokens) < block_size:
        tokens = torch.cat([tokens, eot])
    tokens = torch.stack([tokens])
    xs = Timesteps(
        {
            "text": tokens[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "text": tokens[:, 1:],
        }
    )
    return xs, ys


def create_dataloader(
    tokenizer: Any, batch_size: int, split: str = "train", block_size: int = 1024
) -> DataLoader[Any]:
    datasets = initialize()
    transform_dataset = TransformDataset(datasets[split], partial(tokenize, tokenizer))
    return DataLoader(
        transform_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            generic_collate_fn, sequence_length=block_size, mask_keys=["text"]
        ),
    )


def create_infinite_dataloader(
    tokenizer: Any, batch_size: int, split: str = "train", block_size: int = 1024
) -> Any:
    datasets = initialize()
    transform_dataset = TransformDataset(datasets[split], partial(tokenize, tokenizer))
    return infinite_dataloader(
        partial(
            DataLoader,
            transform_dataset,
            batch_size=batch_size,
            collate_fn=partial(
                generic_collate_fn,
                sequence_length=block_size,
                mask_keys=["text"]
            )
        )
    )
