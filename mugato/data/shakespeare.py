import os
import re
from functools import partial
from typing import Any

import requests  # type: ignore
import torch
from torch.utils.data import DataLoader

from mugato.data.utils import infinite_dataloader, splits
from mugato.utils import Timesteps, TransformDataset, data_home, generic_collate_fn


def initialize() -> dict[str, list[str]]:
    shakespeare_filepath = data_home / "shakespeare.txt"
    if not os.path.exists(shakespeare_filepath):
        os.makedirs(data_home, exist_ok=True)
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(shakespeare_filepath, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

    with open(shakespeare_filepath, encoding="utf-8") as f:
        data = f.read()

    # Split the dataset into each character's lines.
    # Continue taking lines until you have at least 150 words in the sample.
    # Add that sample to the dataset. I chose 150 because empirically it
    # seems like that's less than the 1024 tokens I was using for my
    # initial experiments on small hardware.
    characters_lines = re.split(r"\n\s*\n", data.strip())
    MIN_WORDS_PER_BATCH = 50
    sample = [characters_lines[0]]
    num_words_in_sample = len(characters_lines[0].split())
    text_dataset = []
    i = 1
    while i < len(characters_lines):
        if num_words_in_sample > MIN_WORDS_PER_BATCH:
            text_dataset.append("\n\n".join(sample))
            num_words_in_sample -= len(sample[0].split())
            sample = sample[1:]
        sample += [characters_lines[i]]
        num_words_in_sample += len(characters_lines[i].split())
        i += 1

    train_data, val_data, test_data = splits(text_dataset)
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def tokenize(tokenizer: Any, sample: str) -> tuple[Timesteps, Timesteps]:
    eot = torch.tensor([[tokenizer.eot_token_id]], dtype=torch.long)
    text = torch.stack([torch.concat([eot, tokenizer.encode_text(sample), eot])])
    xs = Timesteps(
        {
            "text": text[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "text": text[:, 1:],
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
