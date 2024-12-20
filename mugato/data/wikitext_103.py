import random
from datasets import load_dataset

from functools import partial
import torch
from torch.utils.data import DataLoader
from mugato.data.utils import splits
from mugato.utils import Timesteps, data_home, TransformDataset, generic_collate_fn


def initialize():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return {
        "train": dataset["train"],
        "val": dataset["validation"],
        "test": dataset["test"],
    }


def tokenize(tokenizer, sample, block_size=1024):
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


def create_dataloader(tokenizer, batch_size, split="train"):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=generic_collate_fn)
