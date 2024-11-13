from functools import partial
import os
import re
import requests
import torch
from torch.utils.data import DataLoader
from mugato.data.utils import splits
from mugato.utils import Timesteps, data_home, TransformDataset, generic_collate_fn


def initialize():
    shakespeare_filepath = data_home / "shakespeare.txt"
    if not os.path.exists(shakespeare_filepath):
        os.makedirs(data_home, exist_ok=True)
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(shakespeare_filepath, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

    with open(shakespeare_filepath, "r", encoding="utf-8") as f:
        data = f.read()

    # Split the dataset into each character's lines.
    # Continue taking lines until you have at least 250 words in the sample.
    # Add that sample to the dataset.
    characters_lines = re.split(r"\n\s*\n", data.strip())
    MIN_WORDS_PER_BATCH = 250
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


def tokenize(tokenizer, sample):
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


def create_dataloader(tokenizer, batch_size, split="train"):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=generic_collate_fn)
