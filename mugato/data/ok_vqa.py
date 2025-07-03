import random
from functools import partial
from typing import Any

import torch
from datasets import load_dataset  # type: ignore
from torch.utils.data import DataLoader

from mugato.data.utils import infinite_dataloader
from mugato.utils import (
    Timesteps,
    TransformDataset,
    as_tensor,
    generic_collate_fn,
    image_transform,
)


def initialize() -> dict[str, Any]:
    dataset = load_dataset("eihli/micro-ok-vqa")
    train_data = dataset["train"]
    # I happen to know this is a dataset of 80 train and 20 val.
    # I'm going to just split the val 50/50 to create a test set.
    val_data = dataset["validation"].select(range(10))
    test_data = dataset["validation"].select(range(10, 20))
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def tokenize(tokenizer: Any, sample: Any) -> tuple[Timesteps, Timesteps]:
    question = [tokenizer.encode_text(sample["question"])]
    image = [tokenizer.encode_image(image_transform(as_tensor(sample["image"])))]
    eot = torch.tensor([[tokenizer.eot_token_id]])
    answer = [
        torch.concat(
            [
                eot,
                tokenizer.encode_text(random.choice(sample["answers"])["answer"]),
                eot,
            ]
        )
    ]
    question_tensor = torch.stack(question)
    image_tensor = torch.stack(image)
    answer_tensor = torch.stack(answer).to(torch.long)
    xs = Timesteps(
        {
            "question": question_tensor,
            "image": image_tensor,
            "answer": answer_tensor[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "question": torch.zeros_like(question_tensor),
            "image": torch.zeros(xs["image"].size(0), xs["image"].size(1), 1),
            "answer": answer_tensor[:, 1:],
        }
    )
    return xs, ys


def create_dataloader(
    tokenizer: Any, batch_size: int, split: str = "train", block_size: int = 1024
) -> DataLoader[Any]:
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            generic_collate_fn, sequence_length=block_size, mask_keys=["answer"]
        ),
    )


def create_infinite_dataloader(
    tokenizer: Any, batch_size: int, split: str = "train", block_size: int = 1024
) -> Any:
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return infinite_dataloader(
        partial(
            DataLoader,
            dataset,
            batch_size=batch_size,
            collate_fn=partial(
                generic_collate_fn,
                sequence_length=block_size,
                mask_keys=["answer"]
            )
        )
    )
