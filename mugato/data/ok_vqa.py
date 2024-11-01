from functools import partial
import random
import torch
from torch.utils.data import DataLoader
from mugato.data.utils import splits
from mugato.utils import (
    Timesteps,
    TransformDataset,
    generic_collate_fn,
    image_transform,
    as_tensor,
)
from datasets import load_dataset


def initialize():
    dataset = load_dataset("eihli/micro-ok-vqa")
    train_data = dataset["train"]
    # I happen to know this is a dataset of 80 train and 20 val.
    # I'm going to just split the val 50/50 to create a test set.
    val_data = dataset["validation"][:10]
    test_data = dataset["validation"][10:]
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def tokenize(tokenizer, sample):
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
    question = torch.stack(question)
    image = torch.stack(image)
    answer = torch.stack(answer).to(torch.long)
    xs = Timesteps(
        {
            "question": question,
            "image": image,
            "answer": answer[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "question": torch.zeros_like(question),
            "image": torch.zeros(xs["image"].size(0), xs["image"].size(1), 1),
            "answer": answer[:, 1:],
        }
    )
    return xs, ys


def create_dataloader(tokenizer, batch_size, split="train"):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=generic_collate_fn)
