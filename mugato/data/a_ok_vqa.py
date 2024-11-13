from ast import literal_eval
from functools import partial
import random
import torch
from torch.utils.data import DataLoader, Dataset
from mugato.data.utils import splits
from mugato.utils import (
    Timesteps,
    TransformDataset,
    generic_collate_fn,
    image_transform,
    as_tensor,
)
from datasets import load_dataset


def example():
    """Demonstrate the shape of the original dataset.

    >>> ds = load_dataset("HuggingFaceM4/A-OKVQA")
    >>> ds.keys()
    dict_keys(['train', 'validation', 'test'])
    >>> ds['train'] [0]
    {
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480>,
        'question_id': '22MexNkBPpdZGX6sxbxVBH',
        'question': 'What is the man by the bags awaiting?',
        'choices': ['skateboarder', 'train', 'delivery', 'cab'],
        'correct_choice_idx': 3,
        'direct_answers': "['ride', 'ride', 'bus', 'taxi', 'travelling', 'traffic', 'taxi', 'cab', 'cab', 'his ride']",
        'difficult_direct_answer': False,
        'rationales': ['A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.',
        'He has bags as if he is going someone, and he is on a road waiting for vehicle that can only be moved on the road and is big enough to hold the bags.',
        'He looks to be waiting for a paid ride to pick him up.']
    }
    """


def initialize():
    dataset = load_dataset("HuggingFaceM4/A-OKVQA")
    return {
        "train": dataset["train"],
        "val": dataset["validation"],
        "test": dataset["test"],
    }


def tokenize(tokenizer, sample):
    question = [tokenizer.encode_text(sample["question"])]
    image = [tokenizer.encode_image(image_transform(as_tensor(sample["image"])))]
    eot = torch.tensor([[tokenizer.eot_token_id]])
    answer = random.choice(literal_eval(sample["direct_answers"]))
    rational = random.choice(sample["rationales"])
    if random.random() < 0.5:
        answer = [
            torch.concat(
                [
                    eot,
                    tokenizer.encode_text(answer),
                    eot,
                ]
            )
        ]
    else:
        answer = [
            torch.concat(
                [
                    eot,
                    tokenizer.encode_text(
                        f"{answer.upper()}, because {rational[0].lower()}{rational[1:]}"
                    ),
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
