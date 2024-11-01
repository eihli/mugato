import importlib
from itertools import cycle
import pkgutil
import mugato.data
import torch
from torch.utils.data import DataLoader
from functools import partial
from mugato.utils import TransformDataset, generic_collate_fn


def splits(dataset, train_split=0.8, val_split=0.9):
    num_samples = len(dataset)
    train_split = int(num_samples * train_split)
    val_split = int(num_samples * val_split)
    train_data = dataset[:train_split]
    val_data = dataset[train_split:val_split]
    test_data = dataset[val_split:]
    return train_data, val_data, test_data


def infinite_dataloader(fn):
    it = iter(fn())
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(fn())


def find_datasets():
    # If a module in mugato.data has functions for initialize, tokenize,
    # and create_dataloader, then it's considered a dataset.
    datasets = []
    for _, name, _ in pkgutil.iter_modules(mugato.data.__path__):
        try:
            module = importlib.import_module(f"mugato.data.{name}")
            if all(
                hasattr(module, attr)
                for attr in ["initialize", "tokenize", "create_dataloader"]
            ):
                datasets.append(module)
        except ImportError:
            continue
    return datasets


def initialize_all():
    dataset_modules = find_datasets()
    datasets = {}
    for dataset_module in dataset_modules:
        datasets[dataset_module.__name__] = dataset_module.initialize()
    return datasets


def create_combined_dataloader(tokenizer, batch_size, split="train"):
    datasets = initialize_all()
    all_datasets = []

    for dataset_name, dataset in datasets.items():
        module = importlib.import_module(dataset_name)
        dataset_split = TransformDataset(
            dataset[split], partial(module.tokenize, tokenizer)
        )
        all_datasets.append(dataset_split)

    return cycle(
        [
            infinite_dataloader(
                partial(
                    DataLoader,
                    dataset,
                    batch_size=batch_size,
                    collate_fn=generic_collate_fn,
                )
            )
            for dataset in all_datasets
        ]
    )
