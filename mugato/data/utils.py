import importlib
from itertools import cycle
import pkgutil
import mugato.data
import torch
from torch.utils.data import DataLoader
from functools import partial
from mugato.utils import TransformDataset, generic_collate_fn
import logging

logger = logging.getLogger(__name__)


def splits(dataset, train_split=0.8, val_split=0.9):
    num_samples = len(dataset)
    train_split = int(num_samples * train_split)
    val_split = int(num_samples * val_split)
    train_data = dataset[:train_split]
    val_data = dataset[train_split:val_split]
    test_data = dataset[val_split:]
    return train_data, val_data, test_data


# TODO: This is idea of an "infinite dataloader" is convenient for hacking
# but isn't good for training.
# If one dataset is 100 samples and another is 1000, then the model will
# see each sample from the 100 sample dataset 10 times by the time it sees
# each sample of the 1000 sample dataset once.
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
            module = importlib.reload(
                module
            )  # TODO: hacky thing to get thing working...
            if all(
                hasattr(module, attr)
                for attr in ["initialize", "tokenize", "create_dataloader"]
            ):
                datasets.append(module)
        except ImportError:
            continue
    return datasets


def initialize_all():
    logger.debug("Initializing all datasets")
    dataset_modules = find_datasets()
    logger.debug(f"Found {dataset_modules} dataset modules")
    datasets = {}
    for dataset_module in sorted(dataset_modules, key=lambda x: x.__name__):
        datasets[dataset_module.__name__] = dataset_module.initialize()
    return datasets

def create_combined_dataloader_from_module(tokenizer, batch_size, split="train", block_size=1024):
    dataset_modules = find_datasets()
    all_dataloaders = []
    for dataset_module in dataset_modules:
        data_loader = dataset_module.create_infinite_dataloader(tokenizer, batch_size, split, block_size)
        all_dataloaders.append(data_loader)
    return cycle(all_dataloaders)

def create_combined_dataloader(tokenizer, batch_size, split="train", block_size=1024):
    datasets = initialize_all()
    all_datasets = []

    for dataset_name, dataset in datasets.items():
        module = importlib.import_module(dataset_name)
        dataset_split = TransformDataset(
            dataset[split], partial(module.tokenize, tokenizer)
        )
        all_datasets.append(dataset_split)

    # TODO: It would be nice to have some metadata about the datasets,
    # like the name of the dataset, the number of samples, etc.
    return cycle(
        [
            infinite_dataloader(
                partial(
                    DataLoader,
                    dataset,
                    batch_size=batch_size,
                    collate_fn=partial(generic_collate_fn, sequence_length=block_size),
                )
            )
            for dataset in all_datasets
        ]
    )
