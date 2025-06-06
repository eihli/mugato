"""Datasets and some dataset-related utilities.

Almost every file in this directory is a dataset.

A dataset is a module that has 3 functions:

- initialize(): Initializes the dataset and splits, downloading if necessary.
  - Returns a dictionary with keys 'train', 'val', and 'test' and values that are indexable.
- tokenize(): Given a sample, returns an (input, targets) tuple, each an ordered dict.
- create_dataloader(): A function that takes a tokenizer and returns a dataloader.
"""
import logging
import os

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)