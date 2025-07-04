"""These are bad tests.

They are flaky, slow, and depend on specific filenames.

It's not really a test. Just a sanity check and example."""

import pytest
import tiktoken

from mugato.data.utils import create_combined_dataloader
from mugato.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(tiktoken.get_encoding("r50k_base"))

@pytest.mark.slow
def test_combined_dataloader(tokenizer: Tokenizer) -> None:
    combined_dataloader = create_combined_dataloader(
        tokenizer, batch_size=4, split="train"
    )
    dataloader_iter = iter(combined_dataloader)
    dataloader = next(dataloader_iter)
    batch = next(dataloader)
    xs, ys, mask = batch
    keys_1 = list(xs.keys())
    dataloader = next(dataloader_iter)
    batch = next(dataloader)
    xs, ys, mask = batch
    keys_2 = list(xs.keys())
    assert keys_1 != keys_2, (
        "Iterating over the combined dataloader should change the keys"
    )
