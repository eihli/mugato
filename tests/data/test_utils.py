"""These are bad tests.

They are flaky, slow, and depend on specific filenames.

It's not really a test. Just a sanity check and example."""

import pytest
import tiktoken
from mugato.data.utils import create_combined_dataloader
from mugato.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer(tiktoken.get_encoding("r50k_base"))


def test_combined_dataloader(tokenizer):
    combined_dataloader = create_combined_dataloader(
        tokenizer, batch_size=4, split="train"
    )
    dataloader_iter = iter(combined_dataloader)
    dataloader = next(dataloader_iter)
    batch = next(dataloader)
    xs, ys, mask = batch
    assert list(xs.keys()) == ["mission", "direction", "image", "action"]
    dataloader = next(dataloader_iter)
    batch = next(dataloader)
    xs, ys, mask = batch
    assert list(xs.keys()) == ["question", "image", "answer"]
