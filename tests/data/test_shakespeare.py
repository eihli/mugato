import pytest
import torch
from mugato.data.shakespeare import initialize, tokenize, create_dataloader
from mugato.tokenizer import Tokenizer
import tiktoken


@pytest.fixture
def tokenizer():
    return Tokenizer(tiktoken.get_encoding("r50k_base"))


def test_initialize():
    data = initialize()
    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert len(data["train"]) > 0
    assert len(data["val"]) > 0
    assert len(data["test"]) > 0
    assert isinstance(data["train"][0], str)


def test_create_dataloader(tokenizer):
    batch_size = 4
    dataloader = create_dataloader(tokenizer, batch_size=batch_size)

    # Check we get a DataLoader
    assert hasattr(dataloader, "__iter__")

    # Check batch has expected format
    batch = next(iter(dataloader))
    xs, ys, mask = batch
    assert isinstance(xs, dict)
    assert isinstance(ys, dict)
    assert isinstance(mask, dict)
    assert xs["text"].shape[0] == batch_size
    assert ys["text"].shape[0] == batch_size
    assert mask["text"].shape[0] == batch_size
    assert "<|endoftext|>First Citizen:\n" == tokenizer.decode_text(
        xs["text"][0, 0][:5]
    )
