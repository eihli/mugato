import torch

from mugato.data.utils import generic_collate_fn
from mugato.utils import Timesteps


def test_mugato_block_size_too_small() -> None:
    """
    Test edge cases in Mugato.

    This test checks that Mugato correctly handles edge cases:
    1. When samples are too long for the context window
    2. When samples fit within the context window
    """
    BLOCK_SIZE = 5
    # Test 1: Samples too long - should raise ValueError
    batch_too_long = [
        (
            Timesteps({
                "text": torch.arange(BLOCK_SIZE).reshape(1, BLOCK_SIZE, 1),
            }),
            Timesteps({
                "text": torch.arange(BLOCK_SIZE).reshape(1, BLOCK_SIZE, 1),
            }),
        )
    ]
    try:
        xs, ys, ms = generic_collate_fn(
            batch_too_long, BLOCK_SIZE-1, mask_keys=["text"]
        )
    except ValueError as e:
        assert "No samples in batch could fit" in str(e)
    else:
        raise AssertionError("Expected generic_collate_fn to raise ValueError")


def test_mugato_block_size_acceptable() -> None:
    """
    Test edge cases in Mugato.

    This test checks that Mugato correctly handles edge cases:
    1. When samples are too long for the context window
    2. When samples fit within the context window
    """
    # Test 1: Samples too long - should raise ValueError
    BLOCK_SIZE = 5
    batch_too_long = [
        (
            Timesteps({
                "text": torch.arange(BLOCK_SIZE).reshape(1, BLOCK_SIZE, 1),
            }),
            Timesteps({
                "text": torch.arange(BLOCK_SIZE).reshape(1, BLOCK_SIZE, 1),
            }),
        )
    ]
    try:
        xs, ys, ms = generic_collate_fn(
            batch_too_long, BLOCK_SIZE+1, mask_keys=["text"]
        )
    except ValueError as e:
        raise AssertionError(f"Expected generic_collate_fn to not raise {e}")\
            from e
