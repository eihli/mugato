import torch

from mugato.data.utils import generic_collate_fn
from mugato.utils import Timesteps, slice_to_context_window


def test_slice_to_context_window() -> None:
    sample = Timesteps({
        "text": torch.arange(20).reshape(2, 10, 1),
        "image": torch.arange(20).reshape(2, 10, 1),
    })
    result = slice_to_context_window(15, sample)
    assert result['text'].shape == (1, 10, 1)
    assert result['image'].shape == (1, 5, 1)

    sample = Timesteps({
        "text": torch.arange(4).reshape(2, 2, 1),
        "image": torch.arange(12).reshape(2, 6, 1),
    })
    result = slice_to_context_window(12, sample)
    assert result['text'].shape == (1, 2, 1)
    assert result['image'].shape == (1, 6, 1)

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
        raise AssertionError("Expected not to raise. Do the epsilon thing. {e}") from e



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
