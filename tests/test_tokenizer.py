import pytest
import tiktoken
import torch

from mugato.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(tiktoken.get_encoding("r50k_base"))


def test_text_tokenization(tokenizer: Tokenizer) -> None:
    text = "Hello, world!"
    tokens = tokenizer.encode_text(text)
    assert torch.equal(tokens, torch.tensor([[15496], [11], [995], [0]]))
    decoded = tokenizer.decode_text(tokens)
    assert decoded == text


def test_discrete_tokenization(tokenizer: Tokenizer) -> None:
    values = [0, 1, 2, 3]
    tokens = tokenizer.encode_discrete(values)
    # assert torch.equal(tokens, torch.tensor(values).unsqueeze(-1) + tokenizer.n_text)
    assert torch.equal(tokens, torch.tensor([[50257], [50258], [50259], [50260]]))
    decoded = tokenizer.decode_discrete(tokens)
    assert decoded == values


def test_continuous_tokenization(tokenizer: Tokenizer) -> None:
    values = [-2, -1, 0, 1, 2, 3, 4]
    tokens, min_val, max_val = tokenizer.encode_continuous(values)
    decoded = tokenizer.decode_continuous(tokens, min_val.item(), max_val.item())
    assert torch.allclose(
        torch.tensor(decoded), torch.tensor(values, dtype=torch.float32), atol=0.1
    )


def test_image_tokenization(tokenizer: Tokenizer) -> None:
    # Create dummy 3x192x192 image
    image = torch.rand(3, 192, 192)
    patches = tokenizer.encode_image(image, patch_size=16)
    decoded = tokenizer.decode_image(patches, image_shape=(3, 192, 192), patch_size=16)

    assert patches.shape == (144, 768)  # (12x12) patches, each 16x16x3 flattened
    assert decoded.shape == image.shape
    # Check image is approximately preserved (within normalization error)
    assert torch.allclose(decoded, image, atol=0.1)
