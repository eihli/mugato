import math
import pytest
import tiktoken
import torch
from mugato.tokenizer import (
    Tokenizer,
    discretize,
    undiscretize,
    image_to_patches,
    patches_to_image,
)


@pytest.fixture
def tokenizer():
    return Tokenizer(tiktoken.get_encoding("r50k_base"))


def test_text_tokenization(tokenizer):
    text = "Hello, world!"
    tokens = tokenizer.encode_text(text)
    assert torch.equal(tokens, torch.tensor([[15496], [11], [995], [0]]))
    decoded = tokenizer.decode_text(tokens)
    assert decoded == text


def test_discrete_tokenization(tokenizer):
    values = [0, 1, 2, 3]
    tokens = tokenizer.encode_discrete(values)
    # assert torch.equal(tokens, torch.tensor(values).unsqueeze(-1) + tokenizer.n_text)
    assert torch.equal(tokens, torch.tensor([[50257], [50258], [50259], [50260]]))
    decoded = tokenizer.decode_discrete(tokens)
    assert decoded == values


# def test_continuous_tokenization(tokenizer):
#     values = torch.tensor([0.1, 0.5, 0.9])
#     tokens, min_val, max_val = tokenizer.encode_continuous(values)
#     decoded = tokenizer.decode_continuous(tokens, min_val, max_val)
#     assert isinstance(tokens, torch.Tensor)
#     assert tokens.dim() == 2
#     # Check values are approximately preserved (within discretization error)
#     assert torch.allclose(torch.tensor(decoded), values, atol=0.1)

# def test_image_tokenization(tokenizer):
#     # Create dummy 3x192x192 image
#     image = torch.rand(3, 192, 192)
#     patches = tokenizer.encode_image(image, patch_size=16)
#     decoded = tokenizer.decode_image(patches, image_shape=(3, 192, 192), patch_size=16)

#     assert patches.shape == (144, 768)  # (12x12) patches, each 16x16x3 flattened
#     assert decoded.shape == image.shape
#     # Check image is approximately preserved (within normalization error)
#     assert torch.allclose(decoded, image, atol=0.1)

# def test_discretize_undiscretize():
#     x = torch.linspace(0, 1, 100)
#     tokens = discretize(x)
#     reconstructed = undiscretize(tokens, x.min(), x.max())
#     # Check values are approximately preserved
#     assert torch.allclose(torch.tensor(reconstructed), x, atol=0.1)

# def test_image_patches():
#     image = torch.rand(3, 192, 192)
#     patches = image_to_patches(image, patch_size=16)
#     reconstructed = patches_to_image(patches, (3, 192, 192), patch_size=16)
#     assert patches.shape == (144, 768)
#     assert reconstructed.shape == image.shape
#     assert torch.allclose(reconstructed, image)
