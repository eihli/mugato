import math
from typing import Any, Protocol

import torch
from einops import rearrange

from mugato.utils import (
    clamp,
    discretize,
    mu_law_decode,
    mu_law_encode,
    undiscretize,
)


class TextTokenizer(Protocol):
    @property
    def n_vocab(self) -> int: ...
    @property
    def eot_token(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class Tokenizer:
    def __init__(self, text_tokenizer: TextTokenizer):
        self.text_tokenizer = text_tokenizer
        self.eot_token_id = text_tokenizer.eot_token
        self.eot_token = text_tokenizer.decode([self.eot_token_id])
        self.n_text = text_tokenizer.n_vocab
        self.n_discrete = 1024
        self.separator = self.boa_token = (
            1023  # Separator between observation and action.
        )
        self.vocab_size = self.n_text + self.n_discrete

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self.text_tokenizer.encode(text), dtype=torch.long
        ).unsqueeze(-1)

    def decode_text(self, tokens: torch.Tensor) -> str:
        return self.text_tokenizer.decode(tokens.squeeze(-1).tolist())

    def encode_discrete(self, xs: list[int]) -> torch.Tensor:
        return torch.tensor(xs, dtype=torch.long).unsqueeze(-1) + self.n_text

    def decode_discrete(self, tokens: torch.Tensor) -> list[int]:
        # The model might output a token below the discrete vocabulary.
        # (The discrete vocabulary is 0-1023 shifted by n_text.)
        # How should we deal with tokens that decode (token - self.n_text)
        # to negative tokens? I say we don't. Leave that to the application.
        return (tokens % self.n_text).squeeze(-1).tolist()

    def encode_continuous(
        self, xs: torch.Tensor | Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = mu_law_encode(xs)
        return (
            self.encode_discrete(discretize(clamp(encoded))),
            encoded.min(),
            encoded.max(),
        )

    def decode_continuous(
        self,
        tokens: torch.Tensor,
        encoded_min: float,
        encoded_max: float
    ) -> list[float]:
        encoded = undiscretize(self.decode_discrete(tokens), encoded_min, encoded_max)
        return mu_law_decode(encoded).tolist()

    def encode_image(self, image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        patches = image_to_patches(image, patch_size=patch_size)
        xs = apply_along_dimension(
            normalize_to_between_minus_one_plus_one, 1, patches
        ) / math.sqrt(patch_size)
        return xs

    def decode_image(
        self,
        tokens: torch.Tensor,
        image_shape: tuple[int, int, int] = (3, 192, 192),
        patch_size: int = 16
    ) -> torch.Tensor:
        # Slightly lossy because I'm not saving the values used for scaling
        # from encoding.
        patches = (tokens * math.sqrt(patch_size) + 1) / 2
        images = patches_to_image(patches, image_shape, patch_size=patch_size)
        return images


def image_to_patches(image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    return rearrange(image, "c (h s1) (w s2) -> (h w) (c s1 s2)", s1=16, s2=16)


# We don't need this as part of Gato. It's just here to play with and visually
# test the code.
def patches_to_image(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    patch_size: int = 12
) -> torch.Tensor:
    channels, height, width = image_shape
    reconstructed = rearrange(
        patches,
        "(h w) (c p1 p2) -> c (h p1) (w p2)",
        h=12,
        w=12,
        c=3,
        p1=16,
        p2=16,
    )
    return reconstructed


def apply_along_dimension(func: Any, dim: int, tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


def normalize_to_between_minus_one_plus_one(t: torch.Tensor) -> torch.Tensor:
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized
