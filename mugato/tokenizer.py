import math
from typing import List, Protocol
from einops import rearrange
import torch
from mugato.util import as_tensor


class TextTokenizer(Protocol):
    n_vocab: int
    eot_token: int

    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


def discretize(x):
    x = as_tensor(x)
    bins = torch.linspace(x.min(), x.max(), steps=1024)
    tokens = torch.bucketize(x, bins)
    return tokens.tolist()


# This is going to be a lossy decode. Nothing you can do about that.
def undiscretize(x, original_min, original_max):
    bins = torch.linspace(original_min, original_max, steps=1025)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[x]


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

    def encode_text(self, text):
        return torch.tensor(self.text_tokenizer.encode(text)).unsqueeze(-1)

    def decode_text(self, tokens):
        return self.text_tokenizer.decode(tokens.squeeze(-1).tolist())

    def encode_discrete(self, xs):
        return torch.tensor(xs, dtype=torch.long).unsqueeze(-1) + self.n_text

    def decode_discrete(self, tokens):
        return (tokens - self.n_text).squeeze(-1).tolist()

    def encode_continuous(self, xs):
        return self.encode_discrete(discretize(xs)), min(xs), max(xs)

    def decode_continuous(self, tokens, original_min=0, original_max=1):
        return undiscretize(
            self.decode_discrete(tokens), original_min, original_max
        ).tolist()

    def encode_image(self, image, patch_size=16):
        patches = image_to_patches(image, patch_size=patch_size)
        xs = apply_along_dimension(
            normalize_to_between_minus_one_plus_one, 1, patches
        ) / math.sqrt(patch_size)
        return xs

    def decode_image(self, tokens, image_shape=(3, 192, 192), patch_size=16):
        # Slightly lossy because I'm not saving the values used for scaling from encoding.
        patches = (tokens * math.sqrt(patch_size) + 1) / 2
        images = patches_to_image(patches, image_shape, patch_size=patch_size)
        return images


def image_to_patches(image, patch_size=16):
    return rearrange(image, "c (h s1) (w s2) -> (h w) (c s1 s2)", s1=16, s2=16)


# We don't need this as part of Gato. It's just here to play with and visually test the code.
def patches_to_image(patches, image_shape, patch_size=12):
    channels, height, width = image_shape
    patch_height = height // patch_size
    patch_width = width // patch_size
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


def apply_along_dimension(func, dim, tensor):
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


def normalize_to_between_minus_one_plus_one(t: torch.Tensor):
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized
