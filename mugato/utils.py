import io
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from einops import rearrange
from IPython.display import Image as IPythonImage
from IPython.display import display
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

xdg_data_home = Path(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
)
data_home = xdg_data_home / "mugato"


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], [1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def as_tensor(x):
    if isinstance(x, Image.Image):
        return pil_to_tensor(x)
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)


image_transform = transforms.Compose(
    [
        as_tensor,
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop((192, 192), (1.0, 1.0)),
        normalize,
    ]
)


def select_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()  # For GCP TPU
    except ImportError:
        pass
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class Timesteps(OrderedDict):
    """An ordered dict of tensors with a `to` method to send them to GPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        for k, v in self.items():
            self[k] = v.to(device)
        return self


class TransformDataset(Dataset):
    """Wraps transform(...) around calls to __getitem__."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def generic_collate_fn(batch, sequence_length=1024, mask_keys=None):
    if mask_keys is None:
        mask_keys = []

    # Slice samples and filter out empty ones
    sliced = []
    for xs, ys in batch:
        xs_sliced = slice_to_context_window(sequence_length, xs)
        ys_sliced = slice_to_context_window(sequence_length, ys)

        # Check if the sample has at least one episode
        # (check the first value since all keys should have same episode count)
        if next(iter(xs_sliced.values())).size(0) > 0:
            sliced.append((xs_sliced, ys_sliced))

    if not sliced:
        raise ValueError(
            f"No samples in batch could fit within sequence_length={sequence_length}. "
            "Consider increasing block_size or using datasets with smaller episodes."
        )

    # Process the non-empty samples
    xs, ys = [v for v in zip(*sliced, strict=False)]
    xs, ys, ms = pad(xs), pad(ys), mask(ys, mask_keys)
    return xs, ys, ms


# These next 5 functions are helpers for when we need have a sample with
# a large number of episodes and creating a sequence from all of them would
# be larger than our context window.
#
# These helpers pick a random index for an episode from the sample
# and then slice up to the greatest index that's within our max sequence length.
def episode_num_tokens(sample):
    return sum([len(v[0]) for v in sample.values()])


def sample_num_tokens(sample):
    return episode_num_tokens(sample) * next(iter(sample.values())).size(0)


def sequence_episode_capacity(sequence_length, sample):
    return sequence_length // episode_num_tokens(sample)


def random_episode_start_index(sequence_length, sample):
    n_eps = next(iter(sample.values())).size(0)
    cap = min(n_eps, sequence_episode_capacity(sequence_length, sample))
    return random.randint(0, n_eps - cap)


def slice_to_context_window(sequence_length, sample):
    result = Timesteps()
    n = random_episode_start_index(sequence_length, sample)
    m = sequence_episode_capacity(sequence_length, sample)
    if m < 1:
        # Can't fit even one complete episode - return empty sample
        # This maintains structure but with 0 episodes
        for k in sample.keys():
            result[k] = sample[k][:0]
    else:
        for k in sample.keys():
            result[k] = sample[k][n : n + m]
    return result


def pad(batch, padding_value=0):
    """A specific-to-μGATO padding.

    Expects a *list of OrderedDict* (this is important).
    """
    padded = {}
    for k, v in batch[0].items():
        episode_length = max(sample[k].size(0) for sample in batch)
        token_length = max(sample[k].size(1) for sample in batch)
        for sample in batch:
            pad = (
                0,
                0,
                0,
                token_length - sample[k].size(1),
                0,
                episode_length - sample[k].size(0),
            )
            padded[k] = padded.get(k, [])
            padded[k].append(F.pad(sample[k], pad, value=0))
    return Timesteps([(k, torch.stack(v)) for k, v in padded.items()])


def mask(batch, mask_keys):
    result = Timesteps()
    for k, v in batch[0].items():
        episode_lengths = [sample[k].size(0) for sample in batch]
        token_lengths = [sample[k].size(1) for sample in batch]
        result[k] = torch.zeros(len(batch), max(episode_lengths), max(token_lengths))
        for i in range(len(batch)):
            if k in mask_keys:
                result[k][i][:episode_lengths[i], :token_lengths[i]] = 1
    return result


def tensor_as_gif(images):
    if images.is_cuda:
        images = images.cpu()
    images_np = images.numpy()  # Shape: (16, 3, 192, 192)
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Shape: (16, 192, 192, 3)
    if images_np.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)
    else:
        images_np = images_np.astype(np.uint8)
    image_list = [Image.fromarray(img) for img in images_np]
    buffer = io.BytesIO()
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    image_list[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=100,
        loop=0,
    )
    buffer.seek(0)
    # https://ipython.readthedocs.io/en/8.27.0/api/generated/IPython.display.html
    display(IPythonImage(data=buffer.getvalue(), format="gif"))


def images_to_patches(images, patch_size=16):
    return rearrange(
        images, "c (h s1) (w s2) -> (h w) (c s1 s2)", s1=patch_size, s2=patch_size
    )


def patches_to_images(patches, image_shape, patch_size=16):
    channels, height, width = image_shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    reconstructed = rearrange(
        patches,
        "b (ph pw) (c ps1 ps2) -> b c (ph ps1) (pw ps2)",
        ph=patch_height,
        pw=patch_width,
        ps1=patch_size,
        ps2=patch_size,
    )
    return reconstructed


def normalize_to_between_minus_one_plus_one(t: torch.Tensor):
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized


def apply_along_dimension(func, dim, tensor):
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


def discretize(x):
    x = as_tensor(x)
    bins = torch.linspace(x.min(), x.max(), steps=1023)
    tokens = torch.bucketize(x, bins)
    return tokens.tolist()


# This is going to be a lossy decode. Nothing you can do about that.
def undiscretize(x, scaled_min, scaled_max):
    bins = torch.linspace(scaled_min, scaled_max, steps=1023)
    return bins[x]


def mu_law_encode(x, M=256, mu=100):
    x = as_tensor(x)
    M = torch.tensor(M, dtype=x.dtype)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log(torch.abs(x) * mu + 1.0)
    x_mu = x_mu / torch.log(M * mu + 1.0)
    return x_mu


def mu_law_decode(x_mu, M=256, mu=100):
    M = torch.tensor(M, dtype=x_mu.dtype)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = (
        torch.sign(x_mu)
        * (torch.exp(torch.abs(x_mu) * torch.log(M * mu + 1.0)) - 1.0)
        / mu
    )
    return x


def clamp(x):
    return torch.clamp(x, -1, 1)


def interleave(tensors):
    return torch.cat(tensors, dim=1)


def deinterleave(splits, tensors):
    return torch.split(tensors, splits, dim=1)
