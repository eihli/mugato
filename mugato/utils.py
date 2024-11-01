from collections import OrderedDict
import os
from pathlib import Path
import random
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms
from PIL import Image
from IPython.display import display, Image as IPythonImage
import io


xdg_data_home = Path(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
)
data_home = xdg_data_home / "mugato"


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], [1 / 0.229, 1 / 0.224, 1 / 0.225]
)

image_transform = transforms.Compose(
    [
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop((192, 192), (1.0, 1.0)),
        normalize,
    ]
)


def as_tensor(x):
    if isinstance(x, Image.Image):
        return pil_to_tensor(x)
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)


def select_device(device):
    # Check for TPU support (requires torch_xla library)
    try:
        import torch_xla.core.xla_model as xm

        tpu_available = xm.xla_device_hw() == "TPU"
    except ImportError:
        tpu_available = False

    # Set device based on availability
    if tpu_available:
        device = xm.xla_device()  # For GCP TPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # For NVIDIA GPUs
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon (macOS with MPS)
    else:
        device = torch.device("cpu")  # Fallback to CPU
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


def generic_collate_fn(batch, sequence_length=1024):
    sliced = [
        (
            slice_to_context_window(sequence_length, xs),
            slice_to_context_window(sequence_length, ys),
        )
        for xs, ys in batch
    ]
    # sliced is a (B, 2, ...) list.
    # the 2 is xs, ys
    xs, ys = [v for v in zip(*sliced)]
    xs, ys, ms = pad(xs), pad(ys), mask(ys)
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
    n = random_episode_start_index(1024, sample)
    m = sequence_episode_capacity(1024, sample)
    if m < 1:
        for k in sample.keys():
            result[k] = sample[k][:, :sequence_length]
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


def mask(batch):
    result = Timesteps()
    for k, v in batch[0].items():
        episode_lengths = [sample[k].size(0) for sample in batch]
        token_lengths = [sample[k].size(1) for sample in batch]
        result[k] = torch.zeros(len(batch), max(episode_lengths), max(token_lengths))
        for i in range(len(batch)):
            result[k][i][: episode_lengths[i], : token_lengths[i]] = 1
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


def discretize(x: torch.Tensor) -> torch.Tensor:
    bins = torch.linspace(x.min(), x.max(), steps=1024)
    tokens = torch.bucketize(x, bins)
    return tokens


def undiscretize(tokens: torch.Tensor, original_min: float, original_max: float):
    bins = torch.linspace(original_min, original_max, steps=1025)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[tokens]


def interleave(tensors):
    return torch.cat(tensors, dim=1)


def deinterleave(splits, tensors):
    return torch.split(tensors, splits, dim=1)
