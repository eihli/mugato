import io
import os
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms  # type: ignore
from einops import rearrange
from IPython.display import Image as IPythonImage
from IPython.display import display
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

if TYPE_CHECKING:
    from dataclasses import Field

    @runtime_checkable
    class DataclassProtocol(Protocol):
        """Protocol for dataclass instances."""
        __dataclass_fields__: dict[str, Field[Any]]
else:
    # At runtime, use a simpler protocol that doesn't require Field
    @runtime_checkable
    class DataclassProtocol(Protocol):
        """Protocol for dataclass instances."""
        __dataclass_fields__: dict[str, Any]

xdg_data_home = Path(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
)
data_home = xdg_data_home / "mugato"


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], [1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def as_tensor(x: Image.Image | torch.Tensor | Any) -> torch.Tensor:
    if isinstance(x, Image.Image):
        result: torch.Tensor = pil_to_tensor(x)  # type: ignore
        return result
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)  # type: ignore


def as_dict(x: DataclassProtocol | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(x):
        return asdict(x)  # type: ignore[arg-type, no-any-return, call-overload]
    if isinstance(x, dict):
        return x
    raise TypeError(f"Expected a dataclass or dict, got {type(x)}")

image_transform = transforms.Compose(
    [
        as_tensor,
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop((192, 192), (1.0, 1.0)),
        normalize,
    ]
)


def select_device() -> str:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return str(xm.xla_device())  # For GCP TPU
    except ImportError:
        pass
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return str(device)


class Timesteps(OrderedDict):
    """An ordered dict of tensors with a `to` method to send them to GPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = "cpu"

    def to(self, device: str | torch.device) -> 'Timesteps':
        self.device = str(device)
        for k, v in self.items():
            self[k] = v.to(device)
        return self


class TransformDataset(Dataset):
    """Wraps transform(...) around calls to __getitem__."""

    def __init__(self, dataset: Any, transform: Callable) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.transform(self.dataset[idx])


def generic_collate_fn(
    batch: list[tuple[Timesteps, Timesteps]],
    sequence_length: int = 1024,
    mask_keys: list[str] | None = None
) -> tuple[Timesteps, Timesteps, Timesteps]:
    if mask_keys is None:
        mask_keys = []

    sliced: list[tuple[Timesteps, Timesteps]] = []
    for x, y in batch:
        xs_sliced = slice_to_context_window(sequence_length, x)
        ys_sliced = slice_to_context_window(sequence_length, y)
        sliced.append((xs_sliced, ys_sliced))

    xs = collate_and_pad_timesteps(tuple(x[0] for x in sliced))
    ys = collate_and_pad_timesteps(tuple(y[1] for y in sliced))
    ms = mask([y[1] for y in sliced], mask_keys)
    return xs, ys, ms


# These next 5 functions are helpers for when we need have a sample with
# a large number of episodes and creating a sequence from all of them would
# be larger than our context window.
#
# These helpers pick a random index for an episode from the sample
# and then slice up to the greatest index that's within our max sequence length.
def episode_num_tokens(sample: dict[str, torch.Tensor]) -> int:
    return sum([len(v[0]) for v in sample.values()])


def sample_num_tokens(sample: dict[str, torch.Tensor]) -> int:
    return episode_num_tokens(sample) * next(iter(sample.values())).size(0)


def sequence_episode_capacity(
    sequence_length: int, sample: dict[str, torch.Tensor]
) -> int:
    return sequence_length // episode_num_tokens(sample)


def random_episode_start_index(
    sequence_length: int, sample: dict[str, torch.Tensor]
) -> int:
    n_eps = next(iter(sample.values())).size(0)
    cap = min(n_eps, sequence_episode_capacity(sequence_length, sample))
    return random.randint(0, n_eps - cap)



def slice_to_context_window(
    sequence_length: int, sample: dict[str, torch.Tensor]
) -> Timesteps:
    result = Timesteps()
    n = random_episode_start_index(sequence_length, sample)
    m = sequence_episode_capacity(sequence_length, sample)
    # Timesteps of an Episode:
    # Text-like mission: (Stack the blue box on top of the green box)
    # Image-like observation
    # Some continuous valued observations (arm position, rotation, gripper, etc...)
    #
    # [Episode, Tokens, Channels]
    if m < 1:
        # Can't fit even one complete episode - return empty sample
        # This maintains structure but with 0 episodes
        remaining = sequence_length
        for k in sample.keys():
            to_take = min(remaining, sample[k].size(1))
            result[k] = sample[k][[0], :to_take]
            remaining -= to_take
    else:
        for k in sample.keys():
            result[k] = sample[k][n : n + m]
    return result


def collate_and_pad_timesteps(
    batch: Sequence[Timesteps], padding_value: int = 0
) -> Timesteps:
    """Collate and pad a batch of μGATO timesteps to uniform dimensions.

    Finds the maximum episode and token lengths across the batch and pads
    all samples to these dimensions.

    Args:
        batch: List of Timesteps (OrderedDict) to collate and pad
        padding_value: Value to use for padding (default: 0)

    Returns:
        Timesteps with all samples padded to uniform dimensions
    """
    padded: dict[str, list[torch.Tensor]] = {}
    for k, _ in batch[0].items():
        episode_length = max(sample[k].size(0) for sample in batch)
        token_length = max(sample[k].size(1) for sample in batch)
        padded[k] = []
        for sample in batch:
            pad = (
                0,
                0,
                0,
                token_length - sample[k].size(1),
                0,
                episode_length - sample[k].size(0),
            )
            padded[k].append(F.pad(sample[k], pad, value=0))
    return Timesteps([(k, torch.stack(v)) for k, v in padded.items()])


def mask(batch: list[Timesteps], mask_keys: list[str]) -> Timesteps:
    result = Timesteps()
    for k, _ in batch[0].items():
        episode_lengths = [sample[k].size(0) for sample in batch]
        token_lengths = [sample[k].size(1) for sample in batch]
        result[k] = torch.zeros(len(batch), max(episode_lengths), max(token_lengths))
        for i in range(len(batch)):
            if k in mask_keys:
                result[k][i][:episode_lengths[i], :token_lengths[i]] = 1
    return result


def tensor_as_gif(images: torch.Tensor) -> None:
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


def images_to_patches(images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    return rearrange(
        images, "c (h s1) (w s2) -> (h w) (c s1 s2)", s1=patch_size, s2=patch_size
    )


def patches_to_images(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    patch_size: int = 16
) -> torch.Tensor:
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


def normalize_to_between_minus_one_plus_one(t: torch.Tensor) -> torch.Tensor:
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized


def apply_along_dimension(
    func: Callable, dim: int, tensor: torch.Tensor
) -> torch.Tensor:
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


def discretize(x: torch.Tensor | Any) -> list[int]:
    x = as_tensor(x)
    bins = torch.linspace(x.min(), x.max(), steps=1023)
    tokens = torch.bucketize(x, bins)
    return tokens.tolist()


# This is going to be a lossy decode. Nothing you can do about that.
def undiscretize(x: list[int], scaled_min: float, scaled_max: float) -> torch.Tensor:
    bins = torch.linspace(scaled_min, scaled_max, steps=1023)
    return bins[x]


def mu_law_encode(x: torch.Tensor | Any, M: int = 256, mu: int = 100) -> torch.Tensor:
    x = as_tensor(x)
    M_tensor = torch.tensor(M, dtype=x.dtype)
    mu_tensor = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log(torch.abs(x) * mu_tensor + 1.0)
    x_mu = x_mu / torch.log(M_tensor * mu_tensor + 1.0)
    return x_mu


def mu_law_decode(x_mu: torch.Tensor, M: int = 256, mu: int = 100) -> torch.Tensor:
    M_tensor = torch.tensor(M, dtype=x_mu.dtype)
    mu_tensor = torch.tensor(mu, dtype=x_mu.dtype)
    x = (
        torch.sign(x_mu)
        * (torch.exp(torch.abs(x_mu) * torch.log(M_tensor * mu_tensor + 1.0)) - 1.0)
        / mu_tensor
    )
    return x


def clamp(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1, 1)


def interleave(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def deinterleave(splits: list[int], tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return torch.split(tensors, splits, dim=1)
