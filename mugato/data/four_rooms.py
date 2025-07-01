from functools import partial

import minari
import minigrid
import numpy as np
import torch
from torch.utils.data import DataLoader

from mugato.data.utils import infinite_dataloader
from mugato.utils import (
    Timesteps,
    TransformDataset,
    generic_collate_fn,
    image_transform,
)


def initialize():
    # See: https://minari.farama.org/api/minari_dataset/minari_dataset/
    # You can't slice a Minari Dataset. But you can set the episode_indices.
    train_data = minari.load_dataset("D4RL/minigrid/fourrooms-v0", download=True)
    val_data = minari.load_dataset("D4RL/minigrid/fourrooms-v0", download=True)
    test_data = minari.load_dataset("D4RL/minigrid/fourrooms-v0", download=True)
    train_split = int(train_data.total_episodes * 0.8)
    val_split = int(train_data.total_episodes * 0.9)
    # NOTE: Order matters here!
    # Minari has some hidden *shared* internal state when you set `episode_indices`.
    # If you set anything's `episode_indices` to be *shorter* than `total_episodes`,
    # then any subsequent sets of `episode_indices` will be relative to the new
    # shorter length.
    # TODO: Submit a bug report and a fix.
    test_data.episode_indices = np.arange(val_split, train_data.total_episodes)
    val_data.episode_indices = np.arange(train_split, val_split)
    train_data.episode_indices = np.arange(0, train_split)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


# Some FourRooms/Minigrid-specific stuff to turn
# a 7x7x3 non-pixel observation into an pixel/image observation.
lut = np.zeros((256, 3), dtype=np.uint8)
for idx, color_name in minigrid.core.constants.IDX_TO_COLOR.items():
    lut[idx] = minigrid.core.constants.COLORS[color_name]


def four_rooms_to_rgb(images):
    """Convert discrete "image" observations into actual images.

    I'm expecting this will improve our image modality while not losing
    much. The downside is we can fit less in our context window. Note:
    We might need to overlay the color/type image (index 1) with the
    state image (index 2), if we really don't want to lose any info."""
    # Apply lookup to second channel
    return torch.from_numpy(lut[images[:, :, :, 1]]).permute(0, 3, 1, 2)


def tokenize(tokenizer, episode):
    # slice to -1 on all observations because we have 1 more observations than actions.
    mission_tokens = [
        tokenizer.encode_text(mission)
        for mission in episode.observations["mission"][:-1]
    ]
    direction_tokens = [
        tokenizer.encode_discrete([direction])
        for direction in episode.observations["direction"][:-1]
    ]
    image = episode.observations["image"][:-1]
    image = four_rooms_to_rgb(image)
    image_tokens = [tokenizer.encode_image(image) for image in image_transform(image)]
    action_tokens = [
        tokenizer.encode_discrete([tokenizer.separator, action])
        for action in episode.actions
    ]

    mission = torch.stack(mission_tokens)
    direction = torch.stack(direction_tokens)
    image = torch.stack(image_tokens)
    action = torch.stack(action_tokens)

    xs = Timesteps(
        {
            "mission": mission,
            "direction": direction,
            "image": image,
            "action": action[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "mission": torch.zeros_like(mission),
            "direction": torch.zeros_like(direction),
            # We're not predicting image patches, so we don't need "real" targets.
            # We just need something with the same channel dimensionality as
            # our other tokens
            # so that we can concat them all together and predict on the
            # sequenced tokens.
            "image": torch.zeros(image.size(0), image.size(1), 1),
            "action": action[:, 1:],
        }
    )
    return xs, ys


def create_dataloader(tokenizer, batch_size, split="train", block_size=1024):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            generic_collate_fn, sequence_length=block_size, mask_keys=["action"]
        ),
    )


def create_infinite_dataloader(tokenizer, batch_size, split="train", block_size=1024):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return infinite_dataloader(
        partial(
            DataLoader,
            dataset,
            batch_size=batch_size,
            collate_fn=partial(
                generic_collate_fn,
                sequence_length=block_size,
                mask_keys=["action"]
            )
        )
    )
