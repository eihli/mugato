import numpy as np
import minigrid
import torch

from mugato.util import image_transform, Timesteps

def initialize():


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


def tokenize_four_rooms(tokenizer, episode):
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
            # We just need something with the same channel dimensionality as our other tokens
            # so that we can concat them all together and predict on the sequenced tokens.
            "image": torch.zeros(image.size(0), image.size(1), 1),
            "action": action[:, 1:],
        }
    )
    return xs, ys
