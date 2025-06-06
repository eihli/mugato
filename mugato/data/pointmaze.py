from functools import partial
import numpy as np
import minari
import torch
from torch.utils.data import DataLoader
from mugato.data.utils import infinite_dataloader
from mugato.utils import (
    Timesteps,
    TransformDataset,
    generic_collate_fn,
)


def initialize():
    # See: https://minari.farama.org/api/minari_dataset/minari_dataset/
    # You can't slice a Minari Dataset. But you can set the episode_indices.
    train_data = minari.load_dataset("D4RL/pointmaze/open-v2", download=True)
    val_data = minari.load_dataset("D4RL/pointmaze/open-v2", download=True)
    test_data = minari.load_dataset("D4RL/pointmaze/open-v2", download=True)
    train_split = int(train_data.total_episodes * 0.8)
    val_split = int(train_data.total_episodes * 0.9)
    # NOTE: Order matters here!
    # Minari has some hidden *shared* internal state when you set `episode_indices`.
    # If you set anything's `episode_indices` to be *shorter* than `total_episodes`,
    # then any subsequent sets of `episode_indices` will be relative to the new shorter length.
    # TODO: Submit a bug report and a fix.
    test_data.episode_indices = np.arange(val_split, train_data.total_episodes)
    val_data.episode_indices = np.arange(train_split, val_split)
    train_data.episode_indices = np.arange(0, train_split)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def tokenize(tokenizer, sample):
    observation_tokens = [
        tokenizer.encode_continuous(torch.from_numpy(observation))
        for observation in sample.observations["observation"][:-1]
    ]
    observation_tokens, observation_min, observation_max = zip(*observation_tokens)
    goal_tokens = [
        tokenizer.encode_continuous(torch.from_numpy(goal))
        for goal in sample.observations["desired_goal"][:-1]
    ]
    goal_tokens, goal_min, goal_max = zip(*goal_tokens)
    action_tokens = [
        tokenizer.encode_continuous(torch.from_numpy(action))
        for action in sample.actions
    ]
    action_tokens, action_min, action_max = zip(*action_tokens)
    action_tokens = [
        torch.concat([tokenizer.encode_discrete([tokenizer.separator]), action])
        for action in action_tokens
    ]

    goal = torch.stack(goal_tokens)
    observation = torch.stack(observation_tokens)
    action = torch.stack(action_tokens)
    xs = Timesteps(
        {
            "goal": goal,
            "observation": observation,
            "action": action[:, :-1],
        }
    )
    ys = Timesteps(
        {
            "goal": torch.zeros_like(goal),
            "observation": torch.zeros_like(observation),
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
        collate_fn=partial(generic_collate_fn, sequence_length=block_size, mask_keys=["action"]),
    )


def create_infinite_dataloader(tokenizer, batch_size, split="train", block_size=1024):
    dataset = initialize()
    dataset = TransformDataset(dataset[split], partial(tokenize, tokenizer))
    return infinite_dataloader(
        partial(DataLoader, dataset, batch_size=batch_size, collate_fn=partial(generic_collate_fn, sequence_length=block_size, mask_keys=["action"]))
    )
