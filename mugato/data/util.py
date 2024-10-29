import os
from pathlib import Path
from torch.data.utils import Dataset

xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

def generic_collate_fn(batch, sequence_length=1024):
    sliced = [
        (slice_to_context_window(sequence_length, xs), slice_to_context_window(sequence_length, ys))
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
            result[k] = sample[k][n:n+m]
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
            pad = (0, 0, 0, token_length - sample[k].size(1), 0, episode_length - sample[k].size(0))
            padded[k] = padded.get(k, [])
            padded[k].append(F.pad(sample[k], pad, value=0))
    return Timesteps([
        (k, torch.stack(v))
        for k, v in padded.items()
    ])

def mask(batch):
    result = Timesteps()
    for k, v in batch[0].items():
        episode_lengths = [sample[k].size(0) for sample in batch]
        token_lengths = [sample[k].size(1) for sample in batch]
        result[k] = torch.zeros(len(batch), max(episode_lengths), max(token_lengths))
        for i in range(len(batch)):
            result[k][i][:episode_lengths[i], :token_lengths[i]] = 1
    return result
