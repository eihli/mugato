"""Combined dataloader implementations for proportional sampling."""

import random
import warnings
from collections.abc import Iterator, Sequence, Sized
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader


class DataLoaderLike(Protocol):
    """Protocol for objects that behave like DataLoaders."""
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...


class LimitedDataLoader:
    """Wrapper that limits a dataloader to a specific number of samples.

    Only supports DataLoaders with map-style datasets (i.e., datasets with __len__).
    Iterable-style datasets are not supported.

    Args:
        dataloader: The underlying dataloader to limit (must have a map-style dataset)
        limit: Maximum number of samples to yield
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducible shuffling
    """

    def __init__(
        self,
        dataloader: DataLoader[Any],
        limit: int,
        shuffle: bool = False,
        seed: int | None = None
    ):
        self.dataloader = dataloader
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed

        # Validate that we have a map-style dataset
        if not hasattr(dataloader, 'dataset'):
            raise ValueError(
                "DataLoader must have a dataset attribute. "
                "Only map-style datasets are supported."
            )

        self.dataset = dataloader.dataset

        # Verify the dataset has a length (is map-style, not iterable-style)
        if not isinstance(self.dataset, Sized):
            raise ValueError(
                "Dataset must be a map-style dataset with a __len__ method. "
                "Iterable-style datasets are not supported."
            )

        # Store the dataset size after validation
        self.dataset_size = len(self.dataset)

        # Create generator for reproducibility
        self.generator = None
        if self.seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

    def __iter__(self) -> Iterator[Any]:
        """Yield samples from the dataloader up to the limit."""
        # Calculate effective number of samples
        n_samples = min(self.limit, self.dataset_size)

        if self.shuffle:
            # Use PyTorch's RandomSampler for shuffling
            # Create indices using torch.randperm for better performance
            if self.generator is not None:
                indices = torch.randperm(
                    self.dataset_size, generator=self.generator
                )[:n_samples]
            else:
                indices = torch.randperm(self.dataset_size)[:n_samples]

            # Convert to list for use as sampler
            sampler = indices.tolist()
        else:
            # Use sequential indices up to the limit
            sampler = list(range(n_samples))

        # Create a new dataloader with our sampler
        temp_loader = DataLoader(
            self.dataset,
            batch_size=self.dataloader.batch_size,
            sampler=sampler,
            num_workers=0,  # Avoid multiprocessing complications
            collate_fn=self.dataloader.collate_fn,
            pin_memory=self.dataloader.pin_memory,
            drop_last=self.dataloader.drop_last,
        )

        yield from temp_loader

    def __len__(self) -> int:
        """Return the limited length."""
        return min(self.limit, self.dataset_size)


class ProportionalCombinedDataLoader:
    """Combines multiple dataloaders and samples from them proportionally.

    Ensures that one epoch goes through all samples from all dataloaders.
    Dataloaders are sampled proportionally based on their lengths, and
    exhausted dataloaders are removed from sampling.

    Only supports DataLoaders with map-style datasets (i.e., datasets with __len__).
    Iterable-style datasets are not supported.

    Args:
        dataloaders: List of dataloaders to combine (must have map-style datasets)
    """

    def __init__(self, dataloaders: Sequence[DataLoader[Any] | DataLoaderLike]):
        if not dataloaders:
            raise ValueError("Must provide at least one dataloader")

        # Validate all dataloaders have map-style datasets
        for i, dl in enumerate(dataloaders):
            if not hasattr(dl, 'dataset'):
                raise ValueError(
                    f"DataLoader at index {i} must have a dataset attribute. "
                    "Only map-style datasets are supported."
                )
            if not isinstance(dl.dataset, Sized):
                raise ValueError(
                    f"Dataset at index {i} must be a map-style dataset with "
                    "a __len__ method. Iterable-style datasets are not supported."
                )

        self.dataloaders = list(dataloaders)  # Convert to list for internal use
        self.lengths = [len(dl) for dl in self.dataloaders]
        self.total_batches = sum(self.lengths)

        # Calculate sampling probabilities based on lengths
        self.probabilities = [length / self.total_batches for length in self.lengths]

    def __iter__(self) -> Iterator[Any]:
        """Yield batches from dataloaders proportionally."""
        # Create iterators for each dataloader
        iterators = [iter(dl) for dl in self.dataloaders]
        remaining = list(range(len(self.dataloaders)))
        batches_yielded = [0] * len(self.dataloaders)

        while remaining:
            # Calculate current probabilities based on remaining batches
            remaining_batches = [
                self.lengths[i] - batches_yielded[i]
                for i in remaining
            ]
            total_remaining = sum(remaining_batches)

            if total_remaining == 0:
                break

            probs = [b / total_remaining for b in remaining_batches]

            # Sample a dataloader index
            idx_in_remaining = random.choices(range(len(remaining)), weights=probs)[0]
            actual_idx = remaining[idx_in_remaining]

            try:
                # Get next batch from selected dataloader
                batch = next(iterators[actual_idx])
                batches_yielded[actual_idx] += 1
                yield batch

                # Remove exhausted dataloaders
                if batches_yielded[actual_idx] >= self.lengths[actual_idx]:
                    remaining.remove(actual_idx)

            except StopIteration:
                # This dataloader is exhausted
                remaining.remove(actual_idx)

    def __len__(self) -> int:
        """Return total number of batches across all dataloaders."""
        return self.total_batches


def create_weighted_dataloaders(
    dataloaders: list[DataLoader[Any]],
    weights: list[float],
    shuffle: bool = False,
    seed: int | None = None
) -> list[LimitedDataLoader]:
    """Create limited dataloaders based on weights.

    Given a list of dataloaders and weights, returns limited dataloaders
    where the number of samples from each is proportional to the weights.

    Only supports DataLoaders with map-style datasets (i.e., datasets with __len__).
    Iterable-style datasets are not supported.

    Note: If a dataloader has fewer samples than its weighted allocation
    would require, it will be capped at its actual size. This means the
    final distribution may not exactly match the requested weights if
    some datasets are too small.

    Args:
        dataloaders: List of dataloaders (must have map-style datasets)
        weights: List of weights (same length as dataloaders)
        shuffle: Whether to shuffle each limited dataloader
        seed: Base random seed (each dataloader gets seed + index if provided)

    Returns:
        List of LimitedDataLoader instances
    """
    if len(dataloaders) != len(weights):
        raise ValueError("Number of dataloaders must match number of weights")

    if not dataloaders:
        raise ValueError("Must provide at least one dataloader")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Validate all dataloaders have map-style datasets and get sizes
    dataset_sizes = []
    for i, dl in enumerate(dataloaders):
        if not hasattr(dl, 'dataset'):
            raise ValueError(
                f"DataLoader at index {i} must have a dataset attribute. "
                "Only map-style datasets are supported."
            )
        if not isinstance(dl.dataset, Sized):
            raise ValueError(
                f"Dataset at index {i} must be a map-style dataset with "
                "a __len__ method. Iterable-style datasets are not supported."
            )
        dataset_sizes.append(len(dl.dataset))

    # Calculate total samples across all dataloaders
    total_samples = sum(dataset_sizes)

    # Create limited dataloaders
    limited_dataloaders = []
    actual_sizes = []
    requested_sizes = []

    for i, (dl, weight, dataset_size) in enumerate(
        zip(dataloaders, normalized_weights, dataset_sizes, strict=False)
    ):
        # Calculate number of samples for this dataloader
        n_samples = int(weight * total_samples)
        requested_sizes.append(n_samples)

        # Get actual dataset size
        max_samples = dataset_size

        # Check if dataset is too small for requested allocation
        if n_samples > max_samples:
            # Calculate what percentage this dataset can actually provide
            actual_percentage = (max_samples / total_samples) * 100
            requested_percentage = weight * 100

            # Simple calculation: to get X% of total, need X% of total samples
            # This is the minimum size this dataset would need
            min_size_needed = n_samples

            warnings.warn(
                f"Dataset {i}: Size constraint prevents requested allocation\n"
                f"  Current size: {max_samples} samples\n"
                f"  Requested: {n_samples} samples "
                f"({requested_percentage:.1f}% of total)\n"
                f"  Maximum possible: {actual_percentage:.1f}% of total\n"
                f"  Minimum size needed: {min_size_needed} samples",
                stacklevel=2
            )

        n_samples = min(n_samples, max_samples)
        actual_sizes.append(n_samples)

        # Create limited dataloader with optional unique seed
        dl_seed = None if seed is None else seed + i
        limited_dl = LimitedDataLoader(
            dl,
            limit=n_samples,
            shuffle=shuffle,
            seed=dl_seed
        )
        limited_dataloaders.append(limited_dl)

    # Check if final distribution significantly differs from requested
    actual_total = sum(actual_sizes)
    if actual_total > 0:
        actual_percentages = [s / actual_total * 100 for s in actual_sizes]
        requested_percentages = [w * 100 for w in normalized_weights]

        # Log summary if any allocation differs by more than 5%
        significant_diff = any(
            abs(actual - requested) > 5
            for actual, requested in zip(
                actual_percentages, requested_percentages, strict=False
            )
        )

        if significant_diff:
            req_str = ", ".join(f"{p:.1f}%" for p in requested_percentages)
            act_str = ", ".join(f"{p:.1f}%" for p in actual_percentages)
            warnings.warn(
                f"Final distribution differs significantly from requested weights:\n"
                f"  Requested: [{req_str}]\n"
                f"  Actual:    [{act_str}]\n"
                f"  Note: This happens when datasets are too small "
                f"for their allocations",
                stacklevel=2
            )

    return limited_dataloaders


# Convenience function to create a proportional combined dataloader from weighted inputs
def create_proportional_dataloader(
    dataloaders: list[DataLoader[Any]],
    weights: list[float] | None = None,
    shuffle: bool = False,
    seed: int | None = None
) -> ProportionalCombinedDataLoader:
    """Convenience function to create a proportional combined dataloader.

    Only supports DataLoaders with map-style datasets (i.e., datasets with __len__).
    Iterable-style datasets are not supported.

    Args:
        dataloaders: List of dataloaders to combine (must have map-style datasets)
        weights: Optional weights for sampling (defaults to equal weights)
        shuffle: Whether to shuffle individual dataloaders
        seed: Random seed for reproducible shuffling

    Returns:
        ProportionalCombinedDataLoader that samples from the dataloaders
    """
    if weights is not None:
        # Use weights to create limited dataloaders
        limited_dls = create_weighted_dataloaders(
            dataloaders, weights, shuffle=shuffle, seed=seed
        )
        return ProportionalCombinedDataLoader(limited_dls)
    else:
        # Use dataloaders as-is for proportional sampling
        return ProportionalCombinedDataLoader(dataloaders)
