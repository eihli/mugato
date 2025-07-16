"""Tests for combined dataloader implementations."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mugato.data.combined_dataloader import (
    LimitedDataLoader,
    ProportionalCombinedDataLoader,
    create_proportional_dataloader,
    create_weighted_dataloaders,
)


class TestLimitedDataLoader:
    """Test cases for LimitedDataLoader."""

    def test_basic_limiting(self):
        """Test that dataloader correctly limits samples."""
        # Create a simple dataset with 100 samples
        data = torch.arange(100)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        # Limit to 50 samples
        limited_dl = LimitedDataLoader(dataloader, limit=50)

        # Collect all samples
        all_samples = []
        for batch in limited_dl:
            all_samples.extend(batch[0].tolist())

        assert len(all_samples) == 50
        # Should get first 50 samples without shuffle
        assert all_samples == list(range(50))

    def test_limiting_with_shuffle(self):
        """Test limiting with shuffle enabled."""
        # Create dataset
        data = torch.arange(100)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        # Limit with shuffle and seed for reproducibility
        limited_dl = LimitedDataLoader(dataloader, limit=50, shuffle=True, seed=42)

        # Collect samples
        all_samples = []
        for batch in limited_dl:
            all_samples.extend(batch[0].tolist())

        assert len(all_samples) == 50
        # Should not be in order due to shuffle
        assert all_samples != list(range(50))

        # Test reproducibility with same seed
        limited_dl2 = LimitedDataLoader(dataloader, limit=50, shuffle=True, seed=42)
        all_samples2 = []
        for batch in limited_dl2:
            all_samples2.extend(batch[0].tolist())

        assert all_samples == all_samples2

    def test_limit_exceeds_dataset(self):
        """Test when limit exceeds dataset size."""
        # Small dataset
        data = torch.arange(30)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        # Limit larger than dataset
        limited_dl = LimitedDataLoader(dataloader, limit=100)

        # Should only get 30 samples
        all_samples = []
        for batch in limited_dl:
            all_samples.extend(batch[0].tolist())

        assert len(all_samples) == 30


class TestProportionalCombinedDataLoader:
    """Test cases for ProportionalCombinedDataLoader."""

    def test_basic_combination(self):
        """Test combining multiple dataloaders."""
        # Create three datasets of different sizes
        data1 = torch.arange(50)
        data2 = torch.arange(100, 200)  # 100 samples
        data3 = torch.arange(300, 330)  # 30 samples

        dl1 = DataLoader(TensorDataset(data1), batch_size=10)
        dl2 = DataLoader(TensorDataset(data2), batch_size=10)
        dl3 = DataLoader(TensorDataset(data3), batch_size=10)

        # Combine them
        combined_dl = ProportionalCombinedDataLoader([dl1, dl2, dl3])

        # Collect all samples
        all_samples = []
        for batch in combined_dl:
            all_samples.extend(batch[0].tolist())

        # Should have all 180 samples
        assert len(all_samples) == 180

        # Check that we got samples from all datasets
        assert any(s < 50 for s in all_samples)  # From data1
        assert any(100 <= s < 200 for s in all_samples)  # From data2
        assert any(300 <= s < 330 for s in all_samples)  # From data3

    def test_proportional_sampling(self):
        """Test that sampling is roughly proportional to dataset sizes."""
        # Create datasets with known proportions
        # 20, 60, 20 samples (20%, 60%, 20%)
        data1 = torch.zeros(20)
        data2 = torch.ones(60)
        data3 = torch.full((20,), 2)

        dl1 = DataLoader(TensorDataset(data1), batch_size=10)
        dl2 = DataLoader(TensorDataset(data2), batch_size=10)
        dl3 = DataLoader(TensorDataset(data3), batch_size=10)

        combined_dl = ProportionalCombinedDataLoader([dl1, dl2, dl3])

        # Run multiple times to check rough proportions
        for _ in range(5):
            counts = {0: 0, 1: 0, 2: 0}
            batch_count = 0

            # Check first 10 batches for rough proportions
            for i, batch in enumerate(combined_dl):
                if i >= 10:
                    break
                batch_count += 1
                value = batch[0][0].item()
                counts[int(value)] += 1

            # We should see more samples from dl2 (60% of data)
            # This is probabilistic, so we just check it's the most common
            if batch_count == 10:  # Only check if we got full 10 batches
                assert counts[1] >= counts[0]
                assert counts[1] >= counts[2]

    def test_single_dataloader(self):
        """Test with a single dataloader."""
        data = torch.arange(50)
        dl = DataLoader(TensorDataset(data), batch_size=10)

        combined_dl = ProportionalCombinedDataLoader([dl])

        all_samples = []
        for batch in combined_dl:
            all_samples.extend(batch[0].tolist())

        assert len(all_samples) == 50
        assert all_samples == list(range(50))


class TestWeightedDataLoaders:
    """Test cases for weighted dataloader creation."""

    def test_create_weighted_dataloaders(self):
        """Test creating weighted dataloaders."""
        # Create three dataloaders
        data1 = torch.arange(100)
        data2 = torch.arange(100, 300)  # 200 samples
        data3 = torch.arange(300, 400)  # 100 samples

        dl1 = DataLoader(TensorDataset(data1), batch_size=10)
        dl2 = DataLoader(TensorDataset(data2), batch_size=10)
        dl3 = DataLoader(TensorDataset(data3), batch_size=10)

        # Create with weights [1, 2, 1] - so 25%, 50%, 25%
        weighted_dls = create_weighted_dataloaders(
            [dl1, dl2, dl3],
            weights=[1, 2, 1]
        )

        # Check that we get 3 limited dataloaders
        assert len(weighted_dls) == 3

        # Collect samples from each
        samples = []
        for wdl in weighted_dls:
            dl_samples = []
            for batch in wdl:
                dl_samples.extend(batch[0].tolist())
            samples.append(dl_samples)

        # Total samples should be 400
        total_samples = sum(len(s) for s in samples)
        assert total_samples == 400

        # Check rough proportions (25%, 50%, 25%)
        assert len(samples[0]) == 100  # 25% of 400
        assert len(samples[1]) == 200  # 50% of 400
        assert len(samples[2]) == 100  # 25% of 400

    def test_weights_exceed_data(self):
        """Test when weights request more data than available."""
        # Small datasets
        data1 = torch.arange(10)
        data2 = torch.arange(20)

        dl1 = DataLoader(TensorDataset(data1), batch_size=5)
        dl2 = DataLoader(TensorDataset(data2), batch_size=5)

        # Weights that would request more data
        weighted_dls = create_weighted_dataloaders(
            [dl1, dl2],
            weights=[90, 10]  # 90% of total would exceed dl1's size
        )

        # Collect samples
        samples1 = []
        samples2 = []
        for batch in weighted_dls[0]:
            samples1.extend(batch[0].tolist())
        for batch in weighted_dls[1]:
            samples2.extend(batch[0].tolist())

        # dl1 should be capped at its actual size
        assert len(samples1) == 10
        # dl2 should get remaining proportion
        assert len(samples2) <= 20


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_create_proportional_dataloader(self):
        """Test the convenience function with weights."""
        # Create dataloaders
        data1 = torch.arange(100)
        data2 = torch.arange(100, 300)
        data3 = torch.arange(300, 400)

        dl1 = DataLoader(TensorDataset(data1), batch_size=10)
        dl2 = DataLoader(TensorDataset(data2), batch_size=20)
        dl3 = DataLoader(TensorDataset(data3), batch_size=10)

        # Create with weights
        combined_dl = create_proportional_dataloader(
            [dl1, dl2, dl3],
            weights=[1, 2, 1],
            shuffle=True,
            seed=42
        )

        # Collect all samples
        all_samples = []
        for batch in combined_dl:
            all_samples.extend(batch[0].tolist())

        # Should have 400 total samples
        assert len(all_samples) == 400

        # Test without weights (should use natural proportions)
        combined_dl2 = create_proportional_dataloader([dl1, dl2, dl3])

        samples2 = []
        for batch in combined_dl2:
            samples2.extend(batch[0].tolist())

        # Should have all 400 samples
        assert len(samples2) == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
