"""
Test suite for multi-dataset utilities.

Tests cover:
- WeightedConcatDataset with different weight configurations
- StratifiedConcatDataset for balanced sampling
- UniformConcatDataset for uniform random sampling
- Edge cases and error handling
"""

import pytest
from torch.utils.data import Dataset

from connectomics.data.dataset import (
    WeightedConcatDataset,
    StratifiedConcatDataset,
    UniformConcatDataset,
)


class DummyDataset(Dataset):
    """Simple dataset that returns its index and a constant value."""

    def __init__(self, size: int, value: int = 0):
        self.size = size
        self.value = value

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'index': idx, 'value': self.value}


class TestWeightedConcatDataset:
    """Test suite for WeightedConcatDataset."""

    def test_basic_functionality(self):
        """Test basic weighted concatenation."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(200, value=2)

        weighted = WeightedConcatDataset(
            datasets=[dataset1, dataset2],
            weights=[0.7, 0.3],
            length=1000
        )

        assert len(weighted) == 1000

        # Sample many times and check distribution
        samples = [weighted[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)

        # Should be approximately 70/30 split (with some randomness)
        assert 650 < count_1 < 750, f"Expected ~700, got {count_1}"
        assert 250 < count_2 < 350, f"Expected ~300, got {count_2}"

    def test_equal_weights(self):
        """Test with equal weights (50/50 split)."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(100, value=2)

        weighted = WeightedConcatDataset(
            datasets=[dataset1, dataset2],
            weights=[0.5, 0.5],
            length=1000
        )

        samples = [weighted[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)

        # Should be approximately equal
        assert 450 < count_1 < 550
        assert 450 < count_2 < 550

    def test_extreme_weights(self):
        """Test with extreme weights (95/5 split)."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(100, value=2)

        weighted = WeightedConcatDataset(
            datasets=[dataset1, dataset2],
            weights=[0.95, 0.05],
            length=1000
        )

        samples = [weighted[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)

        # Should be approximately 95/5 split
        assert 900 < count_1 < 1000
        assert 0 < count_2 < 100

    def test_default_length(self):
        """Test default length (minimum dataset size)."""
        dataset1 = DummyDataset(50, value=1)
        dataset2 = DummyDataset(100, value=2)

        weighted = WeightedConcatDataset(
            datasets=[dataset1, dataset2],
            weights=[0.5, 0.5]
        )

        # Should use minimum length
        assert len(weighted) == 50

    def test_multiple_datasets(self):
        """Test with more than 2 datasets."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(100, value=2)
        dataset3 = DummyDataset(100, value=3)

        weighted = WeightedConcatDataset(
            datasets=[dataset1, dataset2, dataset3],
            weights=[0.5, 0.3, 0.2],
            length=1000
        )

        samples = [weighted[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)
        count_3 = values.count(3)

        # Check approximate distribution
        assert 450 < count_1 < 550
        assert 250 < count_2 < 350
        assert 150 < count_3 < 250

    def test_invalid_weights_sum(self):
        """Test error when weights don't sum to 1.0."""
        dataset1 = DummyDataset(100)
        dataset2 = DummyDataset(100)

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            WeightedConcatDataset(
                datasets=[dataset1, dataset2],
                weights=[0.6, 0.5]  # Sums to 1.1
            )

    def test_mismatched_lengths(self):
        """Test error when datasets and weights have different lengths."""
        dataset1 = DummyDataset(100)
        dataset2 = DummyDataset(100)

        with pytest.raises(ValueError, match="Number of datasets.*must match"):
            WeightedConcatDataset(
                datasets=[dataset1, dataset2],
                weights=[0.5, 0.3, 0.2]  # 3 weights for 2 datasets
            )


class TestStratifiedConcatDataset:
    """Test suite for StratifiedConcatDataset."""

    def test_round_robin_sampling(self):
        """Test round-robin sampling pattern."""
        dataset1 = DummyDataset(5, value=1)
        dataset2 = DummyDataset(5, value=2)

        stratified = StratifiedConcatDataset([dataset1, dataset2])

        # Should alternate between datasets
        samples = [stratified[i] for i in range(10)]
        values = [s['value'] for s in samples]

        # Pattern should be: 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
        expected = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        assert values == expected

    def test_unequal_dataset_sizes(self):
        """Test with datasets of different sizes."""
        dataset1 = DummyDataset(3, value=1)
        dataset2 = DummyDataset(5, value=2)

        stratified = StratifiedConcatDataset(
            datasets=[dataset1, dataset2],
            length=10
        )

        samples = [stratified[i] for i in range(10)]
        values = [s['value'] for s in samples]

        # Should alternate, with wrapping for smaller dataset
        assert values == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    def test_default_length(self):
        """Test default length (sum of dataset lengths)."""
        dataset1 = DummyDataset(50, value=1)
        dataset2 = DummyDataset(30, value=2)

        stratified = StratifiedConcatDataset([dataset1, dataset2])

        # Should use sum of lengths
        assert len(stratified) == 80

    def test_three_datasets(self):
        """Test with three datasets."""
        dataset1 = DummyDataset(3, value=1)
        dataset2 = DummyDataset(3, value=2)
        dataset3 = DummyDataset(3, value=3)

        stratified = StratifiedConcatDataset([dataset1, dataset2, dataset3])

        samples = [stratified[i] for i in range(9)]
        values = [s['value'] for s in samples]

        # Should cycle through all three
        assert values == [1, 2, 3, 1, 2, 3, 1, 2, 3]

    def test_empty_datasets_error(self):
        """Test error with no datasets."""
        with pytest.raises(ValueError, match="Must provide at least one dataset"):
            StratifiedConcatDataset([])


class TestUniformConcatDataset:
    """Test suite for UniformConcatDataset."""

    def test_uniform_sampling(self):
        """Test uniform random sampling."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(100, value=2)

        uniform = UniformConcatDataset([dataset1, dataset2], length=1000)

        samples = [uniform[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)

        # Should be approximately equal (each has 100/200 = 50% probability)
        assert 450 < count_1 < 550
        assert 450 < count_2 < 550

    def test_unequal_sizes_proportional(self):
        """Test that larger datasets get proportionally more samples."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(300, value=2)  # 3x larger

        uniform = UniformConcatDataset([dataset1, dataset2], length=1000)

        samples = [uniform[i] for i in range(1000)]
        values = [s['value'] for s in samples]

        count_1 = values.count(1)
        count_2 = values.count(2)

        # Should be approximately 25/75 split (100/400 and 300/400)
        assert 200 < count_1 < 300
        assert 700 < count_2 < 800

    def test_default_length(self):
        """Test default length (sum of dataset lengths)."""
        dataset1 = DummyDataset(50, value=1)
        dataset2 = DummyDataset(30, value=2)

        uniform = UniformConcatDataset([dataset1, dataset2])

        assert len(uniform) == 80

    def test_empty_datasets_error(self):
        """Test error with no datasets."""
        with pytest.raises(ValueError, match="Must provide at least one dataset"):
            UniformConcatDataset([])


class TestIntegration:
    """Integration tests for multi-dataset utilities."""

    def test_all_strategies_compatible(self):
        """Test that all strategies work with same datasets."""
        dataset1 = DummyDataset(100, value=1)
        dataset2 = DummyDataset(100, value=2)

        # All should work without errors
        weighted = WeightedConcatDataset([dataset1, dataset2], [0.5, 0.5], 100)
        stratified = StratifiedConcatDataset([dataset1, dataset2], 100)
        uniform = UniformConcatDataset([dataset1, dataset2], 100)

        assert len(weighted) == 100
        assert len(stratified) == 100
        assert len(uniform) == 100

    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset1 = DummyDataset(50, value=1)
        dataset2 = DummyDataset(50, value=2)

        weighted = WeightedConcatDataset([dataset1, dataset2], [0.5, 0.5], 100)

        loader = DataLoader(weighted, batch_size=10, shuffle=False)

        batches = list(loader)
        assert len(batches) == 10  # 100 samples / batch_size 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
