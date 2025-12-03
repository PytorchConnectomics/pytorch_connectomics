"""
Multi-dataset utilities for mixing multiple data sources.

This module provides dataset classes for combining multiple datasets with
different sampling strategies, useful for domain adaptation and data mixing.

Reference: BANIS baseline (data.py)
Use cases:
- Mix synthetic and real data with controllable proportions
- Combine multiple datasets with equal representation
- Domain adaptation with weighted sampling
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset


class WeightedConcatDataset(Dataset):
    """
    Concatenate multiple datasets and sample from them with specified weights.

    Unlike torch.utils.data.ConcatDataset which samples proportionally to
    dataset sizes, this class samples according to specified weights. This is
    particularly useful for domain adaptation where you want to control the
    ratio of synthetic vs. real data regardless of dataset sizes.

    Args:
        datasets: List of datasets to concatenate
        weights: List of sampling weights (must sum to 1.0)
        length: Total number of samples per epoch. Default: minimum dataset length

    Example:
        >>> from connectomics.data.dataset import WeightedConcatDataset
        >>> synthetic_data = SyntheticDataset(size=10000)
        >>> real_data = RealDataset(size=1000)
        >>> # 80% synthetic, 20% real (regardless of actual sizes)
        >>> mixed = WeightedConcatDataset(
        ...     datasets=[synthetic_data, real_data],
        ...     weights=[0.8, 0.2],
        ...     length=5000  # 5000 samples per epoch
        ... )
        >>> # Each batch will be 80% synthetic, 20% real on average
    """

    def __init__(self, datasets: List[Dataset], weights: List[float], length: Optional[int] = None):
        if len(datasets) != len(weights):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of weights ({len(weights)})"
            )

        weights_sum = sum(weights)
        if abs(weights_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weights_sum}")

        self.datasets = datasets
        self.weights = np.array(weights, dtype=np.float32)

        # Default length: minimum dataset length
        if length is None:
            self.length = min(len(d) for d in datasets)
        else:
            self.length = length

    def __getitem__(self, index: int):
        """
        Sample from datasets according to weights.

        Note: The index parameter is ignored. Instead, we randomly select
        a dataset according to the weights and then randomly sample from it.
        """
        # Randomly select dataset according to weights
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)

        # Random sample from selected dataset
        sample_idx = np.random.randint(len(self.datasets[dataset_idx]))

        return self.datasets[dataset_idx][sample_idx]

    def __len__(self) -> int:
        return self.length


class StratifiedConcatDataset(Dataset):
    """
    Concatenate datasets with stratified (round-robin) sampling.

    Ensures balanced sampling across datasets by cycling through them.
    This is useful when you want equal representation from each dataset
    regardless of their actual sizes.

    Args:
        datasets: List of datasets to concatenate
        length: Total number of samples per epoch. Default: sum of dataset lengths

    Example:
        >>> from connectomics.data.dataset import StratifiedConcatDataset
        >>> dataset1 = Dataset1(size=100)
        >>> dataset2 = Dataset2(size=200)
        >>> stratified = StratifiedConcatDataset([dataset1, dataset2])
        >>> # Will sample: dataset1[0], dataset2[0], dataset1[1], dataset2[1], ...
        >>> # Ensures equal representation even though dataset2 is 2x larger
    """

    def __init__(self, datasets: List[Dataset], length: Optional[int] = None):
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.dataset_lengths = [len(d) for d in datasets]

        # Default length: sum of all dataset lengths
        if length is None:
            self.length = sum(self.dataset_lengths)
        else:
            self.length = length

    def __getitem__(self, index: int):
        """
        Sample from datasets in round-robin fashion.

        The index determines which dataset to sample from by cycling through
        datasets sequentially.
        """
        # Cycle through datasets (round-robin)
        dataset_idx = index % self.n_datasets

        # Sample index within the selected dataset (with wrapping)
        sample_idx = (index // self.n_datasets) % self.dataset_lengths[dataset_idx]

        return self.datasets[dataset_idx][sample_idx]

    def __len__(self) -> int:
        return self.length


class UniformConcatDataset(Dataset):
    """
    Concatenate datasets with uniform random sampling.

    Samples uniformly from all datasets combined, giving equal probability
    to each individual sample across all datasets. This is equivalent to
    WeightedConcatDataset with weights proportional to dataset sizes.

    Args:
        datasets: List of datasets to concatenate
        length: Total number of samples per epoch. Default: sum of dataset lengths

    Example:
        >>> from connectomics.data.dataset import UniformConcatDataset
        >>> dataset1 = Dataset1(size=100)
        >>> dataset2 = Dataset2(size=200)
        >>> uniform = UniformConcatDataset([dataset1, dataset2])
        >>> # Each sample has equal probability (1/300) regardless of source dataset
    """

    def __init__(self, datasets: List[Dataset], length: Optional[int] = None):
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)

        # Default length: sum of all dataset lengths
        if length is None:
            self.length = self.total_length
        else:
            self.length = length

    def __getitem__(self, index: int):
        """
        Sample uniformly from all datasets.

        Randomly selects a sample from the combined pool of all datasets.
        """
        # Random index in the combined dataset
        global_idx = np.random.randint(self.total_length)

        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], global_idx, side="right")

        # Local index within the selected dataset
        sample_idx = global_idx - self.cumulative_lengths[dataset_idx]

        return self.datasets[dataset_idx][sample_idx]

    def __len__(self) -> int:
        return self.length


__all__ = [
    "WeightedConcatDataset",
    "StratifiedConcatDataset",
    "UniformConcatDataset",
]
