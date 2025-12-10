"""Tests for BANIS-inspired features (Phase 12).

Tests the BANIS-inspired utilities that remain in the codebase:
- Affinity connected components decoding
- Weighted dataset concatenation helpers
"""

import numpy as np
import torch


def test_connected_components():
    """Test affinity connected components using the current decoding API."""
    from connectomics.decoding.segmentation import decode_affinity_cc

    # Construct two disjoint 2x2x2 cubes
    affinities = np.zeros((3, 8, 8, 8), dtype=np.float32)

    # Cube A at origin
    affinities[:, 1:3, 1:3, 1:3] = 1.0
    # Cube B offset
    affinities[:, 5:7, 5:7, 5:7] = 1.0

    seg = decode_affinity_cc(affinities, threshold=0.5)

    unique_ids = np.unique(seg)
    # Expect background + two components
    assert len(unique_ids) == 3
    assert seg[1, 1, 1] != 0
    assert seg[5, 5, 5] != 0
    assert seg[1, 1, 1] != seg[5, 5, 5]


def test_weighted_concat_dataset():
    """Test weighted dataset concatenation."""
    from connectomics.data.dataset import WeightedConcatDataset

    # Create dummy datasets
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return self.value

    dataset1 = DummyDataset(1.0)
    dataset2 = DummyDataset(2.0)

    # 80% from dataset1, 20% from dataset2
    mixed = WeightedConcatDataset([dataset1, dataset2], weights=[0.8, 0.2])

    # Sample many times and check distribution
    samples = [mixed[i] for i in range(1000)]

    count_1 = sum(1 for s in samples if s == 1.0)
    count_2 = sum(1 for s in samples if s == 2.0)

    # Should be approximately 80/20 (with some randomness)
    assert 750 < count_1 < 850
    assert 150 < count_2 < 250


def test_slurm_utils_import():
    """Test that SLURM utils can be imported."""
    from connectomics.config import slurm_utils

    # Test basic import
    assert hasattr(slurm_utils, "detect_slurm_resources")
    assert hasattr(slurm_utils, "get_cluster_config")
    assert hasattr(slurm_utils, "filter_partitions")
    assert hasattr(slurm_utils, "get_best_partition")


def test_slurm_detection_no_slurm():
    """Test SLURM detection when SLURM not available."""
    from connectomics.config.slurm_utils import detect_slurm_resources

    # Should return empty dict if SLURM not available
    partitions = detect_slurm_resources()
    assert isinstance(partitions, dict)
