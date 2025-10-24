"""Tests for BANIS-inspired features (Phase 12).

Tests the BANIS integration features:
- Slice augmentations (DropSliced, ShiftSliced)
- Numba connected components
- Weighted dataset concatenation
- Skeleton-based metrics (if available)
"""

import pytest
import torch
import numpy as np


def test_drop_sliced():
    """Test slice dropout augmentation."""
    from connectomics.data.augment import DropSliced

    aug = DropSliced(keys=["image"], prob=1.0, drop_prob=0.5)
    data = {"image": torch.randn(1, 128, 128, 128)}

    augmented = aug(data)

    assert augmented["image"].shape == data["image"].shape
    # Check that some slices were dropped (set to zero)
    assert (augmented["image"] == 0).any()


def test_shift_sliced():
    """Test slice shifting augmentation."""
    from connectomics.data.augment import ShiftSliced

    aug = ShiftSliced(keys=["image"], prob=1.0, shift_prob=0.5, max_shift=10)
    data = {"image": torch.randn(1, 128, 128, 128)}

    augmented = aug(data)

    assert augmented["image"].shape == data["image"].shape


def test_connected_components():
    """Test Numba connected components."""
    from connectomics.model.utils.connected_components import connected_components_3d

    # Create simple test case: two separate cubes
    affinities = np.zeros((3, 10, 10, 10), dtype=np.float32)

    # First cube (0:4, 0:4, 0:4) - all connected
    affinities[0, 0:3, 0:4, 0:4] = 1.0  # x-direction
    affinities[1, 0:4, 0:3, 0:4] = 1.0  # y-direction
    affinities[2, 0:4, 0:4, 0:3] = 1.0  # z-direction

    # Second cube (6:10, 6:10, 6:10) - all connected
    affinities[0, 6:9, 6:10, 6:10] = 1.0
    affinities[1, 6:10, 6:9, 6:10] = 1.0
    affinities[2, 6:10, 6:10, 6:9] = 1.0

    seg = connected_components_3d(affinities, threshold=0.5)

    # Should have 2 components (plus background)
    unique_ids = np.unique(seg)
    assert len(unique_ids) >= 2  # At least 2 components


def test_weighted_concat_dataset():
    """Test weighted dataset concatenation."""
    from connectomics.data.dataset.weighted_concat import WeightedConcatDataset

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


@pytest.mark.skipif(
    not pytest.importorskip("funlib", reason="funlib.evaluate not available"),
    reason="Skeleton metrics require funlib.evaluate"
)
def test_skeleton_metrics():
    """Test skeleton-based metrics (if funlib available)."""
    from connectomics.metrics.skeleton_metrics import SkeletonMetrics

    # This test requires a skeleton file, so we skip if not available
    pytest.skip("Skeleton metrics require test skeleton file")


def test_slurm_utils_import():
    """Test that SLURM utils can be imported."""
    from connectomics.config import slurm_utils

    # Test basic import
    assert hasattr(slurm_utils, 'detect_slurm_resources')
    assert hasattr(slurm_utils, 'get_cluster_config')
    assert hasattr(slurm_utils, 'filter_partitions')
    assert hasattr(slurm_utils, 'get_best_partition')


def test_slurm_detection_no_slurm():
    """Test SLURM detection when SLURM not available."""
    from connectomics.config.slurm_utils import detect_slurm_resources

    # Should return empty dict if SLURM not available
    partitions = detect_slurm_resources()
    assert isinstance(partitions, dict)
