"""
Dataset module for PyTorch Connectomics.

Provides patch-sampling datasets for volumetric EM data:
- CachedVolumeDataset: loads volumes into RAM, crops with numpy
- LazyZarrVolumeDataset: lazy zarr reads (low memory)
- MonaiFilenameDataset: loads pre-tiled images from JSON
- Multi-dataset wrappers: Weighted, Stratified, Uniform concat
"""

# Base class
from .base import PatchDataset

# Shared helpers
from .data_dicts import create_data_dicts_from_paths
from .dataset_filename import (
    MonaiFilenameDataset,
    create_filename_datasets,
)

# Multi-dataset utilities
from .dataset_multi import (
    StratifiedConcatDataset,
    UniformConcatDataset,
    WeightedConcatDataset,
)

# Core datasets
from .dataset_volume_cached import CachedVolumeDataset, crop_volume
from .dataset_volume_zarr_lazy import LazyZarrVolumeDataset

# Sampling and splitting utilities (moved from data/utils/)
from .sampling import compute_total_samples, count_volume
from .split import split_volume_train_val

__all__ = [
    "PatchDataset",
    "CachedVolumeDataset",
    "LazyZarrVolumeDataset",
    "MonaiFilenameDataset",
    "WeightedConcatDataset",
    "StratifiedConcatDataset",
    "UniformConcatDataset",
    "create_data_dicts_from_paths",
    "create_filename_datasets",
    "crop_volume",
    "count_volume",
    "compute_total_samples",
    "split_volume_train_val",
]
