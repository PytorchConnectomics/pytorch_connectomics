"""
MONAI-native dataset module for PyTorch Connectomics.

This module provides MONAI-based dataset classes and PyTorch Lightning DataModules
for connectomics data loading.

Inference Strategy:
    For sliding-window inference, use MONAI's SlidingWindowInferer in the Lightning
    module. Test datasets should return full volumes (no cropping). See
    .claude/INFERENCE_DESIGN.md for details.
"""

# Dataset factory functions (builder pattern)
from .build import (
    create_connectomics_dataset,
    create_tile_data_dicts_from_json,
    create_tile_dataset,
    create_volume_dataset,
)

# Shared data-dict helpers
from .data_dicts import (
    create_data_dicts_from_paths,
    create_volume_data_dicts,
)

# MONAI base datasets
from .dataset_base import (
    MonaiCachedConnectomicsDataset,
    MonaiConnectomicsDataset,
    MonaiPersistentConnectomicsDataset,
)

# Multi-dataset utilities
from .dataset_multi import (
    StratifiedConcatDataset,
    UniformConcatDataset,
    WeightedConcatDataset,
)

# Tile datasets
from .dataset_tile import (
    MonaiCachedTileDataset,
    MonaiTileDataset,
)

# Volume datasets
from .dataset_volume import (
    MonaiCachedVolumeDataset,
    MonaiVolumeDataset,
)

__all__ = [
    # Base MONAI datasets
    "MonaiConnectomicsDataset",
    "MonaiCachedConnectomicsDataset",
    "MonaiPersistentConnectomicsDataset",
    # Volume datasets
    "MonaiVolumeDataset",
    "MonaiCachedVolumeDataset",
    # Tile datasets
    "MonaiTileDataset",
    "MonaiCachedTileDataset",
    # Multi-dataset utilities
    "WeightedConcatDataset",
    "StratifiedConcatDataset",
    "UniformConcatDataset",
    # Factory functions (from build.py)
    "create_data_dicts_from_paths",
    "create_volume_data_dicts",
    "create_tile_data_dicts_from_json",
    "create_connectomics_dataset",
    "create_volume_dataset",
    "create_tile_dataset",
]
