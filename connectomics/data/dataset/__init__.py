"""
MONAI-native dataset module for PyTorch Connectomics.

This module provides MONAI-based dataset classes and PyTorch Lightning DataModules
for connectomics data loading. All legacy dataset classes have been removed.
"""

# MONAI base datasets
from .dataset_base import (
    MonaiConnectomicsDataset,
    MonaiCachedConnectomicsDataset,
    MonaiPersistentConnectomicsDataset,
)

# Volume datasets
from .dataset_volume import (
    MonaiVolumeDataset,
    MonaiCachedVolumeDataset,
)

# Inference datasets
from .dataset_inference import (
    InferenceVolumeDataset,
)

# Tile datasets
from .dataset_tile import (
    MonaiTileDataset,
    MonaiCachedTileDataset,
)

# Multi-dataset utilities
from .dataset_multi import (
    WeightedConcatDataset,
    StratifiedConcatDataset,
    UniformConcatDataset,
)

# Dataset factory functions (builder pattern)
from .build import (
    create_data_dicts_from_paths,
    create_volume_data_dicts,
    create_tile_data_dicts_from_json,
    create_connectomics_dataset,
    create_volume_dataset,
    create_tile_dataset,
)

__all__ = [
    # Base MONAI datasets
    'MonaiConnectomicsDataset',
    'MonaiCachedConnectomicsDataset',
    'MonaiPersistentConnectomicsDataset',

    # Volume datasets
    'MonaiVolumeDataset',
    'MonaiCachedVolumeDataset',

    # Inference datasets
    'InferenceVolumeDataset',

    # Tile datasets
    'MonaiTileDataset',
    'MonaiCachedTileDataset',

    # Multi-dataset utilities
    'WeightedConcatDataset',
    'StratifiedConcatDataset',
    'UniformConcatDataset',

    # Factory functions (from build.py)
    'create_data_dicts_from_paths',
    'create_volume_data_dicts',
    'create_tile_data_dicts_from_json',
    'create_connectomics_dataset',
    'create_volume_dataset',
    'create_tile_dataset',
]