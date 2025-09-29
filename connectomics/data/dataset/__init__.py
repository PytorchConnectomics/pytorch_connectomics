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
    create_data_dicts_from_paths,
    create_connectomics_dataset,
)

# Volume datasets
from .dataset_volume import (
    MonaiVolumeDataset,
    MonaiCachedVolumeDataset,
    create_volume_dataset,
    create_volume_data_dicts,
)

# Tile datasets
from .dataset_tile import (
    MonaiTileDataset,
    MonaiCachedTileDataset,
    TileLoaderd,
    create_tile_dataset,
    create_tile_data_dicts_from_json,
)

__all__ = [
    # Base MONAI datasets
    'MonaiConnectomicsDataset',
    'MonaiCachedConnectomicsDataset',
    'MonaiPersistentConnectomicsDataset',
    'create_data_dicts_from_paths',
    'create_connectomics_dataset',

    # Volume datasets
    'MonaiVolumeDataset',
    'MonaiCachedVolumeDataset',
    'create_volume_dataset',
    'create_volume_data_dicts',

    # Tile datasets
    'MonaiTileDataset',
    'MonaiCachedTileDataset',
    'TileLoaderd',
    'create_tile_dataset',
    'create_tile_data_dicts_from_json',
]