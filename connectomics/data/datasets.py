"""
Main entry point for connectomics datasets.

This module provides easy access to all dataset classes without
needing to import from deep nested paths.

Usage:
    from connectomics.data.datasets import VolumeDataset, TileDataset
"""

# Core dataset infrastructure
from .dataset.dataset_base import (
    BaseConnectomicsDataset,
    WeightedConcatDataset,
    create_multi_dataset,
)

# Specific dataset implementations
from .dataset.dataset_volume import VolumeDataset
from .dataset.dataset_tile import TileDataset
from .dataset.dataset_cloud import CloudVolumeDataset

# Utility functions
from .dataset.collate import collate_fn

__all__ = [
    # Core classes
    'BaseConnectomicsDataset',
    'VolumeDataset',
    'TileDataset',
    'CloudVolumeDataset',
    'WeightedConcatDataset',

    # Utilities
    'collate_fn',
    'create_multi_dataset',
]