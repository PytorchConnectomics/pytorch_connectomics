"""
Main entry point for Lightning DataModules.

This module provides easy access to all DataModule classes and factory functions.
"""

# Core DataModule infrastructure
from .dataset import (
    ConnectomicsDataModule,
    DataModule,
    create_volume_datamodule,
    create_tile_datamodule,
    create_cloud_datamodule,
    create_datamodule_from_config,
    create_datamodule_from_configs,
)

__all__ = [
    # Modern DataModules
    'ConnectomicsDataModule',
    'DataModule',

    # Factory functions
    'create_volume_datamodule',
    'create_tile_datamodule',
    'create_cloud_datamodule',
    'create_datamodule_from_config',
    'create_datamodule_from_configs',
]