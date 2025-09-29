"""
Main entry point for I/O utilities.

This module provides easy access to all I/O functions for reading and writing
medical images and volumes.

Usage:
    from connectomics.data.io_utils import read_volume, write_volume
"""

# Core I/O functions
from .io.io import (
    read_volume,
    write_volume,
    read_image,
    write_image,
    get_volume_info,
)

# Visualization utilities
from .io.visualize import (
    visualize_volume,
    visualize_slices,
    save_volume_visualization,
)

# Tile utilities
from .io.tiles import (
    TileManager,
    create_tile_dataset,
    load_tile_config,
)

# Constants
from .io.const import *

__all__ = [
    # Core I/O
    'read_volume',
    'write_volume',
    'read_image',
    'write_image',
    'get_volume_info',

    # Visualization
    'visualize_volume',
    'visualize_slices',
    'save_volume_visualization',

    # Tiles
    'TileManager',
    'create_tile_dataset',
    'load_tile_config',
]