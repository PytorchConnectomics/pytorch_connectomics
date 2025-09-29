"""
Main entry point for I/O utilities.

This module provides easy access to all consolidated I/O functions for reading and writing
connectomics data in various formats.

Usage:
    from connectomics.data.io_utils import read_volume, save_volume
"""

# Core I/O functions from the consolidated io module
from .io import (
    # Volume I/O
    read_hdf5,
    write_hdf5,
    read_volume,
    save_volume,
    read_image,
    read_images,
    read_image_as_volume,

    # Tile I/O
    create_tile_metadata,
    reconstruct_volume_from_tiles,

    # Utilities
    read_pickle_file,
    vast_to_segmentation,
    normalize_data_range,
    convert_to_uint8,
    split_multichannel_mask,
    squeeze_arrays,
)

__all__ = [
    # Volume I/O
    'read_hdf5',
    'write_hdf5',
    'read_volume',
    'save_volume',
    'read_image',
    'read_images',
    'read_image_as_volume',

    # Tile I/O
    'create_tile_metadata',
    'reconstruct_volume_from_tiles',

    # Utilities
    'read_pickle_file',
    'vast_to_segmentation',
    'normalize_data_range',
    'convert_to_uint8',
    'split_multichannel_mask',
    'squeeze_arrays',
]