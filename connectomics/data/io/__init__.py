"""
I/O utilities for PyTorch Connectomics.

This package provides comprehensive I/O functionality for various data formats
commonly used in connectomics research. All functions follow PEP 8 naming
conventions and include comprehensive type hints.

Modules:
    volume: Volume-based I/O operations (HDF5, TIFF, PNG)
    tiles: Tile-based operations for large-scale datasets
    utils: Utility functions for data processing and conversion
"""

# Volume I/O
from .volume import (
    read_hdf5, write_hdf5, read_volume, save_volume,
    read_image, read_images, read_image_as_volume
)

# Tile I/O
from .tiles import (
    create_tile_metadata, reconstruct_volume_from_tiles
)

# Utilities
from .utils import (
    read_pickle_file, vast_to_segmentation, normalize_data_range,
    convert_to_uint8, split_multichannel_mask, squeeze_arrays
)

__all__ = [
    # Volume I/O
    'read_hdf5', 'write_hdf5', 'read_volume', 'save_volume',
    'read_image', 'read_images', 'read_image_as_volume',

    # Tile I/O
    'create_tile_metadata', 'reconstruct_volume_from_tiles',

    # Utilities
    'read_pickle_file', 'vast_to_segmentation', 'normalize_data_range',
    'convert_to_uint8', 'split_multichannel_mask', 'squeeze_arrays'
]