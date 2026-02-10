"""
I/O utilities for PyTorch Connectomics.

This package provides comprehensive I/O functionality for various data formats
commonly used in connectomics research.

Organization:
    io.py              - All format-specific I/O (HDF5, images, pickle, volume)
    monai_transforms.py - MONAI-compatible data loading transforms
    tiles.py           - Tile-based operations for large-scale datasets
    utils.py           - Utility functions for data processing
"""

# Core I/O functions
from .io import (
    # HDF5 I/O
    read_hdf5,
    write_hdf5,
    list_hdf5_datasets,
    # Image I/O
    read_image,
    read_images,
    read_image_as_volume,
    save_image,
    save_images,
    SUPPORTED_IMAGE_FORMATS,
    # Pickle I/O
    read_pickle_file,
    write_pickle_file,
    # High-level volume I/O
    read_volume,
    save_volume,
    get_vol_shape,
)

# Tile operations
from .tiles import (
    create_tile_metadata,
    reconstruct_volume_from_tiles,
)

# Utilities
from .utils import (
    seg_to_rgb,
    rgb_to_seg,
    normalize_data_range,
    convert_to_uint8,
    split_multichannel_mask,
    squeeze_arrays,
)

# MONAI transforms
from .monai_transforms import (
    LoadVolumed,
    NNUNetPreprocessd,
    SaveVolumed,
    TileLoaderd,
)

__all__ = [
    # HDF5 I/O
    "read_hdf5",
    "write_hdf5",
    "list_hdf5_datasets",
    # Image I/O
    "read_image",
    "read_images",
    "read_image_as_volume",
    "save_image",
    "save_images",
    "SUPPORTED_IMAGE_FORMATS",
    # Pickle I/O
    "read_pickle_file",
    "write_pickle_file",
    # High-level volume I/O
    "read_volume",
    "save_volume",
    "get_vol_shape",
    # Tile operations
    "create_tile_metadata",
    "reconstruct_volume_from_tiles",
    # Utilities
    "seg_to_rgb",
    "rgb_to_seg",
    "normalize_data_range",
    "convert_to_uint8",
    "split_multichannel_mask",
    "squeeze_arrays",
    # MONAI transforms
    "LoadVolumed",
    "NNUNetPreprocessd",
    "SaveVolumed",
    "TileLoaderd",
]
