"""
I/O utilities for PyTorch Connectomics.

Organization:
    io.py              - Format-specific I/O (HDF5, TIFF, PNG, NIfTI)
    transforms.py      - MONAI-compatible data loading transforms
    tiles.py           - Tile-based operations for large datasets
    utils.py           - RGB/seg conversion, mask splitting
"""

from .io import (
    get_vol_shape,
    read_hdf5,
    read_images,
    read_volume,
    save_volume,
    write_hdf5,
)
from .transforms import (
    LoadVolumed,
    SaveVolumed,
    TileLoaderd,
)
from .utils import (
    rgb_to_seg,
)

__all__ = [
    "read_hdf5",
    "write_hdf5",
    "read_images",
    "read_volume",
    "save_volume",
    "get_vol_shape",
    "LoadVolumed",
    "SaveVolumed",
    "TileLoaderd",
    "rgb_to_seg",
]
