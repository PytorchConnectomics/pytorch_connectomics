"""
MONAI transforms for connectomics I/O operations.

This module provides MONAI-compatible transforms for:
- Volume loading (HDF5, TIFF, PNG)
- Volume saving
- Tile-based loading for large datasets
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from .io import read_volume, save_volume
from .tiles import reconstruct_volume_from_tiles


class LoadVolumed(MapTransform):
    """
    MONAI loader for connectomics volume data (HDF5, TIFF, etc.).

    This transform uses the connectomics read_volume function to load various
    file formats and ensures the data has a channel dimension.

    Args:
        keys: Keys to load from the data dictionary
        allow_missing_keys: Whether to allow missing keys in the dictionary

    Examples:
        >>> transform = LoadVolumed(keys=['image', 'label'])
        >>> data = {'image': 'img.h5', 'label': 'lbl.h5'}
        >>> result = transform(data)
        >>> # result['image'] shape: (C, D, H, W)
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load volume data from file paths."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], str):
                volume = read_volume(d[key])
                # Ensure we have at least 4 dimensions (add channel if needed)
                if volume.ndim == 3:
                    volume = np.expand_dims(volume, axis=0)
                d[key] = volume
        return d


class SaveVolumed(MapTransform):
    """
    MONAI transform for saving volume data.

    Args:
        keys: Keys to save from the data dictionary
        output_dir: Output directory for saved volumes
        output_format: File format ('h5' or 'png')
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = SaveVolumed(
        ...     keys=['prediction'],
        ...     output_dir='./outputs',
        ...     output_format='h5'
        ... )
        >>> data = {'prediction': np.random.rand(1, 32, 128, 128)}
        >>> result = transform(data)
        >>> # Saves to ./outputs/prediction.h5
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        output_format: str = 'h5',
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_format = output_format

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save volume data to files."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], np.ndarray):
                filename = os.path.join(self.output_dir, f"{key}.{self.output_format}")
                save_volume(filename, d[key], file_format=self.output_format)
        return d


class TileLoaderd(MapTransform):
    """
    MONAI transform for loading tile-based data.

    This transform reconstructs volumes from tiles based on chunk coordinates
    and metadata information.

    Args:
        keys: Keys to process from the data dictionary
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = TileLoaderd(keys=['image'])
        >>> data = {
        ...     'image': {
        ...         'metadata': tile_metadata,
        ...         'chunk_coords': (0, 10, 0, 128, 0, 128)
        ...     }
        ... }
        >>> result = transform(data)
        >>> # result['image'] is reconstructed volume from tiles
    """

    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load tile data for specified keys."""
        d = dict(data)

        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], dict):
                tile_info = d[key]
                if 'metadata' in tile_info and 'chunk_coords' in tile_info:
                    metadata = tile_info['metadata']
                    coords = tile_info['chunk_coords']
                    volume = self._load_tiles_for_chunk(metadata, coords)
                    d[key] = volume

        return d

    def _load_tiles_for_chunk(
        self,
        metadata: Dict[str, Any],
        coords: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        """Load and reconstruct volume chunk from tiles."""
        z_start, z_end, y_start, y_end, x_start, x_end = coords

        tile_paths = metadata['image'][z_start:z_end]
        volume_coords = [z_start, z_end, y_start, y_end, x_start, x_end]
        tile_coords = [
            0, metadata['depth'],
            0, metadata['height'],
            0, metadata['width']
        ]

        volume = reconstruct_volume_from_tiles(
            tile_paths=tile_paths,
            volume_coords=volume_coords,
            tile_coords=tile_coords,
            tile_size=metadata['tile_size'],
            data_type=np.dtype(metadata['dtype']),
            tile_start=metadata.get('tile_st', [0, 0]),
            tile_ratio=metadata.get('tile_ratio', 1.0),
        )

        return volume


__all__ = [
    'LoadVolumed',
    'SaveVolumed',
    'TileLoaderd',
]
