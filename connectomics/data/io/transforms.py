"""
MONAI transforms for connectomics I/O operations.

This module provides MONAI-compatible transforms for:
- Volume loading (HDF5, TIFF, PNG)
- Volume saving
- Tile-based loading for large datasets
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from .io import read_volume, save_volume
from .tiles import reconstruct_volume_from_tiles


class LoadVolumed(MapTransform):
    """MONAI loader for connectomics volume data.

    Loads HDF5, TIFF, PNG, NIfTI files and ensures
    channel-first format with a channel dimension.

    Args:
        keys: Keys to load from the data dictionary.
        transpose_axes: Axis permutation for spatial dims
            (e.g., [2,1,0] for xyz->zyx). Applied BEFORE
            adding channel dimension.
        allow_missing_keys: Allow missing keys.
    """

    def __init__(
        self,
        keys: KeysCollection,
        transpose_axes: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.transpose_axes = list(transpose_axes) if transpose_axes else []

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d or not isinstance(d[key], str):
                continue
            source_path = d[key]
            volume = read_volume(source_path)

            if self.transpose_axes:
                if volume.ndim == 3:
                    volume = np.transpose(volume, self.transpose_axes)
                elif volume.ndim == 4:
                    spatial_t = [i + 1 for i in self.transpose_axes]
                    volume = np.transpose(volume, [0] + spatial_t)

            # Ensure channel dim: (H,W)->(1,H,W),
            # (D,H,W)->(1,D,H,W)
            if volume.ndim in (2, 3):
                volume = np.expand_dims(volume, axis=0)

            d[key] = volume
            meta_key = f"{key}_meta_dict"
            meta_dict = dict(d.get(meta_key, {}))
            meta_dict.update(
                {
                    "filename_or_obj": source_path,
                    "original_shape": tuple(volume.shape),
                    "spatial_shape": tuple(volume.shape[1:]),
                    "channels_first": True,
                    "transpose_axes": (self.transpose_axes if self.transpose_axes else None),
                }
            )
            d[meta_key] = meta_dict
        return d


class SaveVolumed(MapTransform):
    """MONAI transform for saving volume data.

    Args:
        keys: Keys to save from the data dictionary.
        output_dir: Output directory.
        output_format: File format ('h5' or 'png').
        allow_missing_keys: Allow missing keys.
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        output_format: str = "h5",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_format = output_format

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], np.ndarray):
                fn = os.path.join(
                    self.output_dir,
                    f"{key}.{self.output_format}",
                )
                save_volume(
                    fn,
                    d[key],
                    file_format=self.output_format,
                )
        return d


class TileLoaderd(MapTransform):
    """MONAI transform for loading tile-based data.

    Reconstructs volumes from tiles based on chunk
    coordinates and metadata.

    Args:
        keys: Keys to process.
        allow_missing_keys: Allow missing keys.
    """

    def __init__(
        self,
        keys: Sequence[str],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], dict):
                ti = d[key]
                if "metadata" in ti and "chunk_coords" in ti:
                    d[key] = self._load_tiles_for_chunk(ti["metadata"], ti["chunk_coords"])
        return d

    def _load_tiles_for_chunk(
        self,
        metadata: Dict[str, Any],
        coords: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        z0, z1, y0, y1, x0, x1 = coords
        tile_paths = metadata["image"][z0:z1]
        volume_coords = [z0, z1, y0, y1, x0, x1]
        tile_coords = [
            0,
            metadata["depth"],
            0,
            metadata["height"],
            0,
            metadata["width"],
        ]
        return reconstruct_volume_from_tiles(
            tile_paths=tile_paths,
            volume_coords=volume_coords,
            tile_coords=tile_coords,
            tile_size=metadata["tile_size"],
            data_type=np.dtype(metadata["dtype"]),
            tile_start=metadata.get("tile_st"),
            tile_ratio=metadata.get("tile_ratio", 1.0),
        )
