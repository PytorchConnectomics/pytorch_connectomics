"""
MONAI-native tile dataset for PyTorch Connectomics.

This module provides tile-based dataset classes using MONAI's native dataset
infrastructure for large-scale connectomics data that cannot fit in memory.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple

from monai.data import CacheDataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from .dataset_base import MonaiConnectomicsDataset
from .tile_utils import (
    calculate_chunk_indices,
    create_chunk_data_dicts,
    create_default_tile_transforms,
    load_tile_metadata,
)
from ..io import TileLoaderd


class MonaiTileDataset(MonaiConnectomicsDataset):
    """
    MONAI-native dataset for large-scale tile-based connectomics data.

    This dataset is designed for large-scale volumetric datasets that are stored
    as individual tiles. It constructs smaller chunks for processing without loading
    the entire volume into memory.

    Args:
        volume_json (str): JSON metadata file for input image tiles
        label_json (str, optional): JSON metadata file for label tiles
        mask_json (str, optional): JSON metadata file for valid mask tiles
        transforms (Compose, optional): MONAI transforms pipeline
        chunk_num (Tuple[int, int, int]): Volume splitting parameters (z, y, x). Default: (2, 2, 2)
        chunk_indices (List[Tuple], optional): Predefined list of chunk indices
        chunk_iter (int): Number of iterations on each chunk. Default: -1
        chunk_stride (bool): Allow overlap between chunks. Default: True
        sample_size (Tuple[int, int, int]): Size of samples to extract (z, y, x)
        mode (str): Dataset mode ('train', 'val', 'test'). Default: 'train'
        iter_num (int): Number of iterations per epoch (-1 for inference). Default: -1
        pad_size (Tuple[int, int, int]): Padding parameters (z, y, x). Default: (0, 0, 0)
        **kwargs: Additional arguments passed to base dataset
    """

    def __init__(
        self,
        volume_json: str,
        label_json: Optional[str] = None,
        mask_json: Optional[str] = None,
        transforms: Optional[Compose] = None,
        chunk_num: Tuple[int, int, int] = (2, 2, 2),
        chunk_indices: Optional[List[Tuple[int, int, int]]] = None,
        chunk_iter: int = -1,
        chunk_stride: bool = True,
        sample_size: Tuple[int, int, int] = (32, 256, 256),
        mode: str = "train",
        iter_num: int = -1,
        pad_size: Tuple[int, int, int] = (0, 0, 0),
        **kwargs,
    ):
        # Load tile metadata
        self.volume_metadata = load_tile_metadata(volume_json)
        self.label_metadata = load_tile_metadata(label_json)
        self.mask_metadata = load_tile_metadata(mask_json)

        # Store tile-specific parameters
        self.chunk_num = ensure_tuple_rep(chunk_num, 3)
        self.chunk_indices = chunk_indices
        self.chunk_iter = chunk_iter
        self.chunk_stride = chunk_stride
        self.pad_size = ensure_tuple_rep(pad_size, 3)

        # Calculate chunk coordinates if not provided
        if chunk_indices is None:
            self.chunk_indices = calculate_chunk_indices(self.volume_metadata, self.chunk_num)

        # Create data dictionaries for chunks
        data_dicts = create_chunk_data_dicts(
            self.chunk_indices,
            volume_metadata=self.volume_metadata,
            label_metadata=self.label_metadata,
            mask_metadata=self.mask_metadata,
        )

        # Create transforms if not provided
        if transforms is None:
            transforms = create_default_tile_transforms(
                label_metadata=self.label_metadata,
                mask_metadata=self.mask_metadata,
            )

        # Initialize base dataset
        super().__init__(
            data_dicts=data_dicts,
            transforms=transforms,
            sample_size=sample_size,
            mode=mode,
            iter_num=iter_num,
            **kwargs,
        )


class MonaiCachedTileDataset(MonaiTileDataset):
    """
    Cached version of MONAI tile dataset.

    This dataset caches reconstructed chunks in memory for improved performance.
    Suitable when the total size of cached chunks fits in available memory.

    Args:
        cache_rate (float): Percentage of chunks to cache. Default: 1.0
        num_workers (int): Number of workers for caching. Default: 0
        **kwargs: Arguments passed to MonaiTileDataset
    """

    def __init__(
        self,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        **kwargs,
    ):
        # Initialize tile-specific attributes first
        volume_json = kwargs.pop("volume_json")
        label_json = kwargs.pop("label_json", None)
        mask_json = kwargs.pop("mask_json", None)
        transforms = kwargs.pop("transforms", None)

        # Load metadata
        self.volume_metadata = load_tile_metadata(volume_json)
        self.label_metadata = load_tile_metadata(label_json)
        self.mask_metadata = load_tile_metadata(mask_json)

        # Set tile-specific parameters
        chunk_num = kwargs.get("chunk_num", (2, 2, 2))
        self.chunk_num = ensure_tuple_rep(chunk_num, 3)
        self.chunk_indices = kwargs.get("chunk_indices", None)

        # Calculate chunk coordinates if not provided
        if self.chunk_indices is None:
            self.chunk_indices = calculate_chunk_indices(self.volume_metadata, self.chunk_num)

        # Create data dictionaries
        data_dicts = create_chunk_data_dicts(
            self.chunk_indices,
            volume_metadata=self.volume_metadata,
            label_metadata=self.label_metadata,
            mask_metadata=self.mask_metadata,
        )

        # Create transforms if not provided
        if transforms is None:
            transforms = create_default_tile_transforms(
                label_metadata=self.label_metadata,
                mask_metadata=self.mask_metadata,
            )

        # Initialize as MONAI CacheDataset
        CacheDataset.__init__(
            self,
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # Store connectomics parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        sample_size = kwargs.get("sample_size", (32, 256, 256))
        self.sample_size = ensure_tuple_rep(sample_size, 3)
        self.mode = kwargs.get("mode", "train")
        self.iter_num = kwargs.get("iter_num", -1)

        # Calculate dataset length
        if self.iter_num > 0:
            self.dataset_length = self.iter_num
        else:
            self.dataset_length = len(data_dicts)

    def __len__(self) -> int:
        return self.dataset_length


__all__ = [
    "MonaiTileDataset",
    "MonaiCachedTileDataset",
    "TileLoaderd",
]
