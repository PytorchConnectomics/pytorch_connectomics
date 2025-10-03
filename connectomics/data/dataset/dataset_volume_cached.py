"""
Optimized cached volume dataset for fast random cropping.

This dataset loads volumes into memory once and performs random cropping
in memory, avoiding repeated disk I/O.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random  # Use random (not np.random) for thread safety
import torch
from monai.data import Dataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from ..io import read_volume


def crop_volume(volume: np.ndarray, size: Tuple[int, int, int], start: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop a subvolume from a volume using numpy slicing (fast!).

    Args:
        volume: Input volume (C, Z, Y, X) or (Z, Y, X)
        size: Crop size (z, y, x)
        start: Start position (z, y, x)

    Returns:
        Cropped volume
    """
    z, y, x = start
    sz, sy, sx = size

    if volume.ndim == 4:  # (C, Z, Y, X)
        return volume[:, z:z+sz, y:y+sy, x:x+sx]
    else:  # (Z, Y, X)
        return volume[z:z+sz, y:y+sy, x:x+sx]


class CachedVolumeDataset(Dataset):
    """
    Cached volume dataset that loads volumes once and crops in memory.

    This dramatically speeds up training with iter_num > num_volumes by:
    1. Loading all volumes into memory once during initialization
    2. Performing random crops from cached volumes during iteration
    3. Applying augmentations to crops (not full volumes)

    Args:
        image_paths: List of image volume paths
        label_paths: List of label volume paths
        patch_size: Size of random crops (z, y, x)
        iter_num: Number of iterations per epoch
        transforms: MONAI transforms (applied after cropping)
        mode: 'train' or 'val'
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        patch_size: Tuple[int, int, int] = (112, 112, 112),
        iter_num: int = 500,
        transforms: Optional[Compose] = None,
        mode: str = 'train',
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths if label_paths else [None] * len(image_paths)
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.iter_num = iter_num if iter_num > 0 else len(image_paths)
        self.transforms = transforms
        self.mode = mode

        # Load all volumes into memory
        print(f"  Loading {len(image_paths)} volumes into memory...")
        self.cached_images = []
        self.cached_labels = []

        for i, (img_path, lbl_path) in enumerate(zip(image_paths, self.label_paths)):
            # Load image
            img = read_volume(img_path)
            if img.ndim == 3:
                img = img[None, ...]  # Add channel dimension: (C, Z, Y, X)
            self.cached_images.append(img)

            # Load label if available
            if lbl_path:
                lbl = read_volume(lbl_path)
                if lbl.ndim == 3:
                    lbl = lbl[None, ...]
                self.cached_labels.append(lbl)
            else:
                self.cached_labels.append(None)

            print(f"    Volume {i+1}/{len(image_paths)}: {img.shape}")

        print(f"  ✓ Loaded {len(self.cached_images)} volumes into memory")

        # Store volume sizes for random position generation
        self.volume_sizes = [img.shape[-3:] for img in self.cached_images]  # (Z, Y, X)

    def __len__(self) -> int:
        return self.iter_num

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, int, int]:
        """
        Get a random crop position for training (like v1 VolumeDataset).

        Args:
            vol_idx: Volume index

        Returns:
            Random crop start position (z, y, x)
        """
        vol_size = self.volume_sizes[vol_idx]
        patch_size = self.patch_size

        # Random position ensuring crop fits within volume
        z = random.randint(0, max(0, vol_size[0] - patch_size[0]))
        y = random.randint(0, max(0, vol_size[1] - patch_size[1]))
        x = random.randint(0, max(0, vol_size[2] - patch_size[2]))

        return (z, y, x)

    def _get_center_crop_position(self, vol_idx: int) -> Tuple[int, int, int]:
        """
        Get center crop position for validation/test.

        Args:
            vol_idx: Volume index

        Returns:
            Center crop start position (z, y, x)
        """
        vol_size = self.volume_sizes[vol_idx]
        patch_size = self.patch_size

        z = max(0, (vol_size[0] - patch_size[0]) // 2)
        y = max(0, (vol_size[1] - patch_size[1]) // 2)
        x = max(0, (vol_size[2] - patch_size[2]) // 2)

        return (z, y, x)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a random crop from cached volumes (fast numpy slicing!)."""

        # Select a random volume (use random for thread safety like v1)
        vol_idx = random.randint(0, len(self.cached_images) - 1)

        # Get cached volumes
        image = self.cached_images[vol_idx]
        label = self.cached_labels[vol_idx]

        # Get crop position
        if self.mode == 'train':
            pos = self._get_random_crop_position(vol_idx)
        else:
            pos = self._get_center_crop_position(vol_idx)

        # Crop using fast numpy slicing (like v1)
        image_crop = crop_volume(image, self.patch_size, pos)
        if label is not None:
            label_crop = crop_volume(label, self.patch_size, pos)
        else:
            label_crop = np.zeros_like(image_crop)

        # Create data dict
        data = {
            'image': image_crop,
            'label': label_crop,
        }

        # Apply additional transforms if provided (augmentation, normalization, etc.)
        if self.transforms:
            data = self.transforms(data)

        return data


__all__ = ['CachedVolumeDataset']
