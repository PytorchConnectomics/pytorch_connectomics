"""
Optimized cached volume dataset for fast random cropping.

This dataset loads volumes into memory once and performs random cropping
in memory, avoiding repeated disk I/O.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random  # Use random (not np.random) for thread safety
from monai.data import Dataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from ..io import read_volume


def crop_volume(volume: np.ndarray, size: Tuple[int, ...], start: Tuple[int, ...]) -> np.ndarray:
    """
    Crop a subvolume from a volume using numpy slicing (fast!).

    Args:
        volume: Input volume
                2D: (C, H, W) or (H, W)
                3D: (C, D, H, W) or (D, H, W)
        size: Crop size (d, h, w) for 3D or (h, w) for 2D
        start: Start position (d, h, w) for 3D or (h, w) for 2D

    Returns:
        Cropped volume
    """
    ndim = len(size)
    if ndim not in [2, 3]:
        raise ValueError(f"crop_volume only supports 2D or 3D, got {ndim}D")

    # Build slicing tuple dynamically based on dimensions
    slices = [slice(start[i], start[i] + size[i]) for i in range(ndim)]

    # Check if volume has channel dimension
    has_channel = volume.ndim == ndim + 1

    if has_channel:
        # (C, ...) format - keep all channels, crop spatial dims
        return volume[(slice(None),) + tuple(slices)]
    else:
        # No channel dimension - crop directly
        return volume[tuple(slices)]


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
        mask_paths: List of mask volume paths
        patch_size: Size of random crops (z, y, x)
        iter_num: Number of iterations per epoch
        transforms: MONAI transforms (applied after cropping)
        mode: 'train' or 'val'
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        patch_size: Tuple[int, int, int] = (112, 112, 112),
        iter_num: int = 500,
        transforms: Optional[Compose] = None,
        mode: str = "train",
        pad_size: Optional[Tuple[int, ...]] = None,
        pad_mode: str = "reflect",
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths if label_paths else [None] * len(image_paths)
        self.mask_paths = mask_paths if mask_paths else [None] * len(image_paths)

        # Support both 2D and 3D patch sizes
        if isinstance(patch_size, (list, tuple)):
            ndim = len(patch_size)
            if ndim not in [2, 3]:
                raise ValueError(f"patch_size must be 2D or 3D, got {ndim}D")
            self.patch_size = ensure_tuple_rep(patch_size, ndim)
        else:
            # Single value - assume 3D for backward compatibility
            self.patch_size = ensure_tuple_rep(patch_size, 3)

        self.iter_num = iter_num if iter_num > 0 else len(image_paths)
        self.transforms = transforms
        self.mode = mode
        self.pad_size = pad_size
        self.pad_mode = pad_mode

        # Load all volumes into memory
        print(f"  Loading {len(image_paths)} volumes into memory...")
        self.cached_images = []
        self.cached_labels = []
        self.cached_masks = []

        for i, (img_path, lbl_path, mask_path) in enumerate(
            zip(image_paths, self.label_paths, self.mask_paths)
        ):
            # Load image
            img = read_volume(img_path)
            # Add channel dimension for both 2D and 3D
            # 2D: (H, W) → (1, H, W)
            # 3D: (D, H, W) → (1, D, H, W)
            if img.ndim == 2:
                img = img[None, ...]  # Add channel for 2D
            elif img.ndim == 3:
                img = img[None, ...]  # Add channel for 3D

            # Apply padding if specified
            if self.pad_size is not None:
                img = self._apply_padding(img)

            self.cached_images.append(img)

            # Load label if available
            if lbl_path:
                lbl = read_volume(lbl_path)
                # Add channel dimension for both 2D and 3D
                if lbl.ndim == 2:
                    lbl = lbl[None, ...]  # Add channel for 2D
                elif lbl.ndim == 3:
                    lbl = lbl[None, ...]  # Add channel for 3D

                # Apply padding if specified (same padding as image)
                if self.pad_size is not None:
                    lbl = self._apply_padding(
                        lbl, mode="constant", constant_values=0
                    )  # Use constant 0 for labels

                self.cached_labels.append(lbl)
            else:
                self.cached_labels.append(None)

            # Load mask if available
            if mask_path:
                mask = read_volume(mask_path)
                if mask.ndim == 3:
                    mask = mask[None, ...]

                # Apply padding if specified (same padding as label)
                if self.pad_size is not None:
                    mask = self._apply_padding(mask, mode="constant", constant_values=0)

                self.cached_masks.append(mask)
            else:
                self.cached_masks.append(None)

            print(f"    Volume {i+1}/{len(image_paths)}: {img.shape}")

        print(f"  ✓ Loaded {len(self.cached_images)} volumes into memory")

        # Store volume sizes for random position generation
        # Support both 2D and 3D: get last N dimensions matching patch_size
        ndim = len(self.patch_size)
        self.volume_sizes = [img.shape[-ndim:] for img in self.cached_images]  # (Z, Y, X) or (Y, X)

    def _apply_padding(
        self, volume: np.ndarray, mode: Optional[str] = None, constant_values: float = 0
    ) -> np.ndarray:
        """
        Apply padding to a volume using np.pad.

        Args:
            volume: Input volume with channel dimension (C, D, H, W) or (C, H, W)
            mode: Padding mode ('reflect', 'constant', etc.). If None, uses self.pad_mode
            constant_values: Value for constant padding

        Returns:
            Padded volume
        """
        if self.pad_size is None:
            return volume

        mode = mode if mode is not None else self.pad_mode

        # Build padding tuple for np.pad: ((before, after), ...)
        # For channel dimension: no padding (0, 0)
        # For spatial dimensions: pad according to pad_size
        pad_width = [(0, 0)]  # No padding on channel dimension
        for p in self.pad_size:
            pad_width.append((p, p))

        # Apply padding using np.pad
        if mode == "constant":
            padded = np.pad(volume, pad_width, mode=mode, constant_values=constant_values)
        else:
            padded = np.pad(volume, pad_width, mode=mode)

        return padded

    def __len__(self) -> int:
        return self.iter_num

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        """
        Get a random crop position for training (like v1 VolumeDataset).

        Args:
            vol_idx: Volume index

        Returns:
            Random crop start position (z, y, x) for 3D or (y, x) for 2D
        """
        vol_size = self.volume_sizes[vol_idx]
        patch_size = self.patch_size

        # Random position ensuring crop fits within volume
        # Support both 2D and 3D
        positions = tuple(
            random.randint(0, max(0, vol_size[i] - patch_size[i])) for i in range(len(patch_size))
        )
        return positions

    def _get_center_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        """
        Get center crop position for validation/test.

        Args:
            vol_idx: Volume index

        Returns:
            Center crop start position (z, y, x) for 3D or (y, x) for 2D
        """
        vol_size = self.volume_sizes[vol_idx]
        patch_size = self.patch_size

        # Center position for each dimension
        # Support both 2D and 3D
        positions = tuple(
            max(0, (vol_size[i] - patch_size[i]) // 2) for i in range(len(patch_size))
        )
        return positions

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a random crop from cached volumes (fast numpy slicing!)."""

        # Select a random volume (use random for thread safety like v1)
        vol_idx = random.randint(0, len(self.cached_images) - 1)

        # Get cached volumes
        image = self.cached_images[vol_idx]
        label = self.cached_labels[vol_idx]
        mask = self.cached_masks[vol_idx]

        # Get crop position
        if self.mode == "train":
            pos = self._get_random_crop_position(vol_idx)
        else:
            pos = self._get_center_crop_position(vol_idx)

        # Crop using fast numpy slicing (like v1)
        image_crop = crop_volume(image, self.patch_size, pos)
        if label is not None:
            label_crop = crop_volume(label, self.patch_size, pos)
        else:
            label_crop = np.zeros_like(image_crop)

        if mask is not None:
            mask_crop = crop_volume(mask, self.patch_size, pos)
        else:
            mask_crop = np.zeros_like(image_crop)

        # Create data dict
        data = {
            "image": image_crop,
            "label": label_crop,
            "mask": mask_crop,
        }

        # Apply additional transforms if provided (augmentation, normalization, etc.)
        if self.transforms:
            data = self.transforms(data)

        return data


__all__ = ["CachedVolumeDataset"]
