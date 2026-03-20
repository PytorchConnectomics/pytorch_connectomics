"""
Optimized cached volume dataset for fast random cropping.

Loads volumes into memory once and performs random cropping via numpy slicing,
avoiding repeated disk I/O.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from monai.transforms import Compose

from ..io import read_volume
from .base import PatchDataset
from .crop_sampling import random_crop_position

logger = logging.getLogger(__name__)


def crop_volume(
    volume: np.ndarray,
    size: Tuple[int, ...],
    start: Tuple[int, ...],
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Crop a subvolume from a volume using numpy slicing.

    If the crop extends past volume bounds, pads to the exact requested size.

    Args:
        volume: Input volume (C, D, H, W) or (C, H, W) or without channel dim.
        size: Crop size (d, h, w) for 3D or (h, w) for 2D.
        start: Start position matching size dimensions.
        pad_mode: Padding mode -- "reflect" for images, "constant" for labels/masks.

    Returns:
        Cropped volume with exact requested size.
    """
    ndim = len(size)
    if ndim not in (2, 3):
        raise ValueError(f"crop_volume only supports 2D or 3D, got {ndim}D")

    has_channel = volume.ndim == ndim + 1
    vol_spatial = volume.shape[1:] if has_channel else volume.shape

    slices = []
    pad_width = []
    for i in range(ndim):
        clamped_start = max(0, min(start[i], vol_spatial[i]))
        available = max(0, vol_spatial[i] - clamped_start)
        actual_crop = min(size[i], available)
        slices.append(slice(clamped_start, clamped_start + actual_crop))
        pad_width.append((0, max(0, size[i] - actual_crop)))

    if has_channel:
        cropped = volume[(slice(None),) + tuple(slices)]
    else:
        cropped = volume[tuple(slices)]

    if any(pad > 0 for _, pad in pad_width):
        full_pad = [(0, 0)] + pad_width if has_channel else pad_width
        if cropped.size == 0 or any(s == 0 for s in cropped.shape):
            cropped = np.pad(cropped, full_pad, mode="constant", constant_values=0)
        else:
            cropped = np.pad(cropped, full_pad, mode=pad_mode)

    return cropped


class CachedVolumeDataset(PatchDataset):
    """
    Cached volume dataset that loads volumes once and crops in memory.

    Dramatically speeds up training by:
    1. Loading all volumes into memory once during init
    2. Performing random crops from cached volumes during iteration
    3. Applying augmentations to crops (not full volumes)

    Args:
        image_paths: List of image volume paths.
        label_paths: List of label volume paths (None entries OK).
        mask_paths: List of mask volume paths (None entries OK).
        patch_size: Size of random crops (z, y, x) or (y, x).
        iter_num: Number of iterations per epoch.
        transforms: MONAI transforms applied after cropping.
        pre_cache_transforms: One-time transforms applied before caching.
        mode: 'train' or 'val'.
        pad_size: Padding to apply to each spatial dimension.
        pad_mode: Padding mode ('reflect', 'constant', etc.).
        max_attempts: Max foreground sampling retries.
        foreground_threshold: Min foreground fraction to accept a patch.
        crop_to_nonzero_mask: Constrain crops to intersect mask bounding box.
        sample_nonzero_mask: Center crops on random nonzero mask voxels.
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        label_aux_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        patch_size: Tuple[int, ...] = (112, 112, 112),
        iter_num: int = 500,
        transforms: Optional[Compose] = None,
        pre_cache_transforms: Optional[Any] = None,
        mode: str = "train",
        pad_size: Optional[Tuple[int, ...]] = None,
        pad_mode: str = "reflect",
        max_attempts: int = 10,
        foreground_threshold: float = 0.05,
        crop_to_nonzero_mask: bool = False,
        sample_nonzero_mask: bool = False,
    ):
        super().__init__(
            patch_size=patch_size,
            iter_num=iter_num if iter_num > 0 else len(image_paths),
            transforms=transforms,
            mode=mode,
            max_attempts=max_attempts,
            foreground_threshold=foreground_threshold,
        )

        self.pad_size = pad_size
        self.pad_mode = pad_mode
        self.crop_to_nonzero_mask = crop_to_nonzero_mask
        self.sample_nonzero_mask = sample_nonzero_mask

        label_paths = label_paths or [None] * len(image_paths)
        label_aux_paths = label_aux_paths or [None] * len(image_paths)
        mask_paths = mask_paths or [None] * len(image_paths)

        # Load all volumes into memory
        logger.info("Loading %d volumes into memory...", len(image_paths))
        self.cached_images: List[np.ndarray] = []
        self.cached_labels: List[Optional[np.ndarray]] = []
        self.cached_label_aux: List[Optional[np.ndarray]] = []
        self.cached_masks: List[Optional[np.ndarray]] = []

        for i, (img_path, lbl_path, aux_path, msk_path) in enumerate(
            zip(image_paths, label_paths, label_aux_paths, mask_paths)
        ):
            img = self._load_volume(img_path)
            lbl = self._load_volume(lbl_path) if lbl_path else None
            aux = self._load_volume(aux_path) if aux_path else None
            msk = self._load_volume(msk_path) if msk_path else None

            # Apply one-time preprocessing before caching
            if pre_cache_transforms is not None:
                sample = {"image": img}
                if lbl is not None:
                    sample["label"] = lbl
                if aux is not None:
                    sample["label_aux"] = aux
                if msk is not None:
                    sample["mask"] = msk
                sample = pre_cache_transforms(sample)
                img = sample["image"]
                lbl = sample.get("label")
                aux = sample.get("label_aux")
                msk = sample.get("mask")

            # Pad and ensure minimum size
            img = self._prepare_volume(img)
            lbl = self._prepare_volume(lbl) if lbl is not None else None
            aux = self._prepare_volume(aux) if aux is not None else None
            msk = self._prepare_volume(msk) if msk is not None else None

            self.cached_images.append(img)
            self.cached_labels.append(lbl)
            self.cached_label_aux.append(aux)
            self.cached_masks.append(msk)
            logger.info("Volume %d/%d: %s", i + 1, len(image_paths), img.shape)

        logger.info("Loaded %d volumes into memory", len(self.cached_images))

        # Store volume spatial sizes
        ndim = len(self.patch_size)
        self.volume_sizes = [img.shape[-ndim:] for img in self.cached_images]

        # Precompute mask bounding boxes for crop_to_nonzero_mask
        self.mask_bboxes: List[Optional[List[Tuple[int, int]]]] = []
        if self.crop_to_nonzero_mask:
            for mask in self.cached_masks:
                self.mask_bboxes.append(self._compute_mask_bbox(mask))
            logger.info(
                "[crop_to_nonzero_mask] Bboxes computed for %d volumes",
                len(self.mask_bboxes),
            )
        else:
            self.mask_bboxes = [None] * len(self.cached_images)

        # Precompute nonzero mask coordinates for sample_nonzero_mask
        self.mask_nonzero_coords: List[Optional[np.ndarray]] = []
        if self.sample_nonzero_mask:
            for mask in self.cached_masks:
                if mask is not None:
                    coords = np.argwhere(mask[0] > 0)
                    self.mask_nonzero_coords.append(coords if len(coords) > 0 else None)
                else:
                    self.mask_nonzero_coords.append(None)
            n_valid = sum(1 for c in self.mask_nonzero_coords if c is not None)
            total = sum(len(c) for c in self.mask_nonzero_coords if c is not None)
            logger.info(
                "[sample_nonzero_mask] %d/%d volumes have nonzero mask (%d voxels)",
                n_valid,
                len(self.mask_nonzero_coords),
                total,
            )
        else:
            self.mask_nonzero_coords = [None] * len(self.cached_images)

    # -- PatchDataset abstract methods --

    def _crop_volumes(self, vol_idx: int, pos: Tuple[int, ...]) -> Dict[str, Any]:
        image = self.cached_images[vol_idx]
        label = self.cached_labels[vol_idx]
        label_aux = self.cached_label_aux[vol_idx]
        mask = self.cached_masks[vol_idx]

        image_crop = crop_volume(image, self.patch_size, pos, pad_mode="reflect")
        label_crop = (
            crop_volume(label, self.patch_size, pos, pad_mode="constant")
            if label is not None
            else None
        )
        label_aux_crop = (
            crop_volume(label_aux, self.patch_size, pos, pad_mode="constant")
            if label_aux is not None
            else None
        )
        mask_crop = (
            crop_volume(mask, self.patch_size, pos, pad_mode="constant")
            if mask is not None
            else None
        )

        return {
            "image": image_crop,
            "label": label_crop,
            "label_aux": label_aux_crop,
            "mask": mask_crop,
        }

    def _has_labels(self, vol_idx: int) -> bool:
        return self.cached_labels[vol_idx] is not None

    # -- Override crop position for mask-aware sampling --

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        coords = self.mask_nonzero_coords[vol_idx] if self.sample_nonzero_mask else None
        bbox = self.mask_bboxes[vol_idx] if self.crop_to_nonzero_mask else None
        return random_crop_position(
            self.volume_sizes[vol_idx],
            self.patch_size,
            rng=random,
            mask_nonzero_coords=coords,
            mask_bbox=bbox,
        )

    # -- Volume loading helpers --

    @staticmethod
    def _load_volume(path: str) -> np.ndarray:
        """Load volume and add channel dimension."""
        vol = read_volume(path)
        if vol.ndim in (2, 3):
            vol = vol[None, ...]  # Add channel dim
        return vol

    def _prepare_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply padding and ensure minimum size."""
        if self.pad_size is not None:
            volume = self._apply_padding(volume)
        volume = self._ensure_minimum_size(volume)
        return volume

    def _apply_padding(self, volume: np.ndarray) -> np.ndarray:
        """Apply symmetric padding to spatial dimensions."""
        if self.pad_size is None:
            return volume
        pad_width = [(0, 0)] + [(p, p) for p in self.pad_size]
        if self.pad_mode == "constant":
            return np.pad(volume, pad_width, mode="constant", constant_values=0)
        return np.pad(volume, pad_width, mode=self.pad_mode)

    def _ensure_minimum_size(self, volume: np.ndarray) -> np.ndarray:
        """Pad volume to at least patch_size in all spatial dimensions."""
        ndim = len(self.patch_size)
        has_channel = volume.ndim == ndim + 1
        spatial = volume.shape[1:] if has_channel else volume.shape

        if all(spatial[i] >= self.patch_size[i] for i in range(ndim)):
            return volume

        pad_width = [(0, 0)] if has_channel else []
        for i in range(ndim):
            deficit = max(0, self.patch_size[i] - spatial[i])
            pad_width.append((deficit // 2, deficit - deficit // 2))

        if self.pad_mode == "constant":
            return np.pad(volume, pad_width, mode="constant", constant_values=0)
        return np.pad(volume, pad_width, mode=self.pad_mode)

    @staticmethod
    def _compute_mask_bbox(
        mask: Optional[np.ndarray],
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute axis-aligned bounding box of nonzero voxels."""
        if mask is None:
            return None
        spatial = mask[0] > 0
        if not spatial.any():
            return None
        bbox = []
        for d in range(spatial.ndim):
            axes = tuple(i for i in range(spatial.ndim) if i != d)
            proj = spatial.any(axis=axes) if spatial.ndim > 1 else spatial
            indices = np.where(proj)[0]
            bbox.append((int(indices[0]), int(indices[-1]) + 1))
        return bbox


__all__ = ["CachedVolumeDataset", "crop_volume"]
