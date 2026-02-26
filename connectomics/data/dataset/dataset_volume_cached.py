"""
Optimized cached volume dataset for fast random cropping.

This dataset loads volumes into memory once and performs random cropping
in memory, avoiding repeated disk I/O.
"""

from __future__ import annotations

import random  # Use random (not np.random) for thread safety
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from monai.data import Dataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from ..io import read_volume


def crop_volume(volume: np.ndarray, size: Tuple[int, ...], start: Tuple[int, ...]) -> np.ndarray:
    """
    Crop a subvolume from a volume using numpy slicing (fast!).

    If the crop would go out of bounds, pads the result to the exact requested size.

    Args:
        volume: Input volume
                2D: (C, H, W) or (H, W)
                3D: (C, D, H, W) or (D, H, W)
        size: Crop size (d, h, w) for 3D or (h, w) for 2D
        start: Start position (d, h, w) for 3D or (h, w) for 2D

    Returns:
        Cropped volume with exact size (padded if necessary)
    """
    ndim = len(size)
    if ndim not in [2, 3]:
        raise ValueError(f"crop_volume only supports 2D or 3D, got {ndim}D")

    # Check if volume has channel dimension
    has_channel = volume.ndim == ndim + 1

    # Get actual volume spatial dimensions
    if has_channel:
        vol_spatial_shape = volume.shape[1:]
    else:
        vol_spatial_shape = volume.shape

    # Build slicing tuple dynamically based on dimensions
    slices = []
    pad_width = []
    for i in range(ndim):
        # Clamp start to valid range (can't be negative, can't be beyond volume)
        clamped_start = max(0, min(start[i], vol_spatial_shape[i]))
        # Calculate how much we can actually crop from this position
        available = max(0, vol_spatial_shape[i] - clamped_start)
        actual_crop_size = min(size[i], available)
        actual_end = clamped_start + actual_crop_size
        slices.append(slice(clamped_start, actual_end))

        # Calculate padding needed if crop is smaller than requested
        # This ensures we always return exactly size[i]
        pad_after = max(0, size[i] - actual_crop_size)
        pad_width.append((0, pad_after))

    # Perform the crop
    if has_channel:
        # (C, ...) format - keep all channels, crop spatial dims
        cropped = volume[(slice(None),) + tuple(slices)]
    else:
        # No channel dimension - crop directly
        cropped = volume[tuple(slices)]

    # Pad if necessary to ensure exact size
    if any(pad > 0 for _, pad in pad_width):
        # Build full pad_width including channel dimension if present
        if has_channel:
            full_pad_width = [(0, 0)] + pad_width  # No padding on channel
        else:
            full_pad_width = pad_width

        # Check if cropped array is empty or has zero-sized dimensions
        # Reflect padding cannot work on empty arrays, so fall back to constant padding
        is_empty = cropped.size == 0 or any(s == 0 for s in cropped.shape)
        if is_empty:
            # Use constant padding for empty arrays (reflect padding requires at least one element)
            cropped = np.pad(cropped, full_pad_width, mode="constant", constant_values=0)
        else:
            # Use reflect padding to match the volume's edge values
            cropped = np.pad(cropped, full_pad_width, mode="reflect")

    # Final safety check: ensure output has exactly the requested size
    # This handles any edge cases where padding calculation might be off
    expected_spatial_shape = size
    actual_spatial_shape = cropped.shape[-ndim:] if has_channel else cropped.shape

    if actual_spatial_shape != expected_spatial_shape:
        # Calculate how much padding is still needed
        pad_needed = []
        for i in range(ndim):
            pad = max(0, expected_spatial_shape[i] - actual_spatial_shape[i])
            pad_needed.append((0, pad))

        if any(p > 0 for _, p in pad_needed):
            if has_channel:
                full_pad_width = [(0, 0)] + pad_needed
            else:
                full_pad_width = pad_needed

            # Check if cropped array is empty or has zero-sized dimensions
            # Reflect padding cannot work on empty arrays, so fall back to constant padding
            is_empty = cropped.size == 0 or any(s == 0 for s in cropped.shape)
            if is_empty:
                # Use constant padding for empty arrays (reflect padding requires at least one element)
                cropped = np.pad(cropped, full_pad_width, mode="constant", constant_values=0)
            else:
                # Use reflect padding to match the volume's edge values
                cropped = np.pad(cropped, full_pad_width, mode="reflect")

        # If still wrong (shouldn't happen), trim excess
        if has_channel:
            current_spatial = cropped.shape[-ndim:]
            if current_spatial != expected_spatial_shape:
                slices = [slice(None)] + [slice(0, s) for s in expected_spatial_shape]
                cropped = cropped[tuple(slices)]
        else:
            if cropped.shape != expected_spatial_shape:
                slices = [slice(0, s) for s in expected_spatial_shape]
                cropped = cropped[tuple(slices)]

    return cropped


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
        pre_cache_transforms: Optional[Any] = None,
        mode: str = "train",
        pad_size: Optional[Tuple[int, ...]] = None,
        pad_mode: str = "reflect",
        max_attempts: int = 10,
        foreground_threshold: float = 0.05,
        crop_to_nonzero_mask: bool = False,
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
        self.pre_cache_transforms = pre_cache_transforms
        self.mode = mode
        self.pad_size = pad_size
        self.pad_mode = pad_mode

        # [FIX 1] Store base seed for epoch-based validation reseeding
        # This allows validation to sample different patches each epoch while remaining deterministic
        self.base_seed = 0  # Will be set by set_epoch() or defaults to 0
        self.current_epoch = 0

        # [D2 DIAGNOSTIC] Initialize sampling statistics
        self._d2_total_samples = 0
        self._d2_total_attempts = 0
        self._d2_rejected_patches = 0
        self._d2_foreground_fracs = []
        self._d2_last_report_step = 0
        self.max_attempts = max_attempts
        self.foreground_threshold = foreground_threshold
        self.crop_to_nonzero_mask = crop_to_nonzero_mask

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

            # Load label if available
            lbl = None
            if lbl_path:
                lbl = read_volume(lbl_path)
                # Add channel dimension for both 2D and 3D
                if lbl.ndim == 2:
                    lbl = lbl[None, ...]  # Add channel for 2D
                elif lbl.ndim == 3:
                    lbl = lbl[None, ...]  # Add channel for 3D

            # Load mask if available
            mask = None
            if mask_path:
                mask = read_volume(mask_path)
                if mask.ndim == 2:
                    mask = mask[None, ...]
                elif mask.ndim == 3:
                    mask = mask[None, ...]

            # Apply one-time preprocessing before caching (e.g., nnU-Net crop/resample/normalize).
            if self.pre_cache_transforms is not None:
                sample = {"image": img}
                if lbl is not None:
                    sample["label"] = lbl
                if mask is not None:
                    sample["mask"] = mask
                sample = self.pre_cache_transforms(sample)
                img = sample["image"]
                lbl = sample.get("label")
                mask = sample.get("mask")

            # Apply padding if specified
            if self.pad_size is not None:
                img = self._apply_padding(img)

            # Ensure volume is at least as large as patch_size in all dimensions
            # This prevents crops from being smaller than patch_size
            img = self._ensure_minimum_size(img)
            self.cached_images.append(img)

            if lbl is not None:
                # Apply padding if specified (same padding as image)
                if self.pad_size is not None:
                    lbl = self._apply_padding(
                        lbl, mode="constant", constant_values=0
                    )  # Use constant 0 for labels

                # Ensure label is at least as large as patch_size
                lbl = self._ensure_minimum_size(lbl, mode="constant", constant_values=0)
                self.cached_labels.append(lbl)
            else:
                self.cached_labels.append(None)

            if mask is not None:
                # Apply padding if specified (same padding as label)
                if self.pad_size is not None:
                    mask = self._apply_padding(mask, mode="constant", constant_values=0)

                # Ensure mask is at least as large as patch_size
                mask = self._ensure_minimum_size(mask, mode="constant", constant_values=0)
                self.cached_masks.append(mask)
            else:
                self.cached_masks.append(None)

            print(f"    Volume {i + 1}/{len(image_paths)}: {img.shape}")

        print(f"  ✓ Loaded {len(self.cached_images)} volumes into memory")

        # Store volume sizes for random position generation
        # Support both 2D and 3D: get last N dimensions matching patch_size
        ndim = len(self.patch_size)
        self.volume_sizes = [img.shape[-ndim:] for img in self.cached_images]  # (Z, Y, X) or (Y, X)

        # Precompute mask bounding boxes for crop_to_nonzero_mask sampling
        self.mask_bboxes: List[Optional[List[Tuple[int, int]]]] = []
        if self.crop_to_nonzero_mask:
            for mask in self.cached_masks:
                self.mask_bboxes.append(self._compute_mask_bbox(mask))
            print(
                f"  [crop_to_nonzero_mask] Mask bboxes computed for {len(self.mask_bboxes)} volumes"
            )
        else:
            self.mask_bboxes = [None] * len(self.cached_images)

        # [D2 DIAGNOSTIC] Print foreground sampling configuration
        if self.mode == "train":
            if self.foreground_threshold > 0:
                print("  [D2] Foreground sampling ENABLED:")
                print(f"    - Minimum foreground threshold: {self.foreground_threshold * 100:.1f}%")
                print(f"    - Max retry attempts: {self.max_attempts}")
                print("    - Will report statistics every 100 batches")
            else:
                print("  [D2] Foreground sampling DISABLED (threshold <= 0)")

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

    def _ensure_minimum_size(
        self, volume: np.ndarray, mode: Optional[str] = None, constant_values: float = 0
    ) -> np.ndarray:
        """
        Ensure volume is at least as large as patch_size in all spatial dimensions.
        Pads only if necessary.

        Args:
            volume: Input volume with channel dimension (C, D, H, W) or (C, H, W)
            mode: Padding mode ('reflect', 'constant', etc.). If None, uses self.pad_mode
            constant_values: Value for constant padding

        Returns:
            Volume padded to at least patch_size if necessary
        """
        ndim = len(self.patch_size)
        has_channel = volume.ndim == ndim + 1

        # Get actual volume spatial dimensions
        if has_channel:
            vol_spatial_shape = volume.shape[1:]
        else:
            vol_spatial_shape = volume.shape

        # Check if padding is needed
        needs_padding = any(vol_spatial_shape[i] < self.patch_size[i] for i in range(ndim))
        if not needs_padding:
            return volume

        # Calculate padding needed for each dimension
        mode = mode if mode is not None else self.pad_mode
        pad_width = [(0, 0)]  # No padding on channel dimension
        for i in range(ndim):
            pad_needed = max(0, self.patch_size[i] - vol_spatial_shape[i])
            # Distribute padding evenly on both sides
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            pad_width.append((pad_before, pad_after))

        # Apply padding
        if mode == "constant":
            padded = np.pad(volume, pad_width, mode=mode, constant_values=constant_values)
        else:
            padded = np.pad(volume, pad_width, mode=mode)

        return padded

    def __len__(self) -> int:
        return self.iter_num

    def set_epoch(self, epoch: int, base_seed: int = 0):
        """
        Set current epoch for epoch-based validation reseeding.

        This method enables validation to sample different patches each epoch
        while maintaining determinism. For training, this has no effect since
        training already uses random sampling.

        Args:
            epoch: Current training epoch
            base_seed: Base random seed (typically from cfg.system.seed)

        Usage:
            Called by ValidationReseedingCallback at the start of each validation epoch:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(trainer.current_epoch, base_seed)
        """
        if self.mode == "val":
            # Reseed RNG for validation to get different patches each epoch
            # Use base_seed + epoch to ensure reproducibility across runs
            self.base_seed = base_seed
            self.current_epoch = epoch
            effective_seed = self.base_seed + epoch
            random.seed(effective_seed)

            # IMPORTANT: Print to verify reseeding is happening
            # This should appear in logs at the start of EACH validation epoch
            print(
                f"[Validation] Set epoch={epoch}, base_seed={base_seed}, effective_seed={effective_seed}"
            )
            print(
                f"[Validation] Dataset: {type(self).__name__}@{id(self)}, mode={self.mode}, iter_num={self.iter_num}"
            )

    def get_sampling_fingerprint(self, num_samples: int = 5) -> str:
        """
        Generate a deterministic fingerprint of validation sampling.

        This allows verification that validation patches change across epochs.
        The fingerprint is based on the first N random samples that would be
        generated with the current RNG state.

        Args:
            num_samples: Number of random samples to include in fingerprint

        Returns:
            String representing the sampling fingerprint
        """
        if self.mode != "val":
            return "N/A (training mode)"

        # Save current RNG state
        state = random.getstate()

        try:
            # Generate deterministic samples
            samples = []
            for _ in range(num_samples):
                # Sample volume index
                vol_idx = random.randint(0, len(self.cached_images) - 1)
                # Sample patch position
                pos = self._get_random_crop_position(vol_idx)
                samples.append((vol_idx, pos))

            # Create fingerprint string
            fingerprint = ", ".join([f"v{v}@{p}" for v, p in samples])
            return fingerprint

        finally:
            # Restore RNG state
            random.setstate(state)

    def _compute_mask_bbox(
        self, mask: Optional[np.ndarray]
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute the axis-aligned bounding box of nonzero voxels in the mask.

        Args:
            mask: Mask array with shape (C, D, H, W) or (C, H, W). May be None.

        Returns:
            List of (start, end) pairs for each spatial dimension, or None if
            the mask is None or entirely zero.
        """
        if mask is None:
            return None
        spatial = mask[0] > 0  # drop channel dim: (D, H, W) or (H, W)
        if not spatial.any():
            return None
        ndim = spatial.ndim
        bbox = []
        for d in range(ndim):
            axes = tuple(i for i in range(ndim) if i != d)
            proj = spatial.any(axis=axes) if ndim > 1 else spatial
            indices = np.where(proj)[0]
            bbox.append((int(indices[0]), int(indices[-1]) + 1))
        return bbox

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        """
        Get a random crop position for training (like v1 VolumeDataset).

        When crop_to_nonzero_mask is True, restricts sampling so every crop
        intersects the nonzero bounding box of the mask.

        Args:
            vol_idx: Volume index

        Returns:
            Random crop start position (z, y, x) for 3D or (y, x) for 2D
        """
        vol_size = self.volume_sizes[vol_idx]
        patch_size = self.patch_size
        mask_bbox = self.mask_bboxes[vol_idx] if self.crop_to_nonzero_mask else None

        positions = []
        for i in range(len(patch_size)):
            vol_max = max(0, vol_size[i] - patch_size[i])
            if mask_bbox is not None:
                # Intersection condition: crop [pos, pos+patch) overlaps [bbox_start, bbox_end)
                # => pos < bbox_end  and  pos + patch > bbox_start
                # => pos >= bbox_start - patch + 1  and  pos <= bbox_end - 1
                min_start = max(0, mask_bbox[i][0] - patch_size[i] + 1)
                max_start = min(vol_max, mask_bbox[i][1] - 1)
                if min_start > max_start:  # bbox too small — fall back to full range
                    min_start, max_start = 0, vol_max
            else:
                min_start, max_start = 0, vol_max
            positions.append(random.randint(min_start, max_start))
        return tuple(positions)

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

        # [D2] Foreground-aware patch sampling: optional retry loop for training.
        # Disabled by default when foreground_threshold <= 0.
        max_attempts = self.max_attempts
        foreground_threshold = self.foreground_threshold
        use_foreground_sampling = (
            self.mode == "train" and label is not None and foreground_threshold > 0
        )

        # [D2 DIAGNOSTIC] Track sampling attempts only when foreground sampling is active.
        attempts_used = 1
        final_foreground_frac = 0.0

        if use_foreground_sampling:
            for attempt in range(max_attempts):
                attempts_used = attempt + 1
                pos = self._get_random_crop_position(vol_idx)

                # Crop using fast numpy slicing (like v1)
                image_crop = crop_volume(image, self.patch_size, pos)
                label_crop = crop_volume(label, self.patch_size, pos)
                if mask is not None:
                    mask_crop = crop_volume(mask, self.patch_size, pos)
                else:
                    mask_crop = np.zeros_like(image_crop)

                foreground_frac = (label_crop > 0).sum() / label_crop.size
                final_foreground_frac = foreground_frac

                if foreground_frac >= foreground_threshold:
                    break

                # [D2 DIAGNOSTIC] Patch rejected, increment counter
                self._d2_rejected_patches += 1
        else:
            # Standard single-crop behavior (no foreground-based retry)
            if self.mode == "train":
                pos = self._get_random_crop_position(vol_idx)
            else:
                pos = self._get_center_crop_position(vol_idx)

            image_crop = crop_volume(image, self.patch_size, pos)
            if label is not None:
                label_crop = crop_volume(label, self.patch_size, pos)
            else:
                label_crop = np.zeros_like(image_crop)

            if mask is not None:
                mask_crop = crop_volume(mask, self.patch_size, pos)
            else:
                mask_crop = np.zeros_like(image_crop)

        # [D2 DIAGNOSTIC] Record/report sampling stats only when enabled.
        if use_foreground_sampling:
            self._d2_total_samples += 1
            self._d2_total_attempts += attempts_used
            self._d2_foreground_fracs.append(final_foreground_frac * 100)  # percentage

        # [D2 DIAGNOSTIC] Print report every 100 samples (not too verbose)
        if use_foreground_sampling and self._d2_total_samples % 100 == 0:
            avg_attempts = self._d2_total_attempts / self._d2_total_samples
            reject_rate = (self._d2_rejected_patches / self._d2_total_attempts) * 100
            avg_fg = sum(self._d2_foreground_fracs) / len(self._d2_foreground_fracs)
            min_fg = min(self._d2_foreground_fracs)
            max_fg = max(self._d2_foreground_fracs)

            print(f"[D2 Sampling Stats after {self._d2_total_samples} batches]")
            print(f"  Avg attempts per patch: {avg_attempts:.2f}/{max_attempts}")
            print(
                f"  Patches rejected: {self._d2_rejected_patches}/{self._d2_total_attempts} ({reject_rate:.1f}%)"
            )
            print(f"  Final foreground %: avg={avg_fg:.1f}%, min={min_fg:.1f}%, max={max_fg:.1f}%")
            print(
                f"  Threshold: {foreground_threshold * 100:.1f}% "
                f"({self.foreground_threshold * 100:.1f}% minimum)"
            )

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
