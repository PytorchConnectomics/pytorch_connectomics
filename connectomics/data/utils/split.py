"""
Volume splitting utilities for train/val splits.

Inspired by DeepEM's approach of using spatial splits along the Z-axis
for training and validation, with automatic padding to model input size.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from pathlib import Path


def split_volume_train_val(
    volume_shape: Tuple[int, int, int],
    train_ratio: float = 0.8,
    axis: int = 0,
    min_val_size: Optional[int] = None,
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Split a volume into training and validation regions along a specified axis.

    This follows DeepEM's approach of spatial splitting where:
    - First 80% (or specified ratio) of volume is used for training
    - Last 20% is used for validation
    - Split is along Z-axis by default (axis=0 for [D,H,W] volumes)

    Args:
        volume_shape: Shape of the volume (D, H, W)
        train_ratio: Ratio of volume to use for training (default: 0.8)
        axis: Axis along which to split (0=D, 1=H, 2=W). Default: 0 (Z-axis)
        min_val_size: Minimum size for validation split (default: None)

    Returns:
        train_slices: Tuple of slices for training region
        val_slices: Tuple of slices for validation region

    Example:
        >>> volume_shape = (100, 256, 256)  # [D, H, W]
        >>> train_slices, val_slices = split_volume_train_val(volume_shape, train_ratio=0.8)
        >>> # train_slices = (slice(0, 80), slice(None), slice(None))
        >>> # val_slices = (slice(80, 100), slice(None), slice(None))
    """
    # Validate inputs
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if axis < 0 or axis >= len(volume_shape):
        raise ValueError(f"axis must be between 0 and {len(volume_shape) - 1}, got {axis}")

    # Calculate split point
    split_dim = volume_shape[axis]
    train_size = int(split_dim * train_ratio)
    val_size = split_dim - train_size

    # Ensure minimum validation size
    if min_val_size is not None and val_size < min_val_size:
        train_size = split_dim - min_val_size
        val_size = min_val_size
        actual_ratio = train_size / split_dim
        print(
            f"Warning: Adjusted train_ratio from {train_ratio:.2f} to {actual_ratio:.2f} "
            f"to satisfy min_val_size={min_val_size}"
        )

    # Create slices
    train_slices = [slice(None)] * len(volume_shape)
    val_slices = [slice(None)] * len(volume_shape)

    train_slices[axis] = slice(0, train_size)
    val_slices[axis] = slice(train_size, split_dim)

    return tuple(train_slices), tuple(val_slices)


def create_split_masks(
    volume_shape: Tuple[int, int, int],
    train_ratio: float = 0.8,
    axis: int = 0,
    min_val_size: Optional[int] = None,
    dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary masks for train/val splits.

    Similar to DeepEM's msk_train.h5 and msk_val.h5 approach, but computed
    automatically based on train_ratio.

    Args:
        volume_shape: Shape of the volume (D, H, W)
        train_ratio: Ratio of volume to use for training (default: 0.8)
        axis: Axis along which to split (0=D, 1=H, 2=W). Default: 0
        min_val_size: Minimum size for validation split (default: None)
        dtype: Data type for masks (default: np.uint8)

    Returns:
        train_mask: Binary mask for training region (1=valid, 0=invalid)
        val_mask: Binary mask for validation region (1=valid, 0=invalid)

    Example:
        >>> volume_shape = (100, 256, 256)
        >>> train_mask, val_mask = create_split_masks(volume_shape, train_ratio=0.8)
        >>> print(train_mask.shape, train_mask.sum())  # (100, 256, 256), first 80 slices = 1
        >>> print(val_mask.shape, val_mask.sum())      # (100, 256, 256), last 20 slices = 1
    """
    train_slices, val_slices = split_volume_train_val(volume_shape, train_ratio, axis, min_val_size)

    # Create masks
    train_mask = np.zeros(volume_shape, dtype=dtype)
    val_mask = np.zeros(volume_shape, dtype=dtype)

    train_mask[train_slices] = 1
    val_mask[val_slices] = 1

    return train_mask, val_mask


def pad_volume_to_size(
    volume: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int, int],
    mode: str = "reflect",
    constant_value: float = 0.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad a volume to match a target size.

    This is useful when validation data is smaller than the model's input size
    and needs to be padded for inference.

    Args:
        volume: Input volume of shape (D, H, W) or (C, D, H, W)
        target_size: Target size (D, H, W)
        mode: Padding mode. Options:
            - 'constant': Pad with constant value
            - 'reflect': Reflect values at boundaries
            - 'replicate': Replicate edge values
            - 'circular': Circular padding
        constant_value: Value to use for constant padding (default: 0.0)

    Returns:
        Padded volume of shape matching target_size (+ channel dim if input had it)

    Example:
        >>> volume = np.random.randn(18, 160, 160)  # Small validation volume
        >>> target_size = (32, 256, 256)  # Model input size
        >>> padded = pad_volume_to_size(volume, target_size, mode='reflect')
        >>> print(padded.shape)  # (32, 256, 256)
    """
    is_tensor = isinstance(volume, torch.Tensor)

    # Handle channel dimension
    has_channel = volume.ndim == 4
    if has_channel:
        if is_tensor:
            c, d, h, w = volume.shape
        else:
            c, d, h, w = volume.shape
        spatial_shape = (d, h, w)
    else:
        if is_tensor:
            d, h, w = volume.shape
        else:
            d, h, w = volume.shape
        spatial_shape = (d, h, w)

    # Calculate padding needed
    pad_d = max(0, target_size[0] - spatial_shape[0])
    pad_h = max(0, target_size[1] - spatial_shape[1])
    pad_w = max(0, target_size[2] - spatial_shape[2])

    # No padding needed
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume

    # Calculate symmetric padding (before, after) for each dimension
    pad_d_before = pad_d // 2
    pad_d_after = pad_d - pad_d_before
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    if is_tensor:
        # PyTorch padding: (left, right, top, bottom, front, back)
        # Order is reversed: last dim first
        padding = (
            pad_w_before,
            pad_w_after,  # W
            pad_h_before,
            pad_h_after,  # H
            pad_d_before,
            pad_d_after,  # D
        )

        # Map mode names
        if mode == "constant":
            padded = torch.nn.functional.pad(volume, padding, mode="constant", value=constant_value)
        elif mode == "reflect":
            padded = torch.nn.functional.pad(volume, padding, mode="reflect")
        elif mode == "replicate":
            padded = torch.nn.functional.pad(volume, padding, mode="replicate")
        elif mode == "circular":
            padded = torch.nn.functional.pad(volume, padding, mode="circular")
        else:
            raise ValueError(f"Unknown padding mode: {mode}")
    else:
        # NumPy padding
        if has_channel:
            pad_width = (
                (0, 0),  # C - no padding
                (pad_d_before, pad_d_after),  # D
                (pad_h_before, pad_h_after),  # H
                (pad_w_before, pad_w_after),  # W
            )
        else:
            pad_width = (
                (pad_d_before, pad_d_after),  # D
                (pad_h_before, pad_h_after),  # H
                (pad_w_before, pad_w_after),  # W
            )

        # Map mode names
        np_mode = mode
        if mode == "reflect":
            np_mode = "reflect"
        elif mode == "replicate":
            np_mode = "edge"
        elif mode == "circular":
            np_mode = "wrap"
        elif mode == "constant":
            padded = np.pad(volume, pad_width, mode="constant", constant_values=constant_value)
            return padded
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

        padded = np.pad(volume, pad_width, mode=np_mode)

    return padded


def split_and_pad_volume(
    volume: Union[np.ndarray, torch.Tensor],
    train_ratio: float = 0.8,
    target_size: Optional[Tuple[int, int, int]] = None,
    axis: int = 0,
    pad_mode: str = "reflect",
    min_val_size: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Split volume into train/val and pad validation if needed.

    This combines splitting and padding in one operation, following DeepEM's
    approach of using 80% for training and padding the smaller validation
    volume to match model input requirements.

    Args:
        volume: Input volume (D, H, W) or (C, D, H, W)
        train_ratio: Ratio for training split (default: 0.8)
        target_size: Target size for padding validation (default: None, no padding)
        axis: Split axis (default: 0 for Z-axis)
        pad_mode: Padding mode for validation volume (default: 'reflect')
        min_val_size: Minimum validation size (default: None)

    Returns:
        train_volume: Training portion (unchanged)
        val_volume: Validation portion (padded to target_size if specified)

    Example:
        >>> # Volume with 100 slices
        >>> volume = np.random.randn(100, 256, 256)
        >>>
        >>> # Split 80/20 and pad validation to model size
        >>> train_vol, val_vol = split_and_pad_volume(
        ...     volume,
        ...     train_ratio=0.8,
        ...     target_size=(32, 256, 256),  # Model input size
        ...     pad_mode='reflect'
        ... )
        >>>
        >>> print(train_vol.shape)  # (80, 256, 256) - first 80 slices
        >>> print(val_vol.shape)    # (32, 256, 256) - last 20 slices, padded to 32
    """
    # Determine spatial shape
    has_channel = volume.ndim == 4
    if has_channel:
        spatial_shape = volume.shape[1:]
    else:
        spatial_shape = volume.shape

    # Get split slices
    train_slices, val_slices = split_volume_train_val(
        spatial_shape, train_ratio, axis, min_val_size
    )

    # Split volume
    train_volume = volume[train_slices] if not has_channel else volume[:][train_slices]
    val_volume = volume[val_slices] if not has_channel else volume[:][val_slices]

    # Pad validation if target size specified
    if target_size is not None:
        val_volume = pad_volume_to_size(val_volume, target_size, mode=pad_mode)

    return train_volume, val_volume


# Convenience function for saving masks to HDF5 (DeepEM style)
def save_split_masks_h5(
    output_dir: Union[str, Path],
    volume_shape: Tuple[int, int, int],
    train_ratio: float = 0.8,
    axis: int = 0,
    train_filename: str = "msk_train.h5",
    val_filename: str = "msk_val.h5",
):
    """
    Save train/val split masks to HDF5 files (DeepEM compatible).

    Args:
        output_dir: Directory to save mask files
        volume_shape: Shape of the volume (D, H, W)
        train_ratio: Training split ratio (default: 0.8)
        axis: Split axis (default: 0)
        train_filename: Filename for training mask (default: 'msk_train.h5')
        val_filename: Filename for validation mask (default: 'msk_val.h5')
    """
    import h5py

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create masks
    train_mask, val_mask = create_split_masks(volume_shape, train_ratio, axis)

    # Save to HDF5
    train_path = output_dir / train_filename
    val_path = output_dir / val_filename

    with h5py.File(train_path, "w") as f:
        f.create_dataset("main", data=train_mask, compression="gzip")

    with h5py.File(val_path, "w") as f:
        f.create_dataset("main", data=val_mask, compression="gzip")

    print(f"Saved training mask to: {train_path}")
    print(f"Saved validation mask to: {val_path}")
    print(f"Training region: {train_mask.sum()} voxels")
    print(f"Validation region: {val_mask.sum()} voxels")


# ============================================================================
# MONAI Transform for applying splits during data loading
# ============================================================================

try:
    from monai.config import KeysCollection
    from monai.transforms import MapTransform

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


if HAS_MONAI:

    class ApplyVolumetricSplitd(MapTransform):
        """
        MONAI transform to apply volumetric split during data loading.

        This transform extracts a spatial region from loaded volumes based on
        split metadata stored in the data dictionary. It supports automatic
        padding for validation regions.

        Expected metadata in data dict:
            - 'split_slices': Tuple of slice objects defining the region
            - 'split_mode': 'train' or 'val'
            - 'split_pad' (optional): Whether to pad validation
            - 'split_pad_size' (optional): Target size for padding
            - 'split_pad_mode' (optional): Padding mode ('reflect', etc.)

        Args:
            keys: Keys of data dict to apply split to (e.g., ['image', 'label'])
            allow_missing_keys: Don't raise error if key is missing

        Example:
            >>> from monai.transforms import Compose
            >>> transforms = Compose([
            ...     LoadImaged(keys=['image', 'label']),
            ...     ApplyVolumetricSplitd(keys=['image', 'label']),
            ...     # ... other transforms
            ... ])
        """

        def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
        ):
            super().__init__(keys, allow_missing_keys)

        def __call__(self, data: dict) -> dict:
            """Apply volumetric split to data dictionary."""
            d = dict(data)

            # Check if split metadata exists
            if "split_slices" not in d:
                return d  # No split to apply

            split_slices = d["split_slices"]
            split_mode = d.get("split_mode", "train")

            # Apply split to each key
            for key in self.key_iterator(d):
                if key in d:
                    volume = d[key]

                    # Apply slicing
                    # Handle both (C, D, H, W) and (D, H, W) shapes
                    if volume.ndim == 4:
                        # Has channel dimension - apply split to spatial dims
                        d[key] = volume[:, split_slices[0], split_slices[1], split_slices[2]]
                    elif volume.ndim == 3:
                        # No channel dimension - apply directly
                        d[key] = volume[split_slices]
                    else:
                        raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")

            # Apply padding if validation mode and padding requested
            if split_mode == "val" and d.get("split_pad", False):
                target_size = d.get("split_pad_size")
                pad_mode = d.get("split_pad_mode", "reflect")

                if target_size is not None:
                    for key in self.key_iterator(d):
                        if key in d:
                            d[key] = pad_volume_to_size(
                                d[key],
                                target_size=target_size,
                                mode=pad_mode,
                            )

            return d

    def apply_volumetric_split(
        volume: Union[np.ndarray, torch.Tensor],
        split_slices: Tuple[slice, ...],
        pad: bool = False,
        pad_size: Optional[Tuple[int, ...]] = None,
        pad_mode: str = "reflect",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply volumetric split to a single volume (convenience function).

        Args:
            volume: Input volume (D, H, W) or (C, D, H, W)
            split_slices: Tuple of slice objects
            pad: Whether to pad after splitting
            pad_size: Target size for padding (only if pad=True)
            pad_mode: Padding mode ('reflect', 'replicate', 'constant')

        Returns:
            Split (and optionally padded) volume
        """
        # Apply split
        if volume.ndim == 4:
            result = volume[:, split_slices[0], split_slices[1], split_slices[2]]
        elif volume.ndim == 3:
            result = volume[split_slices]
        else:
            raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")

        # Apply padding if requested
        if pad and pad_size is not None:
            result = pad_volume_to_size(result, target_size=pad_size, mode=pad_mode)

        return result


__all__ = [
    "split_volume_train_val",
    "create_split_masks",
    "pad_volume_to_size",
    "split_and_pad_volume",
    "save_split_masks_h5",
    "apply_volumetric_split",
]

if HAS_MONAI:
    __all__.append("ApplyVolumetricSplitd")
