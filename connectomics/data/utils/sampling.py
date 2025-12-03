"""Sampling utility functions for volumetric data."""

import numpy as np
from typing import Tuple, Union, List


def count_volume(
    data_size: Union[np.ndarray, Tuple[int, int, int], List[int]],
    patch_size: Union[np.ndarray, Tuple[int, int, int], List[int]],
    stride: Union[np.ndarray, Tuple[int, int, int], List[int]],
) -> np.ndarray:
    """
    Calculate the number of patches that can be extracted from a volume.

    This function computes how many non-overlapping or overlapping patches
    of a given size can be extracted from a volume using a specified stride.

    Args:
        data_size: Size of the input volume (z, y, x)
        patch_size: Size of each patch (z, y, x)
        stride: Stride for sampling (z, y, x)

    Returns:
        Array of shape (3,) containing the number of patches along each dimension

    Examples:
        >>> data_size = np.array([165, 768, 1024])
        >>> patch_size = np.array([112, 112, 112])
        >>> stride = np.array([1, 1, 1])
        >>> count = count_volume(data_size, patch_size, stride)
        >>> # count = [54, 657, 913] along z, y, x
        >>> total_samples = np.prod(count)  # Total possible patches

    Note:
        The formula is: 1 + ceil((data_size - patch_size) / stride)
        This matches the legacy PyTorch Connectomics v1 implementation.
    """
    data_size = np.array(data_size)
    patch_size = np.array(patch_size)
    stride = np.array(stride).astype(float)

    # Calculate number of patches along each dimension
    # Formula: 1 + ceil((data_size - patch_size) / stride)
    num_patches = 1 + np.ceil((data_size - patch_size) / stride).astype(int)

    return num_patches


def compute_total_samples(
    volume_sizes: List[Tuple[int, int, int]],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tuple[int, List[int]]:
    """
    Compute total number of samples across multiple volumes.

    Args:
        volume_sizes: List of volume sizes [(z1, y1, x1), (z2, y2, x2), ...]
        patch_size: Size of each patch (z, y, x)
        stride: Stride for sampling (z, y, x)

    Returns:
        Tuple of (total_samples, samples_per_volume)
        - total_samples: Total number of possible patches across all volumes
        - samples_per_volume: List of sample counts per volume

    Examples:
        >>> volume_sizes = [(165, 768, 1024)]
        >>> patch_size = (112, 112, 112)
        >>> stride = (1, 1, 1)
        >>> total, per_vol = compute_total_samples(volume_sizes, patch_size, stride)
        >>> print(f"Total samples: {total}")
        >>> # Total samples: 32,380,302 (54 * 657 * 913)
    """
    samples_per_volume = []

    for vol_size in volume_sizes:
        # Count patches per dimension
        num_patches = count_volume(vol_size, patch_size, stride)
        # Total patches for this volume = product of all dimensions
        total_patches = int(np.prod(num_patches))
        samples_per_volume.append(total_patches)

    total_samples = sum(samples_per_volume)

    return total_samples, samples_per_volume


def calculate_inference_grid(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Calculate grid of patch positions for sliding-window inference.

    This function generates all patch positions needed to cover a volume
    with overlapping patches, using the specified stride.

    Args:
        volume_shape: Shape of the input volume (D, H, W)
        patch_size: Size of each patch (D, H, W)
        stride: Stride between patch centers (D, H, W)

    Returns:
        positions: Array of shape (N, 3) containing (z, y, x) start positions
        grid_shape: Tuple (num_z, num_y, num_x) indicating grid dimensions

    Examples:
        >>> volume_shape = (256, 256, 256)
        >>> patch_size = (128, 128, 128)
        >>> stride = (64, 64, 64)
        >>> positions, grid = calculate_inference_grid(volume_shape, patch_size, stride)
        >>> print(f"Grid shape: {grid}")
        >>> # Grid shape: (3, 3, 3)
        >>> print(f"Total patches: {len(positions)}")
        >>> # Total patches: 27

    Note:
        The last patch in each dimension is "tucked in" to ensure it fits
        within the volume boundaries, matching the legacy v1 behavior.
    """
    volume_shape = np.array(volume_shape)
    patch_size = np.array(patch_size)
    stride = np.array(stride)

    # Calculate grid dimensions
    grid_shape = count_volume(volume_shape, patch_size, stride)
    grid_shape = tuple(grid_shape)

    positions = []

    # Generate all grid positions
    for z_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            for x_idx in range(grid_shape[2]):
                # Calculate position with boundary handling
                # Normal case: multiply by stride
                # Boundary case: tuck in to ensure patch fits
                z = (
                    z_idx * stride[0]
                    if z_idx < grid_shape[0] - 1
                    else volume_shape[0] - patch_size[0]
                )
                y = (
                    y_idx * stride[1]
                    if y_idx < grid_shape[1] - 1
                    else volume_shape[1] - patch_size[1]
                )
                x = (
                    x_idx * stride[2]
                    if x_idx < grid_shape[2] - 1
                    else volume_shape[2] - patch_size[2]
                )

                positions.append([z, y, x])

    return np.array(positions, dtype=np.int32), grid_shape


__all__ = [
    "count_volume",
    "compute_total_samples",
    "calculate_inference_grid",
]
