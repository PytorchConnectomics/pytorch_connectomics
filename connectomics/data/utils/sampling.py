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


__all__ = [
    'count_volume',
    'compute_total_samples',
]
