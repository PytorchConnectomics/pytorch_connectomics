"""
Utility functions for data I/O operations.

This module provides various utility functions for data processing,
conversion, and manipulation in connectomics workflows.
"""

from __future__ import print_function, division
import numpy as np
import pickle


def read_pickle_file(filename: str):
    """Read data from a pickle file.

    Args:
        filename: Path to the pickle file to read

    Returns:
        The data stored in the pickle file. If multiple objects are stored,
        returns a list. If only one object, returns the object directly.
    """
    data = []
    with open(filename, "rb") as file_handle:
        while True:
            try:
                data.append(pickle.load(file_handle))
            except EOFError:
                break

    if len(data) == 1:
        return data[0]
    return data


def vast_to_segmentation(segmentation_data: np.ndarray) -> np.ndarray:
    """Convert VAST segmentation format to standard format.

    VAST format uses RGB encoding where each pixel's RGB values are combined
    to create a unique 24-bit segmentation ID.

    Args:
        segmentation_data: Input segmentation data in VAST format

    Returns:
        Converted segmentation with proper ID encoding
    """
    # Convert to 24 bits
    if segmentation_data.ndim == 2 or segmentation_data.shape[-1] == 1:
        return np.squeeze(segmentation_data)
    elif segmentation_data.ndim == 3:  # Single RGB image
        return (segmentation_data[:, :, 0].astype(np.uint32) * 65536 +
                segmentation_data[:, :, 1].astype(np.uint32) * 256 +
                segmentation_data[:, :, 2].astype(np.uint32))
    elif segmentation_data.ndim == 4:  # Multiple RGB images
        return (segmentation_data[:, :, :, 0].astype(np.uint32) * 65536 +
                segmentation_data[:, :, :, 1].astype(np.uint32) * 256 +
                segmentation_data[:, :, :, 2].astype(np.uint32))


def normalize_data_range(data: np.ndarray, target_min: float = 0.0,
                        target_max: float = 1.0, ignore_uint8: bool = True) -> np.ndarray:
    """Normalize array values to a target range.

    Args:
        data: Input array to normalize
        target_min: Minimum value of target range. Default: 0.0
        target_max: Maximum value of target range. Default: 1.0
        ignore_uint8: Whether to skip normalization for uint8 arrays. Default: True

    Returns:
        Normalized array with values in the target range
    """
    if ignore_uint8 and data.dtype == np.uint8:
        return data

    epsilon = 1e-6
    data_min = data.min()
    data_max = data.max()

    # Avoid division by zero
    if data_max - data_min < epsilon:
        return np.full_like(data, target_min)

    normalized = (data - data_min) / (data_max - data_min + epsilon)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized


def convert_to_uint8(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert data to uint8 format.

    Args:
        data: Input data array
        normalize: Whether to normalize the data to [0, 255] range first. Default: True

    Returns:
        Data converted to uint8 format
    """
    if normalize:
        data = normalize_data_range(data, 0.0, 255.0, ignore_uint8=False)

    return data.astype(np.uint8)


def split_multichannel_mask(label_data: np.ndarray) -> np.ndarray:
    """Split multichannel label data into separate masks.

    Args:
        label_data: Input label data with multiple classes/instances

    Returns:
        Array with shape (num_classes, ...) where each channel
        contains a binary mask for one class/instance
    """
    unique_indices = np.unique(label_data)
    if len(unique_indices) > 1:
        if unique_indices[0] == 0:
            unique_indices = unique_indices[1:]  # Remove background
        masks = [(label_data == idx).astype(np.uint8) for idx in unique_indices]
        return np.stack(masks, 0)

    return np.ones_like(label_data).astype(np.uint8)[np.newaxis]


def squeeze_arrays(*arrays):
    """Squeeze multiple numpy arrays.

    Args:
        *arrays: Variable number of numpy arrays to squeeze

    Returns:
        Tuple of squeezed arrays (or None for None inputs)
    """
    squeezed = []
    for array in arrays:
        if array is not None:
            squeezed.append(np.squeeze(array))
        else:
            squeezed.append(None)
    return squeezed


__all__ = [
    'read_pickle_file', 'vast_to_segmentation', 'normalize_data_range',
    'convert_to_uint8', 'split_multichannel_mask', 'squeeze_arrays'
]