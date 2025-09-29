"""
Volume I/O functions for various image formats.
"""

from __future__ import print_function, division
from typing import Optional

# Avoid PIL "IOError: image file truncated"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import h5py
import glob
import numpy as np
import imageio


def read_image_as_volume(filename: str, drop_channel: bool = False) -> np.ndarray:
    """Read a single image file as a volume.

    Args:
        filename: Path to the image file
        drop_channel: Whether to convert multichannel images to grayscale

    Returns:
        Image data as numpy array with shape (c, y, x)
    """
    image_suffix = filename[filename.rfind('.') + 1:]
    assert image_suffix in ['png', 'tif'], f"Unsupported format: {image_suffix}"
    data = imageio.imread(filename)

    if data.ndim == 3 and not drop_channel:
        # convert (y,x,c) to (c,y,x) shape
        data = data.transpose(2, 0, 1)
        return data

    elif drop_channel and data.ndim == 3:
        # convert RGB image to grayscale by average
        data = np.mean(data, axis=-1).astype(np.uint8)

    return data[np.newaxis, :, :]   # return data as (1,y,x) shape


def read_hdf5(filename: str, dataset: Optional[str] = None) -> np.ndarray:
    """Read data from HDF5 file.

    Args:
        filename: Path to the HDF5 file
        dataset: Name of the dataset to read. If None, reads the first dataset

    Returns:
        Data from the HDF5 file as numpy array
    """
    with h5py.File(filename, 'r') as file_handle:
        if dataset is None:
            # load the first dataset in the h5 file
            dataset = list(file_handle)[0]
        return np.array(file_handle[dataset])


def write_hdf5(filename: str, data_array: np.ndarray, dataset: str = 'main') -> None:
    """Write data to HDF5 file.

    Args:
        filename: Path to the output HDF5 file
        data_array: Data to write as numpy array or list of arrays
        dataset: Name of the dataset(s) to create
    """
    with h5py.File(filename, 'w') as file_handle:
        if isinstance(dataset, (list,)):
            for i, dataset_name in enumerate(dataset):
                dataset_obj = file_handle.create_dataset(
                    dataset_name, data_array[i].shape,
                    compression="gzip", dtype=data_array[i].dtype
                )
                dataset_obj[:] = data_array[i]
        else:
            dataset_obj = file_handle.create_dataset(
                dataset, data_array.shape,
                compression="gzip", dtype=data_array.dtype
            )
            dataset_obj[:] = data_array


def read_volume(filename: str, dataset: Optional[str] = None, drop_channel: bool = False) -> np.ndarray:
    """Load volumetric data in HDF5, TIFF or PNG formats.

    Args:
        filename: Path to the volume file
        dataset: HDF5 dataset name (only used for HDF5 files)
        drop_channel: Whether to convert multichannel volumes to single channel

    Returns:
        Volume data as numpy array with shape (z,y,x) or (c,z,y,x)
    """
    image_suffix = filename[filename.rfind('.') + 1:]

    if image_suffix in ['h5', 'hdf5']:
        data = read_hdf5(filename, dataset)
    elif 'tif' in image_suffix:
        data = imageio.volread(filename).squeeze()
        if data.ndim == 4:
            # convert (z,c,y,x) to (c,z,y,x) order
            data = data.transpose(1, 0, 2, 3)
    elif 'png' in image_suffix:
        data = read_images(filename)
        if data.ndim == 4:
            # convert (z,y,x,c) to (c,z,y,x) order
            data = data.transpose(3, 0, 1, 2)
    else:
        raise ValueError(f'Unrecognizable file format for {filename}')

    assert data.ndim in [3, 4], (
        f"Currently supported volume data should be 3D (z,y,x) or 4D (c,z,y,x), "
        f"got {data.ndim}D"
    )

    if drop_channel and data.ndim == 4:
        # merge multiple channels to grayscale by average
        original_dtype = data.dtype
        data = np.mean(data, axis=0).astype(original_dtype)

    return data


def save_volume(filename: str, volume: np.ndarray, dataset: str = 'main',
               file_format: str = 'h5') -> None:
    """Save volumetric data in specified format.

    Args:
        filename: Output filename or directory path
        volume: Volume data to save
        dataset: Dataset name for HDF5 format
        file_format: Output format ('h5' or 'png')
    """
    if file_format == 'h5':
        write_hdf5(filename, volume, dataset=dataset)
    elif file_format == 'png':
        current_directory = os.getcwd()
        image_save_path = os.path.join(current_directory, filename)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        for i in range(volume.shape[0]):
            imageio.imsave(f'{image_save_path}/{i:04d}.png', volume[i])
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def read_image(filename: str, add_channel: bool = False) -> Optional[np.ndarray]:
    """Read a single image file.

    Args:
        filename: Path to the image file
        add_channel: Whether to add a channel dimension for grayscale images

    Returns:
        Image data as numpy array with shape (y, x) or (y, x, c), or None if file doesn't exist
    """
    if not os.path.exists(filename):
        return None

    image = imageio.imread(filename)
    if add_channel and image.ndim == 2:
        image = image[:, :, None]
    return image


def read_images(filename_pattern: str) -> np.ndarray:
    """Read multiple images from a filename pattern.

    Args:
        filename_pattern: Glob pattern for matching image files

    Returns:
        Stack of images as numpy array with shape (n, y, x) or (n, y, x, c)
    """
    file_list = sorted(glob.glob(filename_pattern))
    num_images = len(file_list)

    if num_images == 0:
        raise ValueError(f"No files found matching pattern: {filename_pattern}")

    # Determine array shape from first image
    first_image = imageio.imread(file_list[0])
    if first_image.ndim == 2:
        data = np.zeros((num_images, first_image.shape[0], first_image.shape[1]), dtype=np.uint8)
    elif first_image.ndim == 3:
        data = np.zeros((num_images, first_image.shape[0], first_image.shape[1], first_image.shape[2]), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image dimensions: {first_image.ndim}")

    data[0] = first_image

    # Load remaining images
    for i in range(1, num_images):
        data[i] = imageio.imread(file_list[i])

    return data


__all__ = [
    'read_image_as_volume', 'read_hdf5', 'write_hdf5', 'read_volume', 'save_volume',
    'read_image', 'read_images'
]