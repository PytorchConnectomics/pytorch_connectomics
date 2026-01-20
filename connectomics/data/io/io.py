"""
Consolidated I/O operations for all formats.

This module provides I/O functions for:
- HDF5 files (.h5, .hdf5)
- Image files (PNG, TIFF)
- Pickle files (.pkl)
- High-level volume operations
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import List, Optional, Union

import h5py
import imageio
import nibabel as nib
import numpy as np

from .utils import rgb_to_seg

# =============================================================================
# HDF5 I/O
# =============================================================================


def read_hdf5(
    filename: str, dataset: Optional[str] = None, slice_obj: Optional[tuple] = None
) -> np.ndarray:
    """Read data from HDF5 file.

    Args:
        filename: Path to the HDF5 file
        dataset: Name of the dataset to read. If None, reads the first dataset
        slice_obj: Optional slice for partial loading (e.g., np.s_[0:10,:,:])

    Returns:
        Data from the HDF5 file as numpy array
    """
    with h5py.File(filename, "r") as file_handle:
        if dataset is None:
            dataset = list(file_handle)[0]

        if slice_obj is not None:
            return np.array(file_handle[dataset][slice_obj])
        return np.array(file_handle[dataset])


def write_hdf5(
    filename: str,
    data_array: Union[np.ndarray, List[np.ndarray]],
    dataset: Union[str, List[str]] = "main",
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Write data to HDF5 file.

    Args:
        filename: Path to the output HDF5 file
        data_array: Data to write as numpy array or list of arrays
        dataset: Name of the dataset(s) to create
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level (0-9 for gzip)
    """
    with h5py.File(filename, "w") as file_handle:
        if isinstance(dataset, list):
            for i, dataset_name in enumerate(dataset):
                file_handle.create_dataset(
                    dataset_name,
                    data=data_array[i],
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                    dtype=data_array[i].dtype,
                )
        else:
            file_handle.create_dataset(
                dataset,
                data=data_array,
                compression=compression,
                compression_opts=compression_level if compression == "gzip" else None,
                dtype=data_array.dtype,
            )


def list_hdf5_datasets(filename: str) -> List[str]:
    """List all datasets in an HDF5 file.

    Args:
        filename: Path to the HDF5 file

    Returns:
        List of dataset names
    """
    with h5py.File(filename, "r") as file_handle:
        return list(file_handle.keys())


# =============================================================================
# Image I/O
# =============================================================================

SUPPORTED_IMAGE_FORMATS = ["png", "tif", "tiff", "jpg", "jpeg"]


def read_image(
    filename: str, add_channel: bool = False, image_type: str = "image"
) -> Optional[np.ndarray]:
    """Read a single image file.

    Args:
        filename: Path to the image file
        add_channel: Whether to add a channel dimension for grayscale images

    Returns:
        Image data as numpy array with shape (H, W) or (H, W, C), or None if file doesn't exist
    """
    if not os.path.exists(filename):
        return None

    image = imageio.imread(filename)
    if image_type == "seg" and image.ndim == 3:
        image = rgb_to_seg(image)
    if add_channel and image.ndim == 2:
        image = image[:, :, None]
    return image


def read_images(filename_pattern: str, image_type: str = "image") -> np.ndarray:
    """Read multiple images from a filename pattern.

    Args:
        filename_pattern: Glob pattern for matching image files

    Returns:
        Stack of images as numpy array with shape (N, H, W) or (N, H, W, C)

    Raises:
        ValueError: If no files found or unsupported dimensions
    """
    file_list = sorted(glob.glob(filename_pattern))
    if len(file_list) == 0:
        raise ValueError(f"No files found matching pattern: {filename_pattern}")

    # Determine array shape from first image
    first_image = read_image(file_list[0], image_type=image_type)
    data = np.zeros((len(file_list), *first_image.shape), dtype=first_image.dtype)
    # Load all images
    for i, filepath in enumerate(file_list):
        data[i] = read_image(filepath, image_type=image_type)

    return data


def read_image_as_volume(filename: str, drop_channel: bool = False) -> np.ndarray:
    """Read a single image file as a volume with channel-first format.

    Args:
        filename: Path to the image file
        drop_channel: Whether to convert multichannel images to grayscale

    Returns:
        Image data as numpy array with shape (C, H, W)

    Raises:
        ValueError: If file format is not supported
    """
    image_suffix = filename[filename.rfind(".") + 1 :].lower()
    if image_suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported format: {image_suffix}. Supported formats: {SUPPORTED_IMAGE_FORMATS}"
        )

    data = imageio.imread(filename)

    if data.ndim == 3 and not drop_channel:
        # Convert (H, W, C) to (C, H, W) shape
        data = data.transpose(2, 0, 1)
        return data

    if drop_channel and data.ndim == 3:
        # Convert RGB image to grayscale by average
        data = np.mean(data, axis=-1).astype(np.uint8)

    return data[np.newaxis, :, :]  # Return data as (1, H, W) shape


def save_image(filename: str, data: np.ndarray) -> None:
    """Save a single image to file.

    Args:
        filename: Output filename
        data: Image data with shape (H, W) or (H, W, C)
    """
    imageio.imsave(filename, data)


def save_images(directory: str, data: np.ndarray, prefix: str = "", format: str = "png") -> None:
    """Save a stack of images to a directory.

    Args:
        directory: Output directory path
        data: Image stack with shape (N, H, W) or (N, H, W, C)
        prefix: Filename prefix (default: '')
        format: Image format (default: 'png')
    """
    os.makedirs(directory, exist_ok=True)

    for i in range(data.shape[0]):
        filename = os.path.join(directory, f"{prefix}{i:04d}.{format}")
        imageio.imsave(filename, data[i])


# =============================================================================
# Pickle I/O
# =============================================================================


def read_pickle_file(filename: str) -> Union[object, List[object]]:
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


def write_pickle_file(filename: str, data: object) -> None:
    """Write data to a pickle file.

    Args:
        filename: Path to the output pickle file
        data: Data to pickle
    """
    with open(filename, "wb") as file_handle:
        pickle.dump(data, file_handle)


# =============================================================================
# High-level Volume I/O
# =============================================================================


def read_volume(
    filename: str, dataset: Optional[str] = None, drop_channel: bool = False
) -> np.ndarray:
    """Load volumetric data in HDF5, TIFF, PNG, or NIfTI formats.

    Args:
        filename: Path to the volume file
        dataset: HDF5 dataset name (only used for HDF5 files)
        drop_channel: Whether to convert multichannel volumes to single channel

    Returns:
        Volume data as numpy array with shape (D, H, W) or (C, D, H, W)

    Raises:
        ValueError: If file format is not recognized
    """
    # Handle .nii.gz files specially
    if filename.endswith(".nii.gz"):
        image_suffix = "nii.gz"
    else:
        image_suffix = filename[filename.rfind(".") + 1 :].lower()

    if image_suffix in ["h5", "hdf5"]:
        data = read_hdf5(filename, dataset)
    elif "tif" in image_suffix:
        # Check if filename contains glob patterns
        if "*" in filename or "?" in filename:
            # Expand glob pattern to get matching files
            file_list = sorted(glob.glob(filename))
            if len(file_list) == 0:
                raise FileNotFoundError(f"No TIFF files found matching pattern: {filename}")

            # Read each file and stack along depth dimension
            volumes = []
            for filepath in file_list:
                vol = imageio.volread(filepath).squeeze()
                # imageio.volread can return multi-page TIFF as (D, H, W) or single page as (H, W)
                # Ensure all volumes have at least 3D (D, H, W)
                if vol.ndim == 2:
                    vol = vol[np.newaxis, ...]  # Add depth dimension: (H, W) -> (1, H, W)
                # vol.ndim == 3 means (D, H, W), which is what we want
                volumes.append(vol)

            # Stack all volumes along depth dimension
            # Each volume is (D_i, H, W), result will be (sum(D_i), H, W)
            data = np.concatenate(volumes, axis=0)  # Stack along depth (first dimension)
        else:
            # Single file or multi-page TIFF
            data = imageio.volread(filename).squeeze()

        if data.ndim == 4:
            # Convert (D, C, H, W) to (C, D, H, W) order
            data = data.transpose(1, 0, 2, 3)
    elif "png" in image_suffix:
        data = read_images(filename)
        if data.ndim == 4:
            # Convert (D, H, W, C) to (C, D, H, W) order
            data = data.transpose(3, 0, 1, 2)
    elif image_suffix in ["nii", "nii.gz"]:
        # NIfTI format (.nii or .nii.gz)
        nii_img = nib.load(filename)
        data = np.asarray(nii_img.dataobj)
        # NIfTI is typically (X, Y, Z) or (X, Y, Z, C)
        # Convert to our (D, H, W) or (C, D, H, W) format
        # X=W (width), Y=H (height), Z=D (depth)
        if data.ndim == 3:
            # (X, Y, Z) -> (Z, Y, X) = (D, H, W)
            data = data.transpose(2, 1, 0)
        elif data.ndim == 4:
            # (X, Y, Z, C) -> (C, Z, Y, X) = (C, D, H, W)
            data = data.transpose(3, 2, 1, 0)
    else:
        raise ValueError(
            f"Unrecognizable file format for {filename}. "
            f"Expected: h5, hdf5, tif, tiff, png, nii, or nii.gz"
        )

    # if data.ndim not in [3, 4]:
    #     raise ValueError(
    #         f"Currently supported volume data should be 3D (D, H, W) or 4D (C, D, H, W), "
    #         f"got {data.ndim}D"
    #     )

    if drop_channel and data.ndim == 4:
        # Merge multiple channels to grayscale by average
        original_dtype = data.dtype
        data = np.mean(data, axis=0).astype(original_dtype)

    return data


def save_volume(
    filename: str, volume: np.ndarray, dataset: str = "main", file_format: str = "h5"
) -> None:
    """Save volumetric data in specified format.

    Args:
        filename: Output filename or directory path
        volume: Volume data to save
        dataset: Dataset name for HDF5 format
        file_format: Output format ('h5', 'tiff', 'png', 'nii', or 'nii.gz')

    Raises:
        ValueError: If file format is not supported
    """
    if file_format == "h5":
        write_hdf5(filename, volume, dataset=dataset)
    elif file_format in ["tif", "tiff"]:
        # TIFF format - supports both 2D and 3D multi-page TIFF
        import tifffile
        
        # Convert from our internal format to TIFF-compatible format
        # Internal: (D, H, W) or (C, D, H, W)
        # TIFF: (D, H, W) for single-channel or (D, H, W, C) for multi-channel
        if volume.ndim == 3:
            # Single-channel 3D: (D, H, W) - can save directly
            tiff_data = volume
        elif volume.ndim == 4:
            # Multi-channel 3D: (C, D, H, W) -> (D, H, W, C)
            tiff_data = volume.transpose(1, 2, 3, 0)
        elif volume.ndim == 2:
            # Single 2D image: (H, W) - can save directly
            tiff_data = volume
        else:
            tiff_data = volume
        
        # Save with compression for efficiency
        tifffile.imwrite(
            filename, 
            tiff_data,
            compression='zlib',  # Good balance of speed and compression
            photometric='minisblack'  # Grayscale interpretation
        )
    elif file_format == "png":
        save_images(filename, volume)
    elif file_format in ["nii", "nii.gz"]:
        # NIfTI format
        # Convert from our (D, H, W) or (C, D, H, W) to NIfTI (X, Y, Z) or (X, Y, Z, C)
        if volume.ndim == 3:
            # (D, H, W) -> (W, H, D) = (X, Y, Z)
            nii_data = volume.transpose(2, 1, 0)
        elif volume.ndim == 4:
            # (C, D, H, W) -> (W, H, D, C) = (X, Y, Z, C)
            nii_data = volume.transpose(3, 2, 1, 0)
        else:
            nii_data = volume
        nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4))
        nib.save(nii_img, filename)
    else:
        raise ValueError(
            f"Unsupported format: {file_format}. Supported formats: h5, tiff, png, nii, nii.gz"
        )


def get_vol_shape(filename: str, dataset: Optional[str] = None) -> tuple:
    """Get volume shape without loading the entire volume into memory.

    This function efficiently retrieves the shape of volumetric data
    by reading only metadata from the file, not the actual data.

    Args:
        filename: Path to the volume file
        dataset: HDF5 dataset name (only used for HDF5 files, None = first dataset)

    Returns:
        Shape tuple of the volume (e.g., (D, H, W) or (C, D, H, W))

    Raises:
        ValueError: If file format is not recognized
        FileNotFoundError: If file does not exist

    Example:
        >>> shape = get_vol_shape('datasets/train_image.h5')
        >>> print(shape)  # (165, 768, 1024)
        >>>
        >>> shape = get_vol_shape('datasets/train_im.tif')
        >>> print(shape)  # (165, 768, 1024)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Handle .nii.gz files specially
    if filename.endswith(".nii.gz"):
        image_suffix = "nii.gz"
    else:
        image_suffix = filename[filename.rfind(".") + 1 :].lower()

    if image_suffix in ["h5", "hdf5"]:
        # HDF5: Read shape from metadata (no data loading)
        with h5py.File(filename, "r") as f:
            if dataset is None:
                dataset = list(f.keys())[0]
            return f[dataset].shape

    elif "tif" in image_suffix:
        # TIFF: Use imageio to get metadata
        import tifffile

        with tifffile.TiffFile(filename) as tif:
            # Get shape from first series
            if hasattr(tif, "series"):
                # Multi-page TIFF
                shape = tif.series[0].shape
            else:
                # Single page TIFF
                shape = tif.pages[0].shape
            return shape

    elif "png" in image_suffix:
        # PNG stack: Count files and read one image for dimensions
        file_list = sorted(glob.glob(filename))
        if len(file_list) == 0:
            raise ValueError(f"No PNG files found matching pattern: {filename}")

        # Read first image to get spatial dimensions
        first_image = imageio.imread(file_list[0])
        num_slices = len(file_list)

        if first_image.ndim == 2:
            # Grayscale: (D, H, W)
            return (num_slices, *first_image.shape)
        elif first_image.ndim == 3:
            # Color: (D, H, W, C)
            return (num_slices, *first_image.shape)
        else:
            raise ValueError(f"Unsupported PNG dimensions: {first_image.ndim}D")

    elif image_suffix in ["nii", "nii.gz"]:
        # NIfTI: Read shape from header (no data loading)
        nii_img = nib.load(filename)
        nii_shape = nii_img.header.get_data_shape()
        # Convert from NIfTI (X, Y, Z) or (X, Y, Z, C) to our (D, H, W) or (C, D, H, W)
        if len(nii_shape) == 3:
            # (X, Y, Z) -> (Z, Y, X) = (D, H, W)
            return (nii_shape[2], nii_shape[1], nii_shape[0])
        elif len(nii_shape) == 4:
            # (X, Y, Z, C) -> (C, D, H, W)
            return (nii_shape[3], nii_shape[2], nii_shape[1], nii_shape[0])
        else:
            return nii_shape

    else:
        raise ValueError(
            f"Unrecognizable file format for {filename}. "
            f"Expected: h5, hdf5, tif, tiff, png, nii, or nii.gz"
        )


__all__ = [
    # HDF5 I/O
    "read_hdf5",
    "write_hdf5",
    "list_hdf5_datasets",
    # Image I/O
    "read_image",
    "read_images",
    "read_image_as_volume",
    "save_image",
    "save_images",
    "SUPPORTED_IMAGE_FORMATS",
    # Pickle I/O
    "read_pickle_file",
    "write_pickle_file",
    # High-level volume I/O
    "read_volume",
    "save_volume",
    "get_vol_shape",
]
