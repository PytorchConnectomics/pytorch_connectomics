"""
Consolidated I/O operations for all formats.

This module provides I/O functions for:
- HDF5 files (.h5, .hdf5)
- Image files (PNG, TIFF)
- NIfTI files (.nii, .nii.gz)
- High-level volume operations
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Optional, Union

import h5py
import imageio
import numpy as np

from .utils import rgb_to_seg

logger = logging.getLogger(__name__)


# =============================================================================
# Format detection
# =============================================================================


def _detect_format(filename: str) -> str:
    """Detect file format from filename.

    Returns canonical format string:
    'h5', 'tiff', 'png', 'nifti', 'zarr'.
    """
    if ".zarr" in filename:
        return "zarr"
    if filename.endswith(".nii.gz"):
        return "nifti"
    suffix = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )
    _SUFFIX_MAP = {
        "h5": "h5",
        "hdf5": "h5",
        "tif": "tiff",
        "tiff": "tiff",
        "png": "png",
        "nii": "nifti",
    }
    fmt = _SUFFIX_MAP.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unrecognizable file format for {filename}. "
            f"Expected: h5, hdf5, tif, tiff, png, nii, nii.gz"
        )
    return fmt


# =============================================================================
# HDF5 I/O
# =============================================================================


def read_hdf5(
    filename: str,
    dataset: Optional[str] = None,
    slice_obj: Optional[tuple] = None,
) -> np.ndarray:
    """Read data from HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        dataset: Dataset name. If None, reads the first dataset.
        slice_obj: Optional slice for partial loading.
    """
    with h5py.File(filename, "r") as fh:
        if dataset is None:
            dataset = list(fh)[0]
        if slice_obj is not None:
            return np.array(fh[dataset][slice_obj])
        return np.array(fh[dataset])


def write_hdf5(
    filename: str,
    data_array: Union[np.ndarray, List[np.ndarray]],
    dataset: Union[str, List[str]] = "main",
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Write data to HDF5 file."""
    with h5py.File(filename, "w") as fh:
        if isinstance(dataset, list):
            for i, name in enumerate(dataset):
                _opts = (
                    compression_level
                    if compression == "gzip"
                    else None
                )
                fh.create_dataset(
                    name,
                    data=data_array[i],
                    compression=compression,
                    compression_opts=_opts,
                    dtype=data_array[i].dtype,
                )
        else:
            _opts = (
                compression_level
                if compression == "gzip"
                else None
            )
            fh.create_dataset(
                dataset,
                data=data_array,
                compression=compression,
                compression_opts=_opts,
                dtype=data_array.dtype,
            )


# =============================================================================
# Image I/O (internal helpers)
# =============================================================================


def _read_image(
    filename: str,
    image_type: str = "image",
) -> np.ndarray:
    """Read a single image file.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Image file not found: {filename}"
        )
    image = imageio.imread(filename)
    if image_type == "seg" and image.ndim == 3:
        image = rgb_to_seg(image)
    return image


def _read_image_with_channel(
    filename: str,
    image_type: str = "image",
) -> Optional[np.ndarray]:
    """Read image and add trailing channel dim. Returns None
    if file does not exist (used by tile reconstruction)."""
    if not os.path.exists(filename):
        return None
    image = imageio.imread(filename)
    if image_type == "seg" and image.ndim == 3:
        image = rgb_to_seg(image)
    if image.ndim == 2:
        image = image[:, :, None]
    return image


def read_images(
    filename_pattern: str,
    image_type: str = "image",
) -> np.ndarray:
    """Read multiple images from a glob pattern.

    Returns stacked array with shape (N, H, W) or (N, H, W, C).
    """
    file_list = sorted(glob.glob(filename_pattern))
    if len(file_list) == 0:
        raise ValueError(
            f"No files found matching: {filename_pattern}"
        )
    first = _read_image(file_list[0], image_type=image_type)
    data = np.zeros(
        (len(file_list), *first.shape), dtype=first.dtype
    )
    data[0] = first
    for i, fp in enumerate(file_list[1:], start=1):
        data[i] = _read_image(fp, image_type=image_type)
    return data


# =============================================================================
# TIFF helpers
# =============================================================================


def _tiff_series_are_stackable(tif) -> bool:
    """True when multiple TIFF series can be depth-stacked."""
    if len(tif.series) <= 1:
        return False
    ref = tif.series[0]
    ref_shape = tuple(ref.shape)
    ref_axes = ref.axes
    ref_dtype = ref.dtype
    for s in tif.series[1:]:
        if (
            tuple(s.shape) != ref_shape
            or s.axes != ref_axes
            or s.dtype != ref_dtype
        ):
            return False
    return True


def _read_tiff_volume(filename: str) -> np.ndarray:
    """Read TIFF volume with robust multi-page handling."""
    try:
        import tifffile
    except ModuleNotFoundError:
        return imageio.volread(filename).squeeze()

    with tifffile.TiffFile(filename) as tif:
        if len(tif.pages) == 0:
            raise ValueError(
                f"TIFF file has no pages: {filename}"
            )
        if len(tif.series) == 0:
            data = tif.pages[0].asarray()
        elif _tiff_series_are_stackable(tif):
            data = tif.asarray(key=slice(None))
        else:
            data = tif.series[0].asarray()

    return np.asarray(data).squeeze()


def _get_tiff_volume_shape(filename: str) -> tuple:
    """Get TIFF volume shape from metadata."""
    try:
        import tifffile
    except ModuleNotFoundError:
        data = imageio.volread(filename).squeeze()
        return tuple(data.shape)

    with tifffile.TiffFile(filename) as tif:
        if len(tif.pages) == 0:
            raise ValueError(
                f"TIFF file has no pages: {filename}"
            )
        if len(tif.series) == 0:
            return tuple(tif.pages[0].shape)
        if _tiff_series_are_stackable(tif):
            return (
                len(tif.series),
                *tuple(tif.series[0].shape),
            )
        return tuple(tif.series[0].shape)


def _normalize_4d_volume(data: np.ndarray) -> np.ndarray:
    """Normalize a 4D volume to channel-first (C, D, H, W).

    Heuristic: the smallest dimension is the channel dim.
    - If axis 0 is smallest -> already (C, D, H, W)
    - If axis 3 is smallest -> (D, H, W, C) -> transpose
    - If axis 1 is smallest -> (D, C, H, W) -> transpose
    """
    if data.ndim != 4:
        return data
    min_axis = int(np.argmin(data.shape))
    if min_axis == 0:
        return data  # Already (C, D, H, W)
    if min_axis == 3:
        # (D, H, W, C) -> (C, D, H, W)
        return data.transpose(3, 0, 1, 2)
    if min_axis == 1:
        # (D, C, H, W) -> (C, D, H, W)
        return data.transpose(1, 0, 2, 3)
    return data


# =============================================================================
# NIfTI helpers (lazy import)
# =============================================================================


def _read_nifti(filename: str) -> np.ndarray:
    """Read NIfTI volume, converting to (D, H, W) or
    (C, D, H, W)."""
    import nibabel as nib

    nii_img = nib.load(filename)
    data = np.asarray(nii_img.dataobj)
    # NIfTI is (X, Y, Z) -> (Z, Y, X) = (D, H, W)
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    elif data.ndim == 4:
        # (X, Y, Z, C) -> (C, Z, Y, X) = (C, D, H, W)
        data = data.transpose(3, 2, 1, 0)
    return data


def _write_nifti(
    filename: str, volume: np.ndarray
) -> None:
    """Write NIfTI volume."""
    import nibabel as nib

    if volume.ndim == 3:
        nii_data = volume.transpose(2, 1, 0)
    elif volume.ndim == 4:
        nii_data = volume.transpose(3, 2, 1, 0)
    else:
        nii_data = volume
    nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4))
    nib.save(nii_img, filename)


def _get_nifti_shape(filename: str) -> tuple:
    """Get NIfTI shape, converted to our convention."""
    import nibabel as nib

    nii_img = nib.load(filename)
    s = nii_img.header.get_data_shape()
    if len(s) == 3:
        return (s[2], s[1], s[0])
    if len(s) == 4:
        return (s[3], s[2], s[1], s[0])
    return s


# =============================================================================
# High-level Volume I/O
# =============================================================================


def read_volume(
    filename: str,
    dataset: Optional[str] = None,
    drop_channel: bool = False,
) -> np.ndarray:
    """Load volumetric data (HDF5, TIFF, PNG, NIfTI).

    Returns array with shape (D, H, W) or (C, D, H, W).
    """
    fmt = _detect_format(filename)

    if fmt == "h5":
        data = read_hdf5(filename, dataset)

    elif fmt == "tiff":
        if "*" in filename or "?" in filename:
            file_list = sorted(glob.glob(filename))
            if not file_list:
                raise FileNotFoundError(
                    "No TIFF files found matching: "
                    f"{filename}"
                )
            volumes = []
            for fp in file_list:
                vol = _read_tiff_volume(fp)
                if vol.ndim == 2:
                    vol = vol[np.newaxis, ...]
                volumes.append(vol)
            data = np.concatenate(volumes, axis=0)
        else:
            data = _read_tiff_volume(filename)
        if data.ndim == 4:
            data = _normalize_4d_volume(data)

    elif fmt == "png":
        data = read_images(filename)
        if data.ndim == 4:
            # (D, H, W, C) -> (C, D, H, W)
            data = data.transpose(3, 0, 1, 2)

    elif fmt == "nifti":
        data = _read_nifti(filename)

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if data.ndim not in (2, 3, 4):
        raise ValueError(
            f"Expected 2D/3D/4D data, got {data.ndim}D"
        )

    if drop_channel and data.ndim == 4:
        original_dtype = data.dtype
        data = np.mean(data, axis=0).astype(original_dtype)

    return data


def save_volume(
    filename: str,
    volume: np.ndarray,
    dataset: str = "main",
    file_format: str = "h5",
) -> None:
    """Save volumetric data in specified format.

    Args:
        filename: Output filename or directory path.
        volume: Volume data to save.
        dataset: Dataset name for HDF5 format.
        file_format: 'h5', 'tiff', 'png', 'nii', 'nii.gz'.
    """
    if file_format == "h5":
        write_hdf5(filename, volume, dataset=dataset)

    elif file_format in ("tif", "tiff"):
        import tifffile

        if volume.ndim == 4:
            # (C, D, H, W) -> (D, H, W, C)
            tiff_data = volume.transpose(1, 2, 3, 0)
        else:
            tiff_data = volume
        tifffile.imwrite(
            filename,
            tiff_data,
            compression="zlib",
            photometric="minisblack",
        )

    elif file_format == "png":
        _save_images(filename, volume)

    elif file_format in ("nii", "nii.gz"):
        _write_nifti(filename, volume)

    else:
        raise ValueError(
            f"Unsupported format: {file_format}. "
            f"Expected: h5, tiff, png, nii, nii.gz"
        )


def _save_images(
    directory: str,
    data: np.ndarray,
    prefix: str = "",
    fmt: str = "png",
) -> None:
    """Save a stack of images to a directory."""
    os.makedirs(directory, exist_ok=True)
    for i in range(data.shape[0]):
        path = os.path.join(
            directory, f"{prefix}{i:04d}.{fmt}"
        )
        imageio.imsave(path, data[i])


def get_vol_shape(
    filename: str,
    dataset: Optional[str] = None,
) -> tuple:
    """Get volume shape without loading data.

    Returns shape consistent with what read_volume would
    produce: (D, H, W) or (C, D, H, W).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    fmt = _detect_format(filename)

    if fmt == "zarr":
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "zarr required. pip install zarr"
            ) from exc
        obj = zarr.open(filename, mode="r")
        if hasattr(obj, "shape"):
            return tuple(obj.shape)
        if dataset is not None:
            return tuple(obj[dataset].shape)
        keys = list(obj.keys())
        if not keys:
            raise ValueError(
                f"No arrays in zarr group: {filename}"
            )
        return tuple(obj[keys[0]].shape)

    if fmt == "h5":
        with h5py.File(filename, "r") as f:
            if dataset is None:
                dataset = list(f.keys())[0]
            return f[dataset].shape

    if fmt == "tiff":
        return _get_tiff_volume_shape(filename)

    if fmt == "png":
        file_list = sorted(glob.glob(filename))
        if not file_list:
            raise ValueError(
                f"No PNG files found: {filename}"
            )
        first = imageio.imread(file_list[0])
        n = len(file_list)
        if first.ndim == 2:
            return (n, *first.shape)
        if first.ndim == 3:
            # Match read_volume: (C, D, H, W)
            c = first.shape[-1]
            return (c, n, first.shape[0], first.shape[1])
        raise ValueError(
            f"Unsupported PNG dims: {first.ndim}D"
        )

    if fmt == "nifti":
        return _get_nifti_shape(filename)

    raise ValueError(f"Unsupported format: {fmt}")
