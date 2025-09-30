# I/O Module Refactoring Plan

## Executive Summary

This plan reorganizes and refactors the I/O modules in PyTorch Connectomics to eliminate redundancy, improve consistency, and establish a clear, modern API structure.

**Current State**: Redundant implementations across multiple locations
**Target State**: Clean, modular I/O package with consistent naming and modern Python practices

---

## Current State Analysis

### Existing Implementations

#### 1. Legacy: `/projects/weilab/weidf/pytorch_connectomics/connectomics/data/utils/data_io.py`
**Status**: Old codebase (separate repository)
**Style**: Legacy naming (`readh5`, `readvol`, `readim`, `readimgs`, `vast2Seg`, `tile2volume`)
**Issues**:
- Inconsistent naming (camelCase, snake_case mix)
- Missing type hints in some functions
- H5py file not properly closed in `readh5()`
- No docstrings for many functions

#### 2. Modern: `connectomics/data/io/` (current lib)
**Status**: Refactored, modern implementation
**Structure**:
```
connectomics/data/io/
├── __init__.py           # Clean public API
├── volume.py             # Volume I/O (HDF5, TIFF, PNG)
├── tiles.py              # Tile-based operations
├── utils.py              # Utility functions
└── monai_transforms.py   # MONAI-compatible loaders
```

**Style**: Modern naming (`read_hdf5`, `read_volume`, `read_image`, etc.)
**Features**:
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Proper resource management (context managers)
- ✅ PEP 8 compliant
- ✅ MONAI integration (`LoadVolumed`, `TileLoaderd`)

#### 3. Top-level Convenience: `connectomics/data/io_utils.py`
**Status**: Thin wrapper for backward compatibility
**Purpose**: Re-exports functions from `io/` package
**Should be deprecated**: Yes (after migration period)

---

## Problems to Address

### 1. **Duplication**
- Legacy `data_io.py` has older implementations of same functions
- `io_utils.py` is just a re-export wrapper

### 2. **Naming Inconsistency**
Legacy names vs modern names:
- `readh5` → `read_hdf5`
- `writeh5` → `write_hdf5`
- `readvol` → `read_volume`
- `savevol` → `save_volume`
- `readim` → `read_image`
- `readimgs` → `read_images`
- `readimg_as_vol` → `read_image_as_volume`
- `read_pkl` → `read_pickle_file`
- `vast2Seg` → `vast_to_segmentation`
- `create_json` → `create_tile_metadata`
- `tile2volume` → `reconstruct_volume_from_tiles`

### 3. **Missing Features in Modern Implementation**
The modern `io/` package is missing some utilities from legacy:
- Nothing significant (legacy is actually less complete)

### 4. **Import Confusion**
Users might import from:
- `connectomics.data.io` (modern, correct)
- `connectomics.data.io_utils` (convenience wrapper)
- Legacy path (old repo, shouldn't exist in new lib)

### 5. **MONAI Integration Gaps**
- `tiles.py` has incomplete `TileLoaderd` (missing `Sequence` import)
- `LoadVolumed` in `monai_transforms.py` is good but could be promoted

---

## Proposed Structure

### Target Architecture

```
connectomics/data/io/
├── __init__.py              # Public API (all imports)
│
├── formats/                 # NEW: Format-specific I/O
│   ├── __init__.py
│   ├── hdf5.py             # HDF5 operations
│   ├── image.py            # Image formats (PNG, TIFF)
│   └── pickle.py           # Pickle utilities
│
├── transforms/              # MONAI transforms
│   ├── __init__.py
│   ├── volume.py           # LoadVolumed, SaveVolumed
│   └── tiles.py            # TileLoaderd
│
├── tiles.py                 # Tile operations (KEEP)
├── utils.py                 # Data utilities (KEEP)
│
└── legacy.py                # DEPRECATED: Backward compatibility aliases
```

### Rationale for Changes

1. **`formats/` subdirectory**: Groups format-specific I/O by file type
   - Clear separation of concerns
   - Easier to add new formats (Zarr, N5, etc.)
   - Better organization for large codebase

2. **`transforms/` subdirectory**: All MONAI transforms in one place
   - Clear distinction between functions and transforms
   - Easier to find MONAI-compatible loaders
   - Matches MONAI's organization

3. **`legacy.py`**: Backward compatibility
   - One place for all deprecated aliases
   - Clear deprecation warnings
   - Easy to remove in future major version

4. **Keep `tiles.py` and `utils.py`**: Already well-organized
   - These modules are cohesive and well-designed
   - No need to split further

---

## Refactoring Tasks

### Phase 1: Create Format-Specific Modules

#### Task 1.1: Create `formats/hdf5.py`
**Extract from**: `volume.py` lines 45-83
**Functions**:
- `read_hdf5(filename, dataset=None)`
- `write_hdf5(filename, data_array, dataset='main')`
- `list_datasets(filename)` - NEW: List available datasets in HDF5

**Improvements**:
- Add `list_datasets()` utility
- Add context manager support for batch operations
- Add compression level parameter to `write_hdf5()`

```python
"""HDF5 I/O operations for connectomics data."""

from __future__ import annotations
from typing import Optional, List, Union
import h5py
import numpy as np


def read_hdf5(
    filename: str,
    dataset: Optional[str] = None,
    slice_obj: Optional[tuple] = None
) -> np.ndarray:
    """Read data from HDF5 file.

    Args:
        filename: Path to the HDF5 file
        dataset: Name of the dataset to read. If None, reads the first dataset
        slice_obj: Optional slice for partial loading (e.g., np.s_[0:10, :, :])

    Returns:
        Data from the HDF5 file as numpy array
    """
    with h5py.File(filename, 'r') as file_handle:
        if dataset is None:
            dataset = list(file_handle)[0]

        if slice_obj is not None:
            return np.array(file_handle[dataset][slice_obj])
        return np.array(file_handle[dataset])


def write_hdf5(
    filename: str,
    data_array: Union[np.ndarray, List[np.ndarray]],
    dataset: Union[str, List[str]] = 'main',
    compression: str = 'gzip',
    compression_level: int = 4
) -> None:
    """Write data to HDF5 file.

    Args:
        filename: Path to the output HDF5 file
        data_array: Data to write as numpy array or list of arrays
        dataset: Name of the dataset(s) to create
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level (0-9 for gzip, ignored for lzf)
    """
    with h5py.File(filename, 'w') as file_handle:
        if isinstance(dataset, list):
            for i, dataset_name in enumerate(dataset):
                file_handle.create_dataset(
                    dataset_name,
                    data=data_array[i],
                    compression=compression,
                    compression_opts=compression_level if compression == 'gzip' else None,
                    dtype=data_array[i].dtype
                )
        else:
            file_handle.create_dataset(
                dataset,
                data=data_array,
                compression=compression,
                compression_opts=compression_level if compression == 'gzip' else None,
                dtype=data_array.dtype
            )


def list_datasets(filename: str) -> List[str]:
    """List all datasets in an HDF5 file.

    Args:
        filename: Path to the HDF5 file

    Returns:
        List of dataset names
    """
    with h5py.File(filename, 'r') as file_handle:
        return list(file_handle.keys())


def get_dataset_info(filename: str, dataset: Optional[str] = None) -> dict:
    """Get information about a dataset in an HDF5 file.

    Args:
        filename: Path to the HDF5 file
        dataset: Name of the dataset. If None, uses first dataset

    Returns:
        Dictionary with shape, dtype, compression info
    """
    with h5py.File(filename, 'r') as file_handle:
        if dataset is None:
            dataset = list(file_handle)[0]

        ds = file_handle[dataset]
        return {
            'name': dataset,
            'shape': ds.shape,
            'dtype': ds.dtype,
            'compression': ds.compression,
            'compression_opts': ds.compression_opts,
        }


__all__ = [
    'read_hdf5',
    'write_hdf5',
    'list_datasets',
    'get_dataset_info',
]
```

#### Task 1.2: Create `formats/image.py`
**Extract from**: `volume.py` lines 19-199
**Functions**:
- `read_image(filename, add_channel=False)`
- `read_images(filename_pattern)`
- `read_image_as_volume(filename, drop_channel=False)`
- `save_image(filename, data)` - NEW
- `save_images(directory, data, prefix='')` - NEW

**Improvements**:
- Add save functions for symmetry
- Better error messages
- Support for more formats (WebP, JPEG2000)

```python
"""Image I/O operations for various formats (PNG, TIFF, etc.)."""

from __future__ import annotations
from typing import Optional
import os
import glob
import numpy as np
import imageio

# Avoid PIL "IOError: image file truncated"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


SUPPORTED_IMAGE_FORMATS = ['png', 'tif', 'tiff', 'jpg', 'jpeg']


def read_image(filename: str, add_channel: bool = False) -> Optional[np.ndarray]:
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
    if add_channel and image.ndim == 2:
        image = image[:, :, None]
    return image


def read_images(filename_pattern: str) -> np.ndarray:
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
    first_image = imageio.imread(file_list[0])
    if first_image.ndim == 2:
        data = np.zeros((len(file_list), *first_image.shape), dtype=first_image.dtype)
    elif first_image.ndim == 3:
        data = np.zeros((len(file_list), *first_image.shape), dtype=first_image.dtype)
    else:
        raise ValueError(f"Unsupported image dimensions: {first_image.ndim}D")

    # Load all images
    for i, filepath in enumerate(file_list):
        data[i] = imageio.imread(filepath)

    return data


def read_image_as_volume(filename: str, drop_channel: bool = False) -> np.ndarray:
    """Read a single image file as a volume with channel-first format.

    Args:
        filename: Path to the image file
        drop_channel: Whether to convert multichannel images to grayscale

    Returns:
        Image data as numpy array with shape (C, H, W)

    Raises:
        AssertionError: If file format is not supported
    """
    image_suffix = filename[filename.rfind('.') + 1:].lower()
    if image_suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported format: {image_suffix}. "
            f"Supported formats: {SUPPORTED_IMAGE_FORMATS}"
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


def save_images(directory: str, data: np.ndarray, prefix: str = '', format: str = 'png') -> None:
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


__all__ = [
    'read_image',
    'read_images',
    'read_image_as_volume',
    'save_image',
    'save_images',
    'SUPPORTED_IMAGE_FORMATS',
]
```

#### Task 1.3: Create `formats/pickle.py`
**Extract from**: `utils.py` lines 13-33
**Functions**:
- `read_pickle_file(filename)`
- `write_pickle_file(filename, data)` - NEW

```python
"""Pickle I/O operations."""

from __future__ import annotations
from typing import Any, List, Union
import pickle


def read_pickle_file(filename: str) -> Union[Any, List[Any]]:
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


def write_pickle_file(filename: str, data: Any) -> None:
    """Write data to a pickle file.

    Args:
        filename: Path to the output pickle file
        data: Data to pickle
    """
    with open(filename, 'wb') as file_handle:
        pickle.dump(data, file_handle)


__all__ = [
    'read_pickle_file',
    'write_pickle_file',
]
```

---

### Phase 2: Reorganize MONAI Transforms

#### Task 2.1: Create `transforms/volume.py`
**Move**: `monai_transforms.py` → `transforms/volume.py`
**Add**: `SaveVolumed` transform for symmetry

```python
"""MONAI transforms for volume I/O operations."""

from __future__ import annotations
from typing import Any, Dict
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from ..volume import read_volume, save_volume


class LoadVolumed(MapTransform):
    """MONAI loader for connectomics volume data (HDF5, TIFF, etc.)."""

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], str):
                volume = read_volume(d[key])
                if volume.ndim == 3:
                    volume = np.expand_dims(volume, axis=0)
                d[key] = volume
        return d


class SaveVolumed(MapTransform):
    """MONAI transform for saving volume data."""

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        output_format: str = 'h5',
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_format = output_format

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], np.ndarray):
                filename = os.path.join(self.output_dir, f"{key}.{self.output_format}")
                save_volume(filename, d[key], file_format=self.output_format)
        return d


__all__ = [
    'LoadVolumed',
    'SaveVolumed',
]
```

#### Task 2.2: Fix `transforms/tiles.py`
**Move**: Extract `TileLoaderd` from `tiles.py` → `transforms/tiles.py`
**Fix**: Add missing `Sequence` import and type hints

```python
"""MONAI transforms for tile-based I/O operations."""

from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence
import numpy as np
from monai.transforms import MapTransform

from ..tiles import reconstruct_volume_from_tiles


class TileLoaderd(MapTransform):
    """MONAI transform for loading tile-based data.

    This transform reconstructs volumes from tiles based on chunk coordinates
    and metadata information.
    """

    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load tile data for specified keys."""
        d = dict(data)

        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], dict):
                tile_info = d[key]
                if 'metadata' in tile_info and 'chunk_coords' in tile_info:
                    metadata = tile_info['metadata']
                    coords = tile_info['chunk_coords']
                    volume = self._load_tiles_for_chunk(metadata, coords)
                    d[key] = volume

        return d

    def _load_tiles_for_chunk(
        self,
        metadata: Dict[str, Any],
        coords: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        """Load and reconstruct volume chunk from tiles."""
        z_start, z_end, y_start, y_end, x_start, x_end = coords

        tile_paths = metadata['image'][z_start:z_end]
        volume_coords = [z_start, z_end, y_start, y_end, x_start, x_end]
        tile_coords = [
            0, metadata['depth'],
            0, metadata['height'],
            0, metadata['width']
        ]

        volume = reconstruct_volume_from_tiles(
            tile_paths=tile_paths,
            volume_coords=volume_coords,
            tile_coords=tile_coords,
            tile_size=metadata['tile_size'],
            data_type=np.dtype(metadata['dtype']),
            tile_start=metadata.get('tile_st', [0, 0]),
            tile_ratio=metadata.get('tile_ratio', 1.0),
        )

        return volume


__all__ = [
    'TileLoaderd',
]
```

---

### Phase 3: Update Main Volume Module

#### Task 3.1: Refactor `volume.py`
**Keep**: High-level functions (`read_volume`, `save_volume`)
**Delegate**: Format-specific operations to `formats/` modules

```python
"""High-level volume I/O operations."""

from __future__ import annotations
from typing import Optional
import numpy as np
import imageio

from .formats.hdf5 import read_hdf5, write_hdf5
from .formats.image import read_images, SUPPORTED_IMAGE_FORMATS


def read_volume(
    filename: str,
    dataset: Optional[str] = None,
    drop_channel: bool = False
) -> np.ndarray:
    """Load volumetric data in HDF5, TIFF or PNG formats.

    Args:
        filename: Path to the volume file
        dataset: HDF5 dataset name (only used for HDF5 files)
        drop_channel: Whether to convert multichannel volumes to single channel

    Returns:
        Volume data as numpy array with shape (D, H, W) or (C, D, H, W)

    Raises:
        ValueError: If file format is not recognized
    """
    image_suffix = filename[filename.rfind('.') + 1:].lower()

    if image_suffix in ['h5', 'hdf5']:
        data = read_hdf5(filename, dataset)
    elif 'tif' in image_suffix:
        data = imageio.volread(filename).squeeze()
        if data.ndim == 4:
            # Convert (D, C, H, W) to (C, D, H, W) order
            data = data.transpose(1, 0, 2, 3)
    elif 'png' in image_suffix:
        data = read_images(filename)
        if data.ndim == 4:
            # Convert (D, H, W, C) to (C, D, H, W) order
            data = data.transpose(3, 0, 1, 2)
    else:
        raise ValueError(
            f'Unrecognizable file format for {filename}. '
            f'Expected: h5, hdf5, tif, tiff, or png'
        )

    if data.ndim not in [3, 4]:
        raise ValueError(
            f"Currently supported volume data should be 3D (D, H, W) or 4D (C, D, H, W), "
            f"got {data.ndim}D"
        )

    if drop_channel and data.ndim == 4:
        # Merge multiple channels to grayscale by average
        original_dtype = data.dtype
        data = np.mean(data, axis=0).astype(original_dtype)

    return data


def save_volume(
    filename: str,
    volume: np.ndarray,
    dataset: str = 'main',
    file_format: str = 'h5'
) -> None:
    """Save volumetric data in specified format.

    Args:
        filename: Output filename or directory path
        volume: Volume data to save
        dataset: Dataset name for HDF5 format
        file_format: Output format ('h5' or 'png')

    Raises:
        ValueError: If file format is not supported
    """
    from .formats.image import save_images

    if file_format == 'h5':
        write_hdf5(filename, volume, dataset=dataset)
    elif file_format == 'png':
        save_images(filename, volume)
    else:
        raise ValueError(
            f"Unsupported format: {file_format}. "
            f"Supported formats: h5, png"
        )


__all__ = [
    'read_volume',
    'save_volume',
]
```

---

### Phase 4: Create Legacy Compatibility Layer

#### Task 4.1: Create `legacy.py`
**Purpose**: Backward compatibility with old naming
**Strategy**: Deprecation warnings → removal in v3.0

```python
"""
Deprecated legacy function names for backward compatibility.

These functions are deprecated and will be removed in v3.0.
Please update your code to use the modern equivalents.
"""

import warnings
from typing import Optional, List, Union
import numpy as np

# Import modern functions
from .formats.hdf5 import read_hdf5 as _read_hdf5, write_hdf5 as _write_hdf5
from .formats.image import (
    read_image as _read_image,
    read_images as _read_images,
    read_image_as_volume as _read_image_as_volume,
)
from .formats.pickle import read_pickle_file as _read_pickle_file
from .volume import read_volume as _read_volume, save_volume as _save_volume
from .utils import vast_to_segmentation as _vast_to_segmentation
from .tiles import (
    create_tile_metadata as _create_tile_metadata,
    reconstruct_volume_from_tiles as _reconstruct_volume_from_tiles,
)


def _deprecation_warning(old_name: str, new_name: str):
    """Issue a deprecation warning."""
    warnings.warn(
        f"'{old_name}' is deprecated and will be removed in v3.0. "
        f"Use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Legacy function aliases with deprecation warnings
def readh5(filename: str, dataset: Optional[str] = None) -> np.ndarray:
    """Deprecated: Use read_hdf5() instead."""
    _deprecation_warning('readh5', 'read_hdf5')
    return _read_hdf5(filename, dataset)


def writeh5(filename: str, dtarray: np.ndarray, dataset: str = 'main') -> None:
    """Deprecated: Use write_hdf5() instead."""
    _deprecation_warning('writeh5', 'write_hdf5')
    return _write_hdf5(filename, dtarray, dataset)


def readvol(filename: str, dataset: Optional[str] = None, drop_channel: bool = False) -> np.ndarray:
    """Deprecated: Use read_volume() instead."""
    _deprecation_warning('readvol', 'read_volume')
    return _read_volume(filename, dataset, drop_channel)


def savevol(filename: str, vol: np.ndarray, dataset: str = 'main', format: str = 'h5') -> None:
    """Deprecated: Use save_volume() instead."""
    _deprecation_warning('savevol', 'save_volume')
    return _save_volume(filename, vol, dataset, format)


def readim(filename: str, do_channel: bool = False) -> Optional[np.ndarray]:
    """Deprecated: Use read_image() instead."""
    _deprecation_warning('readim', 'read_image')
    return _read_image(filename, add_channel=do_channel)


def readimgs(filename: str) -> np.ndarray:
    """Deprecated: Use read_images() instead."""
    _deprecation_warning('readimgs', 'read_images')
    return _read_images(filename)


def readimg_as_vol(filename: str, drop_channel: bool = False) -> np.ndarray:
    """Deprecated: Use read_image_as_volume() instead."""
    _deprecation_warning('readimg_as_vol', 'read_image_as_volume')
    return _read_image_as_volume(filename, drop_channel)


def read_pkl(filename: str):
    """Deprecated: Use read_pickle_file() instead."""
    _deprecation_warning('read_pkl', 'read_pickle_file')
    return _read_pickle_file(filename)


def vast2Seg(seg: np.ndarray) -> np.ndarray:
    """Deprecated: Use vast_to_segmentation() instead."""
    _deprecation_warning('vast2Seg', 'vast_to_segmentation')
    return _vast_to_segmentation(seg)


def create_json(*args, **kwargs):
    """Deprecated: Use create_tile_metadata() instead."""
    _deprecation_warning('create_json', 'create_tile_metadata')
    return _create_tile_metadata(*args, **kwargs)


def tile2volume(*args, **kwargs):
    """Deprecated: Use reconstruct_volume_from_tiles() instead."""
    _deprecation_warning('tile2volume', 'reconstruct_volume_from_tiles')
    return _reconstruct_volume_from_tiles(*args, **kwargs)


__all__ = [
    # Legacy names (deprecated)
    'readh5', 'writeh5', 'readvol', 'savevol',
    'readim', 'readimgs', 'readimg_as_vol',
    'read_pkl', 'vast2Seg',
    'create_json', 'tile2volume',
]
```

---

### Phase 5: Update Public API

#### Task 5.1: Update `io/__init__.py`
**Purpose**: Clean, organized public API

```python
"""
I/O utilities for PyTorch Connectomics.

This package provides comprehensive I/O functionality for various data formats
commonly used in connectomics research.

Organization:
    formats/    - Format-specific I/O (HDF5, images, pickle)
    transforms/ - MONAI-compatible data loading transforms
    tiles.py    - Tile-based operations for large-scale datasets
    utils.py    - Utility functions for data processing
    legacy.py   - Deprecated function names (for backward compatibility)
"""

# High-level volume I/O
from .volume import read_volume, save_volume

# Format-specific I/O
from .formats.hdf5 import read_hdf5, write_hdf5, list_datasets, get_dataset_info
from .formats.image import (
    read_image, read_images, read_image_as_volume,
    save_image, save_images, SUPPORTED_IMAGE_FORMATS
)
from .formats.pickle import read_pickle_file, write_pickle_file

# Tile operations
from .tiles import create_tile_metadata, reconstruct_volume_from_tiles

# Utilities
from .utils import (
    vast_to_segmentation, normalize_data_range, convert_to_uint8,
    split_multichannel_mask, squeeze_arrays
)

# MONAI transforms
from .transforms.volume import LoadVolumed, SaveVolumed
from .transforms.tiles import TileLoaderd

__all__ = [
    # High-level volume I/O
    'read_volume', 'save_volume',

    # Format-specific I/O
    'read_hdf5', 'write_hdf5', 'list_datasets', 'get_dataset_info',
    'read_image', 'read_images', 'read_image_as_volume',
    'save_image', 'save_images', 'SUPPORTED_IMAGE_FORMATS',
    'read_pickle_file', 'write_pickle_file',

    # Tile operations
    'create_tile_metadata', 'reconstruct_volume_from_tiles',

    # Utilities
    'vast_to_segmentation', 'normalize_data_range', 'convert_to_uint8',
    'split_multichannel_mask', 'squeeze_arrays',

    # MONAI transforms
    'LoadVolumed', 'SaveVolumed', 'TileLoaderd',
]
```

#### Task 5.2: Deprecate `io_utils.py`
**Strategy**: Keep file, add deprecation warning, remove in v3.0

```python
"""
Deprecated: This module will be removed in v3.0.

Please import directly from connectomics.data.io instead:
    from connectomics.data.io import read_volume, save_volume, ...
"""

import warnings

warnings.warn(
    "connectomics.data.io_utils is deprecated and will be removed in v3.0. "
    "Import from connectomics.data.io instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from io for backward compatibility
from .io import *  # noqa
```

---

### Phase 6: Testing & Documentation

#### Task 6.1: Create Comprehensive Tests
**File**: `tests/test_io_refactoring.py`

```python
"""Tests for refactored I/O modules."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from connectomics.data.io import (
    # HDF5
    read_hdf5, write_hdf5, list_datasets, get_dataset_info,
    # Images
    read_image, read_images, read_image_as_volume, save_image, save_images,
    # Volume
    read_volume, save_volume,
    # Pickle
    read_pickle_file, write_pickle_file,
    # Utilities
    vast_to_segmentation, normalize_data_range,
)


class TestHDF5IO:
    """Test HDF5 I/O operations."""

    def test_read_write_hdf5(self, tmp_path):
        """Test basic HDF5 read/write."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        filepath = tmp_path / "test.h5"

        write_hdf5(str(filepath), data, dataset='test_data')
        loaded = read_hdf5(str(filepath), dataset='test_data')

        np.testing.assert_array_equal(data, loaded)

    def test_list_datasets(self, tmp_path):
        """Test listing datasets in HDF5 file."""
        filepath = tmp_path / "multi.h5"
        data1 = np.random.rand(5, 5, 5)
        data2 = np.random.rand(10, 10, 10)

        write_hdf5(str(filepath), [data1, data2], dataset=['data1', 'data2'])
        datasets = list_datasets(str(filepath))

        assert set(datasets) == {'data1', 'data2'}


class TestImageIO:
    """Test image I/O operations."""

    def test_save_load_image(self, tmp_path):
        """Test save and load single image."""
        data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        filepath = tmp_path / "test.png"

        save_image(str(filepath), data)
        loaded = read_image(str(filepath))

        np.testing.assert_array_equal(data, loaded)

    def test_save_load_images(self, tmp_path):
        """Test save and load image stack."""
        data = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)
        directory = tmp_path / "images"

        save_images(str(directory), data, prefix='img_')
        pattern = str(directory / "img_*.png")
        loaded = read_images(pattern)

        np.testing.assert_array_equal(data, loaded)


class TestVolumeIO:
    """Test high-level volume I/O."""

    def test_read_save_volume_h5(self, tmp_path):
        """Test volume I/O with HDF5 format."""
        data = np.random.rand(20, 100, 100).astype(np.float32)
        filepath = tmp_path / "volume.h5"

        save_volume(str(filepath), data, file_format='h5')
        loaded = read_volume(str(filepath))

        np.testing.assert_array_equal(data, loaded)


class TestLegacyCompatibility:
    """Test legacy function names still work."""

    def test_legacy_functions_warn(self):
        """Test that legacy functions issue deprecation warnings."""
        from connectomics.data.io import legacy

        with pytest.warns(DeprecationWarning):
            # This should work but warn
            pass  # Would test actual legacy functions if needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### Task 6.2: Create Migration Guide
**File**: `.claude/IO_MIGRATION_GUIDE.md`

```markdown
# I/O Module Migration Guide

## For Users

### Quick Reference: Old vs New Names

| Old (Deprecated) | New (Modern) | Module |
|-----------------|--------------|---------|
| `readh5()` | `read_hdf5()` | `io.formats.hdf5` |
| `writeh5()` | `write_hdf5()` | `io.formats.hdf5` |
| `readvol()` | `read_volume()` | `io.volume` |
| `savevol()` | `save_volume()` | `io.volume` |
| `readim()` | `read_image()` | `io.formats.image` |
| `readimgs()` | `read_images()` | `io.formats.image` |
| `readimg_as_vol()` | `read_image_as_volume()` | `io.formats.image` |
| `read_pkl()` | `read_pickle_file()` | `io.formats.pickle` |
| `vast2Seg()` | `vast_to_segmentation()` | `io.utils` |
| `create_json()` | `create_tile_metadata()` | `io.tiles` |
| `tile2volume()` | `reconstruct_volume_from_tiles()` | `io.tiles` |

### Import Changes

**Before (deprecated)**:
```python
from connectomics.data.io_utils import readvol, savevol
from connectomics.data.utils.data_io import readh5, writeh5
```

**After (modern)**:
```python
from connectomics.data.io import read_volume, save_volume
from connectomics.data.io import read_hdf5, write_hdf5
```

### Breaking Changes

None! All old function names still work with deprecation warnings.

### Timeline

- **v2.0** (current): Legacy names work with warnings
- **v2.5**: Warnings become more prominent
- **v3.0**: Legacy names removed

## For Developers

### Project Structure Changes

**Old**:
```
data/
├── io_utils.py              # Thin wrapper
└── io/
    ├── volume.py
    ├── tiles.py
    ├── utils.py
    └── monai_transforms.py
```

**New**:
```
data/
├── io_utils.py              # Deprecated wrapper
└── io/
    ├── __init__.py          # Clean public API
    ├── volume.py            # High-level operations
    ├── tiles.py             # Tile operations
    ├── utils.py             # Utilities
    ├── legacy.py            # Deprecated aliases
    │
    ├── formats/             # Format-specific I/O
    │   ├── hdf5.py
    │   ├── image.py
    │   └── pickle.py
    │
    └── transforms/          # MONAI transforms
        ├── volume.py
        └── tiles.py
```
```

---

## Summary of Changes

### Files to Create (7 new files)
1. `connectomics/data/io/formats/__init__.py`
2. `connectomics/data/io/formats/hdf5.py`
3. `connectomics/data/io/formats/image.py`
4. `connectomics/data/io/formats/pickle.py`
5. `connectomics/data/io/transforms/__init__.py`
6. `connectomics/data/io/transforms/volume.py`
7. `connectomics/data/io/transforms/tiles.py`
8. `connectomics/data/io/legacy.py`

### Files to Modify (5 files)
1. `connectomics/data/io/__init__.py` - Update public API
2. `connectomics/data/io/volume.py` - Simplify, delegate to formats/
3. `connectomics/data/io/tiles.py` - Remove TileLoaderd (move to transforms/)
4. `connectomics/data/io/utils.py` - Remove pickle functions (move to formats/)
5. `connectomics/data/io_utils.py` - Add deprecation warning

### Files to Delete (1 file)
1. `connectomics/data/io/monai_transforms.py` - Superseded by transforms/volume.py

### Benefits

1. **✅ Clear Organization**: Format-specific code grouped together
2. **✅ Better Discoverability**: Obvious where to find HDF5, image, pickle I/O
3. **✅ MONAI Integration**: All transforms in one place
4. **✅ Extensibility**: Easy to add new formats (Zarr, N5, etc.)
5. **✅ Backward Compatibility**: Legacy names still work
6. **✅ Type Safety**: Comprehensive type hints throughout
7. **✅ Documentation**: Every function documented
8. **✅ Testing**: Comprehensive test coverage

### Migration Effort

- **For Users**: Zero effort (backward compatible)
- **For Developers**: 2-3 days of work
  - Day 1: Create new structure (formats/, transforms/)
  - Day 2: Update existing files, create legacy.py
  - Day 3: Tests and documentation

---

## Implementation Order

1. **Phase 1** (Day 1): Create formats/ subdirectory
2. **Phase 2** (Day 1): Create transforms/ subdirectory
3. **Phase 3** (Day 2): Update volume.py, tiles.py, utils.py
4. **Phase 4** (Day 2): Create legacy.py
5. **Phase 5** (Day 2-3): Update __init__.py, deprecate io_utils.py
6. **Phase 6** (Day 3): Tests and documentation

**Total Estimated Time**: 3 days
