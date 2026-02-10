# I/O Module Refactoring Plan (Simplified)

## Executive Summary

This plan reorganizes and refactors the I/O modules in PyTorch Connectomics to eliminate redundancy, improve consistency, and establish a clear, modern API structure.

**Current State**: Redundant implementations across multiple locations
**Target State**: Clean, consolidated I/O package with consistent naming and modern Python practices

**Simplification Strategy**:
- Consolidate all format-specific I/O into single `io.py` file
- Consolidate all MONAI transforms into single `monai_transforms.py` file
- No backward compatibility layer (clean break from legacy)
- Minimal file changes for easier maintenance

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
**Action**: Delete this file

---

## Problems to Address

### 1. **Duplication**
- Legacy `data_io.py` has older implementations of same functions
- `io_utils.py` is just a re-export wrapper
- Functions spread across multiple files

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

### 3. **File Fragmentation**
- Format-specific code spread across `volume.py` and `utils.py`
- MONAI transforms in separate file but `TileLoaderd` in `tiles.py`

### 4. **MONAI Integration Gaps**
- `tiles.py` has incomplete `TileLoaderd` (missing `Sequence` import)
- Transforms not all in one place

---

## Proposed Structure (Simplified)

### Target Architecture

```
connectomics/data/io/
├── __init__.py              # Public API (all imports)
├── io.py                    # NEW: All format-specific I/O (HDF5, image, pickle)
├── monai_transforms.py      # UPDATED: All MONAI transforms (volume + tiles)
├── tiles.py                 # UPDATED: Tile operations only (no TileLoaderd)
└── utils.py                 # KEEP: Data utilities
```

**DELETE**:
- `connectomics/data/io_utils.py` (top-level wrapper)
- `connectomics/data/io/volume.py` (merged into `io.py`)

### Rationale for Simplified Design

1. **Single `io.py` file**: All format-specific I/O in one place
   - HDF5, image (PNG/TIFF), pickle operations
   - Easier to find and maintain
   - No subdirectory navigation needed
   - Still well-organized with clear section comments

2. **Single `monai_transforms.py` file**: All MONAI transforms together
   - `LoadVolumed`, `SaveVolumed`, `TileLoaderd` in one file
   - Clear separation: functions vs transforms
   - Easier to understand MONAI integration

3. **Clean `tiles.py`**: Pure tile operations
   - No MONAI dependencies
   - Just reconstruction functions
   - Simpler imports

4. **No legacy.py**: Clean break
   - Users must update to modern names
   - No maintenance burden of deprecated code
   - Clearer codebase

---

## Refactoring Tasks

### Phase 1: Create Consolidated I/O Module

#### Task 1.1: Create `io/io.py`
**Purpose**: Consolidate all format-specific I/O operations
**Sections**:
1. HDF5 operations
2. Image operations (PNG, TIFF)
3. Pickle operations
4. High-level volume I/O

**File**: `connectomics/data/io/io.py` (NEW)

```python
"""
Consolidated I/O operations for all formats.

This module provides I/O functions for:
- HDF5 files (.h5, .hdf5)
- Image files (PNG, TIFF)
- Pickle files (.pkl)
- High-level volume operations
"""

from __future__ import annotations
from typing import Optional, List, Union
import os
import glob
import pickle
import h5py
import numpy as np
import imageio

# Avoid PIL "IOError: image file truncated"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# HDF5 I/O
# =============================================================================

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
        compression_level: Compression level (0-9 for gzip)
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


def list_hdf5_datasets(filename: str) -> List[str]:
    """List all datasets in an HDF5 file.

    Args:
        filename: Path to the HDF5 file

    Returns:
        List of dataset names
    """
    with h5py.File(filename, 'r') as file_handle:
        return list(file_handle.keys())


# =============================================================================
# Image I/O
# =============================================================================

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
        ValueError: If file format is not supported
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
    with open(filename, 'wb') as file_handle:
        pickle.dump(data, file_handle)


# =============================================================================
# High-level Volume I/O
# =============================================================================

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
    # HDF5 I/O
    'read_hdf5', 'write_hdf5', 'list_hdf5_datasets',

    # Image I/O
    'read_image', 'read_images', 'read_image_as_volume',
    'save_image', 'save_images', 'SUPPORTED_IMAGE_FORMATS',

    # Pickle I/O
    'read_pickle_file', 'write_pickle_file',

    # High-level volume I/O
    'read_volume', 'save_volume',
]
```

---

### Phase 2: Consolidate MONAI Transforms

#### Task 2.1: Update `monai_transforms.py`
**Add**: `TileLoaderd` from `tiles.py` and `SaveVolumed` for completeness
**Fix**: Missing imports and type hints

**File**: `connectomics/data/io/monai_transforms.py` (UPDATED)

```python
"""
MONAI transforms for connectomics I/O operations.

This module provides MONAI-compatible transforms for:
- Volume loading (HDF5, TIFF, PNG)
- Volume saving
- Tile-based loading for large datasets
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from .io import read_volume, save_volume
from .tiles import reconstruct_volume_from_tiles


class LoadVolumed(MapTransform):
    """
    MONAI loader for connectomics volume data (HDF5, TIFF, etc.).

    This transform uses the connectomics read_volume function to load various
    file formats and ensures the data has a channel dimension.

    Args:
        keys: Keys to load from the data dictionary
        allow_missing_keys: Whether to allow missing keys in the dictionary

    Examples:
        >>> transform = LoadVolumed(keys=['image', 'label'])
        >>> data = {'image': 'img.h5', 'label': 'lbl.h5'}
        >>> result = transform(data)
        >>> # result['image'] shape: (C, D, H, W)
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load volume data from file paths."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], str):
                volume = read_volume(d[key])
                # Ensure we have at least 4 dimensions (add channel if needed)
                if volume.ndim == 3:
                    volume = np.expand_dims(volume, axis=0)
                d[key] = volume
        return d


class SaveVolumed(MapTransform):
    """
    MONAI transform for saving volume data.

    Args:
        keys: Keys to save from the data dictionary
        output_dir: Output directory for saved volumes
        output_format: File format ('h5' or 'png')
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = SaveVolumed(
        ...     keys=['prediction'],
        ...     output_dir='./outputs',
        ...     output_format='h5'
        ... )
        >>> data = {'prediction': np.random.rand(1, 32, 128, 128)}
        >>> result = transform(data)
        >>> # Saves to ./outputs/prediction.h5
    """

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
        """Save volume data to files."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], np.ndarray):
                filename = os.path.join(self.output_dir, f"{key}.{self.output_format}")
                save_volume(filename, d[key], file_format=self.output_format)
        return d


class TileLoaderd(MapTransform):
    """
    MONAI transform for loading tile-based data.

    This transform reconstructs volumes from tiles based on chunk coordinates
    and metadata information.

    Args:
        keys: Keys to process from the data dictionary
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = TileLoaderd(keys=['image'])
        >>> data = {
        ...     'image': {
        ...         'metadata': tile_metadata,
        ...         'chunk_coords': (0, 10, 0, 128, 0, 128)
        ...     }
        ... }
        >>> result = transform(data)
        >>> # result['image'] is reconstructed volume from tiles
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
    'LoadVolumed',
    'SaveVolumed',
    'TileLoaderd',
]
```

---

### Phase 3: Update Existing Files

#### Task 3.1: Clean up `tiles.py`
**Remove**: `TileLoaderd` class (moved to `monai_transforms.py`)
**Keep**: Tile reconstruction functions only

**File**: `connectomics/data/io/tiles.py` (UPDATED)

```python
"""
Tile-based I/O operations for large-scale connectomics data.

This module provides functions for working with tiled datasets, including
volume reconstruction from tiles and metadata creation.
"""

from __future__ import annotations
from typing import List, Union
import math
import numpy as np
from scipy.ndimage import zoom

from .io import read_image
from .utils import vast_to_segmentation


def create_tile_metadata(
    num_dimensions: int = 1,
    data_type: str = "uint8",
    data_path: str = "/path/to/data/",
    height: int = 10000,
    width: int = 10000,
    depth: int = 500,
    num_columns: int = 3,
    num_rows: int = 3,
    tile_size: int = 4096,
    tile_ratio: int = 1,
    tile_start: List[int] = [0, 0]
) -> dict:
    """Create metadata dictionary for large-scale tiled volumes.

    The dictionary is usually saved as a JSON file and can be read by the TileDataset.

    Args:
        num_dimensions: Number of dimensions in the data. Default: 1
        data_type: Data type string (e.g., "uint8", "float32"). Default: "uint8"
        data_path: Path to the data directory. Default: "/path/to/data/"
        height: Height of the volume in pixels. Default: 10000
        width: Width of the volume in pixels. Default: 10000
        depth: Depth of the volume in pixels. Default: 500
        num_columns: Number of tile columns. Default: 3
        num_rows: Number of tile rows. Default: 3
        tile_size: Size of each tile in pixels. Default: 4096
        tile_ratio: Ratio for tile scaling. Default: 1
        tile_start: Starting position for tiles [row, column]. Default: [0, 0]

    Returns:
        Dictionary containing metadata for the tiled volume
    """
    metadata = {}
    metadata["ndim"] = num_dimensions
    metadata["dtype"] = data_type

    digits = int(math.log10(depth)) + 1
    metadata["image"] = [
        data_path + str(i).zfill(digits) + r"/{row}_{column}.png"
        for i in range(depth)
    ]

    metadata["height"] = height
    metadata["width"] = width
    metadata["depth"] = depth

    metadata["n_columns"] = num_columns
    metadata["n_rows"] = num_rows

    metadata["tile_size"] = tile_size
    metadata["tile_ratio"] = tile_ratio
    metadata["tile_st"] = tile_start

    return metadata


def reconstruct_volume_from_tiles(
    tile_paths: List[str],
    volume_coords: List[int],
    tile_coords: List[int],
    tile_size: Union[int, List[int]],
    data_type: type = np.uint8,
    tile_start: List[int] = [0, 0],
    tile_ratio: float = 1.0,
    is_image: bool = True,
    background_value: int = 128
) -> np.ndarray:
    """Construct a volume from image tiles based on the given volume coordinate.

    Args:
        tile_paths: List of paths to the image tiles
        volume_coords: Coordinate of the volume to be constructed [z0, z1, y0, y1, x0, x1]
        tile_coords: Coordinate of the whole dataset with the tiles [z0, z1, y0, y1, x0, x1]
        tile_size: Height and width of the tiles (int or [height, width])
        data_type: Data type of the constructed volume. Default: np.uint8
        tile_start: Start position of the tiles [row, column]. Default: [0, 0]
        tile_ratio: Scale factor for resizing the tiles. Default: 1.0
        is_image: Whether to construct an image volume (apply linear interpolation for resizing). Default: True
        background_value: Background value for filling the constructed volume. Default: 128

    Returns:
        Reconstructed 3D volume as numpy array
    """
    z0_output, z1_output, y0_output, y1_output, x0_output, x1_output = volume_coords
    z0_max, z1_max, y0_max, y1_max, x0_max, x1_max = tile_coords

    # Calculate boundary conditions
    boundary_diffs = [
        max(-z0_output, z0_max), max(0, z1_output - z1_max),
        max(-y0_output, y0_max), max(0, y1_output - y1_max),
        max(-x0_output, x0_max), max(0, x1_output - x1_max)
    ]

    z0 = max(z0_output, z0_max)
    y0 = max(y0_output, y0_max)
    x0 = max(x0_output, x0_max)
    z1 = min(z1_output, z1_max)
    y1 = min(y1_output, y1_max)
    x1 = min(x1_output, x1_max)

    result = background_value * np.ones((z1 - z0, y1 - y0, x1 - x0), data_type)

    # Handle different tile size formats
    tile_height = tile_size[0] if isinstance(tile_size, list) else tile_size
    tile_width = tile_size[1] if isinstance(tile_size, list) else tile_size

    # Calculate tile grid bounds
    column_start = x0 // tile_width  # floor
    column_end = (x1 + tile_width - 1) // tile_width  # ceil
    row_start = y0 // tile_height
    row_end = (y1 + tile_height - 1) // tile_height

    for z in range(z0, z1):
        pattern = tile_paths[z]
        for row in range(row_start, row_end):
            for column in range(column_start, column_end):
                if r'{row}_{column}' in pattern:
                    path = pattern.format(
                        row=row + tile_start[0],
                        column=column + tile_start[1]
                    )
                else:
                    path = pattern

                patch = read_image(path, add_channel=True)
                if patch is not None:
                    if tile_ratio != 1:  # Apply scaling: image=1, label=0
                        patch = zoom(
                            patch, [tile_ratio, tile_ratio, 1],
                            order=int(is_image)
                        )

                    # Handle potentially different tile sizes
                    x_patch_start = column * tile_width
                    x_patch_end = x_patch_start + patch.shape[1]
                    y_patch_start = row * tile_height
                    y_patch_end = y_patch_start + patch.shape[0]

                    # Calculate intersection with target region
                    x_actual_start = max(x0, x_patch_start)
                    x_actual_end = min(x1, x_patch_end)
                    y_actual_start = max(y0, y_patch_start)
                    y_actual_end = min(y1, y_patch_end)

                    if is_image:  # Image data
                        result[z - z0,
                               y_actual_start - y0:y_actual_end - y0,
                               x_actual_start - x0:x_actual_end - x0] = \
                            patch[y_actual_start - y_patch_start:y_actual_end - y_patch_start,
                                  x_actual_start - x_patch_start:x_actual_end - x_patch_start, 0]
                    else:  # Label data
                        result[z - z0,
                               y_actual_start - y0:y_actual_end - y0,
                               x_actual_start - x0:x_actual_end - x0] = \
                            vast_to_segmentation(patch[y_actual_start - y_patch_start:y_actual_end - y_patch_start,
                                                        x_actual_start - x_patch_start:x_actual_end - x_patch_start])

    # Apply padding for chunks touching the border of the large input volume
    if max(boundary_diffs) > 0:
        result = np.pad(
            result,
            ((boundary_diffs[0], boundary_diffs[1]),
             (boundary_diffs[2], boundary_diffs[3]),
             (boundary_diffs[4], boundary_diffs[5])),
            'reflect'
        )

    return result


__all__ = [
    'create_tile_metadata',
    'reconstruct_volume_from_tiles',
]
```

#### Task 3.2: Update `utils.py`
**Remove**: `read_pickle_file` (moved to `io.py`)
**Keep**: All other utility functions

**File**: `connectomics/data/io/utils.py` (UPDATED)

```python
"""
Utility functions for data I/O operations.

This module provides various utility functions for data processing,
conversion, and manipulation in connectomics workflows.
"""

from __future__ import annotations
import numpy as np


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


def normalize_data_range(
    data: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
    ignore_uint8: bool = True
) -> np.ndarray:
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
    'vast_to_segmentation',
    'normalize_data_range',
    'convert_to_uint8',
    'split_multichannel_mask',
    'squeeze_arrays',
]
```

---

### Phase 4: Update Public API

#### Task 4.1: Update `io/__init__.py`
**Purpose**: Clean, organized public API with all imports

**File**: `connectomics/data/io/__init__.py` (UPDATED)

```python
"""
I/O utilities for PyTorch Connectomics.

This package provides comprehensive I/O functionality for various data formats
commonly used in connectomics research.

Organization:
    io.py              - All format-specific I/O (HDF5, images, pickle, volume)
    monai_transforms.py - MONAI-compatible data loading transforms
    tiles.py           - Tile-based operations for large-scale datasets
    utils.py           - Utility functions for data processing
"""

# Core I/O functions
from .io import (
    # HDF5 I/O
    read_hdf5, write_hdf5, list_hdf5_datasets,

    # Image I/O
    read_image, read_images, read_image_as_volume,
    save_image, save_images, SUPPORTED_IMAGE_FORMATS,

    # Pickle I/O
    read_pickle_file, write_pickle_file,

    # High-level volume I/O
    read_volume, save_volume,
)

# Tile operations
from .tiles import (
    create_tile_metadata,
    reconstruct_volume_from_tiles,
)

# Utilities
from .utils import (
    vast_to_segmentation,
    normalize_data_range,
    convert_to_uint8,
    split_multichannel_mask,
    squeeze_arrays,
)

# MONAI transforms
from .monai_transforms import (
    LoadVolumed,
    SaveVolumed,
    TileLoaderd,
)

__all__ = [
    # HDF5 I/O
    'read_hdf5', 'write_hdf5', 'list_hdf5_datasets',

    # Image I/O
    'read_image', 'read_images', 'read_image_as_volume',
    'save_image', 'save_images', 'SUPPORTED_IMAGE_FORMATS',

    # Pickle I/O
    'read_pickle_file', 'write_pickle_file',

    # High-level volume I/O
    'read_volume', 'save_volume',

    # Tile operations
    'create_tile_metadata', 'reconstruct_volume_from_tiles',

    # Utilities
    'vast_to_segmentation', 'normalize_data_range', 'convert_to_uint8',
    'split_multichannel_mask', 'squeeze_arrays',

    # MONAI transforms
    'LoadVolumed', 'SaveVolumed', 'TileLoaderd',
]
```

---

### Phase 5: Delete Redundant Files

#### Task 5.1: Delete files
**Files to delete**:
1. `connectomics/data/io_utils.py` - Top-level wrapper (no longer needed)
2. `connectomics/data/io/volume.py` - Merged into `io.py`

```bash
# Delete redundant files
rm connectomics/data/io_utils.py
rm connectomics/data/io/volume.py
```

---

## Summary of Changes

### Files to Create (1 new file)
1. `connectomics/data/io/io.py` - Consolidated format-specific I/O

### Files to Modify (4 files)
1. `connectomics/data/io/__init__.py` - Update imports
2. `connectomics/data/io/monai_transforms.py` - Add TileLoaderd, SaveVolumed
3. `connectomics/data/io/tiles.py` - Remove TileLoaderd class
4. `connectomics/data/io/utils.py` - Remove read_pickle_file

### Files to Delete (2 files)
1. `connectomics/data/io_utils.py` - Redundant wrapper
2. `connectomics/data/io/volume.py` - Merged into io.py

### Final Structure

```
connectomics/data/io/
├── __init__.py              # Public API
├── io.py                    # NEW: All format I/O (HDF5, image, pickle, volume)
├── monai_transforms.py      # UPDATED: All MONAI transforms
├── tiles.py                 # UPDATED: Tile operations only
└── utils.py                 # UPDATED: Data utilities (no pickle)
```

### Benefits

1. **✅ Simpler Structure**: 5 files instead of subdirectories
2. **✅ Clear Organization**: All format I/O in one place, all transforms in one place
3. **✅ Easier Maintenance**: Fewer files to manage
4. **✅ No Legacy Baggage**: Clean break from old naming
5. **✅ Complete MONAI Integration**: All transforms together
6. **✅ Type Safety**: Comprehensive type hints throughout
7. **✅ Well Documented**: Every function documented

### Migration for Users

**Breaking Changes**:
- Must use modern function names (no `readh5`, `readvol`, etc.)
- Must import from `connectomics.data.io` (not `io_utils`)

**Migration**:
```python
# Old (will break)
from connectomics.data.io_utils import readvol, savevol
from connectomics.data.utils.data_io import readh5

# New (correct)
from connectomics.data.io import read_volume, save_volume, read_hdf5
```

### Implementation Effort

- **Time**: 1-2 days
- **Complexity**: Low (mostly file consolidation)
- **Testing**: Update imports in existing code

---

## Implementation Order

1. **Phase 1** (4 hours): Create `io.py` with all format I/O
2. **Phase 2** (2 hours): Update `monai_transforms.py`
3. **Phase 3** (2 hours): Update `tiles.py` and `utils.py`
4. **Phase 4** (1 hour): Update `__init__.py`
5. **Phase 5** (1 hour): Delete old files, update imports in codebase

**Total Estimated Time**: 1-2 days
