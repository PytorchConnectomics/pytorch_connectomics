"""
Dataset builder for PyTorch Connectomics.

Factory functions to create various types of MONAI-based datasets for connectomics:
- Base datasets (standard, cached, persistent)
- Volume datasets (for 3D volumetric data)
- Tile datasets (for large-scale tiled volumes)

All factory functions follow the consistent `create_*` naming pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from monai.transforms import Compose

from .data_dicts import (
    create_data_dicts_from_paths,
)
from .dataset_base import (
    MonaiCachedConnectomicsDataset,
    MonaiConnectomicsDataset,
    MonaiPersistentConnectomicsDataset,
)
from .dataset_tile import (
    MonaiCachedTileDataset,
    MonaiTileDataset,
)
from .tile_utils import calculate_chunk_indices, create_chunk_data_dicts

if TYPE_CHECKING:
    from .dataset_volume import MonaiCachedVolumeDataset, MonaiVolumeDataset


__all__ = [
    # Data dict creation
    "create_data_dicts_from_paths",
    "create_tile_data_dicts_from_json",
    # Dataset creation
    "create_connectomics_dataset",
    "create_volume_dataset",
    "create_tile_dataset",
]


def create_tile_data_dicts_from_json(
    volume_json: str,
    label_json: Optional[str] = None,
    mask_json: Optional[str] = None,
    chunk_num: Tuple[int, int, int] = (2, 2, 2),
    chunk_indices: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Create MONAI data dictionaries from tile JSON metadata files.

    This function loads tile metadata from JSON files and creates data dictionaries
    for each chunk of the volume. It's useful for preparing data before creating
    a dataset, or for custom dataset implementations.

    JSON Schema:
        The JSON file should contain volume metadata in the following format:
        {
            "depth": int,       # Volume depth in pixels/voxels
            "height": int,      # Volume height in pixels/voxels
            "width": int,       # Volume width in pixels/voxels
            "tiles": [          # List of tile files (optional)
                {
                    "file": str,           # Path to tile file
                    "z_start": int,        # Starting z coordinate
                    "z_end": int,          # Ending z coordinate
                    "y_start": int,        # Starting y coordinate
                    "y_end": int,          # Ending y coordinate
                    "x_start": int,        # Starting x coordinate
                    "x_end": int           # Ending x coordinate
                },
                ...
            ],
            "tile_size": [int, int, int],    # Optional: default tile size (z, y, x)
            "overlap": [int, int, int],      # Optional: tile overlap (z, y, x)
            "format": str,                   # Optional: file format (e.g., "tif", "h5")
            "metadata": {...}                # Optional: additional metadata
        }

    Args:
        volume_json: Path to JSON metadata file for input image tiles
        label_json: Optional path to JSON metadata file for label tiles
        mask_json: Optional path to JSON metadata file for mask tiles
        chunk_num: Volume splitting parameters (z, y, x). Default: (2, 2, 2)
        chunk_indices: Optional predefined list of chunk information dicts.
                      Each dict should have 'chunk_id' and 'coords' keys.

    Returns:
        List of MONAI-style data dictionaries for tile chunks.
        Each dictionary contains nested dicts for 'image', 'label' (if provided),
        and 'mask' (if provided) with metadata and chunk coordinates.

    Examples:
        >>> # Create data dicts from JSON with automatic chunking
        >>> data_dicts = create_tile_data_dicts_from_json(
        ...     volume_json='tiles/image.json',
        ...     label_json='tiles/label.json',
        ...     chunk_num=(2, 2, 2)
        ... )
        >>> len(data_dicts)  # 2*2*2 = 8 chunks
        8

        >>> # Create with custom chunk indices
        >>> custom_chunks = [
        ...     {'chunk_id': (0, 0, 0), 'coords': (0, 100, 0, 200, 0, 200)},
        ...     {'chunk_id': (0, 0, 1), 'coords': (0, 100, 0, 200, 200, 400)},
        ... ]
        >>> data_dicts = create_tile_data_dicts_from_json(
        ...     'tiles/image.json',
        ...     chunk_indices=custom_chunks
        ... )

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is malformed or missing required fields
        KeyError: If required keys are missing from JSON
    """
    import json
    from pathlib import Path

    # Load volume metadata
    volume_path = Path(volume_json)
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume JSON file not found: {volume_json}")

    with open(volume_path, "r") as f:
        volume_metadata = json.load(f)

    # Validate required fields
    required_fields = ["depth", "height", "width"]
    missing_fields = [field for field in required_fields if field not in volume_metadata]
    if missing_fields:
        raise KeyError(
            f"Volume JSON missing required fields: {missing_fields}. "
            f"Required fields: {required_fields}"
        )

    # Load label metadata if provided
    label_metadata = None
    if label_json is not None:
        label_path = Path(label_json)
        if not label_path.exists():
            raise FileNotFoundError(f"Label JSON file not found: {label_json}")
        with open(label_path, "r") as f:
            label_metadata = json.load(f)

    # Load mask metadata if provided
    mask_metadata = None
    if mask_json is not None:
        mask_path = Path(mask_json)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask JSON file not found: {mask_json}")
        with open(mask_path, "r") as f:
            mask_metadata = json.load(f)

    # Calculate chunk indices if not provided
    if chunk_indices is None:
        chunk_indices = calculate_chunk_indices(volume_metadata, chunk_num)

    return create_chunk_data_dicts(
        chunk_indices,
        volume_metadata=volume_metadata,
        label_metadata=label_metadata,
        mask_metadata=mask_metadata,
    )


# ============================================================================
# Dataset Creation - Base
# ============================================================================


def create_connectomics_dataset(
    data_dicts: Sequence[Dict[str, Any]],
    transforms: Optional[Compose] = None,
    dataset_type: str = "standard",
    **kwargs,
) -> Union[
    MonaiConnectomicsDataset, MonaiCachedConnectomicsDataset, MonaiPersistentConnectomicsDataset
]:
    """
    Factory function to create appropriate MONAI connectomics dataset.

    Args:
        data_dicts: List of data dictionaries
        transforms: MONAI transforms pipeline
        dataset_type: Type of dataset ('standard', 'cached', 'persistent')
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Appropriate MONAI connectomics dataset instance

    Examples:
        >>> data_dicts = create_data_dicts_from_paths(['img.h5'], ['lbl.h5'])
        >>> dataset = create_connectomics_dataset(
        ...     data_dicts, transforms=my_transforms, dataset_type='cached'
        ... )
    """
    if dataset_type == "cached":
        return MonaiCachedConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )
    elif dataset_type == "persistent":
        return MonaiPersistentConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )
    else:
        return MonaiConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )


# ============================================================================
# Dataset Creation - Volume
# ============================================================================


def create_volume_dataset(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
    transforms: Optional[Compose] = None,
    dataset_type: str = "standard",
    cache_rate: float = 1.0,
    **kwargs,
) -> Union[MonaiVolumeDataset, MonaiCachedVolumeDataset]:
    """
    Factory function to create MONAI volume datasets.

    Args:
        image_paths: List of image volume file paths
        label_paths: Optional list of label volume file paths
        mask_paths: Optional list of valid mask file paths
        transforms: MONAI transforms pipeline
        dataset_type: Type of dataset ('standard', 'cached')
        cache_rate: Cache rate for cached datasets
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Appropriate MONAI volume dataset instance

    Examples:
        >>> dataset = create_volume_dataset(
        ...     image_paths=['train_img.tif'],
        ...     label_paths=['train_lbl.tif'],
        ...     transforms=my_transforms,
        ...     dataset_type='cached',
        ...     cache_rate=1.0,
        ... )
    """
    # Lazy import to avoid circular dependency during module import
    from .dataset_volume import MonaiCachedVolumeDataset, MonaiVolumeDataset

    if dataset_type == "cached":
        return MonaiCachedVolumeDataset(
            image_paths=image_paths,
            label_paths=label_paths,
            mask_paths=mask_paths,
            transforms=transforms,
            cache_rate=cache_rate,
            **kwargs,
        )
    else:
        return MonaiVolumeDataset(
            image_paths=image_paths,
            label_paths=label_paths,
            mask_paths=mask_paths,
            transforms=transforms,
            **kwargs,
        )


# ============================================================================
# Dataset Creation - Tile
# ============================================================================


def create_tile_dataset(
    volume_json: str,
    label_json: Optional[str] = None,
    mask_json: Optional[str] = None,
    transforms: Optional[Compose] = None,
    dataset_type: str = "standard",
    cache_rate: float = 1.0,
    **kwargs,
) -> Union[MonaiTileDataset, MonaiCachedTileDataset]:
    """
    Factory function to create MONAI tile datasets.

    Args:
        volume_json: JSON metadata file for input image tiles
        label_json: Optional JSON metadata file for label tiles
        mask_json: Optional JSON metadata file for mask tiles
        transforms: MONAI transforms pipeline
        dataset_type: Type of dataset ('standard', 'cached')
        cache_rate: Cache rate for cached datasets
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Appropriate MONAI tile dataset instance

    Examples:
        >>> dataset = create_tile_dataset(
        ...     volume_json='tiles.json',
        ...     label_json='labels.json',
        ...     transforms=my_transforms,
        ...     dataset_type='cached',
        ... )
    """
    if dataset_type == "cached":
        return MonaiCachedTileDataset(
            volume_json=volume_json,
            label_json=label_json,
            mask_json=mask_json,
            transforms=transforms,
            cache_rate=cache_rate,
            **kwargs,
        )
    else:
        return MonaiTileDataset(
            volume_json=volume_json,
            label_json=label_json,
            mask_json=mask_json,
            transforms=transforms,
            **kwargs,
        )
