"""
Dataset builder for PyTorch Connectomics.

Factory functions to create various types of MONAI-based datasets for connectomics:
- Base datasets (standard, cached, persistent)
- Volume datasets (for 3D volumetric data)
- Tile datasets (for large-scale tiled volumes)

All factory functions follow the consistent `create_*` naming pattern.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from monai.transforms import Compose

from .dataset_base import (
    MonaiConnectomicsDataset,
    MonaiCachedConnectomicsDataset,
    MonaiPersistentConnectomicsDataset,
)
from .dataset_volume import (
    MonaiVolumeDataset,
    MonaiCachedVolumeDataset,
)
from .dataset_tile import (
    MonaiTileDataset,
    MonaiCachedTileDataset,
)


__all__ = [
    # Data dict creation
    'create_data_dicts_from_paths',
    'create_volume_data_dicts',
    'create_tile_data_dicts_from_json',
    
    # Dataset creation
    'create_connectomics_dataset',
    'create_volume_dataset',
    'create_tile_dataset',
]


# ============================================================================
# Data Dictionary Creation
# ============================================================================

def create_data_dicts_from_paths(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create MONAI-style data dictionaries from file paths.

    Args:
        image_paths: List of image file paths
        label_paths: Optional list of label file paths
        mask_paths: Optional list of mask file paths

    Returns:
        List of dictionaries with 'image', 'label', and/or 'mask' keys
        
    Examples:
        >>> image_paths = ['img1.h5', 'img2.h5']
        >>> label_paths = ['lbl1.h5', 'lbl2.h5']
        >>> data_dicts = create_data_dicts_from_paths(image_paths, label_paths)
        >>> # [{'image': 'img1.h5', 'label': 'lbl1.h5'}, ...]
    """
    data_dicts = []

    for i, image_path in enumerate(image_paths):
        data_dict = {'image': image_path}

        if label_paths is not None:
            data_dict['label'] = label_paths[i]

        if mask_paths is not None:
            data_dict['mask'] = mask_paths[i]

        data_dicts.append(data_dict)

    return data_dicts


def create_volume_data_dicts(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create MONAI data dictionaries for volume datasets.

    This is a convenience wrapper around create_data_dicts_from_paths
    for volume-specific use cases.

    Args:
        image_paths: List of image volume file paths
        label_paths: Optional list of label volume file paths
        mask_paths: Optional list of valid mask file paths

    Returns:
        List of MONAI-style data dictionaries
        
    Examples:
        >>> data_dicts = create_volume_data_dicts(['vol1.tif'], ['lbl1.tif'])
    """
    return create_data_dicts_from_paths(
        image_paths=image_paths,
        label_paths=label_paths,
        mask_paths=mask_paths,
    )


def create_tile_data_dicts_from_json(
    volume_json: str,
    label_json: Optional[str] = None,
    mask_json: Optional[str] = None,
    chunk_num: Tuple[int, int, int] = (2, 2, 2),
) -> List[Dict[str, Any]]:
    """
    Create MONAI data dictionaries from tile JSON metadata files.

    Args:
        volume_json: JSON metadata file for input image tiles
        label_json: Optional JSON metadata file for label tiles
        mask_json: Optional JSON metadata file for mask tiles
        chunk_num: Volume splitting parameters (z, y, x)

    Returns:
        List of MONAI-style data dictionaries for tile chunks
        
    Examples:
        >>> data_dicts = create_tile_data_dicts_from_json('tiles.json')
    """
    # This would use the same logic as in MonaiTileDataset._create_chunk_data_dicts
    # but as a standalone function
    # TODO: Implement if needed
    raise NotImplementedError(
        "create_tile_data_dicts_from_json is not yet implemented. "
        "Use create_tile_dataset() directly instead."
    )


# ============================================================================
# Dataset Creation - Base
# ============================================================================

def create_connectomics_dataset(
    data_dicts: Sequence[Dict[str, Any]],
    transforms: Optional[Compose] = None,
    dataset_type: str = 'standard',
    **kwargs,
) -> Union[MonaiConnectomicsDataset, MonaiCachedConnectomicsDataset, MonaiPersistentConnectomicsDataset]:
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
    if dataset_type == 'cached':
        return MonaiCachedConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )
    elif dataset_type == 'persistent':
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
    dataset_type: str = 'standard',
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
    if dataset_type == 'cached':
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
    dataset_type: str = 'standard',
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
    if dataset_type == 'cached':
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
