"""Shared helpers for constructing MONAI-style dataset dictionaries."""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "create_data_dicts_from_paths",
    "create_volume_data_dicts",
]


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
    """
    data_dicts = []

    for i, image_path in enumerate(image_paths):
        data_dict = {"image": image_path}

        if label_paths is not None:
            data_dict["label"] = label_paths[i]

        if mask_paths is not None:
            data_dict["mask"] = mask_paths[i]

        data_dicts.append(data_dict)

    return data_dicts


def create_volume_data_dicts(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create MONAI data dictionaries for volume datasets.

    This is a convenience wrapper around ``create_data_dicts_from_paths``
    for volume-specific use cases.
    """
    return create_data_dicts_from_paths(
        image_paths=image_paths,
        label_paths=label_paths,
        mask_paths=mask_paths,
    )
