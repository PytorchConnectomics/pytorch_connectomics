"""Shared helpers for constructing MONAI-style dataset dictionaries."""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "create_data_dicts_from_paths",
]


def create_data_dicts_from_paths(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
    extra_paths: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, object]]:
    """
    Create MONAI-style data dictionaries from file paths.

    Args:
        image_paths: List of image file paths
        label_paths: Optional list of label file paths
        mask_paths: Optional list of mask file paths
        extra_paths: Optional dict of additional keys to include, e.g.
            ``{"sdt": ["/path/to/sdt1.h5", "/path/to/sdt2.h5"]}``

    Returns:
        List of dictionaries with 'image', 'label', and/or 'mask' keys
    """
    data_dicts: List[Dict[str, object]] = []

    for i, image_path in enumerate(image_paths):
        data_dict: Dict[str, object] = {"image": image_path}

        if label_paths is not None:
            data_dict["label"] = label_paths[i]

        if mask_paths is not None:
            data_dict["mask"] = mask_paths[i]

        if extra_paths is not None:
            for key, paths in extra_paths.items():
                data_dict[key] = paths[i]

        data_dicts.append(data_dict)

    return data_dicts
