"""Shared helpers for tile dataset metadata/chunk preparation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from monai.transforms import Compose, EnsureChannelFirstd

from ..io import TileLoaderd


def load_tile_metadata(json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load tile metadata from JSON file."""
    if json_path is None:
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def calculate_chunk_indices(
    volume_metadata: Dict[str, Any],
    chunk_num: Sequence[int],
) -> List[Dict[str, Any]]:
    """Calculate chunk coordinates for a tiled volume."""
    depth = volume_metadata["depth"]
    height = volume_metadata["height"]
    width = volume_metadata["width"]

    chunk_z = depth // chunk_num[0]
    chunk_y = height // chunk_num[1]
    chunk_x = width // chunk_num[2]

    chunk_indices: List[Dict[str, Any]] = []
    for z in range(chunk_num[0]):
        for y in range(chunk_num[1]):
            for x in range(chunk_num[2]):
                z_start = z * chunk_z
                z_end = min((z + 1) * chunk_z, depth)
                y_start = y * chunk_y
                y_end = min((y + 1) * chunk_y, height)
                x_start = x * chunk_x
                x_end = min((x + 1) * chunk_x, width)

                chunk_indices.append(
                    {
                        "chunk_id": (z, y, x),
                        "coords": (z_start, z_end, y_start, y_end, x_start, x_end),
                    }
                )

    return chunk_indices


def create_chunk_data_dicts(
    chunk_indices: Sequence[Dict[str, Any]],
    *,
    volume_metadata: Dict[str, Any],
    label_metadata: Optional[Dict[str, Any]] = None,
    mask_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create MONAI-style data dictionaries for each chunk."""
    data_dicts: List[Dict[str, Any]] = []

    for chunk_info in chunk_indices:
        chunk_id = chunk_info["chunk_id"]
        coords = chunk_info["coords"]
        data_dict: Dict[str, Any] = {
            "image": {
                "metadata": volume_metadata,
                "chunk_coords": coords,
                "chunk_id": chunk_id,
            },
        }

        if label_metadata is not None:
            data_dict["label"] = {
                "metadata": label_metadata,
                "chunk_coords": coords,
                "chunk_id": chunk_id,
            }

        if mask_metadata is not None:
            data_dict["mask"] = {
                "metadata": mask_metadata,
                "chunk_coords": coords,
                "chunk_id": chunk_id,
            }

        data_dicts.append(data_dict)

    return data_dicts


def create_default_tile_transforms(
    *,
    label_metadata: Optional[Dict[str, Any]],
    mask_metadata: Optional[Dict[str, Any]],
) -> Compose:
    """Create default transforms for tile datasets."""
    keys = ["image"]
    if label_metadata is not None:
        keys.append("label")
    if mask_metadata is not None:
        keys.append("mask")

    return Compose(
        [
            TileLoaderd(keys=keys),
            EnsureChannelFirstd(keys=keys),
        ]
    )
