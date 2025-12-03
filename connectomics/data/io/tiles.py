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
from .utils import rgb_to_seg


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
    tile_start: List[int] = [0, 0],
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
        data_path + str(i).zfill(digits) + r"/{row}_{column}.png" for i in range(depth)
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
    background_value: int = 128,
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
        max(-z0_output, z0_max),
        max(0, z1_output - z1_max),
        max(-y0_output, y0_max),
        max(0, y1_output - y1_max),
        max(-x0_output, x0_max),
        max(0, x1_output - x1_max),
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
                if r"{row}_{column}" in pattern:
                    path = pattern.format(row=row + tile_start[0], column=column + tile_start[1])
                else:
                    path = pattern

                patch = read_image(path, add_channel=True)
                if patch is not None:
                    if tile_ratio != 1:  # Apply scaling: image=1, label=0
                        patch = zoom(patch, [tile_ratio, tile_ratio, 1], order=int(is_image))

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
                        result[
                            z - z0,
                            y_actual_start - y0 : y_actual_end - y0,
                            x_actual_start - x0 : x_actual_end - x0,
                        ] = patch[
                            y_actual_start - y_patch_start : y_actual_end - y_patch_start,
                            x_actual_start - x_patch_start : x_actual_end - x_patch_start,
                            0,
                        ]
                    else:  # Label data
                        result[
                            z - z0,
                            y_actual_start - y0 : y_actual_end - y0,
                            x_actual_start - x0 : x_actual_end - x0,
                        ] = rgb_to_seg(
                            patch[
                                y_actual_start - y_patch_start : y_actual_end - y_patch_start,
                                x_actual_start - x_patch_start : x_actual_end - x_patch_start,
                            ]
                        )

    # Apply padding for chunks touching the border of the large input volume
    if max(boundary_diffs) > 0:
        result = np.pad(
            result,
            (
                (boundary_diffs[0], boundary_diffs[1]),
                (boundary_diffs[2], boundary_diffs[3]),
                (boundary_diffs[4], boundary_diffs[5]),
            ),
            "reflect",
        )

    return result


__all__ = [
    "create_tile_metadata",
    "reconstruct_volume_from_tiles",
]
