"""
Tile-based I/O operations for large-scale connectomics data.

This module provides functions for working with tiled datasets,
including volume reconstruction from tiles.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from scipy.ndimage import zoom

from .io import _read_image_with_channel
from .utils import rgb_to_seg


def reconstruct_volume_from_tiles(
    tile_paths: List[str],
    volume_coords: List[int],
    tile_coords: List[int],
    tile_size: Union[int, List[int]],
    data_type: type = np.uint8,
    tile_start: Optional[List[int]] = None,
    tile_ratio: float = 1.0,
    is_image: bool = True,
    background_value: int = 128,
) -> np.ndarray:
    """Construct a volume from image tiles.

    Args:
        tile_paths: Paths to image tiles.
        volume_coords: [z0, z1, y0, y1, x0, x1].
        tile_coords: Full dataset coords [z0,z1,y0,y1,x0,x1].
        tile_size: Tile height/width (int or [h, w]).
        data_type: Output dtype.
        tile_start: Start position [row, col]. Default [0,0].
        tile_ratio: Scale factor for tiles. Default 1.0.
        is_image: If True, use linear interp for resize.
        background_value: Fill value. Default 128.
    """
    if tile_start is None:
        tile_start = [0, 0]

    z0o, z1o, y0o, y1o, x0o, x1o = volume_coords
    z0m, z1m, y0m, y1m, x0m, x1m = tile_coords

    boundary_diffs = [
        max(-z0o, z0m),
        max(0, z1o - z1m),
        max(-y0o, y0m),
        max(0, y1o - y1m),
        max(-x0o, x0m),
        max(0, x1o - x1m),
    ]

    z0 = max(z0o, z0m)
    y0 = max(y0o, y0m)
    x0 = max(x0o, x0m)
    z1 = min(z1o, z1m)
    y1 = min(y1o, y1m)
    x1 = min(x1o, x1m)

    result = background_value * np.ones((z1 - z0, y1 - y0, x1 - x0), data_type)

    tile_h = tile_size[0] if isinstance(tile_size, list) else tile_size
    tile_w = tile_size[1] if isinstance(tile_size, list) else tile_size

    col_start = x0 // tile_w
    col_end = (x1 + tile_w - 1) // tile_w
    row_start = y0 // tile_h
    row_end = (y1 + tile_h - 1) // tile_h

    for z in range(z0, z1):
        pattern = tile_paths[z]
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                if r"{row}_{column}" in pattern:
                    path = pattern.format(
                        row=row + tile_start[0],
                        column=col + tile_start[1],
                    )
                else:
                    path = pattern

                patch = _read_image_with_channel(path)
                if patch is None:
                    continue

                if tile_ratio != 1:
                    patch = zoom(
                        patch,
                        [tile_ratio, tile_ratio, 1],
                        order=int(is_image),
                    )

                xps = col * tile_w
                xpe = xps + patch.shape[1]
                yps = row * tile_h
                ype = yps + patch.shape[0]

                xa = max(x0, xps)
                xe = min(x1, xpe)
                ya = max(y0, yps)
                ye = min(y1, ype)

                if is_image:
                    result[
                        z - z0,
                        ya - y0 : ye - y0,
                        xa - x0 : xe - x0,
                    ] = patch[
                        ya - yps : ye - yps,
                        xa - xps : xe - xps,
                        0,
                    ]
                else:
                    result[
                        z - z0,
                        ya - y0 : ye - y0,
                        xa - x0 : xe - x0,
                    ] = rgb_to_seg(
                        patch[
                            ya - yps : ye - yps,
                            xa - xps : xe - xps,
                        ]
                    )

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
