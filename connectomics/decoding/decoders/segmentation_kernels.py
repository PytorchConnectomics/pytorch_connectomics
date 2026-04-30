"""Algorithm kernels used by segmentation decoders."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


try:
    import edt

    EDT_AVAILABLE = True
except ImportError:
    EDT_AVAILABLE = False


def compute_edt(
    foreground_mask: np.ndarray,
    use_fast_edt: bool = True,
    edt_parallel: int = 4,
    edt_anisotropy: Optional[Tuple[float, ...]] = None,
    edt_downsample_factor: int = 1,
) -> np.ndarray:
    """Compute Euclidean Distance Transform on a foreground mask."""
    resolved_anisotropy = edt_anisotropy

    def _edt_on_mask(mask: np.ndarray, anisotropy: Optional[Tuple[float, ...]]) -> np.ndarray:
        if use_fast_edt and EDT_AVAILABLE:
            ani = anisotropy if anisotropy is not None else tuple(1.0 for _ in range(mask.ndim))
            return edt.edt(
                mask.astype(np.uint8),
                anisotropy=ani,
                black_border=True,
                parallel=edt_parallel,
            )

        if use_fast_edt and not EDT_AVAILABLE:
            warnings.warn(
                "Fast edt library not available. Using scipy.ndimage (slower). "
                "Install edt for 10-50x speedup: pip install edt",
                UserWarning,
            )

        if anisotropy is not None:
            return ndimage.distance_transform_edt(mask, sampling=anisotropy)
        return ndimage.distance_transform_edt(mask)

    if edt_downsample_factor > 1:
        zoom_factors = tuple(1.0 / edt_downsample_factor for _ in range(foreground_mask.ndim))
        foreground_ds = ndimage.zoom(
            foreground_mask.astype(np.float32), zoom_factors, order=0
        ).astype(bool)

        if resolved_anisotropy is not None:
            ds_anisotropy = tuple(a / edt_downsample_factor for a in resolved_anisotropy)
        else:
            ds_anisotropy = None

        distance_ds = _edt_on_mask(foreground_ds, ds_anisotropy)
        distance_fg = ndimage.zoom(
            distance_ds,
            tuple(edt_downsample_factor for _ in range(distance_ds.ndim)),
            order=1,
        )
        distance_fg *= edt_downsample_factor

        if distance_fg.shape != foreground_mask.shape:
            slices = tuple(
                slice(0, min(s1, s2)) for s1, s2 in zip(distance_fg.shape, foreground_mask.shape)
            )
            corrected = np.zeros(foreground_mask.shape, dtype=distance_fg.dtype)
            corrected[slices] = distance_fg[slices]
            distance_fg = corrected

        return distance_fg

    return _edt_on_mask(foreground_mask, resolved_anisotropy)


@jit(nopython=True, cache=True)
def connected_components_affinity_3d_numba(
    hard_aff: np.ndarray, edge_offset: int = 0
) -> np.ndarray:
    """Numba-accelerated connected components from 3D affinities."""
    visited = np.zeros(hard_aff.shape[1:], dtype=np.uint8)
    seg = np.zeros(hard_aff.shape[1:], dtype=np.uint32)
    cur_id = 1
    max_stack = visited.shape[0] * visited.shape[1] * visited.shape[2]
    stack_x = np.zeros(max_stack, dtype=np.int32)
    stack_y = np.zeros(max_stack, dtype=np.int32)
    stack_z = np.zeros(max_stack, dtype=np.int32)

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:
                    stack_size = 0

                    stack_x[stack_size] = i
                    stack_y[stack_size] = j
                    stack_z[stack_size] = k
                    stack_size += 1
                    visited[i, j, k] = True

                    while stack_size > 0:
                        stack_size -= 1
                        x = stack_x[stack_size]
                        y = stack_y[stack_size]
                        z = stack_z[stack_size]

                        seg[x, y, z] = cur_id

                        if (
                            x + 1 < visited.shape[0]
                            and hard_aff[0, x + edge_offset, y, z]
                            and not visited[x + 1, y, z]
                        ):
                            stack_x[stack_size] = x + 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x + 1, y, z] = True

                        if (
                            y + 1 < visited.shape[1]
                            and hard_aff[1, x, y + edge_offset, z]
                            and not visited[x, y + 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y + 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y + 1, z] = True

                        if (
                            z + 1 < visited.shape[2]
                            and hard_aff[2, x, y, z + edge_offset]
                            and not visited[x, y, z + 1]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z + 1
                            stack_size += 1
                            visited[x, y, z + 1] = True

                        if (
                            x - 1 >= 0
                            and hard_aff[0, x - 1 + edge_offset, y, z]
                            and not visited[x - 1, y, z]
                        ):
                            stack_x[stack_size] = x - 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x - 1, y, z] = True

                        if (
                            y - 1 >= 0
                            and hard_aff[1, x, y - 1 + edge_offset, z]
                            and not visited[x, y - 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y - 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y - 1, z] = True

                        if (
                            z - 1 >= 0
                            and hard_aff[2, x, y, z - 1 + edge_offset]
                            and not visited[x, y, z - 1]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z - 1
                            stack_size += 1
                            visited[x, y, z - 1] = True

                    cur_id += 1

    return seg


__all__ = [
    "NUMBA_AVAILABLE",
    "compute_edt",
    "connected_components_affinity_3d_numba",
]
