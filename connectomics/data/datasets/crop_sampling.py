"""Shared random/center crop position helpers for dataset sampling."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _randint_inclusive(rng, low: int, high: int) -> int:
    """Sample integer in [low, high] across random API variants."""
    if high <= low:
        return int(low)

    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(low, high + 1))
    if hasattr(rng, "randint"):
        # np.random.RandomState.randint uses exclusive upper bound, while
        # random.Random.randint is inclusive; clamp for consistent behavior.
        value = int(rng.randint(low, high + 1))
        return min(max(value, low), high)

    raise TypeError(f"Unsupported RNG type for integer sampling: {type(rng)}")


def _rand_index(rng, size: int) -> int:
    """Sample index in [0, size-1] across random API variants."""
    if size <= 0:
        raise ValueError(f"size must be > 0 for index sampling, got {size}")
    return _randint_inclusive(rng, 0, size - 1)


def random_crop_position(
    vol_size: Sequence[int],
    patch_size: Sequence[int],
    *,
    rng,
    mask_nonzero_coords: Optional[np.ndarray] = None,
    mask_bbox: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[int, ...]:
    """Sample a random crop start index for 2D/3D volumes."""
    ndim = len(patch_size)
    if len(vol_size) != ndim:
        raise ValueError(
            f"vol_size and patch_size must have same ndim, got {len(vol_size)} vs {ndim}"
        )

    if mask_nonzero_coords is not None and len(mask_nonzero_coords) > 0:
        idx = _rand_index(rng, len(mask_nonzero_coords))
        center = mask_nonzero_coords[idx]
        positions = []
        for i in range(ndim):
            vol_max = max(0, vol_size[i] - patch_size[i])
            pos = int(center[i]) - patch_size[i] // 2
            positions.append(max(0, min(pos, vol_max)))
        return tuple(positions)

    if mask_bbox is not None:
        positions = []
        for i in range(ndim):
            vol_max = max(0, vol_size[i] - patch_size[i])
            min_start = max(0, mask_bbox[i][0] - patch_size[i] + 1)
            max_start = min(vol_max, mask_bbox[i][1] - 1)
            if min_start > max_start:
                min_start, max_start = 0, vol_max
            positions.append(_randint_inclusive(rng, min_start, max_start))
        return tuple(positions)

    return tuple(
        _randint_inclusive(rng, 0, max(0, vol_size[i] - patch_size[i])) for i in range(ndim)
    )


def center_crop_position(
    vol_size: Sequence[int],
    patch_size: Sequence[int],
) -> Tuple[int, ...]:
    """Compute center crop start index for 2D/3D volumes."""
    if len(vol_size) != len(patch_size):
        raise ValueError(
            f"vol_size and patch_size must have same ndim, got {len(vol_size)} vs {len(patch_size)}"
        )
    return tuple(max(0, (vol_size[i] - patch_size[i]) // 2) for i in range(len(patch_size)))
