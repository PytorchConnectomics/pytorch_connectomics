"""Shared helpers for MONAI map-style augmentation transforms."""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np
import torch


def to_numpy(img: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, bool, Any]:
    """Convert to numpy, returning (array, was_tensor, device)."""
    if isinstance(img, torch.Tensor):
        return img.cpu().numpy(), True, img.device
    return img, False, None


def from_numpy(img: np.ndarray, was_tensor: bool, device: Any) -> Union[np.ndarray, torch.Tensor]:
    """Convert back to tensor if input was tensor."""
    if was_tensor:
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        return torch.from_numpy(img).to(device)
    return img


def infer_depth_axis(arr: np.ndarray) -> int:
    """Infer the depth axis for channel-first volumes."""
    if arr.ndim >= 4 and arr.shape[0] <= 4:
        return 1
    return 0


def has_channel_axis(arr: np.ndarray) -> bool:
    """Heuristic channel-first detection for image-like tensors."""
    return arr.ndim >= 4 and arr.shape[0] <= 4


def infer_spatial_rank(arr: np.ndarray) -> int:
    """Infer the number of spatial axes."""
    return arr.ndim - 1 if has_channel_axis(arr) else arr.ndim


def spatial_axis_to_array_axis(arr: np.ndarray, spatial_axis: int) -> int:
    """Convert a spatial-axis index to an ndarray axis index."""
    return spatial_axis + 1 if has_channel_axis(arr) else spatial_axis


def sample_spatial_axis(
    rng: np.random.RandomState,
    spec: Union[int, str, Tuple[int, ...], List[int]],
    spatial_rank: int,
) -> int:
    """Sample a spatial axis index from an int/list/random spec."""
    if spatial_rank <= 0:
        raise ValueError("spatial_rank must be positive")

    if isinstance(spec, str):
        if spec not in {"random", "all", "any"}:
            raise ValueError(f"Unsupported spatial axis spec: {spec}")
        choices = list(range(spatial_rank))
    elif isinstance(spec, (tuple, list)):
        choices = [int(a) for a in spec]
    else:
        choices = [int(spec)]

    if not choices:
        raise ValueError("spatial axis choices cannot be empty")
    invalid = [ax for ax in choices if ax < 0 or ax >= spatial_rank]
    if invalid:
        raise ValueError(f"Spatial axis choices {invalid} out of range for rank {spatial_rank}")

    if len(choices) == 1:
        return choices[0]
    return int(rng.choice(np.asarray(choices, dtype=np.int64)))


def sample_non_identity_permutation(
    rng: np.random.RandomState,
    spatial_rank: int,
) -> np.ndarray:
    """Sample a spatial permutation that is not the identity."""
    identity = np.arange(spatial_rank, dtype=np.int64)
    for _ in range(8):
        permutation = rng.permutation(spatial_rank)
        if not np.array_equal(permutation, identity):
            return permutation.astype(np.int64)
    return permutation.astype(np.int64)


def sample_non_identity_rotate_ks(rng: np.random.RandomState) -> Tuple[int, int, int]:
    """Sample quarter-turn counts that are not all zero."""
    rotate_ks = (0, 0, 0)
    for _ in range(8):
        rotate_ks = tuple(int(rng.randint(0, 4)) for _ in range(3))
        if any(k != 0 for k in rotate_ks):
            return rotate_ks
    return rotate_ks


def sample_count(
    rng: np.random.RandomState,
    spec: Union[int, Tuple[int, int], List[int]],
    max_count: int,
) -> int:
    """Sample a non-negative count from an int or inclusive [min, max] pair."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 0:
            return 0
        if len(spec) == 1:
            count = int(spec[0])
        else:
            low = int(spec[0])
            high = int(spec[1])
            if high < low:
                low, high = high, low
            count = int(rng.randint(low, high + 1))
    else:
        count = int(spec)
    return min(max(count, 0), max_count)


__all__ = [
    "from_numpy",
    "has_channel_axis",
    "infer_depth_axis",
    "infer_spatial_rank",
    "sample_count",
    "sample_non_identity_permutation",
    "sample_non_identity_rotate_ks",
    "sample_spatial_axis",
    "spatial_axis_to_array_axis",
    "to_numpy",
]
