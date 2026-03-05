"""Shared label-overlap helpers for segmentation metrics/decoding."""

from __future__ import annotations

import numpy as np


def compute_label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute overlap matrix between two label arrays."""
    x_flat = np.asarray(x).ravel()
    y_flat = np.asarray(y).ravel()
    if x_flat.shape != y_flat.shape:
        raise ValueError(
            f"Label arrays must have the same flattened shape, got {x_flat.shape} vs {y_flat.shape}"
        )
    if x_flat.size == 0:
        return np.zeros((0, 0), dtype=np.uint64)

    overlap = np.zeros((1 + int(x_flat.max()), 1 + int(y_flat.max())), dtype=np.uint64)
    np.add.at(overlap, (x_flat, y_flat), 1)
    return overlap


__all__ = ["compute_label_overlap"]
