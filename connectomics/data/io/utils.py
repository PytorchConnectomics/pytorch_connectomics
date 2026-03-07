"""Utility functions for data I/O operations."""

from __future__ import annotations

import numpy as np


def rgb_to_seg(rgb: np.ndarray) -> np.ndarray:
    """Convert VAST RGB segmentation format to IDs.

    Each pixel's RGB values are combined to create a
    unique 24-bit segmentation ID.
    """
    if rgb.ndim == 2 or rgb.shape[-1] == 1:
        return np.squeeze(rgb)
    if rgb.ndim == 3:
        return (
            rgb[:, :, 0].astype(np.uint32) * 65536
            + rgb[:, :, 1].astype(np.uint32) * 256
            + rgb[:, :, 2].astype(np.uint32)
        )
    if rgb.ndim == 4:
        return (
            rgb[:, :, :, 0].astype(np.uint32) * 65536
            + rgb[:, :, :, 1].astype(np.uint32) * 256
            + rgb[:, :, :, 2].astype(np.uint32)
        )
    raise ValueError(f"Unsupported ndim: {rgb.ndim}")
