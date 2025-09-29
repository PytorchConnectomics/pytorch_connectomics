"""
Quantization functions for PyTorch Connectomics processing.
"""

from __future__ import print_function, division
import numpy as np


def quantize_volume(volume, num_levels=256):
    """Quantize volume to specified number of levels."""
    min_val, max_val = volume.min(), volume.max()
    if max_val > min_val:
        volume_norm = (volume - min_val) / (max_val - min_val)
        volume_quantized = np.floor(volume_norm * (num_levels - 1)).astype(np.uint8)
    else:
        volume_quantized = np.zeros_like(volume, dtype=np.uint8)

    return volume_quantized


__all__ = ['quantize_volume']