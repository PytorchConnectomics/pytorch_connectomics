"""
Quantization functions for PyTorch Connectomics processing.
"""

import numpy as np
import torch
import scipy


def energy_quantize(energy, levels=10):
    """Convert the continuous energy map into the quantized version.

    Args:
        energy: Continuous energy map with values typically in [-1, 1] or [0, 1].
        levels: Number of quantization levels (default: 10).

    Returns:
        Quantized energy map (int64) with values in [0, levels+1].
    """
    bins = np.concatenate([[-1.0], np.linspace(0.0, 1.0, levels + 1)])
    bins[-1] = 1.1  # ensure values at exactly 1.0 are captured
    quantized = np.digitize(energy, bins) - 1
    return quantized.astype(np.int64)


def decode_quantize(output, mode="max", levels=None):
    """Decode quantized energy maps back to continuous values.

    Args:
        output: Quantized output tensor/array. Shape (B, C, *) for torch or (C, *) for numpy.
        mode: 'max' (argmax) or 'mean' (weighted average).
        levels: Number of quantization levels. If None, inferred from channel dimension.
    """
    if not isinstance(output, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(output)}")
    if mode not in ("max", "mean"):
        raise ValueError(f"mode must be 'max' or 'mean', got {mode!r}")
    if isinstance(output, torch.Tensor):
        return _decode_quant_torch(output, mode, levels)
    else:
        return _decode_quant_numpy(output, mode, levels)


def _decode_quant_torch(output, mode="max", levels=None):
    # output: torch tensor of size (B, C, *)
    num_channels = output.size()[1]
    if levels is None:
        levels = num_channels

    if mode == "max":
        pred = torch.argmax(output, axis=1)
        energy = pred / float(levels)
    elif mode == "mean":
        out_shape = output.shape
        bins = torch.linspace(-1.0 / levels, 1.0 - 1.0 / levels, num_channels,
                              dtype=torch.float32, device=output.device)
        bins = bins.view(1, -1, *([1] * (output.ndim - 2)))

        pred = torch.softmax(output, dim=1)
        energy = (pred * bins).sum(1)

    return energy


def _decode_quant_numpy(output, mode="max", levels=None):
    # output: numpy array of shape (C, *)
    num_channels = output.shape[0]
    if levels is None:
        levels = num_channels

    if mode == "max":
        pred = np.argmax(output, axis=0)
        energy = pred / float(levels)
    elif mode == "mean":
        out_shape = output.shape
        bins = np.linspace(-1.0 / num_channels, 1.0 - 1.0 / num_channels, num_channels)
        bins = bins.reshape(-1, *([1] * (output.ndim - 1)))

        pred = scipy.special.softmax(output, axis=0)
        energy = (pred * bins).sum(0)

    return energy


__all__ = ["energy_quantize", "decode_quantize"]
