"""
Quantization functions for PyTorch Connectomics processing.
"""

from __future__ import print_function, division
import numpy as np
import torch
import scipy


def energy_quantize(energy, levels=10):
    """Convert the continuous energy map into the quantized version."""
    # np.digitize returns the indices of the bins to which each
    # value in input array belongs. The default behavior is bins[i-1] <= x < bins[i].
    bins = [-1.0]
    for i in range(levels):
        bins.append(float(i) / float(levels))
    bins.append(1.1)
    bins = np.array(bins)
    quantized = np.digitize(energy, bins) - 1
    return quantized.astype(np.int64)


def decode_quantize(output, mode="max"):
    assert isinstance(output, (torch.Tensor, np.ndarray))
    assert mode in ["max", "mean"]
    if isinstance(output, torch.Tensor):
        return _decode_quant_torch(output, mode)
    else:
        return _decode_quant_numpy(output, mode)


def _decode_quant_torch(output, mode="max"):
    # output: torch tensor of size (B, C, *)
    if mode == "max":
        pred = torch.argmax(output, axis=1)
        max_value = output.size()[1]
        energy = pred / float(max_value)
    elif mode == "mean":
        out_shape = output.shape
        bins = np.array([0.1 * float(x - 1) for x in range(11)])
        bins = torch.from_numpy(bins.astype(np.float32))
        bins = bins.view(1, -1, 1)
        bins = bins.to(output.device)

        output = output.view(out_shape[0], out_shape[1], -1)  # (B, C, *)
        pred = torch.softmax(output, axis=1)
        energy = (pred * bins).view(out_shape).sum(1)

    return energy


def _decode_quant_numpy(output, mode="max"):
    # output: numpy array of shape (C, *)
    if mode == "max":
        pred = np.argmax(output, axis=0)
        max_value = output.shape[0]
        energy = pred / float(max_value)
    elif mode == "mean":
        out_shape = output.shape
        bins = np.array([0.1 * float(x - 1) for x in range(11)])
        bins = bins.reshape(-1, 1)

        output = output.reshape(out_shape[0], -1)  # (C, *)
        pred = scipy.special.softmax(output, axis=0)
        energy = (pred * bins).reshape(out_shape).sum(0)

    return energy


__all__ = ["energy_quantize", "decode_quantize"]
