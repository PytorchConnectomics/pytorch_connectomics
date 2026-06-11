"""Pure array transforms for decoder graphs."""

from __future__ import annotations

import numpy as np

from ...utils.channel_slices import resolve_channel_indices


def channel_gate(
    predictions: np.ndarray,
    *,
    signal_channels,
    gate_channel,
) -> np.ndarray:
    """Multiply selected signal channels by a single gate channel."""
    arr = np.asarray(predictions)
    if arr.ndim < 1:
        raise ValueError("channel_gate expects an array with a channel axis.")

    signal_indices = resolve_channel_indices(
        signal_channels,
        num_channels=int(arr.shape[0]),
        context="channel_gate.signal_channels",
    )
    gate_indices = resolve_channel_indices(
        gate_channel,
        num_channels=int(arr.shape[0]),
        context="channel_gate.gate_channel",
    )
    if len(gate_indices) != 1:
        raise ValueError(
            f"channel_gate.gate_channel must resolve to one channel, got {gate_indices}."
        )

    gated = arr[signal_indices] * arr[gate_indices[0] : gate_indices[0] + 1]
    return gated.astype(arr.dtype, copy=False)


__all__ = ["channel_gate"]
