"""Streaming validity-aware aggregation for test-time augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from .tta_affinity import ValidityEntry, ViewValidity


class TTAEnsembleAccumulator:
    """Aggregate fully valid channels through the legacy path and partial channels safely."""

    def __init__(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        mode_map: Sequence[str],
        partial_channels: Sequence[int],
        distributed_sharding: bool,
        max_views: int,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.dtype = dtype
        self.device = device
        self.mode_map = tuple(str(mode) for mode in mode_map)
        if len(self.shape) < 3 or len(self.mode_map) != self.shape[1]:
            raise ValueError(
                f"Invalid TTA accumulator shape/modes: shape={self.shape}, "
                f"modes={len(self.mode_map)}."
            )
        invalid_modes = sorted(set(self.mode_map) - {"mean", "min", "max"})
        if invalid_modes:
            raise ValueError(f"Unknown TTA ensemble modes: {invalid_modes}.")

        self.partial_channels = tuple(sorted({int(channel) for channel in partial_channels}))
        if any(channel < 0 or channel >= self.shape[1] for channel in self.partial_channels):
            raise ValueError(
                f"Partial TTA channels {self.partial_channels} are invalid for {self.shape[1]} "
                "output channels."
            )
        partial_set = set(self.partial_channels)
        self.full_channels = tuple(
            channel for channel in range(self.shape[1]) if channel not in partial_set
        )
        self.distributed_sharding = bool(distributed_sharding)
        self.num_predictions = 0
        self.legacy_result = torch.zeros(self.shape, device=device, dtype=dtype)

        partial_shape = (self.shape[0], len(self.partial_channels), *self.shape[2:])
        self.partial_statistics = torch.empty(partial_shape, device=device, dtype=torch.float32)
        for partial_index, channel in enumerate(self.partial_channels):
            mode = self.mode_map[channel]
            fill = 0.0 if mode == "mean" else (float("inf") if mode == "min" else float("-inf"))
            self.partial_statistics[:, partial_index, ...].fill_(fill)
        count_dtype = torch.uint8 if int(max_views) < 256 else torch.int16
        self.partial_counts = torch.zeros(partial_shape, device=device, dtype=count_dtype)

    @property
    def has_partial_channels(self) -> bool:
        return bool(self.partial_channels)

    @staticmethod
    def _consecutive_groups(channels: Sequence[int], mode_map: Sequence[str]):
        channels = tuple(channels)
        index = 0
        while index < len(channels):
            start = channels[index]
            mode = mode_map[start]
            stop = start + 1
            index += 1
            while (
                index < len(channels)
                and channels[index] == stop
                and mode_map[channels[index]] == mode
            ):
                stop += 1
                index += 1
            yield slice(start, stop), mode

    def _add_full_channels(self, prediction: torch.Tensor) -> None:
        for channel_slice, mode in self._consecutive_groups(self.full_channels, self.mode_map):
            incoming = prediction[:, channel_slice]
            current = self.legacy_result[:, channel_slice]
            if self.num_predictions == 0:
                current.copy_(incoming)
            elif mode == "mean":
                if self.distributed_sharding:
                    current += incoming
                else:
                    delta = incoming - current
                    current += delta / (self.num_predictions + 1)
            elif mode == "min":
                current.copy_(torch.minimum(current, incoming))
            else:
                current.copy_(torch.maximum(current, incoming))

    @staticmethod
    def _normalize_boolean_validity(validity: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        mask = validity.to(device=values.device, dtype=torch.bool)
        if mask.ndim == values.ndim - 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != values.ndim:
            raise ValueError(
                f"TTA validity tensor rank {mask.ndim} does not match channel value rank "
                f"{values.ndim}."
            )
        if mask.shape[0] == 1 and values.shape[0] != 1:
            mask = mask.expand(values.shape[0], *mask.shape[1:])
        if tuple(mask.shape) != tuple(values.shape):
            raise ValueError(
                f"TTA validity shape {tuple(mask.shape)} does not match channel value shape "
                f"{tuple(values.shape)}."
            )
        return mask

    def _add_partial_channel(
        self,
        prediction: torch.Tensor,
        *,
        channel: int,
        partial_index: int,
        validity: ValidityEntry,
    ) -> None:
        values = prediction[:, channel, ...].to(torch.float32)
        statistics = self.partial_statistics[:, partial_index, ...]
        counts = self.partial_counts[:, partial_index, ...]
        mode = self.mode_map[channel]

        if validity is None:
            if mode == "mean":
                statistics += values
            elif mode == "min":
                statistics.copy_(torch.minimum(statistics, values))
            else:
                statistics.copy_(torch.maximum(statistics, values))
            counts += 1
            return

        if isinstance(validity, tuple):
            box = (slice(None), *validity)
            if mode == "mean":
                statistics[box] += values[box]
            elif mode == "min":
                statistics[box] = torch.minimum(statistics[box], values[box])
            else:
                statistics[box] = torch.maximum(statistics[box], values[box])
            counts[box] += 1
            return

        mask = self._normalize_boolean_validity(validity, values)
        if mode == "mean":
            statistics += torch.where(mask, values, torch.zeros_like(values))
        elif mode == "min":
            statistics.copy_(torch.where(mask, torch.minimum(statistics, values), statistics))
        else:
            statistics.copy_(torch.where(mask, torch.maximum(statistics, values), statistics))
        counts += mask.to(dtype=counts.dtype)

    def add(self, prediction: torch.Tensor, validity: ViewValidity) -> None:
        """Stream one preprocessed canonical prediction into the accumulator."""
        if tuple(prediction.shape) != self.shape:
            raise ValueError(
                f"TTA prediction shape {tuple(prediction.shape)} does not match accumulator "
                f"shape {self.shape}."
            )
        if len(validity.channels) != self.shape[1]:
            raise ValueError(
                f"TTA validity describes {len(validity.channels)} channels, expected "
                f"{self.shape[1]}."
            )
        incoming = prediction.to(device=self.device, dtype=self.dtype)
        self._add_full_channels(incoming)
        for partial_index, channel in enumerate(self.partial_channels):
            self._add_partial_channel(
                incoming,
                channel=channel,
                partial_index=partial_index,
                validity=validity.channels[channel],
            )
        self.num_predictions += 1

    def finalize(
        self,
        *,
        legacy_result: Optional[torch.Tensor] = None,
        partial_statistics: Optional[torch.Tensor] = None,
        partial_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the aggregate and reject any partial channel with zero coverage."""
        result = (self.legacy_result if legacy_result is None else legacy_result).clone()
        statistics = self.partial_statistics if partial_statistics is None else partial_statistics
        counts = self.partial_counts if partial_counts is None else partial_counts
        for partial_index, channel in enumerate(self.partial_channels):
            channel_counts = counts[:, partial_index, ...]
            zero = channel_counts == 0
            if torch.any(zero):
                first = tuple(int(value) for value in torch.nonzero(zero, as_tuple=False)[0])
                raise RuntimeError(
                    f"TTA ensemble has zero valid contributions for channel {channel} at "
                    f"voxel index {first}."
                )
            channel_statistics = statistics[:, partial_index, ...]
            if self.mode_map[channel] == "mean":
                channel_statistics = channel_statistics / channel_counts.to(torch.float32)
            result[:, channel, ...] = channel_statistics.to(dtype=self.dtype)
        return result


__all__ = ["TTAEnsembleAccumulator"]
