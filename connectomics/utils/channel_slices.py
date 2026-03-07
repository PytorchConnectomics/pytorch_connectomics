"""Shared channel-slice range resolution helpers."""

from __future__ import annotations


def resolve_channel_slice_bounds(
    channel_slice: tuple[int, int],
    *,
    num_channels: int,
    context: str = "channel slice",
    negative_index_offset: int = 0,
    end_minus_one_full_span: bool = False,
) -> tuple[int, int]:
    """Resolve possibly-negative channel bounds to absolute half-open indices.

    Args:
        channel_slice: Half-open slice ``(start, end)``.
        num_channels: Channel count of the tensor/output.
        context: Label used in validation error messages.
        negative_index_offset: Offset added for negative bounds before resolution.
            ``0`` matches python-style negative indexing.
            ``1`` matches legacy boundary-offset indexing used by some TTA profiles.
        end_minus_one_full_span: When true, ``end=-1`` resolves to ``num_channels``
            (i.e., "all remaining channels").
    """
    if negative_index_offset not in (0, 1):
        raise ValueError(
            f"negative_index_offset must be 0 or 1, got {negative_index_offset}."
        )

    start_ch, end_ch = int(channel_slice[0]), int(channel_slice[1])

    start_idx = (
        start_ch
        if start_ch >= 0
        else num_channels + start_ch + negative_index_offset
    )
    if end_ch >= 0:
        end_idx = end_ch
    elif end_minus_one_full_span and end_ch == -1:
        end_idx = num_channels
    else:
        end_idx = num_channels + end_ch + negative_index_offset

    if start_idx < 0 or start_idx >= num_channels:
        raise ValueError(
            f"Invalid {context} {channel_slice} for tensor with {num_channels} channels: "
            f"start index resolves to {start_idx} (expected in [0, {num_channels - 1}])."
        )
    if end_idx < 0 or end_idx > num_channels:
        raise ValueError(
            f"Invalid {context} {channel_slice} for tensor with {num_channels} channels: "
            f"end index resolves to {end_idx} (expected in [0, {num_channels}])."
        )
    if end_idx <= start_idx:
        raise ValueError(
            f"Invalid {context} {channel_slice} for tensor with {num_channels} channels: "
            f"resolved range [{start_idx}, {end_idx}) is empty or inverted."
        )

    return start_idx, end_idx
