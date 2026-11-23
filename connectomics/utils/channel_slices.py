"""Shared channel-selector parsing and resolution helpers."""

from __future__ import annotations

from typing import Any, Sequence, TypeAlias

ChannelRangeSelector: TypeAlias = int | str
ChannelSelector: TypeAlias = int | str | Sequence[int]


def _parse_selector_string(value: str, *, context: str) -> int | slice:
    """Parse a Python-style channel selector string.

    Supported forms:
    - ``"3"`` or ``"-1"`` for a single channel index
    - ``":"``, ``"0:"``, ``":-1"``, ``"1:3"`` for half-open slices

    Step values are intentionally unsupported to keep selector semantics simple.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{context} must not be empty.")

    if ":" not in text:
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(
                f"{context} must be an integer index or a Python-style slice string, got {value!r}."
            ) from exc

    if text.count(":") != 1:
        raise ValueError(
            f"{context} must use step-free Python slice syntax 'start:end', got {value!r}."
        )

    start_text, stop_text = text.split(":", 1)
    try:
        start = int(start_text.strip()) if start_text.strip() else None
        stop = int(stop_text.strip()) if stop_text.strip() else None
    except ValueError as exc:
        raise ValueError(
            f"{context} must use integer slice bounds in 'start:end', got {value!r}."
        ) from exc

    return slice(start, stop)


def _format_slice(selector: slice) -> str:
    start = "" if selector.start is None else str(selector.start)
    stop = "" if selector.stop is None else str(selector.stop)
    return f"{start}:{stop}"


def normalize_channel_range_selector(
    selector: Any,
    *,
    context: str = "channel selector",
) -> ChannelRangeSelector | None:
    """Normalize a contiguous channel-range selector.

    Accepted forms:
    - ``None`` for "all channels"
    - ``int`` for a single channel
    - ``str`` for an integer index or a Python-style slice string
    """
    if selector is None:
        return None

    if isinstance(selector, int):
        return int(selector)

    if isinstance(selector, str):
        parsed = _parse_selector_string(selector, context=context)
        if isinstance(parsed, int):
            return parsed
        return _format_slice(parsed)

    raise TypeError(
        f"{context} must be an int or a Python-style slice string, got {type(selector).__name__}."
    )


def normalize_channel_selector(
    selector: Any,
    *,
    context: str = "channel selector",
) -> int | str | list[int] | None:
    """Normalize a general channel selector.

    Accepted forms:
    - ``None`` for "all channels"
    - ``int`` for a single channel
    - ``str`` for an integer index or a Python-style slice string
    - ``list[int]`` / ``tuple[int, ...]`` for explicit channel indices
    """
    if selector is None:
        return None

    if isinstance(selector, (int, str)):
        return normalize_channel_range_selector(selector, context=context)

    if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes)):
        if len(selector) == 0:
            raise ValueError(f"{context} must not be an empty channel list.")
        indices: list[int] = []
        for raw in selector:
            if isinstance(raw, int):
                indices.append(int(raw))
                continue
            if isinstance(raw, str):
                try:
                    indices.append(int(raw.strip()))
                    continue
                except ValueError as exc:
                    raise ValueError(
                        f"{context} channel lists must contain only integer indices, got {raw!r}."
                    ) from exc
            raise TypeError(
                f"{context} channel lists must contain only integers, got {type(raw).__name__}."
            )
        return indices

    raise TypeError(
        f"{context} must be an int, a Python-style slice string, or an explicit list of ints; "
        f"got {type(selector).__name__}."
    )


def resolve_channel_index(
    index_value: int,
    *,
    num_channels: int,
    context: str = "channel selector",
) -> int:
    """Resolve a possibly-negative channel index using Python indexing rules."""
    index = int(index_value)
    if index < 0:
        index += num_channels
    if index < 0 or index >= num_channels:
        raise ValueError(
            f"Invalid {context} {index_value!r} for tensor with {num_channels} channels: "
            f"resolved index {index} is out of bounds."
        )
    return index


def resolve_channel_range(
    selector: ChannelRangeSelector | None,
    *,
    num_channels: int,
    context: str = "channel selector",
) -> tuple[int, int]:
    """Resolve a contiguous channel selector to absolute half-open bounds."""
    if num_channels <= 0:
        raise ValueError(f"{context} requires num_channels > 0, got {num_channels}.")

    normalized = normalize_channel_range_selector(selector, context=context)
    if normalized is None:
        return (0, num_channels)

    if isinstance(normalized, int):
        index = resolve_channel_index(normalized, num_channels=num_channels, context=context)
        return (index, index + 1)

    parsed = _parse_selector_string(normalized, context=context)
    if not isinstance(parsed, slice):
        raise ValueError(f"{context} must resolve to a contiguous slice, got {normalized!r}.")

    start_idx = 0 if parsed.start is None else int(parsed.start)
    stop_idx = num_channels if parsed.stop is None else int(parsed.stop)

    if start_idx < 0:
        start_idx += num_channels
    if stop_idx < 0:
        stop_idx += num_channels

    if start_idx < 0 or start_idx >= num_channels:
        raise ValueError(
            f"Invalid {context} {normalized!r} for tensor with {num_channels} channels: "
            f"resolved start index {start_idx} is out of bounds."
        )
    if stop_idx < 0 or stop_idx > num_channels:
        raise ValueError(
            f"Invalid {context} {normalized!r} for tensor with {num_channels} channels: "
            f"resolved stop index {stop_idx} is out of bounds."
        )
    if stop_idx <= start_idx:
        raise ValueError(
            f"Invalid {context} {normalized!r} for tensor with {num_channels} channels: "
            f"resolved range [{start_idx}, {stop_idx}) is empty or inverted."
        )
    return (start_idx, stop_idx)


def resolve_channel_indices(
    selector: ChannelSelector | None,
    *,
    num_channels: int,
    context: str = "channel selector",
) -> list[int]:
    """Resolve a channel selector to explicit channel indices."""
    if num_channels <= 0:
        raise ValueError(f"{context} requires num_channels > 0, got {num_channels}.")

    normalized = normalize_channel_selector(selector, context=context)
    if normalized is None:
        return list(range(num_channels))

    if isinstance(normalized, list):
        return [
            resolve_channel_index(index, num_channels=num_channels, context=context)
            for index in normalized
        ]

    if isinstance(normalized, int):
        return [resolve_channel_index(normalized, num_channels=num_channels, context=context)]

    start_idx, stop_idx = resolve_channel_range(
        normalized,
        num_channels=num_channels,
        context=context,
    )
    return list(range(start_idx, stop_idx))


def infer_min_required_channels(
    selector: ChannelSelector | None,
    *,
    context: str = "channel selector",
) -> int | None:
    """Infer the minimum channel count needed for a selector to be valid."""
    normalized = normalize_channel_selector(selector, context=context)
    if normalized is None:
        return None

    if isinstance(normalized, list):
        if not normalized:
            raise ValueError(f"{context} must not be an empty channel list.")
        return max(infer_min_required_channels(index, context=context) or 1 for index in normalized)

    if isinstance(normalized, int):
        return normalized + 1 if normalized >= 0 else -normalized

    parsed = _parse_selector_string(normalized, context=context)
    if not isinstance(parsed, slice):
        raise ValueError(f"{context} must resolve to a slice, got {normalized!r}.")

    bounds = [abs(v) + 2 for v in (parsed.start, parsed.stop) if v is not None]
    search_limit = max([1, *bounds])
    for num_channels in range(1, search_limit + 1):
        try:
            resolve_channel_range(
                normalized,
                num_channels=num_channels,
                context=context,
            )
        except ValueError:
            continue
        return num_channels

    raise ValueError(f"Could not infer a valid channel count for {context} {normalized!r}.")


__all__ = [
    "ChannelRangeSelector",
    "ChannelSelector",
    "infer_min_required_channels",
    "normalize_channel_range_selector",
    "normalize_channel_selector",
    "resolve_channel_index",
    "resolve_channel_indices",
    "resolve_channel_range",
]
