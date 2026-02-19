"""Decoding pipeline helpers."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .base import DecodeStep
from .registry import DecoderRegistry, DEFAULT_DECODER_REGISTRY, register_builtin_decoders


def _coerce_kwargs(kwargs: Any) -> dict:
    if kwargs is None:
        return {}
    if hasattr(kwargs, "items"):
        return dict(kwargs)
    raise TypeError(
        f"Decode kwargs must be a mapping, got {type(kwargs).__name__}"
    )


def normalize_decode_modes(decode_modes: Iterable[Any]) -> List[DecodeStep]:
    """Normalize decode configuration entries into DecodeStep objects."""
    steps: List[DecodeStep] = []
    for mode in decode_modes:
        if isinstance(mode, DecodeStep):
            steps.append(DecodeStep(name=mode.name, kwargs=_coerce_kwargs(mode.kwargs)))
            continue

        if hasattr(mode, "name"):
            name = mode.name
            kwargs = _coerce_kwargs(getattr(mode, "kwargs", {}))
            steps.append(DecodeStep(name=name, kwargs=kwargs))
            continue

        if isinstance(mode, dict):
            name = mode.get("name")
            kwargs = _coerce_kwargs(mode.get("kwargs", {}))
            steps.append(DecodeStep(name=name, kwargs=kwargs))
            continue

        raise TypeError(f"Unsupported decode mode type: {type(mode).__name__}")

    for step in steps:
        if not step.name:
            raise ValueError("Decode step is missing required field 'name'.")

    return steps


def _prepare_batched_input(data: np.ndarray) -> Tuple[np.ndarray, int]:
    arr = np.asarray(data)
    if arr.ndim == 5:
        return arr, arr.shape[0]
    if arr.ndim == 4:
        return arr[np.newaxis, ...], 1
    if arr.ndim == 3:
        return arr[np.newaxis, np.newaxis, ...], 1
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, np.newaxis, ...], 1
    raise ValueError(
        f"Expected input with 2-5 dimensions, got shape {arr.shape}."
    )


def apply_decode_pipeline(
    data: np.ndarray,
    decode_modes: Sequence[Any] | None,
    registry: DecoderRegistry | None = None,
) -> np.ndarray:
    """Apply configured decode steps to prediction data."""
    if not decode_modes:
        return data

    register_builtin_decoders()
    registry = registry or DEFAULT_DECODER_REGISTRY
    steps = normalize_decode_modes(decode_modes)
    batched, batch_size = _prepare_batched_input(data)

    results: List[np.ndarray] = []
    for batch_idx in range(batch_size):
        sample = batched[batch_idx]
        for step in steps:
            try:
                decoder = registry.get(step.name)
            except KeyError as exc:
                available = ", ".join(registry.available())
                raise ValueError(
                    f"Unknown decode function '{step.name}'. "
                    f"Available functions: [{available}]."
                ) from exc

            try:
                sample = decoder(sample, **step.kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Error applying decode function '{step.name}': {exc}"
                ) from exc

        results.append(sample)

    if len(results) == 1:
        return results[0]
    return np.stack(results, axis=0)


def resolve_decode_modes_from_cfg(cfg: Any) -> Sequence[Any] | None:
    """Resolve decode mode list from config.

    Priority:
    1. ``cfg.test.decoding``
    2. ``cfg.inference.decoding``
    """
    if hasattr(cfg, "test") and cfg.test and hasattr(cfg.test, "decoding"):
        return cfg.test.decoding
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "decoding"):
        return cfg.inference.decoding
    return None


def apply_decode_mode(cfg: Any, data: np.ndarray, *, verbose: bool = True) -> np.ndarray:
    """Apply decode pipeline resolved from ``test.decoding`` or ``inference.decoding``."""
    decode_modes = resolve_decode_modes_from_cfg(cfg)
    if not decode_modes:
        if verbose:
            print("  No decoding configuration found (test.decoding or inference.decoding)")
        return data

    if verbose:
        source = "test.decoding"
        if not (hasattr(cfg, "test") and cfg.test and hasattr(cfg.test, "decoding")):
            source = "inference.decoding"
        print(f"  Using {source}: {decode_modes}")

    return apply_decode_pipeline(data, decode_modes)
