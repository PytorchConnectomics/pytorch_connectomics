"""Decoding pipeline helpers."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from ..utils.channel_slices import resolve_channel_indices
from .base import DecodeStep
from .registry import DEFAULT_DECODER_REGISTRY, DecoderRegistry, ensure_builtin_decoders_registered

logger = logging.getLogger(__name__)


def _coerce_kwargs(kwargs: Any) -> dict:
    if kwargs is None:
        return {}
    if hasattr(kwargs, "items"):
        return dict(kwargs)
    raise TypeError(f"Decode kwargs must be a mapping, got {type(kwargs).__name__}")


def _resolve_enabled(mode: Any) -> bool:
    """Extract the ``enabled`` flag from a decode mode entry (default ``True``)."""
    if isinstance(mode, dict):
        return bool(mode.get("enabled", True))
    return bool(getattr(mode, "enabled", True))


def normalize_decode_modes(decode_modes: Iterable[Any]) -> List[DecodeStep]:
    """Normalize decode configuration entries into DecodeStep objects.

    Entries with ``enabled: false`` are silently skipped.
    """
    steps: List[DecodeStep] = []
    for mode in decode_modes:
        enabled = _resolve_enabled(mode)

        if isinstance(mode, DecodeStep):
            steps.append(
                DecodeStep(enabled=enabled, name=mode.name, kwargs=_coerce_kwargs(mode.kwargs))
            )
            continue

        if hasattr(mode, "name"):
            name = mode.name
            kwargs = _coerce_kwargs(getattr(mode, "kwargs", {}))
            steps.append(DecodeStep(enabled=enabled, name=name, kwargs=kwargs))
            continue

        if isinstance(mode, dict):
            name = mode.get("name")
            kwargs = _coerce_kwargs(mode.get("kwargs", {}))
            steps.append(DecodeStep(enabled=enabled, name=name, kwargs=kwargs))
            continue

        raise TypeError(f"Unsupported decode mode type: {type(mode).__name__}")

    for step in steps:
        if step.enabled and not step.name:
            raise ValueError("Decode step is missing required field 'name'.")

    return steps


def _invert_axis_permutation(perm: Sequence[int]) -> List[int]:
    inv = [0, 0, 0]
    for i, p in enumerate(perm):
        inv[int(p)] = i
    return inv


def _apply_spatial_transpose(arr: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Permute the trailing 3 spatial axes by ``perm`` (a permutation of [0,1,2]).

    Accepts 3D ``(Z, Y, X)`` or 4D ``(C, Z, Y, X)`` arrays so this hook can
    run both before a decoder (4D affinity input) and on its 3D segmentation
    output (for the inverse transpose).
    """
    perm = tuple(int(p) for p in perm)
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"spatial_transpose must be a permutation of [0, 1, 2], got {list(perm)}")
    if arr.ndim == 3:
        return np.ascontiguousarray(np.transpose(arr, perm))
    if arr.ndim == 4:
        return np.ascontiguousarray(np.transpose(arr, (0, perm[0] + 1, perm[1] + 1, perm[2] + 1)))
    raise ValueError(
        f"spatial_transpose expects a 3D or 4D array, got {arr.ndim}D with shape {arr.shape}."
    )


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
    raise ValueError(f"Expected input with 2-5 dimensions, got shape {arr.shape}.")


def apply_decode_pipeline(
    data: np.ndarray,
    decode_modes: Sequence[Any] | None,
    registry: DecoderRegistry | None = None,
    *,
    on_step_complete: Any = None,
) -> np.ndarray:
    """Apply configured decode steps to prediction data.

    ``on_step_complete``, if provided, is invoked as
    ``on_step_complete(batch_idx, step, sample)`` after each decoder step
    runs, with the step's output array. Used by the decoding stage to write
    per-step intermediates without keeping them all in memory.
    """
    if not decode_modes:
        return data

    if registry is None:
        ensure_builtin_decoders_registered()
        registry = DEFAULT_DECODER_REGISTRY
    steps = [s for s in normalize_decode_modes(decode_modes) if s.enabled]
    if not steps:
        return data
    batched, batch_size = _prepare_batched_input(data)

    results: List[np.ndarray] = []
    for batch_idx in range(batch_size):
        sample = batched[batch_idx]
        original_sample = sample
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
                decoder_kwargs = dict(step.kwargs)
                for key, value in list(decoder_kwargs.items()):
                    if key.endswith("_channels"):
                        decoder_kwargs[key] = resolve_channel_indices(
                            value,
                            num_channels=int(sample.shape[0]),
                            context=f"decode kwargs {step.name}.{key}",
                        )
                spatial_transpose = decoder_kwargs.pop("spatial_transpose", None)
                use_original_input = bool(decoder_kwargs.pop("use_original_input", False))
                original_input_kwarg = decoder_kwargs.pop("original_input_kwarg", "affinities")
                if spatial_transpose:
                    sample = _apply_spatial_transpose(sample, spatial_transpose)
                if use_original_input:
                    original_for_decoder = original_sample
                    if spatial_transpose:
                        original_for_decoder = _apply_spatial_transpose(
                            original_for_decoder, spatial_transpose
                        )
                    decoder_kwargs[original_input_kwarg] = original_for_decoder
                sample = decoder(sample, **decoder_kwargs)
                if spatial_transpose:
                    sample = _apply_spatial_transpose(
                        sample, _invert_axis_permutation(spatial_transpose)
                    )
            except Exception as exc:
                raise RuntimeError(f"Error applying decode function '{step.name}': {exc}") from exc

            if on_step_complete is not None:
                on_step_complete(batch_idx, step, sample)

        results.append(sample)

    if len(results) == 1:
        return results[0]
    return np.stack(results, axis=0)


def resolve_decode_modes_from_cfg(cfg: Any) -> Sequence[Any] | None:
    """Resolve decode mode list from config.

    Uses the top-level ``cfg.decoding.steps`` section.
    """
    decoding = getattr(cfg, "decoding", None)
    if not decoding:
        return None
    steps = getattr(decoding, "steps", None)
    if steps:
        return steps
    return None


def apply_decode_mode(
    cfg: Any,
    data: np.ndarray,
    *,
    verbose: bool = True,
    on_step_complete: Any = None,
) -> np.ndarray:
    """Apply decode pipeline resolved from top-level ``decoding``."""
    decode_modes = resolve_decode_modes_from_cfg(cfg)
    if not decode_modes:
        if verbose:
            logger.info("No decoding configuration found (decoding)")
        return data

    if verbose:
        logger.info("Using decoding: %s", decode_modes)

    return apply_decode_pipeline(data, decode_modes, on_step_complete=on_step_complete)
