"""Decoder registry for configurable decode pipelines."""

from __future__ import annotations

from typing import Dict, List

from .base import DecodeFunction


class DecoderRegistry:
    """Name -> decoder function registry."""

    def __init__(self) -> None:
        self._decoders: Dict[str, DecodeFunction] = {}

    def register(self, name: str, fn: DecodeFunction, *, overwrite: bool = False) -> None:
        """Register a decoder function."""
        if not name or not isinstance(name, str):
            raise ValueError("Decoder name must be a non-empty string.")
        if not callable(fn):
            raise TypeError(f"Decoder '{name}' must be callable.")

        existing = self._decoders.get(name)
        if existing is not None and not overwrite and existing is not fn:
            raise ValueError(
                f"Decoder '{name}' already registered. Use overwrite=True to replace it."
            )
        self._decoders[name] = fn

    def get(self, name: str) -> DecodeFunction:
        """Get decoder function by name."""
        try:
            return self._decoders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._decoders))
            raise KeyError(
                f"Unknown decoder '{name}'. Available decoders: [{available}]"
            ) from exc

    def available(self) -> List[str]:
        """Return registered decoder names."""
        return sorted(self._decoders.keys())


DEFAULT_DECODER_REGISTRY = DecoderRegistry()


def register_decoder(name: str, fn: DecodeFunction, *, overwrite: bool = False) -> None:
    """Register decoder in the default registry."""
    DEFAULT_DECODER_REGISTRY.register(name, fn, overwrite=overwrite)


def get_decoder(name: str) -> DecodeFunction:
    """Get decoder from the default registry."""
    return DEFAULT_DECODER_REGISTRY.get(name)


def list_decoders() -> List[str]:
    """List names in the default registry."""
    return DEFAULT_DECODER_REGISTRY.available()


def register_builtin_decoders() -> None:
    """Populate registry with built-in decoders."""
    from .abiss import decode_abiss
    from .segmentation import (
        decode_affinity_cc,
        decode_distance_watershed,
        decode_instance_binary_contour_distance,
    )
    from .synapse import polarity2instance

    register_decoder(
        "decode_instance_binary_contour_distance",
        decode_instance_binary_contour_distance,
        overwrite=True,
    )
    register_decoder("decode_affinity_cc", decode_affinity_cc, overwrite=True)
    register_decoder("decode_distance_watershed", decode_distance_watershed, overwrite=True)
    register_decoder("decode_abiss", decode_abiss, overwrite=True)
    register_decoder("polarity2instance", polarity2instance, overwrite=True)
