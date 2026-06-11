"""Decoder registry for configurable decode pipelines."""

from __future__ import annotations

from functools import wraps
from typing import Dict, List

from .base import DecodeFunction, GraphOp


def as_graph_op(fn: DecodeFunction) -> GraphOp:
    """Wrap a unary decoder so it satisfies the graph-op call contract."""

    @wraps(fn)
    def _wrapped(inputs, **kwargs):
        if len(inputs) != 1:
            raise ValueError(
                f"Unary decoder '{getattr(fn, '__name__', type(fn).__name__)}' "
                f"expects exactly one input, got {len(inputs)}."
            )
        return fn(inputs[0], **kwargs)

    return _wrapped


class DecoderRegistry:
    """Name -> decoder function registry."""

    def __init__(self) -> None:
        self._decoders: Dict[str, GraphOp] = {}

    def register(self, name: str, fn: DecodeFunction, *, overwrite: bool = False) -> None:
        """Register a decoder function."""
        self.register_graph_op(name, as_graph_op(fn), overwrite=overwrite)

    def register_graph_op(self, name: str, fn: GraphOp, *, overwrite: bool = False) -> None:
        """Register a native graph operation without unary wrapping."""
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

    def get(self, name: str) -> GraphOp:
        """Get decoder function by name."""
        try:
            return self._decoders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._decoders))
            raise KeyError(f"Unknown decoder '{name}'. Available decoders: [{available}]") from exc

    def available(self) -> List[str]:
        """Return registered decoder names."""
        return sorted(self._decoders.keys())


DEFAULT_DECODER_REGISTRY = DecoderRegistry()
_BUILTINS_REGISTERED = False


def register_decoder(name: str, fn: DecodeFunction, *, overwrite: bool = False) -> None:
    """Register decoder in the default registry."""
    DEFAULT_DECODER_REGISTRY.register(name, fn, overwrite=overwrite)


def register_graph_op(name: str, fn: GraphOp, *, overwrite: bool = False) -> None:
    """Register a native graph operation in the default registry."""
    DEFAULT_DECODER_REGISTRY.register_graph_op(name, fn, overwrite=overwrite)


def get_decoder(name: str) -> GraphOp:
    """Get decoder from the default registry."""
    ensure_builtin_decoders_registered()
    return DEFAULT_DECODER_REGISTRY.get(name)


def list_decoders() -> List[str]:
    """List names in the default registry."""
    ensure_builtin_decoders_registered()
    return DEFAULT_DECODER_REGISTRY.available()


def ensure_builtin_decoders_registered() -> None:
    """Register built-in decoders on first use."""
    register_builtin_decoders()


def register_builtin_decoders() -> None:
    """Populate registry with built-in decoders."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from .decoders.abiss import decode_abiss
    from .decoders.branch_merge import branch_merge
    from .decoders.branch_split import branch_split
    from .decoders.combine import combine_split
    from .decoders.longrange_guided_split import longrange_guided_split
    from .decoders.segmentation import (
        decode_affinity_cc,
        decode_distance_watershed,
        decode_instance_binary_contour_distance,
    )
    from .decoders.segmentation_grow import segmentation_grow
    from .decoders.synapse import polarity2instance
    from .decoders.transforms import channel_gate
    from .decoders.waterz import decode_waterz

    register_decoder("channel_gate", channel_gate, overwrite=True)
    register_decoder(
        "decode_instance_binary_contour_distance",
        decode_instance_binary_contour_distance,
        overwrite=True,
    )
    register_decoder(
        "decode_binary_contour_distance_watershed",
        decode_instance_binary_contour_distance,
        overwrite=True,
    )
    register_decoder("decode_affinity_cc", decode_affinity_cc, overwrite=True)
    register_decoder("decode_distance_watershed", decode_distance_watershed, overwrite=True)
    register_decoder("decode_waterz", decode_waterz, overwrite=True)
    register_decoder("branch_merge", branch_merge, overwrite=True)
    register_decoder("branch_split", branch_split, overwrite=True)
    register_decoder("longrange_guided_split", longrange_guided_split, overwrite=True)
    register_decoder("segmentation_grow", segmentation_grow, overwrite=True)
    register_decoder("decode_abiss", decode_abiss, overwrite=True)
    register_decoder("polarity2instance", polarity2instance, overwrite=True)
    register_graph_op("combine_split", combine_split, overwrite=True)
    _BUILTINS_REGISTERED = True
