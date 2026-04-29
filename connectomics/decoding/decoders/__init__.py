"""Segmentation decoder implementations."""

from importlib import import_module

_LAZY_DECODERS = {
    "branch_merge": "connectomics.decoding.decoders.branch_merge",
    "decode_abiss": "connectomics.decoding.decoders.abiss",
    "decode_affinity_cc": "connectomics.decoding.decoders.segmentation",
    "decode_distance_watershed": "connectomics.decoding.decoders.segmentation",
    "decode_instance_binary_contour_distance": "connectomics.decoding.decoders.segmentation",
    "decode_waterz": "connectomics.decoding.decoders.waterz",
    "polarity2instance": "connectomics.decoding.decoders.synapse",
}


def __getattr__(name: str):
    module_name = _LAZY_DECODERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "decode_instance_binary_contour_distance",
    "decode_distance_watershed",
    "decode_affinity_cc",
    "decode_waterz",
    "branch_merge",
    "polarity2instance",
    "decode_abiss",
]
