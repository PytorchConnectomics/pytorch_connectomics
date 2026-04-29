"""Decoding package for PyTorch Connectomics.

Subpackages:
    decoders/    - Segmentation decoder implementations
    postprocess.py - Post-processing utilities for segmentation refinement
    tuning/      - Hyperparameter tuning for decoding parameters
"""

from importlib import import_module

# --- Framework / Infrastructure ---
from .base import DecodeStep
from .pipeline import (
    apply_decode_mode,
    apply_decode_pipeline,
    normalize_decode_modes,
    resolve_decode_modes_from_cfg,
)

# --- Post-processing & Utilities ---
from .postprocess import (
    add_masks,
    apply_binary_postprocessing,
    binarize_and_median,
    intersection_over_union,
    merge_masks,
    remove_masks,
    stitch_3d,
    watershed_split,
)
from .registry import (
    DecoderRegistry,
    get_decoder,
    list_decoders,
    register_builtin_decoders,
    register_decoder,
)
from .stage import (
    DecodingStageResult,
    apply_decoding_postprocessing,
    run_decoding_stage,
)
from .utils import (
    cast2dtype,
    merge_small_objects,
    remove_large_instances,
    remove_small_instances,
)

_LAZY_DECODER_EXPORTS = {
    "branch_merge": "connectomics.decoding.decoders.branch_merge",
    "decode_abiss": "connectomics.decoding.decoders.abiss",
    "decode_affinity_cc": "connectomics.decoding.decoders.segmentation",
    "decode_distance_watershed": "connectomics.decoding.decoders.segmentation",
    "decode_instance_binary_contour_distance": "connectomics.decoding.decoders.segmentation",
    "decode_waterz": "connectomics.decoding.decoders.waterz",
    "polarity2instance": "connectomics.decoding.decoders.synapse",
    "run_chunked_affinity_cc_inference": "connectomics.decoding.streamed_chunked",
}


def __getattr__(name: str):
    module_name = _LAZY_DECODER_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    # Registry / pipeline
    "DecodeStep",
    "DecoderRegistry",
    "register_decoder",
    "get_decoder",
    "list_decoders",
    "normalize_decode_modes",
    "apply_decode_pipeline",
    "resolve_decode_modes_from_cfg",
    "apply_decode_mode",
    "DecodingStageResult",
    "apply_decoding_postprocessing",
    "run_decoding_stage",
    # Segmentation decoding
    "decode_instance_binary_contour_distance",
    "decode_affinity_cc",
    "decode_distance_watershed",
    "decode_waterz",
    "run_chunked_affinity_cc_inference",
    "branch_merge",
    "decode_abiss",
    # Synapse decoding
    "polarity2instance",
    # Post-processing
    "binarize_and_median",
    "remove_masks",
    "add_masks",
    "merge_masks",
    "watershed_split",
    "stitch_3d",
    "intersection_over_union",
    "apply_binary_postprocessing",
    # Utilities
    "cast2dtype",
    "remove_small_instances",
    "remove_large_instances",
    "merge_small_objects",
]
