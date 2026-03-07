"""Decoding package for PyTorch Connectomics.

Subpackages:
    decoders/    - Segmentation decoder implementations
    postprocess/ - Post-processing utilities for segmentation refinement
    tuning/      - Hyperparameter tuning for decoding parameters
"""

# --- Framework / Infrastructure ---
from .base import DecodeStep
from .pipeline import (
    apply_decode_mode,
    apply_decode_pipeline,
    normalize_decode_modes,
    resolve_decode_modes_from_cfg,
)
from .registry import (
    DecoderRegistry,
    get_decoder,
    list_decoders,
    register_builtin_decoders,
    register_decoder,
)

# --- Segmentation Decoders ---
from .decoders import (
    decode_abiss,
    decode_affinity_cc,
    decode_distance_watershed,
    decode_instance_binary_contour_distance,
    polarity2instance,
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
from .utils import (
    cast2dtype,
    merge_small_objects,
    remove_large_instances,
    remove_small_instances,
)

# --- Hyperparameter Tuning ---
from .tuning import (
    OptunaDecodingTuner,
    SkeletonMetrics,
    grid_search_threshold,
    load_and_apply_best_params,
    optimize_parameters,
    optimize_threshold,
    run_tuning,
)

register_builtin_decoders()

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
    # Segmentation decoding
    "decode_instance_binary_contour_distance",
    "decode_affinity_cc",
    "decode_distance_watershed",
    "decode_abiss",
    # Synapse decoding
    "polarity2instance",
    # Auto-tuning
    "optimize_threshold",
    "optimize_parameters",
    "grid_search_threshold",
    "SkeletonMetrics",
    "OptunaDecodingTuner",
    "run_tuning",
    "load_and_apply_best_params",
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
