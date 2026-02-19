"""Decoding package for PyTorch Connectomics."""

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

from .auto_tuning import (
    SkeletonMetrics,
    grid_search_threshold,
    optimize_parameters,
    optimize_threshold,
)
from .optuna_tuner import (
    OptunaDecodingTuner,
    load_and_apply_best_params,
    run_tuning,
)
from .postprocess import (
    add_masks,
    binarize_and_median,
    intersection_over_union,
    merge_masks,
    remove_masks,
    stitch_3d,
    watershed_split,
)
from .segmentation import (
    decode_affinity_cc,
    decode_instance_binary_contour_distance,
    decode_distance_watershed,
)
from .synapse import (
    polarity2instance,
)
from .utils import (
    cast2dtype,
    merge_small_objects,
    remove_large_instances,
    remove_small_instances,
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
    # Auto-tuning
    "optimize_threshold",
    "optimize_parameters",
    "grid_search_threshold",
    "SkeletonMetrics",
    "OptunaDecodingTuner",
    "run_tuning",
    "load_and_apply_best_params",
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
    # Utilities
    "cast2dtype",
    "remove_small_instances",
    "remove_large_instances",
    "merge_small_objects",
]
