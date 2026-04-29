"""Inference utilities package."""

from .artifact import (
    PredictionArtifactMetadata,
    read_prediction_artifact,
    write_prediction_artifact,
    write_prediction_artifact_attrs,
)
from .chunked import (
    is_chunked_inference_enabled,
    run_chunked_affinity_cc_inference,
    run_chunked_prediction_inference,
)
from .manager import InferenceManager
from .output import (
    apply_prediction_transform,
    apply_storage_dtype_transform,
    resolve_output_filenames,
    write_outputs,
)
from .sliding import (
    build_sliding_inferer,
    is_2d_inference_mode,
    resolve_inferer_overlap,
    resolve_inferer_roi_size,
)
from .stage import run_prediction_inference
from .tta import TTAPredictor

__all__ = [
    "InferenceManager",
    "PredictionArtifactMetadata",
    "read_prediction_artifact",
    "write_prediction_artifact",
    "write_prediction_artifact_attrs",
    "run_prediction_inference",
    "is_chunked_inference_enabled",
    "run_chunked_affinity_cc_inference",
    "run_chunked_prediction_inference",
    "apply_prediction_transform",
    "apply_storage_dtype_transform",
    "resolve_output_filenames",
    "write_outputs",
    "build_sliding_inferer",
    "resolve_inferer_roi_size",
    "resolve_inferer_overlap",
    "is_2d_inference_mode",
    "TTAPredictor",
]
