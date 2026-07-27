"""Evaluation-stage adapter for ground-truth-free tube analysis."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from ..metrics.tube import (
    TubeAnalysis,
    TubeAnalysisConfig,
    analyze_tubes,
    format_tube_analysis,
)
from .context import EvaluationContext

logger = logging.getLogger(__name__)


def _tube_config(context: EvaluationContext) -> tuple[TubeAnalysisConfig, int]:
    cfg = context.cfg_value(context.evaluation_cfg, "tube", None)
    defaults = TubeAnalysisConfig()
    config = TubeAnalysisConfig(
        substantial_min_z_slices=int(
            context.cfg_value(
                cfg,
                "substantial_min_z_slices",
                defaults.substantial_min_z_slices,
            )
        ),
        substantial_min_voxels=int(
            context.cfg_value(
                cfg,
                "substantial_min_voxels",
                defaults.substantial_min_voxels,
            )
        ),
        long_span_fraction=float(
            context.cfg_value(cfg, "long_span_fraction", defaults.long_span_fraction)
        ),
        decent_min_voxels=int(
            context.cfg_value(cfg, "decent_min_voxels", defaults.decent_min_voxels)
        ),
        border_margin=int(context.cfg_value(cfg, "border_margin", defaults.border_margin)),
        border_patch_min_voxels=int(
            context.cfg_value(
                cfg,
                "border_patch_min_voxels",
                defaults.border_patch_min_voxels,
            )
        ),
        multi_component_min_voxels=int(
            context.cfg_value(
                cfg,
                "multi_component_min_voxels",
                defaults.multi_component_min_voxels,
            )
        ),
        multi_component_slice_step=int(
            context.cfg_value(
                cfg,
                "multi_component_slice_step",
                defaults.multi_component_slice_step,
            )
        ),
        parallel_min_slices=int(
            context.cfg_value(cfg, "parallel_min_slices", defaults.parallel_min_slices)
        ),
        parallel_fraction_threshold=float(
            context.cfg_value(
                cfg,
                "parallel_fraction_threshold",
                defaults.parallel_fraction_threshold,
            )
        ),
        disconnected_component_min_voxels=int(
            context.cfg_value(
                cfg,
                "disconnected_component_min_voxels",
                defaults.disconnected_component_min_voxels,
            )
        ),
        bump_min_slices=int(context.cfg_value(cfg, "bump_min_slices", defaults.bump_min_slices)),
        bump_relative_excess=float(
            context.cfg_value(
                cfg,
                "bump_relative_excess",
                defaults.bump_relative_excess,
            )
        ),
        bump_absolute_excess=int(
            context.cfg_value(
                cfg,
                "bump_absolute_excess",
                defaults.bump_absolute_excess,
            )
        ),
        bump_max_slices=int(context.cfg_value(cfg, "bump_max_slices", defaults.bump_max_slices)),
        bump_median_window=int(
            context.cfg_value(cfg, "bump_median_window", defaults.bump_median_window)
        ),
    )
    return config, int(context.cfg_value(cfg, "top_incomplete", 8))


def _as_3d_segmentation(decoded_predictions: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(decoded_predictions)
    while segmentation.ndim > 3 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    if segmentation.ndim != 3:
        raise ValueError(
            "Tube evaluation requires one 3D instance segmentation, "
            f"got shape {segmentation.shape}"
        )
    return segmentation


def compute_tube_metrics(
    context: EvaluationContext,
    decoded_predictions: np.ndarray,
    volume_prefix: str,
    metrics_dict: dict[str, Any],
) -> TubeAnalysis:
    """Compute and append GT-free tube metrics for one decoded volume."""

    config, top_incomplete = _tube_config(context)
    analysis = analyze_tubes(_as_3d_segmentation(decoded_predictions), config)
    for name, value in asdict(analysis.summary).items():
        metrics_dict[f"tube_{name}"] = value
    metrics_dict["tube_volume_shape"] = analysis.volume_shape
    metrics_dict["tube_report"] = format_tube_analysis(
        analysis,
        top_incomplete=top_incomplete,
    )
    # The report writer persists the detailed records as a compressed NPZ.
    metrics_dict["_tube_analysis"] = analysis
    logger.info("%s%s", volume_prefix, metrics_dict["tube_report"])
    return analysis


__all__ = ["compute_tube_metrics"]
