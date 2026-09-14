from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TubeEvaluationConfig:
    """Thresholds for ground-truth-free tube analysis."""

    substantial_min_z_slices: int = 21
    substantial_min_voxels: int = 10000
    long_span_fraction: float = 0.25
    decent_min_voxels: int = 20000
    border_margin: int = 2
    border_patch_min_voxels: int = 10
    multi_component_min_voxels: int = 50
    multi_component_slice_step: int = 3
    parallel_min_slices: int = 15
    parallel_fraction_threshold: float = 0.30
    disconnected_component_min_voxels: int = 1000
    bump_min_slices: int = 40
    bump_relative_excess: float = 0.20
    bump_absolute_excess: int = 200
    bump_max_slices: int = 30
    bump_median_window: int = 31
    top_incomplete: int = 8


@dataclass
class MorphologyEvaluationConfig:
    """Physical morphology estimates without ground truth, in micrometers."""

    voxel_size_um: Optional[List[float]] = None  # Otherwise data.test.resolution / 1000.
    min_voxels: int = 1000
    min_axon_length_um: float = 2.0
    max_axon_radius_um: float = 0.35
    min_elongation: float = 3.0
    border_margin_um: float = 0.08
    component_min_voxels: int = 100
    multi_component_min_area_um2: float = 0.005
    multi_component_fraction: float = 0.2
    profile_smoothing_um: float = 0.2
    radius_bins_um: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.35, 0.5, 1.0])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    enabled: bool = False
    metrics: Optional[List[str]] = None
    prediction_threshold: float = 0.5
    instance_iou_threshold: float = 0.5
    nerl_resolution: Optional[List[float]] = None
    nerl_merge_threshold: int = 50
    nerl_chunk_num: int = 1
    nerl_num_workers: int = 1
    nerl_skeleton_id_attribute: str = "id"
    nerl_skeleton_position_attribute: str = "index_position"
    nerl_skeleton_edge_length_attribute: str = "edge_length"
    nerl_skeleton_position_order: str = "xyz"
    nerl_prediction_position_order: Optional[str] = None
    tube: TubeEvaluationConfig = field(default_factory=TubeEvaluationConfig)
    morphology: MorphologyEvaluationConfig = field(default_factory=MorphologyEvaluationConfig)
