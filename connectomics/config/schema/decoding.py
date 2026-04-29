from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConnectedComponentsConfig:
    """Connected components filtering configuration."""

    enabled: bool = False
    top_k: Optional[int] = None
    min_size: int = 0
    connectivity: int = 1


@dataclass
class BinaryPostprocessingConfig:
    """Binary postprocessing pipeline configuration."""

    enabled: bool = False
    median_filter_size: Optional[Tuple[int, ...]] = None
    opening_iterations: int = 0
    closing_iterations: int = 0
    connected_components: Optional[ConnectedComponentsConfig] = None


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration for decoded outputs."""

    enabled: bool = False
    binary: Optional[Dict[str, Any]] = field(default_factory=dict)
    instance_cc3d: Optional[Dict[str, Any]] = None
    output_transpose: List[int] = field(default_factory=list)


@dataclass
class DecodeBinaryContourDistanceWatershedConfig:
    """Parameters for binary+contour+distance watershed decoding."""

    binary_threshold: Tuple[float, float] = (0.9, 0.8)
    contour_threshold: Tuple[float, float] = (0.5, 0.5)
    distance_threshold: Tuple[float, float] = (0.5, -0.5)
    min_instance_size: int = 10
    min_seed_size: int = 10
    seed_distance_scale: float = 1.0


@dataclass
class DecodeModeConfig:
    """Single decode mode configuration."""

    enabled: bool = True
    name: str = "decode_semantic"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingConfig:
    """Decoded-output orchestration configuration."""

    steps: List[DecodeModeConfig] = field(default_factory=list)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output_path: str = ""
    input_prediction_path: str = ""
    tuning: Optional[Dict[str, Any]] = None
