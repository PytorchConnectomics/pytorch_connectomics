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
class GraphNodeConfig:
    """Single node in a decoder graph."""

    enabled: bool = True
    name: str = ""
    op: str = ""
    inputs: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeGraphConfig:
    """Decoder graph configuration."""

    nodes: List[GraphNodeConfig] = field(default_factory=list)
    output: str = ""


@dataclass
class TuningParameterConfig:
    """Single tunable decoding/postprocessing parameter."""

    type: str = "float"
    range: List[Any] = field(default_factory=list)
    choices: List[Any] = field(default_factory=list)
    step: Optional[float] = None
    log: bool = False
    param_group: Optional[str] = None
    tuple_index: Optional[int] = None
    description: Optional[str] = None


@dataclass
class TuningFunctionSpaceConfig:
    """Search space for one decoding or postprocessing function."""

    enabled: bool = False
    function_name: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, TuningParameterConfig] = field(default_factory=dict)


@dataclass
class DecodingTuningParameterSpaceConfig:
    """Parameter spaces attached to a decoding pipeline."""

    decoding: TuningFunctionSpaceConfig = field(default_factory=TuningFunctionSpaceConfig)
    postprocessing: TuningFunctionSpaceConfig = field(default_factory=TuningFunctionSpaceConfig)


@dataclass
class DecodingTuningConfig:
    """Structured tuning metadata for decoded-output workflows."""

    enabled: bool = False
    parameter_space: DecodingTuningParameterSpaceConfig = field(
        default_factory=DecodingTuningParameterSpaceConfig
    )


@dataclass
class AffinityQCConfig:
    """Pre-decoding QC over the affinity prediction.

    When enabled, the decoding stage scans the prediction array, identifies
    outlier slabs along the last spatial axis (per-slice mean drift), and
    builds a (X, Y, Z) uint8 keep/drop mask consumed via ``affinity_mask_path``.
    """

    enabled: bool = False
    # ``post_save`` scans the in-memory prediction inside the decoding stage.
    # ``streaming`` accumulates per-Z stats inline during chunked inference
    # stitching (requires ``inference.strategy == "chunked"``); the mask is
    # built right after stitching and ``decoding.affinity_mask_path`` is wired
    # automatically, so the decoding stage's QC step becomes a no-op.
    mode: str = "post_save"
    # Source image (zarr or h5). Required for the XY-border + low-intensity
    # check; empty disables that check (z-cut detection still runs).
    image_path: str = ""
    # Output mask h5. If empty, defaults to ``<save_root>/affinity_mask.h5``
    # where ``save_root`` = ``decoding.save_path`` or ``inference.save_path``.
    mask_path: str = ""
    # Markdown report path. Empty → ``<save_root>/affinity_qc_report.md``.
    report_path: str = ""
    # Stride over Z for the coarse per-slice scan.
    z_stride: int = 10
    # How many sampled slices on each Z edge to summarize for drift detection.
    k_edge: int = 20
    # Stride-1 refinement window on each Z edge for low_z/high_z resolution.
    refine_window: int = 30
    # Slice-mean drift threshold below interior baseline to declare a slice bad.
    drift_thresh: float = 0.05
    # XY-border ring width for the intensity-cue check.
    border_width: int = 32
    # Image-intensity threshold marking background voxels for the border check.
    bg_thresh: int = 30
    # Sampled z-slices for the XY-border + intensity check.
    n_z_border: int = 8


@dataclass
class DecodingConfig:
    """Decoded-output orchestration configuration."""

    enabled: bool = True
    # Save the per-step output of each decoder in `steps`. Off by default;
    # only useful for debugging chained decoders. Elided when len(steps)==1
    # because the per-step output equals the final result.
    save_intermediate: bool = False
    # Save the final decoded (and optionally postprocessed) artifact.
    save_results: bool = True
    # Optional override for the decoded-artifact write directory. Defaults to
    # ``inference.save_path`` when unset.
    save_path: str = ""
    # Optional user-controlled filename suffix appended to decoded outputs.
    save_suffix: str = ""
    graph: Optional[DecodeGraphConfig] = None
    steps: List[DecodeModeConfig] = field(default_factory=list)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    # Optional explicit raw-prediction file (.h5). If set, pipeline loads
    # this file and proceeds to decoding (decode-only mode).
    load_prediction_path: str = ""
    affinity_mask_path: str = ""
    affinity_qc: AffinityQCConfig = field(default_factory=AffinityQCConfig)
    tuning: Optional[DecodingTuningConfig] = None

    def __post_init__(self) -> None:
        if self.graph is not None and self.steps:
            raise ValueError("decoding.graph cannot be set together with non-empty decoding.steps.")
