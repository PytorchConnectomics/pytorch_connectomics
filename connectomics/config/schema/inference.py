from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .system import SystemConfig


@dataclass
class InferenceDataConfig:
    """Inference data override configuration.

    Optional dataset/data loader overrides for inference jobs.
    """

    input_path: Optional[str] = None
    output_path: Optional[str] = None
    resolution: Optional[List[float]] = None


@dataclass
class SlidingWindowConfig:
    """Sliding window inference configuration."""

    enabled: bool = False
    window_size: Optional[List[int]] = None  # None = fallback to data.dataloader.patch_size
    sw_batch_size: Optional[int] = None  # If None, falls back to data.dataloader.batch_size
    overlap: float = 0.5
    blending: str = "gaussian"  # "constant" or "gaussian"
    sigma_scale: float = 0.125
    padding_mode: str = "reflect"
    cval: float = 0.0
    keep_input_on_cpu: bool = False  # Legacy option; ignored when unsupported
    pad_size: Optional[List[int]] = None  # Legacy alias for explicit padding before inference
    sw_device: Optional[str] = None
    output_device: Optional[str] = None


@dataclass
class TestTimeAugmentationConfig:
    """Test-time augmentation (TTA) configuration."""

    enabled: bool = False
    distributed_sharding: bool = True  # Split TTA variants across DDP ranks for one-volume tests
    distributed_reduce_chunk_mb: int = 128  # Chunk size for rank reduction of large volumes
    # Optional channel-wise activation overrides applied before aggregation.
    # Each entry is a mapping:
    #   {channels: "start:end" | int | [int, ...], activation: "sigmoid"|"softmax"|"tanh"|"none"}
    # Channel selectors follow Python/NumPy indexing rules:
    #   -1 means the last channel, ":" means all channels, ":-1" excludes the last.
    channel_activations: List[Dict[str, Any]] = field(default_factory=list)
    # Optional channel selector applied after channel activations and before TTA transforms.
    # Accepts: None (all channels), int, slice string like ":"/"0:3"/"-1:", or [int, ...].
    select_channel: Optional[Any] = None

    # Legacy/simple controls
    flip_axes: Any = "all"  # "all" | "none" | list[int]
    flip_combinations: Optional[List[List[int]]] = None  # explicit list of axis subsets
    rotate90_axes: Optional[List[List[int]]] = None  # e.g. [[3,4]] for H/W in NCDHW
    rotation90_axes: Optional[List[List[int]]] = None  # Legacy alias for rotate90_axes
    rotate90_k: Optional[List[int]] = None  # e.g. [1,2,3]
    apply_mask: bool = True
    transforms: Optional[List[Dict[str, Any]]] = None  # advanced explicit transforms

    # Aggregation
    ensemble_mode: str = "mean"  # "mean", "max", "gmean"
    # CUDA cache cleanup cadence during TTA loop (0 disables cleanup).
    empty_cache_interval: int = 4


@dataclass
class SavePredictionConfig:
    """Prediction saving configuration."""

    enabled: bool = False
    output_formats: List[str] = field(default_factory=lambda: ["h5"])  # Any of: h5, tiff, png
    output_path: Optional[str] = None
    cache_suffix: str = "_prediction.h5"

    # Data scaling and output typing
    # -1 keeps native float probabilities/logits; >0 scales and casts to integer dtype if chosen.
    intensity_scale: float = -1.0
    intensity_dtype: str = "float32"  # float32/float16/uint8/uint16/int16/int32

    # File writing behavior
    compression: Optional[str] = "gzip"


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

    name: str = "decode_semantic"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BinaryPostprocessingConfig:
    """Binary postprocessing pipeline configuration."""

    enabled: bool = False  # Enable binary postprocessing pipeline
    median_filter_size: Optional[Tuple[int, ...]] = (
        None  # Median filter kernel size (e.g., (3, 3) for 2D)
    )
    opening_iterations: int = 0  # Number of morphological opening iterations
    closing_iterations: int = 0  # Number of morphological closing iterations
    connected_components: Optional[ConnectedComponentsConfig] = None  # CC filtering config


@dataclass
class ConnectedComponentsConfig:
    """Connected components filtering configuration."""

    enabled: bool = False  # Enable connected components filtering
    top_k: Optional[int] = None  # Keep only top-k largest components (None = keep all)
    min_size: int = 0  # Minimum component size in voxels
    connectivity: int = 1  # Connectivity for CC (1=face, 2=face+edge, 3=face+edge+corner)


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration for inference output.

    Controls how predictions are transformed after saving:
    - Binary refinement: Morphological operations and connected components filtering
    - Transpose: Reorder axes (e.g., [2,1,0] for zyx->xyz)

    Note: Intensity scaling and dtype conversion are handled by SavePredictionConfig.
    """

    enabled: bool = False  # Enable postprocessing pipeline

    # Binary segmentation refinement (morphological ops, connected components)
    binary: Optional[Dict[str, Any]] = field(
        default_factory=dict
    )  # Binary postprocessing config (e.g., {'opening_iterations': 2})

    # Axis permutation
    output_transpose: List[int] = field(
        default_factory=list
    )  # Axis permutation for output (e.g., [2,1,0] for zyx->xyz)
    crop_pad: Optional[List[int]] = None  # Crop [D,H,W]/[H,W] sym or [Db,Da,Hb,Ha,Wb,Wa] asym.


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    enabled: bool = False  # Auto-enabled when evaluation keys are provided in YAML
    metrics: Optional[List[str]] = None  # e.g., ['dice', 'jaccard', 'accuracy']
    prediction_threshold: float = 0.5  # Probability/logit threshold for binary metrics
    instance_iou_threshold: float = 0.5  # IoU threshold for instance matching metrics


@dataclass
class InferenceConfig:
    """Inference configuration.

    Shared inference settings for sliding window and test-time augmentation.
    Data paths are resolved in merged runtime `cfg.data`.

    Key Features:
    - Sliding window inference for large volumes
    - Test-time augmentation (TTA) support
    - Saving intermediate predictions
    - Multiple decoding strategies
    - Postprocessing and evaluation
    - Runtime resource overrides for inference

    Note: stage-specific overrides are merged before runtime; consumers should read `cfg.inference`.
    """

    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    test_time_augmentation: TestTimeAugmentationConfig = field(
        default_factory=TestTimeAugmentationConfig
    )
    # Optional explicit intermediate TTA prediction file (.h5). If set in test
    # mode, pipeline loads this file directly and proceeds to decoding.
    tta_result_path: str = ""
    save_prediction: SavePredictionConfig = field(default_factory=SavePredictionConfig)
    decoding: Optional[List[DecodeModeConfig]] = None  # List of decode modes to apply sequentially
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # If True, switch model to eval() during test/predict. Set False to keep
    # train() mode (useful for BatchNorm recalibration or MC-Dropout workflows).
    do_eval: bool = True

    # Inference-specific runtime overrides (applied in test/tune modes)
    # `system` overrides selected keys on top-level cfg.system during stage resolution.
    system: SystemConfig = field(default_factory=lambda: SystemConfig(num_gpus=-1, num_workers=-1))
    batch_size: int = -1  # Overrides data.dataloader.batch_size if >= 0
