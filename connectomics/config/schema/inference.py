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
    keep_input_on_cpu: bool = False  # Move full volume to CPU between sliding-window batches
    lazy_load: bool = False  # Stream ROIs from disk instead of materializing the full volume
    sw_device: Optional[str] = None
    output_device: Optional[str] = None


@dataclass
class ChunkStitchingConfig:
    """Boundary stitching configuration for chunked decoded outputs."""

    method: str = "affinity_cc_boundary_union"
    threshold: Optional[float] = None
    edge_offset: Optional[int] = None
    min_contact: int = 1


@dataclass
class ChunkingConfig:
    """Chunked inference+decoding for volumes too large to materialize at once.

    Chunking is an inference execution strategy: data sections still describe
    whole volumes, while this section controls how each volume is partitioned
    and stitched.
    """

    enabled: bool = False
    chunk_size: Optional[List[int]] = None  # ZYX after test-time val_transpose.
    halo: List[int] = field(default_factory=lambda: [0, 0, 0])
    axes: str = "all"  # "all" or "z"; "z" keeps full YX in each chunk.
    temp_dir: str = ""
    save_intermediate: bool = False
    stitching: ChunkStitchingConfig = field(default_factory=ChunkStitchingConfig)


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
    flip_axes: Any = "all"  # "all" | "none" | list[int] (0-indexed spatial: 0=z, 1=y, 2=x)
    flip_combinations: Optional[List[List[int]]] = None  # explicit list of axis subsets
    rotation90_axes: Any = None  # "all" | None | [[int, int], ...] spatial plane pairs
    rotate90_k: Optional[List[int]] = None  # subset of quarter-turns, defaults to [0,1,2,3]
    patch_first_local: bool = True  # slide once, apply local TTA inside each ROI batch
    apply_mask: bool = True
    transforms: Optional[List[Dict[str, Any]]] = None  # advanced explicit transforms

    # Aggregation — single mode or per-channel list.
    # Single string: "mean", "min", "max" applied to all channels.
    # Per-channel list: [["0:3", "min"], ["3:", "mean"]] where each entry is
    # [channel_selector, mode].  Channel selectors use the same syntax as
    # loss pred_slice / target_slice (e.g. "0:3", "-1:", "3").
    ensemble_mode: Any = "mean"
    # CUDA cache cleanup cadence during TTA loop (0 disables cleanup).
    empty_cache_interval: int = 4


@dataclass
class SavePredictionConfig:
    """Prediction saving configuration."""

    enabled: bool = False
    output_formats: List[str] = field(default_factory=lambda: ["h5"])  # Any of: h5, tiff, png
    output_path: Optional[str] = None
    cache_suffix: str = "_x1_prediction.h5"
    save_all_heads: bool = False

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

    enabled: bool = True
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

    # Instance cc3d relabeling: split disconnected components and remove small ones
    instance_cc3d: Optional[Dict[str, Any]] = None
    # Example: {connectivity: 6, min_size: 100, remove_boundary: false}

    # Axis permutation
    output_transpose: List[int] = field(
        default_factory=list
    )  # Axis permutation for output (e.g., [2,1,0] for zyx->xyz)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    enabled: bool = False  # Auto-enabled when evaluation keys are provided in YAML
    metrics: Optional[List[str]] = None  # e.g., ['dice', 'jaccard', 'accuracy']
    prediction_threshold: float = 0.5  # Probability/logit threshold for binary metrics
    instance_iou_threshold: float = 0.5  # IoU threshold for instance matching metrics
    # Neurite ERL evaluation via lib/em_erl. nerl_graph accepts an ERLGraph
    # .npz or a BANIS/NISB-style NetworkX skeleton.pkl.
    nerl_graph: Any = None
    nerl_mask: Any = None
    nerl_resolution: Optional[List[float]] = None
    nerl_merge_threshold: int = 1
    nerl_chunk_num: int = 1
    nerl_skeleton_id_attribute: str = "id"
    nerl_skeleton_position_attribute: str = "index_position"
    nerl_skeleton_edge_length_attribute: str = "edge_length"
    nerl_skeleton_position_order: str = "xyz"
    nerl_prediction_position_order: Optional[str] = None


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

    # Named output head selection for multi-head models. When unset, falls back
    # to model.primary_head or the sole configured head.
    head: Optional[str] = None
    # Optional channel selector applied after channel activations and before
    # decoding/saving. Accepts None, int, slice string, or explicit index list.
    select_channel: Optional[Any] = None
    strategy: str = "whole_volume"  # "whole_volume" or "chunked"
    # Crop context padding before decoding/saving predictions:
    # [D,H,W]/[H,W] symmetric or [Db,Da,Hb,Ha,Wb,Wa] asymmetric.
    crop_pad: Optional[List[int]] = None
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    test_time_augmentation: TestTimeAugmentationConfig = field(
        default_factory=TestTimeAugmentationConfig
    )
    # Optional explicit intermediate TTA prediction file (.h5). If set in test
    # mode, pipeline loads this file directly and proceeds to decoding.
    tta_result_path: str = ""
    # Path to pre-computed affinity prediction HDF5 (dataset "main").
    # When set, skips model inference — loads and decodes directly.
    saved_prediction_path: str = ""
    # Path to save decoded instance segmentation (separate from raw prediction).
    decoding_path: str = ""
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
