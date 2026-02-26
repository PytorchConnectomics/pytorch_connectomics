"""
Modern Hydra-based configuration system for PyTorch Connectomics.

This module provides a comprehensive, type-safe configuration system using dataclasses
and OmegaConf for PyTorch Connectomics. It supports:

- Type-safe configuration with dataclasses
- Composable configurations for different experiments
- Integration with PyTorch Lightning
- Support for multiple model architectures (UNet, MedNeXt, RSUNet, etc.)
- Comprehensive data loading and augmentation options
- Advanced training and inference configurations

The configuration system is organized into logical sections:
- System: Hardware and parallelization settings
- Model: Architecture and loss function configurations
- Data: Dataset loading and transformation settings
- Optimization: Training parameters and schedulers
- Monitor: Logging, checkpointing, and early stopping
- Inference: Test-time augmentation and postprocessing

Example usage:
    from connectomics.config.hydra_config import Config
    from connectomics.config.hydra_utils import load_config

    # Load configuration from YAML
    cfg = load_config("tutorials/mito_lucchi++.yaml")

    # Access configuration sections
    print(f"Model architecture: {cfg.model.architecture}")
    print(f"Batch size: {cfg.system.training.batch_size}")

    # Configure edge detection for instance segmentation
    cfg.data.label_transform.edge_mode.mode = "seg-all"  # Instance boundaries only
    cfg.data.label_transform.edge_mode.thickness = 5     # 5-pixel boundary thickness
    cfg.data.label_transform.edge_mode.processing_mode = "3d"  # Full 3D processing
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Note: MISSING can be imported from omegaconf if needed for required fields


@dataclass
class SystemTrainingConfig:
    """System configuration for training mode.

    Controls hardware resources and parallelization settings during training.
    These settings affect data loading, model training, and memory usage.

    Attributes:
        num_gpus: Number of GPUs to use for training (0 for CPU-only)
        num_workers: Number of parallel data loading workers
        batch_size: Training batch size (per GPU)
    """

    num_gpus: int = 1
    num_workers: int = 8
    batch_size: int = 4


@dataclass
class SystemInferenceConfig:
    """System configuration for inference mode.

    Controls hardware resources and parallelization settings during inference.
    Typically uses fewer resources than training to optimize for memory efficiency.

    Attributes:
        num_gpus: Number of GPUs to use for inference (0 for CPU-only)
        num_workers: Number of parallel data loading workers
        batch_size: Inference batch size (usually 1 for large volumes)
    """

    num_gpus: int = 1
    num_workers: int = 1
    batch_size: int = 1


@dataclass
class SystemConfig:
    """System configuration for hardware and parallelization.

    Central configuration for all system-level settings including hardware resources,
    parallelization, reproducibility, and automatic hyperparameter planning.

    Attributes:
        training: Training-specific system settings
        inference: Inference-specific system settings
        seed: Random seed for reproducibility (None for random)
        auto_plan: Enable automatic hyperparameter planning based on available GPU
        print_auto_plan: Print auto-planning results to console
    """

    training: SystemTrainingConfig = field(default_factory=SystemTrainingConfig)
    inference: SystemInferenceConfig = field(default_factory=SystemInferenceConfig)
    seed: Optional[int] = None

    # Auto-planning
    auto_plan: bool = False  # Enable automatic hyperparameter planning based on GPU
    print_auto_plan: bool = True  # Print auto-planning results


@dataclass
class LossBalancingConfig:
    """Configuration for adaptive loss weighting."""

    strategy: Optional[str] = None  # None, "uncertainty", or "gradnorm"
    gradnorm_alpha: float = 0.5
    gradnorm_lambda: float = 1.0
    gradnorm_parameter_strategy: str = "last"  # "first", "last", or "all"


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Comprehensive configuration for neural network architectures, loss functions,
    and multi-task learning setups. Supports multiple architectures including
    UNet variants, MedNeXt, RSUNet, and transformer-based models.

    Key Features:
    - Multiple architecture support (UNet, MedNeXt, RSUNet, UNETR, etc.)
    - Configurable loss functions with weighting
    - Multi-task learning capabilities
    - Architecture-specific parameter tuning
    - Deep supervision support
    """

    # Architecture
    architecture: str = "monai_basic_unet3d"

    # I/O dimensions
    input_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    output_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 1
    out_channels: int = 1

    # Architecture-specific parameters
    filters: Tuple[int, ...] = (32, 64, 128, 256, 512)
    dropout: float = 0.0
    norm: str = "batch"
    num_groups: int = 8  # Number of groups for GroupNorm
    activation: str = "relu"

    # UNet-specific parameters (MONAI UNet)
    spatial_dims: int = 3  # Spatial dimensions: 2 for 2D, 3 for 3D
    # (auto-inferred from input_size length, not used directly)
    num_res_units: int = 2  # Number of residual units per block
    kernel_size: int = 3  # Convolution kernel size
    strides: Optional[List[int]] = None  # Downsampling strides (e.g., [2, 2, 2, 2] for 4 levels)
    act: str = "relu"  # Activation function: 'relu', 'prelu', 'elu', etc.
    upsample: str = (
        "deconv"  # Upsampling mode: 'deconv' (transposed conv),
        # 'nontrainable' (interpolation + conv), or 'pixelshuffle'
    )

    # Transformer-specific (UNETR, etc.)
    feature_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12

    # MedNeXt-specific parameters
    # For 'mednext' architecture (predefined sizes)
    mednext_size: str = "S"  # S, B, M, or L

    # For 'mednext_custom' architecture (full control)
    mednext_base_channels: int = 32
    mednext_exp_r: Any = 4  # Expansion ratio: int or list of 9 ints (e.g., [2,3,4,4,4,4,4,3,2])
    mednext_kernel_size: int = 3  # 3, 5, or 7
    mednext_do_res: bool = True  # Residual connections in blocks
    mednext_do_res_up_down: bool = True  # Residual connections in up/down blocks
    mednext_block_counts: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2, 2, 2, 2])
    mednext_checkpoint_style: Optional[str] = None  # None or 'outside_block'
    mednext_norm: str = "group"  # 'group' or 'layer'
    mednext_dim: str = "3d"  # '2d' or '3d'
    mednext_grn: bool = False  # Global Response Normalization

    # RSUNet-specific parameters
    rsunet_norm: str = "batch"  # 'batch', 'group', 'instance', or 'none'
    rsunet_activation: str = "relu"  # 'relu', 'leakyrelu', 'prelu', or 'elu'
    rsunet_num_groups: int = 8  # Number of groups for GroupNorm
    rsunet_down_factors: Optional[List[List[int]]] = None  # E.g., [[1,2,2], [1,2,2], [1,2,2]]
    rsunet_depth_2d: int = 0  # Number of shallow layers using 2D convolutions
    rsunet_kernel_2d: List[int] = field(default_factory=lambda: [1, 3, 3])  # Kernel for 2D layers
    rsunet_act_negative_slope: float = 0.01  # For LeakyReLU
    rsunet_act_init: float = 0.25  # For PReLU

    # nnUNet-specific parameters (for loading pretrained models)
    nnunet_checkpoint: Optional[str] = None  # Path to pretrained checkpoint (.pth file)
    nnunet_plans: Optional[str] = None  # Path to plans.json file
    nnunet_dataset: Optional[str] = None  # Path to dataset.json file
    nnunet_device: str = "cuda"  # Device for model loading ('cuda' or 'cpu')

    # Deep supervision (supported by MedNeXt, RSUNet, and some MONAI models)
    deep_supervision: bool = False
    deep_supervision_weights: Optional[List[float]] = (
        None  # None = auto: [1.0, 0.5, 0.25, 0.125, 0.0625]
    )
    deep_supervision_clamp_min: float = -20.0  # Clamp logits to prevent numerical instability
    deep_supervision_clamp_max: float = 20.0  # Especially important at coarser scales
    enable_nan_detection: bool = True  # Detect NaN/Inf losses and print diagnostics
    debug_on_nan: bool = True  # Enter pdb on NaN/Inf detection (disable for non-interactive jobs)

    # Loss configuration
    loss_functions: List[str] = field(default_factory=lambda: ["DiceLoss", "BCEWithLogitsLoss"])
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    loss_kwargs: List[dict] = field(default_factory=lambda: [{}, {}])  # Per-loss kwargs
    # Explicit loss routing plan (required by LossOrchestrator for nontrivial setups)
    # Each term declares a loss index plus pred/target/mask slices and task grouping.
    loss_terms: Optional[List[Dict[str, Any]]] = None
    loss_balancing: LossBalancingConfig = field(default_factory=LossBalancingConfig)

    # Multi-task learning configuration
    # Defines which output channels correspond to which targets
    # Format: list of (start_ch, end_ch, target_name, loss_indices)
    # Example: [[0, 1, "binary", [0]], [1, 2, "boundary", [1]], [2, 3, "edt", [2]]]
    multi_task_config: Optional[List[List[Any]]] = (
        None  # None = single task (apply all losses to all channels)
    )

    # External model weights loading
    # For loading pretrained weights from external checkpoints (e.g., BANIS, nnUNet)
    external_weights_path: Optional[str] = None  # Path to external checkpoint file
    external_weights_key_prefix: str = "model."  # Prefix to strip from state_dict keys


# Label transformation configurations
@dataclass
class AffinityConfig:
    """Affinity map generation configuration. Enabled when offsets is non-empty."""

    offsets: List[str] = field(default_factory=list)  # Offsets in "z-y-x" format (empty = disabled)


@dataclass
class SkeletonDistanceConfig:
    """Skeleton-aware distance transform configuration."""

    enabled: bool = False
    resolution: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    alpha: float = 0.8
    smooth: bool = True
    bg_value: float = -1.0


@dataclass
class EdgeModeConfig:
    """Edge detection mode configuration for boundary generation.

    Controls how boundaries are detected and generated from segmentation masks.
    Different modes are optimized for different segmentation tasks.

    Edge Modes:
    - "all": Detects all boundaries including instance-to-background edges
    - "seg-all": Only instance-to-instance boundaries (recommended for instance segmentation)
    - "seg-no-bg": Instance boundaries excluding background interactions

    Attributes:
        mode: Edge detection mode ("all", "seg-all", "seg-no-bg")
        thickness: Boundary thickness in pixels (default: 1)
        processing_mode: Processing mode ("2d" for slice-by-slice, "3d" for full 3D)
    """

    mode: str = "seg-all"  # Edge detection mode
    thickness: int = 1  # Boundary thickness in pixels
    processing_mode: str = "3d"  # Processing mode: "2d" or "3d"


@dataclass
class LabelTargetConfig:
    """Configuration block describing an individual label target."""

    name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output_key: Optional[str] = None


@dataclass
class LabelTransformConfig:
    """Multi-channel label transformation configuration.

    Comprehensive configuration for label transformations including boundary detection,
    affinity maps, distance transforms, and multi-task learning setups. Supports
    various edge detection modes and transformation strategies.

    Key Features:
    - Multiple edge detection modes for boundary generation
    - Affinity map generation for instance segmentation
    - Skeleton-aware distance transforms
    - Multi-task learning support
    - Configurable output formats and data types

    Edge Mode Options:
    - "all": All boundaries (instance-to-instance and instance-to-background)
    - "seg-all": Only instance-to-instance boundaries (no background edges)
    - "seg-no-bg": Instance boundaries excluding background interactions
    """

    normalize: bool = True  # Convert labels to 0-1 range
    erosion: int = 0  # Border erosion kernel half-size (0 = disabled, uses seg_widen_border)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    skeleton_distance: SkeletonDistanceConfig = field(default_factory=SkeletonDistanceConfig)
    edge_mode: EdgeModeConfig = field(
        default_factory=EdgeModeConfig
    )  # Edge detection configuration
    keys: List[str] = field(default_factory=lambda: ["label"])
    stack_outputs: bool = True
    retain_original: bool = False
    output_dtype: Optional[str] = "float32"
    output_key_format: str = "{key}_{task}"
    allow_missing_keys: bool = False
    segment_id: Optional[List[int]] = None
    boundary_thickness: int = 1
    targets: List[Any] = field(default_factory=list)


@dataclass
class DataTransformConfig:
    """Data transformation configuration applied to all data (image/label/mask).

    These transforms are applied to paired data (image, label, mask) to ensure
    spatial alignment. Each transform uses appropriate interpolation:
    - Image: bilinear interpolation (smooth)
    - Label/Mask: nearest-neighbor interpolation (preserves integer values)
    """

    resize: Optional[List[float]] = (
        None  # Resize to target size [H, W] for 2D or [D, H, W] for 3D. None = no resize.
    )


@dataclass
class ImageTransformConfig:
    """Image transformation configuration (applied to image only)."""

    normalize: str = "0-1"  # "none", "normal" (z-score), or "0-1" (min-max)
    clip_percentile_low: float = (
        0.0  # Lower percentile for clipping (0.0 = no clip, 0.05 = 5th percentile)
    )
    clip_percentile_high: float = (
        1.0  # Upper percentile for clipping (1.0 = no clip, 0.95 = 95th percentile)
    )
    pad_size: Optional[List[int]] = None  # Reflection padding for context [D, H, W] or [H, W]
    pad_mode: str = "reflect"  # Padding mode: 'reflect', 'replicate', 'constant'


@dataclass
class NNUNetPreprocessingConfig:
    """nnU-Net-style preprocessing configuration.

    This block controls preprocessing that should mirror nnU-Net's behavior:
    - foreground crop
    - spacing-aware resampling
    - z-score or simple intensity normalization
    - optional restoration to input space before saving predictions
    """

    enabled: bool = False
    crop_to_nonzero: bool = True
    target_spacing: Optional[List[float]] = None  # [z, y, x] for 3D, [y, x] for 2D
    source_spacing: Optional[List[float]] = None  # If None, falls back to *_resolution fields
    normalization: str = "zscore"  # "zscore", "none", "0-1", or "divide-K"
    normalization_use_nonzero_mask: bool = True
    clip_percentile_low: float = 0.0  # Optional clipping before normalization (fraction)
    clip_percentile_high: float = 1.0  # Optional clipping before normalization (fraction)
    force_separate_z: Optional[bool] = None  # None = auto
    anisotropy_threshold: float = 3.0
    image_order: int = 3
    label_order: int = 0
    order_z: int = 0
    restore_to_input_space: bool = True


@dataclass
class DataConfig:
    """Dataset and data loading configuration.

    Comprehensive configuration for data loading, preprocessing, augmentation,
    and dataset management. Supports both volume-based and file-based datasets
    with extensive transformation and augmentation options.

    Key Features:
    - Support for volume-based and JSON-based datasets
    - Configurable data augmentation pipelines
    - Multi-channel label transformations
    - Train/validation splitting options
    - Caching and performance optimization
    - 2D data support with do_2d parameter
    """

    # Dataset type
    dataset_type: Optional[str] = None  # Type of dataset: None (volume), 'filename', 'tile', etc.

    # 2D data support
    do_2d: bool = False  # Enable 2D data processing (extract 2D slices from 3D volumes)

    # Base path (prepended to train_image, train_label, etc. if set)
    train_path: str = ""  # Base path for training data (e.g., "/path/to/dataset/")
    val_path: str = ""  # Base path for validation data

    # Paths - Volume-based datasets
    # These can be strings (single file), lists (multiple files), or None
    # Using Any to support both str and List[str] (OmegaConf doesn't support Union of containers)
    train_image: Any = None  # str, List[str], or None
    train_label: Any = None  # str, List[str], or None
    train_mask: Any = None  # str, List[str], or None (Valid region mask for training)
    val_image: Any = None  # str, List[str], or None
    val_label: Any = None  # str, List[str], or None
    val_mask: Any = None  # str, List[str], or None (Valid region mask for validation)

    # Paths - JSON/filename-based datasets
    train_json: Optional[str] = None  # JSON file with image/label file lists
    val_json: Optional[str] = None
    train_image_key: str = "images"  # Key in JSON for image files
    train_label_key: str = "masks"  # Key in JSON for label files
    val_image_key: str = "images"
    val_label_key: str = "masks"
    train_val_split: Optional[float] = None  # Auto split ratio (e.g., 0.9 = 90% train, 10% val)

    # Data properties
    patch_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    pad_size: List[int] = field(default_factory=lambda: [0, 0, 0])
    pad_mode: str = "reflect"  # Padding mode: 'reflect', 'replicate', 'constant', 'edge'
    stride: List[int] = field(default_factory=lambda: [1, 1, 1])  # Sampling stride (z, y, x)

    # Voxel resolution (physical dimensions in nm)
    train_resolution: Optional[List[float]] = (
        None  # Training data resolution [z, y, x] in nm (e.g., [30, 6, 6])
    )
    val_resolution: Optional[List[float]] = (
        None  # Validation data resolution [z, y, x] in nm; if None, falls back to train_resolution
    )

    # Axis transposition (empty list = no transpose)
    train_transpose: List[int] = field(
        default_factory=list
    )  # Axis permutation for training data (e.g., [2,1,0] for xyz->zyx)
    val_transpose: List[int] = field(default_factory=list)  # Axis permutation for validation data

    # Dataset statistics (for auto-planning)
    target_spacing: Optional[List[float]] = None  # Target voxel spacing [z, y, x] in mm
    median_shape: Optional[List[int]] = None  # Median dataset shape [D, H, W] in voxels

    # Train/Val Split (inspired by DeepEM)
    # If enabled, splits single volume into train/val regions
    split_enabled: bool = False  # Enable automatic train/val split (default: False)
    split_train_range: List[float] = field(default_factory=lambda: [0.0, 0.8])  # Train: 0-80%
    split_val_range: List[float] = field(default_factory=lambda: [0.8, 1.0])  # Val: 80-100%
    split_axis: int = 0  # Axis to split along (0=Z, 1=Y, 2=X)
    split_pad_val: bool = True  # Pad validation to patch_size if smaller
    split_pad_mode: str = "reflect"  # Padding mode: 'reflect', 'replicate', 'constant'

    # Data loading (batch_size and num_workers moved to system.training/system.inference)
    pin_memory: bool = True
    persistent_workers: bool = True

    # Caching (MONAI)
    use_cache: bool = False
    cache_rate: float = 1.0

    # Data transformation (applied to image/label/mask)
    data_transform: DataTransformConfig = field(default_factory=DataTransformConfig)

    # Image transformation (applied to image only)
    image_transform: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    nnunet_preprocessing: NNUNetPreprocessingConfig = field(
        default_factory=NNUNetPreprocessingConfig
    )

    # Sampling (for volumetric datasets)
    iter_num_per_epoch: Optional[int] = None  # Alias for iter_num (if set, overrides iter_num)
    val_iter_num: Optional[int] = None  # Validation iterations per epoch (auto-calculated if None)
    use_preloaded_cache_train: bool = True  # Preload training volumes into memory
    use_preloaded_cache_val: bool = True  # Preload validation volumes into memory
    use_lazy_zarr: bool = False  # Lazy crop-on-read for zarr volumes (no full preload)
    cached_sampling_max_attempts: int = 10  # Retry attempts for foreground-aware sampling
    cached_sampling_foreground_threshold: float = (
        0.0  # Minimum (label > 0) fraction required for training crops; 0 disables foreground sampling
    )
    cached_sampling_crop_to_nonzero_mask: bool = (
        False  # Restrict random crops to intersect the nonzero bounding box of the mask volume
    )

    # Reject sampling configuration (for volumetric patch sampling)
    reject_sampling: Optional[Dict[str, Any]] = None  # Dict with 'size_thres' and 'p' keys

    # Multi-channel label transformation (for affinity maps, distance transforms, etc.)
    label_transform: LabelTransformConfig = field(default_factory=LabelTransformConfig)

    # Augmentation configuration (nested under data in YAML)
    augmentation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # CellMap-specific configuration (for CellMap Segmentation Challenge)
    cellmap: Optional[Dict[str, Any]] = (
        None  # CellMap-specific data config (see tutorials/cellmap_*.yaml)
    )


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "AdamW"
    lr: float = 0.001
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: str = "CosineAnnealingLR"
    warmup_epochs: int = 10
    warmup_start_lr: float = 0.0001
    min_lr: float = 0.00001

    # Scheduler interval control
    interval: str = "epoch"  # "epoch" or "step" - controls when scheduler steps
    frequency: int = 1  # How often to step the scheduler

    # CosineAnnealing-specific
    t_max: Optional[int] = None

    # CosineAnnealingWarmRestarts-specific
    T_0: int = 200
    T_mult: int = 1

    # ReduceLROnPlateau-specific
    mode: str = "min"  # 'min' or 'max'
    patience: int = 10
    factor: float = 0.5
    threshold: float = 0.0001
    cooldown: int = 0
    monitor: str = "val_loss_total"  # Metric to monitor for ReduceLROnPlateau

    # StepLR-specific
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class EMAConfig:
    """Exponential Moving Average (EMA) configuration."""

    enabled: bool = False
    decay: float = 0.999
    warmup_steps: int = 0
    validate_with_ema: bool = True
    device: Optional[str] = None  # e.g., "cpu" to offload EMA weights
    copy_buffers: bool = True  # Keep BatchNorm buffers in sync with model


@dataclass
class OptimizationConfig:
    """Optimization configuration (optimizer + scheduler + training params).

    Comprehensive configuration for training optimization including optimizers,
    learning rate schedulers, and training parameters. Supports multiple
    optimization strategies and precision modes.

    Key Features:
    - Multiple optimizer support (AdamW, SGD, etc.)
    - Learning rate scheduling (CosineAnnealing, ReduceLROnPlateau, etc.)
    - Mixed precision training support
    - Gradient clipping and accumulation
    - Performance optimization settings
    """

    max_epochs: int = 100
    max_steps: Optional[int] = None
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "32"  # "32", "16", "bf16", "16-mixed", "bf16-mixed"

    # Performance
    deterministic: bool = False
    benchmark: bool = True

    # Validation and logging
    val_check_interval: Union[int, float] = 1.0  # Validate every N epochs (legacy key name)
    log_every_n_steps: int = 50
    num_sanity_val_steps: int = 0

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""

    monitor: str = "train_loss_total_epoch"
    dirpath: Optional[str] = None
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    save_every_n_epochs: int = 10
    checkpoint_filename: Optional[str] = None  # Auto-generated from monitor if None
    use_timestamp: bool = True  # Create timestamped subdirectories (YYYYMMDD_HHMMSS)


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    monitor: str = "train_loss_total_epoch"
    patience: int = 100
    mode: str = "min"  # 'min' or 'max'
    min_delta: float = 1e-5
    check_finite: bool = True  # Stop training if monitored metric becomes NaN or inf
    threshold: Optional[float] = None  # Stop if metric reaches this value
    divergence_threshold: Optional[float] = None  # Stop if metric diverges beyond this value


@dataclass
class ScalarLoggingConfig:
    """Scalar logging configuration."""

    loss: List[str] = field(default_factory=lambda: ["train_loss_total_epoch"])
    loss_every_n_steps: int = 10
    val_check_interval: Union[int, float] = 1.0
    benchmark: bool = True


@dataclass
class ImageLoggingConfig:
    """Image visualization configuration."""

    enabled: bool = True
    max_images: int = 4
    num_slices: int = 8
    log_every_n_epochs: int = 1  # Log visualization every N epochs

    # Channel visualization options
    channel_mode: str = "argmax"  # 'argmax', 'all', or 'selected'
    selected_channels: Optional[List[int]] = None  # Only used when channel_mode='selected'


@dataclass
class PredictionSavingConfig:
    """Configuration for saving intermediate predictions during training/validation."""

    enabled: bool = False  # Enable saving predictions
    save_during_training: bool = False  # Save predictions during training
    save_during_validation: bool = True  # Save predictions during validation
    save_every_n_epochs: int = 10  # Save predictions every N epochs
    save_every_n_steps: Optional[int] = (
        None  # Save predictions every N steps (overrides epochs if set)
    )
    output_dir: str = "outputs/predictions"  # Directory to save predictions
    max_samples: int = 4  # Maximum number of samples to save per epoch/step
    save_labels: bool = True  # Also save ground truth labels
    save_inputs: bool = False  # Also save input images
    apply_activation: bool = True  # Apply sigmoid/softmax before saving
    apply_decoding: bool = False  # Apply instance segmentation decoding before saving


@dataclass
class LoggingConfig:
    """Logging configuration (scalar + images)."""

    scalar: ScalarLoggingConfig = field(default_factory=ScalarLoggingConfig)
    images: ImageLoggingConfig = field(default_factory=ImageLoggingConfig)
    predictions: PredictionSavingConfig = field(default_factory=PredictionSavingConfig)


@dataclass
class MonitorConfig:
    """Monitoring configuration (logging, checkpointing, early stopping).

    Comprehensive configuration for training monitoring, logging, and model
    checkpointing. Includes TensorBoard logging, model checkpointing, and
    early stopping mechanisms.

    Key Features:
    - TensorBoard logging with scalar and image visualization
    - Model checkpointing with multiple strategies
    - Early stopping with configurable metrics
    - Prediction saving during training/validation
    - Anomaly detection for debugging
    """

    detect_anomaly: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


# Augmentation configurations
@dataclass
class FlipConfig:
    """Random flipping augmentation."""

    enabled: Optional[bool] = None  # None=use preset default, True/False=explicit
    prob: float = 0.5
    spatial_axis: Optional[List[int]] = None  # None = all axes


@dataclass
class AffineConfig:
    """Affine transformation augmentation (rotation, scaling, shearing)."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    rotate_range: Tuple[float, float, float] = (0.2, 0.2, 0.2)  # Rotation range in radians (~11°)
    scale_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # Scaling range (±10%)
    shear_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # Shearing range (±10%)


@dataclass
class RotateConfig:
    """Random rotation augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    max_angle: float = 90.0
    spatial_axes: Tuple[int, int] = (1, 2)  # Axes to rotate: (1, 2) = Y-X plane (preserves Z)


@dataclass
class ElasticConfig:
    """Elastic deformation augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.3
    sigma_range: Tuple[float, float] = (5.0, 8.0)
    magnitude_range: Tuple[float, float] = (50.0, 150.0)


@dataclass
class IntensityConfig:
    """Intensity augmentation."""

    enabled: Optional[bool] = None
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: float = 0.05
    shift_intensity_prob: float = 0.3
    shift_intensity_offset: float = 0.1
    contrast_prob: float = 0.3
    contrast_range: Tuple[float, float] = (0.7, 1.4)


@dataclass
class MisalignmentConfig:
    """Misalignment augmentation for EM data."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    displacement: int = 16
    rotate_ratio: float = 0.0


@dataclass
class MissingSectionConfig:
    """Missing section augmentation for EM data."""

    enabled: Optional[bool] = None
    prob: float = 0.3
    num_sections: int = 2


@dataclass
class MotionBlurConfig:
    """Motion blur augmentation for EM data."""

    enabled: Optional[bool] = None
    prob: float = 0.3
    sections: Any = 2
    kernel_size: int = 11


@dataclass
class CutNoiseConfig:
    """CutNoise augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    length_ratio: float = 0.25
    noise_scale: float = 0.2


@dataclass
class CutBlurConfig:
    """CutBlur augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.3
    length_ratio: float = 0.25
    down_ratio_range: Tuple[float, float] = (2.0, 8.0)
    downsample_z: bool = False


@dataclass
class MissingPartsConfig:
    """Missing parts augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    hole_range: Tuple[float, float] = (0.1, 0.3)


@dataclass
class MixupConfig:
    """Mixup augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    alpha_range: Tuple[float, float] = (0.7, 0.9)


@dataclass
class CopyPasteConfig:
    """Copy-Paste augmentation."""

    enabled: Optional[bool] = None
    prob: float = 0.5
    max_obj_ratio: float = 0.7
    rotation_angles: List[int] = field(default_factory=lambda: list(range(30, 360, 30)))
    border: int = 3


@dataclass
class StripeConfig:
    """Random stripe augmentation for EM artifacts.

    Simulates horizontal, vertical, or diagonal stripe artifacts common in
    electron microscopy, such as curtaining or scan line artifacts.

    Attributes:
        enabled: Whether to enable stripe augmentation (None=use preset default)
        prob: Probability of applying the augmentation
        num_stripes_range: Range for number of stripes as (min, max)
        thickness_range: Range for stripe thickness in pixels as (min, max)
        intensity_range: Range for stripe intensity values as (min, max)
        angle_range: Optional range for stripe angles in degrees as (min, max)
            0° = horizontal, 90° = vertical, 45° = diagonal
            Use None for predefined orientations (horizontal/vertical/random)
        orientation: Stripe orientation when angle_range is None
            - 'horizontal': 0° stripes
            - 'vertical': 90° stripes
            - 'random': randomly choose between horizontal and vertical
        mode: How to apply stripes - 'add' (additive) or 'replace' (replacement)
    """

    enabled: Optional[bool] = None
    prob: float = 0.3
    num_stripes_range: Tuple[int, int] = (2, 10)
    thickness_range: Tuple[int, int] = (1, 5)
    intensity_range: Tuple[float, float] = (-0.2, 0.2)
    angle_range: Optional[Tuple[float, float]] = None
    orientation: str = "random"
    mode: str = "add"


@dataclass
class AugmentationConfig:
    """Complete augmentation configuration.

    All augmentations default to enabled=False for safety. Choose a preset mode:

    - "none": Disable all augmentations (no changes to data)
    - "some": Enable only augmentations explicitly set to enabled=True in YAML
    - "all": Enable all augmentations by default (set to True in config)

    For preset="all", users should set enabled=True for augmentations they want.
    """

    preset: str = "some"  # "all", "some", or "none" - controls how enabled flags are interpreted

    # Standard augmentations
    flip: FlipConfig = field(default_factory=FlipConfig)
    affine: AffineConfig = field(default_factory=AffineConfig)
    rotate: RotateConfig = field(default_factory=RotateConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)

    # EM-specific augmentations
    misalignment: MisalignmentConfig = field(default_factory=MisalignmentConfig)
    missing_section: MissingSectionConfig = field(default_factory=MissingSectionConfig)
    motion_blur: MotionBlurConfig = field(default_factory=MotionBlurConfig)
    cut_noise: CutNoiseConfig = field(default_factory=CutNoiseConfig)
    cut_blur: CutBlurConfig = field(default_factory=CutBlurConfig)
    missing_parts: MissingPartsConfig = field(default_factory=MissingPartsConfig)
    stripe: StripeConfig = field(default_factory=StripeConfig)

    # Advanced augmentations
    mixup: MixupConfig = field(default_factory=MixupConfig)
    copy_paste: CopyPasteConfig = field(default_factory=CopyPasteConfig)


@dataclass
class InferenceDataConfig:
    """Inference data configuration.

    Output path is automatically computed from checkpoint directory:
    - Checkpoint: outputs/experiment_name/YYYYMMDD_HHMMSS/checkpoints/last.ckpt
    - Inference: outputs/experiment_name/YYYYMMDD_HHMMSS/inference/last.ckpt/{output_name}
    """

    test_image: Any = None  # str, List[str], or None - Can be single file or list of files
    test_label: Any = None  # str, List[str], or None - Can be single file or list of files
    test_mask: Any = None  # str, List[str], or None - Optional mask for inference
    test_resolution: Optional[List[float]] = (
        None  # Test data resolution [z, y, x] in nm (e.g., [30, 6, 6])
    )
    test_transpose: List[int] = field(
        default_factory=list
    )  # Axis permutation for test data (e.g., [2,1,0] for xyz->zyx)

    # Tuning data (for parameter optimization)
    tune_image: Any = None  # str, List[str], or None - Data for parameter tuning
    tune_label: Any = None  # str, List[str], or None - Labels for tuning evaluation
    tune_mask: Any = None  # str, List[str], or None - Valid region mask for tuning
    tune_resolution: Optional[List[float]] = (
        None  # Tuning data resolution [z, y, x] in nm (e.g., [30, 8, 8])
    )
    tune_transpose: List[int] = field(default_factory=list)  # Axis permutation for tuning data
    output_path: Optional[str] = None  # Optional explicit directory for inference outputs
    output_name: str = (
        "predictions.h5"  # Output filename (auto-pathed to inference/{checkpoint}/{output_name})
    )

    # Image transformation (applied to test images during inference)
    image_transform: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    nnunet_preprocessing: NNUNetPreprocessingConfig = field(
        default_factory=NNUNetPreprocessingConfig
    )

    # 2D data support
    do_2d: bool = False  # Enable 2D data processing for inference


@dataclass
class SlidingWindowConfig:
    """MONAI SlidingWindowInferer configuration."""

    window_size: Optional[List[int]] = None
    sw_batch_size: Optional[int] = None  # If None, will use system.inference.batch_size
    overlap: Optional[Any] = None  # Overlap between window passes (float or sequence)
    stride: Optional[List[int]] = None  # Explicit stride for controlling window movement
    blending: str = "gaussian"  # 'gaussian' or 'constant' - blending mode for overlapping patches
    sigma_scale: float = (
        0.125  # Gaussian sigma scale (only for blending='gaussian'); larger = smoother blending
    )
    padding_mode: str = "constant"  # Padding mode at volume boundaries
    pad_size: Optional[List[int]] = None  # Padding size for context (e.g., [16, 32, 32])
    save_channels: Optional[List[int]] = None  # Channel indices to save (None = all channels)
    keep_input_on_cpu: bool = False  # Keep full input volume on CPU; MONAI moves windows to sw_device
    sw_device: Optional[str] = None  # MONAI window compute device (e.g., "cuda")
    output_device: Optional[str] = None  # MONAI stitched output device (e.g., "cpu")


@dataclass
class TestTimeAugmentationConfig:
    """Test-time augmentation configuration.

    Note: Saving predictions is now handled by SavePredictionConfig.

    Axis Indexing:
    - flip_axes: Uses full tensor indices (e.g., [2, 3] for H, W in 5D tensor (B, C, D, H, W))
    - rotation90_axes: Uses spatial-only indices (e.g., [1, 2] for H-W plane where 0=D, 1=H, 2=W)
    """

    enabled: bool = True  # Enable flip/rotation TTA (preprocessing still applies)
    flip_axes: Any = None  # TTA flip strategy: "all" (8 flips), null (no aug),
    # or list like [[2], [3]] (full tensor indices)
    rotation90_axes: Any = None  # TTA rotation90 strategy: "all" (3 planes × 4 rotations),
    # null, or list like [[1, 2]] (spatial indices: 0=D, 1=H, 2=W)
    channel_activations: Optional[List[Any]] = (
        None  # Per-channel activations: [[start_ch, end_ch, 'activation'], ...]
        # e.g., [[0, 2, 'softmax'], [2, 3, 'sigmoid'], [3, 4, 'tanh']]
    )
    select_channel: Any = None  # Channel selection: null (all), [1] (foreground), -1 (all)
    # (applied even with null flip_axes)
    ensemble_mode: str = "mean"  # Ensemble mode for TTA: 'mean', 'min', 'max'
    apply_mask: bool = False  # Multiply each channel by corresponding test_mask after ensemble


@dataclass
class SavePredictionConfig:
    """Configuration for saving intermediate predictions during inference.

    Controls how raw model predictions are saved before any decoding or postprocessing.
    Useful for debugging, visualization, or running multiple decoding strategies.

    Attributes:
        enabled: Enable saving intermediate predictions (default: True)
        intensity_scale: Scale factor for predictions (e.g., 255 for uint8 visualization)
        intensity_dtype: Data type for saved predictions (e.g., 'uint8', 'float32')
        analyze_h5: Print HDF5 summary statistics after saving each .h5 file
        output_formats: List of output formats to save predictions
                       (e.g., ['h5', 'tiff', 'nii.gz'])
                       Supported formats: 'h5', 'tiff', 'nii', 'nii.gz', 'png'
                       Default: ['h5']
    """

    enabled: bool = True  # Enable saving intermediate predictions
    intensity_scale: float = (
        -1.0  # If < 0, keep raw predictions (no normalization/scaling).
        # If > 0, normalize to [0,1] then scale.
    )
    intensity_dtype: str = (
        "uint8"  # Save as uint8 for visualization (ignored if intensity_scale < 0)
    )
    analyze_h5: bool = False  # Print H5 analysis summary after saving HDF5 output
    output_formats: List[str] = field(
        default_factory=lambda: ["h5"]  # Default: HDF5 only (NIfTI is opt-in)
    )


@dataclass
class DecodeBinaryContourDistanceWatershedConfig:
    """Configuration for decode_binary_contour_distance_watershed function."""

    binary_threshold: Tuple[float, float] = (
        0.9,
        0.85,
    )  # (seed_threshold, foreground_threshold) for binary mask
    contour_threshold: Tuple[float, float] = (
        0.8,
        1.1,
    )  # (seed_threshold, foreground_threshold) for instance contours
    distance_threshold: Tuple[float, float] = (
        0.5,
        -0.5,
    )  # (seed_threshold, foreground_threshold) for signed distance
    min_instance_size: int = 128  # Minimum size threshold for instances to keep
    scale_factors: Tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )  # Scale factors for resizing in (Z, Y, X) order
    remove_small_mode: str = "background"  # 'background', 'neighbor', or 'none'
    min_seed_size: int = 32  # Minimum size of seed objects
    return_seed: bool = False  # Whether to return the seed map
    precomputed_seed: Optional[Any] = None  # Precomputed seed map (numpy array)
    prediction_scale: int = 255  # Scale of input predictions (255 for uint8 range, 1 for float)


@dataclass
class DecodeModeConfig:
    """Configuration for a single decode mode/function."""

    name: str = "decode_binary_watershed"  # Function name: decode_binary_cc,
    # decode_binary_watershed, decode_binary_contour_distance_watershed, etc.
    kwargs: Dict[str, Any] = field(
        default_factory=dict
    )  # Keyword arguments for the decode function


@dataclass
class BinaryPostprocessingConfig:
    """Binary segmentation postprocessing configuration.

    Applies morphological operations and connected components filtering to binary predictions.
    Input should already be binary (from binary_thresholding decoding).
    Pipeline order:
        1. Ensure input is binary (auto-threshold if needed: 0.5 for [0,1], 0 for >1)
        2. Apply morphological opening (erosion + dilation)
        3. Extract connected components
        4. Keep top-k largest components
    """

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


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    enabled: bool = True  # Use eval mode (vs train mode for BatchNorm)
    metrics: Optional[List[str]] = None  # e.g., ['dice', 'jaccard', 'accuracy']


@dataclass
class InferenceConfig:
    """Inference configuration.

    Shared inference settings for sliding window and test-time augmentation.
    Data paths are specified in separate 'test' and 'tune' sections.

    Key Features:
    - Sliding window inference for large volumes
    - Test-time augmentation (TTA) support
    - Saving intermediate predictions
    - Multiple decoding strategies
    - Postprocessing and evaluation
    - System resource overrides for inference

    Note: Test data is in 'test.data', tuning data is in 'tune.data'
    """

    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    test_time_augmentation: TestTimeAugmentationConfig = field(
        default_factory=TestTimeAugmentationConfig
    )
    save_prediction: SavePredictionConfig = field(default_factory=SavePredictionConfig)
    decoding: Optional[List[DecodeModeConfig]] = None  # List of decode modes to apply sequentially
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Inference-specific overrides (override system settings during inference)
    # Use -1 to keep training values, or >= 0 to override
    num_gpus: int = -1  # Override system.training.num_gpus if >= 0
    batch_size: int = -1  # Override system.training.batch_size if >= 0 (typically 1 for inference)
    num_workers: int = -1  # Override system.training.num_workers if >= 0


@dataclass
class TestDataConfig:
    """Test data configuration."""

    # These can be strings (single file), lists (multiple files), or None
    # Using Any to support both str and List[str] (OmegaConf doesn't support Union of containers)
    test_path: str = ""  # Base path for test data (prepended to test_* paths if set)
    test_image: Any = None  # str, List[str], or None
    test_label: Any = None  # str, List[str], or None
    test_mask: Any = None  # str, List[str], or None
    test_resolution: Optional[List[float]] = None
    test_transpose: Optional[List[int]] = None
    output_path: Optional[str] = None
    cache_suffix: str = "_prediction.h5"
    # Image transformation (applied to test images during inference)
    image_transform: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    nnunet_preprocessing: NNUNetPreprocessingConfig = field(
        default_factory=NNUNetPreprocessingConfig
    )
    # Label transformation (optional). Typically unused in test mode to preserve raw labels
    label_transform: Optional[LabelTransformConfig] = None


@dataclass
class TestConfig:
    """Test-specific configuration (data paths, decoding, evaluation)."""

    data: TestDataConfig = field(default_factory=TestDataConfig)
    decoding: Optional[List[Dict[str, Any]]] = None
    evaluation: Optional[Dict[str, Any]] = field(default_factory=lambda: {"enabled": False})


@dataclass
class TuneDataConfig:
    """Tuning data configuration."""

    # These can be strings (single file), lists (multiple files), or None
    # Using Any to support both str and List[str] (OmegaConf doesn't support Union of containers)
    tune_image: Any = None  # str, List[str], or None
    tune_label: Any = None  # str, List[str], or None
    tune_mask: Any = None  # str, List[str], or None
    tune_resolution: Optional[List[int]] = None
    # Image transformation (applied to tune images during inference)
    image_transform: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    nnunet_preprocessing: NNUNetPreprocessingConfig = field(
        default_factory=NNUNetPreprocessingConfig
    )
    # Label transformation (optional). Typically unused in tune mode to preserve raw labels
    label_transform: Optional[LabelTransformConfig] = None


@dataclass
class TuneOutputConfig:
    """Tuning output configuration."""

    output_dir: Optional[str] = None
    output_pred: Optional[str] = None
    cache_suffix: str = "_tta_prediction.h5"
    save_all_trials: bool = False
    save_best_segmentation: bool = True
    save_study: bool = True
    visualizations: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


@dataclass
class ParameterConfig:
    """Single parameter configuration for optimization."""

    type: str  # "float", "int", "categorical"
    range: List[Any]  # [min, max] for numeric, [options...] for categorical
    step: Optional[float] = None
    log: bool = False
    param_group: Optional[str] = None  # For tuple parameters
    tuple_index: Optional[int] = None  # Position in tuple
    description: Optional[str] = None


@dataclass
class DecodingParameterSpace:
    """Decoding function parameter space configuration."""

    function_name: str = "decode_binary_contour_distance_watershed"
    defaults: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)  # Dict[str, ParameterConfig]


@dataclass
class PostprocessingParameterSpace:
    """Post-processing function parameter space configuration."""

    enabled: bool = False
    function_name: str = "remove_small_instances"
    defaults: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)  # Dict[str, ParameterConfig]


@dataclass
class ParameterSpaceConfig:
    """Parameter space configuration for Optuna optimization."""

    decoding: DecodingParameterSpace = field(default_factory=DecodingParameterSpace)
    postprocessing: PostprocessingParameterSpace = field(
        default_factory=PostprocessingParameterSpace
    )


@dataclass
class TuneConfig:
    """Parameter tuning configuration (Optuna settings)."""

    enabled: bool = True
    n_trials: int = 100
    timeout: Optional[int] = None
    study_name: str = "parameter_optimization"
    storage: Optional[str] = None
    load_if_exists: bool = True
    sampler: Dict[str, Any] = field(default_factory=lambda: {"name": "TPE"})
    pruner: Optional[Dict[str, Any]] = None
    optimization: Dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "single",
            "single_objective": {"metric": "adapted_rand", "direction": "minimize"},
        }
    )
    data: TuneDataConfig = field(default_factory=TuneDataConfig)
    output: TuneOutputConfig = field(default_factory=TuneOutputConfig)
    logging: Dict[str, Any] = field(default_factory=lambda: {"verbose": True})
    parameter_space: ParameterSpaceConfig = field(default_factory=ParameterSpaceConfig)


@dataclass
class Config:
    """Main configuration for PyTorch Connectomics.

    The central configuration class that combines all configuration sections
    into a single, type-safe configuration object. This is the main entry point
    for all PyTorch Connectomics experiments.

    Configuration Sections:
        system: Hardware and parallelization settings
        model: Neural network architecture and loss functions
        data: Training dataset loading and preprocessing
        optimization: Training parameters and schedulers
        monitor: Logging, checkpointing, and monitoring
        inference: Shared inference settings (sliding window, TTA, decoding, postprocessing)
        test: Test-specific configuration (test data paths, decoding, evaluation)
        tune: Parameter tuning configuration (tuning data paths, optimization settings)

    Attributes:
        experiment_name: Name of the experiment for organization
        description: Optional description of the experiment
        system: System configuration for hardware and parallelization
        model: Model architecture and loss configuration
        data: Training data loading and preprocessing configuration
        optimization: Training optimization configuration
        monitor: Monitoring and logging configuration
        inference: Shared inference settings (no data paths)
        test: Test-specific configuration (includes test.data with test_image, test_label, etc.)
        tune: Parameter tuning configuration (includes tune.data with tune_image, tune_label, etc.)
    """

    # Metadata
    experiment_name: str = "connectomics_experiment"
    description: str = ""

    # Core components
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Optional: Test-specific configuration (test data paths, decoding, evaluation)
    test: Optional[TestConfig] = None

    # Optional: Parameter tuning configuration (tuning data paths, optimization settings)
    tune: Optional[TuneConfig] = None


# Allow safe loading of checkpoints with PyTorch 2.6+ weights_only defaults
try:
    import torch

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        # Register every dataclass defined in this module so Lightning checkpoints
        # can be loaded safely when torch.load defaults to weights_only=True.
        safe_dataclasses = [
            obj
            for obj in globals().values()
            if inspect.isclass(obj) and obj.__module__ == __name__ and is_dataclass(obj)
        ]
        torch.serialization.add_safe_globals(safe_dataclasses)
except Exception:
    # Best-effort registration; ignore if torch not available at import time
    pass


# Utility functions for common configuration tasks
def configure_edge_mode(
    cfg: Config, mode: str = "seg-all", thickness: int = 1, processing_mode: str = "3d"
) -> None:
    """Configure edge detection mode for instance segmentation.

    Args:
        cfg: Configuration object to modify
        mode: Edge detection mode ("all", "seg-all", "seg-no-bg")
        thickness: Boundary thickness in pixels
        processing_mode: Processing mode ("2d" or "3d")
    """
    cfg.data.label_transform.edge_mode.mode = mode
    cfg.data.label_transform.edge_mode.thickness = thickness
    cfg.data.label_transform.edge_mode.processing_mode = processing_mode


def configure_instance_segmentation(cfg: Config, boundary_thickness: int = 5) -> None:
    """Configure for instance segmentation with boundary detection.

    Args:
        cfg: Configuration object to modify
        boundary_thickness: Thickness of instance boundaries
    """
    configure_edge_mode(cfg, mode="seg-all", thickness=boundary_thickness, processing_mode="3d")
    cfg.data.label_transform.boundary_thickness = boundary_thickness


__all__ = [
    # Main configuration class
    "Config",
    # System configuration
    "SystemConfig",
    "SystemTrainingConfig",
    "SystemInferenceConfig",
    # Model configuration
    "ModelConfig",
    # Data configuration
    "DataConfig",
    "ImageTransformConfig",
    "LabelTransformConfig",
    "AffinityConfig",
    "SkeletonDistanceConfig",
    "LabelTargetConfig",
    "EdgeModeConfig",
    # Optimization configuration
    "OptimizationConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    # Monitoring configuration
    "MonitorConfig",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "LoggingConfig",
    "ScalarLoggingConfig",
    "ImageLoggingConfig",
    "PredictionSavingConfig",
    # Inference configuration
    "InferenceConfig",
    "InferenceDataConfig",
    "SlidingWindowConfig",
    "TestTimeAugmentationConfig",
    "SavePredictionConfig",
    "PostprocessingConfig",
    "BinaryPostprocessingConfig",
    "ConnectedComponentsConfig",
    "EvaluationConfig",
    "DecodeModeConfig",
    "DecodeBinaryContourDistanceWatershedConfig",
    # Test configuration
    "TestDataConfig",
    "TestConfig",
    # Tuning configuration
    "TuneDataConfig",
    "TuneOutputConfig",
    "ParameterConfig",
    "DecodingParameterSpace",
    "PostprocessingParameterSpace",
    "ParameterSpaceConfig",
    "TuneConfig",
    # Augmentation configuration
    "AugmentationConfig",
    "FlipConfig",
    "AffineConfig",  # Added AffineConfig
    "RotateConfig",
    "ElasticConfig",
    "IntensityConfig",
    "MisalignmentConfig",
    "MissingSectionConfig",
    "MotionBlurConfig",
    "CutNoiseConfig",
    "CutBlurConfig",
    "MissingPartsConfig",
    "StripeConfig",
    "MixupConfig",
    "CopyPasteConfig",
    # Utility functions
    "configure_edge_mode",
    "configure_instance_segmentation",
]
