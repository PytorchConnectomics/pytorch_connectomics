from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Label transformation configurations
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

    Modes:
    - "all": Includes all boundaries (instance-to-instance and instance-to-background)
    - "seg-all": Only boundaries between different instances
    - "seg-no-bg": Boundaries between instances excluding background interactions
    """

    mode: str = "all"  # "all", "seg-all", or "seg-no-bg"
    thickness: int = 1  # Boundary thickness in pixels
    processing_mode: str = "2d"  # "2d" or "3d" processing


@dataclass
class LabelTargetConfig:
    """Configuration for one label transform target."""

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
    align_to_image: bool = (
        False  # For mask transforms: allow minor center pad/crop to match image/prediction size.
    )
    binarize: bool = False  # If True, convert values to {0,1} via `value > threshold`
    threshold: float = 0.0  # Threshold used when binarize=True

    # Data properties
    pad_size: List[int] = field(default_factory=lambda: [0, 0, 0])
    pad_mode: str = "reflect"  # Padding mode: 'reflect', 'replicate', 'constant', 'edge'
    stride: List[int] = field(default_factory=lambda: [1, 1, 1])  # Sampling stride (z, y, x)

    # Axis transposition (empty list = no transpose)
    train_transpose: List[int] = field(
        default_factory=list
    )  # Axis permutation for training data (e.g., [2,1,0] for xyz->zyx)
    val_transpose: List[int] = field(default_factory=list)  # Axis permutation for validation data

    # Dataset statistics (for auto-planning)
    target_spacing: Optional[List[float]] = None  # Target voxel spacing [z, y, x] in mm
    median_shape: Optional[List[int]] = None  # Median dataset shape [D, H, W] in voxels


@dataclass
class ImageTransformConfig:
    """Image transformation configuration (applied to image only)."""

    transform_profile: Optional[str] = None
    normalize: str = "0-1"  # "none", "normal" (z-score), or "0-1" (min-max)
    clip_percentile_low: float = (
        0.0  # Lower percentile for clipping (0.0 = no clip, 0.05 = 5th percentile)
    )
    clip_percentile_high: float = (
        1.0  # Upper percentile for clipping (1.0 = no clip, 0.95 = 95th percentile)
    )


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
class DataTransformProfileConfig:
    """Reusable profile for deterministic data preprocessing transforms."""

    data_transform: Optional[DataTransformConfig] = None
    image_transform: Optional[ImageTransformConfig] = None
    mask_transform: Optional[DataTransformConfig] = None
    nnunet_preprocessing: Optional[NNUNetPreprocessingConfig] = None


@dataclass
class DataloaderConfig:
    """Data loading and sampling configuration."""

    batch_size: int = 4
    patch_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    pin_memory: bool = True
    use_preloaded_cache_train: bool = True  # Preload training volumes into memory
    use_preloaded_cache_val: bool = True  # Preload validation volumes into memory
    persistent_workers: bool = True
    # Caching (MONAI)
    use_cache: bool = False
    cache_rate: float = 1.0
    use_lazy_zarr: bool = False  # Lazy crop-on-read for zarr volumes (no full preload)
    cached_sampling_max_attempts: int = 10  # Retry attempts for foreground-aware sampling
    cached_sampling_foreground_threshold: float = (
        0.0  # Minimum (label > 0) fraction required for training crops.
        # Zero disables foreground sampling.
    )
    cached_sampling_crop_to_nonzero_mask: bool = (
        False  # Bbox approach: constrain crops to intersect the nonzero mask bounding box
    )
    cached_sampling_sample_nonzero_mask: bool = (
        False  # Voxel approach: center crops on random nonzero mask voxels (stronger guarantee)
    )
    reject_sampling: Optional[Dict[str, Any]] = None  # Dict with 'size_thres' and 'p' keys


@dataclass
class DataInputConfig:
    """Dataset input paths and source selection configuration."""

    # Dataset type
    dataset_type: Optional[str] = None  # Type of dataset: None (volume), 'filename', 'tile', etc.

    # 2D data support
    do_2d: bool = False  # Enable 2D data processing (extract 2D slices from 3D volumes)

    # Base path (prepended to image/label/mask/json if set)
    path: str = ""  # Base path for this data split (e.g., "/path/to/dataset/")

    # Paths - Volume-based datasets
    # These can be strings (single file), lists (multiple files), or None
    # Using Any to support both str and List[str] (OmegaConf doesn't support Union of containers)
    image: Any = None  # str, List[str], or None
    label: Any = None  # str, List[str], or None
    mask: Any = None  # str, List[str], or None (Valid region mask)

    # Paths - JSON/filename-based datasets
    json: Optional[str] = None  # JSON file with image/label file lists
    image_key: str = "images"  # Key in JSON for image files
    label_key: str = "masks"  # Key in JSON for label files
    split_ratio: Optional[float] = None  # Auto split ratio (e.g., 0.9 = 90% train, 10% val)

    # Voxel resolution (physical dimensions in nm)
    resolution: Optional[List[float]] = None  # Data resolution [z, y, x] in nm


@dataclass
class FlipConfig:
    """Flip augmentation configuration."""

    enabled: bool = True
    prob: float = 0.5
    spatial_axis: Any = 0


@dataclass
class AffineConfig:
    """Affine augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    rotate_range: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_range: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    shear_range: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RotateConfig:
    """Rotation augmentation configuration."""

    enabled: bool = True
    prob: float = 0.5
    spatial_axes: Tuple[int, int] = (1, 2)


@dataclass
class ElasticConfig:
    """Elastic deformation configuration."""

    enabled: bool = False
    prob: float = 0.0
    sigma_range: Tuple[float, float] = (3.0, 6.0)
    magnitude_range: Tuple[float, float] = (10.0, 20.0)


@dataclass
class IntensityConfig:
    """Intensity augmentation configuration."""

    enabled: bool = True
    gaussian_noise_prob: float = 0.5
    gaussian_noise_std: float = 0.1
    shift_intensity_prob: float = 0.5
    shift_intensity_offset: float = 0.1
    contrast_prob: float = 0.5
    contrast_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class MisalignmentConfig:
    """Misalignment augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    displacement: int = 16
    rotate_ratio: float = 0.0


@dataclass
class MissingSectionConfig:
    """Missing section augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    num_sections: Any = 2
    full_section_prob: float = 0.5
    partial_ratio_range: Tuple[float, float] = (0.25, 0.75)
    fill_value_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class MotionBlurConfig:
    """Out-of-focus blur augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    sections: Any = (1, 3)
    kernel_size: int = 9
    sigma_range: Tuple[float, float] = (1.0, 3.0)
    full_section_prob: float = 0.5
    partial_ratio_range: Tuple[float, float] = (0.25, 0.75)


@dataclass
class CutNoiseConfig:
    """Cut noise augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    length_ratio: Tuple[float, float] = (0.1, 0.4)
    noise_scale: Tuple[float, float] = (0.05, 0.15)


@dataclass
class CutBlurConfig:
    """Cut blur augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    length_ratio: Tuple[float, float] = (0.1, 0.4)
    down_ratio_range: Tuple[float, float] = (2.0, 8.0)
    downsample_z: bool = False


@dataclass
class MissingPartsConfig:
    """Missing parts augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    hole_range: Tuple[float, float] = (8, 32)


@dataclass
class MixupConfig:
    """Mixup augmentation configuration."""

    enabled: bool = False
    alpha: float = 0.2
    prob: float = 0.0
    alpha_range: Tuple[float, float] = (0.1, 0.4)


@dataclass
class CopyPasteConfig:
    """Copy-paste augmentation configuration."""

    enabled: bool = False
    prob: float = 0.0
    max_obj_ratio: float = 0.05
    rotation_angles: Tuple[int, int, int] = (0, 90, 180)
    border: int = 8


@dataclass
class StripeConfig:
    """Stripe artifact augmentation configuration.

    Simulates stripe artifacts often seen in microscopy images.
    """

    enabled: bool = False
    prob: float = 0.0
    num_stripes_range: Tuple[int, int] = (1, 5)
    thickness_range: Tuple[int, int] = (1, 3)
    intensity_range: Tuple[float, float] = (0.7, 1.3)
    angle_range: Tuple[float, float] = (-10.0, 10.0)
    orientation: str = "random"
    mode: str = "multiply"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration.

    Comprehensive augmentation pipeline for training data including geometric,
    intensity, and artifact simulation augmentations. Controlled by presets
    and individual component toggles.

    Presets:
    - "none": No augmentations
    - "some": Basic augmentations (flip, rotate, intensity)
    - "all": All available augmentations
    """

    # Preset for controlling enabled groups: "none", "some", "all"
    preset: str = "some"

    # Mutual exclusion: when True, at most one defect augmentation
    # (misalignment, missing_section, motion_blur) fires per sample.
    # Matches the DeepEM Blend(mutex) behavior.
    defect_mutex: bool = False

    # Individual augmentation blocks (auto-enabled when keys are present)
    flip: FlipConfig = field(default_factory=FlipConfig)
    affine: AffineConfig = field(default_factory=AffineConfig)
    rotate: RotateConfig = field(default_factory=RotateConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)

    # Artifact simulation augmentations
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

    # Train/Val Split (inspired by DeepEM)
    split_enabled: bool = False  # Enable automatic train/val split (default: False)
    split_train_range: List[float] = field(default_factory=lambda: [0.0, 0.8])  # Train: 0-80%
    split_val_range: List[float] = field(default_factory=lambda: [0.8, 1.0])  # Val: 80-100%
    split_axis: int = 0  # Axis to split along (0=Z, 1=Y, 2=X)
    split_pad_val: bool = True  # Pad validation to patch_size if smaller
    split_pad_mode: str = "reflect"  # Padding mode: 'reflect', 'replicate', 'constant'

    # Structured train/val input configuration
    train: DataInputConfig = field(default_factory=DataInputConfig)
    val: DataInputConfig = field(default_factory=DataInputConfig)
    # Optional explicit test split for inference-mode datasets.
    # If left empty, runtime falls back to `val`.
    test: DataInputConfig = field(default_factory=DataInputConfig)

    # Data loading
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    # Data transformation (applied to image/label/mask)
    data_transform: DataTransformConfig = field(default_factory=DataTransformConfig)

    # Image transformation (applied to image only)
    image_transform: ImageTransformConfig = field(default_factory=ImageTransformConfig)

    # nnU-Net-style preprocessing
    nnunet_preprocessing: NNUNetPreprocessingConfig = field(
        default_factory=NNUNetPreprocessingConfig
    )

    # Mask-specific transformation (overrides data_transform for masks when present)
    mask_transform: Optional[DataTransformConfig] = None

    # Multi-channel label transformation (for affinity maps, distance transforms, etc.)
    label_transform: LabelTransformConfig = field(default_factory=LabelTransformConfig)

    # Augmentation configuration (nested under data in YAML)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
