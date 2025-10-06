"""
Modern Hydra-based configuration system for PyTorch Connectomics.

Uses dataclasses and OmegaConf for type-safe, composable configurations
that integrate seamlessly with PyTorch Lightning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union
# Note: MISSING can be imported from omegaconf if needed for required fields


@dataclass
class SystemConfig:
    """System configuration for hardware and parallelization."""
    num_gpus: int = 1
    num_cpus: int = 4
    seed: Optional[int] = None

    # Auto-planning
    auto_plan: bool = False  # Enable automatic hyperparameter planning based on GPU
    print_auto_plan: bool = True  # Print auto-planning results


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Architecture
    architecture: str = 'monai_basic_unet3d'

    # I/O dimensions
    input_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    output_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 1
    out_channels: int = 1

    # Architecture-specific parameters
    filters: Tuple[int, ...] = (32, 64, 128, 256, 512)
    dropout: float = 0.0
    norm: str = "batch"
    activation: str = "relu"

    # UNet-specific parameters (MONAI UNet)
    num_res_units: int = 2         # Number of residual units per block
    kernel_size: int = 3            # Convolution kernel size

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
    mednext_block_counts: List[int] = field(default_factory=lambda: [2,2,2,2,2,2,2,2,2])
    mednext_checkpoint_style: Optional[str] = None  # None or 'outside_block'
    mednext_norm: str = "group"  # 'group' or 'layer'
    mednext_dim: str = "3d"  # '2d' or '3d'
    mednext_grn: bool = False  # Global Response Normalization

    # Deep supervision (supported by MedNeXt and some MONAI models)
    deep_supervision: bool = False

    # Loss configuration
    loss_functions: List[str] = field(default_factory=lambda: ["DiceLoss", "BCEWithLogitsLoss"])
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    loss_kwargs: List[dict] = field(default_factory=lambda: [{}, {}])  # Per-loss kwargs


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    # Paths
    train_image: str = "datasets/train_image.h5"
    train_label: str = "datasets/train_label.h5"
    val_image: Optional[str] = None
    val_label: Optional[str] = None
    test_image: Optional[str] = None
    test_label: Optional[str] = None

    # Data properties
    patch_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    pad_size: List[int] = field(default_factory=lambda: [8, 32, 32])
    stride: List[int] = field(default_factory=lambda: [1, 1, 1])  # Sampling stride (z, y, x)

    # Dataset statistics (for auto-planning)
    target_spacing: Optional[List[float]] = None  # Target voxel spacing [z, y, x] in mm
    median_shape: Optional[List[int]] = None  # Median dataset shape [D, H, W] in voxels

    # Train/Val Split (inspired by DeepEM)
    # If enabled, splits single volume into train/val regions
    split_enabled: bool = False  # Enable automatic train/val split (default: False)
    split_train_range: List[float] = field(default_factory=lambda: [0.0, 0.8])  # Train: 0-80%
    split_val_range: List[float] = field(default_factory=lambda: [0.8, 1.0])    # Val: 80-100%
    split_axis: int = 0  # Axis to split along (0=Z, 1=Y, 2=X)
    split_pad_val: bool = True  # Pad validation to patch_size if smaller
    split_pad_mode: str = 'reflect'  # Padding mode: 'reflect', 'replicate', 'constant'

    # Data loading
    batch_size: int = 4
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

    # Caching (MONAI)
    use_cache: bool = False
    cache_rate: float = 1.0

    # Normalization
    normalize: bool = True
    mean: float = 0.5
    std: float = 0.5
    normalize_labels: bool = True  # Convert labels to 0-1 range (default: True)

    # Sampling (for volumetric datasets)
    iter_num: int = 500  # Number of random crops per epoch (default: 500)
    use_preloaded_cache: bool = True  # Preload volumes into memory for fast random cropping (default: True)


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
    
    # CosineAnnealing-specific
    t_max: Optional[int] = None
    
    # ReduceLROnPlateau-specific
    patience: int = 10
    factor: float = 0.5
    
    # StepLR-specific
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    max_steps: Optional[int] = None
    
    # Gradient control
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Precision
    precision: str = "32"  # "32", "16", "bf16", "16-mixed", "bf16-mixed"
    
    # Validation
    val_check_interval: Union[int, float] = 1.0  # Float (1.0) = every epoch, int = every N steps
    
    # Logging
    log_every_n_steps: int = 50
    
    # Performance
    deterministic: bool = False
    benchmark: bool = True
    detect_anomaly: bool = False


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""
    save_top_k: int = 1
    monitor: str = "val/loss"
    mode: str = "min"
    save_last: bool = True
    save_every_n_epochs: int = 10
    dirpath: str = "checkpoints/"
    filename: str = "epoch={epoch:03d}-val_loss={val/loss:.4f}"


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    monitor: str = "val/loss"
    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0


# Augmentation configurations
@dataclass
class FlipConfig:
    """Random flip augmentation."""
    enabled: bool = True
    prob: float = 0.5
    spatial_axis: Optional[List[int]] = None  # None = all axes


@dataclass
class RotateConfig:
    """Random rotation augmentation."""
    enabled: bool = True
    prob: float = 0.5
    max_angle: float = 90.0


@dataclass
class ElasticConfig:
    """Elastic deformation augmentation."""
    enabled: bool = True
    prob: float = 0.3
    sigma_range: Tuple[float, float] = (5.0, 8.0)
    magnitude_range: Tuple[float, float] = (50.0, 150.0)


@dataclass
class IntensityConfig:
    """Intensity augmentation."""
    enabled: bool = True
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: float = 0.05
    shift_intensity_prob: float = 0.3
    shift_intensity_offset: float = 0.1
    contrast_prob: float = 0.3
    contrast_range: Tuple[float, float] = (0.7, 1.4)


@dataclass
class MisalignmentConfig:
    """Misalignment augmentation for EM data."""
    enabled: bool = True
    prob: float = 0.5
    displacement: int = 16
    rotate_ratio: float = 0.0


@dataclass
class MissingSectionConfig:
    """Missing section augmentation for EM data."""
    enabled: bool = True
    prob: float = 0.3
    num_sections: int = 2


@dataclass
class MotionBlurConfig:
    """Motion blur augmentation for EM data."""
    enabled: bool = True
    prob: float = 0.3
    sections: int = 2
    kernel_size: int = 11


@dataclass
class CutNoiseConfig:
    """CutNoise augmentation."""
    enabled: bool = False
    prob: float = 0.5
    length_ratio: float = 0.25
    noise_scale: float = 0.2


@dataclass
class CutBlurConfig:
    """CutBlur augmentation."""
    enabled: bool = True
    prob: float = 0.3
    length_ratio: float = 0.25
    down_ratio_range: Tuple[float, float] = (2.0, 8.0)
    downsample_z: bool = False


@dataclass
class MissingPartsConfig:
    """Missing parts augmentation."""
    enabled: bool = True
    prob: float = 0.5
    hole_range: Tuple[float, float] = (0.1, 0.3)


@dataclass
class MixupConfig:
    """Mixup augmentation."""
    enabled: bool = True
    prob: float = 0.5
    alpha_range: Tuple[float, float] = (0.7, 0.9)


@dataclass
class CopyPasteConfig:
    """Copy-Paste augmentation."""
    enabled: bool = True
    prob: float = 0.5
    max_obj_ratio: float = 0.7
    rotation_angles: List[int] = field(default_factory=lambda: list(range(30, 360, 30)))
    border: int = 3


@dataclass
class AugmentationConfig:
    """Complete augmentation configuration."""
    enabled: bool = False
    
    # Standard augmentations
    flip: FlipConfig = field(default_factory=FlipConfig)
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
    
    # Advanced augmentations
    mixup: MixupConfig = field(default_factory=MixupConfig)
    copy_paste: CopyPasteConfig = field(default_factory=CopyPasteConfig)


@dataclass
class VisualizationConfig:
    """Visualization configuration for TensorBoard."""
    enabled: bool = True
    max_images: int = 4
    num_slices: int = 8


@dataclass
class InferenceConfig:
    """Inference configuration."""
    test_image: Optional[str] = None
    test_label: Optional[str] = None
    output_path: str = "results/"
    stride: List[int] = field(default_factory=lambda: [64, 64, 64])
    overlap: float = 0.5
    test_time_augmentation: bool = False
    tta_num: int = 4


@dataclass
class Config:
    """Main configuration for PyTorch Connectomics."""

    # Core components
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Metadata
    experiment_name: str = "connectomics_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)


__all__ = ['Config']