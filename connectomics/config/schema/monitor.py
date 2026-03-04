from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""

    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    save_top_k: int = 3
    save_last: bool = True
    dirpath: Optional[str] = None
    filename: Optional[str] = None
    checkpoint_filename: str = "{epoch:03d}-{train_loss_total_epoch:.4f}"
    save_every_n_epochs: int = 1
    use_timestamp: bool = True


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 100
    min_delta: float = 0.0001
    check_finite: bool = True
    threshold: Optional[float] = 0.02
    divergence_threshold: Optional[float] = 2.0


@dataclass
class ScalarLoggingConfig:
    """Scalar logging configuration."""

    enabled: bool = True
    interval: str = "step"  # "step" or "epoch"
    loss: Optional[List[str]] = None
    loss_every_n_steps: int = 100
    val_check_interval: Optional[float] = None  # Legacy alias (use optimization.val_check_interval)
    benchmark: Optional[bool] = None


@dataclass
class ImageLoggingConfig:
    """Image logging configuration."""

    enabled: bool = False
    interval: str = "epoch"  # "step" or "epoch"
    max_images: int = 4
    num_slices: int = 4
    slice_sampling: str = "uniform"  # "uniform" or "consecutive"
    log_every_n_epochs: int = 1
    channels: Optional[Tuple[int, ...]] = None
    channel_mode: Optional[str] = None
    selected_channels: Optional[List[int]] = None


@dataclass
class PredictionSavingConfig:
    """Prediction saving configuration during validation."""

    enabled: bool = False
    interval: str = "epoch"
    max_samples: int = 2
    output_subdir: str = "val_predictions"


@dataclass
class LoggingConfig:
    """Logging behavior configuration."""

    scalar: ScalarLoggingConfig = field(default_factory=ScalarLoggingConfig)
    images: ImageLoggingConfig = field(default_factory=ImageLoggingConfig)
    prediction_saving: PredictionSavingConfig = field(default_factory=PredictionSavingConfig)


@dataclass
class NaNDetectionConfig:
    """NaN/Inf detection and debugging configuration."""

    enabled: bool = True
    debug_on_nan: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases backend configuration."""

    use_wandb: bool = False
    project: str = "connectomics"
    entity: Optional[str] = None
    tags: Optional[Tuple[str, ...]] = None
    name: Optional[str] = None


@dataclass
class MonitorConfig:
    """Monitoring and logging configuration.

    Configuration for experiment monitoring including checkpointing,
    early stopping, TensorBoard logging, and optional Weights & Biases integration.
    """

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    nan_detection: NaNDetectionConfig = field(default_factory=NaNDetectionConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    detect_anomaly: bool = False
