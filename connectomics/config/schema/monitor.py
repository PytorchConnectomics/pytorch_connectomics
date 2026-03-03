from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""

    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    save_top_k: int = 3
    save_last: bool = True
    dirpath: Optional[str] = None
    filename: Optional[str] = None
    every_n_epochs: Optional[int] = 1


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 20
    min_delta: float = 0.0


@dataclass
class ScalarLoggingConfig:
    """Scalar logging configuration."""

    enabled: bool = True
    interval: str = "step"  # "step" or "epoch"


@dataclass
class ImageLoggingConfig:
    """Image logging configuration."""

    enabled: bool = False
    interval: str = "epoch"  # "step" or "epoch"
    max_images: int = 4
    channels: Optional[Tuple[int, ...]] = None


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
    image: ImageLoggingConfig = field(default_factory=ImageLoggingConfig)
    prediction_saving: PredictionSavingConfig = field(default_factory=PredictionSavingConfig)


@dataclass
class NaNDetectionConfig:
    """NaN/Inf detection and debugging configuration."""

    enable_nan_detection: bool = True
    debug_on_nan: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases backend configuration."""

    enabled: bool = False
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
