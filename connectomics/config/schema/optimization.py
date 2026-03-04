from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


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
    """Learning rate scheduler configuration.

    Follows the same extensible pattern as ModelArchConfig:
    - ``name`` selects the scheduler implementation
    - ``params`` carries scheduler-specific arguments
    - typed shared fields remain available for discoverability
    """

    profile: Optional[str] = None
    name: str = "CosineAnnealingLR"
    params: Dict[str, Any] = field(default_factory=dict)

    # Shared scheduler controls
    monitor: Optional[str] = None
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    eps: float = 1e-8

    # Warmup controls
    warmup_epochs: int = 10
    warmup_start_lr: float = 0.0001

    # Minimum LR floor
    min_lr: float = 0.00001

    # Scheduler interval control
    interval: str = "epoch"  # "epoch" or "step" - controls when scheduler steps
    frequency: int = 1  # How often to step the scheduler


@dataclass
class EMAConfig:
    """Exponential moving average (EMA) configuration."""

    enabled: bool = False
    decay: float = 0.999
    warmup_steps: int = 0
    validate_with_ema: bool = True
    device: Optional[str] = None
    copy_buffers: bool = True


@dataclass
class OptimizationConfig:
    """Training optimization configuration.

    Comprehensive training configuration including optimizer, scheduler,
    precision settings, and training loop parameters.

    Key Features:
    - Multiple optimizer support (AdamW, SGD, etc.)
    - Learning rate scheduling with warmup
    - Mixed precision training support
    - Gradient accumulation and clipping
    - Flexible validation scheduling
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)

    # Training loop
    max_epochs: int = 200
    n_steps_per_epoch: int = -1  # Optimizer steps per epoch (-1 = auto from dataset size)
    val_steps_per_epoch: Optional[int] = None  # Validation steps per epoch (auto-calculated if None)
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"  # "32", "16-mixed", "bf16-mixed"

    # Validation scheduling
    val_check_interval: Union[int, float] = 1.0  # Validate every N epochs (legacy key name)

    # Logging
    log_every_n_steps: int = 100
