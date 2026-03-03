from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


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


@dataclass
class EMAConfig:
    """Exponential moving average (EMA) configuration."""

    enabled: bool = False
    decay: float = 0.999
    validate_with_ema: bool = True


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
    iter_num: Union[int, float] = 20000  # Absolute iterations or epochs if float<=1000 heuristic in data factory
    val_iter_num: Optional[int] = None  # Validation iterations per epoch (auto-calculated if None)
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"  # "32", "16-mixed", "bf16-mixed"

    # Validation scheduling
    val_check_interval: Union[int, float] = 1.0  # Validate every N epochs (legacy key name)

    # Logging
    log_every_n_steps: int = 100
