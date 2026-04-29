from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set

from ...runtime.torch_safe_globals import register_torch_safe_globals
from .data import DataConfig
from .decoding import DecodingConfig
from .evaluation import EvaluationConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .monitor import MonitorConfig
from .optimization import OptimizationConfig
from .stages import DefaultConfig, TestConfig, TrainConfig, TuneConfig
from .system import SystemConfig


@dataclass
class MergeContext:
    """Internal merge bookkeeping for YAML explicit path tracking."""

    explicit_field_paths: Set[str] = field(default_factory=set)


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
        inference: Global inference settings (sliding window, TTA, prediction I/O)
        decoding: Decoding pipeline settings
        evaluation: Evaluation and metric settings
        test: Test-specific configuration (test data paths, decoding, evaluation)
        tune: Parameter tuning configuration (tuning data paths, optimization settings)

    Attributes:
        experiment_name: Name of the experiment for organization
        description: Optional description of the experiment
        system: System configuration for hardware and parallelization
        model: Model architecture and loss configuration
        default: Default-stage profile registry (system/data-transform/inference)
        train: Train-stage profile selectors
        data: Training data loading and preprocessing configuration
        optimization: Training optimization configuration
        monitor: Monitoring and logging configuration
        inference: Global inference settings (no data paths)
        decoding: Decoding pipeline settings
        evaluation: Evaluation and metric settings
        test: Test-specific configuration (includes test.data as DataConfig)
        tune: Parameter tuning configuration (includes tune.data as DataConfig)
    """

    # Metadata
    experiment_name: str = "connectomics_experiment"
    description: str = ""

    # Core components
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    default: DefaultConfig = field(default_factory=DefaultConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Optional: Test-specific configuration (test data paths, decoding, evaluation)
    test: Optional[TestConfig] = None

    # Optional: Parameter tuning configuration (tuning data paths, optimization settings)
    tune: Optional[TuneConfig] = None

    # Internal runtime merge context; excluded from structured config serialization.
    _merge_context: MergeContext = field(
        default_factory=MergeContext,
        init=False,
        repr=False,
        compare=False,
        metadata={"omegaconf_ignore": True},
    )


register_torch_safe_globals()
