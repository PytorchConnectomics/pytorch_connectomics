from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

from .data import DataConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .monitor import MonitorConfig
from .optimization import OptimizationConfig
from .stages import SharedConfig, TestConfig, TrainConfig, TuneConfig
from .system import SystemConfig


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
        shared: Shared profile registry (system/data-transform/inference)
        train: Train-stage profile selectors
        data: Training data loading and preprocessing configuration
        optimization: Training optimization configuration
        monitor: Monitoring and logging configuration
        inference: Shared inference settings (no data paths)
        test: Test-specific configuration (includes test.data as DataConfig)
        tune: Parameter tuning configuration (includes tune.data as DataConfig)
    """

    # Metadata
    experiment_name: str = "connectomics_experiment"
    description: str = ""

    # Core components
    system: SystemConfig = field(
        default_factory=lambda: SystemConfig(num_gpus=1, num_workers=8, seed=42)
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    shared: SharedConfig = field(default_factory=SharedConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Optional: Test-specific configuration (test data paths, decoding, evaluation)
    test: Optional[TestConfig] = None

    # Optional: Parameter tuning configuration (tuning data paths, optimization settings)
    tune: Optional[TuneConfig] = None


def _register_torch_safe_globals() -> None:
    """Register schema dataclasses for torch 2.6+ weights_only checkpoint loading."""
    try:
        import torch

        if not (
            hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals")
        ):
            return

        safe_dataclasses = []
        schema_modules = [
            "connectomics.config.schema.system",
            "connectomics.config.schema.model",
            "connectomics.config.schema.model_monai",
            "connectomics.config.schema.model_mednext",
            "connectomics.config.schema.model_rsunet",
            "connectomics.config.schema.model_nnunet",
            "connectomics.config.schema.data",
            "connectomics.config.schema.optimization",
            "connectomics.config.schema.monitor",
            "connectomics.config.schema.inference",
            "connectomics.config.schema.stages",
            "connectomics.config.schema.root",
        ]
        for module_name in schema_modules:
            module = importlib.import_module(module_name)
            safe_dataclasses.extend(
                obj
                for obj in module.__dict__.values()
                if inspect.isclass(obj) and is_dataclass(obj)
            )

        # De-duplicate while preserving order.
        deduped = list(dict.fromkeys(safe_dataclasses))
        torch.serialization.add_safe_globals(deduped)
    except Exception:
        # Best-effort registration; ignore if torch not available at import time.
        pass


# Register safe globals on import.
_register_torch_safe_globals()

