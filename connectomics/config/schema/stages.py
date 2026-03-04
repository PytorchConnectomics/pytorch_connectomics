from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data import DataConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .monitor import MonitorConfig
from .optimization import OptimizationConfig
from .system import SystemConfig


@dataclass
class DefaultConfig:
    """Default stage settings reused across train/test/tune stages."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass
class TrainConfig:
    """Train-stage profile selectors."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


@dataclass
class TestConfig:
    """Test-specific configuration (data paths, decoding, evaluation)."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    output_path: Optional[str] = None
    cache_suffix: str = "_prediction.h5"


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

    enabled: bool = False
    function_name: str = "decode_semantic"
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

    enabled: bool = False
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: TuneOutputConfig = field(default_factory=TuneOutputConfig)
    logging: Dict[str, Any] = field(default_factory=lambda: {"verbose": True})
    parameter_space: ParameterSpaceConfig = field(default_factory=ParameterSpaceConfig)

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
