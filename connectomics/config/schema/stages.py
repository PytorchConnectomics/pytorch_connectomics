from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data import DataConfig, DataTransformProfileConfig
from .inference import InferenceConfig
from .system import SystemConfig


@dataclass
class SharedConfig:
    """Shared defaults and profiles reused across train/test/tune stages."""

    # Stage selector/override
    system: SystemConfig = field(default_factory=SystemConfig)

    # Profile selectors
    pipeline_profile: Optional[str] = None
    arch_profile: Optional[str] = None
    data_transform_profile: Optional[str] = None
    augmentation_profile: Optional[str] = None
    dataloader_profile: Optional[str] = None
    optimizer_profile: Optional[str] = None
    loss_profile: Optional[str] = None
    label_profile: Optional[str] = None
    decoding_profile: Optional[str] = None
    activation_profile: Optional[str] = None

    # Shared partial overrides for runtime sections (merged as defaults for active mode)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    monitor: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)

    # Profile registries
    system_profiles: Dict[str, SystemConfig] = field(default_factory=dict)
    data_transform_profiles: Dict[str, DataTransformProfileConfig] = field(default_factory=dict)
    dataloader_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimizer_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inference_profiles: Dict[str, InferenceConfig] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Train-stage profile selectors."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    monitor: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestConfig:
    """Test-specific configuration (data paths, decoding, evaluation)."""

    system: SystemConfig = field(default_factory=SystemConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    inference: Dict[str, Any] = field(default_factory=dict)

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
    model: Dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)

    inference: Dict[str, Any] = field(default_factory=dict)
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
