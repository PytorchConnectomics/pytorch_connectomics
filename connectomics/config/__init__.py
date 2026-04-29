"""
Modern Hydra-based configuration system for PyTorch Connectomics.
"""

from .pipeline.config_io import load_config, save_config, validate_config
from .pipeline.dict_utils import as_plain_dict, cfg_get
from .pipeline.stage_resolver import resolve_default_profiles

# New Hydra config system (primary)
from .schema import (
    Config,
    DataConfig,
    DecodingConfig,
    DefaultConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    MonitorConfig,
    OptimizationConfig,
    SystemConfig,
    TestConfig,
    TrainConfig,
    TuneConfig,
)

__all__ = [
    # Hydra config system
    "Config",
    "load_config",
    "save_config",
    "validate_config",
    "resolve_default_profiles",
    # Config utilities
    "as_plain_dict",
    "cfg_get",
    # Structured schema contracts
    "DefaultConfig",
    "TrainConfig",
    "TestConfig",
    "TuneConfig",
    "SystemConfig",
    "ModelConfig",
    "DataConfig",
    "OptimizationConfig",
    "MonitorConfig",
    "InferenceConfig",
    "DecodingConfig",
    "EvaluationConfig",
]
