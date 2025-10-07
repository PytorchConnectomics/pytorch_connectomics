"""
Utility functions for Hydra configuration system.

Provides helpers for loading, saving, validating, and manipulating configs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from omegaconf import OmegaConf, DictConfig

from .hydra_config import Config


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with defaults merged
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    yaml_conf = OmegaConf.load(config_path)

    # Merge with structured config defaults
    default_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(default_conf, yaml_conf)

    # Convert to dataclass instance
    cfg = OmegaConf.to_object(merged)

    # Apply top-level overrides if specified (>= 0)
    if cfg.num_gpus >= 0:
        cfg.system.num_gpus = cfg.num_gpus
    if cfg.num_cpus >= 0:
        cfg.system.num_cpus = cfg.num_cpus
    if cfg.batch_size >= 0:
        cfg.data.batch_size = cfg.batch_size
    if cfg.num_workers >= 0:
        cfg.data.num_workers = cfg.num_workers

    return cfg


def save_config(cfg: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        cfg: Config object to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    omega_conf = OmegaConf.structured(cfg)
    OmegaConf.save(omega_conf, save_path)


def merge_configs(
    base_cfg: Config,
    *override_cfgs: Union[Config, Dict, str, Path]
) -> Config:
    """
    Merge multiple configurations together.
    
    Args:
        base_cfg: Base configuration
        *override_cfgs: One or more override configs (Config, dict, or path to YAML)
        
    Returns:
        Merged Config object
    """
    result = OmegaConf.structured(base_cfg)
    
    for override_cfg in override_cfgs:
        if isinstance(override_cfg, (str, Path)):
            override_omega = OmegaConf.load(override_cfg)
        elif isinstance(override_cfg, Config):
            override_omega = OmegaConf.structured(override_cfg)
        elif isinstance(override_cfg, (dict, DictConfig)):
            override_omega = OmegaConf.create(override_cfg)
        else:
            raise TypeError(f"Unsupported config type: {type(override_cfg)}")
        
        result = OmegaConf.merge(result, override_omega)
    
    return OmegaConf.to_object(result)


def update_from_cli(cfg: Config, overrides: List[str]) -> Config:
    """
    Update config from command-line overrides.
    
    Supports dot notation: ['data.batch_size=4', 'model.architecture=unet3d']
    
    Args:
        cfg: Base Config object
        overrides: List of 'key=value' strings
        
    Returns:
        Updated Config object
    """
    cfg_omega = OmegaConf.structured(cfg)
    cli_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg_omega, cli_conf)
    return OmegaConf.to_object(merged)


def to_dict(cfg: Config, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert Config to dictionary.
    
    Args:
        cfg: Config object
        resolve: Whether to resolve variable interpolations
        
    Returns:
        Dictionary representation
    """
    omega_conf = OmegaConf.structured(cfg)
    return OmegaConf.to_container(omega_conf, resolve=resolve)


def from_dict(d: Dict[str, Any]) -> Config:
    """
    Create Config from dictionary.
    
    Args:
        d: Dictionary with configuration values
        
    Returns:
        Config object
    """
    default_conf = OmegaConf.structured(Config)
    dict_conf = OmegaConf.create(d)
    merged = OmegaConf.merge(default_conf, dict_conf)
    return OmegaConf.to_object(merged)


def print_config(cfg: Config, resolve: bool = True) -> None:
    """
    Pretty print configuration.
    
    Args:
        cfg: Config to print
        resolve: Whether to resolve variable interpolations
    """
    omega_conf = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(omega_conf, resolve=resolve))


def validate_config(cfg: Config) -> None:
    """
    Validate configuration values.
    
    Args:
        cfg: Config object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Model validation
    if cfg.model.in_channels <= 0:
        raise ValueError("model.in_channels must be positive")
    if cfg.model.out_channels <= 0:
        raise ValueError("model.out_channels must be positive")
    if len(cfg.model.input_size) != 3:
        raise ValueError("model.input_size must be 3D")
    
    # Data validation
    if cfg.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if cfg.data.num_workers < 0:
        raise ValueError("data.num_workers must be non-negative")
    if len(cfg.data.patch_size) != 3:
        raise ValueError("data.patch_size must be 3D")
    
    # Optimizer validation
    if cfg.optimizer.lr <= 0:
        raise ValueError("optimizer.lr must be positive")
    if cfg.optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    
    # Training validation
    if cfg.training.max_epochs <= 0 and cfg.training.max_steps is None:
        raise ValueError("Either training.max_epochs or training.max_steps must be set")
    if cfg.training.gradient_clip_val < 0:
        raise ValueError("training.gradient_clip_val must be non-negative")
    if cfg.training.accumulate_grad_batches <= 0:
        raise ValueError("training.accumulate_grad_batches must be positive")
    
    # Loss validation
    if len(cfg.model.loss_functions) != len(cfg.model.loss_weights):
        raise ValueError("loss_functions and loss_weights must have same length")
    if any(w < 0 for w in cfg.model.loss_weights):
        raise ValueError("loss_weights must be non-negative")


def get_config_hash(cfg: Config) -> str:
    """
    Generate a hash string for the configuration.
    
    Useful for experiment tracking and reproducibility.
    
    Args:
        cfg: Config object
        
    Returns:
        Hash string
    """
    import hashlib
    omega_conf = OmegaConf.structured(cfg)
    yaml_str = OmegaConf.to_yaml(omega_conf, resolve=True)
    return hashlib.md5(yaml_str.encode()).hexdigest()[:8]


def create_experiment_name(cfg: Config) -> str:
    """
    Create a descriptive experiment name from config.
    
    Args:
        cfg: Config object
        
    Returns:
        Experiment name string
    """
    parts = [
        cfg.model.architecture,
        f"bs{cfg.data.batch_size}",
        f"lr{cfg.optimizer.lr:.0e}",
        get_config_hash(cfg)
    ]
    return "_".join(parts)


__all__ = [
    'load_config',
    'save_config',
    'merge_configs',
    'update_from_cli',
    'to_dict',
    'from_dict',
    'print_config',
    'validate_config',
    'get_config_hash',
    'create_experiment_name',
]