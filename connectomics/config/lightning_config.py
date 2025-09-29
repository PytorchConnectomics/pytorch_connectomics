"""
Configuration utilities for PyTorch Lightning integration.

This module provides helper functions to bridge YACS configurations
with PyTorch Lightning's expected configuration format.
"""

from typing import Dict, Any, Optional
from yacs.config import CfgNode
import os


def adapt_cfg_for_lightning(cfg: CfgNode) -> Dict[str, Any]:
    """
    Convert YACS config to Lightning-compatible dictionary.

    This helps transition from YACS to more modern configuration systems
    while maintaining backward compatibility.

    Args:
        cfg: YACS configuration node

    Returns:
        Dictionary suitable for Lightning hyperparameters
    """
    lightning_cfg = {}

    # Model configuration
    lightning_cfg['model'] = {
        'architecture': cfg.MODEL.ARCHITECTURE,
        'in_channels': cfg.MODEL.IN_PLANES,
        'out_channels': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'blocks': cfg.MODEL.BLOCKS,
        'block_type': cfg.MODEL.BLOCK_TYPE,
        'attention': cfg.MODEL.get('ATTENTION', 'none'),
        'mixed_precision': cfg.MODEL.get('MIXED_PRECESION', False),
    }

    # Training configuration
    lightning_cfg['training'] = {
        'max_epochs': cfg.SOLVER.get('MAX_EPOCHS', None),
        'max_steps': cfg.SOLVER.get('ITERATION_TOTAL', -1),
        'learning_rate': cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.get('WEIGHT_DECAY', 0.0),
        'optimizer': cfg.SOLVER.get('OPTIMIZER', 'Adam'),
        'scheduler': cfg.SOLVER.get('LR_SCHEDULER', 'none'),
    }

    # Data configuration
    lightning_cfg['data'] = {
        'batch_size': cfg.SOLVER.SAMPLES_PER_BATCH,
        'num_workers': cfg.SYSTEM.NUM_CPUS,
        'input_size': cfg.MODEL.INPUT_SIZE,
        'output_size': cfg.MODEL.OUTPUT_SIZE,
    }

    # System configuration
    lightning_cfg['system'] = {
        'num_gpus': cfg.SYSTEM.NUM_GPUS,
        'distributed': cfg.SYSTEM.get('DISTRIBUTED', False),
        'precision': 16 if cfg.MODEL.get('MIXED_PRECESION', False) else 32,
    }

    # Dataset paths
    lightning_cfg['dataset'] = {
        'output_path': cfg.DATASET.OUTPUT_PATH,
        'image_name': cfg.DATASET.IMAGE_NAME,
        'label_name': cfg.DATASET.LABEL_NAME,
        'val_image_name': cfg.DATASET.get('VAL_IMAGE_NAME', None),
    }

    return lightning_cfg


def create_lightning_config_yaml(cfg: CfgNode, output_path: str) -> str:
    """
    Create a Lightning-compatible YAML config from YACS config.

    Args:
        cfg: YACS configuration
        output_path: Where to save the YAML file

    Returns:
        Path to the created YAML file
    """
    import yaml

    lightning_cfg = adapt_cfg_for_lightning(cfg)

    yaml_path = os.path.join(output_path, 'lightning_config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(lightning_cfg, f, default_flow_style=False, indent=2)

    return yaml_path


def validate_lightning_config(cfg: CfgNode) -> bool:
    """
    Validate that the YACS config is compatible with Lightning training.

    Args:
        cfg: YACS configuration to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'MODEL.ARCHITECTURE',
        'MODEL.IN_PLANES',
        'MODEL.OUT_PLANES',
        'SOLVER.BASE_LR',
        'SOLVER.SAMPLES_PER_BATCH',
        'DATASET.OUTPUT_PATH',
        'DATASET.IMAGE_NAME',
        'DATASET.LABEL_NAME',
    ]

    missing_fields = []
    for field in required_fields:
        try:
            # Navigate nested configuration
            obj = cfg
            for key in field.split('.'):
                obj = getattr(obj, key)
        except AttributeError:
            missing_fields.append(field)

    if missing_fields:
        print(f"Missing required configuration fields: {missing_fields}")
        return False

    return True


def print_config_comparison(cfg: CfgNode):
    """
    Print a comparison between YACS and Lightning config formats.

    This helps users understand the mapping during migration.
    """
    print("Configuration Mapping (YACS -> Lightning):")
    print("=" * 50)

    mappings = [
        ("MODEL.ARCHITECTURE", "model.architecture"),
        ("MODEL.IN_PLANES", "model.in_channels"),
        ("MODEL.OUT_PLANES", "model.out_channels"),
        ("SOLVER.BASE_LR", "training.learning_rate"),
        ("SOLVER.SAMPLES_PER_BATCH", "data.batch_size"),
        ("SYSTEM.NUM_GPUS", "system.num_gpus"),
        ("MODEL.MIXED_PRECESION", "system.precision (16 if True else 32)"),
    ]

    for yacs_key, lightning_key in mappings:
        try:
            # Get value from YACS config
            obj = cfg
            for key in yacs_key.split('.'):
                obj = getattr(obj, key)
            value = obj

            print(f"{yacs_key:25} -> {lightning_key:25} = {value}")
        except AttributeError:
            print(f"{yacs_key:25} -> {lightning_key:25} = <not found>")

    print("=" * 50)


# Example Lightning-style config for reference
EXAMPLE_LIGHTNING_CONFIG = {
    'model': {
        'architecture': 'unet_3d',
        'in_channels': 1,
        'out_channels': 1,
        'filters': [28, 36, 48, 64, 80],
        'blocks': [2, 2, 2, 2],
        'block_type': 'residual',
        'attention': 'squeeze_excitation',
        'mixed_precision': True,
    },
    'training': {
        'max_steps': 50000,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'AdamW',
        'scheduler': 'cosine',
    },
    'data': {
        'batch_size': 8,
        'num_workers': 4,
        'input_size': [112, 112, 112],
        'output_size': [112, 112, 112],
    },
    'system': {
        'num_gpus': 1,
        'distributed': False,
        'precision': 16,
    },
    'dataset': {
        'output_path': './outputs',
        'image_name': 'train_im.tif',
        'label_name': 'train_label.tif',
    }
}