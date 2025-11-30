"""
PyTorch Lightning integration for Connectomics.

This package provides Lightning-specific components with Hydra/OmegaConf config:
- lit_data.py: Lightning DataModules
- lit_model.py: Lightning Module wrapper
- lit_trainer.py: Lightning Trainer utilities
- callbacks.py: Lightning callbacks (visualization, checkpointing)
- utils.py: CLI argument parsing and config utilities
- lit_config.py: Factory functions for creating Lightning components

Import patterns:
    from connectomics.lightning import ConnectomicsDataModule
    from connectomics.lightning import ConnectomicsModule, create_lightning_module
    from connectomics.lightning import create_trainer, create_datamodule
    from connectomics.lightning import VisualizationCallback, create_callbacks
    from connectomics.lightning import parse_args, setup_config
"""

from .lit_data import *
from .lit_model import *
from .lit_trainer import *
from .callbacks import *
from .utils import *
from .lit_config import *

__all__ = [
    # DataModule
    'ConnectomicsDataModule',
    'create_datamodule',

    # LightningModule
    'ConnectomicsModule',
    'create_lightning_module',

    # Trainer
    'create_trainer',

    # Callbacks
    'VisualizationCallback',
    'NaNDetectionCallback',
    'create_callbacks',

    # Utilities
    'parse_args',
    'setup_config',
    'expand_file_paths',
    'extract_best_score_from_checkpoint',
    'setup_seed_everything',
    'setup_run_directory',
    'cleanup_run_directory',
    'modify_checkpoint_state',
]