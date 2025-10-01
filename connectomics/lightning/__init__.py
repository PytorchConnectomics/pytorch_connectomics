"""
PyTorch Lightning integration for Connectomics.

This package provides Lightning-specific components with Hydra/OmegaConf config:
- lit_data.py: Lightning DataModules
- lit_model.py: Lightning Module wrapper
- lit_trainer.py: Lightning Trainer utilities
- callbacks.py: Lightning callbacks (visualization, checkpointing)

Import patterns:
    from connectomics.lightning import ConnectomicsDataModule
    from connectomics.lightning import ConnectomicsModule, create_lightning_module
    from connectomics.lightning import create_trainer, ConnectomicsTrainer
    from connectomics.lightning import VisualizationCallback, create_callbacks
"""

from .lit_data import *
from .lit_model import *
from .lit_trainer import *
from .callbacks import *

__all__ = [
    # DataModule
    'ConnectomicsDataModule',

    # LightningModule
    'ConnectomicsModule',
    'create_lightning_module',

    # Trainer
    'create_trainer',
    'ConnectomicsTrainer',

    # Callbacks
    'VisualizationCallback',
    'create_callbacks',
]