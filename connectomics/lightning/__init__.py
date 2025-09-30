"""
PyTorch Lightning integration for Connectomics.

This package provides Lightning-specific components:
- lit_data.py: Lightning DataModules
- lit_model.py: Lightning Module wrapper
- lit_trainer.py: Lightning Trainer utilities

Import patterns:
    from connectomics.lightning import ConnectomicsDataModule
    from connectomics.lightning.lit_model import ConnectomicsModule
    from connectomics.lightning.lit_trainer import create_trainer
"""

from .lit_data import *
from .lit_model import *
from .lit_trainer import *

__all__ = [
    'ConnectomicsDataModule',
    'ConnectomicsModule', 
    'create_trainer',
]