"""
Data package for PyTorch Connectomics.

This package provides:
- Dataset classes (dataset/)
- Data augmentation (augment/)
- Data processing transforms (process/)
- I/O utilities (io/)
- DataModules for PyTorch Lightning (datamodules.py)

Recommended imports:
    from connectomics.data.dataset import MonaiVolumeDataset
    from connectomics.data.augment import RandMisAlignmentd, build_train_transforms
    from connectomics.data.process import SegToBinaryMaskd, create_binary_segmentation_pipeline
"""

from .dataset.dataset_base import *
from .dataset import *
from .io_utils import *

# Make submodules available
from . import augment
from . import process

__all__ = [
    # Submodules
    'augment',
    'process',
]