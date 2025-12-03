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
    from connectomics.data.process import MultiTaskLabelTransformd, create_label_transform_pipeline
"""

# Make submodules available
from . import augment
from . import process

__all__ = [
    # Submodules
    "augment",
    "process",
]
