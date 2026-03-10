"""
Data package for PyTorch Connectomics.

This package provides:
- Dataset classes (dataset/)
- Data augmentation (augment/)
- Data processing transforms (process/)
- I/O utilities (io/)
- DataModules for PyTorch Lightning (see training/lightning/data.py)

Recommended imports:
    from connectomics.data.dataset import CachedVolumeDataset
    from connectomics.data.augment import RandMisAlignmentd, build_train_transforms
    from connectomics.data.process import MultiTaskLabelTransformd, create_label_transform_pipeline
"""

# Make submodules available
from . import augment, dataset, io, process

__all__ = [
    # Submodules
    "augment",
    "dataset",
    "io",
    "process",
]
