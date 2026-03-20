"""Data package for PyTorch Connectomics.

This package provides:
- Dataset classes (datasets/)
- Data augmentation (augmentation/)
- Data processing transforms (processing/)
- I/O utilities (io/)
- DataModules for PyTorch Lightning (see training/lightning/data.py)

Recommended imports:
    from connectomics.data.datasets import CachedVolumeDataset
    from connectomics.data.augmentation import RandMisAlignmentd, build_train_transforms
    from connectomics.data.processing import MultiTaskLabelTransformd, create_label_transform_pipeline
"""

from . import augmentation, datasets, io, processing

__all__ = [
    "augmentation",
    "datasets",
    "io",
    "processing",
]
