"""
Transform modules for PyTorch Connectomics.

This package provides clean entry points for transforms:

Clean import patterns:
    # Option 1: Import directly from main module (recommended)
    from connectomics.transforms import SegToBinaryMaskd, RandMisAlignmentd
    from connectomics.transforms import create_binary_segmentation_pipeline

    # Option 2: Import from specific entry files
    from connectomics.transforms.processor import SegToBinaryMaskd, create_binary_segmentation_pipeline
    from connectomics.transforms.augmentor import RandMisAlignmentd, build_augmentor

Package organization:
    transforms/
    ├── __init__.py          # This file - main entry point
    ├── augmentor.py         # Augmentation entry point
    ├── processor.py         # Processing entry point
    ├── augment/             # Augmentation implementation
    └── process/             # Processing implementation
"""

# Import MONAI-native augmentation transforms
from .augment import (
    build_augmentor, create_inference_transforms,
    RandMisAlignmentd, RandMissingSectiond, RandMissingPartsd,
    RandMotionBlurd, RandCutNoised, RandCutBlurd
)

# Process transforms can be imported separately when needed
# from .process import (...)

__all__ = [
    # === AUGMENTATION (MONAI-native) ===
    'build_augmentor',
    'create_inference_transforms',
    'RandMisAlignmentd',
    'RandMissingSectiond',
    'RandMissingPartsd',
    'RandMotionBlurd',
    'RandCutNoised',
    'RandCutBlurd',
]
