"""
Pure MONAI-style augmentor for PyTorch Connectomics.

This module provides a clean, MONAI-native approach to data augmentation,
replacing the hybrid system with pure MONAI transforms composition.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from yacs.config import CfgNode

# MONAI imports for standard transforms
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandAffined, RandZoomd,
    RandGaussianNoised, RandShiftIntensityd, Rand3DElasticd,
    RandGaussianSmoothd, RandBiasFieldd, RandAdjustContrastd,
    LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord,
    RandSpatialCropd, CenterSpatialCropd, SpatialPadd
)

# Import our connectomics-specific MONAI custom transforms
from .monai_transforms import (
    RandMisAlignmentd, RandMissingSectiond, RandMissingPartsd,
    RandMotionBlurd, RandCutNoised, RandCutBlurd
)


def build_augmentor(cfg: CfgNode) -> Dict[str, Compose]:
    """
    Build pure MONAI augmentor for Lightning DataModule.
    """
    # Determine keys for transforms
    keys = ['image', 'label']

    # Base preprocessing transforms (always applied)
    base_transforms = [
        EnsureChannelFirstd(keys=keys, channel_dim=0),  # Specify channel_dim
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0.05, a_max=0.95,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=keys)
    ]

    if not cfg.AUGMENTOR.ENABLED:
        # No augmentation - return base transforms only
        base_compose = Compose(base_transforms)
        return {
            'train': base_compose,
            'val': base_compose,
            'test': base_compose
        }

    # Build training augmentation pipeline
    train_transforms = base_transforms.copy()
    augmentation_transforms = []

    # Add standard MONAI augmentations (adjusted for 3D connectomics data)
    augmentation_transforms.extend([
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),  # Rotate in xy-plane
        RandFlipd(keys=keys, prob=0.5, spatial_axis=[0, 1]),      # Flip in xy-plane
        RandGaussianNoised(keys=['image'], prob=0.2, std=0.05)
    ])

    # Add other standard MONAI transforms if configured
    if hasattr(cfg.AUGMENTOR, 'ELASTIC') and cfg.AUGMENTOR.ELASTIC.ENABLED:
        augmentation_transforms.append(
            Rand3DElasticd(
                keys=keys,
                prob=cfg.AUGMENTOR.ELASTIC.get('PROB', 0.5),
                sigma_range=cfg.AUGMENTOR.ELASTIC.get('SIGMA_RANGE', (1.0, 2.0)),
                magnitude_range=cfg.AUGMENTOR.ELASTIC.get('MAGNITUDE_RANGE', (1.0, 2.0)),
                spatial_size=cfg.AUGMENTOR.ELASTIC.get('SPATIAL_SIZE', None),
            )
        )

    if hasattr(cfg.AUGMENTOR, 'ZOOM') and cfg.AUGMENTOR.ZOOM.ENABLED:
        augmentation_transforms.append(
            RandZoomd(
                keys=keys,
                prob=cfg.AUGMENTOR.ZOOM.get('PROB', 0.5),
                min_zoom=cfg.AUGMENTOR.ZOOM.get('MIN_ZOOM', 0.8),
                max_zoom=cfg.AUGMENTOR.ZOOM.get('MAX_ZOOM', 1.2),
                keep_size=cfg.AUGMENTOR.ZOOM.get('KEEP_SIZE', True),
            )
        )

    if hasattr(cfg.AUGMENTOR, 'INTENSITY') and cfg.AUGMENTOR.INTENSITY.ENABLED:
        augmentation_transforms.extend([
            RandShiftIntensityd(
                keys=['image'],
                prob=cfg.AUGMENTOR.INTENSITY.get('SHIFT_PROB', 0.3),
                offsets=cfg.AUGMENTOR.INTENSITY.get('SHIFT_OFFSET', 0.1)
            ),
            RandAdjustContrastd(
                keys=['image'],
                prob=cfg.AUGMENTOR.INTENSITY.get('CONTRAST_PROB', 0.3),
                gamma=cfg.AUGMENTOR.INTENSITY.get('CONTRAST_RANGE', (0.7, 1.4))
            )
        ])

    # Add connectomics-specific augmentations if enabled
    if hasattr(cfg.AUGMENTOR, 'MISALIGNMENT') and cfg.AUGMENTOR.MISALIGNMENT.ENABLED:
        augmentation_transforms.append(
            RandMisAlignmentd(
                keys=keys,
                prob=cfg.AUGMENTOR.MISALIGNMENT.get('PROB', 0.5),
                displacement=cfg.AUGMENTOR.MISALIGNMENT.get('DISPLACEMENT', 16),
                rotate_ratio=cfg.AUGMENTOR.MISALIGNMENT.get('ROTATE_RATIO', 0.0)
            )
        )

    if hasattr(cfg.AUGMENTOR, 'MISSING_SECTION') and cfg.AUGMENTOR.MISSING_SECTION.ENABLED:
        augmentation_transforms.append(
            RandMissingSectiond(
                keys=keys,
                prob=cfg.AUGMENTOR.MISSING_SECTION.get('PROB', 0.5),
                num_sections=cfg.AUGMENTOR.MISSING_SECTION.get('NUM_SECTIONS', 2)
            )
        )

    if hasattr(cfg.AUGMENTOR, 'MOTION_BLUR') and cfg.AUGMENTOR.MOTION_BLUR.ENABLED:
        augmentation_transforms.append(
            RandMotionBlurd(
                keys=['image'],  # Only apply to images
                prob=cfg.AUGMENTOR.MOTION_BLUR.get('PROB', 0.5),
                sections=cfg.AUGMENTOR.MOTION_BLUR.get('SECTIONS', 2),
                kernel_size=cfg.AUGMENTOR.MOTION_BLUR.get('KERNEL_SIZE', 11)
            )
        )

    if hasattr(cfg.AUGMENTOR, 'CUT_NOISE') and cfg.AUGMENTOR.CUT_NOISE.ENABLED:
        augmentation_transforms.append(
            RandCutNoised(
                keys=['image'],  # Only apply to images
                prob=cfg.AUGMENTOR.CUT_NOISE.get('PROB', 0.5),
                length_ratio=cfg.AUGMENTOR.CUT_NOISE.get('LENGTH_RATIO', 0.25),
                noise_scale=cfg.AUGMENTOR.CUT_NOISE.get('NOISE_SCALE', 0.2)
            )
        )

    if hasattr(cfg.AUGMENTOR, 'MISSING_PARTS') and cfg.AUGMENTOR.MISSING_PARTS.ENABLED:
        augmentation_transforms.append(
            RandMissingPartsd(
                keys=keys,
                prob=cfg.AUGMENTOR.MISSING_PARTS.get('PROB', 0.5),
                hole_range=cfg.AUGMENTOR.MISSING_PARTS.get('HOLE_RANGE', (0.1, 0.3))
            )
        )

    if hasattr(cfg.AUGMENTOR, 'CUT_BLUR') and cfg.AUGMENTOR.CUT_BLUR.ENABLED:
        augmentation_transforms.append(
            RandCutBlurd(
                keys=['image'],  # Only apply to images
                prob=cfg.AUGMENTOR.CUT_BLUR.get('PROB', 0.5),
                length_ratio=cfg.AUGMENTOR.CUT_BLUR.get('LENGTH_RATIO', 0.25),
                down_ratio_range=cfg.AUGMENTOR.CUT_BLUR.get('DOWN_RATIO_RANGE', (2.0, 8.0)),
                downsample_z=cfg.AUGMENTOR.CUT_BLUR.get('DOWNSAMPLE_Z', False)
            )
        )

    # Insert augmentation transforms before the final ToTensord
    train_transforms = base_transforms[:-1] + augmentation_transforms + [base_transforms[-1]]

    # Create transform pipelines
    train_compose = Compose(train_transforms)
    val_test_compose = Compose(base_transforms)

    return {
        'train': train_compose,
        'val': val_test_compose,
        'test': val_test_compose
    }


def create_inference_transforms(cfg: CfgNode) -> Compose:
    """Create transforms for inference (no augmentation)."""
    keys = ['image']

    transforms = [
        EnsureChannelFirstd(keys=keys, channel_dim=0),  # Specify channel_dim
        ScaleIntensityRanged(
            keys=keys,
            a_min=0.05, a_max=0.95,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=keys)
    ]

    return Compose(transforms)


__all__ = ['build_augmentor', 'create_inference_transforms']