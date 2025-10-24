"""
Build MONAI transform pipelines from Hydra configuration.

Modern replacement for monai_compose.py that works with the new Hydra config system.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandAffined, RandZoomd,
    RandGaussianNoised, RandShiftIntensityd, Rand3DElasticd,
    RandGaussianSmoothd, RandAdjustContrastd, RandSpatialCropd,
    ScaleIntensityRanged, ToTensord, CenterSpatialCropd, SpatialPadd,
    Resized
)

# Import custom loader for HDF5/TIFF volumes
from connectomics.data.dataset.dataset_volume import LoadVolumed

from .monai_transforms import (
    RandMisAlignmentd, RandMissingSectiond, RandMissingPartsd,
    RandMotionBlurd, RandCutNoised, RandCutBlurd,
    RandMixupd, RandCopyPasted, NormalizeLabelsd,
    SmartNormalizeIntensityd
)
from ...config.hydra_config import Config, AugmentationConfig, LabelTransformConfig


def build_train_transforms(cfg: Config, keys: list[str] = None, skip_loading: bool = False) -> Compose:
    """
    Build training transforms from Hydra config.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: ['image', 'label'] or ['image', 'label', 'mask'] if masks are used)
        skip_loading: Skip LoadVolumed (for pre-cached datasets)

    Returns:
        Composed MONAI transforms
    """
    if keys is None:
        # Auto-detect keys based on config
        keys = ['image', 'label']
        # Add mask to keys if it's specified in the config
        if hasattr(cfg.data, 'train_mask') and cfg.data.train_mask is not None:
            keys.append('mask')

    transforms = []

    # Load images first (unless using pre-cached dataset)
    if not skip_loading:
        # Get transpose axes for training data
        train_transpose = cfg.data.train_transpose if cfg.data.train_transpose else []
        transforms.append(LoadVolumed(keys=keys, transpose_axes=train_transpose if train_transpose else None))

    # Apply volumetric split if enabled
    if cfg.data.split_enabled:
        from connectomics.data.utils import ApplyVolumetricSplitd
        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
    # NOTE: Resize uses scale factors, but target size must be computed based on input
    # This is handled by a custom transform that computes target size dynamically
    if hasattr(cfg.data.image_transform, 'resize') and cfg.data.image_transform.resize is not None:
        resize_factors = cfg.data.image_transform.resize
        if resize_factors:
            from .monai_transforms import ResizeByFactord
            # Use bilinear for images, nearest for labels/masks
            transforms.append(
                ResizeByFactord(
                    keys=['image'],
                    scale_factors=resize_factors,  # Scale factors for each spatial dimension
                    mode='bilinear',               # Bilinear interpolation for images
                    align_corners=True
                )
            )
            # Resize labels and masks with nearest-neighbor to preserve integer values
            label_mask_keys = [k for k in keys if k in ['label', 'mask']]
            if label_mask_keys:
                transforms.append(
                    ResizeByFactord(
                        keys=label_mask_keys,
                        scale_factors=resize_factors,
                        mode='nearest',                 # Nearest neighbor for labels/masks
                        align_corners=None              # Not used for nearest mode
                    )
                )

    # Ensure target patch size is respected (unless using pre-cached dataset)
    if not skip_loading:
        patch_size = tuple(cfg.data.patch_size) if hasattr(cfg.data, 'patch_size') else None
        if patch_size and all(size > 0 for size in patch_size):
            # Pad smaller volumes so random crops always succeed
            transforms.append(
                SpatialPadd(
                    keys=keys,
                    spatial_size=patch_size,
                    constant_values=0.0,
                )
            )
            transforms.append(
                RandSpatialCropd(
                    keys=keys,
                    roi_size=patch_size,
                    random_center=True,
                    random_size=False,
                )
            )

    # Normalization - use smart normalization
    if cfg.data.image_transform.normalize != "none":
        transforms.append(
            SmartNormalizeIntensityd(
                keys=['image'],
                mode=cfg.data.image_transform.normalize,
                clip_percentile_low=cfg.data.image_transform.clip_percentile_low,
                clip_percentile_high=cfg.data.image_transform.clip_percentile_high
            )
        )

    # Add augmentations if enabled
    # Support both new data.augmentation_enabled and old augmentation.enabled
    augmentation_enabled = getattr(cfg.data, 'augmentation_enabled',
                                   getattr(cfg.augmentation, 'enabled', False) if hasattr(cfg, 'augmentation') and cfg.augmentation else False)

    if augmentation_enabled and hasattr(cfg, 'augmentation') and cfg.augmentation is not None:
        transforms.extend(_build_augmentations(cfg.augmentation, keys))

    # Normalize labels to 0-1 range if enabled
    if getattr(cfg.data, 'normalize_labels', False):
        transforms.append(
            NormalizeLabelsd(keys=['label'])
        )

    # Label transformations (affinity, distance transform, etc.)
    if hasattr(cfg.data, 'label_transform'):
        from ..process.build import create_label_transform_pipeline
        from ..process.monai_transforms import SegErosionInstanced
        label_cfg = cfg.data.label_transform

        # Apply instance erosion first if specified
        if hasattr(label_cfg, 'erosion') and label_cfg.erosion > 0:
            transforms.append(SegErosionInstanced(keys=['label'], tsz_h=label_cfg.erosion))

        # Build label transform pipeline directly from label_transform config
        label_pipeline = create_label_transform_pipeline(label_cfg)
        transforms.extend(label_pipeline.transforms)

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def build_val_transforms(cfg: Config, keys: list[str] = None) -> Compose:
    """
    Build validation transforms from Hydra config.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: ['image', 'label'] or ['image', 'label', 'mask'] if masks are used)

    Returns:
        Composed MONAI transforms (no augmentation)
    """
    if keys is None:
        # Auto-detect keys based on config
        keys = ['image', 'label']
        # Add mask to keys if it's specified in the config (check both train and val masks)
        if (hasattr(cfg.data, 'val_mask') and cfg.data.val_mask is not None) or \
           (hasattr(cfg.data, 'train_mask') and cfg.data.train_mask is not None):
            keys.append('mask')

    transforms = []

    # Load images first
    # Get transpose axes for validation data
    val_transpose = cfg.data.val_transpose if cfg.data.val_transpose else []
    transforms.append(LoadVolumed(keys=keys, transpose_axes=val_transpose if val_transpose else None))

    # Apply volumetric split if enabled
    if cfg.data.split_enabled:
        from connectomics.data.utils import ApplyVolumetricSplitd
        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
    if hasattr(cfg.data.image_transform, 'resize') and cfg.data.image_transform.resize is not None:
        resize_factors = cfg.data.image_transform.resize
        if resize_factors:
            # Use bilinear for images, nearest for labels/masks
            transforms.append(
                Resized(
                    keys=['image'],
                    scale=resize_factors,
                    mode='bilinear',
                    align_corners=True
                )
            )
            # Resize labels and masks with nearest-neighbor
            label_mask_keys = [k for k in keys if k in ['label', 'mask']]
            if label_mask_keys:
                transforms.append(
                    Resized(
                        keys=label_mask_keys,
                        scale=resize_factors,
                        mode='nearest',
                        align_corners=None
                    )
                )

    patch_size = tuple(cfg.data.patch_size) if hasattr(cfg.data, 'patch_size') else None
    if patch_size and all(size > 0 for size in patch_size):
        transforms.append(
            SpatialPadd(
                keys=keys,
                spatial_size=patch_size,
                constant_values=0.0,
            )
        )

    # Add spatial cropping to prevent loading full volumes (OOM fix)
    # NOTE: If split is enabled with padding, this crop will be applied AFTER padding
    if patch_size and all(size > 0 for size in patch_size):
        transforms.append(
            CenterSpatialCropd(
                keys=keys,
                roi_size=patch_size,
            )
        )

    # Normalization - use smart normalization
    if cfg.data.image_transform.normalize != "none":
        transforms.append(
            SmartNormalizeIntensityd(
                keys=['image'],
                mode=cfg.data.image_transform.normalize,
                clip_percentile_low=cfg.data.image_transform.clip_percentile_low,
                clip_percentile_high=cfg.data.image_transform.clip_percentile_high
            )
        )

    # Normalize labels to 0-1 range if enabled
    if getattr(cfg.data, 'normalize_labels', False):
        transforms.append(
            NormalizeLabelsd(keys=['label'])
        )

    # Label transformations (affinity, distance transform, etc.)
    if hasattr(cfg.data, 'label_transform'):
        from ..process.build import create_label_transform_pipeline
        from ..process.monai_transforms import SegErosionInstanced
        label_cfg = cfg.data.label_transform

        # Apply instance erosion first if specified
        if hasattr(label_cfg, 'erosion') and label_cfg.erosion > 0:
            transforms.append(SegErosionInstanced(keys=['label'], tsz_h=label_cfg.erosion))

        # Build label transform pipeline directly from label_transform config
        label_pipeline = create_label_transform_pipeline(label_cfg)
        transforms.extend(label_pipeline.transforms)

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def build_test_transforms(cfg: Config, keys: list[str] = None) -> Compose:
    """
    Build test/inference transforms from Hydra config.

    Similar to validation transforms but WITHOUT cropping to enable
    sliding window inference on full volumes.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: ['image', 'label'] or ['image', 'label', 'mask'] if masks are used)

    Returns:
        Composed MONAI transforms (no augmentation, no cropping)
    """
    if keys is None:
        # Auto-detect keys based on config
        keys = ['image']
        # Only add label if test_label is specified in the config
        if hasattr(cfg, 'inference') and hasattr(cfg.inference, 'data') and \
           hasattr(cfg.inference.data, 'test_label') and cfg.inference.data.test_label is not None:
            keys.append('label')
        # Add mask to keys if it's specified in the config (check test mask)
        if hasattr(cfg, 'inference') and hasattr(cfg.inference, 'data') and \
           hasattr(cfg.inference.data, 'test_mask') and cfg.inference.data.test_mask is not None:
            keys.append('mask')

    transforms = []

    # Load images first
    # Get transpose axes for test data (check both data.test_transpose and inference.data.test_transpose)
    test_transpose = []
    if cfg.data.test_transpose:
        test_transpose = cfg.data.test_transpose
    if hasattr(cfg, 'inference') and hasattr(cfg.inference, 'data') and \
       hasattr(cfg.inference.data, 'test_transpose') and cfg.inference.data.test_transpose:
        test_transpose = cfg.inference.data.test_transpose  # inference takes precedence
    transforms.append(LoadVolumed(keys=keys, transpose_axes=test_transpose if test_transpose else None))

    # Apply volumetric split if enabled (though typically not used for test)
    if cfg.data.split_enabled:
        from connectomics.data.utils import ApplyVolumetricSplitd
        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before padding)
    if hasattr(cfg.data.image_transform, 'resize') and cfg.data.image_transform.resize is not None:
        resize_factors = cfg.data.image_transform.resize
        if resize_factors:
            # Use bilinear for images, nearest for labels/masks
            transforms.append(
                Resized(
                    keys=['image'],
                    scale=resize_factors,
                    mode='bilinear',
                    align_corners=True
                )
            )
            # Resize labels and masks with nearest-neighbor
            label_mask_keys = [k for k in keys if k in ['label', 'mask']]
            if label_mask_keys:
                transforms.append(
                    Resized(
                        keys=label_mask_keys,
                        scale=resize_factors,
                        mode='nearest',
                        align_corners=None
                    )
                )

    patch_size = tuple(cfg.data.patch_size) if hasattr(cfg.data, 'patch_size') else None
    if patch_size and all(size > 0 for size in patch_size):
        transforms.append(
            SpatialPadd(
                keys=keys,
                spatial_size=patch_size,
                constant_values=0.0,
            )
        )

    # NOTE: No CenterSpatialCropd here - we want full volumes for sliding window inference!

    # Normalization - use smart normalization
    if cfg.data.image_transform.normalize != "none":
        transforms.append(
            SmartNormalizeIntensityd(
                keys=['image'],
                mode=cfg.data.image_transform.normalize,
                clip_percentile_low=cfg.data.image_transform.clip_percentile_low,
                clip_percentile_high=cfg.data.image_transform.clip_percentile_high
            )
        )

    # Only apply label transforms if 'label' is in keys
    if 'label' in keys:
        # Normalize labels to 0-1 range if enabled
        if getattr(cfg.data, 'normalize_labels', False):
            transforms.append(
                NormalizeLabelsd(keys=['label'])
            )

        # Check if any evaluation metric is enabled (requires original instance labels)
        skip_label_transform = False
        if hasattr(cfg, 'inference') and hasattr(cfg.inference, 'evaluation'):
            evaluation_enabled = getattr(cfg.inference.evaluation, 'enabled', False)
            metrics = getattr(cfg.inference.evaluation, 'metrics', [])
            if evaluation_enabled and metrics:
                skip_label_transform = True
                print(f"  ⚠️  Skipping label transforms for metric evaluation (keeping original labels for {metrics})")

        # Label transformations (affinity, distance transform, etc.)
        # Skip if evaluation metrics are enabled (need original labels for metric computation)
        if hasattr(cfg.data, 'label_transform') and not skip_label_transform:
            from ..process.build import create_label_transform_pipeline
            from ..process.monai_transforms import SegErosionInstanced
            label_cfg = cfg.data.label_transform

            # Apply instance erosion first if specified
            if hasattr(label_cfg, 'erosion') and label_cfg.erosion > 0:
                transforms.append(SegErosionInstanced(keys=['label'], tsz_h=label_cfg.erosion))

            # Build label transform pipeline directly from label_transform config
            label_pipeline = create_label_transform_pipeline(label_cfg)
            transforms.extend(label_pipeline.transforms)

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def build_inference_transforms(cfg: Config) -> Compose:
    """
    Build inference transforms from Hydra config.
    
    Args:
        cfg: Hydra Config object
        
    Returns:
        Composed MONAI transforms for inference
    """
    keys = ['image']
    transforms = []
    
    # Normalization - use smart normalization
    if cfg.data.normalize:
        transforms.append(
            SmartNormalizeIntensityd(
                keys=keys,
                mean=cfg.data.mean,
                std=cfg.data.std,
                clip_percentile_low=getattr(cfg.data, 'clip_percentile_low', 0.0),
                clip_percentile_high=getattr(cfg.data, 'clip_percentile_high', 1.0)
            )
        )

    # Convert to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def _build_augmentations(aug_cfg: AugmentationConfig, keys: list[str]) -> list:
    """
    Build augmentation transforms from config.
    
    Args:
        aug_cfg: AugmentationConfig object
        keys: Keys to augment
        
    Returns:
        List of MONAI transforms
    """
    transforms = []
    
    # Standard geometric augmentations
    if aug_cfg.flip.enabled:
        transforms.append(
            RandFlipd(
                keys=keys,
                prob=aug_cfg.flip.prob,
                spatial_axis=aug_cfg.flip.spatial_axis
            )
        )
    
    if aug_cfg.rotate.enabled:
        transforms.append(
            RandRotate90d(
                keys=keys,
                prob=aug_cfg.rotate.prob,
                spatial_axes=(1, 2)  # Rotate in Y-X plane to preserve anisotropic Z
            )
        )
    
    if aug_cfg.elastic.enabled:
        transforms.append(
            Rand3DElasticd(
                keys=keys,
                prob=aug_cfg.elastic.prob,
                sigma_range=aug_cfg.elastic.sigma_range,
                magnitude_range=aug_cfg.elastic.magnitude_range
            )
        )
    
    # Intensity augmentations (only for images)
    if aug_cfg.intensity.enabled:
        if aug_cfg.intensity.gaussian_noise_prob > 0:
            transforms.append(
                RandGaussianNoised(
                    keys=['image'],
                    prob=aug_cfg.intensity.gaussian_noise_prob,
                    std=aug_cfg.intensity.gaussian_noise_std
                )
            )
        
        if aug_cfg.intensity.shift_intensity_prob > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=['image'],
                    prob=aug_cfg.intensity.shift_intensity_prob,
                    offsets=aug_cfg.intensity.shift_intensity_offset
                )
            )
        
        if aug_cfg.intensity.contrast_prob > 0:
            transforms.append(
                RandAdjustContrastd(
                    keys=['image'],
                    prob=aug_cfg.intensity.contrast_prob,
                    gamma=aug_cfg.intensity.contrast_range
                )
            )
    
    # EM-specific augmentations
    if aug_cfg.misalignment.enabled:
        transforms.append(
            RandMisAlignmentd(
                keys=keys,
                prob=aug_cfg.misalignment.prob,
                displacement=aug_cfg.misalignment.displacement,
                rotate_ratio=aug_cfg.misalignment.rotate_ratio
            )
        )
    
    if aug_cfg.missing_section.enabled:
        transforms.append(
            RandMissingSectiond(
                keys=keys,
                prob=aug_cfg.missing_section.prob,
                num_sections=aug_cfg.missing_section.num_sections
            )
        )
    
    if aug_cfg.motion_blur.enabled:
        transforms.append(
            RandMotionBlurd(
                keys=['image'],
                prob=aug_cfg.motion_blur.prob,
                sections=aug_cfg.motion_blur.sections,
                kernel_size=aug_cfg.motion_blur.kernel_size
            )
        )
    
    if aug_cfg.cut_noise.enabled:
        transforms.append(
            RandCutNoised(
                keys=['image'],
                prob=aug_cfg.cut_noise.prob,
                length_ratio=aug_cfg.cut_noise.length_ratio,
                noise_scale=aug_cfg.cut_noise.noise_scale
            )
        )
    
    if aug_cfg.cut_blur.enabled:
        transforms.append(
            RandCutBlurd(
                keys=['image'],
                prob=aug_cfg.cut_blur.prob,
                length_ratio=aug_cfg.cut_blur.length_ratio,
                down_ratio_range=aug_cfg.cut_blur.down_ratio_range,
                downsample_z=aug_cfg.cut_blur.downsample_z
            )
        )
    
    if aug_cfg.missing_parts.enabled:
        transforms.append(
            RandMissingPartsd(
                keys=keys,
                prob=aug_cfg.missing_parts.prob,
                hole_range=aug_cfg.missing_parts.hole_range
            )
        )
    
    # Advanced augmentations
    if aug_cfg.mixup.enabled:
        transforms.append(
            RandMixupd(
                keys=['image'],
                prob=aug_cfg.mixup.prob,
                alpha_range=aug_cfg.mixup.alpha_range
            )
        )
    
    if aug_cfg.copy_paste.enabled:
        transforms.append(
            RandCopyPasted(
                keys=['image'],
                label_key='label',
                prob=aug_cfg.copy_paste.prob,
                max_obj_ratio=aug_cfg.copy_paste.max_obj_ratio,
                rotation_angles=aug_cfg.copy_paste.rotation_angles,
                border=aug_cfg.copy_paste.border
            )
        )
    
    return transforms


def build_transform_dict(cfg: Config) -> Dict[str, Compose]:
    """
    Build dictionary of transforms for train/val/test splits.
    
    Args:
        cfg: Hydra Config object
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    return {
        'train': build_train_transforms(cfg),
        'val': build_val_transforms(cfg),
        'test': build_val_transforms(cfg)
    }


__all__ = [
    'build_train_transforms',
    'build_val_transforms',
    'build_test_transforms',
    'build_inference_transforms',
    'build_transform_dict',
]
