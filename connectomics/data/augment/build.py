"""
Build MONAI transform pipelines from Hydra configuration.

Modern replacement for monai_compose.py that works with the new Hydra config system.
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
from monai.transforms import (
    Compose,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandSpatialCropd,
    ToTensord,
    CenterSpatialCropd,
    SpatialPadd,
    Resized,
    LoadImaged,  # For filename-based datasets (PNG, JPG, etc.)
    EnsureChannelFirstd,  # Ensure channel-first format for 2D/3D images
)

# Import custom loader for HDF5/TIFF volumes
from connectomics.data.dataset.dataset_volume import LoadVolumed

from .monai_transforms import (
    RandMisAlignmentd,
    RandMissingSectiond,
    RandMissingPartsd,
    RandMotionBlurd,
    RandCutNoised,
    RandCutBlurd,
    RandMixupd,
    RandCopyPasted,
    RandStriped,
    NormalizeLabelsd,
    SmartNormalizeIntensityd,
    RandElasticd,
)
from ...config.hydra_config import Config, AugmentationConfig


def build_train_transforms(
    cfg: Config, keys: list[str] = None, skip_loading: bool = False
) -> Compose:
    """
    Build training transforms from Hydra config.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: ['image', 'label'] or
            ['image', 'label', 'mask'] if masks are used)
        skip_loading: Skip LoadVolumed (for pre-cached datasets)

    Returns:
        Composed MONAI transforms
    """
    if keys is None:
        # Auto-detect keys based on config
        keys = ["image", "label"]
        # Add mask to keys if it's specified in the config
        if hasattr(cfg.data, "train_mask") and cfg.data.train_mask is not None:
            keys.append("mask")

    transforms = []

    # Load images first (unless using pre-cached dataset)
    if not skip_loading:
        # Use appropriate loader based on dataset type
        dataset_type = getattr(cfg.data, "dataset_type", "volume")

        if dataset_type == "filename":
            # For filename-based datasets (PNG, JPG, etc.), use MONAI's LoadImaged
            transforms.append(LoadImaged(keys=keys, image_only=False))
            # Ensure channel-first format [C, H, W] or [C, D, H, W]
            transforms.append(EnsureChannelFirstd(keys=keys))
        else:
            # For volume-based datasets (HDF5, TIFF volumes), use custom LoadVolumed
            train_transpose = cfg.data.train_transpose if cfg.data.train_transpose else []
            transforms.append(
                LoadVolumed(keys=keys, transpose_axes=train_transpose if train_transpose else None)
            )

    # Apply volumetric split if enabled
    if cfg.data.split_enabled:
        from connectomics.data.utils import ApplyVolumetricSplitd

        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
    resize_size = None
    if (
        hasattr(cfg.data, "data_transform")
        and hasattr(cfg.data.data_transform, "resize")
        and cfg.data.data_transform.resize is not None
    ):
        resize_size = cfg.data.data_transform.resize

    if resize_size:
        # Use bilinear for images, nearest for labels/masks
        transforms.append(
            Resized(
                keys=["image"],
                spatial_size=resize_size,  # Target size [H, W] or [D, H, W]
                mode="bilinear",  # Bilinear interpolation for images
                align_corners=True,
            )
        )
        # Resize labels and masks with nearest-neighbor to preserve integer values
        label_mask_keys = [k for k in keys if k in ["label", "mask"]]
        if label_mask_keys:
            transforms.append(
                Resized(
                    keys=label_mask_keys,
                    spatial_size=resize_size,  # Same target size
                    mode="nearest",  # Nearest neighbor for labels/masks
                    align_corners=None,  # Not used for nearest mode
                )
            )

    # Ensure target patch size is respected (unless using pre-cached dataset)
    if not skip_loading:
        patch_size = tuple(cfg.data.patch_size) if hasattr(cfg.data, "patch_size") else None
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
                keys=["image"],
                mode=cfg.data.image_transform.normalize,
                clip_percentile_low=cfg.data.image_transform.clip_percentile_low,
                clip_percentile_high=cfg.data.image_transform.clip_percentile_high,
            )
        )

    # Add augmentations if enabled
    # Check if augmentation is configured and preset is not "none"
    augmentation_enabled = False
    if hasattr(cfg.data, "augmentation") and cfg.data.augmentation is not None:
        preset = getattr(cfg.data.augmentation, "preset", "some")
        augmentation_enabled = preset != "none"

    if augmentation_enabled:
        # Pass do_2d flag to augmentation builder
        do_2d = getattr(cfg.data, "do_2d", False)
        transforms.extend(_build_augmentations(cfg.data.augmentation, keys, do_2d=do_2d))

    # Normalize labels to 0-1 range if enabled
    if getattr(cfg.data, "normalize_labels", False):
        transforms.append(NormalizeLabelsd(keys=["label"]))

    # Label transformations (affinity, distance transform, etc.)
    if hasattr(cfg.data, "label_transform"):
        from ..process.build import create_label_transform_pipeline
        from ..process.monai_transforms import SegErosionInstanced

        label_cfg = cfg.data.label_transform

        # Apply instance erosion first if specified
        if hasattr(label_cfg, "erosion") and label_cfg.erosion > 0:
            transforms.append(SegErosionInstanced(keys=["label"], tsz_h=label_cfg.erosion))

        # Build label transform pipeline directly from label_transform config
        label_pipeline = create_label_transform_pipeline(label_cfg)
        transforms.extend(label_pipeline.transforms)

    # NOTE: Do NOT squeeze labels here!
    # - DiceLoss needs (B, 1, H, W) with to_onehot_y=True
    # - CrossEntropyLoss needs (B, H, W)
    # Squeezing is handled in the loss wrapper instead

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def _build_eval_transforms_impl(cfg: Config, mode: str = "val", keys: list[str] = None) -> Compose:
    """
    Internal implementation for building evaluation transforms (validation or test).

    This function contains the shared logic between validation and test transforms,
    with mode-specific branching for key differences.

    Args:
        cfg: Hydra Config object
        mode: 'val' or 'test' mode
        keys: Keys to transform (default: auto-detected based on mode)

    Returns:
        Composed MONAI transforms (no augmentation)
    """
    if keys is None:
        # Auto-detect keys based on mode
        if mode == "val":
            # Validation: default to image+label
            keys = ["image", "label"]
            # Add mask if val_mask or train_mask exists
            if (hasattr(cfg.data, "val_mask") and cfg.data.val_mask is not None) or (
                hasattr(cfg.data, "train_mask") and cfg.data.train_mask is not None
            ):
                keys.append("mask")
        else:  # mode == "test"
            # Test/inference: default to image only
            keys = ["image"]
            # Only add label if test_label or tune_label is explicitly specified
            has_test_label = (
                hasattr(cfg, "test")
                and hasattr(cfg.test, "data")
                and hasattr(cfg.test.data, "test_label")
                and cfg.test.data.test_label is not None
            )
            has_tune_label = (
                hasattr(cfg, "tune")
                and cfg.tune is not None
                and hasattr(cfg.tune, "data")
                and cfg.tune.data.tune_label is not None
            )
            if has_test_label or has_tune_label:
                keys.append("label")

            # Add mask if test_mask or tune_mask is explicitly specified
            has_test_mask = (
                hasattr(cfg, "test")
                and hasattr(cfg.test, "data")
                and hasattr(cfg.test.data, "test_mask")
                and cfg.test.data.test_mask is not None
            )
            has_tune_mask = (
                hasattr(cfg, "tune")
                and cfg.tune is not None
                and hasattr(cfg.tune, "data")
                and cfg.tune.data.tune_mask is not None
            )
            if has_test_mask or has_tune_mask:
                keys.append("mask")

    transforms = []

    # Load images first - use appropriate loader based on dataset type
    dataset_type = getattr(cfg.data, "dataset_type", "volume")

    if dataset_type == "filename":
        # For filename-based datasets (PNG, JPG, etc.), use MONAI's LoadImaged
        transforms.append(LoadImaged(keys=keys, image_only=False))
        # Ensure channel-first format [C, H, W] or [C, D, H, W]
        transforms.append(EnsureChannelFirstd(keys=keys))
    else:
        # For volume-based datasets (HDF5, TIFF volumes), use custom LoadVolumed
        # Get transpose axes based on mode
        if mode == "val":
            transpose_axes = cfg.data.val_transpose if cfg.data.val_transpose else []
        else:  # mode == "test"
            # Use test.data.test_transpose
            transpose_axes = []
            if (
                hasattr(cfg, "test")
                and hasattr(cfg.test, "data")
                and hasattr(cfg.test.data, "test_transpose")
                and cfg.test.data.test_transpose
            ):
                transpose_axes = cfg.test.data.test_transpose

        transforms.append(
            LoadVolumed(keys=keys, transpose_axes=transpose_axes if transpose_axes else None)
        )

    # Apply volumetric split if enabled
    if cfg.data.split_enabled:
        from connectomics.data.utils import ApplyVolumetricSplitd

        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
    # For test/tune mode, only use test.data.image_transform or
    # tune.data.image_transform (no fallback)
    # For val mode, use data.image_transform
    resize_factors = None
    if mode == "test":
        if (
            hasattr(cfg, "test")
            and hasattr(cfg.test, "data")
            and hasattr(cfg.test.data, "image_transform")
        ):
            if (
                hasattr(cfg.test.data.image_transform, "resize")
                and cfg.test.data.image_transform.resize is not None
            ):
                resize_factors = cfg.test.data.image_transform.resize
    elif mode == "tune":
        if (
            hasattr(cfg, "tune")
            and cfg.tune
            and hasattr(cfg.tune, "data")
            and hasattr(cfg.tune.data, "image_transform")
        ):
            if (
                hasattr(cfg.tune.data.image_transform, "resize")
                and cfg.tune.data.image_transform.resize is not None
            ):
                resize_factors = cfg.tune.data.image_transform.resize
    else:  # mode == "val"
        if (
            hasattr(cfg.data.image_transform, "resize")
            and cfg.data.image_transform.resize is not None
        ):
            resize_factors = cfg.data.image_transform.resize

    if resize_factors is not None:
        if resize_factors:
            # Use bilinear for images, nearest for labels/masks
            transforms.append(
                Resized(keys=["image"], scale=resize_factors, mode="bilinear", align_corners=True)
            )
            # Resize labels and masks with nearest-neighbor
            label_mask_keys = [k for k in keys if k in ["label", "mask"]]
            if label_mask_keys:
                transforms.append(
                    Resized(
                        keys=label_mask_keys,
                        scale=resize_factors,
                        mode="nearest",
                        align_corners=None,
                    )
                )

    patch_size = tuple(cfg.data.patch_size) if hasattr(cfg.data, "patch_size") else None
    if patch_size and all(size > 0 for size in patch_size):
        transforms.append(
            SpatialPadd(
                keys=keys,
                spatial_size=patch_size,
                constant_values=0.0,
            )
        )

    # Add spatial cropping - MODE-SPECIFIC
    # Validation: Apply center crop for patch-based validation
    # Test: Skip cropping to enable sliding window inference on full volumes
    if mode == "val":
        if patch_size and all(size > 0 for size in patch_size):
            transforms.append(
                CenterSpatialCropd(
                    keys=keys,
                    roi_size=patch_size,
                )
            )
    # else: mode == "test" -> no cropping for sliding window inference

    # Normalization - use smart normalization
    # For test/tune mode, only use test.data.image_transform or
    # tune.data.image_transform (no fallback)
    # For val mode, use data.image_transform
    image_transform = None
    if (
        mode == "test"
        and hasattr(cfg, "test")
        and hasattr(cfg.test, "data")
        and hasattr(cfg.test.data, "image_transform")
    ):
        image_transform = cfg.test.data.image_transform
    elif (
        mode == "tune"
        and hasattr(cfg, "tune")
        and cfg.tune
        and hasattr(cfg.tune, "data")
        and hasattr(cfg.tune.data, "image_transform")
    ):
        image_transform = cfg.tune.data.image_transform
    elif mode == "val":
        image_transform = cfg.data.image_transform

    if image_transform is not None and image_transform.normalize != "none":
        transforms.append(
            SmartNormalizeIntensityd(
                keys=["image"],
                mode=image_transform.normalize,
                clip_percentile_low=getattr(image_transform, "clip_percentile_low", 0.0),
                clip_percentile_high=getattr(image_transform, "clip_percentile_high", 1.0),
            )
        )

    # Only process labels if 'label' is in keys
    if "label" in keys:
        # Normalize labels to 0-1 range if enabled
        if getattr(cfg.data, "normalize_labels", False):
            transforms.append(NormalizeLabelsd(keys=["label"]))

        # Label transformations (affinity, distance transform, etc.)
        # For test/tune modes: NEVER apply label transforms
        # (keep raw instance labels for evaluation)
        # For val mode: use training label_transform config
        label_cfg = None
        if mode == "val":
            # Validation always uses training label_transform
            if hasattr(cfg.data, "label_transform"):
                label_cfg = cfg.data.label_transform

        # Apply label transforms if configured
        if label_cfg is not None:
            from ..process.build import create_label_transform_pipeline
            from ..process.monai_transforms import SegErosionInstanced

            # Apply instance erosion first if specified
            if hasattr(label_cfg, "erosion") and label_cfg.erosion > 0:
                transforms.append(SegErosionInstanced(keys=["label"], tsz_h=label_cfg.erosion))

            # Build label transform pipeline directly from label_transform config
            label_pipeline = create_label_transform_pipeline(label_cfg)
            transforms.extend(label_pipeline.transforms)

    # NOTE: Do NOT squeeze labels here!
    # - DiceLoss needs (B, 1, H, W) with to_onehot_y=True
    # - CrossEntropyLoss needs (B, H, W)
    # Squeezing is handled in the loss wrapper instead

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def build_val_transforms(cfg: Config, keys: list[str] = None) -> Compose:
    """
    Build validation transforms from Hydra config.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: auto-detected as ['image', 'label'])

    Returns:
        Composed MONAI transforms (no augmentation, center cropping)
    """
    return _build_eval_transforms_impl(cfg, mode="val", keys=keys)


def build_test_transforms(cfg: Config, keys: list[str] = None) -> Compose:
    """
    Build test/inference transforms from Hydra config.

    Similar to validation transforms but WITHOUT cropping to enable
    sliding window inference on full volumes.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: auto-detected as ['image'] only)

    Returns:
        Composed MONAI transforms (no augmentation, no cropping)
    """
    return _build_eval_transforms_impl(cfg, mode="test", keys=keys)


def build_inference_transforms(cfg: Config) -> Compose:
    """
    Build inference transforms from Hydra config.

    Args:
        cfg: Hydra Config object

    Returns:
        Composed MONAI transforms for inference
    """
    keys = ["image"]
    transforms = []

    # Normalization - use smart normalization
    if cfg.data.normalize:
        transforms.append(
            SmartNormalizeIntensityd(
                keys=keys,
                mean=cfg.data.mean,
                std=cfg.data.std,
                clip_percentile_low=getattr(cfg.data, "clip_percentile_low", 0.0),
                clip_percentile_high=getattr(cfg.data, "clip_percentile_high", 1.0),
            )
        )

    # Convert to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def _build_augmentations(aug_cfg: AugmentationConfig, keys: list[str], do_2d: bool = False) -> list:
    """
    Build augmentation transforms from config.

    Args:
        aug_cfg: AugmentationConfig object
        keys: Keys to augment
        do_2d: Whether data is 2D (True) or 3D (False)

    Returns:
        List of MONAI transforms
    """
    transforms = []

    # Get preset mode
    preset = getattr(aug_cfg, "preset", "some")

    # Validate preset choice
    valid_presets = {"none", "some", "all"}
    if preset not in valid_presets:
        raise ValueError(
            f"Invalid augmentation preset: '{preset}'. "
            f"Valid choices are: {', '.join(sorted(valid_presets))}. "
            f"Got: '{preset}'. Please use one of the valid options."
        )

    # Helper function to check if augmentation should be applied
    def should_augment(aug_name: str, aug_enabled: Optional[bool]) -> bool:
        """
        Determine if augmentation should be applied based on preset mode and enabled flag.

        All augmentations default to enabled=None (use preset default). The preset mode controls
        the behavior:

        - "none": Disable all augmentations
        - "some": Disable all by default (only True enables)
        - "all": Enable all by default (only False disables)

        The enabled field can be:
        - None: Use preset default (not specified in YAML)
        - True: Explicitly enable (overrides preset)
        - False: Explicitly disable (overrides preset)

        Examples:
            preset="some" with enabled=None:
                → Disabled (default off, must opt-in)
                → User enables in YAML: flip.enabled: true

            preset="some" with enabled=true:
                → Enabled

            preset="all" with enabled=None:
                → Enabled (default on)
                → User can disable: flip.enabled: false

            preset="all" with enabled=false:
                → Disabled (explicit override)

            preset="none":
                → Always disabled
        """
        if preset == "none":
            return False
        elif preset == "all":
            # Enable all by default, unless explicitly disabled (False)
            # None (default) → True
            # True (explicit) → True
            # False (explicit disable) → False
            if aug_enabled is False:
                return False
            else:  # None or True
                return True
        else:  # preset == "some"
            # Disable all by default, unless explicitly enabled (True)
            # None (default) → False
            # True (explicit enable) → True
            # False (explicit) → False
            if aug_enabled is True:
                return True
            else:  # None or False
                return False

    # Standard geometric augmentations
    if should_augment("flip", aug_cfg.flip.enabled):
        transforms.append(
            RandFlipd(keys=keys, prob=aug_cfg.flip.prob, spatial_axis=aug_cfg.flip.spatial_axis)
        )

    if should_augment("rotate", aug_cfg.rotate.enabled):
        # Determine spatial_axes based on data dimensionality
        # MONAI transforms work on (C, *spatial) tensors (no batch dimension)
        # - 2D data: (C, H, W) → spatial_axes=(0, 1) rotates H-W plane
        # - 3D data: (C, D, H, W) → spatial_axes=(1, 2) rotates H-W plane

        # Auto-detect based on do_2d flag (default behavior for 2D/3D)
        spatial_axes = (0, 1) if do_2d else (1, 2)

        transforms.append(
            RandRotate90d(
                keys=keys,
                prob=aug_cfg.rotate.prob,
                spatial_axes=spatial_axes,
            )
        )

    if should_augment("affine", aug_cfg.affine.enabled):
        # Adjust affine parameters for 2D vs 3D data
        # For 2D: use only the first element of each range
        # For 3D: use all three elements
        if do_2d:
            rotate_range = (aug_cfg.affine.rotate_range[0],)
            scale_range = (aug_cfg.affine.scale_range[0],)
            shear_range = (aug_cfg.affine.shear_range[0],)
        else:
            rotate_range = aug_cfg.affine.rotate_range
            scale_range = aug_cfg.affine.scale_range
            shear_range = aug_cfg.affine.shear_range

        # Interpolation per key: bilinear for images, nearest for labels/masks
        affine_modes = ["bilinear" if k == "image" else "nearest" for k in keys]

        transforms.append(
            RandAffined(
                keys=keys,
                prob=aug_cfg.affine.prob,
                rotate_range=rotate_range,
                scale_range=scale_range,
                shear_range=shear_range,
                mode=affine_modes,
                padding_mode="reflection",
            )
        )

    if should_augment("elastic", aug_cfg.elastic.enabled):
        # Unified elastic deformation that supports both 2D and 3D
        elastic_modes = ["bilinear" if k == "image" else "nearest" for k in keys]
        transforms.append(
            RandElasticd(
                keys=keys,
                do_2d=do_2d,
                prob=aug_cfg.elastic.prob,
                sigma_range=aug_cfg.elastic.sigma_range,
                magnitude_range=aug_cfg.elastic.magnitude_range,
                mode=elastic_modes,
            )
        )

    # Intensity augmentations (only for images)
    if should_augment("intensity", aug_cfg.intensity.enabled):
        if aug_cfg.intensity.gaussian_noise_prob > 0:
            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    prob=aug_cfg.intensity.gaussian_noise_prob,
                    std=aug_cfg.intensity.gaussian_noise_std,
                )
            )

        if aug_cfg.intensity.shift_intensity_prob > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    prob=aug_cfg.intensity.shift_intensity_prob,
                    offsets=aug_cfg.intensity.shift_intensity_offset,
                )
            )

        if aug_cfg.intensity.contrast_prob > 0:
            transforms.append(
                RandAdjustContrastd(
                    keys=["image"],
                    prob=aug_cfg.intensity.contrast_prob,
                    gamma=aug_cfg.intensity.contrast_range,
                )
            )

    # EM-specific augmentations
    if should_augment("misalignment", aug_cfg.misalignment.enabled):
        transforms.append(
            RandMisAlignmentd(
                keys=keys,
                prob=aug_cfg.misalignment.prob,
                displacement=aug_cfg.misalignment.displacement,
                rotate_ratio=aug_cfg.misalignment.rotate_ratio,
            )
        )

    if should_augment("missing_section", aug_cfg.missing_section.enabled):
        transforms.append(
            RandMissingSectiond(
                keys=keys,
                prob=aug_cfg.missing_section.prob,
                num_sections=aug_cfg.missing_section.num_sections,
            )
        )

    if should_augment("motion_blur", aug_cfg.motion_blur.enabled):
        transforms.append(
            RandMotionBlurd(
                keys=["image"],
                prob=aug_cfg.motion_blur.prob,
                sections=aug_cfg.motion_blur.sections,
                kernel_size=aug_cfg.motion_blur.kernel_size,
            )
        )

    if should_augment("cut_noise", aug_cfg.cut_noise.enabled):
        transforms.append(
            RandCutNoised(
                keys=["image"],
                prob=aug_cfg.cut_noise.prob,
                length_ratio=aug_cfg.cut_noise.length_ratio,
                noise_scale=aug_cfg.cut_noise.noise_scale,
            )
        )

    if should_augment("cut_blur", aug_cfg.cut_blur.enabled):
        transforms.append(
            RandCutBlurd(
                keys=["image"],
                prob=aug_cfg.cut_blur.prob,
                length_ratio=aug_cfg.cut_blur.length_ratio,
                down_ratio_range=aug_cfg.cut_blur.down_ratio_range,
                downsample_z=aug_cfg.cut_blur.downsample_z,
            )
        )

    if should_augment("missing_parts", aug_cfg.missing_parts.enabled):
        transforms.append(
            RandMissingPartsd(
                keys=keys,
                prob=aug_cfg.missing_parts.prob,
                hole_range=aug_cfg.missing_parts.hole_range,
            )
        )

    if should_augment("stripe", aug_cfg.stripe.enabled):
        transforms.append(
            RandStriped(
                keys=["image"],
                prob=aug_cfg.stripe.prob,
                num_stripes_range=aug_cfg.stripe.num_stripes_range,
                thickness_range=aug_cfg.stripe.thickness_range,
                intensity_range=aug_cfg.stripe.intensity_range,
                angle_range=aug_cfg.stripe.angle_range,
                orientation=aug_cfg.stripe.orientation,
                mode=aug_cfg.stripe.mode,
            )
        )

    # Advanced augmentations
    if should_augment("mixup", aug_cfg.mixup.enabled):
        transforms.append(
            RandMixupd(
                keys=["image"], prob=aug_cfg.mixup.prob, alpha_range=aug_cfg.mixup.alpha_range
            )
        )

    if should_augment("copy_paste", aug_cfg.copy_paste.enabled):
        transforms.append(
            RandCopyPasted(
                keys=["image"],
                label_key="label",
                prob=aug_cfg.copy_paste.prob,
                max_obj_ratio=aug_cfg.copy_paste.max_obj_ratio,
                rotation_angles=aug_cfg.copy_paste.rotation_angles,
                border=aug_cfg.copy_paste.border,
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
        "train": build_train_transforms(cfg),
        "val": build_val_transforms(cfg),
        "test": build_val_transforms(cfg),
    }


__all__ = [
    "build_train_transforms",
    "build_val_transforms",
    "build_test_transforms",
    "build_inference_transforms",
    "build_transform_dict",
]
