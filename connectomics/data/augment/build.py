"""
Build MONAI transform pipelines from Hydra configuration.

Modern replacement for monai_compose.py that works with the new Hydra config system.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from monai.transforms import EnsureChannelFirstd  # Ensure channel-first format for 2D/3D images
from monai.transforms import LoadImaged  # For filename-based datasets (PNG, JPG, etc.)
from monai.transforms import (
    BorderPadd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    OneOf,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    ToTensord,
)

# Import custom loader for HDF5/TIFF volumes
from connectomics.data.io.transforms import LoadVolumed
from connectomics.data.process.nnunet_preprocess import NNUNetPreprocessd

from ...config.schema import AugmentationConfig, Config
from .transforms import (
    RandCopyPasted,
    RandCutBlurd,
    RandCutNoised,
    RandElasticd,
    RandMisAlignmentd,
    RandMissingPartsd,
    RandMissingSectiond,
    RandMixupd,
    RandMotionBlurd,
    RandStriped,
    ResizeByFactord,
    SmartNormalizeIntensityd,
)


def _strict_binarize_mask(mask, threshold: float = 0.0):
    """Binarize mask with strict greater-than semantics (mask > threshold)."""
    if torch.is_tensor(mask):
        return (mask > threshold).to(dtype=mask.dtype)
    return (mask > threshold).astype(mask.dtype, copy=False)


def _build_nnunet_preprocess_transform(keys, nnunet_pre_cfg, source_spacing):
    """Build NNUNetPreprocessd transform from config."""
    source_spacing = getattr(nnunet_pre_cfg, "source_spacing", None) or source_spacing
    return NNUNetPreprocessd(
        keys=keys,
        image_key="image",
        enabled=True,
        crop_to_nonzero=getattr(nnunet_pre_cfg, "crop_to_nonzero", True),
        source_spacing=source_spacing,
        target_spacing=getattr(nnunet_pre_cfg, "target_spacing", None),
        normalization=getattr(nnunet_pre_cfg, "normalization", "zscore"),
        normalization_use_nonzero_mask=getattr(
            nnunet_pre_cfg, "normalization_use_nonzero_mask", True
        ),
        clip_percentile_low=getattr(nnunet_pre_cfg, "clip_percentile_low", 0.0),
        clip_percentile_high=getattr(nnunet_pre_cfg, "clip_percentile_high", 1.0),
        force_separate_z=getattr(nnunet_pre_cfg, "force_separate_z", None),
        anisotropy_threshold=getattr(nnunet_pre_cfg, "anisotropy_threshold", 3.0),
        image_order=getattr(nnunet_pre_cfg, "image_order", 3),
        label_order=getattr(nnunet_pre_cfg, "label_order", 0),
        order_z=getattr(nnunet_pre_cfg, "order_z", 0),
    )


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
        keys = ["image", "label"]
        if cfg.data.train.mask is not None:
            keys.append("mask")

    transforms = []

    # Load images first (unless using pre-cached dataset)
    if not skip_loading:
        # Use appropriate loader based on dataset type
        dataset_type = (
            getattr(cfg.data.train, "dataset_type", None)
            or getattr(cfg.data.val, "dataset_type", None)
            or "volume"
        )

        if dataset_type == "filename":
            # For filename-based datasets (PNG, JPG, etc.), use MONAI's LoadImaged
            transforms.append(LoadImaged(keys=keys, image_only=False))
            # Ensure channel-first format [C, H, W] or [C, D, H, W]
            transforms.append(EnsureChannelFirstd(keys=keys))
        else:
            # For volume-based datasets (HDF5, TIFF volumes), use custom LoadVolumed
            train_transpose = (
                cfg.data.data_transform.train_transpose
                if cfg.data.data_transform.train_transpose
                else []
            )
            transforms.append(
                LoadVolumed(keys=keys, transpose_axes=train_transpose if train_transpose else None)
            )

    nnunet_pre_cfg = getattr(cfg.data, "nnunet_preprocessing", None)
    nnunet_pre_enabled = bool(getattr(nnunet_pre_cfg, "enabled", False))
    if not skip_loading and nnunet_pre_enabled:
        source_spacing = getattr(cfg.data.train, "resolution", None)
        transforms.append(_build_nnunet_preprocess_transform(keys, nnunet_pre_cfg, source_spacing))

    # Apply volumetric split if enabled
    if cfg.data.split_enabled:
        from connectomics.data.dataset.split import ApplyVolumetricSplitd

        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
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
        patch_size = (
            tuple(cfg.data.dataloader.patch_size) if cfg.data.dataloader.patch_size else None
        )
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
    if (not nnunet_pre_enabled) and cfg.data.image_transform.normalize != "none":
        transforms.append(
            SmartNormalizeIntensityd(
                keys=["image"],
                mode=cfg.data.image_transform.normalize,
                clip_percentile_low=cfg.data.image_transform.clip_percentile_low,
                clip_percentile_high=cfg.data.image_transform.clip_percentile_high,
            )
        )

    # Add augmentations if enabled
    if cfg.data.augmentation is not None and cfg.data.augmentation.preset != "none":
        # Pass do_2d flag to augmentation builder
        do_2d = bool(
            getattr(getattr(cfg.data, "train", None), "do_2d", False)
            or getattr(getattr(cfg.data, "val", None), "do_2d", False)
        )
        transforms.extend(_build_augmentations(cfg.data.augmentation, keys, do_2d=do_2d))

    # Label transformations (affinity, distance transform, etc.)
    if hasattr(cfg.data, "label_transform"):
        from ..process.build import create_label_transform_pipeline
        from ..process.transforms import SegErosionInstanced

        label_cfg = cfg.data.label_transform

        # Apply instance erosion first if specified
        if hasattr(label_cfg, "erosion") and label_cfg.erosion > 0:
            transforms.append(SegErosionInstanced(keys=["label"], tsz_h=label_cfg.erosion))

        # Build label transform pipeline directly from label_transform config
        label_transform = create_label_transform_pipeline(label_cfg)
        if isinstance(label_transform, Compose):
            transforms.extend(label_transform.transforms)
        else:
            transforms.append(label_transform)

    # NOTE: Do NOT squeeze labels here!
    # - DiceLoss needs (B, 1, H, W) with to_onehot_y=True
    # - CrossEntropyLoss needs (B, H, W)
    # Squeezing is handled in the loss wrapper instead

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def _build_eval_transforms_impl(
    cfg: Config, mode: str = "val", keys: list[str] = None, skip_loading: bool = False
) -> Compose:
    """
    Internal implementation for building evaluation transforms (validation or test).

    This function contains the shared logic between validation and test transforms,
    with mode-specific branching for key differences.

    Args:
        cfg: Hydra Config object
        mode: 'val' or 'test' mode
        keys: Keys to transform (default: auto-detected based on mode)
        skip_loading: Skip LoadVolumed (for pre-cached datasets)

    Returns:
        Composed MONAI transforms (no augmentation)
    """
    data_cfg = cfg.data

    def _resolve_eval_split():
        if mode == "val":
            return data_cfg.val
        if mode == "tune":
            return data_cfg.val
        return data_cfg.test

    if keys is None:
        # Auto-detect keys based on mode
        if mode == "val":
            # Validation: default to image+label
            keys = ["image", "label"]
            # Add mask if val_mask or train_mask exists
            if (
                hasattr(data_cfg, "val")
                and hasattr(data_cfg.val, "mask")
                and data_cfg.val.mask is not None
            ) or (
                hasattr(data_cfg, "train")
                and hasattr(data_cfg.train, "mask")
                and data_cfg.train.mask is not None
            ):
                keys.append("mask")
        else:  # mode == "test" or "tune"
            # Test/inference: default to image only
            eval_split = _resolve_eval_split()
            keys = ["image"]
            if eval_split.label is not None:
                keys.append("label")

            if eval_split.mask is not None:
                keys.append("mask")

    transforms = []

    # Load images first - use appropriate loader based on dataset type
    # Skip loading if using pre-cached datasets
    if not skip_loading:
        dataset_type = (
            getattr(data_cfg.train, "dataset_type", None)
            or getattr(data_cfg.val, "dataset_type", None)
            or "volume"
        )

        if dataset_type == "filename":
            # For filename-based datasets (PNG, JPG, etc.), use MONAI's LoadImaged
            transforms.append(LoadImaged(keys=keys, image_only=False))
            # Ensure channel-first format [C, H, W] or [C, D, H, W]
            transforms.append(EnsureChannelFirstd(keys=keys))
        else:
            # For volume-based datasets (HDF5, TIFF volumes), use custom LoadVolumed
            transpose_axes = (
                data_cfg.data_transform.val_transpose
                if data_cfg.data_transform.val_transpose
                else []
            )

            transforms.append(
                LoadVolumed(keys=keys, transpose_axes=transpose_axes if transpose_axes else None)
            )

    nnunet_pre_cfg = getattr(data_cfg, "nnunet_preprocessing", None)
    if mode in {"test", "tune"}:
        source_spacing = _resolve_eval_split().resolution
    elif mode == "val":
        source_spacing = getattr(data_cfg.val, "resolution", None) or getattr(
            data_cfg.train, "resolution", None
        )
    else:
        source_spacing = getattr(data_cfg.train, "resolution", None)

    nnunet_pre_enabled = bool(getattr(nnunet_pre_cfg, "enabled", False))
    if not skip_loading and nnunet_pre_enabled:
        transforms.append(_build_nnunet_preprocess_transform(keys, nnunet_pre_cfg, source_spacing))

    # Apply volumetric split if enabled
    if data_cfg.split_enabled:
        from connectomics.data.dataset.split import ApplyVolumetricSplitd

        transforms.append(ApplyVolumetricSplitd(keys=keys))

    # Apply resize if configured (before cropping)
    image_resize_factors = getattr(data_cfg.image_transform, "resize", None)

    # Prefer mask_transform over data_transform for mask-specific settings.
    mask_cfg = getattr(data_cfg, "mask_transform", None) or data_cfg.data_transform
    mask_resize_factors = None
    if mode in {"test", "tune"} and mask_cfg.resize is not None:
        mask_resize_factors = mask_cfg.resize

    if image_resize_factors is not None and image_resize_factors:
        transforms.append(
            ResizeByFactord(
                keys=["image"],
                scale_factors=image_resize_factors,
                mode="bilinear",
                align_corners=True,
            )
        )
        if "label" in keys:
            transforms.append(
                ResizeByFactord(
                    keys=["label"],
                    scale_factors=image_resize_factors,
                    mode="nearest",
                    align_corners=None,
                )
            )
        # By default, mask follows image resize unless mask_transform explicitly overrides it.
        if "mask" in keys and mask_resize_factors is None:
            transforms.append(
                ResizeByFactord(
                    keys=["mask"],
                    scale_factors=image_resize_factors,
                    mode="nearest",
                    align_corners=None,
                )
            )

    if mask_resize_factors is not None and mask_resize_factors and "mask" in keys:
        transforms.append(
            ResizeByFactord(
                keys=["mask"],
                scale_factors=mask_resize_factors,
                mode="nearest",
                align_corners=None,
            )
        )

    # Optional mask binarization for inference masks (e.g., enforce mask > 0).
    mask_binarize = False
    mask_threshold = 0.0
    if mode in {"test", "tune"}:
        mask_binarize = bool(getattr(mask_cfg, "binarize", False))
        mask_threshold = float(getattr(mask_cfg, "threshold", 0.0))

    if "mask" in keys and mask_binarize:
        transforms.append(
            Lambdad(
                keys=["mask"],
                func=partial(_strict_binarize_mask, threshold=mask_threshold),
            )
        )

    patch_size = tuple(data_cfg.dataloader.patch_size) if data_cfg.dataloader.patch_size else None
    if patch_size and all(size > 0 for size in patch_size):
        transforms.append(
            SpatialPadd(
                keys=keys,
                spatial_size=patch_size,
                constant_values=0.0,
            )
        )

    if mode in {"test", "tune"}:
        context_pad = getattr(data_cfg.data_transform, "pad_size", None)
        if context_pad and any(int(v) > 0 for v in context_pad):
            # Explicit test-time context padding is only for inference inputs.
            # Labels stay in the original FOV; masks get zero-padded to match the image.
            transforms.append(
                BorderPadd(
                    keys=["image"],
                    spatial_border=tuple(int(v) for v in context_pad),
                    mode=getattr(data_cfg.data_transform, "pad_mode", "reflect"),
                )
            )
            if "mask" in keys:
                # Keep mask context empty outside the source FOV.
                transforms.append(
                    BorderPadd(
                        keys=["mask"],
                        spatial_border=tuple(int(v) for v in context_pad),
                        mode="constant",
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
    image_transform = data_cfg.image_transform
    if (not nnunet_pre_enabled) and image_transform.normalize != "none":
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
        # Label transformations (affinity, distance transform, etc.)
        # For test/tune modes: NEVER apply label transforms
        # (keep raw instance labels for evaluation)
        # For val mode: use training label_transform config
        label_cfg = None
        if mode == "val":
            # Validation always uses training label_transform
            if hasattr(data_cfg, "label_transform"):
                label_cfg = data_cfg.label_transform

        # Apply label transforms if configured
        if label_cfg is not None:
            from ..process.build import create_label_transform_pipeline
            from ..process.transforms import SegErosionInstanced

            # Apply instance erosion first if specified
            if hasattr(label_cfg, "erosion") and label_cfg.erosion > 0:
                transforms.append(SegErosionInstanced(keys=["label"], tsz_h=label_cfg.erosion))

            # Build label transform pipeline directly from label_transform config
            label_transform = create_label_transform_pipeline(label_cfg)
            if isinstance(label_transform, Compose):
                transforms.extend(label_transform.transforms)
            else:
                transforms.append(label_transform)

    # NOTE: Do NOT squeeze labels here!
    # - DiceLoss needs (B, 1, H, W) with to_onehot_y=True
    # - CrossEntropyLoss needs (B, H, W)
    # Squeezing is handled in the loss wrapper instead

    # Final conversion to tensor with float32 dtype
    transforms.append(ToTensord(keys=keys, dtype=torch.float32))

    return Compose(transforms)


def build_val_transforms(
    cfg: Config, keys: list[str] = None, skip_loading: bool = False
) -> Compose:
    """
    Build validation transforms from Hydra config.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: auto-detected as ['image', 'label'])
        skip_loading: Skip LoadVolumed (for pre-cached datasets)

    Returns:
        Composed MONAI transforms (no augmentation, center cropping)
    """
    return _build_eval_transforms_impl(cfg, mode="val", keys=keys, skip_loading=skip_loading)


def build_test_transforms(cfg: Config, keys: list[str] = None, mode: str = "test") -> Compose:
    """
    Build test/tune inference transforms from Hydra config.

    Similar to validation transforms but WITHOUT cropping to enable
    sliding window inference on full volumes.

    Args:
        cfg: Hydra Config object
        keys: Keys to transform (default: auto-detected as ['image'] only)
        mode: 'test' or 'tune' to choose the correct eval split

    Returns:
        Composed MONAI transforms (no augmentation, no cropping)
    """
    return _build_eval_transforms_impl(cfg, mode=mode, keys=keys)


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

    def should_augment(aug_name: str, aug_enabled: Optional[bool]) -> bool:
        """Check if aug should apply: 'none'=off, 'some'=opt-in, 'all'=opt-out."""
        if preset == "none":
            return False
        if preset == "all":
            return aug_enabled is not False
        # preset == "some"
        return aug_enabled is True

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

    # EM-specific defect augmentations.
    # When defect_mutex is True, at most one defect fires per sample (DeepEM
    # Blend(mutex) behaviour).  Otherwise they are applied independently.
    defect_transforms = []

    if should_augment("misalignment", aug_cfg.misalignment.enabled):
        defect_transforms.append(
            RandMisAlignmentd(
                keys=["image"],
                prob=aug_cfg.misalignment.prob,
                displacement=aug_cfg.misalignment.displacement,
                rotate_ratio=aug_cfg.misalignment.rotate_ratio,
            )
        )

    if should_augment("missing_section", aug_cfg.missing_section.enabled):
        defect_transforms.append(
            RandMissingSectiond(
                keys=["image"],
                prob=aug_cfg.missing_section.prob,
                num_sections=aug_cfg.missing_section.num_sections,
                full_section_prob=aug_cfg.missing_section.full_section_prob,
                partial_ratio_range=aug_cfg.missing_section.partial_ratio_range,
                fill_value_range=aug_cfg.missing_section.fill_value_range,
            )
        )

    if should_augment("motion_blur", aug_cfg.motion_blur.enabled):
        defect_transforms.append(
            RandMotionBlurd(
                keys=["image"],
                prob=aug_cfg.motion_blur.prob,
                sections=aug_cfg.motion_blur.sections,
                kernel_size=aug_cfg.motion_blur.kernel_size,
                sigma_range=aug_cfg.motion_blur.sigma_range,
                full_section_prob=aug_cfg.motion_blur.full_section_prob,
                partial_ratio_range=aug_cfg.motion_blur.partial_ratio_range,
            )
        )

    if defect_transforms:
        if getattr(aug_cfg, "defect_mutex", False) and len(defect_transforms) > 1:
            # Mutual exclusion: randomly pick one defect per sample.
            transforms.append(OneOf(transforms=defect_transforms))
        else:
            transforms.extend(defect_transforms)

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
                keys=["image"],
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


__all__ = [
    "build_train_transforms",
    "build_val_transforms",
    "build_test_transforms",
]
