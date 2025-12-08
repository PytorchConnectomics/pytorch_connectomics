#!/usr/bin/env python
"""
Standalone training script for CellMap Segmentation Challenge using PyTC models.

This script uses:
- cellmap-data for data loading (official challenge library)
- PyTC models (MONAI model zoo)
- PyTorch Lightning for training orchestration

NO modifications to PyTC core required.

Usage:
    python scripts/cellmap/train_cellmap.py configs/mednext_cos7.py
    python scripts/cellmap/train_cellmap.py configs/mednext_mito.py

Requirements:
    pip install cellmap-data cellmap-segmentation-challenge
"""

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

# Add PyTC to path
PYTC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PYTC_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# CellMap data loading (official)
from cellmap_segmentation_challenge.utils import (
    get_dataloader,  # Official dataloader factory
    make_datasplit_csv,  # Auto-generate train/val split
    make_s3_datasplit_csv,
    get_tested_classes,  # Official class list
    load_safe_config,
)
from cellmap_segmentation_challenge import config as cellmap_cfg
from upath import UPath

# ---------------------------------------------------------------------------
# Compatibility patch:
# xarray-tensorstore>=0.3.0 expects callers to pass the zarr_format argument to
# _zarr_spec_from_path. cellmap-data currently calls it with a single argument.
# Monkey patch the helper so we fall back to zarr v2 automatically.
# ---------------------------------------------------------------------------
try:
    import inspect
    import xarray_tensorstore as xt

    _orig_zarr_spec = xt._zarr_spec_from_path
    _spec_sig = inspect.signature(_orig_zarr_spec)
    needs_patch = (
        len(_spec_sig.parameters) >= 2
        and list(_spec_sig.parameters.values())[1].default is inspect._empty
    )

    if needs_patch:
        def _compat_zarr_spec(path: str, zarr_format: int | None = None):
            if zarr_format is None:
                zarr_format = 2
            return _orig_zarr_spec(path, zarr_format)

        xt._zarr_spec_from_path = _compat_zarr_spec
except Exception as patch_err:  # pragma: no cover - best-effort guard
    print(f"[WARN] Failed to patch xarray_tensorstore: {patch_err}")

# PyTC model building (import only, no modification)
from connectomics.models import build_model
from connectomics.models.loss import create_loss

# Import config utilities

class CellMapBalancedLoss(nn.Module):
    """Combines class-balanced BCE with soft Dice while masking NaNs."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target).float()
        valid_voxels = mask.sum()

        if valid_voxels == 0:
            # No supervision in this patch; return zero loss but keep gradients defined.
            return logits.sum() * 0.0

        target = target.nan_to_num(0.0)

        # Clamp logits to prevent numerical instability (overflow in sigmoid/BCE)
        logits = torch.clamp(logits, -10.0, 10.0)
        # Compute per-channel pos_weight to counter extreme imbalance
        dims = (0, 2, 3, 4) if logits.dim() == 5 else tuple(range(logits.dim() - 1))
        pos = (target * mask).sum(dim=dims)
        neg = ((1.0 - target) * mask).sum(dim=dims)

        # For sparse segmentation (like CellMap), compute loss on all classes
        # even when some classes have no positive examples in a batch

        # Check if any classes have positive examples
        valid_classes = (pos > 0)

        # Debug: check for potential issues
        if torch.isnan(pos).any() or torch.isinf(pos).any() or torch.isnan(neg).any() or torch.isinf(neg).any():
            # If pos/neg have NaN/inf, return safe loss
            return logits.sum() * 0.0 + 1e-6

        if not valid_classes.any():
            # No classes have positive examples; return small finite loss to keep gradients flowing
            return logits.sum() * 0.0 + 1e-6

        # For multi-class sparse segmentation, extreme pos_weight can cause numerical issues
        # Use a simpler approach: uniform weight for all classes to avoid extreme imbalance
        pos_weight = torch.ones_like(pos)

        # Compute per-class losses and average across classes
        num_classes = logits.shape[1]
        bce_losses = []
        dice_losses = []

        for c in range(num_classes):
            logits_c = logits[:, c]
            target_c = target[:, c]
            mask_c = mask[:, c] if mask is not None else None

            # BCE for this class with pos_weight to handle class imbalance
            if mask_c is not None and mask_c.sum() > 0:
                # Get pos_weight as a tensor (0D tensor) for this class
                pos_weight_c = pos_weight[c]  # Keep as tensor, not scalar
                bce_c = F.binary_cross_entropy_with_logits(
                    logits_c,
                    target_c,
                    weight=mask_c,
                    pos_weight=pos_weight_c,  # Add pos_weight to handle class imbalance
                    reduction="mean",  # Mean for this class
                )
                # Safety check: if BCE is not finite, use 0 but maintain gradient connection
                if not torch.isfinite(bce_c):
                    bce_c = logits_c.sum() * 0.0
                bce_losses.append(bce_c)
            else:
                bce_losses.append(logits_c.sum() * 0.0)

            # Dice for this class
            probs_c = torch.sigmoid(logits_c) * (mask_c if mask_c is not None else torch.ones_like(logits_c))
            # Clamp probabilities to prevent numerical issues
            probs_c = torch.clamp(probs_c, 1e-7, 1.0 - 1e-7)
            target_masked_c = target_c * (mask_c if mask_c is not None else torch.ones_like(target_c))

            # Use correct spatial dimensions for the per-class tensor (4D: B, D, H, W)
            # After removing channel dim, spatial dims are (1, 2, 3) for 4D tensor
            spatial_dims = tuple(range(1, logits_c.dim()))
            intersection_c = (probs_c * target_masked_c).sum(dim=spatial_dims)  # Sum spatial dims, shape: (B,)
            denom_c = probs_c.sum(dim=spatial_dims) + target_masked_c.sum(dim=spatial_dims)  # Shape: (B,)

            # Replace any NaN/inf values with 0 to prevent propagation
            intersection_c = intersection_c.nan_to_num(0.0)
            denom_c = denom_c.nan_to_num(0.0)

            # Compute Dice per batch element, handling edge cases
            # Use torch.where to handle denom_c == 0 case, and ensure no NaN in computation
            numerator = 2.0 * intersection_c + self.eps
            denominator = denom_c + self.eps
            
            # Replace any NaN/inf in numerator/denominator
            numerator = numerator.nan_to_num(0.0)
            denominator = denominator.nan_to_num(self.eps)  # Ensure denominator is never 0
            
            dice_per_batch = torch.where(
                denom_c > 0,
                1.0 - numerator / denominator,
                torch.zeros_like(intersection_c)
            )
            
            # Replace any NaN/inf in dice_per_batch before averaging
            dice_per_batch = dice_per_batch.nan_to_num(0.0)
            
            # Average over batch dimension to get scalar for this class
            dice_c = dice_per_batch.mean()
            
            # Final safety check: if Dice is not finite, use 0 but maintain gradient connection
            if not torch.isfinite(dice_c):
                dice_c = logits_c.sum() * 0.0
            dice_losses.append(dice_c)

        # Ensure all individual losses are finite before stacking (use gradient-connected replacements)
        bce_losses = [l if torch.isfinite(l) else (logits.sum() * 0.0) for l in bce_losses]
        dice_losses = [l if torch.isfinite(l) else (logits.sum() * 0.0) for l in dice_losses]
        
        # Average across classes
        bce = torch.stack(bce_losses).mean()
        dice = torch.stack(dice_losses).mean()

        # Safety check for averaged losses
        if not torch.isfinite(bce) or not torch.isfinite(dice):
            return logits.sum() * 0.0 + 1e-6

        loss = self.bce_weight * bce + self.dice_weight * dice

        # Final safety check: if loss is NaN or inf, return a small finite loss that maintains gradients
        if not torch.isfinite(loss):
            return logits.sum() * 0.0 + 1e-6

        return loss


class CellMapLightningModule(pl.LightningModule):
    """
    Minimal Lightning wrapper around PyTC models for CellMap training.

    Uses PyTC models as-is, no modifications needed.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer_config,
        scheduler_config=None,
        classes=None,
        target_shape=None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.classes = classes or []
        self.target_shape = target_shape

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Cast raw EM voxels to float and normalize if they arrive as uint8."""
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        else:
            images = images.float()
        return images

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Ensure supervision tensors are float32 before loss computations."""
        if labels.dtype != torch.float32:
            labels = labels.float()
        return labels

    def _maybe_resample(self, images: torch.Tensor, labels: torch.Tensor):
        """Optionally resample images/labels to a fixed shape to avoid scale filtering."""
        if self.target_shape is None:
            return images, labels
        # Expect 5D tensors (B, C, D, H, W). Use nearest for labels.
        target = tuple(self.target_shape)
        images_rs = torch.nn.functional.interpolate(
            images, size=target, mode="trilinear", align_corners=False
        )
        labels_rs = torch.nn.functional.interpolate(
            labels, size=target, mode="nearest"
        )
        return images_rs, labels_rs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = self._prepare_images(batch['input'])
        labels = self._prepare_labels(batch['output'])
        batch_size = images.shape[0]

        images, labels = self._maybe_resample(images, labels)

        # Let the loss function handle all edge cases (empty batches, NaN, etc.)
        # Don't skip batches - the loss function returns safe finite values for all cases
        predictions = self(images)
        predictions = self._normalize_predictions(predictions, labels.shape[-3:])
        loss = self.criterion(predictions, labels)

        # The loss function should always return a finite value, but double-check
        if not torch.isfinite(loss):
            # This should never happen if loss function is correct, but handle gracefully
            loss = predictions.sum() * 0.0 + 1e-6

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = self._prepare_images(batch['input'])
        labels = self._prepare_labels(batch['output'])
        batch_size = images.shape[0]

        images, labels = self._maybe_resample(images, labels)

        # Let the loss function handle all edge cases (empty batches, NaN, etc.)
        # Don't skip batches - the loss function returns safe finite values for all cases
        valid_mask = torch.isfinite(labels).float()
        labels = labels.nan_to_num(0.0)

        predictions = self(images)
        predictions = self._normalize_predictions(predictions, labels.shape[-3:])
        loss = self.criterion(predictions, labels)

        # The loss function should always return a finite value, but double-check
        if not torch.isfinite(loss):
            # This should never happen if loss function is correct, but handle gracefully
            loss = predictions.sum() * 0.0 + 1e-6

        # Compute Dice score per class
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()

            # Average Dice across classes
            dice_scores = []
            eps = 1e-7
            for c in range(predictions.shape[1]):
                mask_c = valid_mask[:, c]
                valid_vox = mask_c.sum()
                if valid_vox == 0:
                    dice = torch.tensor(1.0, device=labels.device)
                else:
                    pred_c = pred_binary[:, c] * mask_c
                    label_c = labels[:, c] * mask_c
                    intersection = (pred_c * label_c).sum()
                    denom = pred_c.sum() + label_c.sum()
                    dice = (2. * intersection + eps) / (denom + eps)
                dice_scores.append(dice)

                # Log per-class Dice if we have class names
                if c < len(self.classes):
                    self.log(
                        f'val/dice_{self.classes[c]}',
                        dice,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

            mean_dice = torch.stack(dice_scores).mean()

        self.log('val/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/dice', mean_dice, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def _normalize_predictions(self, predictions, target_shape):
        """
        Convert model outputs to a single tensor aligned with target_shape.

        MedNeXt (and other deep-supervision models) can return a dict of multi-scale logits.
        We upsample each prediction to the target shape and average them so downstream loss
        functions see consistent tensors.
        """
        if not isinstance(predictions, dict):
            return predictions

        merged = []
        for tensor in predictions.values():
            if tensor.shape[-3:] != tuple(target_shape):
                tensor = F.interpolate(
                    tensor,
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            merged.append(tensor)

        if not merged:
            raise ValueError("Prediction dictionary is empty; cannot compute loss.")

        return torch.mean(torch.stack(merged, dim=0), dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config.get('weight_decay', 1e-5),
        )

        if self.scheduler_config is None or self.scheduler_config.get('name') == 'constant':
            return optimizer

        scheduler_name = self.scheduler_config.get('name', 'cosine')

        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('min_lr', 1e-6),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def train_cellmap(config_path: str, data_root: str | None = None, target_shape=None):
    """
    Main training function using CellMap's official tools + PyTC models.

    Args:
        config_path: Path to Python config file (CellMap style)
        data_root: Optional override for the CellMap dataset root
    """
    # Load config (CellMap's safe config loader)
    print(f"Loading config from: {config_path}")
    config = load_safe_config(config_path)
    base_experiment_path = getattr(config, "base_experiment_path", None)
    if base_experiment_path is not None:
        base_experiment_path = UPath(base_experiment_path)

    def _resolve_path(value, default):
        """Resolve relative paths against the configured experiment root."""
        if value is None:
            value = default
        if value is None:
            return None
        path = UPath(value)
        if not path.is_absolute() and base_experiment_path is not None:
            path = base_experiment_path / path
        return path.path

    def _infer_scale(filter_value, array_info):
        """Infer a scale tuple based on filter settings and array metadata."""
        if filter_value in (False, None):
            return None

        def _extract(info):
            if isinstance(info, Mapping) and "scale" in info:
                return info["scale"]
            if isinstance(info, Mapping):
                for value in info.values():
                    result = _extract(value)
                    if result is not None:
                        return result
            return None

        if filter_value is True:
            scale = _extract(array_info)
            return tuple(scale) if scale is not None else None
        if isinstance(filter_value, (int, float)):
            return (float(filter_value),) * 3
        if isinstance(filter_value, Sequence) and not isinstance(
            filter_value, (str, bytes)
        ):
            seq = list(filter_value)
            if len(seq) == 1:
                return tuple(seq * 3)
            return tuple(seq)
        return tuple(filter_value)


    # Allow CLI overrides
    if data_root:
        setattr(config, "data_root", data_root)
    if target_shape:
        setattr(config, "target_shape", target_shape)

    # Extract config values
    model_name = getattr(config, "model_name", "mednext")
    classes = getattr(config, "classes", get_tested_classes())
    learning_rate = getattr(config, "learning_rate", 1e-3)
    batch_size = getattr(config, "batch_size", 2)
    batch_size = getattr(config, "train_micro_batch_size_per_gpu", batch_size)
    max_epochs = getattr(config, "epochs", 1000)
    num_gpus = getattr(config, "num_gpus", 1)
    precision = getattr(config, "precision", "16-mixed")
    validation_prob = getattr(config, "validation_prob", 0.15)
    filter_by_scale = getattr(config, "filter_by_scale", False)
    force_classes_mode = getattr(config, "force_all_classes", "both")
    use_s3 = getattr(config, "use_s3", False)
    weighted_sampler = getattr(config, "weighted_sampler", False)
    use_mutual_exclusion = getattr(config, "use_mutual_exclusion", False)
    train_raw_value_transforms = getattr(
        config, "train_raw_value_transforms", None
    )
    val_raw_value_transforms = getattr(config, "val_raw_value_transforms", None)
    target_value_transforms = getattr(config, "target_value_transforms", None)
    dataloader_kwargs = dict(getattr(config, "dataloader_kwargs", {}))
    datasplit_kwargs = dict(getattr(config, "datasplit_kwargs", {}))
    train_batches_per_epoch = getattr(config, "train_batches_per_epoch", None)
    val_batches_per_epoch = getattr(config, "val_batches_per_epoch", None)

    output_dir = _resolve_path(
        getattr(config, "output_dir", "outputs/cellmap"), "outputs/cellmap"
    )
    os.makedirs(output_dir, exist_ok=True)

    datasplit_path = _resolve_path(
        getattr(config, "datasplit_path", None),
        os.path.join(output_dir, "datasplit.csv"),
    )
    datasplit_dir = os.path.dirname(datasplit_path)
    if datasplit_dir:
        os.makedirs(datasplit_dir, exist_ok=True)

    tensorboard_dir = _resolve_path(
        getattr(config, "logs_save_path", None),
        os.path.join(output_dir, "tensorboard"),
    )
    checkpoint_dir = _resolve_path(
        getattr(config, "checkpoint_dir", None),
        os.path.join(output_dir, "checkpoints"),
    )
    if tensorboard_dir:
        os.makedirs(tensorboard_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    input_array_info = getattr(
        config,
        "input_array_info",
        {
            "shape": (128, 128, 128),
            "scale": (8, 8, 8),
        },
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    spatial_transforms = getattr(
        config,
        "spatial_transforms",
        {
            "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},
            "transpose": {"axes": ["x", "y", "z"]},
            "rotate": {
                "axes": {
                    "x": [-180, 180],
                    "y": [-180, 180],
                    "z": [-180, 180],
                }
            },
        },
    )
    iterations_per_epoch = getattr(config, "iterations_per_epoch", None)
    validation_time_limit = getattr(config, "validation_time_limit", None)
    validation_batch_limit = getattr(config, "validation_batch_limit", None)

    print(f"Training configuration:")
    print(f"  Model: {model_name}")
    print(f"  Classes: {classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Precision: {precision}")
    iter_msg = (
        iterations_per_epoch
        if iterations_per_epoch is not None
        else "auto (full dataset shuffle)"
    )
    print(f"  Iterations per epoch: {iter_msg}")
    print(f"  Weighted sampler: {weighted_sampler}")
    if train_batches_per_epoch is not None:
        print(f"  Trainer train batches/epoch: {train_batches_per_epoch}")
    if val_batches_per_epoch is not None:
        print(f"  Trainer val batches/epoch: {val_batches_per_epoch}")
    if target_shape:
        print(f"  Target resample shape: {target_shape}")

    # Resolve data root override (defaults to package SEARCH_PATH under repo/data)
    data_root = getattr(config, "data_root", None)
    target_shape = getattr(config, "target_shape", target_shape)
    search_path = cellmap_cfg.SEARCH_PATH
    if data_root:
        search_path = os.path.normpath(
            os.path.join(data_root, "{dataset}/{dataset}.zarr/recon-1/{name}")
        )
        print(f"Using data root: {data_root}")
    else:
        print(f"Using default CellMap search path: {search_path}")

    # Generate datasplit CSV if it doesn't exist
    if not os.path.exists(datasplit_path):
        print(f"Generating datasplit CSV: {datasplit_path}")
        scale_filter = None
        if not target_shape:
            scale_filter = _infer_scale(filter_by_scale, input_array_info)
        if force_classes_mode not in {"train", "validate", "both", None}:
            raise ValueError(
                "force_all_classes must be one of {'train', 'validate', 'both', None}"
            )
        effective_force_mode = force_classes_mode or "both"
        print(f"Forcing class coverage in datasplit: {effective_force_mode}")
        datasplit_args = dict(datasplit_kwargs)
        datasplit_args.setdefault("search_path", search_path)
        datasplit_args.setdefault("csv_path", datasplit_path)
        if use_s3:
            make_s3_datasplit_csv(
                classes=classes,
                scale=scale_filter,
                force_all_classes=effective_force_mode,
                validation_prob=validation_prob,
                **datasplit_args,
            )
        else:
            make_datasplit_csv(
                classes=classes,
                scale=scale_filter,
                force_all_classes=effective_force_mode,
                validation_prob=validation_prob,
                **datasplit_args,
            )
    else:
        print(f"Using existing datasplit: {datasplit_path}")

    # Get dataloaders (CellMap's official dataloader)
    print("Creating dataloaders...")
    dataloader_args = dict(
        datasplit_path=datasplit_path,
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        target_value_transforms=target_value_transforms,
        train_raw_value_transforms=train_raw_value_transforms,
        val_raw_value_transforms=val_raw_value_transforms,
        random_validation=bool(validation_time_limit or validation_batch_limit),
        use_mutual_exclusion=use_mutual_exclusion,
        weighted_sampler=weighted_sampler,
        **dataloader_kwargs,
    )

    # Always pass iterations_per_epoch (even None) so we override the default 1000
    dataloader_args["iterations_per_epoch"] = iterations_per_epoch

    train_loader, val_loader = get_dataloader(**dataloader_args)

    # Wrap loaders to satisfy PyTorch Lightning's expectation of a batch_sampler attribute
    class _LightningDataLoaderWrapper:
        """Adds batch_sampler attribute so custom loaders work with Lightning."""

        def __init__(self, loader):
            self._loader = loader
            self.batch_sampler = None  # Lightning inspects this attribute

        def __iter__(self):
            return iter(self._loader)

        def __len__(self):
            inner_loader = getattr(self._loader, "loader", None)
            if inner_loader is not None:
                try:
                    return len(inner_loader)
                except TypeError:
                    pass

            if hasattr(self._loader, "__len__"):
                try:
                    return len(self._loader)
                except TypeError:
                    pass

            iterations = getattr(self._loader, "iterations_per_epoch", None)
            if isinstance(iterations, int) and iterations > 0:
                return iterations

            dataset = getattr(self._loader, "dataset", None)
            if dataset is not None:
                try:
                    return len(dataset)
                except TypeError:
                    pass

            raise TypeError(
                "Wrapped loader does not define a finite length required for progress bars."
            )

        def __getattr__(self, name):
            return getattr(self._loader, name)

    train_loader = _LightningDataLoaderWrapper(train_loader)
    if val_loader is not None:
        val_loader = _LightningDataLoaderWrapper(val_loader)

    # Build model using PyTC's model factory (MONAI models)
    print(f"Building model: {model_name}")

    # Create minimal config for PyTC's build_model
    from omegaconf import OmegaConf
    # Get input shape from config (D, H, W) for 3D
    input_shape = input_array_info.get('shape', (64, 64, 64))
    model_config = OmegaConf.create({
        'model': {
            'architecture': model_name,
            'in_channels': 1,
            'out_channels': len(classes),
            'input_size': list(input_shape),  # [D, H, W] for 3D
            'mednext_size': getattr(config, 'mednext_size', 'B'),
            'mednext_kernel_size': getattr(config, 'mednext_kernel_size', 5),
            'deep_supervision': getattr(config, 'deep_supervision', True),
        }
    })

    model = build_model(model_config)
    print(f"Model built successfully")

    # Create loss (balanced BCE + Dice, NaN-aware)
    print("Creating loss function...")
    criterion = CellMapBalancedLoss(bce_weight=0.7, dice_weight=0.3)

    # Create Lightning module
    lit_model = CellMapLightningModule(
        model=model,
        criterion=criterion,
        optimizer_config={'lr': learning_rate, 'weight_decay': 1e-5},
        scheduler_config=getattr(config, 'scheduler_config', {'name': 'constant'}),
        classes=classes,
        target_shape=target_shape,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{{epoch:02d}}-{{val/dice:.3f}}',
        monitor='val/dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val/dice',
        patience=50,
        mode='max',
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup loggers
    tb_logger = TensorBoardLogger(
        tensorboard_dir,
        name=model_name,
    )

    # Create trainer
    limit_train_batches = (
        train_batches_per_epoch if train_batches_per_epoch is not None else 1.0
    )
    if validation_batch_limit is not None:
        limit_val_batches = validation_batch_limit
    elif val_batches_per_epoch is not None:
        limit_val_batches = val_batches_per_epoch
    else:
        limit_val_batches = 1.0

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=num_gpus,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    # Train!
    print("Starting training...")
    print(f"Monitor progress: tensorboard --logdir {tensorboard_dir}")
    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/dice: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PyTC models on CellMap data')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument(
        '--data-root',
        type=str,
        help='Override dataset root (e.g., /projects/weilab/dataset/cellmap)',
        default=None,
    )
    parser.add_argument(
        '--target-shape',
        nargs=3,
        metavar=('D', 'H', 'W'),
        type=int,
        help='Resample input/label to this shape (e.g., --target-shape 128 128 128) to include all scales',
        default=None,
    )
    args = parser.parse_args()

    target_shape = tuple(args.target_shape) if args.target_shape else None
    train_cellmap(args.config, data_root=args.data_root, target_shape=target_shape)
