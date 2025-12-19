"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing

The implementation delegates to specialized modules:
- connectomics.training.deep_supervision: Deep supervision and multi-task learning
- connectomics.inference: Sliding window inference and test-time augmentation
- connectomics.training.debugging: NaN detection and debugging utilities
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torchmetrics

# Import existing components
from ...models import build_model
from ...models.loss import create_loss
from ...models.solver import build_optimizer, build_lr_scheduler
from ...config import Config

# Import training/inference components
from ..deep_supervision import DeepSupervisionHandler, match_target_to_output
from ..debugging import DebugManager
from ..loss_balancing import build_loss_weighter
from ...inference import (
    InferenceManager,
    apply_save_prediction_transform,
    apply_postprocessing,
    apply_decode_mode,
    resolve_output_filenames,
    write_outputs,
)


class ConnectomicsModule(pl.LightningModule):
    """
    PyTorch Lightning module for connectomics tasks.

    This module provides automatic training features including:
    - Distributed training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    - Learning rate scheduling

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model (if None, builds from config)
    """

    def __init__(
        self,
        cfg: Union[Config, DictConfig],
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['model'])

        # Build model
        self.model = model if model is not None else self._build_model(cfg)

        # Build loss functions
        self.loss_functions = self._build_losses(cfg)
        self.loss_weights = cfg.model.loss_weights if hasattr(cfg.model, 'loss_weights') else [1.0] * len(self.loss_functions)

        # Build adaptive loss weighter (for multi-task learning)
        num_tasks = len(cfg.model.multi_task_config) if hasattr(cfg.model, 'multi_task_config') and cfg.model.multi_task_config else 1
        self.loss_weighter = build_loss_weighter(cfg, num_tasks, model=self.model)

        # Track multi-task configuration state for downstream logic/tests
        self.multi_task_config = getattr(cfg.model, 'multi_task_config', None)
        self.multi_task_enabled = bool(self.multi_task_config)

        # Enable inline NaN detection (can be disabled via config)
        self.enable_nan_detection = getattr(cfg.model, 'enable_nan_detection', True)
        self.debug_on_nan = getattr(cfg.model, 'debug_on_nan', True)

        # Activation clamping to prevent inf (can be configured)
        self.clamp_activations = getattr(cfg.model, 'clamp_activations', False)
        self.clamp_min = getattr(cfg.model, 'clamp_min', -10.0)
        self.clamp_max = getattr(cfg.model, 'clamp_max', 10.0)

        # Initialize specialized handlers
        self.deep_supervision_handler = DeepSupervisionHandler(
            cfg=cfg,
            loss_functions=self.loss_functions,
            loss_weights=self.loss_weights,
            enable_nan_detection=self.enable_nan_detection,
            debug_on_nan=self.debug_on_nan,
            loss_weighter=self.loss_weighter,
        )

        self.inference_manager = InferenceManager(
            cfg=cfg,
            model=self.model,
            forward_fn=self.forward,
        )

        self.debug_manager = DebugManager(model=self.model)

        # Test metrics (initialized lazily during test mode if specified in config)
        self.test_jaccard = None
        self.test_dice = None
        self.test_accuracy = None
        self.test_adapted_rand = None  # Adapted Rand error (instance segmentation metric)

        # Prediction saving state
        self._prediction_save_counter = 0  # Track number of samples saved

    def _build_model(self, cfg) -> nn.Module:
        """Build model from configuration."""
        return build_model(cfg)

    def _build_losses(self, cfg) -> nn.ModuleList:
        """Build loss functions from configuration."""
        loss_names = cfg.model.loss_functions if hasattr(cfg.model, 'loss_functions') else ['DiceLoss']
        loss_kwargs_list = cfg.model.loss_kwargs if hasattr(cfg.model, 'loss_kwargs') else [{}] * len(loss_names)

        losses = nn.ModuleList()
        for loss_name, kwargs in zip(loss_names, loss_kwargs_list):
            loss_fn = create_loss(loss_name, **kwargs)
            losses.append(loss_fn)

        return losses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightning forward pass that delegates to the underlying model.

        This is required so Lightning can execute the module during training/inference.
        """
        return self.model(x)

    def _setup_test_metrics(self):
        """Initialize test metrics based on test or inference config."""
        # Check test.evaluation first, then fall back to inference.evaluation
        evaluation_config = None
        if hasattr(self.cfg, 'test') and self.cfg.test and hasattr(self.cfg.test, 'evaluation'):
            evaluation_config = self.cfg.test.evaluation
        elif hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            evaluation_config = self.cfg.inference.evaluation
        
        if not evaluation_config:
            return

        # Check if evaluation is enabled
        enabled = evaluation_config.get('enabled', False) if isinstance(evaluation_config, dict) else getattr(evaluation_config, 'enabled', False)
        if not enabled:
            return

        metrics = evaluation_config.get('metrics', None) if isinstance(evaluation_config, dict) else getattr(evaluation_config, 'metrics', None)
        if metrics is None:
            return

        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2

        # Create only the specified metrics
        if 'jaccard' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_jaccard = torchmetrics.JaccardIndex(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)
        if 'dice' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_dice = torchmetrics.Dice(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(self.device)
        if 'accuracy' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_accuracy = torchmetrics.Accuracy(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
        if 'adapted_rand' in metrics:
            from ...metrics.metrics_seg import AdaptedRandError
            self.test_adapted_rand = AdaptedRandError().to(self.device)

    def _invert_save_prediction_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Invert the save_prediction transform to convert saved predictions back to [0,1] range.

        This is needed when loading intermediate predictions that were saved with
        intensity_scale and intensity_dtype applied. We need to convert them back
        to the original [0,1] float range for decoding.

        Args:
            data: Saved predictions (e.g., uint8 in [0, 255])

        Returns:
            Predictions in original [0,1] float range
        """
        if not hasattr(self.cfg, "inference") or not hasattr(self.cfg.inference, "save_prediction"):
            # No save_prediction config, assume data is already in correct format
            return data.astype(np.float32)

        save_pred_cfg = self.cfg.inference.save_prediction

        # Get the scale and dtype that were used for saving
        intensity_scale = getattr(save_pred_cfg, "intensity_scale", None)
        intensity_dtype = getattr(save_pred_cfg, "intensity_dtype", None)

        # Convert to float first
        data = data.astype(np.float32)

        # Invert the scaling if it was applied
        # Note: intensity_scale < 0 means scaling was disabled, so no inversion needed
        if intensity_scale is not None and intensity_scale > 0 and intensity_scale != 1.0:
            data = data / float(intensity_scale)
            print(f"  🔄 Inverted intensity scaling by {intensity_scale}")
        elif intensity_scale is not None and intensity_scale < 0:
            print(f"  ℹ️  Intensity scaling was disabled (scale={intensity_scale}), no inversion needed")

        return data

    def _resolve_test_output_config(self, batch: Dict[str, Any]) -> tuple[str, Optional[str], str, List[str]]:
        """Determine mode, output dir, cache suffix, and filenames for test/tune."""
        mode = "test"
        output_dir_value = None
        cache_suffix = "_prediction.h5"

        if (
            hasattr(self.cfg, "tune")
            and self.cfg.tune
            and hasattr(self.cfg.tune, "output")
            and self.cfg.tune.output.output_pred is not None
        ):
            mode = "tune"
            output_dir_value = self.cfg.tune.output.output_pred
            cache_suffix = self.cfg.tune.output.cache_suffix
        elif hasattr(self.cfg, "test") and hasattr(self.cfg.test, "data"):
            output_dir_value = getattr(self.cfg.test.data, "output_path", None)
            cache_suffix = getattr(self.cfg.test.data, "cache_suffix", "_prediction.h5")

        filenames = resolve_output_filenames(self.cfg, batch, global_step=self.global_step)
        return mode, output_dir_value, cache_suffix, filenames

    def _load_cached_predictions(
        self, output_dir_value: Optional[str], filenames: List[str], cache_suffix: str, mode: str
    ):
        """Attempt to load cached predictions from disk."""
        if not output_dir_value:
            return None, False, cache_suffix

        output_dir = Path(output_dir_value)
        existing_predictions = []
        loaded_suffix = cache_suffix
        all_exist = True

        for filename in filenames:
            from connectomics.data.io import read_hdf5
            pred_file = output_dir / f"{filename}{cache_suffix}"
            if not pred_file.exists() and mode == "test" and cache_suffix != "_tta_prediction.h5":
                tta_pred_file = output_dir / f"{filename}_tta_prediction.h5"
                if tta_pred_file.exists():
                    pred_file = tta_pred_file
                    loaded_suffix = "_tta_prediction.h5"

            if pred_file.exists():
                try:
                    pred = read_hdf5(str(pred_file), dataset="main")
                    existing_predictions.append(pred)
                except Exception as e:
                    print(f"  ⚠️  Failed to load {pred_file}: {e}, will re-run inference")
                    all_exist = False
                    break
            else:
                all_exist = False
                break

        if all_exist and len(existing_predictions) == len(filenames):
            print(
                f"  ✅ All prediction files exist! Loading {len(existing_predictions)} predictions and skipping inference."
            )
            if len(existing_predictions) == 1:
                predictions_np = existing_predictions[0]
                if predictions_np.ndim < 4:
                    predictions_np = predictions_np[np.newaxis, ...]
            else:
                predictions_np = np.stack(
                    [p[np.newaxis, ...] if p.ndim < 4 else p for p in existing_predictions], axis=0
                )
            return predictions_np, True, loaded_suffix

        return None, False, loaded_suffix

    def _compute_test_metrics(self, decoded_predictions: np.ndarray, labels: torch.Tensor, volume_name: str = None):
        """Update configured torchmetrics using decoded predictions and print per-volume metrics."""
        pred_tensor = torch.from_numpy(decoded_predictions).float().to(self.device)
        labels_tensor = labels.float()

        # Remove batch and channel dimensions
        pred_tensor = pred_tensor.squeeze()
        labels_tensor = labels_tensor.squeeze()

        # Ensure both tensors have the same shape
        if pred_tensor.shape != labels_tensor.shape:
            print(f"  ⚠️  Shape mismatch: pred={pred_tensor.shape}, labels={labels_tensor.shape}")

            # Try to align dimensions
            if pred_tensor.ndim != labels_tensor.ndim:
                if pred_tensor.ndim == labels_tensor.ndim - 1:
                    pred_tensor = pred_tensor.unsqueeze(0)
                elif labels_tensor.ndim == pred_tensor.ndim - 1:
                    labels_tensor = labels_tensor.unsqueeze(0)

            # If still mismatched after dimension alignment, skip metrics
            if pred_tensor.shape != labels_tensor.shape:
                print(f"  ❌ Cannot compute metrics: incompatible shapes after alignment")
                print(f"     pred={pred_tensor.shape}, labels={labels_tensor.shape}")
                return

        # Compute per-volume metrics (print immediately)
        volume_prefix = f"[{volume_name}] " if volume_name else ""

        # Determine if this is instance segmentation or binary/semantic segmentation
        # Instance segmentation: predictions have integer instance IDs (0, 1, 2, ..., N)
        # Binary/semantic segmentation: predictions are probabilities [0, 1] or logits
        is_instance_segmentation = (
            pred_tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
            or (pred_tensor.dtype == torch.float32 and pred_tensor.max() > 1.0)
        )

        if is_instance_segmentation:
            # For instance segmentation: use instance IDs directly
            pred_instances = pred_tensor.long()
            labels_instances = labels_tensor.long()

            # Adapted Rand Error is for instance segmentation
            if hasattr(self, "test_adapted_rand") and isinstance(self.test_adapted_rand, torchmetrics.Metric):
                from ...metrics.metrics_seg import AdaptedRandError
                per_volume_metric = AdaptedRandError().to(self.device)
                per_volume_metric.update(pred_instances.cpu(), labels_instances.cpu())
                adapted_rand_value = per_volume_metric.compute()
                print(f"  {volume_prefix}Adapted Rand Error: {adapted_rand_value.item():.6f}")

                # Update running metric for epoch-level aggregation
                self.test_adapted_rand.update(pred_instances.cpu(), labels_instances.cpu())
                self.log("test_adapted_rand", self.test_adapted_rand, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        else:
            # For binary/semantic segmentation: binarize predictions
            if pred_tensor.max() <= 1.0:
                pred_binary = (pred_tensor > 0.5).long()
            else:
                pred_binary = (torch.sigmoid(pred_tensor) > 0.5).long()

            labels_binary = (labels_tensor > 0.5).long() if labels_tensor.max() <= 1.0 else labels_tensor.long()

            if hasattr(self, "test_jaccard") and self.test_jaccard is not None:
                jaccard_value = torchmetrics.functional.jaccard_index(
                    pred_binary, labels_binary, task='binary'
                )
                print(f"  {volume_prefix}Jaccard: {jaccard_value.item():.6f}")
                self.test_jaccard.update(pred_binary, labels_binary)
                self.log("test_jaccard", self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if hasattr(self, "test_dice") and self.test_dice is not None:
                dice_value = torchmetrics.functional.dice(pred_binary, labels_binary)
                print(f"  {volume_prefix}Dice: {dice_value.item():.6f}")
                self.test_dice.update(pred_binary, labels_binary)
                self.log("test_dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if hasattr(self, "test_accuracy") and self.test_accuracy is not None:
                accuracy_value = torchmetrics.functional.accuracy(
                    pred_binary, labels_binary, task='binary'
                )
                print(f"  {volume_prefix}Accuracy: {accuracy_value.item():.6f}")
                self.test_accuracy.update(pred_binary, labels_binary)
                self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss using deep supervision handler
        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="train"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="train"
            )

        # Log losses (sync across GPUs for distributed training)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss using deep supervision handler
        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="val"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="val"
            )

        # Compute evaluation metrics if enabled
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            if getattr(self.cfg.inference.evaluation, 'enabled', False):
                metrics = getattr(self.cfg.inference.evaluation, 'metrics', None)
                if metrics is not None:
                    # Get the main output for metric computation
                    if is_deep_supervision:
                        main_output = outputs['output']
                    else:
                        main_output = outputs

                    # Check if this is multi-task learning
                    is_multi_task = hasattr(self.cfg.model, 'multi_task_config') and self.cfg.model.multi_task_config is not None

                    # Convert logits/probabilities to predictions
                    if is_multi_task:
                        # Multi-task learning: use first channel (usually binary segmentation)
                        # Extract first channel for both output and target
                        binary_output = main_output[:, 0:1, ...]  # (B, 1, H, W)
                        binary_target = labels[:, 0:1, ...]  # (B, 1, H, W)
                        preds = (binary_output.squeeze(1) > 0.5).long()  # (B, H, W)
                        targets = binary_target.squeeze(1).long()  # (B, H, W)
                    elif main_output.shape[1] > 1:
                        # Multi-class segmentation: use argmax
                        preds = torch.argmax(main_output, dim=1)  # (B, D, H, W)
                        targets = labels.squeeze(1).long()  # (B, D, H, W)
                    else:
                        # Single channel output (already predicted class or probability)
                        preds = (main_output.squeeze(1) > 0.5).long()  # (B, D, H, W)
                        targets = labels.squeeze(1).long()  # (B, D, H, W)

                    # Compute and log metrics
                    if 'jaccard' in metrics:
                        if not hasattr(self, 'val_jaccard'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_jaccard = torchmetrics.JaccardIndex(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)
                        self.val_jaccard(preds, targets)
                        self.log('val_jaccard', self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True)

                    if 'dice' in metrics:
                        if not hasattr(self, 'val_dice'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_dice = torchmetrics.Dice(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(self.device)
                        self.val_dice(preds, targets)
                        self.log('val_dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

                    if 'accuracy' in metrics:
                        if not hasattr(self, 'val_accuracy'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_accuracy = torchmetrics.Accuracy(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
                        self.val_accuracy(preds, targets)
                        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Log losses (sync across GPUs for distributed training)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def on_test_start(self):
        """Called at the beginning of testing to initialize metrics and inferer."""
        self._setup_test_metrics()

        # Explicitly set eval mode if configured (Lightning does this by default, but be explicit)
        if hasattr(self.cfg, 'inference') and getattr(self.cfg.inference, 'do_eval', True):
            self.eval()
        else:
            # Keep in training mode (e.g., for Monte Carlo Dropout uncertainty estimation)
            self.train()

    def on_test_end(self):
        """Called at the end of testing."""
        # Note: Metrics are logged in test_step with on_epoch=True,
        # which automatically computes and logs them at the end of testing.
        # Logging here causes a Lightning warning since self.log() is not allowed in on_test_end.
        pass

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Test step with optional sliding-window inference and metrics computation.

        Workflow:
        1. If final prediction exists → directly do evaluation
        2. If intermediate prediction exists → apply decoding → postprocessing → evaluation
        3. Else → run inference (using cfg.test for data loading/transform) → save → decode → evaluate
        """
        images = batch["image"]
        labels = batch.get("label")
        mask = batch.get("mask")

        mode, output_dir_value, cache_suffix, filenames = self._resolve_test_output_config(batch)
        predictions_np, loaded_from_file, loaded_suffix = self._load_cached_predictions(
            output_dir_value, filenames, cache_suffix, mode
        )

        # Determine what type of prediction was loaded
        loaded_final_predictions = loaded_from_file and loaded_suffix == "_prediction.h5"
        loaded_intermediate_predictions = loaded_from_file and loaded_suffix == "_tta_prediction.h5"

        # Extract volume name for logging
        volume_name = filenames[0] if filenames else f"volume_{batch_idx}"

        # CASE 1: Final predictions exist → directly evaluate
        if loaded_final_predictions:
            print(f"  ✅ Loaded final predictions from disk, skipping inference/decoding/postprocessing")
            if labels is not None:
                self._compute_test_metrics(predictions_np, labels, volume_name=volume_name)
            return torch.tensor(0.0, device=self.device)

        # CASE 2: Intermediate predictions exist → decode and postprocess
        if loaded_intermediate_predictions:
            print(f"  ✅ Loaded intermediate predictions from disk, skipping inference")
            # Convert back from saved format to [0,1] predictions if needed
            predictions_np = self._invert_save_prediction_transform(predictions_np)

            # Decode and postprocess
            decoded_predictions = apply_decode_mode(self.cfg, predictions_np)
            postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)

            # Save final predictions
            write_outputs(self.cfg, postprocessed_predictions, filenames, suffix="prediction", mode=mode)

            # Evaluate if labels provided
            if labels is not None:
                self._compute_test_metrics(decoded_predictions, labels, volume_name=volume_name)
            return torch.tensor(0.0, device=self.device)

        # CASE 3: No cached predictions → run full inference pipeline
        print(f"  🔄 No cached predictions found, running inference")

        # Run inference (cfg.test used for data loading and transforms via datamodule)
        predictions = self.inference_manager.predict_with_tta(images, mask=mask)
        predictions_np = predictions.detach().cpu().float().numpy()

        # Save intermediate predictions if configured
        save_intermediate = False
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "save_prediction"):
            save_intermediate = getattr(self.cfg.inference.save_prediction, "enabled", False)

        if save_intermediate:
            # Apply intensity scaling and dtype conversion before saving
            predictions_to_save = apply_save_prediction_transform(self.cfg, predictions_np)
            write_outputs(self.cfg, predictions_to_save, filenames, suffix="tta_prediction", mode=mode)
            print(f"  💾 Saved intermediate predictions")

        # Decode and postprocess
        decoded_predictions = apply_decode_mode(self.cfg, predictions_np)
        postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)

        # Save final predictions
        write_outputs(self.cfg, postprocessed_predictions, filenames, suffix="prediction", mode=mode)
        print(f"  💾 Saved final predictions")

        # Evaluate if labels provided
        if labels is not None:
            self._compute_test_metrics(decoded_predictions, labels, volume_name=volume_name)

        return torch.tensor(0.0, device=self.device)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = build_optimizer(self.cfg, self.model)

        # Build scheduler if configured (check both cfg.scheduler and cfg.optimization.scheduler)
        has_scheduler = (
            (hasattr(self.cfg, 'scheduler') and self.cfg.scheduler is not None) or
            (hasattr(self.cfg, 'optimization') and hasattr(self.cfg.optimization, 'scheduler') and self.cfg.optimization.scheduler is not None)
        )
        
        if has_scheduler:
            scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Check if this is ReduceLROnPlateau (requires metric monitoring)
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
            
            # ReduceLROnPlateau requires the 'monitor' key to pass the metric value
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Get monitor metric from scheduler config
                monitor_metric = None
                if hasattr(self.cfg, 'optimization') and hasattr(self.cfg.optimization, 'scheduler'):
                    monitor_metric = getattr(self.cfg.optimization.scheduler, 'monitor', None)
                elif hasattr(self.cfg, 'scheduler'):
                    monitor_metric = getattr(self.cfg.scheduler, 'monitor', None)
                
                if monitor_metric:
                    scheduler_config['monitor'] = monitor_metric
                    print(f"  ✅ ReduceLROnPlateau will monitor: {monitor_metric}")
                else:
                    # Default to validation loss
                    scheduler_config['monitor'] = 'val_loss_total'
                    print(f"  ⚠️  ReduceLROnPlateau will monitor: val_loss_total (default, no monitor specified in config)")

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config,
            }
        else:
            return optimizer

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log learning rate
        if self.optimizers():
            optimizer = self.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            lr = optimizer.param_groups[0]['lr']
            self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def create_lightning_module(
    cfg: Union[Config, DictConfig],
    model: Optional[nn.Module] = None,
) -> ConnectomicsModule:
    """
    Factory function to create ConnectomicsModule.

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model

    Returns:
        ConnectomicsModule instance
    """
    return ConnectomicsModule(cfg, model)
