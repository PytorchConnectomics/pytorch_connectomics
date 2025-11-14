"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing

The implementation delegates to specialized modules:
- deep_supervision.py: Deep supervision and multi-task learning
- inference.py: Sliding window inference and test-time augmentation
- debugging.py: NaN detection and debugging utilities
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
from ..models import build_model
from ..models.loss import create_loss
from ..models.solver import build_optimizer, build_lr_scheduler
from ..config import Config

# Import new modular components
from .deep_supervision import DeepSupervisionHandler, match_target_to_output
from .inference import (
    InferenceManager,
    apply_postprocessing,
    apply_decode_mode,
    resolve_output_filenames,
    write_outputs,
)
from .debugging import DebugManager


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
        self.test_adapted_rand_results = []  # Store per-batch results for averaging

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

    def _setup_test_metrics(self):
        """Initialize test metrics based on inference config."""
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'evaluation'):
            return

        # Check if evaluation is enabled
        if not getattr(self.cfg.inference.evaluation, 'enabled', False):
            return

        metrics = getattr(self.cfg.inference.evaluation, 'metrics', None)
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
            # Adapted Rand error for instance segmentation
            # This is a custom metric that needs manual computation
            self.test_adapted_rand = True  # Flag to enable adapted_rand computation
            self.test_adapted_rand_results = []  # Store per-batch results

    def _compute_adapted_rand(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        filenames: List[str]
    ) -> None:
        """
        Compute Adapted Rand Error for instance segmentation evaluation.

        Args:
            predictions: Decoded instance segmentation predictions (B, C, D, H, W) or (B, D, H, W)
            labels: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            filenames: List of filenames for each sample in batch
        """
        if not self.test_adapted_rand:
            return

        from connectomics.metrics import adapted_rand

        batch_size = predictions.shape[0]

        # Compare decoded predictions with ground truth
        for idx in range(batch_size):
            pred = predictions[idx]
            # Remove channel dimension if present
            if pred.ndim == 4:
                pred = pred[0]  # Take first channel

            # Handle label indexing - labels might be a dict (multi-task) or ndarray
            try:
                if isinstance(labels, dict):
                    # Multi-task labels - use the first task (typically 'label')
                    label_key = list(labels.keys())[0]
                    gt = labels[label_key][idx]
                else:
                    gt = labels[idx]

                gt = gt.astype(np.uint32)
                if gt.ndim == 4:
                    gt = gt[0]  # Remove channel dimension

                are, prec, rec = adapted_rand(pred, gt, all_stats=True)
                self.test_adapted_rand_results.append({
                    'are': are,
                    'precision': prec,
                    'recall': rec,
                    'filename': filenames[idx]
                })
            except Exception as e:
                warnings.warn(
                    f"Error computing adapted_rand for {filenames[idx]}: {e}",
                    UserWarning
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        output = self.model(x)

        # Optionally clamp activations to prevent inf/nan
        if self.clamp_activations:
            if isinstance(output, dict):
                # Deep supervision - clamp all outputs
                output = {k: torch.clamp(v, self.clamp_min, self.clamp_max) for k, v in output.items()}
            else:
                output = torch.clamp(output, self.clamp_min, self.clamp_max)

        return output

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
        """Called at the end of testing to compute and log final metrics."""
        # Compute average adapted_rand if results were collected
        if self.test_adapted_rand and len(self.test_adapted_rand_results) > 0:
            # Compute average metrics
            avg_are = np.mean([r['are'] for r in self.test_adapted_rand_results])
            avg_prec = np.mean([r['precision'] for r in self.test_adapted_rand_results])
            avg_rec = np.mean([r['recall'] for r in self.test_adapted_rand_results])

            # Log to console
            print(f"\n{'='*60}")
            print(f"Adapted Rand Error (ARE) Results:")
            print(f"{'='*60}")
            print(f"Average ARE:       {avg_are:.6f}")
            print(f"Average Precision: {avg_prec:.6f}")
            print(f"Average Recall:    {avg_rec:.6f}")
            print(f"Number of samples: {len(self.test_adapted_rand_results)}")
            print(f"{'='*60}\n")

            # Log individual results
            print("Per-sample results:")
            for r in self.test_adapted_rand_results:
                print(f"  {r['filename']}: ARE={r['are']:.6f}, Prec={r['precision']:.6f}, Rec={r['recall']:.6f}")
            print()

            # Log to tensorboard/wandb directly (can't use self.log in on_test_end)
            if self.logger is not None:
                try:
                    # For TensorBoardLogger
                    if hasattr(self.logger, 'experiment'):
                        self.logger.experiment.add_scalar('test_adapted_rand_mean', avg_are, self.global_step)
                        self.logger.experiment.add_scalar('test_adapted_rand_precision', avg_prec, self.global_step)
                        self.logger.experiment.add_scalar('test_adapted_rand_recall', avg_rec, self.global_step)
                except Exception as e:
                    warnings.warn(f"Could not log metrics to logger: {e}", UserWarning)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step with optional sliding-window inference and metrics computation."""
        images = batch['image']
        labels = batch.get('label')
        mask = batch.get('mask')  # Get test mask if available

        # Get batch size from images
        actual_batch_size = images.shape[0]
        print(f"  DEBUG: test_step - images shape: {images.shape}, batch_size: {actual_batch_size}")

        # Always use TTA (handles no-transform case) + sliding window
        # TTA preprocessing (activation, masking) is applied regardless of flip augmentation
        # Note: TTA always returns a simple tensor, not a dict (deep supervision not supported in test mode)
        predictions = self.inference_manager.predict_with_tta(images, mask=mask)

        # Convert predictions to numpy for saving/decoding
        predictions_np = predictions.detach().cpu().float().numpy()
        print(f"  DEBUG: test_step - predictions_np shape: {predictions_np.shape}")

        # Resolve filenames once for all saving operations
        filenames = resolve_output_filenames(self.cfg, batch, self.global_step)
        print(f"  DEBUG: test_step - filenames count: {len(filenames)}, filenames: {filenames[:5]}...")

        # Ensure filenames list matches actual batch size
        # If we don't have enough filenames, generate default ones
        while len(filenames) < actual_batch_size:
            filenames.append(f"volume_{self.global_step}_{len(filenames)}")
        print(f"  DEBUG: test_step - after padding, filenames count: {len(filenames)}")

        # Check if we should save intermediate predictions (before decoding)
        save_intermediate = False
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'test_time_augmentation'):
            save_intermediate = getattr(self.cfg.inference.test_time_augmentation, 'save_predictions', False)

        # Save intermediate TTA predictions (before decoding) if requested
        if save_intermediate:
            write_outputs(self.cfg, predictions_np, filenames, suffix="tta_prediction")

        # Apply decode mode (instance segmentation decoding)
        print(f"  DEBUG: test_step - before decode, predictions_np shape: {predictions_np.shape}")
        decoded_predictions = apply_decode_mode(self.cfg, predictions_np)
        print(f"  DEBUG: test_step - after decode, decoded_predictions shape: {decoded_predictions.shape}")

        # Apply postprocessing (scaling and dtype conversion) if configured
        postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)
        print(f"  DEBUG: test_step - after postprocess, postprocessed_predictions shape: {postprocessed_predictions.shape}")

        # Save final decoded and postprocessed predictions
        write_outputs(self.cfg, postprocessed_predictions, filenames, suffix="prediction")

        # Compute adapted_rand if enabled and labels available
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            self._compute_adapted_rand(decoded_predictions, labels_np, filenames)

        # Determine if we should skip loss computation
        skip_loss = False
        if labels is None:
            skip_loss = True
        elif hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            # Skip loss when any evaluation metric is enabled (labels are not transformed)
            evaluation_enabled = getattr(self.cfg.inference.evaluation, 'enabled', False)
            if evaluation_enabled:
                skip_loss = True

        # Skip loss computation if needed
        if skip_loss:
            return torch.tensor(0.0, device=self.device)

        # Compute test loss if labels are available and evaluation is not enabled
        outputs = self(images)
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="test"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="test"
            )

        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = build_optimizer(self.cfg, self.model.parameters())

        # Build scheduler if configured
        if hasattr(self.cfg, 'scheduler') and self.cfg.scheduler is not None:
            scheduler = build_lr_scheduler(self.cfg, optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                },
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
