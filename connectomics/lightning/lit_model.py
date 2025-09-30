"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface to replace the custom training loop
with automatic distributed training, mixed precision, checkpointing, and more.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
from argparse import Namespace
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np

# Import existing components
from ..models import build_model
from ..models.loss import Criterion
from ..config import CfgNode
# from ..utils import setup_logger  # Not implemented


class ConnectomicsModule(pl.LightningModule):
    """
    PyTorch Lightning module for connectomics tasks.

    This module replaces the custom training loop in trainer.py with Lightning's
    automatic training features including:
    - Distributed training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    - Learning rate scheduling

    Args:
        cfg: Configuration object containing model, training, and data parameters
        model: Optional pre-built model (if None, builds from config)
        criterion: Optional pre-built criterion (if None, builds from config)
    """

    def __init__(self,
                 cfg: CfgNode,
                 model: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None):
        super().__init__()

        # Store config and setup logging
        self.cfg = cfg
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.logger_name = "connectomics"  # Lightning handles logging

        # Build model and criterion
        self.model = model if model is not None else build_model(cfg)
        self.criterion = criterion if criterion is not None else self._create_criterion(cfg)

        # Store task configuration
        self.task_type = getattr(cfg.MODEL, 'TASK_TYPE', 'segmentation')
        self.output_types = cfg.MODEL.TARGET_OPT
        self.loss_weights = cfg.MODEL.LOSS_WEIGHT
        self.weight_opts = cfg.MODEL.WEIGHT_OPT

        # Training configuration
        self.automatic_optimization = True
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Metrics storage
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Training step - replaces the custom training loop.

        Args:
            batch: Dictionary containing 'image' and target tensors
            batch_idx: Batch index

        Returns:
            Loss tensor and logging dict
        """
        return self._shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Validation step.

        Args:
            batch: Dictionary containing 'image' and target tensors
            batch_idx: Batch index

        Returns:
            Loss tensor and logging dict
        """
        return self._shared_step(batch, batch_idx, stage='val')

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str) -> STEP_OUTPUT:
        """
        Shared logic for training and validation steps.

        This replaces the complex logic in the original trainer's train() method.
        """
        # Extract input and targets
        images = batch['image']
        batch_size = images.size(0)

        # Forward pass
        predictions = self.forward(images)
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        # Compute losses for each output head
        total_loss = 0
        loss_dict = {}

        for i, (pred, target_opt, weight_opt, loss_weight) in enumerate(
            zip(predictions, self.output_types, self.weight_opts, self.loss_weights)):

            # Get target and weight from batch
            target_key = f'target_{target_opt}' if f'target_{target_opt}' in batch else f'target'
            weight_key = f'weight_{weight_opt}' if f'weight_{weight_opt}' in batch else None

            if target_key not in batch:
                warnings.warn(f"Target key {target_key} not found in batch, skipping loss computation")
                continue

            target = batch[target_key]
            weight = batch.get(weight_key) if weight_key else None

            # Compute loss
            if weight is not None:
                loss = self.criterion[i](pred, target, weight)
            else:
                loss = self.criterion[i](pred, target)

            # Weight the loss
            weighted_loss = loss * loss_weight[i] if isinstance(loss_weight, list) else loss * loss_weight
            total_loss += weighted_loss

            # Store individual losses for logging
            loss_dict[f'{stage}_loss_{i}'] = loss.detach()
            loss_dict[f'{stage}_weighted_loss_{i}'] = weighted_loss.detach()

        # Log metrics
        loss_dict[f'{stage}_loss'] = total_loss.detach()
        loss_dict[f'{stage}_batch_size'] = batch_size

        # Log to Lightning
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True,
                     batch_size=batch_size, sync_dist=True)

        return {
            'loss': total_loss,
            'log': loss_dict,
            'batch_size': batch_size
        }

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Compute epoch-level metrics
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            self.log('train_epoch_loss', avg_loss, prog_bar=True, sync_dist=True)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Compute epoch-level metrics
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_epoch_loss', avg_loss, prog_bar=True, sync_dist=True)
            self.validation_step_outputs.clear()

    def _create_criterion(self, cfg):
        """Create criterion from configuration."""
        return Criterion(
            device=self.device,
            target_opt=cfg.MODEL.TARGET_OPT,
            loss_opt=cfg.MODEL.LOSS_OPTION,
            output_act=cfg.MODEL.OUTPUT_ACT,
            loss_weight=cfg.MODEL.LOSS_WEIGHT,
            regu_opt=getattr(cfg.MODEL, 'REGU_OPT', None),
            regu_target=getattr(cfg.MODEL, 'REGU_TARGET', None),
            regu_weight=getattr(cfg.MODEL, 'REGU_WEIGHT', None),
            do_2d=cfg.DATASET.DO_2D,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure modern optimizers and learning rate schedulers."""
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Modern optimizer (AdamW is recommended for most tasks)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.SOLVER.BASE_LR,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Modern learning rate scheduler
        scheduler_name = getattr(self.cfg.SOLVER, 'LR_SCHEDULER_NAME', 'WarmupCosineLR')

        if scheduler_name == 'WarmupCosineLR':
            # Warmup + Cosine Annealing (modern best practice)
            warmup_epochs = max(1, self.cfg.SOLVER.ITERATION_TOTAL // 10000)  # 10% warmup
            total_epochs = self.cfg.SOLVER.ITERATION_TOTAL // 1000  # Approximate epochs

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=self.cfg.SOLVER.BASE_LR * 0.01
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        else:
            # Simple cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.SOLVER.ITERATION_TOTAL,
                eta_min=self.cfg.SOLVER.BASE_LR * 0.01
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }

    def configure_callbacks(self) -> List[pl.Callback]:
        """Configure Lightning callbacks."""
        callbacks = []

        # Model checkpointing
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.DATASET.OUTPUT_PATH,
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_epochs=self.cfg.SOLVER.get('CHECKPOINT_PERIOD', 1)
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        if self.cfg.SOLVER.get('EARLY_STOPPING', False):
            from pytorch_lightning.callbacks import EarlyStopping
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.SOLVER.get('EARLY_STOPPING_PATIENCE', 10),
                mode='min'
            )
            callbacks.append(early_stop_callback)

        # Learning rate monitoring
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        return callbacks

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference."""
        images = batch['image']
        predictions = self.forward(images)

        # Apply output activations if specified
        if hasattr(self.cfg.MODEL, 'OUTPUT_ACT') and self.cfg.MODEL.OUTPUT_ACT:
            if not isinstance(predictions, (list, tuple)):
                predictions = [predictions]

            activated_preds = []
            for pred, act_name in zip(predictions, self.cfg.MODEL.OUTPUT_ACT):
                if act_name == 'sigmoid':
                    pred = torch.sigmoid(pred)
                elif act_name == 'softmax':
                    pred = F.softmax(pred, dim=1)
                elif act_name == 'tanh':
                    pred = torch.tanh(pred)
                # 'none' or other: no activation
                activated_preds.append(pred)
            predictions = activated_preds

        return predictions[0] if len(predictions) == 1 else predictions

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint."""
        # Add any custom state that needs to be saved
        checkpoint['cfg'] = self.cfg
        checkpoint['task_type'] = self.task_type

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        # Restore any custom state
        if 'cfg' in checkpoint:
            self.cfg = checkpoint['cfg']
        if 'task_type' in checkpoint:
            self.task_type = checkpoint['task_type']


def create_lightning_module(cfg: CfgNode, **kwargs) -> ConnectomicsModule:
    """
    Factory function to create a Lightning module from config.

    Args:
        cfg: Configuration object
        **kwargs: Additional arguments passed to ConnectomicsModule

    Returns:
        Configured Lightning module
    """
    return ConnectomicsModule(cfg, **kwargs)