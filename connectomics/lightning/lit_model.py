"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np
from omegaconf import DictConfig

# Import existing components
from ..models import build_model
from ..models.loss import create_loss
from ..models.solver import build_optimizer, build_lr_scheduler
from ..config import Config


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

    def _build_model(self, cfg) -> nn.Module:
        """Build model from configuration."""
        return build_model(cfg)

    def _build_losses(self, cfg) -> nn.ModuleList:
        """Build loss functions from configuration."""
        loss_names = cfg.model.loss_functions if hasattr(cfg.model, 'loss_functions') else ['DiceLoss']
        loss_kwargs_list = cfg.model.loss_kwargs if hasattr(cfg.model, 'loss_kwargs') else [{}] * len(loss_names)

        losses = nn.ModuleList()
        for i, loss_name in enumerate(loss_names):
            # Get kwargs for this loss (default to empty dict if not specified)
            kwargs = loss_kwargs_list[i] if i < len(loss_kwargs_list) else {}
            loss = create_loss(loss_name=loss_name, **kwargs)
            losses.append(loss)

        return losses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss
        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
            # Weights decrease for smaller scales: [1.0, 0.5, 0.25, 0.125, 0.0625]
            main_output = outputs['output']
            ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
            all_outputs = [main_output] + ds_outputs

            for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
                # Match target to output size
                target = self._match_target_to_output(labels, output)

                # Compute loss for this scale
                scale_loss = 0.0
                for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                    loss = loss_fn(output, target)
                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'train_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['train_loss_total'] = total_loss.item()

        else:
            # Standard single-scale loss
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)
                weighted_loss = loss * weight
                total_loss += weighted_loss

                loss_dict[f'train_loss_{i}'] = loss.item()

            loss_dict['train_loss_total'] = total_loss.item()

        # Log losses
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss
        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
            main_output = outputs['output']
            ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
            all_outputs = [main_output] + ds_outputs

            for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
                # Match target to output size
                target = self._match_target_to_output(labels, output)

                # Compute loss for this scale
                scale_loss = 0.0
                for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                    loss = loss_fn(output, target)
                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'val_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['val_loss_total'] = total_loss.item()

        else:
            # Standard single-scale loss
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)
                weighted_loss = loss * weight
                total_loss += weighted_loss

                loss_dict[f'val_loss_{i}'] = loss.item()

            loss_dict['val_loss_total'] = total_loss.item()

        # Log losses
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss
        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
            main_output = outputs['output']
            ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
            all_outputs = [main_output] + ds_outputs

            for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
                # Match target to output size
                target = self._match_target_to_output(labels, output)

                # Compute loss for this scale
                scale_loss = 0.0
                for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                    loss = loss_fn(output, target)
                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'test_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['test_loss_total'] = total_loss.item()

        else:
            # Standard single-scale loss
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)
                weighted_loss = loss * weight
                total_loss += weighted_loss

                loss_dict[f'test_loss_{i}'] = loss.item()

            loss_dict['test_loss_total'] = total_loss.item()

        # Log losses
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def _match_target_to_output(
        self,
        target: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Match target size to output size for deep supervision.

        Uses interpolation to downsample labels to match output resolution.
        For segmentation masks, uses nearest-neighbor interpolation to preserve labels.
        For continuous targets, uses trilinear interpolation.

        Args:
            target: Target tensor of shape (B, C, D, H, W)
            output: Output tensor of shape (B, C, D', H', W')

        Returns:
            Resized target tensor matching output shape
        """
        if target.shape == output.shape:
            return target

        # Determine interpolation mode based on data type
        if target.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.uint8, torch.ByteTensor]:
            # Integer labels (including Byte/uint8): use nearest-neighbor
            mode = 'nearest'
            target_resized = nn.functional.interpolate(
                target.float(),
                size=output.shape[2:],
                mode=mode,
            ).long()
        else:
            # Continuous values: use trilinear
            mode = 'trilinear'
            target_resized = nn.functional.interpolate(
                target,
                size=output.shape[2:],
                mode=mode,
                align_corners=False,
            )

        return target_resized

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Use the unified builder that supports both YACS and Hydra configs
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_total',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

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
    Factory function to create a Lightning module from configuration.
    
    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model
        
    Returns:
        ConnectomicsModule instance
    """
    return ConnectomicsModule(cfg=cfg, model=model)


__all__ = [
    'ConnectomicsModule',
    'create_lightning_module',
]