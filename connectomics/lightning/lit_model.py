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
from ..models.loss import create_loss, create_combined_loss
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
        
        losses = nn.ModuleList()
        for loss_name in loss_names:
            # Use the new MONAI-native loss creation
            loss = create_loss(
                loss_name=loss_name,
                to_onehot_y=False,  # Adjust based on your task
                sigmoid=True if 'BCE' in loss_name else False,
                softmax=False,
            )
            losses.append(loss)
        
        return losses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        total_loss = 0.0
        loss_dict = {}
        
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
        """Validation step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        total_loss = 0.0
        loss_dict = {}
        
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
        """Test step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        total_loss = 0.0
        loss_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
            loss = loss_fn(outputs, labels)
            weighted_loss = loss * weight
            total_loss += weighted_loss
            
            loss_dict[f'test_loss_{i}'] = loss.item()
        
        loss_dict['test_loss_total'] = total_loss.item()
        
        # Log losses
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Build optimizer using the new solver API
        optimizer = build_optimizer(
            self.cfg,
            self.model,
        )
        
        # Build scheduler using the new solver API
        scheduler = build_lr_scheduler(
            self.cfg,
            optimizer,
        )
        
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