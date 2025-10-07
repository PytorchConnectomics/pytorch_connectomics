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
import pdb

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np
from omegaconf import DictConfig
import torchmetrics

# Import existing components
from ..models import build_model
from ..models.loss import create_loss
from ..models.solver import build_optimizer, build_lr_scheduler
from ..config import Config
from ..utils.debug_hooks import NaNDetectionHookManager


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

        # Hook manager for intermediate layer debugging
        self._hook_manager: Optional[NaNDetectionHookManager] = None

        # Test metrics (initialized lazily during test mode if specified in config)
        self.test_jaccard = None
        self.test_dice = None
        self.test_accuracy = None

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

    def _setup_test_metrics(self):
        """Initialize test metrics based on inference config."""
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'metrics'):
            return

        metrics = self.cfg.inference.metrics
        if metrics is None:
            return

        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2

        # Create only the specified metrics
        if 'jaccard' in metrics:
            self.test_jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)
        if 'dice' in metrics:
            self.test_dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(self.device)
        if 'accuracy' in metrics:
            self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)

    def _setup_inference_blending(self):
        """Initialize blending matrices and result buffers for inference."""
        if not hasattr(self.cfg, 'inference'):
            return

        # Import blending utilities
        from connectomics.data.process.blend import build_blending_matrix

        # Get output size from config
        output_size = tuple(self.cfg.model.output_size) if hasattr(self.cfg.model, 'output_size') else tuple(self.cfg.data.patch_size)

        # Build blending matrix
        blending_mode = getattr(self.cfg.inference, 'blending', 'gaussian')
        self.blending_matrix = build_blending_matrix(output_size, mode=blending_mode)
        print(f"  Inference blending: {blending_mode} (patch size: {output_size})")

        # Initialize result buffers (created dynamically per volume)
        self.inference_results = {}  # vol_id -> accumulated predictions
        self.inference_weights = {}  # vol_id -> accumulated weights
        self.volume_shapes = {}      # vol_id -> volume shape

    def _accumulate_prediction(self, output: torch.Tensor, pos: torch.Tensor, vol_id: int):
        """
        Accumulate a single prediction with blending weights.

        Args:
            output: Model output of shape (C, D, H, W)
            pos: Position tensor [z, y, x]
            vol_id: Volume ID
        """
        # Convert to numpy
        pred = output.detach().cpu().numpy()
        z, y, x = int(pos[0]), int(pos[1]), int(pos[2])

        # Get shape
        num_channels = pred.shape[0]
        d, h, w = pred.shape[1:]

        # Initialize buffers for this volume if needed
        if vol_id not in self.inference_results:
            # Estimate volume shape from first patch position and size
            # This is a simplified approach - ideally get from dataset
            vol_shape = (max(z + d, 256), max(y + h, 256), max(x + w, 256))
            self.volume_shapes[vol_id] = vol_shape

            self.inference_results[vol_id] = np.zeros(
                [num_channels] + list(vol_shape),
                dtype=np.float32
            )
            self.inference_weights[vol_id] = np.zeros(
                vol_shape,
                dtype=np.float32
            )
            print(f"  Initialized inference buffers for volume {vol_id}: {vol_shape}")

        # Accumulate with blending
        self.inference_results[vol_id][
            :, z:z+d, y:y+h, x:x+w
        ] += pred * self.blending_matrix[np.newaxis, :]

        self.inference_weights[vol_id][
            z:z+d, y:y+h, x:x+w
        ] += self.blending_matrix

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

                    # Check for NaN/Inf immediately after computing loss
                    if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"\n{'='*80}")
                        print(f"⚠️  NaN/Inf detected in loss computation!")
                        print(f"{'='*80}")
                        print(f"Loss function: {loss_fn.__class__.__name__}")
                        print(f"Loss value: {loss.item()}")
                        print(f"Scale: {scale_idx}, Weight: {weight}")
                        print(f"Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")
                        print(f"Target shape: {target.shape}, range: [{target.min():.4f}, {target.max():.4f}]")
                        print(f"Output contains NaN: {torch.isnan(output).any()}")
                        print(f"Target contains NaN: {torch.isnan(target).any()}")
                        if self.debug_on_nan:
                            print(f"\nEntering debugger...")
                            pdb.set_trace()
                        raise ValueError(f"NaN/Inf in loss at scale {scale_idx}")

                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'train_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['train_loss_total'] = total_loss.item()

        else:
            # Standard single-scale loss
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)

                # Check for NaN/Inf immediately after computing loss
                if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"\n{'='*80}")
                    print(f"⚠️  NaN/Inf detected in loss computation!")
                    print(f"{'='*80}")
                    print(f"Loss function: {loss_fn.__class__.__name__}")
                    print(f"Loss value: {loss.item()}")
                    print(f"Loss index: {i}, Weight: {weight}")
                    print(f"Output shape: {outputs.shape}, range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                    print(f"Label shape: {labels.shape}, range: [{labels.min():.4f}, {labels.max():.4f}]")
                    print(f"Output contains NaN: {torch.isnan(outputs).any()}")
                    print(f"Label contains NaN: {torch.isnan(labels).any()}")
                    if self.debug_on_nan:
                        print(f"\nEntering debugger...")
                        pdb.set_trace()
                    raise ValueError(f"NaN/Inf in loss at index {i}")

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

    def on_test_start(self):
        """Called at the beginning of testing to initialize metrics and blending."""
        self._setup_test_metrics()
        self._setup_inference_blending()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step with deep supervision support and metrics computation."""
        images = batch['image']
        labels = batch.get('label', None)

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Get the main output for metrics
        if is_deep_supervision:
            main_output = outputs['output']
        else:
            main_output = outputs

        # Accumulate predictions for blending if position metadata available
        if 'pos' in batch and hasattr(self, 'blending_matrix'):
            positions = batch['pos']  # (B, 4) - [vol_id, z, y, x]
            batch_size = main_output.shape[0]
            for i in range(batch_size):
                vol_id = int(positions[i, 0])
                pos = positions[i, 1:]  # [z, y, x]
                self._accumulate_prediction(main_output[i], pos, vol_id)

        # Compute loss only if labels are available
        if labels is None:
            # No labels - inference only mode
            return None

        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
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

        # Compute metrics on main output (only if metrics are configured)
        if self.test_jaccard is not None or self.test_dice is not None or self.test_accuracy is not None:
            # Convert logits to predictions
            preds = torch.argmax(main_output, dim=1)  # (B, D, H, W)
            targets = labels.squeeze(1).long()  # (B, D, H, W)

            # Update and log metrics (only if initialized)
            if self.test_jaccard is not None:
                self.test_jaccard(preds, targets)
                self.log('test_jaccard', self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True)
            if self.test_dice is not None:
                self.test_dice(preds, targets)
                self.log('test_dice', self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
            if self.test_accuracy is not None:
                self.test_accuracy(preds, targets)
                self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Log losses
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def on_test_epoch_end(self):
        """Normalize and save blended inference results."""
        if not hasattr(self, 'inference_results') or len(self.inference_results) == 0:
            return

        print("\n" + "=" * 60)
        print("Finalizing inference results with blending")
        print("=" * 60)

        # Import saving utilities
        from connectomics.data.io import write_hdf5
        from pathlib import Path

        # Normalize and save each volume
        for vol_id in sorted(self.inference_results.keys()):
            print(f"\nProcessing volume {vol_id}...")

            # Normalize by accumulated weights
            weights = self.inference_weights[vol_id][np.newaxis, :]  # (1, D, H, W)
            result = self.inference_results[vol_id] / (weights + 1e-8)

            # Convert to uint8 (0-255 range)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")
            print(f"  Result range: [{result.min()}, {result.max()}]")

            # Save result
            if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'output_path'):
                output_dir = Path(self.cfg.inference.output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                output_filename = f"vol{vol_id}_prediction.h5"
                output_path = output_dir / output_filename

                # Save as HDF5
                write_hdf5(str(output_path), result, dataset=f'vol{vol_id}')
                print(f"  Saved to: {output_path}")

        print("\n" + "=" * 60)
        print(f"Inference complete! Saved {len(self.inference_results)} volume(s)")
        print("=" * 60 + "\n")

        # Cleanup to free memory
        del self.inference_results
        del self.inference_weights
        del self.blending_matrix

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

    def enable_nan_hooks(
        self,
        debug_on_nan: bool = True,
        verbose: bool = False,
        layer_types: Optional[Tuple] = None,
    ) -> NaNDetectionHookManager:
        """
        Enable forward hooks to detect NaN in intermediate layer outputs.

        This attaches hooks to all layers in the model that will check for NaN/Inf
        in layer outputs during forward pass. When NaN is detected, it will print
        diagnostics and optionally enter the debugger.

        Useful for debugging in pdb:
            (Pdb) pl_module.enable_nan_hooks()
            (Pdb) outputs = pl_module(batch['image'])
            # Will stop at first layer producing NaN

        Args:
            debug_on_nan: If True, enter pdb when NaN detected (default: True)
            verbose: If True, print stats for every layer (slow, default: False)
            layer_types: Tuple of layer types to hook (default: all common layers)

        Returns:
            NaNDetectionHookManager instance
        """
        if self._hook_manager is not None:
            print("⚠️  Hooks already enabled. Call disable_nan_hooks() first.")
            return self._hook_manager

        self._hook_manager = NaNDetectionHookManager(
            model=self.model,
            debug_on_nan=debug_on_nan,
            verbose=verbose,
            collect_stats=True,
            layer_types=layer_types,
        )

        return self._hook_manager

    def disable_nan_hooks(self):
        """
        Disable forward hooks for NaN detection.

        Removes all hooks that were attached by enable_nan_hooks().
        """
        if self._hook_manager is not None:
            self._hook_manager.remove_hooks()
            self._hook_manager = None
        else:
            print("⚠️  No hooks to remove.")

    def get_hook_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get statistics from NaN detection hooks.

        Returns:
            Dictionary mapping layer names to their statistics, or None if hooks not enabled
        """
        if self._hook_manager is not None:
            return self._hook_manager.get_stats()
        else:
            print("⚠️  Hooks not enabled. Call enable_nan_hooks() first.")
            return None

    def print_hook_summary(self):
        """
        Print summary of NaN detection hook statistics.

        Shows which layers detected NaN/Inf and how many times.
        """
        if self._hook_manager is not None:
            self._hook_manager.print_summary()
        else:
            print("⚠️  Hooks not enabled. Call enable_nan_hooks() first.")

    def check_for_nan(self, check_grads: bool = True, verbose: bool = True) -> dict:
        """
        Debug utility to check for NaN/Inf in model parameters and gradients.

        Useful when debugging in pdb. Call as: pl_module.check_for_nan()

        Args:
            check_grads: Also check gradients
            verbose: Print detailed information

        Returns:
            Dictionary with NaN/Inf information
        """
        nan_params = []
        inf_params = []
        nan_grads = []
        inf_grads = []

        for name, param in self.named_parameters():
            # Check parameters
            if torch.isnan(param).any():
                nan_params.append((name, param.shape))
                if verbose:
                    print(f"⚠️  NaN in parameter: {name}, shape={param.shape}")
            if torch.isinf(param).any():
                inf_params.append((name, param.shape))
                if verbose:
                    print(f"⚠️  Inf in parameter: {name}, shape={param.shape}")

            # Check gradients
            if check_grads and param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append((name, param.grad.shape))
                    if verbose:
                        print(f"⚠️  NaN in gradient: {name}, shape={param.grad.shape}")
                if torch.isinf(param.grad).any():
                    inf_grads.append((name, param.grad.shape))
                    if verbose:
                        print(f"⚠️  Inf in gradient: {name}, shape={param.grad.shape}")

        result = {
            'nan_params': nan_params,
            'inf_params': inf_params,
            'nan_grads': nan_grads,
            'inf_grads': inf_grads,
            'has_nan': len(nan_params) > 0 or len(nan_grads) > 0,
            'has_inf': len(inf_params) > 0 or len(inf_grads) > 0,
        }

        if verbose:
            if not result['has_nan'] and not result['has_inf']:
                print("✅ No NaN/Inf found in parameters or gradients")
            else:
                print(f"\n📊 Summary:")
                print(f"   NaN parameters: {len(nan_params)}")
                print(f"   Inf parameters: {len(inf_params)}")
                print(f"   NaN gradients: {len(nan_grads)}")
                print(f"   Inf gradients: {len(inf_grads)}")

        return result

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