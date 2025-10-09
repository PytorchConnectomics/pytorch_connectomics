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
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torchmetrics
from monai.inferers import SlidingWindowInferer
from monai.transforms import Flip

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
        self.sliding_inferer: Optional[SlidingWindowInferer] = None

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

    def _setup_sliding_window_inferer(self):
        """Initialize MONAI's SlidingWindowInferer based on config."""
        self.sliding_inferer = None

        if not hasattr(self.cfg, 'inference'):
            return

        roi_size = self._resolve_inferer_roi_size()
        if roi_size is None:
            warnings.warn(
                "Sliding-window inference disabled: unable to determine ROI size. "
                "Set inference.window_size or model.output_size in the config.",
                UserWarning,
            )
            return

        overlap = self._resolve_inferer_overlap(roi_size)
        sw_batch_size = max(1, int(getattr(self.cfg.inference, 'sw_batch_size', 1)))
        mode = getattr(self.cfg.inference, 'blending', 'gaussian')
        sigma_scale = float(getattr(self.cfg.inference, 'sigma_scale', 0.125))
        padding_mode = getattr(self.cfg.inference, 'padding_mode', 'constant')

        self.sliding_inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            progress=True,
        )

        print(
            "  Sliding-window inference configured: "
            f"roi_size={roi_size}, overlap={overlap}, sw_batch={sw_batch_size}, "
            f"mode={mode}, sigma_scale={sigma_scale}, padding={padding_mode}"
        )

    def _resolve_inferer_roi_size(self) -> Optional[Tuple[int, ...]]:
        """Determine the ROI size for sliding-window inference."""
        if hasattr(self.cfg, 'inference'):
            window_size = getattr(self.cfg.inference, 'window_size', None)
            if window_size:
                return tuple(int(v) for v in window_size)

        if hasattr(self.cfg, 'model') and hasattr(self.cfg.model, 'output_size'):
            output_size = getattr(self.cfg.model, 'output_size', None)
            if output_size:
                return tuple(int(v) for v in output_size)

        if hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'patch_size'):
            patch_size = getattr(self.cfg.data, 'patch_size', None)
            if patch_size:
                return tuple(int(v) for v in patch_size)

        return None

    def _resolve_inferer_overlap(self, roi_size: Tuple[int, ...]) -> Union[float, Tuple[float, ...]]:
        """Resolve overlap parameter using inference config."""
        if not hasattr(self.cfg, 'inference'):
            return 0.5

        overlap = getattr(self.cfg.inference, 'overlap', None)
        if overlap is not None:
            if isinstance(overlap, (list, tuple)):
                return tuple(float(max(0.0, min(o, 0.99))) for o in overlap)
            return float(max(0.0, min(overlap, 0.99)))

        stride = getattr(self.cfg.inference, 'stride', None)
        if stride:
            values: List[float] = []
            for size, step in zip(roi_size, stride):
                if size <= 0:
                    values.append(0.0)
                    continue
                ratio = 1.0 - float(step) / float(size)
                values.append(float(max(0.0, min(ratio, 0.99))))
            if len(set(values)) == 1:
                return values[0]
            return tuple(values)

        return 0.5

    def _extract_main_output(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Extract the primary segmentation logits from model outputs."""
        if isinstance(outputs, dict):
            if 'output' not in outputs:
                raise KeyError("Expected key 'output' in model outputs for deep supervision.")
            return outputs['output']
        return outputs

    def _sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Wrapper used by MONAI inferer to obtain primary model predictions."""
        outputs = self.forward(inputs)
        return self._extract_main_output(outputs)

    def _apply_tta_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply activation and channel selection before TTA ensemble.

        Args:
            tensor: Raw predictions (B, C, D, H, W)

        Returns:
            Preprocessed tensor for TTA ensembling
        """
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Get TTA-specific activation or fall back to output_act
        tta_act = getattr(self.cfg.inference, 'tta_act', None)
        if tta_act is None:
            tta_act = getattr(self.cfg.inference, 'output_act', None)

        # Apply activation function
        if tta_act == 'softmax':
            tensor = torch.softmax(tensor, dim=1)
        elif tta_act == 'sigmoid':
            tensor = torch.sigmoid(tensor)
        elif tta_act is not None and tta_act.lower() != 'none':
            warnings.warn(
                f"Unknown TTA activation function '{tta_act}'. Supported: 'softmax', 'sigmoid', None",
                UserWarning,
            )

        # Get TTA-specific channel selection or fall back to output_channel
        tta_channel = getattr(self.cfg.inference, 'tta_channel', None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, 'output_channel', None)

        # Apply channel selection
        if tta_channel is not None:
            if isinstance(tta_channel, int):
                if tta_channel == -1:
                    # -1 means all channels
                    pass
                else:
                    # Single channel selection
                    tensor = tensor[:, tta_channel:tta_channel+1, ...]
            elif isinstance(tta_channel, (list, tuple)):
                # Multiple channel selection
                tensor = tensor[:, list(tta_channel), ...]

        return tensor

    def _predict_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform test-time augmentation using flips and ensemble predictions.

        Args:
            images: Input volume (B, C, D, H, W) or (B, D, H, W) or (D, H, W)

        Returns:
            Averaged predictions from all TTA variants
        """
        # Normalize input to 5D tensor (B, C, D, H, W)
        if images.ndim == 3:
            # (D, H, W) -> (1, 1, D, H, W)
            images = images.unsqueeze(0).unsqueeze(0)
            warnings.warn(
                f"Input shape {images.shape} (D, H, W) automatically expanded to (1, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            images = images.unsqueeze(1)
            warnings.warn(
                f"Input shape (B, D, H, W) automatically expanded to (B, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim != 5:
            raise ValueError(
                f"TTA requires 3D, 4D, or 5D input tensor. Got {images.ndim}D tensor with shape {images.shape}. "
                f"Expected shapes: (D, H, W), (B, D, H, W), or (B, C, D, H, W)"
            )

        # Get TTA configuration
        tta_flip_axes_config = getattr(self.cfg.inference, 'tta_flip_axes', 'all')

        # Handle different tta_flip_axes configurations
        if tta_flip_axes_config is None:
            # null: No augmentation, skip TTA preprocessing and return raw prediction
            if self.sliding_inferer is not None:
                return self.sliding_inferer(inputs=images, network=self._sliding_window_predict)
            else:
                return self._sliding_window_predict(images)

        elif tta_flip_axes_config == 'all' or tta_flip_axes_config == []:
            # "all" or []: All 8 flips (all combinations of Z, Y, X)
            # IMPORTANT: MONAI Flip spatial_axis behavior for (B, C, D, H, W) tensors:
            #   spatial_axis=[0] flips C (channel) - WRONG for TTA!
            #   spatial_axis=[1] flips D (depth/Z) - CORRECT
            #   spatial_axis=[2] flips H (height/Y) - CORRECT
            #   spatial_axis=[3] flips W (width/X) - CORRECT
            # Must use [1, 2, 3] for [D, H, W] flips, NOT [0, 1, 2]!
            tta_flip_axes = [
                [],           # No flip
                [1],          # Flip Z (depth)
                [2],          # Flip Y (height)
                [3],          # Flip X (width)
                [1, 2],       # Flip Z+Y
                [1, 3],       # Flip Z+X
                [2, 3],       # Flip Y+X
                [1, 2, 3],    # Flip Z+Y+X
            ]
        elif isinstance(tta_flip_axes_config, (list, tuple)):
            # Custom list: Add no-flip baseline + user-specified flips
            tta_flip_axes = [[]] + list(tta_flip_axes_config)
        else:
            raise ValueError(
                f"Invalid tta_flip_axes: {tta_flip_axes_config}. "
                f"Expected 'all' (8 flips), null (no aug), or list of flip axes."
            )

        # Apply TTA with flips, preprocessing, and ensembling
        predictions = []

        for flip_axes in tta_flip_axes:
            # Apply flip augmentation
            if flip_axes:
                x_aug = Flip(spatial_axis=flip_axes)(images)
            else:
                x_aug = images

            # Inference with sliding window
            if self.sliding_inferer is not None:
                pred = self.sliding_inferer(
                    inputs=x_aug,
                    network=self._sliding_window_predict,
                )
            else:
                pred = self._sliding_window_predict(x_aug)

            # Invert flip for prediction
            if flip_axes:
                pred = Flip(spatial_axis=flip_axes)(pred)

            # Apply TTA preprocessing (activation + channel selection) if configured
            # Note: This is applied BEFORE ensembling for probability-space averaging
            pred_processed = self._apply_tta_preprocessing(pred)

            predictions.append(pred_processed)

        # Ensemble predictions based on configured mode
        ensemble_mode = getattr(self.cfg.inference, 'tta_ensemble_mode', 'mean')
        stacked_preds = torch.stack(predictions, dim=0)

        if ensemble_mode == 'mean':
            ensemble_result = stacked_preds.mean(dim=0)
        elif ensemble_mode == 'min':
            ensemble_result = stacked_preds.min(dim=0)[0]  # min returns (values, indices)
        elif ensemble_mode == 'max':
            ensemble_result = stacked_preds.max(dim=0)[0]  # max returns (values, indices)
        else:
            raise ValueError(f"Unknown TTA ensemble mode: {ensemble_mode}. Use 'mean', 'min', or 'max'.")

        return ensemble_result

    def _apply_postprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply postprocessing to predictions: activation and channel selection.

        Args:
            tensor: Raw predictions (B, C, D, H, W)

        Returns:
            Postprocessed tensor with activation and/or channel selection applied
        """
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Apply activation function
        output_act = getattr(self.cfg.inference, 'output_act', None)
        if output_act == 'softmax':
            tensor = torch.softmax(tensor, dim=1)
        elif output_act == 'sigmoid':
            tensor = torch.sigmoid(tensor)
        elif output_act is not None and output_act.lower() != 'none':
            warnings.warn(
                f"Unknown activation function '{output_act}'. Supported: 'softmax', 'sigmoid', None",
                UserWarning,
            )

        # Apply channel selection
        output_channel = getattr(self.cfg.inference, 'output_channel', None)
        if output_channel is not None:
            if isinstance(output_channel, int):
                if output_channel == -1:
                    # -1 means all channels
                    pass
                else:
                    # Single channel selection
                    tensor = tensor[:, output_channel:output_channel+1, ...]
            elif isinstance(output_channel, (list, tuple)):
                # Multiple channel selection
                tensor = tensor[:, list(output_channel), ...]

        return tensor

    def _apply_output_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply optional per-channel scaling before saving."""
        if not hasattr(self.cfg, 'inference'):
            return tensor

        scale = getattr(self.cfg.inference, 'output_scale', None)
        if not scale:
            return tensor

        scale_tensor = torch.tensor(scale, dtype=tensor.dtype, device=tensor.device)
        if scale_tensor.numel() == 1:
            return tensor * scale_tensor.view(1, 1, 1, 1, 1)

        if scale_tensor.numel() == tensor.shape[1]:
            return tensor * scale_tensor.view(1, -1, 1, 1, 1)

        warnings.warn(
            "inference.output_scale length does not match output channels; skipping scaling.",
            UserWarning,
        )
        return tensor

    def _apply_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply output dtype conversion."""
        if not hasattr(self.cfg, 'inference'):
            return tensor

        output_dtype = getattr(self.cfg.inference, 'output_dtype', None)
        if not output_dtype or output_dtype == 'float32':
            return tensor

        # Map string dtype to numpy dtype
        dtype_map = {
            'uint8': torch.uint8,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }

        if output_dtype not in dtype_map:
            warnings.warn(
                f"Unknown output_dtype '{output_dtype}'. Supported: {list(dtype_map.keys())}. "
                f"Keeping float32.",
                UserWarning,
            )
            return tensor

        target_dtype = dtype_map[output_dtype]

        # Clamp to valid range before conversion for integer types
        if output_dtype == 'uint8':
            tensor = torch.clamp(tensor, 0, 255)
        elif output_dtype == 'int8':
            tensor = torch.clamp(tensor, -128, 127)
        elif output_dtype == 'int16':
            tensor = torch.clamp(tensor, -32768, 32767)

        return tensor.to(target_dtype)

    def _write_outputs(self, logits: torch.Tensor, batch: Dict[str, Any]) -> None:
        """Persist predictions to disk using MONAI metadata when available."""
        if not hasattr(self.cfg, 'inference'):
            return

        output_dir_value = getattr(self.cfg.inference, 'output_path', None)
        if not output_dir_value:
            return

        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = batch.get('image_meta_dict')
        filenames: List[Optional[str]] = []

        if isinstance(meta, dict):
            meta_filenames = meta.get('filename_or_obj')
            if isinstance(meta_filenames, (list, tuple)):
                filenames = list(meta_filenames)
            elif meta_filenames is not None:
                filenames = [meta_filenames]
        elif isinstance(meta, list):
            for meta_item in meta:
                if isinstance(meta_item, dict):
                    filenames.append(meta_item.get('filename_or_obj'))

        batch_size = int(logits.shape[0])
        resolved_names: List[str] = []
        for idx in range(batch_size):
            if idx < len(filenames) and filenames[idx]:
                resolved_names.append(Path(str(filenames[idx])).stem)
            else:
                resolved_names.append(f"volume_{self.global_step}_{idx}")

        # Apply postprocessing (activation + channel selection)
        tensor = self._apply_postprocessing(logits.detach().cpu().float())
        # Apply scaling
        tensor = self._apply_output_scale(tensor)
        # Apply dtype conversion
        tensor = self._apply_output_dtype(tensor)
        data = tensor.numpy()

        from connectomics.data.io import write_hdf5

        for idx, name in enumerate(resolved_names):
            prediction = data[idx]
            destination = output_dir / f"{name}_prediction.h5"
            write_hdf5(str(destination), prediction, dataset="prediction")

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
        """Called at the beginning of testing to initialize metrics and inferer."""
        self._setup_test_metrics()
        self._setup_sliding_window_inferer()

        # Explicitly set eval mode if configured (Lightning does this by default, but be explicit)
        if hasattr(self.cfg, 'inference') and getattr(self.cfg.inference, 'do_eval', True):
            self.eval()
        else:
            # Keep in training mode (e.g., for Monte Carlo Dropout uncertainty estimation)
            self.train()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step with optional sliding-window inference and metrics computation."""
        images = batch['image']
        labels = batch.get('label')

        # Check if TTA is enabled and actually doing augmentation
        tta_enabled = hasattr(self.cfg, 'inference') and getattr(self.cfg.inference, 'test_time_augmentation', False)
        tta_flip_axes = getattr(self.cfg.inference, 'tta_flip_axes', 'all') if hasattr(self.cfg, 'inference') else 'all'
        use_tta = tta_enabled and tta_flip_axes is not None  # Only use TTA if flip_axes is not null
        use_sliding_inferer = self.sliding_inferer is not None

        if use_tta:
            # Use test-time augmentation
            main_output = self._predict_with_tta(images)
            outputs = main_output
            is_deep_supervision = False
        elif use_sliding_inferer:
            main_output = self.sliding_inferer(
                inputs=images,
                network=self._sliding_window_predict,
            )
            outputs = main_output
            is_deep_supervision = False
        else:
            outputs = self(images)
            is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())
            main_output = self._extract_main_output(outputs)

        # Persist predictions if requested
        self._write_outputs(main_output, batch)

        # Determine if we should skip loss computation
        skip_loss = False
        if labels is None:
            skip_loss = True
        elif use_tta:
            # Skip loss computation if TTA preprocessing changed the output shape
            # (e.g., channel selection makes it incompatible with loss functions)
            tta_channel = getattr(self.cfg.inference, 'tta_channel', None)
            if tta_channel is None:
                tta_channel = getattr(self.cfg.inference, 'output_channel', None)

            # If channel selection was applied, skip loss computation but keep metrics
            if tta_channel is not None and tta_channel != -1:
                skip_loss = True
                print("⚠️  Skipping loss computation (TTA channel selection enabled). Metrics will still be computed.")

        total_loss = 0.0
        loss_dict = {}

        # Compute loss only if not skipped
        if not skip_loss:
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

        # Compute metrics on main output (only if metrics are configured and labels available)
        # Metrics can be computed even if loss is skipped (e.g., with TTA channel selection)
        if labels is not None and (self.test_jaccard is not None or self.test_dice is not None or self.test_accuracy is not None):
            # Convert logits/probabilities to predictions
            # Check if main_output has multiple channels (need argmax) or single channel (already predicted)
            if main_output.shape[1] > 1:
                preds = torch.argmax(main_output, dim=1)  # (B, D, H, W)
            else:
                # Single channel output (already predicted class or probability)
                preds = (main_output.squeeze(1) > 0.5).long()  # (B, D, H, W)

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

        # Log losses (only if loss was computed)
        if not skip_loss and loss_dict:
            self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return loss if computed, otherwise None
        return total_loss if not skip_loss else None

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
