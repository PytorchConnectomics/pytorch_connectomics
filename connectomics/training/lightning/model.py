"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing

The implementation delegates to specialized modules:
- connectomics.training.loss: Loss orchestration and weighting (PyTorch-only)
- connectomics.inference: Sliding window inference and test-time augmentation
- connectomics.training.debugging: NaN detection and debugging utilities
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

# Import existing components
from ...config import Config
from ...inference import (
    InferenceManager,
    resolve_output_filenames,
)
from ...models import build_model
from ...models.loss import create_loss, get_loss_metadata_for_module
from ..debugging import DebugManager

# Import training/inference components
from ..loss import LossOrchestrator, build_loss_weighter, infer_num_loss_tasks_from_config
from ..model_weights import load_external_weights
from ..optim import build_lr_scheduler, build_optimizer
from .test_pipeline import compute_test_metrics, run_test_step


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
        self.save_hyperparameters(ignore=["model"])

        # Build model
        self.model = model if model is not None else self._build_model(cfg)

        # Build loss functions
        self.loss_functions = self._build_losses(cfg)
        self.loss_weights = self._extract_loss_weights(cfg)
        self.loss_metadata = [
            get_loss_metadata_for_module(loss_fn) for loss_fn in self.loss_functions
        ]

        # Build adaptive loss weighter (for multi-task learning)
        num_tasks = infer_num_loss_tasks_from_config(cfg)
        self.loss_weighter = build_loss_weighter(cfg, num_tasks, model=self.model)

        # Enable inline NaN detection
        nan_cfg = getattr(getattr(cfg, "monitor", None), "nan_detection", None)
        self.enable_nan_detection = getattr(nan_cfg, "enabled", True)
        self.debug_on_nan = getattr(nan_cfg, "debug_on_nan", True)

        # Initialize specialized handlers
        self.loss_orchestrator = LossOrchestrator(
            cfg=cfg,
            loss_functions=self.loss_functions,
            loss_weights=self.loss_weights,
            enable_nan_detection=self.enable_nan_detection,
            debug_on_nan=self.debug_on_nan,
            loss_weighter=self.loss_weighter,
            loss_metadata=self.loss_metadata,
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
        self.val_jaccard = None
        self.val_dice = None
        self.val_accuracy = None
        self._val_metrics_initialized = False

        # Prediction saving state
        self._prediction_save_counter = 0  # Track number of samples saved

    def _build_model(self, cfg) -> nn.Module:
        """Build model from configuration."""
        model = build_model(cfg)
        external_weights_path = getattr(cfg.model, "external_weights_path", None)
        if external_weights_path:
            print(f"\n  Loading external weights from: {external_weights_path}")
            model = load_external_weights(model, cfg)
            print("")
        return model

    @staticmethod
    def _get_losses_list(cfg) -> list:
        """Return the unified losses list from config, with defaults."""
        loss_cfg = getattr(cfg.model, "loss", None)
        losses = getattr(loss_cfg, "losses", None)
        if losses is not None:
            return list(losses)
        # Default: DiceLoss + BCEWithLogitsLoss applied to all channels
        return [
            {"function": "DiceLoss", "weight": 1.0},
            {"function": "BCEWithLogitsLoss", "weight": 1.0},
        ]

    def _build_losses(self, cfg) -> nn.ModuleList:
        """Build loss functions from unified losses configuration."""
        losses_list = self._get_losses_list(cfg)
        result = nn.ModuleList()
        for entry in losses_list:
            fn_name = entry["function"]
            kwargs = dict(entry.get("kwargs", {}))
            if fn_name == "WeightedBCEWithLogitsLoss":
                raw_pos_weight = entry.get("pos_weight", None)
                if raw_pos_weight is not None and not isinstance(raw_pos_weight, str):
                    kwargs["pos_weight"] = float(raw_pos_weight)
            result.append(create_loss(fn_name, **kwargs))
        return result

    def _extract_loss_weights(self, cfg) -> list:
        """Extract per-loss weights from unified losses configuration."""
        losses_list = self._get_losses_list(cfg)
        return [float(entry.get("weight", 1.0)) for entry in losses_list]

    def _get_runtime_inference_config(self):
        """Return merged runtime inference config (resolved before module construction)."""
        inference_cfg = getattr(self.cfg, "inference", None)
        if inference_cfg is None:
            raise ValueError("Missing runtime cfg.inference configuration")
        return inference_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightning forward pass that delegates to the underlying model.

        This is required so Lightning can execute the module during training/inference.
        """
        return self.model(x)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load checkpoint state with compatibility filtering for stale loss-function buffers."""
        if strict and isinstance(state_dict, dict):
            current_keys = set(self.state_dict().keys())
            dropped_keys = [
                key
                for key in state_dict.keys()
                if key not in current_keys and key.startswith("loss_functions.")
            ]
            if dropped_keys:
                state_dict = {
                    key: value for key, value in state_dict.items() if key not in dropped_keys
                }
                preview = ", ".join(dropped_keys[:3])
                if len(dropped_keys) > 3:
                    preview += f", ... (+{len(dropped_keys) - 3} more)"
                print(
                    "  ℹ️  Ignoring stale loss-function checkpoint key(s): "
                    f"{preview}"
                )

        return super().load_state_dict(state_dict, strict=strict)

    def _get_test_evaluation_config(self):
        """Return merged runtime evaluation config from cfg.inference."""
        inference_cfg = self._get_runtime_inference_config()
        return getattr(inference_cfg, "evaluation", None)

    def _is_test_evaluation_enabled(self) -> bool:
        """Return whether test-time metric computation is enabled."""
        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            return False

        return bool(self._cfg_value(evaluation_config, "enabled", False))

    @staticmethod
    def _cfg_value(cfg_obj: Any, key: str, default: Any = None) -> Any:
        """Unified dict/attribute config accessor (delegates to shared utility)."""
        from ...config.dict_utils import cfg_get
        return cfg_get(cfg_obj, key, default)

    @classmethod
    def _cfg_float(cls, cfg_obj: Any, key: str, default: float) -> float:
        """Unified float accessor for dict/attribute config objects."""
        value = cls._cfg_value(cfg_obj, key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            warnings.warn(
                f"Config key '{key}' value {value!r} cannot be converted to float, "
                f"using default {default}"
            )
            return float(default)

    def _has_multiple_supervised_loss_tasks(self) -> bool:
        """Infer multi-task supervised setup from compiled explicit loss terms."""
        pred_target_terms = [
            term for term in self.loss_orchestrator.loss_term_specs
            if term.call_kind == "pred_target"
        ]
        return len(pred_target_terms) > 1

    def _create_metrics(self, prefix: str, metrics: list, num_classes: int, use_binary: bool,
                        instance_iou_threshold: float = 0.5):
        """Create and attach torchmetrics with the given prefix (test_ or val_)."""
        if "jaccard" in metrics:
            setattr(self, f"{prefix}jaccard", (
                torchmetrics.JaccardIndex(task="binary").to(self.device)
                if use_binary
                else torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(self.device)
            ))
        if "dice" in metrics:
            setattr(self, f"{prefix}dice", (
                torchmetrics.Dice(task="binary").to(self.device)
                if use_binary
                else torchmetrics.Dice(num_classes=num_classes, average="macro").to(self.device)
            ))
        if "accuracy" in metrics:
            setattr(self, f"{prefix}accuracy", (
                torchmetrics.Accuracy(task="binary").to(self.device)
                if use_binary
                else torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            ))
        # Instance-level metrics only for test
        if prefix == "test_":
            if "adapted_rand" in metrics:
                from ...metrics.metrics_seg import AdaptedRandError
                setattr(self, f"{prefix}adapted_rand", AdaptedRandError().to(self.device))
            if "voi" in metrics:
                from ...metrics.metrics_seg import VariationOfInformation
                setattr(self, f"{prefix}voi", VariationOfInformation().to(self.device))
            if "instance_accuracy" in metrics:
                from ...metrics.metrics_seg import InstanceAccuracy
                setattr(self, f"{prefix}instance_accuracy", InstanceAccuracy(
                    thresh=instance_iou_threshold, criterion="iou",
                ).to(self.device))
            if "instance_accuracy_detail" in metrics:
                from ...metrics.metrics_seg import InstanceAccuracySimple
                setattr(self, f"{prefix}instance_accuracy_detail", InstanceAccuracySimple(
                    thresh=instance_iou_threshold, criterion="iou",
                ).to(self.device))

    def _setup_test_metrics(self):
        """Initialize test metrics based on test or inference config."""
        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            return
        if not self._is_test_evaluation_enabled():
            return
        metrics = self._cfg_value(evaluation_config, "metrics", None)
        if metrics is None:
            return

        inference_eval_defaults = self._get_runtime_inference_config().evaluation
        instance_iou_threshold = float(
            self._cfg_value(
                evaluation_config,
                "instance_iou_threshold",
                inference_eval_defaults.instance_iou_threshold,
            )
        )
        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, "out_channels") else 2
        self._create_metrics("test_", metrics, num_classes, num_classes == 1, instance_iou_threshold)

    def _setup_validation_metrics(self):
        """Initialize validation metrics once (avoid lazy init in validation loop)."""
        if self._val_metrics_initialized:
            return

        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            self._val_metrics_initialized = True
            return
        if not self._is_test_evaluation_enabled():
            self._val_metrics_initialized = True
            return
        metrics = self._cfg_value(evaluation_config, "metrics", None)
        if metrics is None:
            self._val_metrics_initialized = True
            return

        is_multi_task = self._has_multiple_supervised_loss_tasks()
        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, "out_channels") else 2
        use_binary = is_multi_task or num_classes == 1
        self._create_metrics("val_", metrics, num_classes, use_binary)
        self._val_metrics_initialized = True

    def on_validation_start(self) -> None:
        """Called before validation starts."""
        self._setup_validation_metrics()

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
        inference_cfg = self._get_runtime_inference_config()
        save_pred_cfg = inference_cfg.save_prediction

        # Get the scale and dtype that were used for saving
        intensity_scale = getattr(save_pred_cfg, "intensity_scale", None)

        # Convert to float first
        data = data.astype(np.float32)

        # Invert the scaling if it was applied
        # Note: intensity_scale < 0 means scaling was disabled, so no inversion needed
        if intensity_scale is not None and intensity_scale > 0 and intensity_scale != 1.0:
            data = data / float(intensity_scale)
            print(f"  🔄 Inverted intensity scaling by {intensity_scale}")
        elif intensity_scale is not None and intensity_scale < 0:
            print(
                f"  ℹ️  Intensity scaling was disabled (scale={intensity_scale}), no inversion needed"
            )

        return data

    def _resolve_test_output_config(
        self, batch: Dict[str, Any]
    ) -> tuple[str, Optional[str], str, List[str]]:
        """Determine output dir/cache suffix from merged runtime inference config."""
        mode = "test"
        save_pred_cfg = self._get_runtime_inference_config().save_prediction
        output_dir_value = getattr(save_pred_cfg, "output_path", None)
        cache_suffix = getattr(save_pred_cfg, "cache_suffix", "_prediction.h5")

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

    def _save_metrics_to_file(self, metrics_dict: Dict[str, Any]):
        """
        Save evaluation metrics to a text file in the output directory.

        Args:
            metrics_dict: Dictionary containing metric names and values
        """
        metric_keys = [k for k in metrics_dict.keys() if k != "volume_name"]
        if not metric_keys:
            return

        mode = "test"
        output_path = getattr(self._get_runtime_inference_config().save_prediction, "output_path", None)

        if output_path is None:
            print(f"  ⚠️  Cannot save metrics: output_path not found for mode={mode}")
            return

        from datetime import datetime
        from pathlib import Path

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with volume name and timestamp
        volume_name = metrics_dict.get("volume_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = output_dir / f"evaluation_metrics_{volume_name}.txt"

        # Write metrics to file
        try:
            with open(metrics_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("EVALUATION METRICS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Volume: {volume_name}\n")
                f.write("=" * 80 + "\n\n")

                # Write instance segmentation metrics
                if "adapted_rand_error" in metrics_dict:
                    f.write("Instance Segmentation Metrics:\n")
                    f.write("-" * 80 + "\n")
                    f.write(
                        f"  Adapted Rand Error:           {metrics_dict['adapted_rand_error']:.6f}\n"
                    )

                    if "voi_split" in metrics_dict:
                        f.write(
                            f"  VOI Split:                    {metrics_dict['voi_split']:.6f}\n"
                        )
                        f.write(
                            f"  VOI Merge:                    {metrics_dict['voi_merge']:.6f}\n"
                        )
                        f.write(
                            f"  VOI Total:                    {metrics_dict['voi_total']:.6f}\n"
                        )

                    if "instance_accuracy" in metrics_dict:
                        f.write(
                            f"  Instance Accuracy:            {metrics_dict['instance_accuracy']:.6f}\n"
                        )

                    if "instance_accuracy_detail" in metrics_dict:
                        f.write(
                            f"\n  Instance Accuracy (Detail):   {metrics_dict['instance_accuracy_detail']:.6f}\n"
                        )
                        f.write(
                            f"    ├─ Precision:               {metrics_dict['instance_precision_detail']:.6f}\n"
                        )
                        f.write(
                            f"    ├─ Recall:                  {metrics_dict['instance_recall_detail']:.6f}\n"
                        )
                        f.write(
                            f"    └─ F1:                      {metrics_dict['instance_f1_detail']:.6f}\n"
                        )
                    f.write("\n")

                # Write binary/semantic segmentation metrics
                if "jaccard" in metrics_dict or "dice" in metrics_dict:
                    f.write("Binary/Semantic Segmentation Metrics:\n")
                    f.write("-" * 80 + "\n")
                    if "jaccard" in metrics_dict:
                        f.write(f"  Jaccard Index:                {metrics_dict['jaccard']:.6f}\n")
                    if "dice" in metrics_dict:
                        f.write(f"  Dice Score:                   {metrics_dict['dice']:.6f}\n")
                    if "accuracy" in metrics_dict:
                        f.write(f"  Accuracy:                     {metrics_dict['accuracy']:.6f}\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")

            print(f"  💾 Metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"  ⚠️  Failed to save metrics to file: {e}")

    def _compute_test_metrics(
        self, decoded_predictions: np.ndarray, labels: torch.Tensor, volume_name: str = None
    ):
        """Update configured test metrics."""
        compute_test_metrics(self, decoded_predictions, labels, volume_name=volume_name)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""

        images = batch["image"]
        labels = batch["label"]
        raw_mask = batch.get("mask", None)
        # Binarize mask: (B, 1, D, H, W) float, 1 = valid, 0 = ignore
        mask = (raw_mask > 0).float() if raw_mask is not None else None

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(
            k.startswith("ds_") for k in outputs.keys()
        )

        # Compute loss using the loss orchestrator
        if is_deep_supervision:
            total_loss, loss_dict = self.loss_orchestrator.compute_deep_supervision_loss(
                outputs, labels, stage="train", mask=mask
            )
        else:
            total_loss, loss_dict = self.loss_orchestrator.compute_standard_loss(
                outputs, labels, stage="train", mask=mask
            )

        # Keep full training curves in TensorBoard while avoiding console spam.
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step with deep supervision support."""
        images = batch["image"]
        labels = batch["label"]
        raw_mask = batch.get("mask", None)
        mask = (raw_mask > 0).float() if raw_mask is not None else None

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(
            k.startswith("ds_") for k in outputs.keys()
        )

        # Compute loss using the loss orchestrator
        if is_deep_supervision:
            total_loss, loss_dict = self.loss_orchestrator.compute_deep_supervision_loss(
                outputs, labels, stage="val", mask=mask
            )
        else:
            total_loss, loss_dict = self.loss_orchestrator.compute_standard_loss(
                outputs, labels, stage="val", mask=mask
            )

        # Compute evaluation metrics if enabled
        evaluation_cfg = self._get_test_evaluation_config()
        evaluation_enabled = bool(self._cfg_value(evaluation_cfg, "enabled", False))
        metrics = self._cfg_value(evaluation_cfg, "metrics", None)

        if evaluation_enabled and metrics is not None:
            if is_deep_supervision:
                main_output = outputs["output"]
            else:
                main_output = outputs

            is_multi_task = self._has_multiple_supervised_loss_tasks()
            inference_eval_defaults = self._get_runtime_inference_config().evaluation
            prediction_threshold = self._cfg_float(
                evaluation_cfg,
                "prediction_threshold",
                inference_eval_defaults.prediction_threshold,
            )

            if is_multi_task:
                binary_output = main_output[:, 0:1, ...]
                binary_target = labels[:, 0:1, ...]
                preds = (binary_output.squeeze(1) > prediction_threshold).long()
                targets = binary_target.squeeze(1).long()
            elif main_output.shape[1] > 1:
                preds = torch.argmax(main_output, dim=1)
                targets = labels.squeeze(1).long()
            else:
                preds = (main_output.squeeze(1) > prediction_threshold).long()
                targets = labels.squeeze(1).long()

            if "jaccard" in metrics and self.val_jaccard is not None:
                self.val_jaccard(preds, targets)
                self.log(
                    "val_jaccard",
                    self.val_jaccard,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            if "dice" in metrics and self.val_dice is not None:
                self.val_dice(preds, targets)
                self.log("val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

            if "accuracy" in metrics and self.val_accuracy is not None:
                self.val_accuracy(preds, targets)
                self.log(
                    "val_accuracy",
                    self.val_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        # Show only validation total loss on the progress bar.
        if "val_loss_total" in loss_dict:
            self.log(
                "val_loss",
                loss_dict["val_loss_total"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=False,
                sync_dist=True,
            )

        # Log full validation losses to logger at epoch granularity.
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        return total_loss

    def on_test_start(self):
        """Called at the beginning of testing to initialize metrics and inferer."""
        self._setup_test_metrics()
        inference_cfg = self._get_runtime_inference_config()

        # Explicitly set eval mode if configured (Lightning does this by default, but be explicit)
        if getattr(inference_cfg, "do_eval", True):
            self.eval()
        else:
            # Keep in training mode (e.g., for Monte Carlo Dropout uncertainty estimation)
            self.train()

        sliding_cfg = getattr(inference_cfg, "sliding_window", None)
        if bool(getattr(sliding_cfg, "keep_input_on_cpu", False)):
            print(
                "  Sliding-window CPU input mode enabled: keeping test image tensors on CPU "
                "and letting MONAI move window batches to the configured sw_device."
            )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Keep large test/predict input volumes on CPU for MONAI sliding-window inference."""
        trainer = getattr(self, "_trainer", None)
        is_test_or_predict = bool(
            getattr(trainer, "testing", False) or getattr(trainer, "predicting", False)
        )
        inference_cfg = self._get_runtime_inference_config() if is_test_or_predict else None
        sliding_cfg = getattr(inference_cfg, "sliding_window", None)
        keep_input_on_cpu = bool(getattr(sliding_cfg, "keep_input_on_cpu", False))

        preserve_cpu_input = keep_input_on_cpu and is_test_or_predict and isinstance(batch, dict)
        cpu_image = None
        cpu_label = None
        cpu_mask = None
        if preserve_cpu_input:
            image = batch.get("image")
            label = batch.get("label")
            mask = batch.get("mask")
            if torch.is_tensor(image):
                cpu_image = image
            if torch.is_tensor(label):
                cpu_label = label
            if torch.is_tensor(mask):
                cpu_mask = mask

        moved_batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        if preserve_cpu_input and isinstance(moved_batch, dict):
            if cpu_image is not None:
                moved_batch["image"] = cpu_image
            if cpu_label is not None:
                moved_batch["label"] = cpu_label
            if cpu_mask is not None:
                moved_batch["mask"] = cpu_mask

        return moved_batch

    @staticmethod
    def _tta_cfg_len(value: Any) -> int:
        """Return sequence length for TTA config lists (supports OmegaConf ListConfig)."""
        if value is None or isinstance(value, str):
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    def _summarize_tta_plan(self, image_ndim: int) -> str:
        """Build a concise, accurate TTA summary for inference logs."""
        inference_cfg = self._get_runtime_inference_config()
        tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)

        if tta_cfg is None:
            return "Disabled"

        if not bool(getattr(tta_cfg, "enabled", False)):
            return "Disabled"

        flip_axes_cfg = getattr(tta_cfg, "flip_axes", None)
        rotation90_axes_cfg = getattr(tta_cfg, "rotation90_axes", None)
        channel_activations_cfg = getattr(tta_cfg, "channel_activations", None)

        spatial_dims = 3 if image_ndim == 5 else 2 if image_ndim == 4 else 0

        if flip_axes_cfg == "all" or flip_axes_cfg == []:
            flip_variants = 2**spatial_dims if spatial_dims > 0 else 1
        elif flip_axes_cfg is None:
            flip_variants = 1
        else:
            flip_variants = 1 + self._tta_cfg_len(flip_axes_cfg)

        if rotation90_axes_cfg == "all":
            rotation_planes = 3 if image_ndim == 5 else 1 if image_ndim == 4 else 0
        elif rotation90_axes_cfg is None:
            rotation_planes = 0
        else:
            rotation_planes = self._tta_cfg_len(rotation90_axes_cfg)

        passes_per_flip = 1 if rotation_planes == 0 else rotation_planes * 4
        total_passes = flip_variants * passes_per_flip
        geometric_transforms = max(total_passes - 1, 0)
        channel_activation_groups = self._tta_cfg_len(channel_activations_cfg)

        if geometric_transforms > 0:
            return f"Enabled ({geometric_transforms} geometric transforms, {total_passes} passes)"
        if channel_activation_groups > 0:
            return (
                f"Enabled (0 geometric transforms; channel_activations="
                f"{channel_activation_groups})"
            )
        return "Enabled (0 transforms; single pass)"

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        return run_test_step(self, batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = build_optimizer(self.cfg, self.model)

        # Build scheduler if configured (check both cfg.scheduler and cfg.optimization.scheduler)
        has_scheduler = (hasattr(self.cfg, "scheduler") and self.cfg.scheduler is not None) or (
            hasattr(self.cfg, "optimization")
            and hasattr(self.cfg.optimization, "scheduler")
            and self.cfg.optimization.scheduler is not None
        )

        if has_scheduler:
            scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Get scheduler interval from config (default: 'epoch')
            # Can be 'epoch' or 'step' to control when scheduler steps
            scheduler_interval = "epoch"  # default
            scheduler_frequency = 1  # default

            if hasattr(self.cfg, "optimization") and hasattr(self.cfg.optimization, "scheduler"):
                scheduler_interval = getattr(self.cfg.optimization.scheduler, "interval", "epoch")
                scheduler_frequency = getattr(self.cfg.optimization.scheduler, "frequency", 1)
            elif hasattr(self.cfg, "scheduler"):
                scheduler_interval = getattr(self.cfg.scheduler, "interval", "epoch")
                scheduler_frequency = getattr(self.cfg.scheduler, "frequency", 1)

            # Check if this is ReduceLROnPlateau (requires metric monitoring)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": scheduler_interval,  # Now configurable!
                "frequency": scheduler_frequency,
            }

            # Print scheduler configuration for verification
            print(
                f"  📅 Scheduler interval: '{scheduler_interval}' (frequency: {scheduler_frequency})"
            )
            if scheduler_interval == "step":
                print(f"  ℹ️  Scheduler will step every {scheduler_frequency} training step(s)")
            else:
                print(f"  ℹ️  Scheduler will step every {scheduler_frequency} epoch(s)")

            # ReduceLROnPlateau requires the 'monitor' key to pass the metric value
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Get monitor metric from scheduler config
                monitor_metric = None
                if hasattr(self.cfg, "optimization") and hasattr(
                    self.cfg.optimization, "scheduler"
                ):
                    monitor_metric = getattr(self.cfg.optimization.scheduler, "monitor", None)
                elif hasattr(self.cfg, "scheduler"):
                    monitor_metric = getattr(self.cfg.scheduler, "monitor", None)

                if monitor_metric:
                    scheduler_config["monitor"] = monitor_metric
                    print(f"  ✅ ReduceLROnPlateau will monitor: {monitor_metric}")
                else:
                    # Default to validation loss
                    scheduler_config["monitor"] = "val_loss_total"
                    print(
                        "  ⚠️  ReduceLROnPlateau will monitor: val_loss_total (default, no monitor specified in config)"
                    )

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
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
            lr = optimizer.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


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
