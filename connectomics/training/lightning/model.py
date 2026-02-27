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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

DEBUG_D1 = os.environ.get("PYTC_DEBUG_D1", "0").lower() in {"1", "true", "yes", "on"}
DEBUG_PDB = os.environ.get("PYTC_DEBUG_PDB", "0").lower() in {"1", "true", "yes", "on"}

# Import existing components
from ...config import Config
from ...decoding import apply_decode_mode, resolve_decode_modes_from_cfg
from ...inference import (
    InferenceManager,
    apply_postprocessing,
    apply_save_prediction_transform,
    resolve_output_filenames,
    write_outputs,
)
from ...models import build_model
from ...models.loss import create_loss, get_loss_metadata_for_module
from ..debugging import DebugManager

# Import training/inference components
from ..loss import LossOrchestrator, build_loss_weighter, infer_num_loss_tasks_from_config
from ..model_weights import load_external_weights
from ..optim import build_lr_scheduler, build_optimizer


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

        # Enable inline NaN detection (can be disabled via config)
        self.enable_nan_detection = getattr(cfg.model, "enable_nan_detection", True)
        self.debug_on_nan = getattr(cfg.model, "debug_on_nan", True)

        # Activation clamping to prevent inf (can be configured)
        self.clamp_activations = getattr(cfg.model, "clamp_activations", False)
        self.clamp_min = getattr(cfg.model, "clamp_min", -10.0)
        self.clamp_max = getattr(cfg.model, "clamp_max", 10.0)

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
        losses = getattr(cfg.model, "losses", None)
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
            result.append(create_loss(fn_name, **kwargs))
        return result

    def _extract_loss_weights(self, cfg) -> list:
        """Extract per-loss weights from unified losses configuration."""
        losses_list = self._get_losses_list(cfg)
        return [float(entry.get("weight", 1.0)) for entry in losses_list]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightning forward pass that delegates to the underlying model.

        This is required so Lightning can execute the module during training/inference.
        """
        return self.model(x)

    def _get_test_evaluation_config(self):
        """Resolve test evaluation config, falling back to inference evaluation if needed."""
        if hasattr(self.cfg, "test") and self.cfg.test is not None:
            test_eval = getattr(self.cfg.test, "evaluation", None)
            if test_eval is not None:
                return test_eval

        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "evaluation"):
            return self.cfg.inference.evaluation

        return None

    def _is_test_evaluation_enabled(self) -> bool:
        """Return whether test-time metric computation is enabled."""
        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            return False

        if isinstance(evaluation_config, dict):
            return bool(evaluation_config.get("enabled", False))

        return bool(getattr(evaluation_config, "enabled", False))

    def _has_multiple_supervised_loss_tasks(self) -> bool:
        """Infer multi-task supervised setup from compiled explicit loss terms."""
        pred_target_terms = [
            term for term in self.loss_orchestrator.loss_term_specs
            if term.call_kind == "pred_target"
        ]
        return len(pred_target_terms) > 1

    def _setup_test_metrics(self):
        """Initialize test metrics based on test or inference config."""
        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            return

        # Check if evaluation is enabled
        if not self._is_test_evaluation_enabled():
            return

        metrics = (
            evaluation_config.get("metrics", None)
            if isinstance(evaluation_config, dict)
            else getattr(evaluation_config, "metrics", None)
        )
        if metrics is None:
            return

        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, "out_channels") else 2

        # Create only the specified metrics
        if "jaccard" in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_jaccard = torchmetrics.JaccardIndex(task="binary").to(self.device)
            else:
                # Multi-class segmentation
                self.test_jaccard = torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=num_classes
                ).to(self.device)
        if "dice" in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_dice = torchmetrics.Dice(task="binary").to(self.device)
            else:
                # Multi-class segmentation
                self.test_dice = torchmetrics.Dice(num_classes=num_classes, average="macro").to(
                    self.device
                )
        if "accuracy" in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_accuracy = torchmetrics.Accuracy(task="binary").to(self.device)
            else:
                # Multi-class segmentation
                self.test_accuracy = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                ).to(self.device)
        if "adapted_rand" in metrics:
            from ...metrics.metrics_seg import AdaptedRandError

            self.test_adapted_rand = AdaptedRandError().to(self.device)
        if "voi" in metrics:
            from ...metrics.metrics_seg import VariationOfInformation

            self.test_voi = VariationOfInformation().to(self.device)
        if "instance_accuracy" in metrics:
            from ...metrics.metrics_seg import InstanceAccuracy

            self.test_instance_accuracy = InstanceAccuracy(thresh=0.5, criterion="iou").to(
                self.device
            )
        if "instance_accuracy_detail" in metrics:
            from ...metrics.metrics_seg import InstanceAccuracySimple

            self.test_instance_accuracy_detail = InstanceAccuracySimple(
                thresh=0.5, criterion="iou"
            ).to(self.device)

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
            print(
                f"  ℹ️  Intensity scaling was disabled (scale={intensity_scale}), no inversion needed"
            )

        return data

    def _resolve_test_output_config(
        self, batch: Dict[str, Any]
    ) -> tuple[str, Optional[str], str, List[str]]:
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

    def _save_metrics_to_file(self, metrics_dict: Dict[str, Any]):
        """
        Save evaluation metrics to a text file in the output directory.

        Args:
            metrics_dict: Dictionary containing metric names and values
        """
        metric_keys = [k for k in metrics_dict.keys() if k != "volume_name"]
        if not metric_keys:
            return

        # Get output path from config
        output_path = None
        if hasattr(self.cfg, "test") and hasattr(self.cfg.test, "data"):
            output_path = getattr(self.cfg.test.data, "output_path", None)

        if output_path is None:
            print("  ⚠️  Cannot save metrics: output_path not found in config")
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
        """Update configured torchmetrics using decoded predictions and print per-volume metrics."""
        if not self._is_test_evaluation_enabled():
            return

        pred_tensor = torch.from_numpy(decoded_predictions).float().to(self.device)
        labels_tensor = labels.float().to(pred_tensor.device)

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
                print("  ❌ Cannot compute metrics: incompatible shapes after alignment")
                print(f"     pred={pred_tensor.shape}, labels={labels_tensor.shape}")
                return

        # Compute per-volume metrics (print immediately)
        volume_prefix = f"[{volume_name}] " if volume_name else ""

        # Dictionary to collect all metrics for saving to file
        metrics_dict = {}
        metrics_dict["volume_name"] = volume_name if volume_name else "unknown"

        # Determine if this is instance segmentation or binary/semantic segmentation
        # Instance segmentation: predictions have integer instance IDs (0, 1, 2, ..., N)
        # Binary/semantic segmentation: predictions are probabilities [0, 1] or logits
        is_instance_segmentation = pred_tensor.dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ] or (pred_tensor.dtype == torch.float32 and pred_tensor.max() > 1.0)

        if is_instance_segmentation:
            # For instance segmentation: use instance IDs directly
            pred_instances = pred_tensor.long()
            labels_instances = labels_tensor.long()

            # Adapted Rand Error is for instance segmentation
            if hasattr(self, "test_adapted_rand") and isinstance(
                self.test_adapted_rand, torchmetrics.Metric
            ):
                from ...metrics.metrics_seg import AdaptedRandError

                # Use return_all_stats=True to get precision and recall
                per_volume_metric = AdaptedRandError(return_all_stats=True).to(self.device)
                per_volume_metric.update(pred_instances.cpu(), labels_instances.cpu())
                adapted_rand_value = per_volume_metric.compute()
                if isinstance(adapted_rand_value, dict):
                    are_score = adapted_rand_value.get(
                        "adapted_rand_error",
                        adapted_rand_value.get("are", list(adapted_rand_value.values())[0]),
                    )
                    are_score = are_score.item() if hasattr(are_score, "item") else float(are_score)
                else:
                    are_score = adapted_rand_value.item()
                print(f"  {volume_prefix}Adapted Rand Error: {are_score:.6f}")
                if isinstance(adapted_rand_value, dict):
                    for k, v in adapted_rand_value.items():
                        val = v.item() if hasattr(v, "item") else float(v)
                        print(f"  {volume_prefix}  {k}: {val:.6f}")

                # Collect metric
                metrics_dict["adapted_rand_error"] = are_score

                # Update running metric for epoch-level aggregation
                self.test_adapted_rand.update(pred_instances.cpu(), labels_instances.cpu())

                # Log metrics - handle both dict and tensor return values
                epoch_stats = self.test_adapted_rand.compute()
                if isinstance(epoch_stats, dict):
                    self.log(
                        "test_adapted_rand",
                        epoch_stats["adapted_rand_error"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
                    self.log(
                        "test_adapted_rand_precision",
                        epoch_stats["adapted_rand_precision"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
                    self.log(
                        "test_adapted_rand_recall",
                        epoch_stats["adapted_rand_recall"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )
                else:
                    self.log(
                        "test_adapted_rand",
                        epoch_stats,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

            # VOI (Variation of Information) is for instance segmentation
            if hasattr(self, "test_voi") and isinstance(self.test_voi, torchmetrics.Metric):
                from ...metrics.segmentation_numpy import voi

                # Compute per-volume VOI for immediate feedback
                split, merge = voi(pred_instances.cpu().numpy(), labels_instances.cpu().numpy())
                print(f"  {volume_prefix}VOI Split: {split:.6f}")
                print(f"  {volume_prefix}VOI Merge: {merge:.6f}")
                print(f"  {volume_prefix}VOI Total: {split + merge:.6f}")

                # Collect metrics
                metrics_dict["voi_split"] = split
                metrics_dict["voi_merge"] = merge
                metrics_dict["voi_total"] = split + merge

                # Update running metric for epoch-level aggregation
                self.test_voi.update(pred_instances.cpu(), labels_instances.cpu())
                self.log(
                    "test_voi",
                    self.test_voi,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    "test_voi_split",
                    self.test_voi.compute_split(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "test_voi_merge",
                    self.test_voi.compute_merge(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

            # Instance Accuracy (from instance_matching)
            if hasattr(self, "test_instance_accuracy") and isinstance(
                self.test_instance_accuracy, torchmetrics.Metric
            ):
                from ...metrics.segmentation_numpy import instance_matching

                # Compute per-volume instance accuracy for immediate feedback
                stats = instance_matching(
                    labels_instances.cpu().numpy(),
                    pred_instances.cpu().numpy(),
                    thresh=0.5,
                    criterion="iou",
                )
                print(f"  {volume_prefix}Instance Accuracy: {stats['accuracy']:.6f}")

                # Collect metric
                metrics_dict["instance_accuracy"] = stats["accuracy"]

                # Update running metric for epoch-level aggregation
                self.test_instance_accuracy.update(pred_instances.cpu(), labels_instances.cpu())
                self.log(
                    "test_instance_accuracy",
                    self.test_instance_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            # Instance Accuracy Detail (relaxed, no Hungarian matching - for debugging only)
            if hasattr(self, "test_instance_accuracy_detail") and isinstance(
                self.test_instance_accuracy_detail, torchmetrics.Metric
            ):
                from ...metrics.segmentation_numpy import instance_matching_simple

                # Compute per-volume relaxed instance accuracy for immediate feedback
                stats_simple = instance_matching_simple(
                    labels_instances.cpu().numpy(),
                    pred_instances.cpu().numpy(),
                    thresh=0.5,
                    criterion="iou",
                )
                print(
                    f"  {volume_prefix}Instance Accuracy (Detail): {stats_simple['accuracy']:.6f} [relaxed, non-Hungarian]"
                )
                print(f"  {volume_prefix}  ├─ Precision: {stats_simple['precision']:.6f}")
                print(f"  {volume_prefix}  ├─ Recall: {stats_simple['recall']:.6f}")
                print(f"  {volume_prefix}  └─ F1: {stats_simple['f1']:.6f}")

                # Collect metrics
                metrics_dict["instance_accuracy_detail"] = stats_simple["accuracy"]
                metrics_dict["instance_precision_detail"] = stats_simple["precision"]
                metrics_dict["instance_recall_detail"] = stats_simple["recall"]
                metrics_dict["instance_f1_detail"] = stats_simple["f1"]

                # Update running metric for epoch-level aggregation
                self.test_instance_accuracy_detail.update(
                    pred_instances.cpu(), labels_instances.cpu()
                )
                self.log(
                    "test_instance_accuracy_detail",
                    self.test_instance_accuracy_detail,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    "test_instance_precision_detail",
                    self.test_instance_accuracy_detail.compute_precision(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "test_instance_recall_detail",
                    self.test_instance_accuracy_detail.compute_recall(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "test_instance_f1_detail",
                    self.test_instance_accuracy_detail.compute_f1(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        else:
            # For binary/semantic segmentation: binarize predictions
            if pred_tensor.max() <= 1.0:
                pred_binary = (pred_tensor > 0.5).long()
            else:
                pred_binary = (torch.sigmoid(pred_tensor) > 0.5).long()

            labels_binary = (
                (labels_tensor > 0.5).long() if labels_tensor.max() <= 1.0 else labels_tensor.long()
            )

            if hasattr(self, "test_jaccard") and self.test_jaccard is not None:
                jaccard_value = torchmetrics.functional.jaccard_index(
                    pred_binary, labels_binary, task="binary"
                )
                print(f"  {volume_prefix}Jaccard: {jaccard_value.item():.6f}")
                metrics_dict["jaccard"] = jaccard_value.item()
                self.test_jaccard.update(pred_binary, labels_binary)
                self.log(
                    "test_jaccard",
                    self.test_jaccard,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            if hasattr(self, "test_dice") and self.test_dice is not None:
                dice_value = torchmetrics.functional.dice(pred_binary, labels_binary)
                print(f"  {volume_prefix}Dice: {dice_value.item():.6f}")
                metrics_dict["dice"] = dice_value.item()
                self.test_dice.update(pred_binary, labels_binary)
                self.log(
                    "test_dice",
                    self.test_dice,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            if hasattr(self, "test_accuracy") and self.test_accuracy is not None:
                accuracy_value = torchmetrics.functional.accuracy(
                    pred_binary, labels_binary, task="binary"
                )
                print(f"  {volume_prefix}Accuracy: {accuracy_value.item():.6f}")
                metrics_dict["accuracy"] = accuracy_value.item()
                self.test_accuracy.update(pred_binary, labels_binary)
                self.log(
                    "test_accuracy",
                    self.test_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        # Save metrics to file
        self._save_metrics_to_file(metrics_dict)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""

        images = batch["image"]
        labels = batch["label"]
        raw_mask = batch.get("mask", None)
        # Binarize mask: (B, 1, D, H, W) float, 1 = valid, 0 = ignore
        mask = (raw_mask > 0).float() if raw_mask is not None else None

        if DEBUG_PDB and batch_idx == 0 and self.current_epoch == 0:
            import pdb; pdb.set_trace()

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

        # [D1] Training diagnostics: log prediction and target statistics to detect SDT collapse
        # Print every 50 global steps to monitor if model learns positive SDT values
        if DEBUG_D1 and self.global_step % 50 == 0:
            with torch.no_grad():
                # Get main output for diagnostics (use 'output' key for deep supervision, or outputs directly)
                pred_raw = outputs["output"] if is_deep_supervision else outputs

                # Apply tanh to match the loss function's behavior (predictions should be in [-1, 1])
                pred_for_stats = torch.tanh(pred_raw)

                # Compute prediction statistics (after tanh)
                pred_min = pred_for_stats.min().item()
                pred_max = pred_for_stats.max().item()
                pred_mean = pred_for_stats.mean().item()
                pred_positive_frac = (pred_for_stats > 0).float().mean().item() * 100
                pred_near_minus1_frac = (pred_for_stats < -0.98).float().mean().item() * 100

                # Also show raw logit range for debugging
                pred_raw_min = pred_raw.min().item()
                pred_raw_max = pred_raw.max().item()

                # Compute target statistics
                target_min = labels.min().item()
                target_max = labels.max().item()
                target_mean = labels.mean().item()
                target_positive_frac = (labels > 0).float().mean().item() * 100

                print(
                    f"\n[D1 Step {self.global_step}] PRED (after tanh): min={pred_min:.3f}, max={pred_max:.3f}, "
                    f"mean={pred_mean:.3f}, >0: {pred_positive_frac:.1f}%, <-0.98: {pred_near_minus1_frac:.1f}%"
                )
                print(
                    f"[D1 Step {self.global_step}] PRED (raw logits): min={pred_raw_min:.3f}, max={pred_raw_max:.3f}"
                )
                print(
                    f"[D1 Step {self.global_step}] TARGET: min={target_min:.3f}, max={target_max:.3f}, "
                    f"mean={target_mean:.3f}, >0: {target_positive_frac:.1f}%"
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
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "evaluation"):
            if getattr(self.cfg.inference.evaluation, "enabled", False):
                metrics = getattr(self.cfg.inference.evaluation, "metrics", None)
                if metrics is not None:
                    # Get the main output for metric computation
                    if is_deep_supervision:
                        main_output = outputs["output"]
                    else:
                        main_output = outputs

                    # Check if this is a multi-task supervised loss setup.
                    is_multi_task = self._has_multiple_supervised_loss_tasks()

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
                    if "jaccard" in metrics:
                        if not hasattr(self, "val_jaccard"):
                            num_classes = (
                                self.cfg.model.out_channels
                                if hasattr(self.cfg.model, "out_channels")
                                else 2
                            )
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_jaccard = torchmetrics.JaccardIndex(task="binary").to(
                                    self.device
                                )
                            else:
                                # Multi-class segmentation
                                self.val_jaccard = torchmetrics.JaccardIndex(
                                    task="multiclass", num_classes=num_classes
                                ).to(self.device)
                        self.val_jaccard(preds, targets)
                        self.log(
                            "val_jaccard",
                            self.val_jaccard,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        )

                    if "dice" in metrics:
                        if not hasattr(self, "val_dice"):
                            num_classes = (
                                self.cfg.model.out_channels
                                if hasattr(self.cfg.model, "out_channels")
                                else 2
                            )
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_dice = torchmetrics.Dice(task="binary").to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_dice = torchmetrics.Dice(
                                    num_classes=num_classes, average="macro"
                                ).to(self.device)
                        self.val_dice(preds, targets)
                        self.log(
                            "val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True
                        )

                    if "accuracy" in metrics:
                        if not hasattr(self, "val_accuracy"):
                            num_classes = (
                                self.cfg.model.out_channels
                                if hasattr(self.cfg.model, "out_channels")
                                else 2
                            )
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_accuracy = torchmetrics.Accuracy(task="binary").to(
                                    self.device
                                )
                            else:
                                # Multi-class segmentation
                                self.val_accuracy = torchmetrics.Accuracy(
                                    task="multiclass", num_classes=num_classes
                                ).to(self.device)
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

        # Explicitly set eval mode if configured (Lightning does this by default, but be explicit)
        if hasattr(self.cfg, "inference") and getattr(self.cfg.inference, "do_eval", True):
            self.eval()
        else:
            # Keep in training mode (e.g., for Monte Carlo Dropout uncertainty estimation)
            self.train()

        sliding_cfg = getattr(getattr(self.cfg, "inference", None), "sliding_window", None)
        if bool(getattr(sliding_cfg, "keep_input_on_cpu", False)):
            print(
                "  Sliding-window CPU input mode enabled: keeping test image tensors on CPU "
                "and letting MONAI move window batches to the configured sw_device."
            )

    def on_test_end(self):
        """Called at the end of testing."""
        # Note: Metrics are logged in test_step with on_epoch=True,
        # which automatically computes and logs them at the end of testing.
        # Logging here causes a Lightning warning since self.log() is not allowed in on_test_end.
        pass

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Keep large test/predict input volumes on CPU for MONAI sliding-window inference."""
        sliding_cfg = getattr(getattr(self.cfg, "inference", None), "sliding_window", None)
        keep_input_on_cpu = bool(getattr(sliding_cfg, "keep_input_on_cpu", False))

        trainer = getattr(self, "_trainer", None)
        is_test_or_predict = bool(
            getattr(trainer, "testing", False) or getattr(trainer, "predicting", False)
        )

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
            print(
                "  ✅ Loaded final predictions from disk, skipping inference/decoding/postprocessing"
            )
            if labels is not None and self._is_test_evaluation_enabled():
                self._compute_test_metrics(predictions_np, labels, volume_name=volume_name)
            return torch.tensor(0.0, device=self.device)

        # CASE 2: Intermediate predictions exist → decode and postprocess
        if loaded_intermediate_predictions:
            print("  ✅ Loaded intermediate predictions from disk, skipping inference")
            print(f"\n{'=' * 70}")
            print(f"PROCESSING VOLUME: {volume_name}")
            print(f"{'=' * 70}")

            # Convert back from saved format to [0,1] predictions if needed
            predictions_np = self._invert_save_prediction_transform(predictions_np)

            # Decode and postprocess
            import time

            print("\n  🔄 [STAGE: Decoding Instances]")
            decode_start = time.time()

            has_decoding_cfg = bool(resolve_decode_modes_from_cfg(self.cfg))
            decoded_predictions = apply_decode_mode(self.cfg, predictions_np)

            decode_duration = time.time() - decode_start
            print(f"  ✅ Decoding completed ({decode_duration:.1f}s)")

            if has_decoding_cfg:
                postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)

                # Summary of decoded output
                print("\n  📊 Decoded Segmentation Summary:")
                print(f"      Shape:      {decoded_predictions.shape}")
                print(f"      Dtype:      {decoded_predictions.dtype}")
                print(f"      Min:        {decoded_predictions.min()}")
                print(f"      Max:        {decoded_predictions.max()}")
                print(f"      Instances:  {decoded_predictions.max()} (max label)")
                unique_count = len(np.unique(decoded_predictions))
                print(f"      Unique IDs: {unique_count}")
                print("")

                # Save final predictions
                print("  💾 [STAGE: Saving Final Predictions]")
                save_start = time.time()

                write_outputs(
                    self.cfg,
                    postprocessed_predictions,
                    filenames,
                    suffix="prediction",
                    mode=mode,
                    batch_meta=batch.get("image_meta_dict"),
                )

                save_duration = time.time() - save_start
                print(f"  ✅ Final predictions saved ({save_duration:.1f}s)")
            else:
                print("  ⏭️  Skipping postprocessing (no decoding configuration)")
                print("  ⏭️  Skipping decoded segmentation summary (no decoding configuration)")
                print("  ⏭️  Skipping final prediction save (no decoding configuration)")

            # Evaluate if labels provided
            if labels is not None and self._is_test_evaluation_enabled():
                print("\n  📈 [STAGE: Computing Evaluation Metrics]")
                eval_start = time.time()

                self._compute_test_metrics(decoded_predictions, labels, volume_name=volume_name)

                eval_duration = time.time() - eval_start
                print(f"  ✅ Evaluation completed ({eval_duration:.1f}s)")
            elif labels is None:
                print("\n  ⏭️  [STAGE: Evaluation] Skipped (no ground truth labels)")
            else:
                print("\n  ⏭️  [STAGE: Evaluation] Skipped (evaluation disabled)")

            print(f"\n{'=' * 70}")
            print(f"VOLUME COMPLETE: {volume_name}")
            print(f"{'=' * 70}\n")

            return torch.tensor(0.0, device=self.device)

        # CASE 3: No cached predictions → run full inference pipeline
        print("  🔄 No cached predictions found, running inference")

        # ============================================================
        # PART 1: Inference Plan Summary
        # ============================================================
        print(f"\n{'=' * 70}")
        print(f"INFERENCE PLAN: {volume_name}")
        print(f"{'=' * 70}")
        print(f"Input shape:       {tuple(images.shape)}")
        print(f"Input device:      {images.device}")

        # Extract sliding window parameters if available
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "sliding_window"):
            sw_cfg = self.cfg.inference.sliding_window
            roi_size = getattr(sw_cfg, "roi_size", "N/A")
            overlap = getattr(sw_cfg, "overlap", "N/A")
            sw_batch = getattr(sw_cfg, "sw_batch_size", "N/A")
            blending = getattr(sw_cfg, "blending", "gaussian")
            print(f"Sliding window ROI: {roi_size}")
            print(f"Overlap:            {overlap}")
            print(f"SW batch size:      {sw_batch}")
            print(f"Blending mode:      {blending}")
        else:
            print("Sliding window:     [Direct inference, no sliding window]")

        # TTA info
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "test_time_augmentation"):
            tta_cfg = self.cfg.inference.test_time_augmentation
            tta_enabled = getattr(tta_cfg, "enabled", False) if tta_cfg else False
            if tta_enabled:
                transforms = getattr(tta_cfg, "transforms", [])
                print(f"TTA:                Enabled ({len(transforms)} transforms)")
            else:
                print("TTA:                Disabled")
        print(f"{'=' * 70}\n")

        # ============================================================
        # PART 2: Timed Sliding-Window Inference
        # ============================================================
        import time

        inference_start = time.time()
        print("  ⏱️  Starting sliding-window inference...")

        # Run inference (cfg.test used for data loading and transforms via datamodule)
        predictions = self.inference_manager.predict_with_tta(images, mask=mask)
        predictions_np = predictions.detach().cpu().float().numpy()

        inference_end = time.time()
        inference_duration = inference_end - inference_start
        print(
            f"  ✅ Inference completed in {inference_duration / 60:.2f} minutes ({inference_duration:.1f}s)"
        )

        # ============================================================
        # PART 3: Prediction Tensor Summary
        # ============================================================
        print("\n  📊 Prediction Summary:")
        print(f"      Shape:  {predictions_np.shape}")
        print(f"      Dtype:  {predictions_np.dtype}")
        print(f"      Min:    {predictions_np.min():.6f}")
        print(f"      Max:    {predictions_np.max():.6f}")
        print(f"      Mean:   {predictions_np.mean():.6f}")
        print("")

        # Save intermediate predictions if configured
        save_intermediate = False
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "save_prediction"):
            save_intermediate = getattr(self.cfg.inference.save_prediction, "enabled", False)

        if save_intermediate:
            print("\n  💾 [STAGE: Saving Intermediate Predictions]")
            save_start = time.time()

            # Apply intensity scaling and dtype conversion before saving
            predictions_to_save = apply_save_prediction_transform(self.cfg, predictions_np)
            write_outputs(
                self.cfg,
                predictions_to_save,
                filenames,
                suffix="tta_prediction",
                mode=mode,
                batch_meta=batch.get("image_meta_dict"),
            )

            save_duration = time.time() - save_start
            print(f"  ✅ Intermediate predictions saved ({save_duration:.1f}s)")

        # ============================================================
        # PART 4: Decoding Stage
        # ============================================================
        print("\n  🔄 [STAGE: Decoding Instances]")
        decode_start = time.time()

        has_decoding_cfg = bool(resolve_decode_modes_from_cfg(self.cfg))
        decoded_predictions = apply_decode_mode(self.cfg, predictions_np)

        decode_duration = time.time() - decode_start
        print(f"  ✅ Decoding completed ({decode_duration:.1f}s)")

        if has_decoding_cfg:
            postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)

            # Summary of decoded output
            print("\n  📊 Decoded Segmentation Summary:")
            print(f"      Shape:      {decoded_predictions.shape}")
            print(f"      Dtype:      {decoded_predictions.dtype}")
            print(f"      Min:        {decoded_predictions.min()}")
            print(f"      Max:        {decoded_predictions.max()}")
            print(f"      Instances:  {decoded_predictions.max()} (max label)")
            unique_count = len(np.unique(decoded_predictions))
            print(f"      Unique IDs: {unique_count}")
            print("")

            # ============================================================
            # PART 5: Saving Final Predictions
            # ============================================================
            print("  💾 [STAGE: Saving Final Predictions]")
            final_save_start = time.time()

            write_outputs(
                self.cfg,
                postprocessed_predictions,
                filenames,
                suffix="prediction",
                mode=mode,
                batch_meta=batch.get("image_meta_dict"),
            )

            final_save_duration = time.time() - final_save_start
            print(f"  ✅ Final predictions saved ({final_save_duration:.1f}s)")
        else:
            print("  ⏭️  Skipping postprocessing (no decoding configuration)")
            print("  ⏭️  Skipping decoded segmentation summary (no decoding configuration)")
            print("  ⏭️  Skipping final prediction save (no decoding configuration)")

        # ============================================================
        # PART 6: Evaluation Stage
        # ============================================================

        # ============================================================
        # PART 6: Evaluation Stage
        # ============================================================
        # Evaluate if labels provided
        if labels is not None and self._is_test_evaluation_enabled():
            print("\n  📈 [STAGE: Computing Evaluation Metrics]")
            eval_start = time.time()

            self._compute_test_metrics(decoded_predictions, labels, volume_name=volume_name)

            eval_duration = time.time() - eval_start
            print(f"  ✅ Evaluation completed ({eval_duration:.1f}s)")
        elif labels is None:
            print("\n  ⏭️  [STAGE: Evaluation] Skipped (no ground truth labels)")
        else:
            print("\n  ⏭️  [STAGE: Evaluation] Skipped (evaluation disabled)")

        print(f"\n{'=' * 70}")
        print(f"VOLUME COMPLETE: {volume_name}")
        print(f"{'=' * 70}\n")

        return torch.tensor(0.0, device=self.device)

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
