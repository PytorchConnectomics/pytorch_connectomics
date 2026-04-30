"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing

The implementation delegates to specialized modules:
- connectomics.training.losses: Loss orchestration and weighting (PyTorch-only)
- connectomics.inference: Sliding window inference and test-time augmentation
- connectomics.training.debugging: NaN detection and debugging utilities
"""

from __future__ import annotations

import logging
import os
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
from ...data.io import read_volume
from ...evaluation import EvaluationContext, log_test_epoch_metrics, save_metrics_to_file
from ...inference import (
    InferenceManager,
    resolve_output_filenames,
)
from ...metrics.metrics_seg import (
    AdaptedRandError,
    InstanceAccuracy,
    InstanceAccuracySimple,
    VariationOfInformation,
)
from ...models import build_model
from ...models.losses import create_loss, get_loss_metadata_for_module
from ...runtime.output_naming import (
    final_prediction_decoded_glob_suffix,
    final_prediction_output_tag,
    is_tta_cache_suffix,
    resolve_prediction_cache_suffix,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
)
from ...utils import (
    resolve_channel_range,
    resolve_configured_output_head,
    resolve_head_target_slice,
    select_output_tensor,
)
from ..debugging import DebugManager

# Import training/inference components
from ..losses import LossOrchestrator, build_loss_weighter, infer_num_loss_tasks_from_config
from ..model_weights import load_external_weights
from ..optimization import build_lr_scheduler, build_optimizer
from .test_pipeline import run_test_step

logger = logging.getLogger(__name__)


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
        skip_loss: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        # Build model
        self.model = model if model is not None else self._build_model(cfg)

        # Skip loss/optimizer setup for decode-only mode
        if skip_loss:
            self.loss_functions = []
            self.loss_weights = []
            self.loss_metadata = []
            self.loss_weighter = None
            self.enable_nan_detection = False
            self.debug_on_nan = False
            self.loss_orchestrator = None
        else:
            self._init_losses(cfg)

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
        self.test_adapted_rand = None
        self.test_voi = None
        self.test_instance_accuracy = None
        self.test_instance_accuracy_detail = None
        self.val_jaccard = None
        self.val_dice = None
        self.val_accuracy = None
        self._val_metrics_initialized = False

        # Prediction saving state
        self._prediction_save_counter = 0

    def _init_losses(self, cfg):
        """Initialize loss functions, weights, and orchestrator."""
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

    def _build_model(self, cfg) -> nn.Module:
        """Build model from configuration."""
        model = build_model(cfg)
        external_weights_path = getattr(cfg.model, "external_weights_path", None)
        if external_weights_path:
            logger.info(f"Loading external weights from: {external_weights_path}")
            model = load_external_weights(model, cfg)
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
                logger.info(f"Ignoring stale loss-function checkpoint key(s): {preview}")

        return super().load_state_dict(state_dict, strict=strict)

    def _get_test_evaluation_config(self):
        """Return merged runtime evaluation config."""
        return getattr(self.cfg, "evaluation", None)

    def _is_test_evaluation_enabled(self) -> bool:
        """Return whether test-time metric computation is enabled."""
        evaluation_config = self._get_test_evaluation_config()
        if evaluation_config is None:
            return False

        return bool(self._cfg_value(evaluation_config, "enabled", False))

    @staticmethod
    def _cfg_value(cfg_obj: Any, key: str, default: Any = None) -> Any:
        """Unified dict/attribute config accessor (delegates to shared utility)."""
        from ...config.pipeline.dict_utils import cfg_get

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
            term
            for term in self.loss_orchestrator.loss_term_specs
            if term.call_kind == "pred_target"
        ]
        return len(pred_target_terms) > 1

    def _resolve_named_output_channels(self, *, purpose: str) -> Optional[int]:
        model_heads = getattr(self.cfg.model, "heads", None) or {}
        if not model_heads:
            return None

        selected_head = resolve_configured_output_head(
            self.cfg,
            purpose=purpose,
            allow_none=True,
        )
        if selected_head is None or selected_head not in model_heads:
            return None
        return int(self._cfg_value(model_heads[selected_head], "out_channels", 0))

    @staticmethod
    def _slice_tensor_channels(
        tensor: torch.Tensor,
        channel_selector,
        *,
        context: str,
    ) -> torch.Tensor:
        start_idx, end_idx = resolve_channel_range(
            channel_selector,
            num_channels=int(tensor.shape[1]),
            context=context,
        )
        return tensor[:, start_idx:end_idx, ...]

    def _resolve_validation_target_slice(self, head_name: str):
        """Infer the supervised label slice corresponding to a named output head."""
        explicit_target_slice = resolve_head_target_slice(self.cfg, head_name)
        if explicit_target_slice is not None:
            return explicit_target_slice

        primary_head = getattr(self.cfg.model, "primary_head", None)
        matching_slices = []
        for term in self.loss_orchestrator.loss_term_specs:
            if term.call_kind != "pred_target":
                continue
            resolved_head = term.pred_head if term.pred_head is not None else primary_head
            if resolved_head == head_name:
                if term.target_slice not in matching_slices:
                    matching_slices.append(term.target_slice)

        if not matching_slices:
            raise ValueError(
                f"Validation metric computation could not find any pred_target loss term for "
                f"head '{head_name}'."
            )
        if len(matching_slices) > 1:
            raise ValueError(
                f"Validation metric computation found multiple target slices for head "
                f"'{head_name}': {matching_slices}. Configure a single supervised label slice "
                "for the evaluated head."
            )
        return matching_slices[0]

    @staticmethod
    def _prepare_metric_predictions(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        prediction_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize prediction/target tensors for torchmetrics consumption."""
        if int(prediction.shape[1]) == 1:
            if int(target.shape[1]) != 1:
                raise ValueError(
                    "Binary metric computation expects a single target channel, got "
                    f"{tuple(target.shape)}."
                )
            preds = (prediction.squeeze(1) > prediction_threshold).long()
            targets = target.squeeze(1).long()
            return preds, targets

        preds = torch.argmax(prediction, dim=1)
        if int(target.shape[1]) == 1:
            targets = target.squeeze(1).long()
        elif int(target.shape[1]) == int(prediction.shape[1]):
            targets = torch.argmax(target, dim=1).long()
        else:
            raise ValueError(
                "Multiclass metric computation requires either a single class-index target "
                "channel or the same number of one-hot channels as the prediction. "
                f"Got prediction.shape={tuple(prediction.shape)} and target.shape={tuple(target.shape)}."
            )
        return preds, targets

    def _create_metrics(
        self,
        prefix: str,
        metrics: list,
        num_classes: int,
        use_binary: bool,
        instance_iou_threshold: float = 0.5,
    ):
        """Create and attach torchmetrics with the given prefix (test_ or val_)."""
        if "jaccard" in metrics:
            setattr(
                self,
                f"{prefix}jaccard",
                (
                    torchmetrics.JaccardIndex(task="binary").to(self.device)
                    if use_binary
                    else torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(
                        self.device
                    )
                ),
            )
        if "dice" in metrics:
            setattr(
                self,
                f"{prefix}dice",
                (
                    torchmetrics.Dice(task="binary").to(self.device)
                    if use_binary
                    else torchmetrics.Dice(num_classes=num_classes, average="macro").to(self.device)
                ),
            )
        if "accuracy" in metrics:
            setattr(
                self,
                f"{prefix}accuracy",
                (
                    torchmetrics.Accuracy(task="binary").to(self.device)
                    if use_binary
                    else torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
                        self.device
                    )
                ),
            )
        # Instance-level metrics only for test
        if prefix == "test_":
            if "adapted_rand" in metrics:
                setattr(self, f"{prefix}adapted_rand", AdaptedRandError().to(self.device))
            if "voi" in metrics:
                setattr(self, f"{prefix}voi", VariationOfInformation().to(self.device))
            if "instance_accuracy" in metrics:
                setattr(
                    self,
                    f"{prefix}instance_accuracy",
                    InstanceAccuracy(
                        thresh=instance_iou_threshold,
                        criterion="iou",
                    ).to(self.device),
                )
            if "instance_accuracy_detail" in metrics:
                setattr(
                    self,
                    f"{prefix}instance_accuracy_detail",
                    InstanceAccuracySimple(
                        thresh=instance_iou_threshold,
                        criterion="iou",
                    ).to(self.device),
                )

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

        evaluation_defaults = getattr(self.cfg, "evaluation", None)
        instance_iou_threshold = float(
            self._cfg_value(
                evaluation_config,
                "instance_iou_threshold",
                self._cfg_value(evaluation_defaults, "instance_iou_threshold", 0.5),
            )
        )
        named_output_channels = self._resolve_named_output_channels(
            purpose="test metric setup",
        )
        if named_output_channels == 1:
            num_classes = 1
        else:
            num_classes = (
                self.cfg.model.out_channels if hasattr(self.cfg.model, "out_channels") else 2
            )
        self._create_metrics(
            "test_", metrics, num_classes, num_classes == 1, instance_iou_threshold
        )

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

        named_output_channels = self._resolve_named_output_channels(
            purpose="validation metric setup",
        )
        if named_output_channels is not None:
            num_classes = named_output_channels
            use_binary = num_classes == 1
        else:
            is_multi_task = self._has_multiple_supervised_loss_tasks()
            num_classes = (
                self.cfg.model.out_channels if hasattr(self.cfg.model, "out_channels") else 2
            )
            use_binary = is_multi_task or num_classes == 1
        self._create_metrics("val_", metrics, num_classes, use_binary)
        self._val_metrics_initialized = True

    def on_validation_start(self) -> None:
        """Called before validation starts."""
        self._setup_validation_metrics()

    def _resolve_test_output_config(
        self, batch: Dict[str, Any]
    ) -> tuple[str, Optional[str], str, List[str]]:
        """Determine output dir/cache suffix from merged runtime inference config."""
        mode = "test"
        save_pred_cfg = self._get_runtime_inference_config().save_prediction
        output_dir_value = getattr(save_pred_cfg, "output_path", None)
        cache_suffix = resolve_prediction_cache_suffix(
            self.cfg,
            mode=mode,
            checkpoint_path=self._get_prediction_checkpoint_path(),
        )

        filenames = resolve_output_filenames(self.cfg, batch, global_step=self.global_step)
        return mode, output_dir_value, cache_suffix, filenames

    def _get_prediction_checkpoint_path(self) -> str:
        """Return the checkpoint/weights path whose stem should tag prediction caches."""
        explicit_path = getattr(self, "_prediction_checkpoint_path", None)
        if explicit_path is not None:
            path_value = str(explicit_path).strip()
            if path_value:
                return path_value

        trainer = getattr(self, "_trainer", None)
        trainer_ckpt_path = getattr(trainer, "ckpt_path", None) if trainer is not None else None
        if trainer_ckpt_path is not None:
            path_value = str(trainer_ckpt_path).strip()
            if path_value:
                return path_value

        external_weights_path = getattr(
            getattr(self.cfg, "model", None), "external_weights_path", None
        )
        if isinstance(external_weights_path, str) and external_weights_path.strip():
            return external_weights_path.strip()

        return ""

    def _resolve_tta_result_path_override(self) -> str:
        """Return explicit intermediate prediction file from inference.tta_result_path."""
        inference_cfg = self._get_runtime_inference_config()
        value = getattr(inference_cfg, "tta_result_path", "")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""

    def _load_cached_predictions(
        self, output_dir_value: Optional[str], filenames: List[str], cache_suffix: str, mode: str
    ):
        """Attempt to load cached predictions from disk."""
        # Check decoding.input_prediction_path first (decode-only mode)
        saved_path = getattr(getattr(self.cfg, "decoding", None), "input_prediction_path", "")
        if saved_path and isinstance(saved_path, str) and saved_path.strip():
            pred_file = Path(saved_path.strip()).expanduser()
            if not pred_file.is_absolute():
                pred_file = Path.cwd() / pred_file
            if pred_file.exists():
                logger.info(f"Loading saved prediction (decode-only): {pred_file}")
                pred = read_volume(str(pred_file), dataset="main")
                if pred.ndim < 4:
                    pred = pred[np.newaxis, ...]
                return pred, True, "_prediction.h5"
            else:
                raise FileNotFoundError(f"decoding.input_prediction_path not found: {pred_file}")

        explicit_prediction = self._resolve_tta_result_path_override()
        if isinstance(explicit_prediction, str) and explicit_prediction.strip():
            pred_file = Path(explicit_prediction).expanduser()
            if not pred_file.is_absolute():
                pred_file = Path.cwd() / pred_file

            if os.path.exists(pred_file):
                try:
                    logger.info(f"Using explicit inference.tta_result_path file: {pred_file}")
                    pred = read_volume(str(pred_file), dataset="main")
                    if pred.ndim < 4:
                        pred = pred[np.newaxis, ...]
                    if len(filenames) > 1:
                        logger.warning(
                            f"inference.tta_result_path is a single file while batch has "
                            f"{len(filenames)} filenames; decoding will use the explicit file only."
                        )
                    # Treat explicit file as intermediate prediction so decoding still runs.
                    return (
                        pred,
                        True,
                        tta_cache_suffix(
                            self.cfg,
                            checkpoint_path=self._get_prediction_checkpoint_path(),
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load explicit inference.tta_result_path file {pred_file}: {e}. "
                        f"Falling back to computed cache paths."
                    )
            else:
                logger.warning(
                    f"inference.tta_result_path file not found: {pred_file}. "
                    f"Falling back to computed cache paths."
                )

        if not output_dir_value:
            return None, False, cache_suffix

        output_dir = Path(output_dir_value)

        # Build ordered list of suffixes to try: final prediction first, then
        # intermediate TTA, then glob fallback.
        suffixes_to_try: list[str] = []
        if is_tta_cache_suffix(cache_suffix):
            # Prefer the final decoded file (e.g. _x16_prediction.h5) over
            # the intermediate TTA file (e.g. _tta_x16_prediction.h5).
            final_suffix = (
                "_"
                + final_prediction_output_tag(
                    self.cfg,
                    checkpoint_path=self._get_prediction_checkpoint_path(),
                )
                + ".h5"
            )
            suffixes_to_try.append(final_suffix)
        suffixes_to_try.append(cache_suffix)

        # Glob fallback before exact suffix matching: any pre-existing decoded
        # final file matching the same TTA/head/channel/checkpoint prefix lets
        # us skip a multi-GB intermediate TTA reload + redecode, even if the
        # current config's decoding kwargs differ from the cached file.
        if is_tta_cache_suffix(cache_suffix):
            decoded_glob = final_prediction_decoded_glob_suffix(
                self.cfg,
                checkpoint_path=self._get_prediction_checkpoint_path(),
            )
            decoded_files: list[Path] = []
            for filename in filenames:
                matches = sorted(output_dir.glob(f"{filename}{decoded_glob}"))
                if not matches:
                    decoded_files = []
                    break
                decoded_files.append(matches[-1])
            if decoded_files and len(decoded_files) == len(filenames):
                try:
                    preds = [read_volume(str(p), dataset="main") for p in decoded_files]
                except Exception as e:
                    logger.warning(
                        f"Failed to load decoded glob match {decoded_files[0]}: {e}; "
                        f"falling back to exact suffix matching."
                    )
                    preds = None
                if preds is not None:
                    logger.info(
                        "Loaded existing decoded final prediction(s) via glob fallback "
                        "(%s); skipping inference and decoding.",
                        decoded_files[0].name,
                    )
                    if len(preds) == 1:
                        predictions_np = preds[0]
                        if predictions_np.ndim < 4:
                            predictions_np = predictions_np[np.newaxis, ...]
                    else:
                        predictions_np = np.stack(
                            [p[np.newaxis, ...] if p.ndim < 4 else p for p in preds],
                            axis=0,
                        )
                    chosen_suffix = decoded_files[0].name[len(filenames[0]):]
                    return predictions_np, True, chosen_suffix

        for try_suffix in suffixes_to_try:
            existing_predictions = []
            all_exist = True
            for filename in filenames:
                pred_file = output_dir / f"{filename}{try_suffix}"
                if os.path.exists(pred_file):
                    try:
                        pred = read_volume(str(pred_file), dataset="main")
                        existing_predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"Failed to load {pred_file}: {e}, will re-run inference")
                        all_exist = False
                        break
                else:
                    all_exist = False
                    break

            if all_exist and len(existing_predictions) == len(filenames):
                logger.info(
                    "All prediction files exist (%s). Loading %d predictions and skipping inference.",
                    try_suffix,
                    len(existing_predictions),
                )
                if len(existing_predictions) == 1:
                    predictions_np = existing_predictions[0]
                    if predictions_np.ndim < 4:
                        predictions_np = predictions_np[np.newaxis, ...]
                else:
                    predictions_np = np.stack(
                        [p[np.newaxis, ...] if p.ndim < 4 else p for p in existing_predictions],
                        axis=0,
                    )
                return predictions_np, True, try_suffix

        # Targeted fallback: look for the exact TTA intermediate cache suffix
        # matching the current config rather than any arbitrary TTA file.
        if mode == "test" and not is_tta_cache_suffix(cache_suffix):
            fallback_suffixes = tta_cache_suffix_candidates(
                self.cfg,
                checkpoint_path=self._get_prediction_checkpoint_path(),
            )
            for try_suffix in fallback_suffixes:
                existing_predictions = []
                all_exist = True
                for filename in filenames:
                    pred_file = output_dir / f"{filename}{try_suffix}"
                    if not os.path.exists(pred_file):
                        all_exist = False
                        break
                    try:
                        pred = read_volume(str(pred_file), dataset="main")
                        existing_predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"Failed to load {pred_file}: {e}")
                        all_exist = False
                        break
                if all_exist and len(existing_predictions) == len(filenames):
                    logger.info(
                        "Loaded fallback TTA prediction(s) using exact suffix %s",
                        try_suffix,
                    )
                    if len(existing_predictions) == 1:
                        predictions_np = existing_predictions[0]
                        if predictions_np.ndim < 4:
                            predictions_np = predictions_np[np.newaxis, ...]
                    else:
                        predictions_np = np.stack(
                            [p[np.newaxis, ...] if p.ndim < 4 else p for p in existing_predictions],
                            axis=0,
                        )
                    return predictions_np, True, try_suffix

        return None, False, cache_suffix

    def _is_distributed_single_volume_sharding_active(self) -> bool:
        manager = self.inference_manager
        tta_active = bool(manager.is_distributed_tta_sharding_enabled())
        window_active = (
            bool(manager.is_distributed_window_sharding_enabled())
            if hasattr(manager, "is_distributed_window_sharding_enabled")
            else False
        )
        return tta_active or window_active

    def _test_metric_handles(self) -> Dict[str, Any]:
        return {
            "jaccard": self.test_jaccard,
            "dice": self.test_dice,
            "accuracy": self.test_accuracy,
            "adapted_rand": self.test_adapted_rand,
            "voi": self.test_voi,
            "instance_accuracy": self.test_instance_accuracy,
            "instance_accuracy_detail": self.test_instance_accuracy_detail,
        }

    def _evaluation_context(self) -> EvaluationContext:
        return EvaluationContext(
            cfg=self.cfg,
            evaluation_cfg=self._get_test_evaluation_config(),
            inference_cfg=self._get_runtime_inference_config(),
            device=self.device,
            enabled=self._is_test_evaluation_enabled(),
            checkpoint_path=self._get_prediction_checkpoint_path(),
            metrics=self._test_metric_handles(),
            log_fn=self.log,
            distributed_single_volume_sharding=self._is_distributed_single_volume_sharding_active(),
        )

    def _save_metrics_to_file(self, metrics_dict: Dict[str, Any]):
        save_metrics_to_file(self._evaluation_context(), metrics_dict)

    def _compute_loss(
        self,
        outputs,
        labels: torch.Tensor,
        stage: str,
        mask: Optional[torch.Tensor] = None,
    ):
        """Compute loss handling both standard and deep supervision outputs."""
        is_deep_supervision = isinstance(outputs, dict) and any(
            k.startswith("ds_") for k in outputs.keys()
        )
        if is_deep_supervision:
            return self.loss_orchestrator.compute_deep_supervision_loss(
                outputs, labels, stage=stage, mask=mask
            )
        return self.loss_orchestrator.compute_standard_loss(outputs, labels, stage=stage, mask=mask)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""

        images = batch["image"]
        labels = batch["label"]
        raw_mask = batch.get("mask", None)
        # Binarize mask: (B, 1, D, H, W) float, 1 = valid, 0 = ignore
        mask = (raw_mask > 0).float() if raw_mask is not None else None

        # Forward pass
        outputs = self(images)

        # Compute loss using the loss orchestrator
        total_loss, loss_dict = self._compute_loss(outputs, labels, stage="train", mask=mask)

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

        # Compute loss using the loss orchestrator
        total_loss, loss_dict = self._compute_loss(outputs, labels, stage="val", mask=mask)

        # Compute evaluation metrics if enabled
        evaluation_cfg = self._get_test_evaluation_config()
        evaluation_enabled = bool(self._cfg_value(evaluation_cfg, "enabled", False))
        metrics = self._cfg_value(evaluation_cfg, "metrics", None)

        if evaluation_enabled and metrics is not None:
            requested_head = resolve_configured_output_head(
                self.cfg,
                purpose="validation metric computation",
                allow_none=True,
            )
            main_output, resolved_head = select_output_tensor(
                outputs,
                requested_head=requested_head,
                primary_head=getattr(self.cfg.model, "primary_head", None),
                purpose="validation metric computation",
            )

            is_multi_task = self._has_multiple_supervised_loss_tasks()
            evaluation_defaults = getattr(self.cfg, "evaluation", None)
            prediction_threshold = self._cfg_float(
                evaluation_cfg,
                "prediction_threshold",
                self._cfg_float(evaluation_defaults, "prediction_threshold", 0.5),
            )

            if resolved_head is not None:
                target_slice = self._resolve_validation_target_slice(resolved_head)
                target_tensor = (
                    labels
                    if target_slice is None
                    else self._slice_tensor_channels(
                        labels,
                        target_slice,
                        context=f"validation target slice for head '{resolved_head}'",
                    )
                )
                preds, targets = self._prepare_metric_predictions(
                    main_output,
                    target_tensor,
                    prediction_threshold=prediction_threshold,
                )
            elif is_multi_task:
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
            logger.info(
                "Sliding-window CPU input mode enabled: keeping test image tensors on CPU "
                "and letting MONAI move window batches to the configured sw_device."
            )

    def on_test_epoch_end(self) -> None:
        """Log aggregated test metrics after all ranks finish their assigned volumes."""
        log_test_epoch_metrics(self._evaluation_context())

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

        def _preserve_cpu_value(value: Any) -> Any:
            if torch.is_tensor(value):
                return value
            if isinstance(value, list) and all(v is None or torch.is_tensor(v) for v in value):
                return list(value)
            if isinstance(value, tuple) and all(v is None or torch.is_tensor(v) for v in value):
                return list(value)
            return None

        cpu_image = None
        cpu_label = None
        cpu_mask = None
        if preserve_cpu_input:
            image = batch.get("image")
            label = batch.get("label")
            mask = batch.get("mask")
            cpu_image = _preserve_cpu_value(image)
            cpu_label = _preserve_cpu_value(label)
            cpu_mask = _preserve_cpu_value(mask)

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
        # Optimize the full Lightning module so trainable auxiliaries such as
        # adaptive loss weighters are included alongside the base network.
        optimizer = build_optimizer(self.cfg, self)

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
            logger.info(
                f"Scheduler interval: '{scheduler_interval}' (frequency: {scheduler_frequency})"
            )
            if scheduler_interval == "step":
                logger.info(f"Scheduler will step every {scheduler_frequency} training step(s)")
            else:
                logger.info(f"Scheduler will step every {scheduler_frequency} epoch(s)")

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
                    logger.info(f"ReduceLROnPlateau will monitor: {monitor_metric}")
                else:
                    # Default to validation loss
                    scheduler_config["monitor"] = "val_loss_total"
                    logger.warning(
                        "ReduceLROnPlateau will monitor: val_loss_total "
                        "(default, no monitor specified in config)"
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
