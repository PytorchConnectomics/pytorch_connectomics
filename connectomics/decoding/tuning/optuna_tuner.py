"""
Optuna-based hyperparameter tuning for decoding/post-processing parameters.

This module provides automated parameter optimization for instance segmentation
post-processing, particularly for watershed-based decoding with binary, contour,
and distance predictions.

Usage:
    from connectomics.decoding.tuning.optuna_tuner import OptunaDecodingTuner

    tuner = OptunaDecodingTuner(cfg, predictions, ground_truth)
    study = tuner.optimize()
    best_params = study.best_params
"""

from __future__ import annotations

import logging
import traceback
from contextlib import contextmanager
from copy import deepcopy
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from connectomics.metrics.metrics_seg import adapted_rand
from connectomics.data.process.affinity import (
    affinity_deepem_crop_enabled,
    compute_affinity_crop_pad,
    crop_spatial_by_pad,
    resolve_affinity_channel_groups_from_cfg,
)

from ..registry import get_decoder
from ..utils import remove_small_instances

logger = logging.getLogger(__name__)

__all__ = ["OptunaDecodingTuner", "run_tuning", "load_and_apply_best_params"]


def _maybe_crop_affinity_array(
    data: np.ndarray,
    *,
    reference_spatial_shape: tuple[int, ...],
    crop_pad: tuple[tuple[int, int], ...],
) -> np.ndarray:
    if not crop_pad:
        return data
    expected_cropped_shape = tuple(
        int(reference_spatial_shape[axis]) - crop_pad[axis][0] - crop_pad[axis][1]
        for axis in range(len(crop_pad))
    )
    data_spatial_shape = tuple(int(v) for v in data.shape[-len(crop_pad):])
    if data_spatial_shape == expected_cropped_shape:
        return data
    if data_spatial_shape != reference_spatial_shape:
        return data
    return crop_spatial_by_pad(data, crop_pad, item_name="tuning array")


@contextmanager
def _temporary_tuning_inference_overrides(*cfg_objects: Any):
    """Force the pre-Optuna inference pass to cache raw predictions only."""
    inference_cfgs = []
    seen_inference_cfgs: set[int] = set()
    for cfg_obj in cfg_objects:
        if cfg_obj is None:
            continue
        inference_cfg = getattr(cfg_obj, "inference", None)
        if inference_cfg is None or id(inference_cfg) in seen_inference_cfgs:
            continue
        inference_cfgs.append(inference_cfg)
        seen_inference_cfgs.add(id(inference_cfg))

    if not inference_cfgs:
        raise ValueError("Missing runtime cfg.inference configuration required for tuning")

    backups = []
    for inference_cfg in inference_cfgs:
        save_prediction_cfg = getattr(inference_cfg, "save_prediction", None)
        if save_prediction_cfg is None:
            raise ValueError("Missing inference.save_prediction configuration required for tuning")

        evaluation_cfg = getattr(inference_cfg, "evaluation", None)
        backups.append(
            {
                "inference_cfg": inference_cfg,
                "save_prediction_enabled": bool(getattr(save_prediction_cfg, "enabled", False)),
                "save_prediction_cache_suffix": getattr(
                    save_prediction_cfg, "cache_suffix", "_prediction.h5"
                ),
                "decoding": deepcopy(getattr(inference_cfg, "decoding", None)),
                "evaluation_cfg": evaluation_cfg,
                "evaluation_enabled": (
                    bool(getattr(evaluation_cfg, "enabled", False))
                    if evaluation_cfg is not None
                    else None
                ),
            }
        )

        save_prediction_cfg.enabled = True
        save_prediction_cfg.cache_suffix = "_tta_prediction.h5"
        inference_cfg.decoding = None
        if evaluation_cfg is not None:
            evaluation_cfg.enabled = False

    try:
        yield "_tta_prediction.h5"
    finally:
        for backup in backups:
            inference_cfg = backup["inference_cfg"]
            save_prediction_cfg = inference_cfg.save_prediction
            save_prediction_cfg.enabled = backup["save_prediction_enabled"]
            save_prediction_cfg.cache_suffix = backup["save_prediction_cache_suffix"]
            inference_cfg.decoding = backup["decoding"]

            evaluation_cfg = backup["evaluation_cfg"]
            if evaluation_cfg is not None and backup["evaluation_enabled"] is not None:
                evaluation_cfg.enabled = backup["evaluation_enabled"]


class OptunaDecodingTuner:
    """
    Optuna-based parameter tuner for decoding/post-processing.

    This class handles automated hyperparameter optimization for instance
    segmentation post-processing, supporting:
    - Binary + Contour + Distance watershed decoding
    - Post-processing (small instance removal)
    - Single and multi-objective optimization
    - Flexible parameter search spaces with tuple support

    Args:
        cfg: Hydra configuration with tune and tune.parameter_space sections
        predictions: Model predictions (C, D, H, W) or path to .h5 file
        ground_truth: Ground truth labels (D, H, W) or path to .h5 file
        mask: Optional foreground mask (D, H, W) or path to .h5 file

    Example:
        >>> tuner = OptunaDecodingTuner(cfg, predictions, ground_truth)
        >>> study = tuner.optimize()
        >>> print(f"Best adapted_rand: {study.best_value:.4f}")
        >>> print(f"Best params: {study.best_params}")
    """

    def __init__(
        self,
        cfg: DictConfig,
        predictions: Union[np.ndarray, List[np.ndarray], str, Path],
        ground_truth: Union[np.ndarray, List[np.ndarray], str, Path],
        mask: Optional[Union[np.ndarray, List[np.ndarray], str, Path]] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.cfg = cfg

        # Load data — supports lists of arrays for per-volume evaluation
        if isinstance(predictions, list):
            # Multi-volume mode: evaluate each volume independently and average
            self.predictions_list = predictions
            self.ground_truth_list = (
                ground_truth if isinstance(ground_truth, list) else [ground_truth]
            )
            self.mask_list = (
                mask
                if isinstance(mask, list)
                else ([mask] * len(self.predictions_list) if mask is not None else None)
            )
            self.multi_volume = True
            logger.info("Multi-volume mode: %d volumes", len(self.predictions_list))
        else:
            # Single-volume mode (backward compatible)
            loaded_pred = self._load_data(predictions, "predictions")
            loaded_gt = self._load_data(ground_truth, "ground_truth")
            loaded_mask = self._load_data(mask, "mask") if mask is not None else None
            self.predictions_list = [loaded_pred]
            self.ground_truth_list = [loaded_gt]
            self.mask_list = [loaded_mask] if loaded_mask is not None else None
            self.multi_volume = False

        # Validate data shapes
        self._validate_data()

        # Extract tune-only optimization config through local handles.
        self.tune_cfg = getattr(cfg, "tune", None)
        if self.tune_cfg is None:
            raise ValueError("Missing tune configuration required for Optuna tuning")

        self.param_space_cfg = getattr(self.tune_cfg, "parameter_space", None)
        if self.param_space_cfg is None:
            raise ValueError("Missing tune.parameter_space configuration for Optuna tuning")

        # Resolve decoder function from registry
        self.decoder_fn_name = getattr(
            self.param_space_cfg.decoding, "function_name",
            "decode_instance_binary_contour_distance",
        )
        self.decoder_fn = get_decoder(self.decoder_fn_name)

        # Initialize trial counter
        self.trial_count = 0

        # ABISS batch merge-threshold optimisation: when the decoder is
        # decode_abiss and ws_merge_threshold is a tunable parameter, we
        # can run the C++ binary once with *all* merge threshold candidates
        # and cache the segmentations, avoiding redundant watershed + RG
        # recomputation.
        self._abiss_batch_enabled = False
        self._abiss_all_merge_thresholds: list[float] = []

        if self.decoder_fn_name == "decode_abiss":
            mt_cfg = None
            if hasattr(self.param_space_cfg, "decoding") and self.param_space_cfg.decoding.parameters:
                mt_cfg = self.param_space_cfg.decoding.parameters.get("ws_merge_threshold", None)
            if mt_cfg is not None:
                lo, hi = mt_cfg["range"]
                step = mt_cfg.get("step", None)
                if step:
                    self._abiss_all_merge_thresholds = [
                        round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)
                    ]
                else:
                    self._abiss_all_merge_thresholds = [round(lo, 10), round(hi, 10)]
                self._abiss_batch_enabled = True
                logger.info(
                    "ABISS batch mode: will sweep %d merge thresholds per ABISS call: %s",
                    len(self._abiss_all_merge_thresholds),
                    self._abiss_all_merge_thresholds,
                )

    def _load_data(self, data: np.ndarray | str | Path, name: str) -> np.ndarray:
        """Load data from array or HDF5 file."""
        if isinstance(data, np.ndarray):
            return data

        # Load from file
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        with h5py.File(path, "r") as f:
            # Try common HDF5 dataset names
            for key in ["main", "data", "volume", "image", "label"]:
                if key in f:
                    return f[key][:]

            # Use first dataset
            first_key = list(f.keys())[0]
            warnings.warn(
                f"Using first dataset '{first_key}' from {path}. "
                f"Available keys: {list(f.keys())}"
            )
            return f[first_key][:]

    def _validate_data(self):
        """Validate data shapes and types for each volume in the list."""
        for i in range(len(self.predictions_list)):
            pred = self.predictions_list[i]
            gt = self.ground_truth_list[i]

            # Handle 2D predictions: (C, H, W) → (C, 1, H, W)
            if pred.ndim == 3:
                expanded_shape = pred.shape[:1] + (1,) + pred.shape[1:]
                logger.info(
                    "Volume %d: 2D data detected, expanding predictions: %s -> %s",
                    i, pred.shape, expanded_shape,
                )
                pred = pred[:, np.newaxis, :, :]
                self.predictions_list[i] = pred

            if pred.ndim != 4:
                raise ValueError(
                    f"Volume {i}: Predictions should be 4D (C, D, H, W), got shape {pred.shape}"
                )

            # Handle 2D ground truth: (H, W) → (1, H, W)
            if gt.ndim == 2:
                expanded_shape = (1,) + gt.shape
                logger.info(
                    "Volume %d: 2D ground truth detected, expanding: %s -> %s",
                    i, gt.shape, expanded_shape,
                )
                gt = gt[np.newaxis, :, :]
                self.ground_truth_list[i] = gt

            if gt.ndim != 3:
                raise ValueError(
                    f"Volume {i}: Ground truth should be 3D (D, H, W), got shape {gt.shape}"
                )

            # Check spatial dimensions match
            if pred.shape[1:] != gt.shape:
                raise ValueError(
                    f"Volume {i}: Spatial dimensions mismatch: "
                    f"predictions {pred.shape[1:]} vs ground_truth {gt.shape}"
                )

            # Handle mask if provided
            if self.mask_list is not None:
                mask = self.mask_list[i]
                if mask is not None:
                    if mask.ndim == 2:
                        logger.info(
                            "Volume %d: 2D mask detected, expanding: %s -> %s",
                            i, mask.shape, (1,) + mask.shape,
                        )
                        mask = mask[np.newaxis, :, :]
                        self.mask_list[i] = mask

                    if mask.shape != gt.shape:
                        raise ValueError(
                            f"Volume {i}: Mask shape {mask.shape} doesn't match "
                            f"ground truth shape {gt.shape}"
                        )

    def optimize(self) -> optuna.Study:
        """
        Run Optuna optimization.

        Returns:
            Optuna study object with optimization results
        """
        # Create sampler
        sampler = self._create_sampler()

        # Create pruner
        pruner = self._create_pruner()

        # Get optimization direction
        direction = self._get_optimization_direction()

        # Create storage directory if using SQLite
        storage = getattr(self.tune_cfg, "storage", None)
        if storage and storage.startswith("sqlite:///"):
            # Extract database file path from SQLite URL
            db_path = storage.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created optuna storage directory: %s", db_dir)

        # Create or load study
        study = optuna.create_study(
            study_name=self.tune_cfg.study_name,
            storage=storage,
            load_if_exists=self.tune_cfg.load_if_exists,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

        # Run optimization
        n_trials = self.tune_cfg.n_trials
        timeout = self.tune_cfg.timeout

        metric = self.tune_cfg.optimization['single_objective']['metric']
        logger.info(
            "Starting Optuna optimization: %s | Trials: %s | Metric: %s | Direction: %s",
            self.tune_cfg.study_name, n_trials, metric, direction,
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=getattr(self.tune_cfg.logging, "show_progress_bar", True),
        )

        # Print results
        self._print_results(study)

        # Save results
        self._save_results(study)

        return study

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from config."""
        sampler_cfg = self.tune_cfg.sampler
        sampler_name = sampler_cfg["name"]
        sampler_kwargs = getattr(sampler_cfg, "kwargs", {})

        # Convert OmegaConf to dict
        if isinstance(sampler_kwargs, DictConfig):
            sampler_kwargs = OmegaConf.to_container(sampler_kwargs, resolve=True)

        if sampler_name == "TPE":
            return TPESampler(**sampler_kwargs)
        elif sampler_name == "CmaEs":
            return CmaEsSampler(**sampler_kwargs)
        elif sampler_name == "Random":
            return RandomSampler(**sampler_kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner from config."""
        pruner_cfg = getattr(self.tune_cfg, "pruner", None)

        if pruner_cfg is None or not getattr(pruner_cfg, "enabled", False):
            return None

        pruner_name = getattr(pruner_cfg, "name", "Median")
        pruner_kwargs = getattr(pruner_cfg, "kwargs", {})

        # Convert OmegaConf to dict
        if isinstance(pruner_kwargs, DictConfig):
            pruner_kwargs = OmegaConf.to_container(pruner_kwargs, resolve=True)

        if pruner_name == "Median":
            return MedianPruner(**pruner_kwargs)
        elif pruner_name == "Hyperband":
            return HyperbandPruner(**pruner_kwargs)
        else:
            warnings.warn(f"Unknown pruner: {pruner_name}, using None")
            return None

    def _get_optimization_direction(self) -> str:
        """Get optimization direction from config."""
        opt_cfg = self.tune_cfg.optimization

        if opt_cfg["mode"] == "single":
            return opt_cfg["single_objective"]["direction"]
        else:
            raise NotImplementedError("Multi-objective optimization not yet implemented")

    # ------------------------------------------------------------------
    # ABISS batch merge-threshold sweep
    # ------------------------------------------------------------------

    def _abiss_batch_objective(
        self,
        trial: "optuna.Trial",
        decoding_params: Dict[str, Any],
        postproc_params: Optional[Dict[str, Any]],
    ) -> float:
        """Inner sweep of merge thresholds for a single ABISS call.

        Runs the C++ binary once with *all* merge threshold candidates
        (watershed + region-graph computed once), evaluates each resulting
        segmentation, and reports the best metric to Optuna.
        """
        metric_name = self.tune_cfg.optimization["single_objective"]["metric"]
        direction = self._get_optimization_direction()
        bad_value = float("inf") if direction == "minimize" else float("-inf")

        # Build batch decoding params: replace single merge threshold with list.
        batch_params: Dict[str, Any] = {}
        for k, v in decoding_params.items():
            batch_params[k] = dict(v) if isinstance(v, dict) else v
        batch_cli = dict(batch_params.get("cli_args", {}))
        batch_cli.pop("ws_merge_threshold", None)
        batch_cli["ws_merge_thresholds"] = self._abiss_all_merge_thresholds
        batch_params["cli_args"] = batch_cli

        # per merge-threshold → per-volume metrics
        mt_are: Dict[float, List[float]] = {mt: [] for mt in self._abiss_all_merge_thresholds}
        mt_prec: Dict[float, List[float]] = {mt: [] for mt in self._abiss_all_merge_thresholds}
        mt_rec: Dict[float, List[float]] = {mt: [] for mt in self._abiss_all_merge_thresholds}

        for vol_idx in range(len(self.predictions_list)):
            pred_vol = self.predictions_list[vol_idx]
            gt_vol = self.ground_truth_list[vol_idx]
            mask_vol = self.mask_list[vol_idx] if self.mask_list else None

            try:
                results = self.decoder_fn(pred_vol, **batch_params)
            except Exception:
                logger.error(
                    "Trial %d ABISS batch failed (vol %d):\n%s",
                    self.trial_count, vol_idx, traceback.format_exc(),
                )
                return bad_value

            if not isinstance(results, dict):
                logger.error("Trial %d: decoder did not return dict in batch mode", self.trial_count)
                return bad_value

            for mt_val in self._abiss_all_merge_thresholds:
                mt_key = round(mt_val, 10)
                seg = results.get(mt_key)
                if seg is None:
                    continue

                if postproc_params is not None:
                    try:
                        seg = remove_small_instances(seg, **postproc_params)
                    except Exception:
                        continue

                gt_m = gt_vol * mask_vol if mask_vol is not None else gt_vol
                seg_m = seg * mask_vol if mask_vol is not None else seg

                if metric_name == "adapted_rand":
                    are_v, prec_v, rec_v = adapted_rand(seg_m, gt_m, all_stats=True)
                    mt_are[mt_val].append(are_v)
                    mt_prec[mt_val].append(prec_v)
                    mt_rec[mt_val].append(rec_v)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")

        # Pick the merge threshold with the best averaged metric.
        best_mt = self._abiss_all_merge_thresholds[0]
        best_avg = bad_value
        for mt_val in self._abiss_all_merge_thresholds:
            if not mt_are[mt_val]:
                continue
            avg = float(np.mean(mt_are[mt_val]))
            if (direction == "minimize" and avg < best_avg) or (
                direction == "maximize" and avg > best_avg
            ):
                best_avg = avg
                best_mt = mt_val

        avg_prec = float(np.mean(mt_prec.get(best_mt, [0.0])))
        avg_rec = float(np.mean(mt_rec.get(best_mt, [0.0])))

        trial.set_user_attr("precision", avg_prec)
        trial.set_user_attr("recall", avg_rec)
        trial.set_user_attr("best_ws_merge_threshold", best_mt)
        trial.set_user_attr("per_vol_are", mt_are.get(best_mt, []))
        trial.set_user_attr("per_vol_precision", mt_prec.get(best_mt, []))
        trial.set_user_attr("per_vol_recall", mt_rec.get(best_mt, []))

        # Store per-threshold metrics for analysis.
        for mt_val in self._abiss_all_merge_thresholds:
            if mt_are[mt_val]:
                trial.set_user_attr(f"are_mt_{mt_val}", float(np.mean(mt_are[mt_val])))

        if getattr(self.tune_cfg.logging, "verbose", True):
            mt_summary = " | ".join(
                f"mt={mt:.2f}:{float(np.mean(mt_are[mt])):.4f}"
                for mt in self._abiss_all_merge_thresholds
                if mt_are[mt]
            )
            logger.info(
                "Trial %3d: best ARE=%.4f (mt=%.2f) Prec=%.4f Rec=%.4f | %s",
                self.trial_count, best_avg, best_mt, avg_prec, avg_rec, mt_summary,
            )

        return best_avg

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Evaluates each volume independently to avoid instance ID collisions
        from concatenating unrelated volumes, then averages the metric.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize (averaged over all volumes)
        """
        self.trial_count += 1

        # Sample parameters (ws_merge_threshold skipped when batch enabled)
        params = self._sample_parameters(trial)

        # Reconstruct decoding parameters from sampled values
        decoding_params = self._reconstruct_decoding_params(params)

        # Reconstruct post-processing parameters if enabled
        postproc_params = None
        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            postproc_params = self._reconstruct_postproc_params(params)

        # ABISS batch: sweep all merge thresholds in a single binary call.
        if self._abiss_batch_enabled:
            return self._abiss_batch_objective(trial, decoding_params, postproc_params)

        metric_name = self.tune_cfg.optimization["single_objective"]["metric"]
        metric_values = []
        precision_values = []
        recall_values = []

        # Evaluate each volume independently
        for vol_idx in range(len(self.predictions_list)):
            pred_vol = self.predictions_list[vol_idx]
            gt_vol = self.ground_truth_list[vol_idx]
            mask_vol = self.mask_list[vol_idx] if self.mask_list else None

            # Decode predictions for this volume
            try:
                segmentation = self.decoder_fn(
                    pred_vol, **decoding_params
                )
            except Exception as e:
                logger.error(
                    "Trial %d failed during decoding (vol %d): params=%s\n%s",
                    self.trial_count, vol_idx, decoding_params, traceback.format_exc(),
                )
                return (
                    float("-inf")
                    if self._get_optimization_direction() == "maximize"
                    else float("inf")
                )

            # Apply post-processing if enabled
            if postproc_params is not None:
                try:
                    segmentation = remove_small_instances(segmentation, **postproc_params)
                except Exception as e:
                    logger.error(
                        "Trial %d failed during post-processing (vol %d): params=%s\n%s",
                        self.trial_count, vol_idx, postproc_params, traceback.format_exc(),
                    )
                    return (
                        float("-inf")
                        if self._get_optimization_direction() == "maximize"
                        else float("inf")
                    )

            # Compute metric for this volume
            try:
                if mask_vol is not None:
                    gt_masked = gt_vol * mask_vol
                    seg_masked = segmentation * mask_vol
                else:
                    gt_masked = gt_vol
                    seg_masked = segmentation

                if metric_name == "adapted_rand":
                    are_val, prec_val, rec_val = adapted_rand(
                        seg_masked, gt_masked, all_stats=True
                    )
                    metric_values.append(are_val)
                    precision_values.append(prec_val)
                    recall_values.append(rec_val)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")

            except Exception as e:
                logger.error(
                    "Trial %d failed during metric computation (vol %d): "
                    "metric=%s, seg shape=%s, dtype=%s, n_labels=%d\n%s",
                    self.trial_count, vol_idx, metric_name,
                    segmentation.shape, segmentation.dtype,
                    len(np.unique(segmentation)), traceback.format_exc(),
                )
                return (
                    float("-inf")
                    if self._get_optimization_direction() == "maximize"
                    else float("inf")
                )

        # Average metrics across volumes
        avg_metric = float(np.mean(metric_values))
        avg_precision = float(np.mean(precision_values)) if precision_values else 0.0
        avg_recall = float(np.mean(recall_values)) if recall_values else 0.0

        # Log progress with precision and recall
        if getattr(self.tune_cfg.logging, "verbose", True):
            per_vol_are = " ".join(f"{v:.3f}" for v in metric_values)
            per_vol_prec = " ".join(f"{v:.3f}" for v in precision_values)
            per_vol_rec = " ".join(f"{v:.3f}" for v in recall_values)
            logger.info(
                "Trial %3d: ARE=%.4f Prec=%.4f Rec=%.4f "
                "(per-vol ARE: [%s] Prec: [%s] Rec: [%s])",
                self.trial_count, avg_metric, avg_precision, avg_recall,
                per_vol_are, per_vol_prec, per_vol_rec,
            )

        # Store precision/recall as user attributes for later analysis
        trial.set_user_attr("precision", avg_precision)
        trial.set_user_attr("recall", avg_recall)
        trial.set_user_attr("per_vol_are", metric_values)
        trial.set_user_attr("per_vol_precision", precision_values)
        trial.set_user_attr("per_vol_recall", recall_values)

        return avg_metric

    @staticmethod
    def _suggest_param(trial: optuna.Trial, name: str, cfg: Any) -> Any:
        """Suggest a single parameter from its config spec."""
        param_type = cfg["type"]
        if param_type == "float":
            return trial.suggest_float(
                name, cfg["range"][0], cfg["range"][1],
                step=cfg.get("step", None), log=cfg.get("log", False),
            )
        elif param_type == "int":
            return trial.suggest_int(
                name, cfg["range"][0], cfg["range"][1],
                step=cfg.get("step", 1), log=cfg.get("log", False),
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(name, cfg.choices)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}

        # Sample decoding parameters
        if hasattr(self.param_space_cfg, "decoding") and self.param_space_cfg.decoding.parameters:
            for name, cfg in self.param_space_cfg.decoding.parameters.items():
                # When ABISS batch mode is active, ws_merge_threshold is
                # swept internally (not sampled by Optuna).
                if self._abiss_batch_enabled and name == "ws_merge_threshold":
                    continue
                params[name] = self._suggest_param(trial, name, cfg)

        # Sample post-processing parameters
        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            for name, cfg in self.param_space_cfg.postprocessing.parameters.items():
                params[name] = self._suggest_param(trial, name, cfg)

        return params

    def _reconstruct_decoding_params(self, sampled_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct decoding function parameters from sampled values.

        Handles:
        - Tuple parameters via ``param_group`` / ``tuple_index`` fields.
        - Nested dict parameters via ``nest_under`` field (e.g. ``cli_args``
          for ``decode_abiss``).

        Args:
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Dictionary of parameters ready for decoding function
        """
        decoding_defaults = self.param_space_cfg.decoding.defaults
        # Deep-copy defaults so nested dicts (like cli_args) are independent.
        decoding_params: Dict[str, Any] = {}
        for k, v in decoding_defaults.items():
            decoding_params[k] = dict(v) if isinstance(v, dict) else v

        # Group tuple parameters
        tuple_params: Dict[str, Dict[int, Any]] = defaultdict(dict)
        scalar_params: Dict[str, Any] = {}
        nested_params: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Collect postprocessing parameter names to skip them
        postproc_param_names = set()
        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            postproc_param_names = set(self.param_space_cfg.postprocessing.parameters.keys())

        for param_name, value in sampled_params.items():
            # Skip post-processing parameters
            if param_name in postproc_param_names:
                continue

            # Check parameter config for grouping hints
            param_cfg = self.param_space_cfg.decoding.parameters.get(param_name, {})

            if "param_group" in param_cfg:
                # Part of a tuple parameter
                group_name = param_cfg["param_group"]
                tuple_index = param_cfg["tuple_index"]
                tuple_params[group_name][tuple_index] = value
            elif "nest_under" in param_cfg:
                # Nested under a dict key (e.g. cli_args for decode_abiss)
                nest_key = param_cfg["nest_under"]
                nested_params[nest_key][param_name] = value
            else:
                # Scalar parameter
                scalar_params[param_name] = value

        # Reconstruct tuples
        for group_name, indexed_values in tuple_params.items():
            sorted_items = sorted(indexed_values.items())
            tuple_values = tuple(val for idx, val in sorted_items)
            decoding_params[group_name] = tuple_values

        # Add scalar parameters
        decoding_params.update(scalar_params)

        # Merge nested parameters into their target dicts
        for nest_key, nested_vals in nested_params.items():
            if nest_key in decoding_params and isinstance(decoding_params[nest_key], dict):
                decoding_params[nest_key].update(nested_vals)
            else:
                decoding_params[nest_key] = nested_vals

        return decoding_params

    def _reconstruct_postproc_params(self, sampled_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct post-processing parameters from sampled values.

        Args:
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Dictionary of parameters for post-processing
        """
        if (
            not hasattr(self.param_space_cfg, "postprocessing")
            or not self.param_space_cfg.postprocessing.enabled
        ):
            return {}

        postproc_defaults = self.param_space_cfg.postprocessing.defaults
        postproc_params = dict(postproc_defaults)  # Start with defaults

        # Update with sampled parameters
        for param_name in self.param_space_cfg.postprocessing.parameters.keys():
            if param_name in sampled_params:
                postproc_params[param_name] = sampled_params[param_name]

        return postproc_params

    def _print_results(self, study: optuna.Study):
        """Log optimization results."""
        best_trial = study.best_trial
        best_prec = best_trial.user_attrs.get("precision", None)
        best_rec = best_trial.user_attrs.get("recall", None)

        lines = [
            f"OPTIMIZATION COMPLETE | {len(study.trials)} trials",
            f"Best trial: #{best_trial.number} | ARE: {study.best_value:.4f}",
        ]
        if best_prec is not None:
            lines.append(f"  Precision: {best_prec:.4f}")
        if best_rec is not None:
            lines.append(f"  Recall:    {best_rec:.4f}")

        per_vol_are = best_trial.user_attrs.get("per_vol_are", None)
        per_vol_prec = best_trial.user_attrs.get("per_vol_precision", None)
        per_vol_rec = best_trial.user_attrs.get("per_vol_recall", None)
        if per_vol_are:
            lines.append(f"  Per-volume ARE:  [{' '.join(f'{v:.3f}' for v in per_vol_are)}]")
        if per_vol_prec:
            lines.append(f"  Per-volume Prec: [{' '.join(f'{v:.3f}' for v in per_vol_prec)}]")
        if per_vol_rec:
            lines.append(f"  Per-volume Rec:  [{' '.join(f'{v:.3f}' for v in per_vol_rec)}]")

        best_decoding_params = self._reconstruct_decoding_params(study.best_params)

        # When ABISS batch mode was active, inject the best merge threshold
        # back into the decoding params for display / saving.
        best_ws_mt = best_trial.user_attrs.get("best_ws_merge_threshold", None)
        if best_ws_mt is not None:
            cli = best_decoding_params.get("cli_args", {})
            cli["ws_merge_threshold"] = best_ws_mt
            best_decoding_params["cli_args"] = cli

        lines.append("  Params:")
        for key, value in best_decoding_params.items():
            lines.append(f"    {key}: {value}")

        if getattr(self.param_space_cfg, "postprocessing", None) and getattr(
            self.param_space_cfg.postprocessing, "enabled", False
        ):
            best_postproc_params = self._reconstruct_postproc_params(study.best_params)
            if best_postproc_params:
                lines.append("  Post-processing params:")
                for key, value in best_postproc_params.items():
                    lines.append(f"    {key}: {value}")

        logger.info("\n".join(lines))

    def _save_results(self, study: optuna.Study):
        """Save optimization results to disk."""
        output_dir = Path(self.tune_cfg.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_file = output_dir / "best_params.yaml"
        best_decoding_params = self._reconstruct_decoding_params(study.best_params)
        best_postproc_params = self._reconstruct_postproc_params(study.best_params)

        # Inject best merge threshold from ABISS batch sweep.
        best_trial = study.best_trial
        best_ws_mt = best_trial.user_attrs.get("best_ws_merge_threshold", None)
        if best_ws_mt is not None:
            cli = best_decoding_params.get("cli_args", {})
            cli["ws_merge_threshold"] = best_ws_mt
            best_decoding_params["cli_args"] = cli

        # Create YAML content
        params_dict = {
            "best_trial": best_trial.number,
            "best_value": float(study.best_value),
            "best_precision": float(best_trial.user_attrs.get("precision", 0.0)),
            "best_recall": float(best_trial.user_attrs.get("recall", 0.0)),
            "metric": self.tune_cfg.optimization["single_objective"]["metric"],
            "decoding_function": self.decoder_fn_name,
            "decoding_params": best_decoding_params,
        }

        # Add per-volume metrics if available
        per_vol_are = best_trial.user_attrs.get("per_vol_are", None)
        per_vol_prec = best_trial.user_attrs.get("per_vol_precision", None)
        per_vol_rec = best_trial.user_attrs.get("per_vol_recall", None)
        if per_vol_are:
            params_dict["per_volume_are"] = [float(v) for v in per_vol_are]
        if per_vol_prec:
            params_dict["per_volume_precision"] = [float(v) for v in per_vol_prec]
        if per_vol_rec:
            params_dict["per_volume_recall"] = [float(v) for v in per_vol_rec]

        if best_postproc_params:
            params_dict["postprocessing_params"] = best_postproc_params

        # Save as YAML
        with open(best_params_file, "w") as f:
            OmegaConf.save(params_dict, f)

        logger.info("Best parameters saved to: %s", best_params_file)

        # Save study if requested
        if self.tune_cfg.output.save_study:
            if self.tune_cfg.storage:
                logger.info("Study saved to database: %s", self.tune_cfg.storage)
            else:
                warnings.warn("No storage configured, study not persisted to database")


# ============================================================================
# High-level API Functions
# ============================================================================


def run_tuning(model, trainer, cfg, checkpoint_path=None):
    """
    Run Optuna-based parameter tuning for instance segmentation decoding.

    This function performs automated hyperparameter optimization for post-processing
    parameters (thresholds, sizes, etc.) using Optuna on a validation/tuning dataset.

    Args:
        model: Lightning module (ConnectomicsModule)
        trainer: PyTorch Lightning Trainer
        cfg: Configuration object (Config dataclass)
        checkpoint_path: Optional path to model checkpoint

    Returns:
        None (results are saved to disk)

    Workflow:
        1. Check if best_params.yaml already exists (skip if it does)
        2. Run inference on tune dataset to get predictions
        3. Load ground truth labels for tuning
        4. Create OptunaDecodingTuner instance
        5. Run optimization study
        6. Save best parameters to YAML file

    Example:
        >>> from connectomics.training.lightning import ConnectomicsModule, create_trainer
        >>> from connectomics.decoding import run_tuning
        >>> model = ConnectomicsModule(cfg)
        >>> trainer = create_trainer(cfg)
        >>> run_tuning(model, trainer, cfg, checkpoint_path='best.ckpt')
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for parameter tuning. " "Install with: pip install optuna"
        )

    output_pred_dir = getattr(cfg.inference.save_prediction, "output_path", None)
    if not output_pred_dir:
        raise ValueError("Missing inference.save_prediction.output_path in configuration")
    output_dir = Path(output_pred_dir).parent / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Propagate the resolved output_dir into the tune config so that
    # OptunaDecodingTuner._save_results() can find it.
    cfg.tune.output.output_dir = str(output_dir)

    best_params_file = output_dir / "best_params.yaml"

    # Check if best parameters already exist
    if best_params_file.exists():
        logger.info(
            "SKIPPING PARAMETER TUNING: best parameters already exist at %s. "
            "Delete this file to re-run tuning.", best_params_file,
        )
        return

    logger.info("STARTING PARAMETER TUNING | Output directory: %s", output_dir)

    # Step 1: Run inference on tune dataset
    import glob

    from connectomics.data.io import read_volume
    from connectomics.training.lightning import create_datamodule

    logger.info("[1/4] Running inference on tuning dataset...")

    tune_data = cfg.data

    # Create datamodule with tune mode using merged runtime cfg.data/cfg.inference.
    datamodule = create_datamodule(cfg, mode="tune")

    logger.info("Using intermediate-only cache generation (decoding/evaluation disabled)")

    # Run test to populate/load raw prediction caches only. Optuna applies its own
    # decoding sweep afterward, so the tune inference pass must not decode with the
    # default config first.
    with _temporary_tuning_inference_overrides(cfg, getattr(model, "cfg", None)) as cache_suffix:
        model._tune_mode = True
        try:
            results = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        finally:
            model._tune_mode = False

    logger.info("Test completed. Results: %s", results)

    # Step 2: Load predictions from saved files
    logger.info("[2/4] Loading predictions from saved files...")
    output_pred_dir = cfg.inference.save_prediction.output_path
    predictions_dir = Path(output_pred_dir)

    # Find all prediction files using cache_suffix from config
    pred_pattern = f"*{cache_suffix}"
    pred_files = sorted(glob.glob(str(predictions_dir / pred_pattern)))

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found in: {predictions_dir}\n"
            f"Expected files matching pattern: {pred_pattern}"
        )

    logger.info("Found %d prediction file(s)", len(pred_files))

    # Load all prediction files as a list (per-volume evaluation)
    all_predictions = []
    for pred_file in pred_files:
        pred = read_volume(pred_file)
        logger.info("Loaded %s: shape %s", Path(pred_file).name, pred.shape)
        all_predictions.append(pred)

    total_slices = sum(p.shape[1] for p in all_predictions)
    logger.info("Loaded %d prediction volumes (%d total slices)", len(all_predictions), total_slices)

    # Step 3: Load ground truth
    logger.info("[3/4] Loading ground truth labels...")
    # Prefer data.val.label, fall back to data.test.label
    tune_label_pattern = getattr(getattr(tune_data, "val", None), "label", None)
    if tune_label_pattern is None:
        tune_label_pattern = getattr(getattr(tune_data, "test", None), "label", None)

    if tune_label_pattern is None:
        raise ValueError("Missing data.val.label or data.test.label in configuration")

    # Handle both string patterns and pre-resolved lists
    if isinstance(tune_label_pattern, list):
        # Already resolved to list of files
        label_files = sorted(tune_label_pattern)
    elif isinstance(tune_label_pattern, str):
        # Glob pattern - expand it
        label_files = sorted(glob.glob(tune_label_pattern))
    else:
        raise TypeError(f"tune.data.val.label must be string or list, got {type(tune_label_pattern)}")

    if not label_files:
        raise FileNotFoundError(f"No label files found matching pattern: {tune_label_pattern}")

    logger.info("Found %d label file(s)", len(label_files))

    # Load all label files as a list (per-volume evaluation)
    all_labels = []
    for label_file in label_files:
        label = read_volume(label_file)
        logger.info("Loaded %s: shape %s", Path(label_file).name, label.shape)
        all_labels.append(label)

    total_label_slices = sum(l.shape[0] for l in all_labels)
    logger.info("Loaded %d ground truth volumes (%d total slices)", len(all_labels), total_label_slices)

    # Load mask if available
    all_masks = None
    # Prefer data.val.mask, fall back to data.test.mask
    tune_mask_pattern = getattr(getattr(tune_data, "val", None), "mask", None)
    if tune_mask_pattern is None:
        tune_mask_pattern = getattr(getattr(tune_data, "test", None), "mask", None)
    if tune_mask_pattern:
        # Handle both string patterns and pre-resolved lists
        if isinstance(tune_mask_pattern, list):
            mask_files = sorted(tune_mask_pattern)
        elif isinstance(tune_mask_pattern, str):
            mask_files = sorted(glob.glob(tune_mask_pattern))
        else:
            raise TypeError(f"data.test.mask must be string or list, got {type(tune_mask_pattern)}")

        if not mask_files:
            logger.warning("No mask files found matching pattern: %s", tune_mask_pattern)
        else:
            logger.info("Found %d mask file(s)", len(mask_files))
            all_masks = []
            for mask_file in mask_files:
                m = read_volume(mask_file)
                logger.info("Loaded %s: shape %s", Path(mask_file).name, m.shape)
                all_masks.append(m)

            logger.info("Loaded %d mask volumes", len(all_masks))

    # Validate pred/label count match
    if len(all_predictions) != len(all_labels):
        raise ValueError(
            f"Mismatch: {len(all_predictions)} prediction files vs "
            f"{len(all_labels)} label files"
        )
    if all_masks is not None and len(all_masks) != len(all_predictions):
        raise ValueError(
            f"Mismatch: {len(all_predictions)} prediction files vs "
            f"{len(all_masks)} mask files"
        )

    if affinity_deepem_crop_enabled(cfg):
        groups = resolve_affinity_channel_groups_from_cfg(cfg)
        all_offsets = []
        for _, offsets in groups:
            all_offsets.extend(offsets)
        crop_pad = compute_affinity_crop_pad(all_offsets)
        if crop_pad and any(before or after for before, after in crop_pad):
            cropped_predictions = []
            cropped_labels = []
            cropped_masks = [] if all_masks is not None else None
            for idx, pred in enumerate(all_predictions):
                reference_spatial_shape = tuple(int(v) for v in all_labels[idx].shape[-len(crop_pad):])
                cropped_predictions.append(
                    _maybe_crop_affinity_array(
                        np.asarray(pred),
                        reference_spatial_shape=reference_spatial_shape,
                        crop_pad=crop_pad,
                    )
                )
                cropped_labels.append(
                    _maybe_crop_affinity_array(
                        np.asarray(all_labels[idx]),
                        reference_spatial_shape=reference_spatial_shape,
                        crop_pad=crop_pad,
                    )
                )
                if cropped_masks is not None:
                    cropped_masks.append(
                        _maybe_crop_affinity_array(
                            np.asarray(all_masks[idx]),
                            reference_spatial_shape=reference_spatial_shape,
                            crop_pad=crop_pad,
                        )
                    )
            all_predictions = cropped_predictions
            all_labels = cropped_labels
            all_masks = cropped_masks

    # Step 4: Create tuner and run optimization (per-volume evaluation)
    logger.info("[4/5] Creating Optuna tuner...")
    tuner = OptunaDecodingTuner(
        cfg=cfg, predictions=all_predictions, ground_truth=all_labels, mask=all_masks
    )

    logger.info("[5/5] Running optimization study...")
    study = tuner.optimize()

    logger.info(
        "TUNING COMPLETED | Best parameters saved to: %s | "
        "Best value: %.4f | Best params: %s",
        best_params_file, study.best_value, study.best_params,
    )


def load_and_apply_best_params(cfg):
    """
    Load best parameters from Optuna tuning and apply them to test config.

    This function loads the best_params.yaml file generated by run_tuning()
    and updates the test.decoding section of the config with optimized parameters.

    Args:
        cfg: Configuration object (Config dataclass)

    Returns:
        cfg: Updated configuration object with best parameters applied

    Example:
        >>> cfg = load_config('tutorials/misc/hydra-lv.yaml')
        >>> cfg = load_and_apply_best_params(cfg)
        >>> # cfg.inference now has optimized decoding parameters
    """
    output_pred_dir = getattr(cfg.inference.save_prediction, "output_path", None)
    if not output_pred_dir:
        raise ValueError("Missing inference.save_prediction.output_path in configuration")
    output_dir = Path(output_pred_dir).parent / "tuning"
    best_params_file = output_dir / "best_params.yaml"

    if not best_params_file.exists():
        raise FileNotFoundError(
            f"Best parameters file not found: {best_params_file}\n"
            f"Run parameter tuning first with --mode tune"
        )

    logger.info("Loading best parameters from: %s", best_params_file)

    # Load best parameters
    best_params = OmegaConf.load(best_params_file)

    logger.info("Loaded best parameters:\n%s", OmegaConf.to_yaml(best_params))

    # Apply to merged runtime inference.decoding config
    if getattr(cfg.inference, "decoding", None) is None:
        cfg.inference.decoding = []

    # Find the decoding function in test.decoding that matches the tuned function
    decoding_function = best_params.get("decoding_function", None)

    if decoding_function is None:
        warnings.warn("No decoding_function found in best_params, applying to first decoder")
        decoder_idx = 0
    else:
        # Find decoder with matching function name
        decoder_idx = None
        for idx, decoder in enumerate(cfg.inference.decoding):
            decoder_name = (
                decoder.get("name") if isinstance(decoder, dict) else getattr(decoder, "name", None)
            )
            if decoder_name == decoding_function:
                decoder_idx = idx
                break

        if decoder_idx is None:
            # Create new decoder entry
            decoder_idx = len(cfg.inference.decoding)
            cfg.inference.decoding.append({"name": decoding_function, "kwargs": {}})

    # Update parameters
    if decoder_idx < len(cfg.inference.decoding):
        decoder = cfg.inference.decoding[decoder_idx]

        # Handle both dict and config object
        if isinstance(decoder, dict):
            if "kwargs" not in decoder:
                decoder["kwargs"] = {}
            decoder["kwargs"].update(OmegaConf.to_container(best_params["decoding_params"]))
        else:
            if not hasattr(decoder, "kwargs") or decoder.kwargs is None:
                decoder.kwargs = {}
            # Update kwargs with best parameters
            for key, value in best_params["decoding_params"].items():
                decoder.kwargs[key] = value

        logger.info("Applied best parameters to inference.decoding[%d]", decoder_idx)

    return cfg
