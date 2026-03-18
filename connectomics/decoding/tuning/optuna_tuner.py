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
import multiprocessing as mp
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
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
from connectomics.training.lightning.utils import (
    tta_cache_suffix,
    tuning_best_params_filename,
    tuning_best_params_filename_candidates,
    tuning_study_db_filename,
)

from ..registry import get_decoder
from ..utils import remove_small_instances

logger = logging.getLogger(__name__)

__all__ = [
    "OptunaDecodingTuner",
    "TrialEvaluationTimeoutError",
    "run_tuning",
    "load_and_apply_best_params",
]


class TrialEvaluationTimeoutError(TimeoutError):
    """Raised when a single tuning trial exceeds the configured wall-clock limit."""


def _bad_objective_value(direction: str) -> float:
    """Return the worst possible objective value for the given direction."""
    return float("inf") if direction == "minimize" else float("-inf")


def _expand_tuning_paths(path_or_pattern: Any, *, field_name: str) -> list[str]:
    """Expand string/list path inputs used by the tuning loader."""
    import glob

    if path_or_pattern is None:
        return []

    if isinstance(path_or_pattern, (str, Path)):
        pattern = str(path_or_pattern)
        if "*" in pattern or "?" in pattern:
            return sorted(glob.glob(pattern))
        return [pattern]

    if isinstance(path_or_pattern, list):
        expanded: list[str] = []
        for entry in path_or_pattern:
            expanded.extend(_expand_tuning_paths(entry, field_name=field_name))
        return expanded

    raise TypeError(f"{field_name} must be string or list, got {type(path_or_pattern)}")


def _resolve_tuning_prediction_files(
    cfg,
    predictions_dir: Path,
    cache_suffix: str,
) -> tuple[list[str], list[str]]:
    """Resolve cached prediction files for the current tune dataset only."""
    tune_image_pattern = getattr(getattr(cfg.data, "val", None), "image", None)
    if tune_image_pattern is None:
        raise ValueError("Missing data.val.image in configuration")

    image_files = _expand_tuning_paths(tune_image_pattern, field_name="data.val.image")
    if not image_files:
        raise FileNotFoundError(f"No image files found matching pattern: {tune_image_pattern}")

    expected_files = [
        predictions_dir / f"{Path(str(path)).stem}{cache_suffix}" for path in image_files
    ]
    existing_files = [str(path) for path in expected_files if path.exists()]
    return existing_files, [str(path) for path in expected_files]


def _print_best_params_yaml(best_params_file: Path) -> None:
    """Print the current best-params YAML to stdout for interactive tune runs."""
    try:
        best_params = OmegaConf.load(best_params_file)
        yaml_text = OmegaConf.to_yaml(best_params).rstrip()
    except Exception:
        yaml_text = best_params_file.read_text().rstrip()

    print("\n" + "=" * 80)
    print(f"BEST PARAMETERS | {best_params_file}")
    print("=" * 80)
    if yaml_text:
        print(yaml_text)
    else:
        print("[empty]")


def _resolve_best_params_file(
    cfg: Any,
    output_dir: Path,
    *,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Return the primary best-params path for the current tuning context."""
    return output_dir / tuning_best_params_filename(cfg, checkpoint_path=checkpoint_path)


def _resolve_existing_best_params_file(
    cfg: Any,
    output_dir: Path,
    *,
    checkpoint_path: str | Path | None = None,
) -> Path | None:
    """Return the first matching best-params file, preferring the current exact name."""
    for candidate in tuning_best_params_filename_candidates(cfg, checkpoint_path=checkpoint_path):
        candidate_path = output_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _compute_segmentation_metric(
    segmentation: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray | None,
    metric_name: str,
) -> tuple[float, float, float]:
    """Compute a segmentation metric triplet for one volume."""
    gt_masked = ground_truth * mask if mask is not None else ground_truth
    seg_masked = segmentation * mask if mask is not None else segmentation

    if metric_name == "adapted_rand":
        are_val, prec_val, rec_val = adapted_rand(seg_masked, gt_masked, all_stats=True)
        return float(are_val), float(prec_val), float(rec_val)

    raise ValueError(f"Unknown metric: {metric_name}")


def _evaluate_standard_trial_payload(
    *,
    decoder_fn,
    predictions_list: list[np.ndarray],
    ground_truth_list: list[np.ndarray],
    mask_list: list[np.ndarray] | None,
    decoding_params: Dict[str, Any],
    postproc_params: Optional[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, Any]:
    """Run one standard decoding trial and return aggregate metrics."""
    metric_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    for vol_idx, pred_vol in enumerate(predictions_list):
        gt_vol = ground_truth_list[vol_idx]
        mask_vol = mask_list[vol_idx] if mask_list else None

        try:
            segmentation = decoder_fn(pred_vol, **decoding_params)
        except Exception as exc:
            raise RuntimeError(
                f"Decoding failed for volume {vol_idx} with params={decoding_params!r}"
            ) from exc

        if postproc_params is not None:
            try:
                segmentation = remove_small_instances(segmentation, **postproc_params)
            except Exception as exc:
                raise RuntimeError(
                    f"Post-processing failed for volume {vol_idx} "
                    f"with params={postproc_params!r}"
                ) from exc

        try:
            are_val, prec_val, rec_val = _compute_segmentation_metric(
                segmentation,
                gt_vol,
                mask_vol,
                metric_name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Metric computation failed for volume {vol_idx} "
                f"(metric={metric_name}, seg_shape={segmentation.shape}, "
                f"dtype={segmentation.dtype})"
            ) from exc

        metric_values.append(are_val)
        precision_values.append(prec_val)
        recall_values.append(rec_val)

    return {
        "avg_metric": float(np.mean(metric_values)),
        "avg_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "avg_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "per_vol_are": metric_values,
        "per_vol_precision": precision_values,
        "per_vol_recall": recall_values,
    }


def _evaluate_batch_trial_payload(
    *,
    decoder_fn,
    predictions_list: list[np.ndarray],
    ground_truth_list: list[np.ndarray],
    mask_list: list[np.ndarray] | None,
    batch_params: Dict[str, Any],
    postproc_params: Optional[Dict[str, Any]],
    metric_name: str,
    direction: str,
    candidate_values: list[float],
) -> Dict[str, Any]:
    """Run one batch-decoder trial and return aggregate metrics."""
    candidate_are: Dict[float, List[float]] = {val: [] for val in candidate_values}
    candidate_prec: Dict[float, List[float]] = {val: [] for val in candidate_values}
    candidate_rec: Dict[float, List[float]] = {val: [] for val in candidate_values}

    for vol_idx, pred_vol in enumerate(predictions_list):
        gt_vol = ground_truth_list[vol_idx]
        mask_vol = mask_list[vol_idx] if mask_list else None

        try:
            results = decoder_fn(pred_vol, **batch_params)
        except Exception as exc:
            raise RuntimeError(
                f"Batch decoding failed for volume {vol_idx} with params={batch_params!r}"
            ) from exc

        if not isinstance(results, dict):
            raise TypeError("Decoder did not return a dict in batch mode")

        for candidate in candidate_values:
            seg = results.get(round(candidate, 10))
            if seg is None:
                continue

            if postproc_params is not None:
                try:
                    seg = remove_small_instances(seg, **postproc_params)
                except Exception:
                    continue

            are_val, prec_val, rec_val = _compute_segmentation_metric(
                seg,
                gt_vol,
                mask_vol,
                metric_name,
            )
            candidate_are[candidate].append(are_val)
            candidate_prec[candidate].append(prec_val)
            candidate_rec[candidate].append(rec_val)

    best_candidate = candidate_values[0]
    best_metric = _bad_objective_value(direction)
    for candidate in candidate_values:
        if not candidate_are[candidate]:
            continue
        avg_metric = float(np.mean(candidate_are[candidate]))
        if (direction == "minimize" and avg_metric < best_metric) or (
            direction == "maximize" and avg_metric > best_metric
        ):
            best_metric = avg_metric
            best_candidate = candidate

    return {
        "best_metric": best_metric,
        "best_candidate": best_candidate,
        "avg_precision": float(np.mean(candidate_prec.get(best_candidate, [0.0]))),
        "avg_recall": float(np.mean(candidate_rec.get(best_candidate, [0.0]))),
        "per_vol_are": list(candidate_are.get(best_candidate, [])),
        "per_vol_precision": list(candidate_prec.get(best_candidate, [])),
        "per_vol_recall": list(candidate_rec.get(best_candidate, [])),
        "per_candidate_metric": {
            candidate: float(np.mean(values))
            for candidate, values in candidate_are.items()
            if values
        },
    }


def _execute_trial_payload(evaluation_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one tuning trial payload in-process."""
    if evaluation_kind == "standard":
        return _evaluate_standard_trial_payload(**payload)
    if evaluation_kind in {"abiss_batch", "waterz_batch"}:
        return _evaluate_batch_trial_payload(**payload)
    raise ValueError(f"Unknown evaluation kind: {evaluation_kind}")


def _trial_evaluation_worker(send_conn, evaluation_kind: str, payload: Dict[str, Any]) -> None:
    """Execute a trial payload in a child process and send back a small summary dict."""
    try:
        send_conn.send({"ok": True, "result": _execute_trial_payload(evaluation_kind, payload)})
    except Exception:
        try:
            send_conn.send({"ok": False, "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        send_conn.close()


def _get_trial_process_context() -> mp.context.BaseContext:
    """Prefer fork to avoid copying large prediction arrays when available."""
    for method in ("fork", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()


def _run_trial_payload_with_timeout(
    evaluation_kind: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: float,
) -> Dict[str, Any]:
    """Execute a trial payload in a child process and abort it on timeout."""
    ctx = _get_trial_process_context()
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_trial_evaluation_worker,
        args=(send_conn, evaluation_kind, payload),
    )
    process.start()
    send_conn.close()

    try:
        process.join(timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join(2.0)
            if process.is_alive():
                process.kill()
                process.join()
            raise TrialEvaluationTimeoutError(
                f"{evaluation_kind} evaluation exceeded timeout of {timeout_sec:g}s"
            )

        if not recv_conn.poll(0.1):
            raise RuntimeError(
                f"Trial worker exited without returning a result (exitcode={process.exitcode})"
            )

        message = recv_conn.recv()
    finally:
        recv_conn.close()

    if not isinstance(message, dict):
        raise RuntimeError("Trial worker returned an invalid response")
    if not message.get("ok", False):
        raise RuntimeError(message.get("traceback", "Trial worker failed without traceback"))
    return message["result"]


@contextmanager
def _temporary_tuning_inference_overrides(*cfg_objects: Any, checkpoint_path: str | None = None):
    """Force the pre-Optuna inference pass to cache raw predictions only."""
    inference_cfgs = []
    seen_inference_cfgs: set[int] = set()
    primary_cfg = None
    for cfg_obj in cfg_objects:
        if cfg_obj is None:
            continue
        if primary_cfg is None:
            primary_cfg = cfg_obj
        inference_cfg = getattr(cfg_obj, "inference", None)
        if inference_cfg is None or id(inference_cfg) in seen_inference_cfgs:
            continue
        inference_cfgs.append(inference_cfg)
        seen_inference_cfgs.add(id(inference_cfg))

    if not inference_cfgs:
        raise ValueError("Missing runtime cfg.inference configuration required for tuning")

    suffix = (
        tta_cache_suffix(primary_cfg, checkpoint_path=checkpoint_path)
        if primary_cfg is not None
        else "_tta_x1_prediction.h5"
    )

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
                    save_prediction_cfg, "cache_suffix", "_x1_prediction.h5"
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
        save_prediction_cfg.cache_suffix = suffix
        inference_cfg.decoding = None
        if evaluation_cfg is not None:
            evaluation_cfg.enabled = False

    try:
        yield suffix
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
            # Single-volume mode
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
            self.param_space_cfg.decoding,
            "function_name",
            "decode_instance_binary_contour_distance",
        )
        self.decoder_fn = get_decoder(self.decoder_fn_name)

        # Initialize trial counter
        self.trial_count = 0

        # Batch threshold optimisation: when the decoder supports multi-
        # threshold evaluation in a single call (watershed + region-graph
        # computed once), we enumerate all threshold candidates and sweep
        # them in one decoder invocation per trial.
        #
        # Supported decoders:
        #   - decode_abiss  (param: ws_merge_threshold → cli_args batch)
        #   - decode_waterz (param: thresholds → return_all_thresholds)
        self._abiss_batch_enabled = False
        self._abiss_all_merge_thresholds: list[float] = []
        self._waterz_batch_enabled = False
        self._waterz_all_thresholds: list[float] = []

        if self.decoder_fn_name == "decode_abiss":
            mt_cfg = None
            if (
                hasattr(self.param_space_cfg, "decoding")
                and self.param_space_cfg.decoding.parameters
            ):
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

        if self.decoder_fn_name == "decode_waterz":
            thr_cfg = None
            if (
                hasattr(self.param_space_cfg, "decoding")
                and self.param_space_cfg.decoding.parameters
            ):
                thr_cfg = self.param_space_cfg.decoding.parameters.get("thresholds", None)
            if thr_cfg is not None:
                lo, hi = thr_cfg["range"]
                step = thr_cfg.get("step", None)
                if step:
                    self._waterz_all_thresholds = [
                        round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)
                    ]
                else:
                    self._waterz_all_thresholds = [round(lo, 10), round(hi, 10)]
                self._waterz_batch_enabled = True
                logger.info(
                    "Waterz batch mode: will sweep %d thresholds per waterz call: %s",
                    len(self._waterz_all_thresholds),
                    self._waterz_all_thresholds,
                )

    def _get_trial_timeout_seconds(self) -> float | None:
        """Return the configured per-trial timeout in seconds, if enabled."""
        timeout = getattr(self.tune_cfg, "trial_timeout", None)
        if timeout is None:
            return None

        timeout_value = float(timeout)
        if timeout_value <= 0:
            return None
        return timeout_value

    def _execute_evaluation(self, evaluation_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run trial evaluation inline or in a timeout-capable child process."""
        timeout_sec = self._get_trial_timeout_seconds()
        if timeout_sec is None:
            return _execute_trial_payload(evaluation_kind, payload)
        return _run_trial_payload_with_timeout(
            evaluation_kind,
            payload,
            timeout_sec=timeout_sec,
        )

    def _handle_timed_out_trial(
        self,
        trial: "optuna.Trial",
        *,
        evaluation_kind: str,
        params: Dict[str, Any],
        direction: str,
    ) -> float:
        """Record timeout metadata and return the worst objective value."""
        timeout_sec = self._get_trial_timeout_seconds()
        trial.set_user_attr("timed_out", True)
        trial.set_user_attr("timeout_stage", evaluation_kind)
        if timeout_sec is not None:
            trial.set_user_attr("trial_timeout", timeout_sec)

        logger.warning(
            "Trial %d timed out after %.1fs during %s evaluation: params=%s",
            self.trial_count,
            timeout_sec or 0.0,
            evaluation_kind,
            params,
        )
        return _bad_objective_value(direction)

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
                    i,
                    pred.shape,
                    expanded_shape,
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
                    i,
                    gt.shape,
                    expanded_shape,
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
                            i,
                            mask.shape,
                            (1,) + mask.shape,
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

        # Resolve storage: auto-generate SQLite path when save_study=True
        storage = getattr(self.tune_cfg, "storage", None)
        if not storage and getattr(self.tune_cfg.output, "save_study", False):
            output_dir = getattr(self.tune_cfg.output, "output_dir", None)
            if output_dir:
                db_path = Path(output_dir) / tuning_study_db_filename(
                    self.cfg,
                    self.tune_cfg.study_name,
                    checkpoint_path=getattr(self, "_prediction_checkpoint_path", None),
                )
                storage = f"sqlite:///{db_path}"
                logger.info("Auto-generated study storage: %s", storage)
        if storage and storage.startswith("sqlite:///"):
            db_path = storage.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_storage = storage

        # Create or load study
        study = optuna.create_study(
            study_name=self.tune_cfg.study_name,
            storage=storage,
            load_if_exists=self.tune_cfg.load_if_exists,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

        # Seed the first trial with known-good defaults so TPE has a strong
        # baseline from the start instead of wasting early trials on random configs.
        default_params = self._build_default_trial_params()
        if default_params:
            study.enqueue_trial(default_params)
            logger.info("Seeded first trial with default parameters: %s", default_params)

        # Run optimization
        n_trials = self.tune_cfg.n_trials
        timeout = self.tune_cfg.timeout

        metric = self.tune_cfg.optimization["single_objective"]["metric"]
        trial_timeout = self._get_trial_timeout_seconds()
        logger.info(
            "Starting Optuna optimization: %s | Trials: %s | Metric: %s | "
            "Direction: %s | Trial timeout: %s",
            self.tune_cfg.study_name,
            n_trials,
            metric,
            direction,
            f"{trial_timeout:g}s" if trial_timeout is not None else "disabled",
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
        bad_value = _bad_objective_value(direction)

        # Build batch decoding params: replace single merge threshold with list.
        batch_params: Dict[str, Any] = {}
        for k, v in decoding_params.items():
            batch_params[k] = dict(v) if isinstance(v, dict) else v
        batch_cli = dict(batch_params.get("cli_args", {}))
        batch_cli.pop("ws_merge_threshold", None)
        batch_cli["ws_merge_thresholds"] = self._abiss_all_merge_thresholds
        batch_params["cli_args"] = batch_cli

        payload = {
            "decoder_fn": self.decoder_fn,
            "predictions_list": self.predictions_list,
            "ground_truth_list": self.ground_truth_list,
            "mask_list": self.mask_list,
            "batch_params": batch_params,
            "postproc_params": postproc_params,
            "metric_name": metric_name,
            "direction": direction,
            "candidate_values": self._abiss_all_merge_thresholds,
        }
        try:
            result = self._execute_evaluation("abiss_batch", payload)
        except TrialEvaluationTimeoutError:
            return self._handle_timed_out_trial(
                trial,
                evaluation_kind="abiss_batch",
                params=batch_params,
                direction=direction,
            )
        except Exception:
            logger.error(
                "Trial %d ABISS batch failed: params=%s\n%s",
                self.trial_count,
                batch_params,
                traceback.format_exc(),
            )
            return bad_value

        best_mt = float(result["best_candidate"])
        best_avg = float(result["best_metric"])
        avg_prec = float(result["avg_precision"])
        avg_rec = float(result["avg_recall"])

        trial.set_user_attr("precision", avg_prec)
        trial.set_user_attr("recall", avg_rec)
        trial.set_user_attr("best_ws_merge_threshold", best_mt)
        trial.set_user_attr("per_vol_are", result["per_vol_are"])
        trial.set_user_attr("per_vol_precision", result["per_vol_precision"])
        trial.set_user_attr("per_vol_recall", result["per_vol_recall"])

        # Store per-threshold metrics for analysis.
        for mt_val, avg_metric in result["per_candidate_metric"].items():
            trial.set_user_attr(f"are_mt_{mt_val}", float(avg_metric))

        if getattr(self.tune_cfg.logging, "verbose", True):
            mt_summary = " | ".join(
                f"mt={mt:.2f}:{float(avg_metric):.4f}"
                for mt, avg_metric in result["per_candidate_metric"].items()
            )
            print(
                f"  Trial {self.trial_count:3d}: best ARE={best_avg:.4f} (mt={best_mt:.2f}) "
                f"Prec={avg_prec:.4f} Rec={avg_rec:.4f} | {mt_summary}"
            )

        return best_avg

    # ------------------------------------------------------------------
    # Waterz batch threshold sweep
    # ------------------------------------------------------------------

    def _waterz_batch_objective(
        self,
        trial: "optuna.Trial",
        decoding_params: Dict[str, Any],
        postproc_params: Optional[Dict[str, Any]],
    ) -> float:
        """Sweep all threshold candidates in a single waterz call.

        Waterz performs watershed + region-graph extraction once, then
        incrementally merges for each threshold — so N thresholds are
        nearly as fast as one.
        """
        metric_name = self.tune_cfg.optimization["single_objective"]["metric"]
        direction = self._get_optimization_direction()
        bad_value = _bad_objective_value(direction)

        # Replace single threshold with full list + return_all_thresholds.
        batch_params = dict(decoding_params)
        batch_params["thresholds"] = self._waterz_all_thresholds
        batch_params["return_all_thresholds"] = True

        payload = {
            "decoder_fn": self.decoder_fn,
            "predictions_list": self.predictions_list,
            "ground_truth_list": self.ground_truth_list,
            "mask_list": self.mask_list,
            "batch_params": batch_params,
            "postproc_params": postproc_params,
            "metric_name": metric_name,
            "direction": direction,
            "candidate_values": self._waterz_all_thresholds,
        }
        try:
            result = self._execute_evaluation("waterz_batch", payload)
        except TrialEvaluationTimeoutError:
            return self._handle_timed_out_trial(
                trial,
                evaluation_kind="waterz_batch",
                params=batch_params,
                direction=direction,
            )
        except Exception:
            logger.error(
                "Trial %d waterz batch failed: params=%s\n%s",
                self.trial_count,
                batch_params,
                traceback.format_exc(),
            )
            return bad_value

        best_thr = float(result["best_candidate"])
        best_avg = float(result["best_metric"])
        avg_prec = float(result["avg_precision"])
        avg_rec = float(result["avg_recall"])

        trial.set_user_attr("precision", avg_prec)
        trial.set_user_attr("recall", avg_rec)
        trial.set_user_attr("best_threshold", best_thr)
        trial.set_user_attr("per_vol_are", result["per_vol_are"])
        trial.set_user_attr("per_vol_precision", result["per_vol_precision"])
        trial.set_user_attr("per_vol_recall", result["per_vol_recall"])

        # Store per-threshold metrics for analysis.
        for thr_val, avg_metric in result["per_candidate_metric"].items():
            trial.set_user_attr(f"are_thr_{thr_val}", float(avg_metric))

        if getattr(self.tune_cfg.logging, "verbose", True):
            thr_summary = " | ".join(
                f"t={thr:.2f}:{float(avg_metric):.4f}"
                for thr, avg_metric in result["per_candidate_metric"].items()
            )
            print(
                f"  Trial {self.trial_count:3d}: best ARE={best_avg:.4f} (thr={best_thr:.2f}) "
                f"Prec={avg_prec:.4f} Rec={avg_rec:.4f} | {thr_summary}"
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

        # Waterz batch: sweep all thresholds in a single waterz call
        # (watershed + region-graph computed once, incremental merging).
        if self._waterz_batch_enabled:
            return self._waterz_batch_objective(trial, decoding_params, postproc_params)

        metric_name = self.tune_cfg.optimization["single_objective"]["metric"]
        direction = self._get_optimization_direction()
        bad_value = _bad_objective_value(direction)

        payload = {
            "decoder_fn": self.decoder_fn,
            "predictions_list": self.predictions_list,
            "ground_truth_list": self.ground_truth_list,
            "mask_list": self.mask_list,
            "decoding_params": decoding_params,
            "postproc_params": postproc_params,
            "metric_name": metric_name,
        }
        try:
            result = self._execute_evaluation("standard", payload)
        except TrialEvaluationTimeoutError:
            return self._handle_timed_out_trial(
                trial,
                evaluation_kind="standard",
                params=decoding_params,
                direction=direction,
            )
        except Exception:
            logger.error(
                "Trial %d failed during evaluation: params=%s postproc=%s\n%s",
                self.trial_count,
                decoding_params,
                postproc_params,
                traceback.format_exc(),
            )
            return bad_value

        avg_metric = float(result["avg_metric"])
        avg_precision = float(result["avg_precision"])
        avg_recall = float(result["avg_recall"])

        # Log progress with precision and recall
        if getattr(self.tune_cfg.logging, "verbose", True):
            per_vol_are = " ".join(f"{v:.3f}" for v in result["per_vol_are"])
            per_vol_prec = " ".join(f"{v:.3f}" for v in result["per_vol_precision"])
            per_vol_rec = " ".join(f"{v:.3f}" for v in result["per_vol_recall"])
            logger.info(
                "Trial %3d: ARE=%.4f Prec=%.4f Rec=%.4f "
                "(per-vol ARE: [%s] Prec: [%s] Rec: [%s])",
                self.trial_count,
                avg_metric,
                avg_precision,
                avg_recall,
                per_vol_are,
                per_vol_prec,
                per_vol_rec,
            )

        # Store precision/recall as user attributes for later analysis
        trial.set_user_attr("precision", avg_precision)
        trial.set_user_attr("recall", avg_recall)
        trial.set_user_attr("per_vol_are", result["per_vol_are"])
        trial.set_user_attr("per_vol_precision", result["per_vol_precision"])
        trial.set_user_attr("per_vol_recall", result["per_vol_recall"])

        return avg_metric

    @staticmethod
    def _suggest_param(trial: optuna.Trial, name: str, cfg: Any) -> Any:
        """Suggest a single parameter from its config spec."""
        param_type = cfg["type"]
        if param_type == "float":
            return trial.suggest_float(
                name,
                cfg["range"][0],
                cfg["range"][1],
                step=cfg.get("step", None),
                log=cfg.get("log", False),
            )
        elif param_type == "int":
            return trial.suggest_int(
                name,
                cfg["range"][0],
                cfg["range"][1],
                step=cfg.get("step", 1),
                log=cfg.get("log", False),
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(name, cfg["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}

        # Sample decoding parameters
        if hasattr(self.param_space_cfg, "decoding") and self.param_space_cfg.decoding.parameters:
            for name, cfg in self.param_space_cfg.decoding.parameters.items():
                # When batch mode is active, the threshold parameter is
                # swept internally (not sampled by Optuna).
                if self._abiss_batch_enabled and name == "ws_merge_threshold":
                    continue
                if self._waterz_batch_enabled and name == "thresholds":
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

    def _build_default_trial_params(self) -> Optional[Dict[str, Any]]:
        """Build a param dict from config defaults to seed the first Optuna trial.

        Maps ``parameter_space.decoding.defaults`` (and postprocessing defaults)
        back to the flat Optuna parameter names used by ``_suggest_param``.
        """
        params: Dict[str, Any] = {}

        # --- decoding defaults ---
        decoding_cfg = getattr(self.param_space_cfg, "decoding", None)
        defaults = getattr(decoding_cfg, "defaults", None) if decoding_cfg else None
        param_defs = getattr(decoding_cfg, "parameters", None) if decoding_cfg else None

        if defaults and param_defs:
            for name, pcfg in param_defs.items():
                if self._abiss_batch_enabled and name == "ws_merge_threshold":
                    continue
                if self._waterz_batch_enabled and name == "thresholds":
                    continue
                val = self._lookup_default(defaults, name, pcfg)
                if val is not None:
                    params[name] = val

        # --- postprocessing defaults ---
        postproc_cfg = getattr(self.param_space_cfg, "postprocessing", None)
        if postproc_cfg and getattr(postproc_cfg, "enabled", False):
            pp_defaults = getattr(postproc_cfg, "defaults", None)
            pp_params = getattr(postproc_cfg, "parameters", None)
            if pp_defaults and pp_params:
                for name, pcfg in pp_params.items():
                    val = self._lookup_default(pp_defaults, name, pcfg)
                    if val is not None:
                        params[name] = val

        return params if params else None

    @staticmethod
    def _lookup_default(defaults: Any, name: str, pcfg: Any) -> Any:
        """Resolve a single default value from the defaults block.

        Handles three layouts:
        - ``nest_under``: ``defaults.<nest_under>.<name>``
        - ``param_group`` + ``tuple_index``: ``defaults.<param_group>[tuple_index]``
        - direct: ``defaults.<name>``
        """
        # nested (e.g. cli_args.ws_high_threshold)
        nest_under = (
            pcfg.get("nest_under", None)
            if hasattr(pcfg, "get")
            else getattr(pcfg, "nest_under", None)
        )
        if nest_under:
            nested = (
                getattr(defaults, nest_under, None)
                if not isinstance(defaults, dict)
                else defaults.get(nest_under)
            )
            if nested is not None:
                val = (
                    nested.get(name, None)
                    if isinstance(nested, dict)
                    else getattr(nested, name, None)
                )
                if val is not None:
                    return val

        # tuple param (e.g. binary_threshold[0])
        param_group = (
            pcfg.get("param_group", None)
            if hasattr(pcfg, "get")
            else getattr(pcfg, "param_group", None)
        )
        tuple_index = (
            pcfg.get("tuple_index", None)
            if hasattr(pcfg, "get")
            else getattr(pcfg, "tuple_index", None)
        )
        if param_group is not None and tuple_index is not None:
            group_val = (
                getattr(defaults, param_group, None)
                if not isinstance(defaults, dict)
                else defaults.get(param_group)
            )
            if isinstance(group_val, (list, tuple)) and int(tuple_index) < len(group_val):
                return group_val[int(tuple_index)]

        # direct
        val = (
            getattr(defaults, name, None) if not isinstance(defaults, dict) else defaults.get(name)
        )
        return val

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
        best_params_file = _resolve_best_params_file(
            self.cfg,
            output_dir,
            checkpoint_path=getattr(self, "_prediction_checkpoint_path", None),
        )
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
            resolved = getattr(self, "_resolved_storage", None)
            if resolved:
                logger.info("Study persisted to database: %s", resolved)
            else:
                logger.warning(
                    "save_study=True but no storage configured and output_dir "
                    "not set — study not persisted to database"
                )


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
        1. Check if the current best-params file already exists (skip if it does)
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
    prediction_checkpoint_path = (
        getattr(model, "_prediction_checkpoint_path", None) or checkpoint_path
    )
    best_params_file = _resolve_best_params_file(
        cfg,
        output_dir,
        checkpoint_path=prediction_checkpoint_path,
    )
    existing_best_params_file = _resolve_existing_best_params_file(
        cfg,
        output_dir,
        checkpoint_path=prediction_checkpoint_path,
    )

    # Check if best parameters already exist
    if existing_best_params_file is not None:
        logger.info(
            "SKIPPING PARAMETER TUNING: best parameters already exist at %s. "
            "Delete this file to re-run tuning.",
            existing_best_params_file,
        )
        _print_best_params_yaml(existing_best_params_file)
        return

    logger.info("STARTING PARAMETER TUNING | Output directory: %s", output_dir)

    # Step 1: Run inference on tune dataset
    from connectomics.data.io import read_volume
    from connectomics.training.lightning import create_datamodule

    logger.info("[1/4] Running inference on tuning dataset...")

    tune_data = cfg.data
    cache_suffix = tta_cache_suffix(cfg, checkpoint_path=prediction_checkpoint_path)

    output_pred_dir = cfg.inference.save_prediction.output_path
    predictions_dir = Path(output_pred_dir)
    pred_files, expected_pred_files = _resolve_tuning_prediction_files(
        cfg, predictions_dir, cache_suffix
    )

    if len(pred_files) == len(expected_pred_files):
        logger.info(
            "Found %d existing tuning prediction file(s) for the current tune dataset in %s "
            "— skipping inference.",
            len(pred_files),
            predictions_dir,
        )
    else:
        if pred_files:
            logger.info(
                "Found %d/%d matching tuning prediction file(s); rerunning inference for missing "
                "volumes instead of mixing partial caches.",
                len(pred_files),
                len(expected_pred_files),
            )
        else:
            logger.info("No matching tuning prediction files found in %s.", predictions_dir)

        # Create datamodule with tune mode using merged runtime cfg.data/cfg.inference.
        datamodule = create_datamodule(cfg, mode="tune")

        logger.info("Using intermediate-only cache generation (decoding/evaluation disabled)")

        # Run test to populate/load raw prediction caches only. Optuna applies its own
        # decoding sweep afterward, so the tune inference pass must not decode with the
        # default config first.
        with _temporary_tuning_inference_overrides(
            cfg,
            getattr(model, "cfg", None),
            checkpoint_path=prediction_checkpoint_path,
        ) as cache_suffix:
            model._tune_mode = True
            try:
                results = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
            finally:
                model._tune_mode = False

        logger.info("Test completed. Results: %s", results)
        pred_files, expected_pred_files = _resolve_tuning_prediction_files(
            cfg, predictions_dir, cache_suffix
        )

    # Step 2: Load predictions from saved files
    logger.info("[2/4] Loading predictions from saved files...")

    if len(pred_files) != len(expected_pred_files):
        missing = sorted(set(expected_pred_files) - set(pred_files))
        raise FileNotFoundError(
            "Missing tuning prediction files for the current tune dataset.\n"
            f"Found: {len(pred_files)}/{len(expected_pred_files)} in {predictions_dir}\n"
            f"Missing: {missing}"
        )

    logger.info("Found %d prediction file(s)", len(pred_files))

    # Load all prediction files as a list (per-volume evaluation)
    all_predictions = []
    for pred_file in pred_files:
        pred = read_volume(pred_file)
        logger.info(
            "Loaded %s: shape %s, dtype %s, range [%.4f, %.4f]",
            Path(pred_file).name,
            pred.shape,
            pred.dtype,
            pred.min(),
            pred.max(),
        )
        all_predictions.append(pred)

    total_slices = sum(p.shape[1] for p in all_predictions)
    logger.info(
        "Loaded %d prediction volumes (%d total slices)",
        len(all_predictions),
        total_slices,
    )

    # Step 3: Load ground truth
    logger.info("[3/4] Loading ground truth labels...")
    tune_label_pattern = getattr(getattr(tune_data, "val", None), "label", None)

    if tune_label_pattern is None:
        raise ValueError("Missing data.val.label in configuration")

    # Handle both string patterns and pre-resolved lists
    label_files = _expand_tuning_paths(tune_label_pattern, field_name="data.val.label")

    if not label_files:
        raise FileNotFoundError(f"No label files found matching pattern: {tune_label_pattern}")

    logger.info("Found %d label file(s)", len(label_files))

    # Load all label files as a list (per-volume evaluation)
    all_labels = []
    for label_file in label_files:
        label = read_volume(label_file)
        logger.info("Loaded %s: shape %s", Path(label_file).name, label.shape)
        all_labels.append(label)

    total_label_slices = sum(label.shape[0] for label in all_labels)
    logger.info(
        "Loaded %d ground truth volumes (%d total slices)",
        len(all_labels),
        total_label_slices,
    )

    # Load mask if available
    all_masks = None
    tune_mask_pattern = getattr(getattr(tune_data, "val", None), "mask", None)
    if tune_mask_pattern:
        mask_files = _expand_tuning_paths(tune_mask_pattern, field_name="data.val.mask")

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
            f"Mismatch: {len(all_predictions)} prediction files vs " f"{len(all_masks)} mask files"
        )

    # Validate that prediction and label spatial shapes match.
    # Cached TTA prediction files are saved after crop_pad + affinity_crop
    # in the test pipeline, so they should already align with the label volume.
    for idx, pred in enumerate(all_predictions):
        pred_spatial = tuple(int(v) for v in pred.shape[-3:])
        label_spatial = tuple(int(v) for v in all_labels[idx].shape[-3:])
        if pred_spatial != label_spatial:
            raise ValueError(
                f"Prediction/label spatial shape mismatch for volume {idx}: "
                f"prediction {pred_spatial} vs label {label_spatial}. "
                f"Cached predictions may be stale — regenerate TTA predictions "
                f"by re-running inference with the real model checkpoint."
            )

    # Step 4: Create tuner and run optimization (per-volume evaluation)
    logger.info("[4/5] Creating Optuna tuner...")
    tuner = OptunaDecodingTuner(
        cfg=cfg, predictions=all_predictions, ground_truth=all_labels, mask=all_masks
    )
    tuner._prediction_checkpoint_path = prediction_checkpoint_path

    logger.info("[5/5] Running optimization study...")
    study = tuner.optimize()

    logger.info(
        "TUNING COMPLETED | Best parameters saved to: %s | " "Best value: %.4f | Best params: %s",
        best_params_file,
        study.best_value,
        study.best_params,
    )
    if best_params_file.exists():
        _print_best_params_yaml(best_params_file)


def load_and_apply_best_params(cfg, checkpoint_path=None):
    """
    Load best parameters from Optuna tuning and apply them to test config.

    This function loads the best-params file generated by run_tuning()
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
    best_params_file = _resolve_existing_best_params_file(
        cfg,
        output_dir,
        checkpoint_path=checkpoint_path,
    )

    if best_params_file is None:
        expected_names = ", ".join(
            tuning_best_params_filename_candidates(cfg, checkpoint_path=checkpoint_path)
        )
        raise FileNotFoundError(
            f"Best parameters file not found in: {output_dir}\n"
            f"Tried: {expected_names}\n"
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
