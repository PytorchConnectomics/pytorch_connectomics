"""Runtime orchestration for decoding parameter tuning."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..decoding.tuning.optuna_tuner import (
    OPTUNA_AVAILABLE,
    OptunaDecodingTuner,
    _expand_tuning_paths,
    _print_best_params_yaml,
    _resolve_best_params_file,
    _resolve_existing_best_params_file,
    _resolve_tuning_prediction_files,
)
from .output_naming import (
    intermediate_prediction_cache_suffix,
    tuning_best_params_filename_candidates,
)

logger = logging.getLogger(__name__)


def _unique_prediction_dirs(*paths: str | Path | None) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        resolved = Path(path)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _resolve_first_complete_tuning_prediction_cache(
    cfg,
    prediction_dirs: list[Path],
    cache_suffix: str,
) -> tuple[Path | None, list[str], list[str]]:
    expected_from_first_dir: list[str] = []
    for idx, predictions_dir in enumerate(prediction_dirs):
        pred_files, expected_pred_files = _resolve_tuning_prediction_files(
            cfg,
            predictions_dir,
            cache_suffix,
        )
        if idx == 0:
            expected_from_first_dir = expected_pred_files
        if len(pred_files) == len(expected_pred_files):
            return predictions_dir, pred_files, expected_pred_files
    return None, [], expected_from_first_dir


def _resolve_tuning_output_dir(cfg, fallback_prediction_dir: str | Path | None) -> Path:
    tune_cfg = getattr(cfg, "tune", None)
    configured_output_dir = getattr(tune_cfg, "save_path", None)
    if configured_output_dir:
        return Path(configured_output_dir)
    configured_prediction_dir = getattr(tune_cfg, "save_predictions_path", None)
    if configured_prediction_dir:
        return Path(configured_prediction_dir).parent
    if fallback_prediction_dir:
        return Path(fallback_prediction_dir).parent / "tuning"
    raise ValueError(
        "Missing tuning output directory. Set tune.save_path "
        "or inference.save_path."
    )


@contextmanager
def temporary_tuning_inference_overrides(
    *cfg_objects: Any,
    checkpoint_path: str | None = None,
    prediction_output_path: str | Path | None = None,
):
    """Force the pre-Optuna inference pass to cache raw predictions only."""
    inference_cfgs = []
    seen_inference_cfgs: set[int] = set()
    decoding_owners = []
    seen_decoding_owners: set[int] = set()
    evaluation_cfgs = []
    seen_evaluation_cfgs: set[int] = set()
    primary_cfg = None
    for cfg_obj in cfg_objects:
        if cfg_obj is None:
            continue
        if primary_cfg is None:
            primary_cfg = cfg_obj
        evaluation_cfg = getattr(cfg_obj, "evaluation", None)
        if evaluation_cfg is not None and id(evaluation_cfg) not in seen_evaluation_cfgs:
            evaluation_cfgs.append(evaluation_cfg)
            seen_evaluation_cfgs.add(id(evaluation_cfg))
        if hasattr(cfg_obj, "decoding") and id(cfg_obj) not in seen_decoding_owners:
            decoding_owners.append(cfg_obj)
            seen_decoding_owners.add(id(cfg_obj))
        inference_cfg = getattr(cfg_obj, "inference", None)
        if inference_cfg is None or id(inference_cfg) in seen_inference_cfgs:
            continue
        inference_cfgs.append(inference_cfg)
        seen_inference_cfgs.add(id(inference_cfg))

    if not inference_cfgs:
        raise ValueError("Missing runtime cfg.inference configuration required for tuning")

    suffix = (
        intermediate_prediction_cache_suffix(primary_cfg, checkpoint_path=checkpoint_path)
        if primary_cfg is not None
        else "_tta_x1_prediction.h5"
    )

    backups = []
    for inference_cfg in inference_cfgs:
        backups.append(
            {
                "inference_cfg": inference_cfg,
                "save_results": bool(getattr(inference_cfg, "save_results", False)),
                "save_cache_suffix": getattr(inference_cfg, "save_cache_suffix", "_x1_prediction.h5"),
                "save_path": getattr(inference_cfg, "save_path", None),
            }
        )

        inference_cfg.save_results = True
        inference_cfg.save_cache_suffix = suffix
        if prediction_output_path is not None:
            inference_cfg.save_path = str(prediction_output_path)

    decoding_backups = [
        (owner, deepcopy(getattr(owner, "decoding", None))) for owner in decoding_owners
    ]
    for owner in decoding_owners:
        owner.decoding = None

    evaluation_backups = [
        (evaluation_cfg, bool(getattr(evaluation_cfg, "enabled", False)))
        for evaluation_cfg in evaluation_cfgs
    ]
    for evaluation_cfg in evaluation_cfgs:
        if evaluation_cfg is not None:
            evaluation_cfg.enabled = False

    try:
        yield suffix
    finally:
        for backup in backups:
            inference_cfg = backup["inference_cfg"]
            inference_cfg.save_results = backup["save_results"]
            inference_cfg.save_cache_suffix = backup["save_cache_suffix"]
            inference_cfg.save_path = backup["save_path"]

        for owner, decoding in decoding_backups:
            owner.decoding = decoding

        for evaluation_cfg, evaluation_enabled in evaluation_backups:
            evaluation_cfg.enabled = evaluation_enabled


def run_tuning(model, trainer_or_factory, cfg, checkpoint_path=None):
    """Run Optuna-based parameter tuning for instance segmentation decoding.

    ``trainer_or_factory`` may be a Lightning Trainer or a zero-arg callable
    that builds one on demand. The trainer is only constructed when the
    intermediate prediction cache misses and inference must be re-run.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for parameter tuning. Install with: pip install optuna"
        )

    tune_cfg = getattr(cfg, "tune", None)
    primary_output_pred_dir = getattr(tune_cfg, "save_predictions_path", None)
    fallback_output_pred_dir = getattr(cfg.inference, "save_path", None)
    if not primary_output_pred_dir:
        primary_output_pred_dir = fallback_output_pred_dir
    if not primary_output_pred_dir:
        raise ValueError(
            "Missing tuning prediction output path. Set tune.save_predictions_path "
            "or inference.save_path."
        )
    predictions_dir = Path(primary_output_pred_dir)
    inference_write_dir = Path(fallback_output_pred_dir or primary_output_pred_dir)
    prediction_search_dirs = _unique_prediction_dirs(
        predictions_dir,
        inference_write_dir,
    )
    inference_write_dir.mkdir(parents=True, exist_ok=True)

    output_dir = _resolve_tuning_output_dir(cfg, fallback_output_pred_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.tune.save_path = str(output_dir)
    cfg.tune.save_predictions_path = str(predictions_dir)
    prediction_checkpoint_path = (
        getattr(model, "_prediction_checkpoint_path", None) if model is not None else None
    ) or checkpoint_path
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

    if existing_best_params_file is not None:
        logger.info(
            "SKIPPING PARAMETER TUNING: best parameters already exist at %s. "
            "Delete this file to re-run tuning.",
            existing_best_params_file,
        )
        _print_best_params_yaml(existing_best_params_file)
        return

    logger.info("STARTING PARAMETER TUNING | Output directory: %s", output_dir)

    from connectomics.data.io import read_volume
    from connectomics.training.lightning import create_datamodule

    logger.info("[1/4] Running inference on tuning dataset...")

    tune_data = cfg.data
    cache_suffix = intermediate_prediction_cache_suffix(
        cfg, checkpoint_path=prediction_checkpoint_path
    )

    cache_dir, pred_files, expected_pred_files = _resolve_first_complete_tuning_prediction_cache(
        cfg,
        prediction_search_dirs,
        cache_suffix,
    )

    if cache_dir is not None:
        cache_kind = (
            "tuning prediction cache"
            if str(cache_dir) == str(predictions_dir)
            else "inference result cache"
        )
        logger.info(
            "Found %d existing %s file(s) for the current tune dataset in %s - "
            "skipping inference.",
            len(pred_files),
            cache_kind,
            cache_dir,
        )
    else:
        logger.info(
            "No complete tuning prediction cache found in %s.",
            ", ".join(str(path) for path in prediction_search_dirs),
        )

        if model is None:
            raise RuntimeError(
                "Cached intermediate predictions are missing from both the tuning "
                "prediction directory and fallback inference results directory, but the "
                "Lightning module was not built. This indicates a bug in the dispatch "
                "logic — the cache miss should have triggered model construction."
            )

        datamodule = create_datamodule(cfg, mode="tune")

        logger.info("Using intermediate-only cache generation (decoding/evaluation disabled)")
        trainer = trainer_or_factory() if callable(trainer_or_factory) else trainer_or_factory
        with temporary_tuning_inference_overrides(
            cfg,
            getattr(model, "cfg", None),
            checkpoint_path=prediction_checkpoint_path,
            prediction_output_path=inference_write_dir,
        ) as cache_suffix:
            model._tune_mode = True
            try:
                results = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
            finally:
                model._tune_mode = False

        logger.info("Test completed. Results: %s", results)
        pred_files, expected_pred_files = _resolve_tuning_prediction_files(
            cfg,
            inference_write_dir,
            cache_suffix,
        )
        cache_dir = inference_write_dir

    logger.info("[2/4] Loading predictions...")

    if len(pred_files) != len(expected_pred_files):
        missing = sorted(set(expected_pred_files) - set(pred_files))
        active_prediction_dir = cache_dir or predictions_dir
        raise FileNotFoundError(
            "Missing tuning prediction files for the current tune dataset.\n"
            f"Found: {len(pred_files)}/{len(expected_pred_files)} in {active_prediction_dir}\n"
            f"Missing: {missing}"
        )

    logger.info("Found %d prediction file(s)", len(pred_files))

    in_memory_cache = (
        getattr(model, "_tune_predictions_cache", None) if model is not None else None
    ) or {}

    all_predictions = []
    for pred_file in pred_files:
        cache_key = Path(pred_file).name.removesuffix(cache_suffix)
        cached = in_memory_cache.get(cache_key)
        if cached is not None:
            pred = cached
            source = "in-memory"
        else:
            pred = read_volume(pred_file)
            source = "disk"
        logger.info(
            "Loaded %s (%s): shape %s, dtype %s, range [%.4f, %.4f]",
            Path(pred_file).name,
            source,
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

    tune_label_pattern = getattr(getattr(tune_data, "val", None), "label", None)
    if tune_label_pattern is None:
        raise ValueError("Missing data.val.label in configuration")

    logger.info("[3/4] Loading ground truth labels for tuning from %s ...", tune_label_pattern)
    label_files = _expand_tuning_paths(tune_label_pattern, field_name="data.val.label")

    if not label_files:
        raise FileNotFoundError(f"No label files found matching pattern: {tune_label_pattern}")

    logger.info("Found %d label file(s)", len(label_files))

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
                mask = read_volume(mask_file)
                logger.info("Loaded %s: shape %s", Path(mask_file).name, mask.shape)
                all_masks.append(mask)

            logger.info("Loaded %d mask volumes", len(all_masks))

    if len(all_predictions) != len(all_labels):
        raise ValueError(
            f"Mismatch: {len(all_predictions)} prediction files vs "
            f"{len(all_labels)} label files"
        )
    if all_masks is not None and len(all_masks) != len(all_predictions):
        raise ValueError(
            f"Mismatch: {len(all_predictions)} prediction files vs {len(all_masks)} mask files"
        )

    for idx, pred in enumerate(all_predictions):
        pred_spatial = tuple(int(v) for v in pred.shape[-3:])
        label_spatial = tuple(int(v) for v in all_labels[idx].shape[-3:])
        if pred_spatial != label_spatial:
            raise ValueError(
                f"Prediction/label spatial shape mismatch for volume {idx}: "
                f"prediction {pred_spatial} vs label {label_spatial}. "
                "Cached predictions may be stale - regenerate TTA predictions "
                "by re-running inference with the real model checkpoint."
            )

    logger.info("[4/5] Creating Optuna tuner...")
    tuner = OptunaDecodingTuner(
        cfg=cfg,
        predictions=all_predictions,
        ground_truth=all_labels,
        mask=all_masks,
    )
    tuner._prediction_checkpoint_path = prediction_checkpoint_path

    logger.info("[5/5] Running optimization study...")
    study = tuner.optimize()

    logger.info(
        "TUNING COMPLETED | Best parameters saved to: %s | Best value: %.4f | Best params: %s",
        best_params_file,
        study.best_value,
        study.best_params,
    )
    if best_params_file.exists():
        _print_best_params_yaml(best_params_file)

    if model is not None and hasattr(model, "_tune_predictions_cache"):
        model._tune_predictions_cache.clear()


def load_and_apply_best_params(cfg, checkpoint_path=None):
    """Load tuned parameters and apply them to the merged runtime decoding config."""
    output_pred_dir = getattr(cfg.inference, "save_path", None)
    output_dir = _resolve_tuning_output_dir(cfg, output_pred_dir)
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
            "Run parameter tuning first with --mode tune"
        )

    logger.info("Loading best parameters from: %s", best_params_file)

    best_params = OmegaConf.load(best_params_file)
    logger.info("Loaded best parameters:\n%s", OmegaConf.to_yaml(best_params))

    decoding_cfg = getattr(cfg, "decoding", None)
    if decoding_cfg is None:
        raise ValueError("Missing top-level decoding configuration")
    if decoding_cfg.steps is None:
        decoding_cfg.steps = []
    decoding_steps = decoding_cfg.steps

    decoding_function = best_params.get("decoding_function", None)

    if decoding_function is None:
        warnings.warn("No decoding_function found in best_params, applying to first decoder")
        decoder_idx = 0
    else:
        decoder_idx = None
        for idx, decoder in enumerate(decoding_steps):
            decoder_name = (
                decoder.get("name") if isinstance(decoder, dict) else getattr(decoder, "name", None)
            )
            if decoder_name == decoding_function:
                decoder_idx = idx
                break

        if decoder_idx is None:
            decoder_idx = len(decoding_steps)
            decoding_steps.append({"name": decoding_function, "kwargs": {}})

    if decoder_idx < len(decoding_steps):
        decoder = decoding_steps[decoder_idx]

        if isinstance(decoder, dict):
            if "kwargs" not in decoder:
                decoder["kwargs"] = {}
            decoder["kwargs"].update(OmegaConf.to_container(best_params["decoding_params"]))
        else:
            if not hasattr(decoder, "kwargs") or decoder.kwargs is None:
                decoder.kwargs = {}
            for key, value in best_params["decoding_params"].items():
                decoder.kwargs[key] = value

        logger.info("Applied best parameters to decoding[%d]", decoder_idx)

    return cfg


def try_skip_tune_with_cached_results(cfg, checkpoint_path: str | None) -> bool:
    """Short-circuit ``--mode tune`` when ``best_params.yaml`` already exists.

    The presence of a previously-saved final-decoded segmentation file is NOT a
    valid skip signal: that file reflects whatever decoding parameters were used
    in a prior test run, not the result of an Optuna sweep. Only an existing
    best-params YAML proves tuning has converged.
    """
    output_pred_dir = getattr(cfg.inference, "save_path", None)

    try:
        tuning_dir = _resolve_tuning_output_dir(cfg, output_pred_dir)
    except ValueError:
        return False
    best_params_file = _resolve_existing_best_params_file(
        cfg, tuning_dir, checkpoint_path=checkpoint_path
    )
    if best_params_file is None:
        return False

    logger.info(
        "SKIPPING PARAMETER TUNING: best parameters already exist at %s. "
        "Delete this file to re-run tuning.",
        best_params_file,
    )
    _print_best_params_yaml(best_params_file)
    return True


__all__ = [
    "load_and_apply_best_params",
    "run_tuning",
    "temporary_tuning_inference_overrides",
    "try_skip_tune_with_cached_results",
]
