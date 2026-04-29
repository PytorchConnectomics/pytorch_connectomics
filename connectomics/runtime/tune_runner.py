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
from .output_naming import tta_cache_suffix, tuning_best_params_filename_candidates

logger = logging.getLogger(__name__)


@contextmanager
def temporary_tuning_inference_overrides(*cfg_objects: Any, checkpoint_path: str | None = None):
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
        tta_cache_suffix(primary_cfg, checkpoint_path=checkpoint_path)
        if primary_cfg is not None
        else "_tta_x1_prediction.h5"
    )

    backups = []
    for inference_cfg in inference_cfgs:
        save_prediction_cfg = getattr(inference_cfg, "save_prediction", None)
        if save_prediction_cfg is None:
            raise ValueError("Missing inference.save_prediction configuration required for tuning")

        backups.append(
            {
                "inference_cfg": inference_cfg,
                "save_prediction_enabled": bool(getattr(save_prediction_cfg, "enabled", False)),
                "save_prediction_cache_suffix": getattr(
                    save_prediction_cfg,
                    "cache_suffix",
                    "_x1_prediction.h5",
                ),
            }
        )

        save_prediction_cfg.enabled = True
        save_prediction_cfg.cache_suffix = suffix

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
            save_prediction_cfg = inference_cfg.save_prediction
            save_prediction_cfg.enabled = backup["save_prediction_enabled"]
            save_prediction_cfg.cache_suffix = backup["save_prediction_cache_suffix"]

        for owner, decoding in decoding_backups:
            owner.decoding = decoding

        for evaluation_cfg, evaluation_enabled in evaluation_backups:
            evaluation_cfg.enabled = evaluation_enabled


def run_tuning(model, trainer, cfg, checkpoint_path=None):
    """Run Optuna-based parameter tuning for instance segmentation decoding."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for parameter tuning. Install with: pip install optuna"
        )

    output_pred_dir = getattr(cfg.inference.save_prediction, "output_path", None)
    if not output_pred_dir:
        raise ValueError("Missing inference.save_prediction.output_path in configuration")
    output_dir = Path(output_pred_dir).parent / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    cache_suffix = tta_cache_suffix(cfg, checkpoint_path=prediction_checkpoint_path)

    output_pred_dir = cfg.inference.save_prediction.output_path
    predictions_dir = Path(output_pred_dir)
    pred_files, expected_pred_files = _resolve_tuning_prediction_files(
        cfg,
        predictions_dir,
        cache_suffix,
    )

    if len(pred_files) == len(expected_pred_files):
        logger.info(
            "Found %d existing tuning prediction file(s) for the current tune dataset in %s "
            "- skipping inference.",
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

        datamodule = create_datamodule(cfg, mode="tune")

        logger.info("Using intermediate-only cache generation (decoding/evaluation disabled)")
        with temporary_tuning_inference_overrides(
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
            cfg,
            predictions_dir,
            cache_suffix,
        )

    logger.info("[2/4] Loading predictions from saved files...")

    if len(pred_files) != len(expected_pred_files):
        missing = sorted(set(expected_pred_files) - set(pred_files))
        raise FileNotFoundError(
            "Missing tuning prediction files for the current tune dataset.\n"
            f"Found: {len(pred_files)}/{len(expected_pred_files)} in {predictions_dir}\n"
            f"Missing: {missing}"
        )

    logger.info("Found %d prediction file(s)", len(pred_files))

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

    logger.info("[3/4] Loading ground truth labels...")
    tune_label_pattern = getattr(getattr(tune_data, "val", None), "label", None)

    if tune_label_pattern is None:
        raise ValueError("Missing data.val.label in configuration")

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


def load_and_apply_best_params(cfg, checkpoint_path=None):
    """Load tuned parameters and apply them to the merged runtime decoding config."""
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


__all__ = [
    "load_and_apply_best_params",
    "run_tuning",
    "temporary_tuning_inference_overrides",
]
