"""Helpers for test-time inference, decoding, postprocessing, and metrics."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ...decoding import apply_decode_mode, resolve_decode_modes_from_cfg
from ...inference import apply_postprocessing, apply_save_prediction_transform, write_outputs


def _align_metric_tensors(
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_tensor = pred_tensor.squeeze()
    labels_tensor = labels_tensor.squeeze()

    if pred_tensor.shape == labels_tensor.shape:
        return pred_tensor, labels_tensor

    print(f"  ⚠️  Shape mismatch: pred={pred_tensor.shape}, labels={labels_tensor.shape}")
    if pred_tensor.ndim == labels_tensor.ndim - 1:
        pred_tensor = pred_tensor.unsqueeze(0)
    elif labels_tensor.ndim == pred_tensor.ndim - 1:
        labels_tensor = labels_tensor.unsqueeze(0)

    if pred_tensor.shape != labels_tensor.shape:
        print("  ❌ Cannot compute metrics: incompatible shapes after alignment")
        print(f"     pred={pred_tensor.shape}, labels={labels_tensor.shape}")
        return None, None

    return pred_tensor, labels_tensor


def _is_instance_segmentation(pred_tensor: torch.Tensor) -> bool:
    return pred_tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ) or (pred_tensor.dtype == torch.float32 and pred_tensor.max() > 1.0)


def _compute_instance_metrics(
    module,
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    instance_iou_threshold: float,
) -> None:
    pred_instances = pred_tensor.long()
    labels_instances = labels_tensor.long()

    if hasattr(module, "test_adapted_rand") and isinstance(module.test_adapted_rand, torchmetrics.Metric):
        from ...metrics.metrics_seg import AdaptedRandError

        per_volume_metric = AdaptedRandError(return_all_stats=True).to(module.device)
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

        metrics_dict["adapted_rand_error"] = are_score
        module.test_adapted_rand.update(pred_instances.cpu(), labels_instances.cpu())

        epoch_stats = module.test_adapted_rand.compute()
        if isinstance(epoch_stats, dict):
            module.log(
                "test_adapted_rand",
                epoch_stats["adapted_rand_error"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            module.log(
                "test_adapted_rand_precision",
                epoch_stats["adapted_rand_precision"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            module.log(
                "test_adapted_rand_recall",
                epoch_stats["adapted_rand_recall"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            module.log(
                "test_adapted_rand",
                epoch_stats,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    if hasattr(module, "test_voi") and isinstance(module.test_voi, torchmetrics.Metric):
        from ...metrics.segmentation_numpy import voi

        split, merge = voi(pred_instances.cpu().numpy(), labels_instances.cpu().numpy())
        print(f"  {volume_prefix}VOI Split: {split:.6f}")
        print(f"  {volume_prefix}VOI Merge: {merge:.6f}")
        print(f"  {volume_prefix}VOI Total: {split + merge:.6f}")

        metrics_dict["voi_split"] = split
        metrics_dict["voi_merge"] = merge
        metrics_dict["voi_total"] = split + merge

        module.test_voi.update(pred_instances.cpu(), labels_instances.cpu())
        module.log("test_voi", module.test_voi, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        module.log(
            "test_voi_split",
            module.test_voi.compute_split(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        module.log(
            "test_voi_merge",
            module.test_voi.compute_merge(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    if hasattr(module, "test_instance_accuracy") and isinstance(
        module.test_instance_accuracy, torchmetrics.Metric
    ):
        from ...metrics.segmentation_numpy import instance_matching

        stats = instance_matching(
            labels_instances.cpu().numpy(),
            pred_instances.cpu().numpy(),
            thresh=instance_iou_threshold,
            criterion="iou",
        )
        print(f"  {volume_prefix}Instance Accuracy: {stats['accuracy']:.6f}")
        metrics_dict["instance_accuracy"] = stats["accuracy"]

        module.test_instance_accuracy.update(pred_instances.cpu(), labels_instances.cpu())
        module.log(
            "test_instance_accuracy",
            module.test_instance_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    if hasattr(module, "test_instance_accuracy_detail") and isinstance(
        module.test_instance_accuracy_detail, torchmetrics.Metric
    ):
        from ...metrics.segmentation_numpy import instance_matching_simple

        stats_simple = instance_matching_simple(
            labels_instances.cpu().numpy(),
            pred_instances.cpu().numpy(),
            thresh=instance_iou_threshold,
            criterion="iou",
        )
        print(
            f"  {volume_prefix}Instance Accuracy (Detail): {stats_simple['accuracy']:.6f} [relaxed, non-Hungarian]"
        )
        print(f"  {volume_prefix}  ├─ Precision: {stats_simple['precision']:.6f}")
        print(f"  {volume_prefix}  ├─ Recall: {stats_simple['recall']:.6f}")
        print(f"  {volume_prefix}  └─ F1: {stats_simple['f1']:.6f}")

        metrics_dict["instance_accuracy_detail"] = stats_simple["accuracy"]
        metrics_dict["instance_precision_detail"] = stats_simple["precision"]
        metrics_dict["instance_recall_detail"] = stats_simple["recall"]
        metrics_dict["instance_f1_detail"] = stats_simple["f1"]

        module.test_instance_accuracy_detail.update(pred_instances.cpu(), labels_instances.cpu())
        module.log(
            "test_instance_accuracy_detail",
            module.test_instance_accuracy_detail,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        module.log(
            "test_instance_precision_detail",
            module.test_instance_accuracy_detail.compute_precision(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        module.log(
            "test_instance_recall_detail",
            module.test_instance_accuracy_detail.compute_recall(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        module.log(
            "test_instance_f1_detail",
            module.test_instance_accuracy_detail.compute_f1(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )


def _compute_binary_metrics(
    module,
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    prediction_threshold: float,
) -> None:
    if pred_tensor.max() <= 1.0:
        pred_binary = (pred_tensor > prediction_threshold).long()
    else:
        pred_binary = (torch.sigmoid(pred_tensor) > prediction_threshold).long()

    labels_binary = (
        (labels_tensor > prediction_threshold).long()
        if labels_tensor.max() <= 1.0
        else labels_tensor.long()
    )

    if hasattr(module, "test_jaccard") and module.test_jaccard is not None:
        jaccard_value = torchmetrics.functional.jaccard_index(
            pred_binary,
            labels_binary,
            task="binary",
        )
        print(f"  {volume_prefix}Jaccard: {jaccard_value.item():.6f}")
        metrics_dict["jaccard"] = jaccard_value.item()
        module.test_jaccard.update(pred_binary, labels_binary)
        module.log(
            "test_jaccard",
            module.test_jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    if hasattr(module, "test_dice") and module.test_dice is not None:
        dice_value = torchmetrics.functional.dice(pred_binary, labels_binary)
        print(f"  {volume_prefix}Dice: {dice_value.item():.6f}")
        metrics_dict["dice"] = dice_value.item()
        module.test_dice.update(pred_binary, labels_binary)
        module.log(
            "test_dice",
            module.test_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    if hasattr(module, "test_accuracy") and module.test_accuracy is not None:
        accuracy_value = torchmetrics.functional.accuracy(
            pred_binary,
            labels_binary,
            task="binary",
        )
        print(f"  {volume_prefix}Accuracy: {accuracy_value.item():.6f}")
        metrics_dict["accuracy"] = accuracy_value.item()
        module.test_accuracy.update(pred_binary, labels_binary)
        module.log(
            "test_accuracy",
            module.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


def compute_test_metrics(
    module,
    decoded_predictions: np.ndarray,
    labels: torch.Tensor,
    volume_name: str | None = None,
) -> None:
    """Update configured metrics and save per-volume evaluation summaries."""
    if not module._is_test_evaluation_enabled():
        return

    pred_tensor = torch.from_numpy(decoded_predictions).float().to(module.device)
    labels_tensor = labels.float().to(pred_tensor.device)
    pred_tensor, labels_tensor = _align_metric_tensors(pred_tensor, labels_tensor)
    if pred_tensor is None or labels_tensor is None:
        return

    volume_prefix = f"[{volume_name}] " if volume_name else ""
    metrics_dict: Dict[str, Any] = {"volume_name": volume_name if volume_name else "unknown"}

    inference_eval_defaults = module._get_runtime_inference_config().evaluation
    evaluation_cfg = module._get_test_evaluation_config()
    prediction_threshold = module._cfg_float(
        evaluation_cfg,
        "prediction_threshold",
        float(inference_eval_defaults.prediction_threshold),
    )
    instance_iou_threshold = module._cfg_float(
        evaluation_cfg,
        "instance_iou_threshold",
        float(inference_eval_defaults.instance_iou_threshold),
    )

    if _is_instance_segmentation(pred_tensor):
        _compute_instance_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            instance_iou_threshold,
        )
    else:
        _compute_binary_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            prediction_threshold,
        )

    module._save_metrics_to_file(metrics_dict)


def _log_volume_header(volume_name: str, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"{title}: {volume_name}")
    print(f"{'=' * 70}")


def _process_decoding_postprocessing(
    module,
    predictions_np: np.ndarray,
    *,
    filenames: list[str],
    mode: str,
    batch_meta: Any,
    save_final_predictions: bool,
) -> np.ndarray:
    print("\n  🔄 [STAGE: Decoding Instances]")
    decode_start = time.time()
    has_decoding_cfg = bool(resolve_decode_modes_from_cfg(module.cfg))
    decoded_predictions = apply_decode_mode(module.cfg, predictions_np)
    print(f"  ✅ Decoding completed ({time.time() - decode_start:.1f}s)")

    if not has_decoding_cfg:
        print("  ⏭️  Skipping postprocessing (no decoding configuration)")
        print("  ⏭️  Skipping decoded segmentation summary (no decoding configuration)")
        print("  ⏭️  Skipping final prediction save (no decoding configuration)")
        return decoded_predictions

    postprocessed_predictions = apply_postprocessing(module.cfg, decoded_predictions)
    print("\n  📊 Decoded Segmentation Summary:")
    print(f"      Shape:      {decoded_predictions.shape}")
    print(f"      Dtype:      {decoded_predictions.dtype}")
    print(f"      Min:        {decoded_predictions.min()}")
    print(f"      Max:        {decoded_predictions.max()}")
    print(f"      Instances:  {decoded_predictions.max()} (max label)")
    print(f"      Unique IDs: {len(np.unique(decoded_predictions))}")
    print("")

    if save_final_predictions:
        print("  💾 [STAGE: Saving Final Predictions]")
        save_start = time.time()
        write_outputs(
            module.cfg,
            postprocessed_predictions,
            filenames,
            suffix="prediction",
            mode=mode,
            batch_meta=batch_meta,
        )
        print(f"  ✅ Final predictions saved ({time.time() - save_start:.1f}s)")

    return decoded_predictions


def _evaluate_decoded_predictions(
    module,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    *,
    filenames: list[str],
    batch_idx: int,
) -> None:
    if labels is not None and module._is_test_evaluation_enabled():
        print("\n  📈 [STAGE: Computing Evaluation Metrics]")
        eval_start = time.time()
        volume_names = filenames if filenames else [f"volume_{batch_idx}"]

        # Multi-volume test batches can occur when batch_size > 1.
        # Evaluate each volume independently so per-volume metrics are emitted.
        if len(volume_names) > 1:
            pred_arr = np.asarray(decoded_predictions)
            can_split_pred = pred_arr.ndim > 0 and pred_arr.shape[0] == len(volume_names)
            can_split_label = labels.ndim > 0 and labels.shape[0] == len(volume_names)
            if can_split_pred and can_split_label:
                for i, name in enumerate(volume_names):
                    compute_test_metrics(module, pred_arr[i], labels[i], volume_name=name)
            else:
                print(
                    "  ⚠️  Could not split batched predictions/labels by volume; "
                    "computing a single aggregate metric."
                )
                compute_test_metrics(module, pred_arr, labels, volume_name=volume_names[0])
        else:
            compute_test_metrics(module, decoded_predictions, labels, volume_name=volume_names[0])

        print(f"  ✅ Evaluation completed ({time.time() - eval_start:.1f}s)")
        return

    if labels is None:
        print("\n  ⏭️  [STAGE: Evaluation] Skipped (no ground truth labels)")
    else:
        print("\n  ⏭️  [STAGE: Evaluation] Skipped (evaluation disabled)")


def run_test_step(module, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
    """End-to-end test-step workflow with cache reuse and staged processing."""
    images = batch["image"]
    labels = batch.get("label")
    mask = batch.get("mask")

    mode, output_dir_value, cache_suffix, filenames = module._resolve_test_output_config(batch)
    predictions_np, loaded_from_file, loaded_suffix = module._load_cached_predictions(
        output_dir_value,
        filenames,
        cache_suffix,
        mode,
    )

    loaded_final_predictions = loaded_from_file and loaded_suffix == "_prediction.h5"
    loaded_intermediate_predictions = loaded_from_file and loaded_suffix == "_tta_prediction.h5"
    volume_name = filenames[0] if filenames else f"volume_{batch_idx}"

    if loaded_final_predictions:
        print("  ✅ Loaded final predictions from disk, skipping inference/decoding/postprocessing")
        _evaluate_decoded_predictions(
            module,
            predictions_np,
            labels,
            filenames=filenames,
            batch_idx=batch_idx,
        )
        return torch.tensor(0.0, device=module.device)

    if loaded_intermediate_predictions:
        print("  ✅ Loaded intermediate predictions from disk, skipping inference")
        _log_volume_header(volume_name, "PROCESSING VOLUME")
        predictions_np = module._invert_save_prediction_transform(predictions_np)
        decoded_predictions = _process_decoding_postprocessing(
            module,
            predictions_np,
            filenames=filenames,
            mode=mode,
            batch_meta=batch.get("image_meta_dict"),
            save_final_predictions=True,
        )
        _evaluate_decoded_predictions(
            module,
            decoded_predictions,
            labels,
            filenames=filenames,
            batch_idx=batch_idx,
        )
        _log_volume_header(volume_name, "VOLUME COMPLETE")
        print("")
        return torch.tensor(0.0, device=module.device)

    print("  🔄 No cached predictions found, running inference")
    _log_volume_header(volume_name, "INFERENCE PLAN")
    print(f"Input shape:       {tuple(images.shape)}")
    print(f"Input device:      {images.device}")

    inference_cfg = module._get_runtime_inference_config()
    sw_cfg = getattr(inference_cfg, "sliding_window", None)
    if sw_cfg is not None:
        roi_size = getattr(sw_cfg, "window_size", "N/A")
        overlap = getattr(sw_cfg, "overlap", "N/A")
        sw_batch = getattr(sw_cfg, "sw_batch_size", "N/A")
        blending = getattr(sw_cfg, "blending", "gaussian")
        print(f"Sliding window ROI: {roi_size}")
        print(f"Overlap:            {overlap}")
        print(f"SW batch size:      {sw_batch}")
        print(f"Blending mode:      {blending}")
    else:
        print("Sliding window:     [Direct inference, no sliding window]")
    print(f"TTA:                {module._summarize_tta_plan(images.ndim)}")
    print(f"{'=' * 70}\n")

    inference_start = time.time()
    print("  ⏱️  Starting sliding-window inference...")

    mask_align_to_image = False
    mask_transform_cfg = getattr(module.cfg.data, "data_transform", None)
    if mask_transform_cfg is not None:
        mask_align_to_image = bool(getattr(mask_transform_cfg, "align_to_image", False))

    predictions = module.inference_manager.predict_with_tta(
        images,
        mask=mask,
        mask_align_to_image=mask_align_to_image,
    )
    predictions_np = predictions.detach().cpu().float().numpy()
    inference_duration = time.time() - inference_start
    print(f"  ✅ Inference completed in {inference_duration / 60:.2f} minutes ({inference_duration:.1f}s)")

    print("\n  📊 Prediction Summary:")
    print(f"      Shape:  {predictions_np.shape}")
    print(f"      Dtype:  {predictions_np.dtype}")
    print(f"      Min:    {predictions_np.min():.6f}")
    print(f"      Max:    {predictions_np.max():.6f}")
    print(f"      Mean:   {predictions_np.mean():.6f}")
    print("")

    save_intermediate = bool(getattr(inference_cfg.save_prediction, "enabled", False))
    if save_intermediate:
        print("\n  💾 [STAGE: Saving Intermediate Predictions]")
        save_start = time.time()
        predictions_to_save = apply_save_prediction_transform(module.cfg, predictions_np)
        write_outputs(
            module.cfg,
            predictions_to_save,
            filenames,
            suffix="tta_prediction",
            mode=mode,
            batch_meta=batch.get("image_meta_dict"),
        )
        print(f"  ✅ Intermediate predictions saved ({time.time() - save_start:.1f}s)")

    decoded_predictions = _process_decoding_postprocessing(
        module,
        predictions_np,
        filenames=filenames,
        mode=mode,
        batch_meta=batch.get("image_meta_dict"),
        save_final_predictions=True,
    )
    _evaluate_decoded_predictions(
        module,
        decoded_predictions,
        labels,
        filenames=filenames,
        batch_idx=batch_idx,
    )
    _log_volume_header(volume_name, "VOLUME COMPLETE")
    print("")
    return torch.tensor(0.0, device=module.device)
