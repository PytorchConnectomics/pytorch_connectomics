#!/usr/bin/env python3
"""Evaluate a saved prediction volume against ground truth.

This is a direct file-based companion to `just test`: it reuses the same PyTC
metric implementations and writes a report with the same metric fields, but it
does not run model inference or decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from connectomics.data.io import read_volume  # noqa: E402
from connectomics.evaluation.context import EvaluationContext  # noqa: E402
from connectomics.evaluation.metric_execution import (  # noqa: E402
    align_metric_tensors,
    compute_binary_metrics,
    compute_instance_metrics,
)
from connectomics.evaluation.report import (  # noqa: E402
    BINARY_METRIC_NAMES,
    INSTANCE_METRIC_NAMES,
)
from connectomics.metrics.nerl import (  # noqa: E402
    NerlGraphOptions,
    extract_nerl_score_outputs,
    import_em_erl,
    load_nerl_graph,
)

NERL_METRIC_NAMES = {"nerl"}
SUPPORTED_METRICS = INSTANCE_METRIC_NAMES | BINARY_METRIC_NAMES | NERL_METRIC_NAMES


def parse_metrics(value: str) -> set[str]:
    metrics = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not metrics:
        raise ValueError("metric must contain at least one metric name")
    unknown = metrics - SUPPORTED_METRICS
    if unknown:
        raise ValueError(
            f"Unsupported metric(s): {sorted(unknown)}. " f"Supported: {sorted(SUPPORTED_METRICS)}"
        )
    if "nerl" in metrics and len(metrics) > 1:
        raise ValueError("nerl uses skeleton ground truth and cannot be mixed with dense metrics")
    return metrics


def _metric_value(metrics_dict: dict[str, Any], metric: str) -> float | None:
    key_by_metric = {
        "adapted_rand": "adapted_rand_error",
        "voi": "voi_total",
        "instance_accuracy": "instance_accuracy",
        "instance_accuracy_detail": "instance_accuracy_detail",
        "jaccard": "jaccard",
        "dice": "dice",
        "accuracy": "accuracy",
        "nerl": "nerl",
    }
    key = key_by_metric.get(metric)
    value = metrics_dict.get(key) if key else None
    return None if value is None else float(value)


def _default_output_dir(prediction_path: Path) -> Path:
    """Default report directory: sibling `evaluation/` under the prediction folder."""
    return prediction_path.parent / "evaluation"


def _resolve_chunk_num(chunk_num: int) -> int:
    """Resolve nonpositive chunk count from the available CPU allocation."""
    if chunk_num > 0:
        return chunk_num
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _report_name(volume_name: str, metric_tag: str) -> str:
    safe_volume = "".join(c if c.isalnum() or c in "._=-" else "-" for c in volume_name)
    safe_metric = "".join(c if c.isalnum() or c in "._=-" else "-" for c in metric_tag)
    return f"eval_{safe_volume}_{safe_metric}.txt"


def write_report(
    metrics_dict: dict[str, Any],
    *,
    prediction_path: Path,
    gt_path: Path,
    output_dir: Path,
    volume_name: str,
    metric_tag: str,
) -> Path:
    """Write an evaluation text report using the same sections as `just test`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / _report_name(volume_name, metric_tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "nerl_per_gt_erl" in metrics_dict:
        per_gt_erl_file = output_dir / f"{metrics_file.stem}_nerl_per_gt_erl.npz"
        np.savez_compressed(
            per_gt_erl_file,
            gt_segment_id=np.asarray(metrics_dict.get("nerl_gt_segment_ids", [])),
            erl=np.asarray(metrics_dict["nerl_per_gt_erl"], dtype=np.float64),
        )
        metrics_dict["nerl_per_gt_erl_file"] = str(per_gt_erl_file)

    with metrics_file.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("EVALUATION METRICS\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Timestamp: {timestamp}\n")
        handle.write(f"Volume: {volume_name}\n")
        handle.write(f"Prediction: {prediction_path}\n")
        handle.write(f"Ground Truth: {gt_path}\n")
        handle.write("=" * 80 + "\n\n")

        if "adapted_rand_error" in metrics_dict:
            handle.write("Instance Segmentation Metrics:\n")
            handle.write("-" * 80 + "\n")
            handle.write(
                f"  Adapted Rand Error:           {metrics_dict['adapted_rand_error']:.6f}\n"
            )
            if "voi_split" in metrics_dict:
                handle.write(f"  VOI Split:                    {metrics_dict['voi_split']:.6f}\n")
                handle.write(f"  VOI Merge:                    {metrics_dict['voi_merge']:.6f}\n")
                handle.write(f"  VOI Total:                    {metrics_dict['voi_total']:.6f}\n")
            if "instance_accuracy" in metrics_dict:
                handle.write(
                    f"  Instance Accuracy:            {metrics_dict['instance_accuracy']:.6f}\n"
                )
            if "instance_accuracy_detail" in metrics_dict:
                handle.write(
                    "\n"
                    f"  Instance Accuracy (Detail):   {metrics_dict['instance_accuracy_detail']:.6f}\n"
                )
                handle.write(
                    f"    Precision:                  {metrics_dict['instance_precision_detail']:.6f}\n"
                )
                handle.write(
                    f"    Recall:                     {metrics_dict['instance_recall_detail']:.6f}\n"
                )
                handle.write(
                    f"    F1:                         {metrics_dict['instance_f1_detail']:.6f}\n"
                )
            handle.write("\n")

        if "jaccard" in metrics_dict or "dice" in metrics_dict or "accuracy" in metrics_dict:
            handle.write("Binary/Semantic Segmentation Metrics:\n")
            handle.write("-" * 80 + "\n")
            if "jaccard" in metrics_dict:
                handle.write(f"  Jaccard Index:                {metrics_dict['jaccard']:.6f}\n")
            if "dice" in metrics_dict:
                handle.write(f"  Dice Score:                   {metrics_dict['dice']:.6f}\n")
            if "accuracy" in metrics_dict:
                handle.write(f"  Accuracy:                     {metrics_dict['accuracy']:.6f}\n")
            handle.write("\n")

        if "nerl" in metrics_dict:
            handle.write("Neurite ERL Metrics:\n")
            handle.write("-" * 80 + "\n")
            handle.write(f"  NERL:                         {metrics_dict['nerl']:.6f}\n")
            handle.write(f"  Pred ERL:                     {metrics_dict['nerl_pred_erl']:.6f}\n")
            handle.write(f"  GT ERL:                       {metrics_dict['nerl_gt_erl']:.6f}\n")
            handle.write(f"  Skeletons:                    {metrics_dict['nerl_num_skeletons']}\n")
            if "nerl_chunk_num" in metrics_dict:
                handle.write(f"  Chunk Num:                    {metrics_dict['nerl_chunk_num']}\n")
            handle.write(f"  Graph:                        {metrics_dict['nerl_graph']}\n")
            if "nerl_per_gt_erl_file" in metrics_dict:
                handle.write(
                    "  Per-GT ERL File:              " f"{metrics_dict['nerl_per_gt_erl_file']}\n"
                )
            handle.write("\n")

        handle.write("=" * 80 + "\n")

    return metrics_file


def evaluate_nerl(
    prediction: str,
    skeleton: str,
    *,
    prediction_dataset: str,
    skeleton_mask: str | None,
    skeleton_mask_dataset: str | None,
    resolution: list[float] | None,
    chunk_num: int,
    num_workers: int,
    merge_threshold: int,
    skeleton_id_attribute: str,
    skeleton_position_attribute: str,
    skeleton_edge_length_attribute: str,
    skeleton_position_order: str,
    prediction_position_order: str | None,
) -> dict[str, Any]:
    """Evaluate NERL by streaming labels at skeleton-node positions."""
    _, compute_erl_score, compute_segment_lut = import_em_erl()
    graph_options = NerlGraphOptions(
        skeleton_id_attribute=skeleton_id_attribute,
        skeleton_position_attribute=skeleton_position_attribute,
        skeleton_edge_length_attribute=skeleton_edge_length_attribute,
        skeleton_position_order=skeleton_position_order,
        prediction_position_order=prediction_position_order,
    )
    graph, voxel_coords = load_nerl_graph(skeleton, graph_options)
    if voxel_coords:
        node_positions = np.asarray(graph.node_coords_zyx, dtype=np.int64)
    else:
        node_positions = graph.get_nodes_position(resolution)

    resolved_chunk_num = _resolve_chunk_num(int(chunk_num))
    resolved_num_workers = _resolve_chunk_num(int(num_workers))
    node_segment_lut, mask_segment_id = compute_segment_lut(
        prediction,
        node_positions,
        mask=skeleton_mask,
        chunk_num=resolved_chunk_num,
        data_type=np.uint64,
        segment_dataset=prediction_dataset,
        mask_dataset=skeleton_mask_dataset,
        num_workers=resolved_num_workers,
    )
    score = compute_erl_score(
        graph,
        node_segment_lut,
        mask_segment_id,
        merge_threshold=int(merge_threshold),
    )
    score.compute_erl()
    pred_erl, gt_erl, num_skeletons, per_gt_erl = extract_nerl_score_outputs(score)
    nerl = pred_erl / gt_erl if gt_erl > 0 else float("nan")
    return {
        "nerl": float(nerl),
        "nerl_pred_erl": float(pred_erl),
        "nerl_gt_erl": float(gt_erl),
        "nerl_erl": float(pred_erl),
        "nerl_max_erl": float(gt_erl),
        "nerl_num_skeletons": int(num_skeletons),
        "nerl_graph": str(skeleton),
        "nerl_chunk_num": int(resolved_chunk_num),
        "nerl_gt_segment_ids": np.asarray(graph.skeleton_id),
        "nerl_per_gt_erl": per_gt_erl,
    }


def evaluate_dense_metrics(
    prediction: str,
    label: str,
    *,
    metrics: set[str],
    prediction_dataset: str,
    gt_dataset: str,
    prediction_threshold: float,
    instance_iou_threshold: float,
) -> dict[str, Any]:
    """Evaluate dense segmentation metrics by loading prediction and label volumes."""
    pred = read_volume(prediction, dataset=prediction_dataset)
    gt = read_volume(label, dataset=gt_dataset)

    pred_tensor = torch.from_numpy(np.asarray(pred))
    gt_tensor = torch.from_numpy(np.asarray(gt))
    pred_tensor, gt_tensor = align_metric_tensors(pred_tensor, gt_tensor)
    if pred_tensor is None or gt_tensor is None:
        raise ValueError(f"Cannot align prediction shape {pred.shape} with GT shape {gt.shape}")

    evaluation_cfg = SimpleNamespace(enabled=True, metrics=sorted(metrics))
    context = EvaluationContext(
        cfg=SimpleNamespace(),
        evaluation_cfg=evaluation_cfg,
        enabled=True,
        metrics={},
        device="cpu",
    )
    metrics_dict: dict[str, Any] = {}
    if metrics & INSTANCE_METRIC_NAMES:
        compute_instance_metrics(
            context,
            pred_tensor,
            gt_tensor,
            "",
            metrics_dict,
            instance_iou_threshold,
        )
    if metrics & BINARY_METRIC_NAMES:
        compute_binary_metrics(
            context,
            pred_tensor,
            gt_tensor,
            "",
            metrics_dict,
            prediction_threshold,
        )
    return metrics_dict


def evaluate_prediction(
    prediction: str,
    gt: str,
    metric: str,
    **kwargs: Any,
) -> tuple[dict[str, Any], Path]:
    """Evaluate one prediction file and write a report.

    Args:
        prediction: HDF5/Zarr/TIFF prediction path. HDF5 defaults to dataset `main`.
        gt: Dense label volume for dense metrics, or skeleton pickle/ERL npz for `nerl`.
        metric: Metric name or comma-separated metric names.
        output_dir: Optional report directory. Defaults to
            `<prediction parent>/evaluation`.

    Returns:
        `(metrics_dict, report_path)`.
    """
    metrics = parse_metrics(metric)
    prediction_path = Path(prediction)
    gt_path = Path(gt)
    volume_name = kwargs.get("volume_name") or prediction_path.stem
    output_dir = Path(kwargs.get("output_dir") or _default_output_dir(prediction_path))
    metric_tag = "-".join(sorted(metrics))

    if metrics == {"nerl"}:
        metrics_dict = evaluate_nerl(
            str(prediction_path),
            str(gt_path),
            prediction_dataset=kwargs.get("prediction_dataset", "main"),
            skeleton_mask=kwargs.get("skeleton_mask"),
            skeleton_mask_dataset=kwargs.get("skeleton_mask_dataset"),
            resolution=kwargs.get("resolution"),
            chunk_num=int(kwargs.get("chunk_num", -1)),
            num_workers=int(kwargs.get("num_workers", 1)),
            merge_threshold=int(kwargs.get("merge_threshold", 1)),
            skeleton_id_attribute=kwargs.get("skeleton_id_attribute", "id"),
            skeleton_position_attribute=kwargs.get(
                "skeleton_position_attribute",
                "index_position",
            ),
            skeleton_edge_length_attribute=kwargs.get(
                "skeleton_edge_length_attribute",
                "edge_length",
            ),
            skeleton_position_order=kwargs.get("skeleton_position_order", "xyz"),
            prediction_position_order=kwargs.get("prediction_position_order"),
        )
    else:
        metrics_dict = evaluate_dense_metrics(
            str(prediction_path),
            str(gt_path),
            metrics=metrics,
            prediction_dataset=kwargs.get("prediction_dataset", "main"),
            gt_dataset=kwargs.get("gt_dataset", "main"),
            prediction_threshold=float(kwargs.get("prediction_threshold", 0.5)),
            instance_iou_threshold=float(kwargs.get("instance_iou_threshold", 0.5)),
        )

    metrics_dict["volume_name"] = volume_name
    report_path = write_report(
        metrics_dict,
        prediction_path=prediction_path,
        gt_path=gt_path,
        output_dir=output_dir,
        volume_name=volume_name,
        metric_tag=metric_tag,
    )
    return metrics_dict, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved prediction volume against GT without running inference."
    )
    parser.add_argument("prediction", help="Prediction/segmentation file, usually HDF5")
    parser.add_argument("gt", help="Dense label GT, or skeleton .pkl/.npz for metric=nerl")
    parser.add_argument(
        "metric",
        help=(
            "Metric name, or comma-separated dense metrics. Supported: "
            f"{', '.join(sorted(SUPPORTED_METRICS))}"
        ),
    )
    parser.add_argument("--prediction-dataset", default="main", help="Prediction HDF5 dataset")
    parser.add_argument("--gt-dataset", default="main", help="Dense GT HDF5 dataset")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory (default: <prediction folder>/evaluation)",
    )
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON")

    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)

    parser.add_argument("--resolution", type=float, nargs=3, default=None)
    parser.add_argument(
        "--chunk-num",
        type=int,
        default=-1,
        help=(
            "Number of z chunks for NERL streaming. "
            "-1 uses SLURM_CPUS_PER_TASK, then os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help=(
            "Worker processes for NERL skeleton-point sampling. "
            "-1 uses SLURM_CPUS_PER_TASK, then os.cpu_count(); 1 = serial."
        ),
    )
    parser.add_argument("--merge-threshold", type=int, default=1)
    parser.add_argument("--skeleton-mask", default=None)
    parser.add_argument("--skeleton-mask-dataset", default=None)
    parser.add_argument("--skeleton-id-attribute", default="id")
    parser.add_argument("--skeleton-position-attribute", default="index_position")
    parser.add_argument("--skeleton-edge-length-attribute", default="edge_length")
    parser.add_argument("--skeleton-position-order", default="xyz")
    parser.add_argument("--prediction-position-order", default=None)
    return parser.parse_args()


def _json_safe_metrics(metrics_dict: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics_dict.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    metrics_dict, report_path = evaluate_prediction(
        args.prediction,
        args.gt,
        args.metric,
        prediction_dataset=args.prediction_dataset,
        gt_dataset=args.gt_dataset,
        output_dir=args.output_dir,
        prediction_threshold=args.prediction_threshold,
        instance_iou_threshold=args.instance_iou_threshold,
        resolution=args.resolution,
        chunk_num=args.chunk_num,
        num_workers=args.num_workers,
        merge_threshold=args.merge_threshold,
        skeleton_mask=args.skeleton_mask,
        skeleton_mask_dataset=args.skeleton_mask_dataset,
        skeleton_id_attribute=args.skeleton_id_attribute,
        skeleton_position_attribute=args.skeleton_position_attribute,
        skeleton_edge_length_attribute=args.skeleton_edge_length_attribute,
        skeleton_position_order=args.skeleton_position_order,
        prediction_position_order=args.prediction_position_order,
    )
    requested_metrics = parse_metrics(args.metric)

    if args.json:
        print(json.dumps(_json_safe_metrics(metrics_dict), indent=2, sort_keys=True))
    else:
        for metric in sorted(requested_metrics):
            value = _metric_value(metrics_dict, metric)
            if value is not None:
                print(f"{metric}: {value:.6f}")
        if "voi" in requested_metrics:
            print(f"voi_split: {metrics_dict['voi_split']:.6f}")
            print(f"voi_merge: {metrics_dict['voi_merge']:.6f}")
        if "nerl" in requested_metrics:
            print(f"pred_erl: {metrics_dict['nerl_pred_erl']:.6f}")
            print(f"gt_erl: {metrics_dict['nerl_gt_erl']:.6f}")
            print(f"skeletons: {metrics_dict['nerl_num_skeletons']}")
            print(f"chunk_num: {metrics_dict['nerl_chunk_num']}")
            if "nerl_per_gt_erl_file" in metrics_dict:
                print(f"per_gt: {metrics_dict['nerl_per_gt_erl_file']}")
        print(f"report: {report_path}")


if __name__ == "__main__":
    main()
