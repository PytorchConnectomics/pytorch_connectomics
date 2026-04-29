"""Decode experiment logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from connectomics.runtime.output_naming import tta_cache_suffix

from .pipeline import normalize_decode_modes, resolve_decode_modes_from_cfg

logger = logging.getLogger(__name__)


def log_decode_experiment(
    *,
    cfg,
    output_dir: str | Path,
    volume_name: str,
    timestamp: str,
    metrics_dict: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> None:
    """Append decode parameters and metrics to ``decode_experiments.tsv``."""
    decode_modes = resolve_decode_modes_from_cfg(cfg)
    if not decode_modes:
        return

    tsv_path = Path(output_dir) / "decode_experiments.tsv"

    steps = normalize_decode_modes(decode_modes)
    decode_params: dict[str, Any] = {}
    for step in steps:
        decode_params["decoder"] = step.name
        decode_params.update(step.kwargs)

    input_tta_prediction_name = (
        f"{volume_name}{tta_cache_suffix(cfg, checkpoint_path=checkpoint_path)}"
    )

    param_keys = [
        "decoder",
        "thresholds",
        "merge_function",
        "aff_threshold",
        "channel_order",
    ]
    if decode_params.get("use_aff_uint8"):
        param_keys.append("use_aff_uint8")
    if decode_params.get("use_seg_uint32"):
        param_keys.append("use_seg_uint32")
    if decode_params.get("compute_fragments"):
        param_keys += ["compute_fragments", "seed_method"]
    if decode_params.get("boundary_threshold", 0) > 0:
        param_keys.append("boundary_threshold")
    if decode_params.get("dust_merge") and decode_params.get("dust_merge_size", 0) > 0:
        param_keys += ["dust_merge_size", "dust_merge_affinity", "dust_remove_size"]
    if decode_params.get("branch_merge"):
        param_keys += [
            "branch_merge",
            "iou_threshold",
            "best_buddy",
            "one_sided_threshold",
            "one_sided_min_size",
        ]
    metric_keys = [
        "adapted_rand_error",
        "voi_split",
        "voi_merge",
        "voi_total",
        "instance_precision_detail",
        "instance_recall_detail",
        "instance_f1_detail",
        "nerl",
        "nerl_pred_erl",
        "nerl_gt_erl",
        "nerl_erl",
        "nerl_max_erl",
        "nerl_num_skeletons",
    ]

    header_cols = ["timestamp", "volume", "input_tta_prediction_name"] + param_keys + metric_keys
    row_vals = [timestamp, volume_name, input_tta_prediction_name]
    for key in param_keys:
        row_vals.append(str(decode_params.get(key, "")))
    for key in metric_keys:
        value = metrics_dict.get(key)
        row_vals.append(f"{value:.6f}" if isinstance(value, float) else str(value or ""))

    try:
        if tsv_path.exists():
            with open(tsv_path) as f:
                existing_header = f.readline().rstrip("\n").split("\t")
            col_set = set(existing_header)
            merged_header = list(existing_header)
            for column in header_cols:
                if column not in col_set:
                    merged_header.append(column)
            row_dict = dict(zip(header_cols, row_vals))
            aligned_row = [row_dict.get(column, "") for column in merged_header]
            if len(merged_header) > len(existing_header):
                content = tsv_path.read_text()
                lines = content.split("\n")
                lines[0] = "\t".join(merged_header)
                with open(tsv_path, "w") as f:
                    f.write("\n".join(lines))
                    if not content.endswith("\n"):
                        f.write("\n")
            with open(tsv_path, "a") as f:
                f.write("\t".join(aligned_row) + "\n")
        else:
            with open(tsv_path, "w") as f:
                f.write("\t".join(header_cols) + "\n")
                f.write("\t".join(row_vals) + "\n")
        logger.info("Decode experiment logged to: %s", tsv_path)
    except Exception as exc:
        logger.warning("Failed to log decode experiment: %s", exc)


__all__ = ["log_decode_experiment"]
