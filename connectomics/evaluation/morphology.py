"""Evaluation adapters and artifacts for morphology estimates without ground truth."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from ..config.pipeline.dict_utils import as_plain_dict
from ..metrics.morphology import (
    MorphologyAnalysis,
    MorphologyConfig,
    MorphologyRecord,
    analyze_morphology,
)
from .context import EvaluationContext

logger = logging.getLogger(__name__)


def _morphology_config(context: EvaluationContext) -> MorphologyConfig:
    options = as_plain_dict(context.cfg_value(context.evaluation_cfg, "morphology", None))
    spacing = options.pop("voxel_size_um", None)
    if spacing is None:
        data_cfg = context.cfg_value(context.cfg, "data", None)
        test_cfg = context.cfg_value(data_cfg, "test", None)
        resolution = context.cfg_value(test_cfg, "resolution", None)
        if resolution is None:
            raise ValueError(
                "Morphology evaluation requires evaluation.morphology.voxel_size_um "
                "or data.test.resolution in nm, in z,y,x order"
            )
        spacing = [float(value) / 1000.0 for value in resolution]
    if "radius_bins_um" in options:
        options["radius_bins_um"] = tuple(options["radius_bins_um"])
    return MorphologyConfig(voxel_size_um=tuple(spacing), **options)


def format_morphology_report(analysis: MorphologyAnalysis) -> str:
    """Format measured geometry and heuristic estimates with their denominators."""
    lines = [
        "Morphology estimates without ground truth (physical lengths in um).",
        "Candidate classes and merge flags are geometric estimates; NERL requires ground truth.",
    ]
    for name, value in analysis.summary.items():
        label = name.replace("_", " ")
        if isinstance(value, dict):
            lines.append(f"  {label}:")
            lines.extend(
                f"    {key.replace('_', ' ')}: {json.dumps(item, allow_nan=False)}"
                for key, item in value.items()
            )
        elif isinstance(value, (list, tuple)):
            lines.append(f"  {label}:")
            lines.extend(f"    {json.dumps(item, allow_nan=False)}" for item in value)
        else:
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def write_morphology_artifacts(
    analysis: MorphologyAnalysis,
    output_dir: str | Path,
    stem: str = "morphology",
) -> dict[str, Path]:
    """Write a JSON summary and a CSV row per analyzed instance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / f"{stem}_summary.json",
        "instances": output_dir / f"{stem}_instances.csv",
    }
    payload = {
        "volume_shape": analysis.volume_shape,
        "config": asdict(analysis.config),
        "summary": analysis.summary,
    }
    paths["summary"].write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["instances"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[item.name for item in fields(MorphologyRecord)])
        writer.writeheader()
        for record in analysis.records:
            writer.writerow(
                {
                    name: (
                        json.dumps(value, allow_nan=False)
                        if isinstance(value, (tuple, list, dict))
                        else value
                    )
                    for name, value in asdict(record).items()
                }
            )
    return paths


def compute_morphology_metrics(
    context: EvaluationContext,
    decoded_predictions: np.ndarray,
    volume_prefix: str,
    metrics_dict: dict[str, Any],
) -> MorphologyAnalysis:
    """Append physical no-GT morphology measurements for one decoded volume."""
    segmentation = np.asarray(decoded_predictions)
    while segmentation.ndim > 3 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    analysis = analyze_morphology(segmentation, _morphology_config(context))
    for name, value in analysis.summary.items():
        metrics_dict[f"morphology_{name}"] = value
    metrics_dict["morphology_report"] = format_morphology_report(analysis)
    metrics_dict["_morphology_analysis"] = analysis
    logger.info("%s%s", volume_prefix, metrics_dict["morphology_report"])
    return analysis


__all__ = [
    "compute_morphology_metrics",
    "format_morphology_report",
    "write_morphology_artifacts",
]
