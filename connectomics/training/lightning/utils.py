"""
Utility functions for PyTorch Lightning training scripts.

This module provides helper functions for:
- Command-line argument parsing
- Configuration setup and validation
- File path expansion
- Checkpoint utilities
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def extract_best_score_from_checkpoint(ckpt_path: str, monitor_metric: str) -> Optional[float]:
    """
    Extract best score from checkpoint filename.

    Args:
        ckpt_path: Path to checkpoint file
        monitor_metric: Metric name to extract (e.g., 'train_loss_total_epoch', 'val/loss')

    Returns:
        Extracted score or None if not found
    """
    if not ckpt_path:
        return None

    filename = Path(ckpt_path).stem  # Get filename without extension

    # Replace '/' with underscore for metric name (e.g., 'val/loss' -> 'val_loss')
    metric_pattern = monitor_metric.replace("/", "_")

    # Try multiple patterns to extract the metric value:
    # 1. Full metric name: "train_loss_total_epoch=0.1234"
    # 2. Abbreviated in filename: "loss=0.1234" (when metric is "train_loss_total_epoch")
    # 3. Other common abbreviations

    patterns = [
        rf"{metric_pattern}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",  # Full name
    ]

    # Add abbreviated patterns by extracting the last part after '_' or '/'
    if "_" in monitor_metric or "/" in monitor_metric:
        short_name = monitor_metric.split("_")[-1].split("/")[-1]
        patterns.append(rf"{short_name}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def setup_seed_everything():
    """
    Return Lightning's canonical seed helper.

    Returns:
        seed_everything function
    """
    from pytorch_lightning import seed_everything

    return seed_everything


__all__ = [
    "extract_best_score_from_checkpoint",
    "setup_seed_everything",
]
