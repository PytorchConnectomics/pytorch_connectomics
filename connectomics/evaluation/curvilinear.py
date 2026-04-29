"""File-backed evaluation helpers for curvilinear structures."""

from __future__ import annotations

import functools
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

from ..metrics.metrics_skel import evaluate_image_pair

logger = logging.getLogger(__name__)


def evaluate_file_pair(
    pred_path: str,
    gt_path: str,
    threshold: int = 128,
    dilation_size: int = 5,
    verbose: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
    """Evaluate one prediction/ground-truth image pair."""
    if imageio is None:
        raise ImportError(
            "imageio is required for loading images. Install with: pip install imageio"
        )

    if not os.path.exists(pred_path):
        return None

    pred = imageio.imread(pred_path)
    gt = imageio.imread(gt_path)
    iou, correctness, completeness, quality = evaluate_image_pair(
        pred, gt, threshold, dilation_size
    )

    if verbose:
        logger.info(
            "%s: IoU=%.4f, Corr=%.4f, Comp=%.4f, Qual=%.4f",
            Path(pred_path).name,
            iou,
            correctness,
            completeness,
            quality,
        )

    return iou, correctness, completeness, quality


def evaluate_directory(
    pred_dir: str,
    gt_dir: str,
    pred_pattern: str = "%03d_pred.png",
    gt_pattern: str = "%03d.png",
    max_index: int = 200,
    threshold: int = 128,
    dilation_size: int = 5,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate all image pairs in two directories."""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    if verbose:
        logger.info("Evaluating with %d workers...", num_workers)

    pred_dir = pred_dir if pred_dir.endswith("/") else pred_dir + "/"
    gt_dir = gt_dir if gt_dir.endswith("/") else gt_dir + "/"
    file_pairs = [
        (pred_dir + (pred_pattern % i), gt_dir + (gt_pattern % i)) for i in range(max_index)
    ]

    eval_fn = functools.partial(
        evaluate_file_pair, threshold=threshold, dilation_size=dilation_size, verbose=verbose
    )
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(eval_fn, file_pairs)

    filtered = [r for r in results if r is not None]
    results_array = np.array(filtered)

    if not filtered:
        logger.warning("No valid results found.")
        return {
            "mean_iou": 0.0,
            "mean_correctness": 0.0,
            "mean_completeness": 0.0,
            "mean_quality": 0.0,
            "num_evaluated": 0,
            "results": results_array,
        }

    mean_metrics = results_array.mean(axis=0)
    output = {
        "mean_iou": mean_metrics[0],
        "mean_correctness": mean_metrics[1],
        "mean_completeness": mean_metrics[2],
        "mean_quality": mean_metrics[3],
        "num_evaluated": len(filtered),
        "results": results_array,
    }

    if verbose:
        logger.info("Evaluated %d images", output["num_evaluated"])
        logger.info("Mean IoU:          %.4f", output["mean_iou"])
        logger.info("Mean Correctness:  %.4f", output["mean_correctness"])
        logger.info("Mean Completeness: %.4f", output["mean_completeness"])
        logger.info("Mean Quality:      %.4f", output["mean_quality"])

    return output


__all__ = ["evaluate_file_pair", "evaluate_directory"]
