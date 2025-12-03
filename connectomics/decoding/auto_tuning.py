"""
Hyperparameter auto-tuning for segmentation post-processing.

This module provides Optuna-based parameter optimization for connectomics
segmentation tasks, particularly affinity threshold tuning using skeleton
metrics (NERL, VOI) from the NISB benchmark.

Reference: BANIS baseline (threshold sweep utilities)
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Optional, Union
from pathlib import Path
import pickle
import warnings

import numpy as np

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from funlib.evaluate import rand_voi, expected_run_length
    from networkx import get_node_attributes

    FUNLIB_AVAILABLE = True
except ImportError:
    FUNLIB_AVAILABLE = False


__all__ = [
    "optimize_threshold",
    "optimize_parameters",
    "grid_search_threshold",
    "SkeletonMetrics",
]


class SkeletonMetrics:
    """
    Compute skeleton-based metrics for neuron segmentation evaluation.

    Uses funlib.evaluate to compute NERL (Normalized Expected Run Length)
    and VOI (Variation of Information) metrics on skeleton graphs.

    Args:
        skeleton_path: Path to skeleton file (.pkl with NetworkX graph)

    Example:
        >>> metrics = SkeletonMetrics("skeleton.pkl")
        >>> segmentation = decode_affinity_cc(affinities, threshold=0.5)
        >>> results = metrics.compute(segmentation)
        >>> print(f"NERL: {results['nerl']:.3f}")
        >>> print(f"VOI: {results['voi_sum']:.3f}")

    Note:
        Requires funlib.evaluate:
        pip install git+https://github.com/funkelab/funlib.evaluate.git
    """

    def __init__(self, skeleton_path: Union[str, Path]):
        if not FUNLIB_AVAILABLE:
            raise ImportError(
                "funlib.evaluate not found. Install with:\n"
                "pip install git+https://github.com/funkelab/funlib.evaluate.git"
            )

        self.skeleton_path = Path(skeleton_path)
        if not self.skeleton_path.exists():
            raise FileNotFoundError(f"Skeleton not found: {skeleton_path}")

        # Load skeleton
        with open(self.skeleton_path, "rb") as f:
            self.skeleton = pickle.load(f)

    def compute(self, pred_seg: np.ndarray, return_details: bool = False) -> Dict[str, Any]:
        """
        Compute skeleton-based metrics.

        Args:
            pred_seg: Predicted segmentation (D, H, W)
            return_details: Include detailed merge/split statistics

        Returns:
            Dictionary with metrics:
                - nerl: Normalized Expected Run Length (0-1, higher better)
                - erl: Expected Run Length
                - max_erl: Maximum possible ERL
                - voi_sum: Variation of Information (lower better)
                - voi_split: VOI split component
                - voi_merge: VOI merge component
                - n_mergers: Number of merge errors
                - n_non0_mergers: Mergers excluding background
                - n_splits: Number of split errors
        """
        # Assign predicted IDs to skeleton nodes
        for node in self.skeleton.nodes:
            x, y, z = self.skeleton.nodes[node]["index_position"]
            self.skeleton.nodes[node]["pred_id"] = pred_seg[x, y, z]

        # Compute VOI
        gt_ids = np.array(list(get_node_attributes(self.skeleton, "id").values())).astype(np.uint64)
        pred_ids = np.array(list(get_node_attributes(self.skeleton, "pred_id").values())).astype(
            np.uint64
        )

        voi_report = rand_voi(
            gt_ids,
            pred_ids,
            return_cluster_scores=False,
        )

        # Compute ERL
        erl_report = expected_run_length(
            self.skeleton,
            "id",
            "edge_length",
            get_node_attributes(self.skeleton, "pred_id"),
            skeleton_position_attributes=["nm_position"],
            return_merge_split_stats=True,
        )

        # Compute max ERL (perfect segmentation)
        max_erl_report = expected_run_length(
            self.skeleton,
            "id",
            "edge_length",
            get_node_attributes(self.skeleton, "id"),
            skeleton_position_attributes=["nm_position"],
            return_merge_split_stats=False,
        )

        # Extract metrics
        erl = erl_report[0]
        max_erl = max_erl_report
        nerl = erl / max_erl

        merge_stats = erl_report[1]["merge_stats"]
        split_stats = erl_report[1]["split_stats"]

        # Count errors
        n_mergers = sum(len(v) for v in merge_stats.values())
        merge_stats_no_bg = {k: v for k, v in merge_stats.items() if k not in [0, 0.0]}
        n_non0_mergers = sum(len(v) for v in merge_stats_no_bg.values())
        n_splits = sum(len(v) for v in split_stats.values())

        metrics = {
            "nerl": nerl,
            "erl": erl,
            "max_erl": max_erl,
            "voi_sum": voi_report["voi_split"] + voi_report["voi_merge"],
            "voi_split": voi_report["voi_split"],
            "voi_merge": voi_report["voi_merge"],
            "n_mergers": n_mergers,
            "n_non0_mergers": n_non0_mergers,
            "n_splits": n_splits,
        }

        if return_details:
            metrics["merge_stats"] = merge_stats
            metrics["split_stats"] = split_stats
            metrics["voi_report"] = voi_report

        return metrics


def grid_search_threshold(
    affinities: np.ndarray,
    skeleton_path: Union[str, Path],
    thresholds: Optional[np.ndarray] = None,
    metric: str = "nerl",
    verbose: bool = True,
    segmentation_fn: Optional[Callable] = None,
    **seg_kwargs,
) -> Dict[str, Any]:
    """
    Grid search over thresholds to find optimal affinity threshold.

    This function systematically evaluates segmentation quality across different
    affinity thresholds using skeleton-based metrics.

    Args:
        affinities: Affinity predictions (3, D, H, W) or (6, D, H, W)
        skeleton_path: Path to skeleton .pkl file
        thresholds: Array of thresholds to try. If None, uses sigmoid-spaced
            thresholds: 1/(1+exp(-x)) for x in [-1, -0.8, ..., 2.2]
        metric: Metric to optimize ("nerl" or "voi_sum")
        verbose: Print progress
        segmentation_fn: Custom segmentation function. If None, uses decode_affinity_cc
        **seg_kwargs: Additional arguments for segmentation function

    Returns:
        Dictionary with:
            - best_threshold: Optimal threshold value
            - best_metrics: Metrics at best threshold
            - all_results: List of metrics for all thresholds

    Example:
        >>> result = grid_search_threshold(
        ...     affinities,
        ...     "skeleton.pkl",
        ...     metric="nerl"
        ... )
        >>> print(f"Best threshold: {result['best_threshold']:.3f}")
        >>> print(f"Best NERL: {result['best_metrics']['nerl']:.3f}")

    See Also:
        - :func:`optimize_threshold`: Optuna-based optimization (more efficient)
        - :func:`decode_affinity_cc`: Default segmentation function
    """
    from .segmentation import decode_affinity_cc

    if segmentation_fn is None:
        segmentation_fn = decode_affinity_cc

    if thresholds is None:
        # Default: sigmoid-spaced thresholds
        logits = np.arange(-1.0, 2.4, 0.2)
        thresholds = 1.0 / (1.0 + np.exp(-logits))

    metrics_calc = SkeletonMetrics(skeleton_path)

    best_value = -np.inf if metric == "nerl" else np.inf
    best_threshold = None
    best_metrics = None
    all_results = []

    if verbose:
        print(f"Grid search over {len(thresholds)} thresholds...")
        print(f"Optimizing metric: {metric}")

    for i, thr in enumerate(thresholds):
        # Convert affinities to segmentation
        seg = segmentation_fn(affinities, threshold=thr, **seg_kwargs)

        # Compute metrics
        metrics_dict = metrics_calc.compute(seg)
        metrics_dict["threshold"] = float(thr)
        all_results.append(metrics_dict)

        # Check if best
        current_value = metrics_dict[metric]
        is_better = (
            (current_value > best_value) if metric == "nerl" else (current_value < best_value)
        )

        if is_better:
            best_value = current_value
            best_threshold = thr
            best_metrics = metrics_dict

        if verbose and (i + 1) % 5 == 0:
            print(
                f"  [{i+1}/{len(thresholds)}] " f"threshold={thr:.3f}, {metric}={current_value:.4f}"
            )

    if verbose:
        print(f"\nBest threshold: {best_threshold:.3f}")
        print(f"Best {metric}: {best_value:.4f}")

    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "all_results": all_results,
    }


def optimize_threshold(
    affinities: np.ndarray,
    skeleton_path: Union[str, Path],
    n_trials: int = 50,
    metric: str = "nerl",
    threshold_range: tuple = (0.1, 0.9),
    verbose: bool = True,
    segmentation_fn: Optional[Callable] = None,
    study_name: Optional[str] = None,
    **seg_kwargs,
) -> Dict[str, Any]:
    """
    Optimize affinity threshold using Optuna (Bayesian optimization).

    This function uses Optuna's Tree-structured Parzen Estimator (TPE) sampler
    to efficiently search for the optimal affinity threshold. More efficient
    than grid search for large search spaces.

    Args:
        affinities: Affinity predictions (3, D, H, W) or (6, D, H, W)
        skeleton_path: Path to skeleton .pkl file
        n_trials: Number of optimization trials
        metric: Metric to optimize ("nerl" or "voi_sum")
        threshold_range: (min, max) threshold range to search
        verbose: Show Optuna progress
        segmentation_fn: Custom segmentation function. If None, uses decode_affinity_cc
        study_name: Name for Optuna study (for logging)
        **seg_kwargs: Additional arguments for segmentation function

    Returns:
        Dictionary with:
            - best_threshold: Optimal threshold value
            - best_metrics: Metrics at best threshold
            - study: Optuna study object (for further analysis)

    Example:
        >>> result = optimize_threshold(
        ...     affinities,
        ...     "skeleton.pkl",
        ...     n_trials=50,
        ...     metric="nerl"
        ... )
        >>> print(f"Best threshold: {result['best_threshold']:.3f}")
        >>> print(f"Best NERL: {result['best_metrics']['nerl']:.3f}")
        >>>
        >>> # Access optimization history
        >>> study = result['study']
        >>> print(f"Number of trials: {len(study.trials)}")

    Note:
        Requires optuna: pip install optuna>=3.0.0

    See Also:
        - :func:`grid_search_threshold`: Exhaustive grid search
        - :func:`optimize_parameters`: Multi-parameter optimization
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not found. Install with: pip install optuna>=3.0.0")

    from .segmentation import decode_affinity_cc

    if segmentation_fn is None:
        segmentation_fn = decode_affinity_cc

    metrics_calc = SkeletonMetrics(skeleton_path)

    # Determine optimization direction
    direction = "maximize" if metric == "nerl" else "minimize"

    def objective(trial):
        """Optuna objective function."""
        # Suggest threshold
        threshold = trial.suggest_float("threshold", threshold_range[0], threshold_range[1])

        # Convert affinities to segmentation
        seg = segmentation_fn(affinities, threshold=threshold, **seg_kwargs)

        # Compute metrics
        metrics_dict = metrics_calc.compute(seg)

        # Return metric to optimize
        return metrics_dict[metric]

    # Create study
    if study_name is None:
        study_name = f"affinity_threshold_{metric}"

    study = optuna.create_study(
        direction=direction, study_name=study_name, sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    if verbose:
        print(f"Optimizing threshold with Optuna ({n_trials} trials)...")
        print(f"Metric: {metric} ({direction})")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get best parameters
    best_threshold = study.best_params["threshold"]

    # Compute full metrics for best threshold
    best_seg = segmentation_fn(affinities, threshold=best_threshold, **seg_kwargs)
    best_metrics = metrics_calc.compute(best_seg)
    best_metrics["threshold"] = best_threshold

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Best threshold: {best_threshold:.3f}")
        print(f"Best {metric}: {study.best_value:.4f}")

    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "study": study,
    }


def optimize_parameters(
    affinities: np.ndarray,
    skeleton_path: Union[str, Path],
    param_space: Dict[str, tuple],
    n_trials: int = 100,
    metric: str = "nerl",
    verbose: bool = True,
    segmentation_fn: Optional[Callable] = None,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optimize multiple post-processing parameters using Optuna.

    Extends single-parameter optimization to jointly optimize multiple
    parameters (e.g., threshold, small object removal, morphological ops).

    Args:
        affinities: Affinity predictions (3, D, H, W) or (6, D, H, W)
        skeleton_path: Path to skeleton .pkl file
        param_space: Dictionary mapping parameter names to (min, max) ranges.
            Example: {"threshold": (0.1, 0.9), "thres_small": (50, 500)}
        n_trials: Number of optimization trials
        metric: Metric to optimize ("nerl" or "voi_sum")
        verbose: Show Optuna progress
        segmentation_fn: Custom segmentation function. If None, uses decode_affinity_cc
        study_name: Name for Optuna study

    Returns:
        Dictionary with:
            - best_params: Optimal parameter values
            - best_metrics: Metrics at best parameters
            - study: Optuna study object

    Example:
        >>> param_space = {
        ...     "threshold": (0.1, 0.9),
        ...     "thres_small": (50, 500),
        ... }
        >>> result = optimize_parameters(
        ...     affinities,
        ...     "skeleton.pkl",
        ...     param_space,
        ...     n_trials=100
        ... )
        >>> print(f"Best params: {result['best_params']}")
        >>> print(f"Best NERL: {result['best_metrics']['nerl']:.3f}")

    See Also:
        - :func:`optimize_threshold`: Single-parameter optimization
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not found. Install with: pip install optuna>=3.0.0")

    from .segmentation import decode_affinity_cc

    if segmentation_fn is None:
        segmentation_fn = decode_affinity_cc

    metrics_calc = SkeletonMetrics(skeleton_path)

    # Determine optimization direction
    direction = "maximize" if metric == "nerl" else "minimize"

    def objective(trial):
        """Optuna objective function."""
        # Suggest parameters
        params = {}
        for param_name, (min_val, max_val) in param_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            else:
                params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))

        # Convert affinities to segmentation with suggested parameters
        try:
            seg = segmentation_fn(affinities, **params)
        except Exception as e:
            # If parameters cause an error, return worst possible value
            warnings.warn(f"Trial failed with params {params}: {e}")
            return -np.inf if direction == "maximize" else np.inf

        # Compute metrics
        metrics_dict = metrics_calc.compute(seg)

        # Return metric to optimize
        return metrics_dict[metric]

    # Create study
    if study_name is None:
        study_name = f"affinity_multiparams_{metric}"

    study = optuna.create_study(
        direction=direction, study_name=study_name, sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    if verbose:
        print(f"Optimizing {len(param_space)} parameters with Optuna...")
        print(f"Parameters: {list(param_space.keys())}")
        print(f"Metric: {metric} ({direction})")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get best parameters
    best_params = study.best_params

    # Compute full metrics for best parameters
    best_seg = segmentation_fn(affinities, **best_params)
    best_metrics = metrics_calc.compute(best_seg)
    best_metrics.update(best_params)

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"Best {metric}: {study.best_value:.4f}")

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "study": study,
    }
