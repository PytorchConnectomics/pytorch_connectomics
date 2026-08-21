"""NERL evaluation-stage integration."""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config.pipeline.dict_utils import cfg_get
from ..metrics.nerl import NerlGraphOptions, compute_nerl_score_details
from ..metrics.oracle import oracle_merge_segmentation
from .context import EvaluationContext

logger = logging.getLogger(__name__)


def cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    return cfg_get(cfg, name, default)


def select_volume_config_value(value: Any, volume_name: str | None) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return value
    if isinstance(value, dict):
        if volume_name and volume_name in value:
            return value[volume_name]
        if "default" in value:
            return value["default"]
        if len(value) == 1:
            return next(iter(value.values()))
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else None
    return value


def _nerl_graph_options(evaluation_cfg: Any) -> NerlGraphOptions:
    return NerlGraphOptions(
        skeleton_id_attribute=cfg_value(evaluation_cfg, "nerl_skeleton_id_attribute", "id"),
        skeleton_position_attribute=cfg_value(
            evaluation_cfg,
            "nerl_skeleton_position_attribute",
            "index_position",
        ),
        skeleton_edge_length_attribute=cfg_value(
            evaluation_cfg,
            "nerl_skeleton_edge_length_attribute",
            "edge_length",
        ),
        skeleton_position_order=cfg_value(
            evaluation_cfg,
            "nerl_skeleton_position_order",
            "xyz",
        ),
        prediction_position_order=cfg_value(
            evaluation_cfg,
            "nerl_prediction_position_order",
            None,
        ),
    )


def _resolve_num_workers(value: int) -> int:
    """Resolve ``nerl_num_workers``: ``-1`` means use all available CPUs.

    Why: ``multiprocessing.cpu_count()`` returns the host's total cores, ignoring
    slurm/cgroup CPU pinning. Under ``srun --cpus-per-task=N`` that oversubscribes
    the allocation. ``os.sched_getaffinity(0)`` respects the cgroup mask.
    """
    if value < 0:
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            return max(1, multiprocessing.cpu_count())
    return max(1, value)


def _nerl_resolution(context: EvaluationContext, evaluation_cfg: Any) -> Any:
    resolution = cfg_value(evaluation_cfg, "nerl_resolution", None)
    if resolution is not None:
        return resolution
    data_cfg = getattr(context.cfg, "data", None)
    test_cfg = getattr(data_cfg, "test", None)
    return getattr(test_cfg, "resolution", None)


def _resolve_graph_value(
    graph_value: Any,
    label_value: Any,
    resolution: Any,
    volume_prefix: str,
) -> Any:
    """Return the skeleton graph to score against, deriving it if necessary.

    ``data.test.skeleton`` may be omitted, or name an ``.erlgraph.npz`` that does
    not exist yet: in both cases the graph is built from the dense
    ``data.test.label`` volume (kimimaro at the default downsample) and cached
    next to it, so a run needs only the segmentation ground truth.
    """
    if not label_value or not resolution:
        return graph_value

    label_path = Path(str(label_value))
    if graph_value:
        npz_path = Path(str(graph_value))
        if npz_path.exists() or npz_path.suffix != ".npz":
            return graph_value
    else:
        # <label stem>.erlgraph.npz next to the labels.
        npz_path = label_path.with_name(f"{label_path.stem}.erlgraph.npz")
        if npz_path.exists():
            return str(npz_path)

    if not label_path.exists():
        return graph_value

    from ..metrics.skeleton_build import ensure_erl_graph

    logger.info(
        "%sderiving the ERL skeleton graph from %s (this is cached at %s)",
        volume_prefix,
        label_path.name,
        npz_path,
    )
    return str(ensure_erl_graph(npz_path, label_path, resolution))


def compute_nerl_metrics(
    context: EvaluationContext,
    decoded_predictions: np.ndarray,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    volume_name: str | None,
    dense_labels: Any = None,
) -> None:
    evaluation_cfg = context.evaluation_cfg
    test_data_cfg = getattr(getattr(context.cfg, "data", None), "test", None)
    graph_value = _resolve_graph_value(
        select_volume_config_value(getattr(test_data_cfg, "skeleton", None), volume_name),
        select_volume_config_value(getattr(test_data_cfg, "label", None), volume_name),
        _nerl_resolution(context, evaluation_cfg),
        volume_prefix,
    )
    if graph_value is None:
        logger.warning(
            "%sSkipping NERL: set data.test.skeleton to an ERLGraph .npz or "
            "BANIS/NISB skeleton.pkl, or data.test.label to a dense volume to "
            "derive one from",
            volume_prefix,
        )
        return

    mask_value = select_volume_config_value(
        getattr(test_data_cfg, "skeleton_mask", None),
        volume_name,
    )
    score_kwargs = {
        "skeleton_mask_value": mask_value,
        "resolution": _nerl_resolution(context, evaluation_cfg),
        "merge_threshold": int(cfg_value(evaluation_cfg, "nerl_merge_threshold", 50)),
        "chunk_num": int(cfg_value(evaluation_cfg, "nerl_chunk_num", 1)),
        "num_workers": _resolve_num_workers(int(cfg_value(evaluation_cfg, "nerl_num_workers", -1))),
        "graph_options": _nerl_graph_options(evaluation_cfg),
    }
    if context.metric_requested("nerl"):
        result = compute_nerl_score_details(
            decoded_predictions,
            graph_value,
            **score_kwargs,
        )

        logger.info("%sNERL: %.6f", volume_prefix, result.nerl)
        logger.info("%s  Pred ERL: %.6f", volume_prefix, result.pred_erl)
        logger.info("%s  GT ERL: %.6f", volume_prefix, result.gt_erl)
        logger.info("%s  # Skeletons: %d", volume_prefix, result.num_skeletons)

        metrics_dict["nerl"] = result.nerl
        metrics_dict["nerl_pred_erl"] = result.pred_erl
        metrics_dict["nerl_gt_erl"] = result.gt_erl
        metrics_dict["nerl_erl"] = result.pred_erl
        metrics_dict["nerl_max_erl"] = result.gt_erl
        metrics_dict["nerl_num_skeletons"] = result.num_skeletons
        metrics_dict["nerl_graph"] = str(graph_value)
        metrics_dict["nerl_gt_segment_ids"] = np.asarray(result.graph.skeleton_id)
        metrics_dict["nerl_per_gt_erl"] = result.per_gt_erl

        try:
            context.log_metric(
                "test_nerl",
                float(result.nerl),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=1,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to log test_nerl metric: %s", exc)

    if not context.metric_requested("nerl_oracle_merge") or dense_labels is None:
        return

    if hasattr(dense_labels, "detach"):
        dense_labels = dense_labels.detach().cpu().numpy()
    dense_labels = np.asarray(dense_labels)
    while dense_labels.ndim > decoded_predictions.ndim and dense_labels.shape[0] == 1:
        dense_labels = dense_labels[0]
    if dense_labels.shape != decoded_predictions.shape:
        logger.warning(
            "%sSkipping NERL oracle-merge: prediction/label shapes differ (%s vs %s)",
            volume_prefix,
            decoded_predictions.shape,
            dense_labels.shape,
        )
        return

    oracle_segmentation = oracle_merge_segmentation(
        decoded_predictions,
        dense_labels,
    )
    oracle_result = compute_nerl_score_details(
        oracle_segmentation,
        graph_value,
        **score_kwargs,
    )
    logger.info(
        "%sNERL oracle-merge: %.6f",
        volume_prefix,
        oracle_result.nerl,
    )
    metrics_dict["nerl_oracle_merge"] = oracle_result.nerl
    metrics_dict["nerl_oracle_merge_pred_erl"] = oracle_result.pred_erl

    try:
        context.log_metric(
            "test_nerl_oracle_merge",
            float(oracle_result.nerl),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to log test_nerl_oracle_merge metric: %s", exc)


__all__ = [
    "compute_nerl_metrics",
    "cfg_value",
    "select_volume_config_value",
]
