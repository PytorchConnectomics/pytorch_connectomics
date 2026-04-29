"""Neurite ERL evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def module_cfg_value(module, cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(module, "_cfg_value"):
        return module._cfg_value(cfg, name, default)
    return getattr(cfg, name, default)


def get_effective_evaluation_config(module) -> Any:
    evaluation_cfg = module._get_test_evaluation_config()
    if evaluation_cfg is not None:
        return evaluation_cfg
    return getattr(getattr(module, "cfg", None), "evaluation", None)


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


def import_em_erl():
    try:
        from em_erl import ERLGraph, compute_erl_score, compute_segment_lut

        return ERLGraph, compute_erl_score, compute_segment_lut
    except ModuleNotFoundError:
        import sys

        repo_root = Path(__file__).resolve().parents[2]
        em_erl_root = repo_root / "lib" / "em_erl"
        if em_erl_root.exists():
            sys.path.insert(0, str(em_erl_root))
        from em_erl import ERLGraph, compute_erl_score, compute_segment_lut

        return ERLGraph, compute_erl_score, compute_segment_lut


def reorder_coordinate_axes(
    coords: np.ndarray,
    *,
    source_order: str,
    target_order: str | None,
) -> np.ndarray:
    source_order = str(source_order).lower()
    target_order = source_order if target_order is None else str(target_order).lower()
    valid_axes = {"x", "y", "z"}
    if len(source_order) != 3 or set(source_order) != valid_axes:
        raise ValueError(f"Invalid skeleton coordinate order: {source_order!r}")
    if len(target_order) != 3 or set(target_order) != valid_axes:
        raise ValueError(f"Invalid prediction coordinate order: {target_order!r}")
    axis_indices = [source_order.index(axis) for axis in target_order]
    return np.asarray(coords)[:, axis_indices]


def networkx_skeleton_to_erl_graph(skeleton: Any, evaluation_cfg: Any, module: Any):
    ERLGraph, _, _ = import_em_erl()

    id_attr = module_cfg_value(module, evaluation_cfg, "nerl_skeleton_id_attribute", "id")
    pos_attr = module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_position_attribute",
        "index_position",
    )
    edge_len_attr = module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_edge_length_attribute",
        "edge_length",
    )
    source_order = module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_position_order",
        "xyz",
    )
    target_order = module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_prediction_position_order",
        None,
    )

    node_ids = list(skeleton.nodes)
    if not node_ids:
        raise ValueError("NERL skeleton has no nodes")

    raw_skeleton_ids = []
    node_coords = []
    for node_id in node_ids:
        node_data = skeleton.nodes[node_id]
        raw_skeleton_ids.append(node_data[id_attr])
        node_coords.append(node_data[pos_attr])

    skeleton_ids = list(dict.fromkeys(raw_skeleton_ids))
    skeleton_index_by_id = {skeleton_id: i for i, skeleton_id in enumerate(skeleton_ids)}
    node_index_by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    node_skeleton_index = np.asarray(
        [skeleton_index_by_id[skeleton_id] for skeleton_id in raw_skeleton_ids],
        dtype=np.uint32,
    )
    node_coords_arr = reorder_coordinate_axes(
        np.asarray(node_coords, dtype=np.float32),
        source_order=source_order,
        target_order=target_order,
    )

    edge_buckets: list[list[tuple[int, int, float]]] = [[] for _ in skeleton_ids]
    skeleton_len = np.zeros(len(skeleton_ids), dtype=np.float64)
    for u, v, edge_data in skeleton.edges(data=True):
        if u not in node_index_by_id or v not in node_index_by_id:
            continue
        u_idx = node_index_by_id[u]
        v_idx = node_index_by_id[v]
        skel_idx = int(node_skeleton_index[u_idx])
        if skel_idx != int(node_skeleton_index[v_idx]):
            continue
        if edge_len_attr in edge_data:
            edge_len = float(edge_data[edge_len_attr])
        else:
            edge_len = float(np.linalg.norm(node_coords_arr[u_idx] - node_coords_arr[v_idx]))
        edge_buckets[skel_idx].append((u_idx, v_idx, edge_len))
        skeleton_len[skel_idx] += edge_len

    edge_ptr = [0]
    edge_u = []
    edge_v = []
    edge_len = []
    for bucket in edge_buckets:
        for u_idx, v_idx, length in bucket:
            edge_u.append(u_idx)
            edge_v.append(v_idx)
            edge_len.append(length)
        edge_ptr.append(len(edge_u))

    return ERLGraph(
        skeleton_id=np.asarray(skeleton_ids),
        skeleton_len=skeleton_len,
        node_skeleton_index=node_skeleton_index,
        node_coords_zyx=node_coords_arr,
        edge_u=np.asarray(edge_u, dtype=np.uint32),
        edge_v=np.asarray(edge_v, dtype=np.uint32),
        edge_len=np.asarray(edge_len, dtype=np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.uint64),
    )


def load_nerl_graph(graph_source: Any, evaluation_cfg: Any, module: Any):
    ERLGraph, _, _ = import_em_erl()
    if isinstance(graph_source, ERLGraph):
        return graph_source, False
    if hasattr(graph_source, "node_coords_zyx") and hasattr(graph_source, "edge_ptr"):
        return graph_source, False

    graph_path = Path(graph_source)
    suffix = graph_path.suffix.lower()
    if suffix == ".npz":
        return ERLGraph.from_npz(graph_path), False
    if suffix in {".pkl", ".pickle"}:
        import pickle

        with open(graph_path, "rb") as f:
            skeleton = pickle.load(f)
        return networkx_skeleton_to_erl_graph(skeleton, evaluation_cfg, module), True
    raise ValueError(
        "evaluation.nerl_graph must be an ERLGraph .npz or "
        f"NetworkX skeleton pickle, got {graph_path}"
    )


def nerl_node_positions(module, graph: Any, voxel_coords: bool, evaluation_cfg: Any) -> np.ndarray:
    if voxel_coords:
        return np.asarray(graph.node_coords_zyx, dtype=np.int64)

    resolution = module_cfg_value(module, evaluation_cfg, "nerl_resolution", None)
    if resolution is None:
        data_cfg = getattr(getattr(module, "cfg", None), "data", None)
        test_cfg = getattr(data_cfg, "test", None)
        resolution = getattr(test_cfg, "resolution", None)
    return graph.get_nodes_position(resolution)


def prepare_nerl_segmentation(decoded_predictions: np.ndarray) -> np.ndarray:
    seg = np.asarray(decoded_predictions)
    while seg.ndim > 3 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim > 3:
        singleton_axes = tuple(i for i, size in enumerate(seg.shape) if size == 1)
        if singleton_axes:
            seg = np.squeeze(seg, axis=singleton_axes)
    if seg.ndim != 3:
        raise ValueError(f"NERL expects a 3D decoded instance volume, got shape {seg.shape}")
    if not np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.uint32, copy=False)
    return seg


def extract_nerl_score_outputs(score: Any) -> tuple[float, float, int, np.ndarray]:
    """Return aggregate and per-GT ERL values from an em_erl score object."""
    score_erl = np.asarray(score.erl)
    if score_erl.ndim > 1:
        score_erl = score_erl[0]

    pred_erl = getattr(score, "pred_erl", None)
    gt_erl = getattr(score, "gt_erl", None)
    if pred_erl is None:
        pred_erl = score_erl[0]
    if gt_erl is None:
        gt_erl = score_erl[1]
    num_skeletons = int(score_erl[2]) if score_erl.size > 2 else int(len(score.skeleton_len))

    per_gt_erl = None
    for attr_name in (
        "per_gt_erl",
        "gt_segment_erl",
        "skeleton_erl_pair",
        "skeleton_erl_pairs",
    ):
        attr_value = getattr(score, attr_name, None)
        if attr_value is not None:
            per_gt_erl = np.asarray(attr_value, dtype=np.float64)
            break

    if per_gt_erl is None:
        skeleton_pred_erl = getattr(score, "skeleton_pred_erl", None)
        if skeleton_pred_erl is None:
            skeleton_pred_erl = score.skeleton_erl
        skeleton_gt_erl = getattr(score, "skeleton_gt_erl", None)
        if skeleton_gt_erl is None:
            skeleton_gt_erl = score.skeleton_len

        skeleton_pred_erl = np.asarray(skeleton_pred_erl, dtype=np.float64)
        skeleton_gt_erl = np.asarray(skeleton_gt_erl, dtype=np.float64)
        if skeleton_pred_erl.ndim == 2 and skeleton_pred_erl.shape[1] >= 2:
            per_gt_erl = skeleton_pred_erl[:, :2]
        else:
            per_gt_erl = np.column_stack([skeleton_pred_erl, skeleton_gt_erl])

    if per_gt_erl.ndim == 1:
        per_gt_erl = per_gt_erl.reshape(0, 2) if per_gt_erl.size == 0 else per_gt_erl.reshape(1, -1)
    if per_gt_erl.ndim != 2 or per_gt_erl.shape[1] != 2:
        raise ValueError(f"NERL per-GT ERL array must have shape [N, 2], got {per_gt_erl.shape}")

    return float(pred_erl), float(gt_erl), num_skeletons, per_gt_erl


def compute_nerl_metrics(
    module,
    decoded_predictions: np.ndarray,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    volume_name: str | None,
) -> None:
    evaluation_cfg = get_effective_evaluation_config(module)
    graph_value = select_volume_config_value(
        module_cfg_value(module, evaluation_cfg, "nerl_graph", None),
        volume_name,
    )
    if graph_value is None:
        logger.warning(
            "%sSkipping NERL: set evaluation.nerl_graph to an "
            "ERLGraph .npz or BANIS/NISB skeleton.pkl",
            volume_prefix,
        )
        return

    mask_value = select_volume_config_value(
        module_cfg_value(module, evaluation_cfg, "nerl_mask", None),
        volume_name,
    )
    _, compute_erl_score, compute_segment_lut = import_em_erl()
    erl_graph, voxel_coords = load_nerl_graph(graph_value, evaluation_cfg, module)
    node_positions = nerl_node_positions(module, erl_graph, voxel_coords, evaluation_cfg)
    segment = prepare_nerl_segmentation(decoded_predictions)

    merge_threshold = int(module_cfg_value(module, evaluation_cfg, "nerl_merge_threshold", 1))
    chunk_num = int(module_cfg_value(module, evaluation_cfg, "nerl_chunk_num", 1))
    node_segment_lut, mask_segment_id = compute_segment_lut(
        segment,
        node_positions,
        mask=mask_value,
        chunk_num=chunk_num,
        data_type=segment.dtype,
    )

    score = compute_erl_score(
        erl_graph,
        node_segment_lut,
        mask_segment_id,
        merge_threshold=merge_threshold,
    )
    score.compute_erl()

    pred_erl, gt_erl, num_skeletons, per_gt_erl = extract_nerl_score_outputs(score)
    nerl = pred_erl / gt_erl if gt_erl > 0 else float("nan")

    logger.info("%sNERL: %.6f", volume_prefix, nerl)
    logger.info("%s  Pred ERL: %.6f", volume_prefix, pred_erl)
    logger.info("%s  GT ERL: %.6f", volume_prefix, gt_erl)
    logger.info("%s  # Skeletons: %d", volume_prefix, num_skeletons)

    metrics_dict["nerl"] = nerl
    metrics_dict["nerl_pred_erl"] = pred_erl
    metrics_dict["nerl_gt_erl"] = gt_erl
    metrics_dict["nerl_erl"] = pred_erl
    metrics_dict["nerl_max_erl"] = gt_erl
    metrics_dict["nerl_num_skeletons"] = num_skeletons
    metrics_dict["nerl_graph"] = str(graph_value)
    metrics_dict["nerl_gt_segment_ids"] = np.asarray(erl_graph.skeleton_id)
    metrics_dict["nerl_per_gt_erl"] = per_gt_erl

    if hasattr(module, "log"):
        try:
            module.log(
                "test_nerl",
                nerl,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to log test_nerl metric: %s", exc)


__all__ = [
    "compute_nerl_metrics",
    "extract_nerl_score_outputs",
    "get_effective_evaluation_config",
    "import_em_erl",
    "load_nerl_graph",
    "module_cfg_value",
    "networkx_skeleton_to_erl_graph",
    "nerl_node_positions",
    "prepare_nerl_segmentation",
    "reorder_coordinate_axes",
    "select_volume_config_value",
]
