"""Reusable NERL scoring helpers.

This module owns PyTC's low-level adapter around ``em_erl``: graph loading,
NetworkX skeleton conversion, segmentation normalization, and score extraction.
Evaluation-stage logging and report wiring lives in ``connectomics.evaluation``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NerlGraphOptions:
    """Options for converting NetworkX skeleton pickles into ERL graphs."""

    skeleton_id_attribute: str = "id"
    skeleton_position_attribute: str = "index_position"
    skeleton_edge_length_attribute: str = "edge_length"
    skeleton_position_order: str = "xyz"
    prediction_position_order: str | None = None


@dataclass(frozen=True)
class NerlScoreResult:
    """Detailed NERL score output for stage-specific adapters."""

    nerl: float
    pred_erl: float
    gt_erl: float
    num_skeletons: int
    per_gt_erl: np.ndarray
    graph: Any
    voi_split: float = float("nan")
    voi_merge: float = float("nan")
    voi_total: float = float("nan")


def _materialize_for_parallel(segment, mask, num_workers):
    """Write ndarray inputs to temp HDF5 so em_erl's multi-process path can use them.

    em_erl's parallel `compute_segment_lut` requires path-based inputs because
    workers each open their own VolumeSource. When the caller hands us an
    in-memory ndarray (as the test pipeline does) we materialize it to a
    NamedTemporaryFile here so the parallel path is actually exercised. Returns
    `(seg_arg, mask_arg, tempfiles_to_cleanup)`.
    """
    if num_workers <= 1:
        return segment, mask, []

    import os
    import tempfile

    import h5py

    # Cluster $TMPDIR/$HOME often lives on a network FS where HDF5 POSIX
    # file locking returns ENOSPC even when there's plenty of space. Prefer
    # /dev/shm (tmpfs in RAM) when available; otherwise fall back to the
    # default temp dir.
    temp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None

    tempfiles = []

    def _maybe_write(arr, prefix):
        if not isinstance(arr, np.ndarray):
            return arr
        fh = tempfile.NamedTemporaryFile(suffix=".h5", prefix=prefix, dir=temp_dir, delete=False)
        fh.close()
        with h5py.File(fh.name, "w") as fid:
            fid.create_dataset("main", data=np.ascontiguousarray(arr))
        tempfiles.append(fh.name)
        return fh.name

    seg_arg = _maybe_write(segment, "nerl_seg_")
    mask_arg = _maybe_write(mask, "nerl_mask_") if mask is not None else None
    return seg_arg, mask_arg, tempfiles


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


def networkx_skeleton_to_erl_graph(
    skeleton: Any,
    options: NerlGraphOptions | None = None,
    resolution: Any = None,
):
    ERLGraph, _, _ = import_em_erl()
    options = options or NerlGraphOptions()

    node_ids = list(skeleton.nodes)
    if not node_ids:
        raise ValueError("NERL skeleton has no nodes")

    raw_skeleton_ids = []
    node_coords = []
    for node_id in node_ids:
        node_data = skeleton.nodes[node_id]
        raw_skeleton_ids.append(node_data[options.skeleton_id_attribute])
        node_coords.append(node_data[options.skeleton_position_attribute])

    skeleton_ids = list(dict.fromkeys(raw_skeleton_ids))
    skeleton_index_by_id = {skeleton_id: i for i, skeleton_id in enumerate(skeleton_ids)}
    node_index_by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    node_skeleton_index = np.asarray(
        [skeleton_index_by_id[skeleton_id] for skeleton_id in raw_skeleton_ids],
        dtype=np.uint32,
    )
    node_coords_arr = reorder_coordinate_axes(
        np.asarray(node_coords, dtype=np.float32),
        source_order=options.skeleton_position_order,
        target_order=options.prediction_position_order,
    )

    edge_coords = node_coords_arr
    if resolution is not None:
        res = np.asarray(resolution, dtype=np.float64).reshape(-1)
        if res.size != 3:
            raise ValueError(f"NERL resolution must have 3 elements, got {res.size}")
        edge_coords = node_coords_arr.astype(np.float64) * res

    edge_buckets: list[list[tuple[int, int, float]]] = [[] for _ in skeleton_ids]
    skeleton_len: np.ndarray = np.zeros(len(skeleton_ids), dtype=np.float64)
    for u, v, edge_data in skeleton.edges(data=True):
        if u not in node_index_by_id or v not in node_index_by_id:
            continue
        u_idx = node_index_by_id[u]
        v_idx = node_index_by_id[v]
        skel_idx = int(node_skeleton_index[u_idx])
        if skel_idx != int(node_skeleton_index[v_idx]):
            continue
        if options.skeleton_edge_length_attribute in edge_data:
            edge_len = float(edge_data[options.skeleton_edge_length_attribute])
        else:
            edge_len = float(np.linalg.norm(edge_coords[u_idx] - edge_coords[v_idx]))
        edge_buckets[skel_idx].append((u_idx, v_idx, edge_len))
        skeleton_len[skel_idx] += edge_len

    edge_ptr = [0]
    edge_u = []
    edge_v = []
    edge_lens: list[float] = []
    for bucket in edge_buckets:
        for u_idx, v_idx, length in bucket:
            edge_u.append(u_idx)
            edge_v.append(v_idx)
            edge_lens.append(length)
        edge_ptr.append(len(edge_u))

    return ERLGraph(
        skeleton_id=np.asarray(skeleton_ids),
        skeleton_len=skeleton_len,
        node_skeleton_index=node_skeleton_index,
        node_coords_zyx=node_coords_arr,
        edge_u=np.asarray(edge_u, dtype=np.uint32),
        edge_v=np.asarray(edge_v, dtype=np.uint32),
        edge_len=np.asarray(edge_lens, dtype=np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.uint64),
    )


_ERL_CACHE_FIELDS = (
    "skeleton_id",
    "skeleton_len",
    "node_skeleton_index",
    "node_coords_zyx",
    "edge_u",
    "edge_v",
    "edge_len",
    "edge_ptr",
)


def _resolution_cache_tag(resolution: Any) -> str:
    if resolution is None:
        return "vox"
    res = np.asarray(resolution, dtype=np.float64).reshape(-1)
    return "res" + "_".join(f"{r:g}" for r in res)


def _erl_cache_path(source: Path, resolution: Any = None) -> Path:
    return source.with_suffix(source.suffix + f".erl_cache.{_resolution_cache_tag(resolution)}.npz")


def _save_erl_cache(graph: Any, voxel_coords: bool, cache_path: Path) -> None:
    try:
        np.savez_compressed(
            cache_path,
            voxel_coords=np.asarray(int(voxel_coords)),
            **{name: getattr(graph, name) for name in _ERL_CACHE_FIELDS},
        )
    except OSError as exc:
        logger.warning("Failed to write ERLGraph cache %s: %s", cache_path, exc)


def _load_erl_cache(cache_path: Path) -> tuple[Any, bool]:
    ERLGraph, _, _ = import_em_erl()
    data = np.load(cache_path, allow_pickle=False)
    voxel_coords = bool(int(np.asarray(data["voxel_coords"]).item()))
    graph = ERLGraph(**{name: data[name] for name in _ERL_CACHE_FIELDS})
    return graph, voxel_coords


def load_nerl_graph(
    graph_source: Any,
    graph_options: NerlGraphOptions | None = None,
    resolution: Any = None,
):
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
        cache_path = _erl_cache_path(graph_path, resolution)
        if cache_path.exists() and cache_path.stat().st_mtime >= graph_path.stat().st_mtime:
            try:
                return _load_erl_cache(cache_path)
            except (KeyError, OSError, ValueError) as exc:
                logger.warning("Ignoring corrupt ERLGraph cache %s: %s", cache_path, exc)

        import pickle

        with open(graph_path, "rb") as f:
            skeleton = pickle.load(f)
        graph = networkx_skeleton_to_erl_graph(skeleton, graph_options, resolution=resolution)
        _save_erl_cache(graph, True, cache_path)
        return graph, True
    raise ValueError(
        "NERL skeleton must be an ERLGraph .npz or " f"NetworkX skeleton pickle, got {graph_path}"
    )


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


def skeleton_voi(node_pred_ids: Any, node_gt_ids: Any) -> tuple[float, float, float]:
    """Skeleton-based Variation of Information, matching ``funlib.evaluate.rand_voi``.

    Faithful numpy port of ``lib/funlib.evaluate`` (``impl/rand_voi.hpp``), which
    is what BANIS uses: build the joint label histogram over skeleton nodes with a
    non-zero GT label — funlib counts a node only where ``labels_a`` (the GT) is
    non-zero, so GT id 0 is ignored while a predicted background id 0 is kept —
    then, with base-2 logs,

        voi_split = H(gt, pred) - H(gt)   = H(pred | gt)   # over-segmentation
        voi_merge = H(gt, pred) - H(pred) = H(gt | pred)   # under-segmentation

    This mirrors BANIS's ``rand_voi(gt_ids, pred_ids)`` sampled at skeleton nodes
    (funlib's argument order is ``rand_voi(labels_a=truth, labels_b=test)``).
    Validated against funlib's own test vectors.

    Parameters
    ----------
    node_pred_ids : predicted segment id per node (em_erl ``node_segment_lut``).
    node_gt_ids   : ground-truth skeleton label per node
                    (``ERLGraph.skeleton_id[node_skeleton_index]``).

    Returns ``(voi_split, voi_merge, voi_total)``.
    """
    gt = np.asarray(node_gt_ids).ravel()
    pred = np.asarray(node_pred_ids).ravel()
    if gt.shape != pred.shape:
        raise ValueError(
            f"node_gt_ids {gt.shape} and node_pred_ids {pred.shape} must have equal length"
        )
    keep = gt != 0  # funlib rand_voi counts only where labels_a (GT) != 0
    gt = gt[keep].astype(np.uint64, copy=False)
    pred = pred[keep].astype(np.uint64, copy=False)
    total = gt.size
    if total == 0:
        return 0.0, 0.0, 0.0

    def _entropy(counts: np.ndarray) -> float:
        p = counts.astype(np.float64) / float(total)
        return float(-np.sum(p * np.log2(p)))

    _, gt_counts = np.unique(gt, return_counts=True)
    _, pred_counts = np.unique(pred, return_counts=True)
    _, joint_counts = np.unique(np.stack([gt, pred], axis=1), axis=0, return_counts=True)

    h_gt = _entropy(gt_counts)
    h_pred = _entropy(pred_counts)
    h_joint = _entropy(joint_counts)
    voi_split = h_joint - h_gt
    voi_merge = h_joint - h_pred
    return float(voi_split), float(voi_merge), float(voi_split + voi_merge)


def compute_nerl_score_details(
    segmentation: np.ndarray,
    skeleton_value: Any,
    *,
    skeleton_mask_value: Any = None,
    resolution: Any = None,
    merge_threshold: int = 1,
    chunk_num: int = 1,
    num_workers: int = 1,
    graph_options: NerlGraphOptions | None = None,
) -> NerlScoreResult:
    """Compute detailed NERL output for one segmentation/skeleton pair."""
    _, compute_erl_score, compute_segment_lut = import_em_erl()
    erl_graph, voxel_coords = load_nerl_graph(skeleton_value, graph_options, resolution=resolution)
    if voxel_coords:
        node_positions = np.asarray(erl_graph.node_coords_zyx, dtype=np.int64)
    else:
        node_positions = erl_graph.get_nodes_position(resolution)

    segment = prepare_nerl_segmentation(segmentation)
    seg_arg, mask_arg, _tempfiles = _materialize_for_parallel(
        segment, skeleton_mask_value, int(num_workers)
    )
    try:
        node_segment_lut, mask_segment_id = compute_segment_lut(
            seg_arg,
            node_positions,
            mask=mask_arg,
            chunk_num=int(chunk_num),
            data_type=segment.dtype,
            num_workers=int(num_workers),
        )
    finally:
        for path in _tempfiles:
            try:
                Path(path).unlink()
            except OSError:
                pass
    score = compute_erl_score(
        erl_graph,
        node_segment_lut,
        mask_segment_id,
        merge_threshold=int(merge_threshold),
    )
    score.compute_erl()
    pred_erl, gt_erl, num_skeletons, per_gt_erl = extract_nerl_score_outputs(score)
    nerl = pred_erl / gt_erl if gt_erl > 0 else float("nan")
    # Skeleton-based VOI reuses the per-node predicted-segment LUT just built for
    # ERL, against each node's GT skeleton label (see skeleton_voi).
    node_gt_ids = np.asarray(erl_graph.skeleton_id)[np.asarray(erl_graph.node_skeleton_index)]
    voi_split, voi_merge, voi_total = skeleton_voi(node_segment_lut, node_gt_ids)
    return NerlScoreResult(
        nerl=float(nerl),
        pred_erl=float(pred_erl),
        gt_erl=float(gt_erl),
        num_skeletons=num_skeletons,
        per_gt_erl=per_gt_erl,
        graph=erl_graph,
        voi_split=voi_split,
        voi_merge=voi_merge,
        voi_total=voi_total,
    )


def compute_nerl_score(
    segmentation: np.ndarray,
    skeleton_value: Any,
    *,
    skeleton_mask_value: Any = None,
    resolution: Any = None,
    merge_threshold: int = 1,
    chunk_num: int = 1,
    num_workers: int = 1,
    graph_options: NerlGraphOptions | None = None,
) -> tuple[float, float, float]:
    """Return ``(nerl, pred_erl, gt_erl)`` for one segmentation/skeleton pair."""
    result = compute_nerl_score_details(
        segmentation,
        skeleton_value,
        skeleton_mask_value=skeleton_mask_value,
        resolution=resolution,
        merge_threshold=merge_threshold,
        chunk_num=chunk_num,
        num_workers=num_workers,
        graph_options=graph_options,
    )
    return result.nerl, result.pred_erl, result.gt_erl


__all__ = [
    "NerlGraphOptions",
    "NerlScoreResult",
    "compute_nerl_score",
    "compute_nerl_score_details",
    "extract_nerl_score_outputs",
    "import_em_erl",
    "load_nerl_graph",
    "networkx_skeleton_to_erl_graph",
    "prepare_nerl_segmentation",
    "reorder_coordinate_axes",
    "skeleton_voi",
]
