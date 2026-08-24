#!/usr/bin/env python3
"""Evaluate a native j0126 precomputed segmentation at test-50 skeleton nodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from connectomics.metrics.nerl import (
    extract_nerl_score_outputs,
    import_em_erl,
    skeleton_voi,
)


def _load_graph(skeleton_path: Path) -> tuple[Any, np.ndarray]:
    import_em_erl()
    from em_erl import skel_to_erlgraph

    class Skeleton:
        pass

    skeletons = {}
    with h5py.File(skeleton_path, "r", locking=False) as handle:
        for key in sorted(handle.keys(), key=lambda value: int(value)):
            skeleton = Skeleton()
            skeleton.vertices = handle[key]["vertices"][:].astype(np.int64)
            skeleton.edges = handle[key]["edges"][:].astype(np.int64)
            skeletons[int(key)] = skeleton
    graph = skel_to_erlgraph(skeletons)
    positions_xyz = np.concatenate(
        [skeletons[int(identifier)].vertices[:, ::-1] for identifier in graph.skeleton_id],
        axis=0,
    )
    if positions_xyz.shape[0] != int(graph.num_nodes):
        raise RuntimeError("Skeleton-node order does not match the ERL graph")
    return graph, positions_xyz


def _sample_node_lut(
    segmentation: Path,
    positions_xyz: np.ndarray,
    *,
    resolution_xyz: tuple[int, int, int],
    read_chunk_xyz: tuple[int, int, int],
) -> np.ndarray:
    from cloudvolume import CloudVolume

    volume = CloudVolume(
        segmentation.as_uri(),
        mip=list(resolution_xyz),
        fill_missing=True,
        bounded=False,
        progress=False,
    )
    chunk = np.asarray(read_chunk_xyz, dtype=np.int64)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, chunk_index in enumerate(positions_xyz // chunk):
        groups[tuple(int(value) for value in chunk_index)].append(index)

    lut = np.zeros(positions_xyz.shape[0], dtype=np.uint64)
    for chunk_index, node_indices in groups.items():
        origin = np.asarray(chunk_index, dtype=np.int64) * chunk
        stop = origin + chunk
        block = np.asarray(
            volume[
                origin[0] : stop[0],
                origin[1] : stop[1],
                origin[2] : stop[2],
            ]
        )
        if block.ndim == 4:
            block = block[..., 0]
        indices = np.asarray(node_indices, dtype=np.int64)
        local = positions_xyz[indices] - origin
        in_bounds = np.all((local >= 0) & (local < np.asarray(block.shape)), axis=1)
        if np.any(in_bounds):
            query = local[in_bounds]
            lut[indices[in_bounds]] = block[query[:, 0], query[:, 1], query[:, 2]]
    return lut


def _score(graph: Any, lut: np.ndarray, merge_threshold: int) -> dict[str, Any]:
    _, compute_erl_score, _ = import_em_erl()
    score = compute_erl_score(graph, lut, None, merge_threshold=merge_threshold)
    score.compute_erl()
    pred_erl, gt_erl, num_skeletons, _ = extract_nerl_score_outputs(score)
    return {
        "merge_threshold": merge_threshold,
        "nerl": float(pred_erl / gt_erl),
        "pred_erl": float(pred_erl),
        "gt_erl": float(gt_erl),
        "num_skeletons": int(num_skeletons),
    }


def _parse_triplet(value: str) -> tuple[int, int, int]:
    parsed = tuple(int(item) for item in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--segmentation", type=Path)
    source.add_argument(
        "--node-lut",
        type=Path,
        help="existing graph-order node-to-segment LUT, for example the canonical FFN LUT",
    )
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-xyz", type=_parse_triplet, default=(9, 9, 20))
    parser.add_argument("--read-chunk-xyz", type=_parse_triplet, default=(256, 256, 256))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    graph, positions_xyz = _load_graph(args.skeletons)
    if args.node_lut is not None:
        from em_erl import load_node_segment_lut

        lut = np.asarray(load_node_segment_lut(args.node_lut), dtype=np.uint64)
        if lut.shape != (positions_xyz.shape[0],):
            raise RuntimeError(
                f"node LUT has shape {lut.shape}; expected {(positions_xyz.shape[0],)}"
            )
    else:
        lut = _sample_node_lut(
            args.segmentation,
            positions_xyz,
            resolution_xyz=args.resolution_xyz,
            read_chunk_xyz=args.read_chunk_xyz,
        )
    foreground = lut != 0
    gt_labels = np.asarray(graph.node_skeleton_index, dtype=np.uint64) + 1
    voi_split, voi_merge, voi_sum = skeleton_voi(lut, gt_labels)
    fg_split, fg_merge, fg_sum = skeleton_voi(lut[foreground], gt_labels[foreground])
    owners: dict[int, set[int]] = defaultdict(set)
    for label, skeleton_index in zip(
        lut[foreground], np.asarray(graph.node_skeleton_index)[foreground]
    ):
        owners[int(label)].add(int(skeleton_index))
    multi_owner_labels = [label for label, values in owners.items() if len(values) > 1]
    multi_owner_nodes = (
        np.isin(lut, multi_owner_labels)
        if multi_owner_labels
        else np.zeros(lut.shape, dtype=bool)
    )
    fragments_per_skeleton = []
    dominant_fraction_per_skeleton = []
    for skeleton_index in range(int(graph.num_skeletons)):
        skeleton_labels = lut[np.asarray(graph.node_skeleton_index) == skeleton_index]
        labels, counts = np.unique(skeleton_labels[skeleton_labels != 0], return_counts=True)
        fragments_per_skeleton.append(int(labels.size))
        dominant_fraction_per_skeleton.append(
            float(counts.max() / skeleton_labels.size) if counts.size else 0.0
        )
    report = {
        "segmentation": str(args.segmentation) if args.segmentation is not None else None,
        "node_lut": str(args.node_lut) if args.node_lut is not None else None,
        "skeletons": str(args.skeletons),
        "coordinate_order": "skeleton ZYX; segmentation XYZ",
        "edge_length_units": "native voxels",
        "nodes": int(lut.size),
        "node_coverage": float(np.mean(foreground)),
        "unique_foreground_labels": int(np.unique(lut[foreground]).size),
        "multi_owner_labels": len(multi_owner_labels),
        "multi_owner_node_fraction": float(np.mean(multi_owner_nodes)),
        "mean_fragments_per_skeleton": float(np.mean(fragments_per_skeleton)),
        "median_dominant_label_fraction": float(
            np.median(dominant_fraction_per_skeleton)
        ),
        "nerl": {
            "mt0": _score(graph, lut, 0),
            "mt5": _score(graph, lut, 5),
        },
        "voi_background_inclusive": {
            "split": voi_split,
            "merge": voi_merge,
            "sum": voi_sum,
        },
        "voi_foreground_only": {
            "split": fg_split,
            "merge": fg_merge,
            "sum": fg_sum,
            "nodes": int(np.count_nonzero(foreground)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
