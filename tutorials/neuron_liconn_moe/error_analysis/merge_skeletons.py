#!/usr/bin/env python3
"""Concatenate sharded skeleton bundles into the single bundle `analyze.py` reads.

Refuses to merge a partial set: a silently incomplete bundle would show up
downstream as segments that are `unmeasured` for no stated reason, which is
indistinguishable from a segment whose skeleton genuinely failed.

    python dev/liconn_moe/merge_skeletons.py --shards 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402
from build_skeletons import load_cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="skeletons_shard{:03d}.npz")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--directory", type=Path, default=V.OUT)
    parser.add_argument("--output", type=Path, default=V.SKELETONS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge what exists and record which shards are absent")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")

    paths = [args.directory / args.pattern.format(i) for i in range(args.shards)]
    missing = [p.name for p in paths if not p.exists()]
    if missing and not args.allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} of {args.shards} shards are missing: {missing[:8]}"
            " -- rerun them, or pass --allow-partial deliberately"
        )
    present = [p for p in paths if p.exists()]

    labels, voxels, vertices, radii, edges = [], [], [], [], []
    vertex_offset, edge_offset = [0], [0]
    metadata = []
    for path in present:
        with np.load(path, allow_pickle=False) as data:
            block_labels = np.asarray(data["labels"], dtype=np.uint32)
            block_v_off = np.asarray(data["vertex_offset"], dtype=np.int64)
            block_e_off = np.asarray(data["edge_offset"], dtype=np.int64)
            block_vertices = np.asarray(data["vertices_um_zyx"], dtype=np.float32)
            block_radii = np.asarray(data["radii_um"], dtype=np.float32)
            block_edges = np.asarray(data["edges"], dtype=np.int32)
            block_voxels = np.asarray(data["voxel_counts"], dtype=np.int64)
            meta = json.loads(str(data["metadata"]))
        labels.append(block_labels)
        voxels.append(block_voxels)
        vertices.append(block_vertices)
        radii.append(block_radii)
        edges.append(block_edges)
        # Edges are local to each label's own vertex block, so per-label offsets
        # concatenate directly; only the running totals need rebasing.
        for index in range(len(block_labels)):
            vertex_offset.append(
                vertex_offset[-1] + int(block_v_off[index + 1] - block_v_off[index])
            )
            edge_offset.append(edge_offset[-1] + int(block_e_off[index + 1] - block_e_off[index]))
        metadata.append(meta)

    all_labels = np.concatenate(labels)
    order = np.argsort(all_labels, kind="stable")
    if len(np.unique(all_labels)) != len(all_labels):
        raise ValueError("a label appears in more than one shard")

    # Duplicate detection alone is not coverage: a label present in *no* shard is
    # the failure that actually happened once, when a size cutoff applied before
    # sharding silently re-partitioned the cohort. Check membership against the
    # cohort itself, and account for every absence as a deliberate skip.
    skipped = {label for m in metadata for label in m.get("labels_skipped_oversized", [])}
    cohort, _, _ = load_cohort(metadata[0]["min_voxels"])
    expected = set(int(v) for v in cohort)
    got = set(int(v) for v in all_labels)
    unexpected = got - expected
    if unexpected:
        raise ValueError(f"{len(unexpected)} merged labels are not in the cohort: "
                         f"{sorted(unexpected)[:8]}")
    absent = expected - got - skipped
    if absent and not args.allow_partial:
        raise ValueError(
            f"{len(absent)} cohort labels are in no shard and were not recorded as "
            f"skipped: {sorted(absent)[:8]} -- the shard set is inconsistent"
        )

    # Reorder whole per-label blocks, not individual rows.
    v_off = np.asarray(vertex_offset, dtype=np.int64)
    e_off = np.asarray(edge_offset, dtype=np.int64)
    flat_vertices = np.concatenate(vertices)
    flat_radii = np.concatenate(radii)
    flat_edges = np.concatenate(edges)
    out_vertices, out_radii, out_edges = [], [], []
    out_v_off, out_e_off = [0], [0]
    for index in order:
        out_vertices.append(flat_vertices[v_off[index] : v_off[index + 1]])
        out_radii.append(flat_radii[v_off[index] : v_off[index + 1]])
        out_edges.append(flat_edges[e_off[index] : e_off[index + 1]])
        out_v_off.append(out_v_off[-1] + len(out_vertices[-1]))
        out_e_off.append(out_e_off[-1] + len(out_edges[-1]))

    merged = {
        "volume": V.NAME,
        "segmentation": str(V.SEG),
        "segmentation_sha256": metadata[0]["segmentation_sha256"],
        "shape_zyx": list(V.SHAPE_ZYX),
        "spacing_nm_zyx": list(V.SPACING_NM_ZYX),
        "min_voxels": metadata[0]["min_voxels"],
        "skeleton_count": int(len(all_labels)),
        "vertex_count": int(out_v_off[-1]),
        "edge_count": int(out_e_off[-1]),
        "shards_expected": args.shards,
        "shards_merged": len(present),
        "shards_missing": missing,
        "complete": not missing and not absent,
        "cohort_labels_expected": len(expected),
        "labels_skipped_oversized": sorted(skipped),
        "labels_absent_unexplained": sorted(absent),
        "teasar_params": metadata[0]["teasar_params"],
        "simplification_nm": metadata[0]["simplification_nm"],
        "coordinate_convention": metadata[0]["coordinate_convention"],
        "edge_indexing": metadata[0]["edge_indexing"],
        "radii_clamped_to_floor": sum(m["radii_clamped_to_floor"] for m in metadata),
        "labels_without_skeleton": sorted(
            label for m in metadata for label in m["labels_without_skeleton"]
        ),
    }
    if len({m["segmentation_sha256"] for m in metadata}) != 1:
        raise ValueError("shards were built from different segmentations")

    temporary = args.output.with_suffix(".TMP.npz")
    np.savez_compressed(
        temporary,
        labels=all_labels[order],
        voxel_counts=np.concatenate(voxels)[order],
        vertex_offset=np.asarray(out_v_off, dtype=np.int64),
        edge_offset=np.asarray(out_e_off, dtype=np.int64),
        vertices_um_zyx=np.concatenate(out_vertices),
        radii_um=np.concatenate(out_radii),
        edges=np.concatenate(out_edges) if out_edges else np.zeros((0, 2), np.int32),
        metadata=np.asarray(json.dumps(merged, sort_keys=True)),
    )
    temporary.replace(args.output)
    args.output.with_name(args.output.stem + "_metadata.json").write_text(
        json.dumps(merged, indent=2) + "\n"
    )
    print(f"merged {len(present)}/{args.shards} shards -> {args.output}")
    print(f"  {merged['skeleton_count']} skeletons, {merged['vertex_count']:,} vertices, "
          f"complete={merged['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
