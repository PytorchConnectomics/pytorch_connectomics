#!/usr/bin/env python3
"""Skeletonize every cohort label of the eb2 segmentation into one graph bundle.

TEASAR imposes forest topology on whatever it is given; a skeleton is not evidence
that a segment is one biological process. What it does supply that a PCA profile
cannot is a local radius at every vertex and terminals that exist regardless of
how tube-like the object is -- the two measurements the continuity analysis needs
in order to see a broken axon terminal that is not a tube.

Settings match the sibling moe volume in
`dev/astra_nogt_eval/glia_revision/build_arbors.py` (const 100 nm, 50 nm
simplification, fix_branching, fix_borders), so the two volumes stay comparable.

    python dev/liconn_moe/build_skeletons.py --parallel 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

from connectomics.decoding.error_correction.skeletonize import simplify_skeleton  # noqa: E402

SPACING_NM = np.asarray(V.SPACING_NM_ZYX)


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(min_voxels: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Cohort IDs and counts, plus the largest ID present anywhere in the volume.

    The maximum is the whole inventory's, not the cohort's: the mask LUT has to
    span every ID the segmentation can contain, or a label merely too small to
    be in the cohort reads as out of range.
    """
    with np.load(V.SIZES) as data:
        ids = np.asarray(data["ids"], dtype=np.int64)
        counts = np.asarray(data["counts"], dtype=np.int64)
    keep = counts >= min_voxels
    order = np.argsort(ids[keep], kind="stable")
    return ids[keep][order].astype(np.uint32), counts[keep][order], int(ids.max())


def select_shard(
    labels: np.ndarray, counts: np.ndarray, shard: int, num_shards: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the cohort into equal-*work* shards, not equal-count ones.

    TEASAR cost grows faster than linearly with object size, and this cohort spans
    1e3 to 1.2e7 voxels. Round-robin on ID would put several of the giants in one
    shard and leave it running long after the rest finished, which is exactly the
    tail that makes the single-process run unpredictable. Largest-first into the
    currently-lightest shard keeps the totals close.
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
    load = np.zeros(num_shards, dtype=np.int64)
    assignment = np.empty(len(labels), dtype=np.int64)
    for index in np.argsort(counts)[::-1]:
        target = int(np.argmin(load))
        assignment[index] = target
        load[target] += int(counts[index])
    keep = assignment == shard
    return labels[keep], counts[keep]


def load_masked(labels: np.ndarray, max_id: int) -> np.ndarray:
    """Read the segmentation, zeroing every label outside the cohort."""
    keep = np.zeros(max_id + 1, dtype=bool)
    keep[labels] = True
    with h5py.File(V.SEG, "r") as handle:
        dataset = handle["main"]
        if tuple(dataset.shape) != V.SHAPE_ZYX:
            raise ValueError(f"shape {dataset.shape} != expected {V.SHAPE_ZYX}")
        dense = np.zeros(dataset.shape, dtype=np.uint32)
        for z0 in range(0, dataset.shape[0], 16):
            slab = np.asarray(dataset[z0 : z0 + 16])
            if slab.size and int(slab.max()) >= len(keep):
                raise ValueError(
                    f"segmentation holds ID {int(slab.max())} above the size inventory max {max_id}"
                )
            small = slab.astype(np.uint32)
            dense[z0 : z0 + 16] = np.where(keep[small], small, 0)
    return dense


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=V.SKELETONS)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--min-voxels", type=int, default=V.MIN_VOXELS)
    parser.add_argument(
        "--simplification-nm",
        type=float,
        default=V.SIMPLIFICATION_NM,
        help="vertex spacing kept along degree-2 chains. 50 nm is fine for length and "
        "caliber, but a terminal branch here is ~160 nm, so it leaves 3-4 vertices and "
        "no room to measure a tip-vs-shaft radius profile. Use ~10 nm to resolve "
        "terminal shape.",
    )
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N labels only")
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=0,
        help="skip labels at or above this size. TEASAR cost is superlinear in object "
        "size, so one pathological label can outlast the other thousand combined; "
        "skipped labels are recorded in the metadata, never silently dropped",
    )
    parser.add_argument("--shard", type=int, default=0, help="0-based index of this shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels, counts, max_id = load_cohort(args.min_voxels)
    if args.limit:
        labels, counts = labels[: args.limit], counts[: args.limit]
    total_labels, total_voxels = len(labels), int(counts.sum())
    if args.num_shards > 1:
        labels, counts = select_shard(labels, counts, args.shard, args.num_shards)
    # The size cutoff is applied AFTER sharding, never before: filtering first
    # would change the greedy balance and hand this shard a different label set
    # than its siblings were assigned, so a merge would duplicate and omit labels.
    oversized: list[int] = []
    if args.max_voxels:
        too_big = counts >= args.max_voxels
        oversized = [int(v) for v in labels[too_big]]
        labels, counts = labels[~too_big], counts[~too_big]
        if oversized:
            log(f"skipping {len(oversized)} oversized label(s) in this shard "
                f"(>= {args.max_voxels:,} voxels): {oversized}")
    log(f"cohort {total_labels} labels >= {args.min_voxels} voxels ({total_voxels:,} voxels); "
        f"shard {args.shard}/{args.num_shards} takes {len(labels)} labels "
        f"({int(counts.sum()):,} voxels)")

    started = time.monotonic()
    dense = load_masked(labels, max_id)
    present = int(np.count_nonzero(dense))
    if present != int(counts.sum()):
        raise ValueError(f"masked foreground {present} != cohort voxel total {counts.sum()}")
    log(f"loaded {dense.shape} masked to cohort ({present:,} voxels); starting kimimaro")

    import kimimaro

    skeletons = kimimaro.skeletonize(
        dense,
        teasar_params=dict(V.TEASAR_PARAMS),
        anisotropy=SPACING_NM,
        dust_threshold=1,
        progress=False,
        parallel=args.parallel,
        parallel_chunk_size=1,
        fix_branching=True,
        fix_borders=True,
    )
    del dense
    log(f"kimimaro returned {len(skeletons)} skeletons in {time.monotonic() - started:.0f}s")

    missing = sorted(set(map(int, labels)) - set(map(int, skeletons)))
    out_labels: list[int] = []
    vertex_blocks: list[np.ndarray] = []
    radius_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    vertex_offsets = [0]
    edge_offsets = [0]
    # A zero EDT radius is a floor artifact, not a measurement, and `arbor`
    # requires strictly positive radii. Clamp to half the finest voxel and count
    # it rather than dropping the vertex.
    floor_um = 0.5 * float(min(V.SPACING_UM_ZYX))
    clamped = 0
    for label in sorted(map(int, skeletons)):
        skeleton = skeletons[label]
        vertices, edges, radii = simplify_skeleton(
            skeleton.vertices, skeleton.edges, skeleton.radii, args.simplification_nm
        )
        vertices = np.asarray((vertices + 0.5 * SPACING_NM) / 1000.0, dtype=np.float32)
        radii = np.asarray(radii / 1000.0, dtype=np.float32)
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        if not len(vertices) or not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(radii)):
            raise ValueError(f"invalid skeleton for label {label}")
        if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
            raise ValueError(f"invalid edge index for label {label}")
        low = radii < floor_um
        clamped += int(np.count_nonzero(low))
        radii[low] = floor_um
        out_labels.append(label)
        vertex_blocks.append(vertices)
        radius_blocks.append(radii)
        edge_blocks.append(edges)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        edge_offsets.append(edge_offsets[-1] + len(edges))

    metadata = {
        "volume": V.NAME,
        "segmentation": str(V.SEG),
        "segmentation_sha256": sha256(V.SEG),
        "shape_zyx": list(V.SHAPE_ZYX),
        "spacing_nm_zyx": SPACING_NM.tolist(),
        "min_voxels": args.min_voxels,
        "cohort_labels": int(len(labels)),
        "cohort_voxels": int(counts.sum()),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "max_voxels": args.max_voxels,
        "labels_skipped_oversized": oversized,
        "skeleton_count": len(out_labels),
        "labels_without_skeleton": missing,
        "vertex_count": int(vertex_offsets[-1]),
        "edge_count": int(edge_offsets[-1]),
        "radii_clamped_to_floor": clamped,
        "radius_floor_um": floor_um,
        "teasar_params": dict(V.TEASAR_PARAMS),
        "simplification_nm": args.simplification_nm,
        "fix_branching": True,
        "fix_borders": True,
        "dust_threshold": 1,
        "parallel": args.parallel,
        "coordinate_convention": "vertices_um_zyx = (voxel_index_zyx + 0.5) * spacing_um_zyx",
        "edge_indexing": "edges are local to each label's vertex block",
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "TEASAR forest topology is imposed, not biological evidence.",
            "No ground truth exists for any moe volume; nothing here is a score.",
            "Physical scale is conditional on the documented expansion factor.",
        ],
    }
    temporary = args.output.with_suffix(".TMP.npz")
    np.savez_compressed(
        temporary,
        labels=np.asarray(out_labels, dtype=np.uint32),
        voxel_counts=counts[np.isin(labels, np.asarray(out_labels, dtype=np.uint32))],
        vertex_offset=np.asarray(vertex_offsets, dtype=np.int64),
        edge_offset=np.asarray(edge_offsets, dtype=np.int64),
        vertices_um_zyx=np.concatenate(vertex_blocks) if vertex_blocks else np.zeros((0, 3), np.float32),
        radii_um=np.concatenate(radius_blocks) if radius_blocks else np.zeros(0, np.float32),
        edges=np.concatenate(edge_blocks) if edge_blocks else np.zeros((0, 2), np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    # Name the sidecar after the bundle, or a smoke run silently clobbers the
    # real run's provenance while leaving its .npz in place.
    args.output.with_name(args.output.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    log(
        f"wrote {args.output} -- {len(out_labels)} skeletons, "
        f"{metadata['vertex_count']:,} vertices, {metadata['edge_count']:,} edges, "
        f"{len(missing)} labels without a skeleton, {clamped:,} radii clamped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
