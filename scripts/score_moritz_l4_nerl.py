#!/usr/bin/env python3
"""NERL + skeleton VOI for a Moritz L4 segmentation, against the 96 manual skeletons.

    python s3_score.py <precomputed_seg_dir> [--out score.json] [--workers 8]

FRAME. The .nml skeletons are in the GLOBAL frame and the segmentation is in the VOLUME
frame, whose origin is global (118, 0, 0) zyx -- so a node at nml z maps to volume
z - 118. This is not a detail: scoring in the wrong frame gives a plausible number
(~0) for a good segmentation. The check that it is right is the node count -- 263938 of
264926 nodes land in bounds under this mapping, and 258764 under the identity.

WHAT IS REPORTED
  base           NERL and skeleton VOI as segmented.
  merge-oracle   every false merge cut: each predicted segment is split by the GT
                 skeleton it covers. Splits are untouched. This is the ceiling that
                 merge repair alone could reach, and the previous unmasked run scored
                 0.9158 here against a base of 0.0007 -- fragments fine, merges fatal.
  nodes on background are reported, not masked away: with an exclusion mask in the
  decode, a GT node inside a masked vessel legitimately has no segment.

The join-oracle the earlier (deleted) version of this script reported is NOT
reimplemented here -- its definition is not recoverable from the log, and a
differently-defined number under the same name is worse than no number.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SKELETONS = Path("/projects/weilab/dataset/segEM/Moritz_l4_2019/manual-neuron-reconstructions")
RESOLUTION_ZYX_NM = (28.0, 11.24, 11.24)
VOLUME_SHAPE_ZYX = (3306, 8534, 5599)
GLOBAL_ORIGIN_ZYX = (118, 0, 0)


def parse_nml(path: Path):
    """One skeleton per file: nodes and edges of every <thing>, in GLOBAL xyz voxels."""
    root = ET.parse(path).getroot()
    coords: dict[int, tuple[int, int, int]] = {}
    edges: list[tuple[int, int]] = []
    for thing in root.iter("thing"):
        for node in thing.iter("node"):
            nid = int(node.attrib["id"])
            if nid in coords:
                raise ValueError(f"{path}: duplicate node id {nid}")
            coords[nid] = (int(node.attrib["x"]), int(node.attrib["y"]), int(node.attrib["z"]))
        for edge in thing.iter("edge"):
            edges.append((int(edge.attrib["source"]), int(edge.attrib["target"])))
    return coords, edges


def build_graph(
    skeleton_dir: Path,
    *,
    volume_shape_zyx=VOLUME_SHAPE_ZYX,
    global_origin_zyx=GLOBAL_ORIGIN_ZYX,
    resolution_zyx_nm=RESOLUTION_ZYX_NM,
):
    """Flat ERLGraph arrays in the VOLUME frame, out-of-bounds nodes dropped."""
    files = sorted(skeleton_dir.glob("*.nml"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not files:
        raise SystemExit(f"no .nml under {skeleton_dir}")

    node_coords, node_skel, edge_u, edge_v, edge_len, edge_ptr = [], [], [], [], [], [0]
    skeleton_id, skeleton_len = [], []
    total_nodes = 0
    res = np.asarray(resolution_zyx_nm)

    for index, path in enumerate(files):
        coords, edges = parse_nml(path)
        total_nodes += len(coords)
        keep = {}
        for nid, (x, y, z) in coords.items():
            zyx = (z - global_origin_zyx[0], y - global_origin_zyx[1], x - global_origin_zyx[2])
            if all(0 <= zyx[a] < volume_shape_zyx[a] for a in range(3)):
                keep[nid] = len(node_coords)
                node_coords.append(zyx)
                node_skel.append(index)
        length = 0.0
        for source, target in edges:
            if source in keep and target in keep:
                u, v = keep[source], keep[target]
                d = float(
                    np.sqrt((((np.asarray(node_coords[u]) - node_coords[v]) * res) ** 2).sum())
                )
                edge_u.append(u)
                edge_v.append(v)
                edge_len.append(d)
                length += d
        edge_ptr.append(len(edge_u))
        skeleton_id.append(index + 1)
        skeleton_len.append(length)

    sys.path.insert(0, str(REPO))
    from connectomics.metrics.nerl import import_em_erl

    ERLGraph, _, _ = import_em_erl()
    graph = ERLGraph(
        skeleton_id=np.asarray(skeleton_id, dtype=np.int64),
        skeleton_len=np.asarray(skeleton_len, dtype=np.float64),
        node_skeleton_index=np.asarray(node_skel, dtype=np.int64),
        node_coords_zyx=np.asarray(node_coords, dtype=np.int64),
        edge_u=np.asarray(edge_u, dtype=np.int64),
        edge_v=np.asarray(edge_v, dtype=np.int64),
        edge_len=np.asarray(edge_len, dtype=np.float64),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
    )
    print(f"graph: {len(files)} skeletons, {len(node_coords)}/{total_nodes} nodes in bounds, "
          f"total {sum(skeleton_len) / 1e6:.4f} mm", flush=True)
    return graph


def read_node_segments(seg_path: str, coords_zyx: np.ndarray, workers: int) -> np.ndarray:
    """Segment id at every node, reading each storage chunk exactly once."""
    from cloudvolume import CloudVolume

    path = seg_path if "://" in seg_path else "file://" + str(Path(seg_path).resolve())
    probe = CloudVolume(path, mip=0, bounded=True, progress=False)
    chunk_xyz = np.asarray(probe.chunk_size, dtype=np.int64)          # (X, Y, Z)
    offset_xyz = np.asarray(probe.voxel_offset, dtype=np.int64)

    xyz = coords_zyx[:, ::-1]
    keys = (xyz - offset_xyz) // chunk_xyz
    order = np.lexsort((keys[:, 0], keys[:, 1], keys[:, 2]))
    out = np.zeros(len(xyz), dtype=np.uint64)
    groups = collections.defaultdict(list)
    for i in order:
        groups[tuple(keys[i].tolist())].append(i)
    print(f"reading segment ids at node voxels from {path}\n"
          f"  {len(xyz)} nodes, {len(groups)} storage chunks to read", flush=True)

    items = list(groups.items())
    done = 0

    def one(item):
        key, idx = item
        start = offset_xyz + np.asarray(key) * chunk_xyz
        stop = np.minimum(start + chunk_xyz, offset_xyz + np.asarray(probe.shape[:3]))
        vol = CloudVolume(path, mip=0, bounded=True, progress=False)
        block = np.asarray(vol[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]])[..., 0]
        local = xyz[idx] - start
        return idx, block[local[:, 0], local[:, 1], local[:, 2]]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, values in pool.map(one, items):
            out[idx] = values
            done += 1
            if done % 5000 == 0:
                print(f"    {done}/{len(items)} chunks", flush=True)
    return out


def score(graph, lut, merge_threshold):
    sys.path.insert(0, str(REPO))
    from connectomics.metrics.nerl import extract_nerl_score_outputs, import_em_erl

    _, compute_erl_score, _ = import_em_erl()
    result = compute_erl_score(graph, lut, None, merge_threshold=int(merge_threshold))
    result.compute_erl()
    pred_erl, gt_erl, _, _ = extract_nerl_score_outputs(result)
    return float(pred_erl / gt_erl) if gt_erl > 0 else float("nan")


def merge_oracle_lut(graph, lut):
    """Cut every false merge: split each predicted segment by the GT skeleton it covers."""
    pairs = np.stack([np.asarray(graph.node_skeleton_index), np.asarray(lut).astype(np.int64)], 1)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    out = (inverse + 1).astype(np.uint64)
    out[np.asarray(lut) == 0] = 0        # background stays background
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--skeletons", default=str(SKELETONS))
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--volume-shape-zyx", nargs=3, type=int, default=VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"), help="segmentation extent in volume voxels",
    )
    ap.add_argument(
        "--global-origin-zyx", nargs=3, type=int, default=GLOBAL_ORIGIN_ZYX,
        metavar=("Z", "Y", "X"), help="global voxel coordinate of the volume origin",
    )
    ap.add_argument(
        "--resolution-xyz-nm", nargs=3, type=float, default=RESOLUTION_ZYX_NM[::-1],
        metavar=("X", "Y", "Z"), help="voxel size in nanometers",
    )
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from connectomics.metrics.nerl import skeleton_voi

    graph = build_graph(
        Path(a.skeletons),
        volume_shape_zyx=a.volume_shape_zyx,
        global_origin_zyx=a.global_origin_zyx,
        resolution_zyx_nm=a.resolution_xyz_nm[::-1],
    )
    lut = read_node_segments(a.seg, np.asarray(graph.node_coords_zyx), a.workers)

    background = float((lut == 0).mean())
    node_gt = np.asarray(graph.skeleton_id)[np.asarray(graph.node_skeleton_index)]
    voi_split, voi_merge, voi_total = skeleton_voi(lut, node_gt)
    oracle = merge_oracle_lut(graph, lut)

    report = {
        "segmentation": a.seg,
        "nodes": int(len(lut)),
        "nodes_on_background": background,
        "n_pred_segments": int(len(np.unique(lut[lut != 0]))),
        "voi_split": float(voi_split),
        "voi_merge": float(voi_merge),
        "voi_total": float(voi_total),
    }
    for mt in (0, 5, 50):
        report[f"nerl_mt{mt}"] = score(graph, lut, mt)
        report[f"nerl_mt{mt}_merge_oracle"] = score(graph, oracle, mt)

    print("\n" + "=" * 66)
    print(f"{'metric':<24}{'base':>12}{'merge-oracle':>16}")
    for mt in (0, 5, 50):
        print(f"nerl_mt{mt:<18}{report[f'nerl_mt{mt}']:>12.4f}"
              f"{report[f'nerl_mt{mt}_merge_oracle']:>16.4f}")
    print(f"{'voi_split':<24}{report['voi_split']:>12.4f}")
    print(f"{'voi_merge':<24}{report['voi_merge']:>12.4f}")
    print(f"{'voi_total':<24}{report['voi_total']:>12.4f}")
    print(f"{'n_pred_segments':<24}{report['n_pred_segments']:>12d}")
    print(f"{'nodes on background':<24}{background:>11.2%}")
    print("=" * 66)

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
