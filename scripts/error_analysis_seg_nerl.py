"""Error analysis for the NISB BANIS-style 200k segmentation.

The default inputs are the 200k cc-numba decode and its NERL per-GT output:

  outputs/nisb_base_banis/20260427_095218/results_step=00200000/
    img_x1_ch0-1-2_ckpt-step=00200000_decoding_affinity_cc_numba-0-0.75.h5
    evaluation_metrics_img_x1_ch0-1-2_ckpt-step=00200000_decoding_affinity_cc_numba-0-0.75_nerl_per_gt_erl.npz

Outputs:
  - per_gt_nerl.csv: GT skeleton ERL ranking. Near-zero nERL is the strongest
    false-merge suspect signal for this benchmark.
  - z_discontinuities.csv: sampled z->z+1 label churn from the predicted
    segmentation, including large births/deaths and same-label continuity.
  - gt_segment_samples.csv and segment_skeleton_ownership.csv when the
    seed101 skeleton pickle is available. These map bad GT skeletons to the
    predicted segment IDs that own their sampled nodes, and identify predicted
    segments that contain nodes from multiple GT skeletons.
  - report.md: compact summary with the highest-signal tables.

This script intentionally avoids loading the full 3000x3000x1350 segmentation
into memory.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

RESULT_DIR = Path("outputs/nisb_base_banis/20260427_095218/results_step=00200000")
DEFAULT_SEG = RESULT_DIR / "img_x1_ch0-1-2_ckpt-step=00200000_decoding_affinity_cc_numba-0-0.75.h5"
DEFAULT_NERL = RESULT_DIR / (
    "evaluation_metrics_img_x1_ch0-1-2_ckpt-step=00200000"
    "_decoding_affinity_cc_numba-0-0.75_nerl_per_gt_erl.npz"
)
DEFAULT_SKELETON = Path("/projects/weilab/dataset/nisb/base/test/seed101/skeleton.pkl")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_axis(axis: int, ndim: int) -> int:
    axis = axis + ndim if axis < 0 else axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} is invalid for ndim={ndim}")
    return axis


def reorder_coords(coords: np.ndarray, source_order: str, target_order: str) -> np.ndarray:
    source_order = source_order.lower()
    target_order = target_order.lower()
    if len(source_order) != len(target_order) or set(source_order) != set(target_order):
        raise ValueError(f"Cannot reorder coordinates from {source_order!r} to {target_order!r}")
    return coords[:, [source_order.index(axis) for axis in target_order]]


def fmt_top(items: Iterable[tuple[int, int | float]], limit: int) -> str:
    out = []
    for label, value in list(items)[:limit]:
        if isinstance(value, float):
            out.append(f"{int(label)}:{value:.4g}")
        else:
            out.append(f"{int(label)}:{int(value)}")
    return ";".join(out)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_nerl_rows(path: Path, false_merge_threshold: float) -> list[dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    gt_ids = np.asarray(data["gt_segment_id"], dtype=np.int64)
    erl = np.asarray(data["erl"], dtype=np.float64)
    if erl.ndim != 2 or erl.shape[1] != 2:
        raise ValueError(f"Expected erl shape [N, 2], got {erl.shape}")
    if len(gt_ids) != len(erl):
        raise ValueError(f"gt_segment_id length {len(gt_ids)} != erl length {len(erl)}")

    pred_erl = erl[:, 0]
    gt_erl = erl[:, 1]
    nerl = np.divide(
        pred_erl,
        gt_erl,
        out=np.full_like(pred_erl, np.nan, dtype=np.float64),
        where=gt_erl > 0,
    )

    rows: list[dict[str, object]] = []
    order = np.argsort(np.nan_to_num(nerl, nan=np.inf))
    for rank, i in enumerate(order, start=1):
        score = float(nerl[i])
        rows.append(
            {
                "rank_low_to_high": rank,
                "gt_segment_id": int(gt_ids[i]),
                "pred_erl": float(pred_erl[i]),
                "gt_erl": float(gt_erl[i]),
                "nerl": score,
                "false_merge_suspect": bool(score <= false_merge_threshold),
            }
        )
    return rows


def read_z_pair(dataset: h5py.Dataset, z_axis: int, z: int) -> tuple[np.ndarray, np.ndarray]:
    slc: list[slice | int] = [slice(None)] * dataset.ndim
    slc[z_axis] = slice(z, z + 2)
    block = np.asarray(dataset[tuple(slc)])
    return np.take(block, 0, axis=z_axis), np.take(block, 1, axis=z_axis)


def label_counts(slice_arr: np.ndarray, ignore_label: int) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(slice_arr, return_counts=True)
    keep = labels != ignore_label
    return labels[keep].astype(np.int64, copy=False), counts[keep].astype(np.int64, copy=False)


def top_count_pairs(labels: np.ndarray, counts: np.ndarray, limit: int) -> list[tuple[int, int]]:
    if labels.size == 0:
        return []
    order = np.argsort(counts)[-limit:][::-1]
    return [(int(labels[i]), int(counts[i])) for i in order]


def summarize_z_pair(
    a: np.ndarray,
    b: np.ndarray,
    *,
    z: int,
    ignore_label: int,
    large_area: int,
    area_jump_factor: float,
    top_k: int,
) -> dict[str, object]:
    labels_a, counts_a = label_counts(a, ignore_label)
    labels_b, counts_b = label_counts(b, ignore_label)

    nz_a = a != ignore_label
    nz_b = b != ignore_label
    union_nz = nz_a | nz_b
    both_nz = nz_a & nz_b
    union_count = int(union_nz.sum())
    both_count = int(both_nz.sum())
    same_count = int(((a == b) & both_nz).sum())
    changed_count = int(((a != b) & both_nz).sum())

    birth_mask = ~np.isin(labels_b, labels_a, assume_unique=True)
    death_mask = ~np.isin(labels_a, labels_b, assume_unique=True)
    birth_labels = labels_b[birth_mask]
    birth_counts = counts_b[birth_mask]
    death_labels = labels_a[death_mask]
    death_counts = counts_a[death_mask]

    common, ia, ib = np.intersect1d(labels_a, labels_b, assume_unique=True, return_indices=True)
    jump_pairs: list[tuple[int, float]] = []
    if common.size:
        ca = counts_a[ia].astype(np.float64)
        cb = counts_b[ib].astype(np.float64)
        small = np.minimum(ca, cb)
        large = np.maximum(ca, cb)
        ratios = np.divide(large, small, out=np.full_like(large, np.inf), where=small > 0)
        jump_keep = (small >= large_area) & (ratios >= area_jump_factor)
        jump_order = np.argsort(ratios[jump_keep])[-top_k:][::-1]
        jump_labels = common[jump_keep]
        jump_ratios = ratios[jump_keep]
        jump_pairs = [(int(jump_labels[i]), float(jump_ratios[i])) for i in jump_order]

    large_birth = birth_counts >= large_area
    large_death = death_counts >= large_area
    birth_voxels = int(birth_counts.sum())
    death_voxels = int(death_counts.sum())

    return {
        "z": int(z),
        "z_next": int(z + 1),
        "union_nonzero_voxels": union_count,
        "same_label_fraction": same_count / union_count if union_count else np.nan,
        "changed_fraction_both_nonzero": changed_count / both_count if both_count else np.nan,
        "birth_voxels": birth_voxels,
        "death_voxels": death_voxels,
        "birth_fraction": birth_voxels / union_count if union_count else np.nan,
        "death_fraction": death_voxels / union_count if union_count else np.nan,
        "num_labels_z": int(labels_a.size),
        "num_labels_z_next": int(labels_b.size),
        "num_large_birth_labels": int(large_birth.sum()),
        "num_large_death_labels": int(large_death.sum()),
        "top_birth_labels": fmt_top(top_count_pairs(birth_labels, birth_counts, top_k), top_k),
        "top_death_labels": fmt_top(top_count_pairs(death_labels, death_counts, top_k), top_k),
        "top_area_jump_labels": fmt_top(jump_pairs, top_k),
    }


def scan_z_discontinuities(
    dataset: h5py.Dataset,
    *,
    z_axis: int,
    z_stride: int,
    max_z_pairs: int | None,
    ignore_label: int,
    large_area: int,
    area_jump_factor: float,
    top_k: int,
) -> list[dict[str, object]]:
    z_len = dataset.shape[z_axis]
    zs = np.arange(0, z_len - 1, max(1, z_stride), dtype=np.int64)
    if max_z_pairs is not None and zs.size > max_z_pairs:
        keep = np.linspace(0, zs.size - 1, max_z_pairs, dtype=np.int64)
        zs = zs[keep]

    rows: list[dict[str, object]] = []
    for i, z in enumerate(zs, start=1):
        a, b = read_z_pair(dataset, z_axis, int(z))
        rows.append(
            summarize_z_pair(
                a,
                b,
                z=int(z),
                ignore_label=ignore_label,
                large_area=large_area,
                area_jump_factor=area_jump_factor,
                top_k=top_k,
            )
        )
        if i == 1 or i == len(zs) or i % 10 == 0:
            print(f"  z-discontinuity scan {i}/{len(zs)}: z={int(z)}", flush=True)
    return rows


def load_skeleton_coords(
    path: Path,
    *,
    gt_ids: set[int],
    id_key: str,
    position_key: str,
    skeleton_coord_order: str,
    dataset_axis_order: str,
    max_nodes_per_gt: int,
) -> dict[int, np.ndarray]:
    with path.open("rb") as f:
        graph = pickle.load(f)

    coords_by_gt: dict[int, list[np.ndarray]] = defaultdict(list)
    for _node_id, node_data in graph.nodes(data=True):
        gt_id = int(node_data[id_key])
        if gt_id not in gt_ids:
            continue
        coords_by_gt[gt_id].append(np.asarray(node_data[position_key], dtype=np.int64))

    sampled: dict[int, np.ndarray] = {}
    for gt_id, coords_list in coords_by_gt.items():
        coords = np.vstack(coords_list)
        coords = reorder_coords(coords, skeleton_coord_order, dataset_axis_order)
        if max_nodes_per_gt > 0 and len(coords) > max_nodes_per_gt:
            idx = np.linspace(0, len(coords) - 1, max_nodes_per_gt, dtype=np.int64)
            coords = coords[idx]
        sampled[gt_id] = coords.astype(np.int64, copy=False)
    return sampled


def sample_dataset_at_coords(dataset: h5py.Dataset, coords: np.ndarray) -> np.ndarray:
    """Chunk-grouped point sampling for a 3D HDF5 dataset."""
    if coords.size == 0:
        return np.empty(0, dtype=dataset.dtype)

    shape = np.asarray(dataset.shape, dtype=np.int64)
    chunks = np.asarray(dataset.chunks or (64, 64, 64), dtype=np.int64)
    valid = np.all((coords >= 0) & (coords < shape), axis=1)
    valid_coords = coords[valid]
    labels = np.full(coords.shape[0], 0, dtype=dataset.dtype)
    if valid_coords.size == 0:
        return labels

    chunk_ids = valid_coords // chunks
    order = np.lexsort((chunk_ids[:, 2], chunk_ids[:, 1], chunk_ids[:, 0]))
    sorted_coords = valid_coords[order]
    sorted_chunks = chunk_ids[order]
    sampled_valid = np.empty(len(valid_coords), dtype=dataset.dtype)

    start = 0
    while start < len(sorted_coords):
        end = start + 1
        while end < len(sorted_coords) and np.array_equal(sorted_chunks[end], sorted_chunks[start]):
            end += 1

        chunk_id = sorted_chunks[start]
        lo = chunk_id * chunks
        hi = np.minimum(lo + chunks, shape)
        block = np.asarray(dataset[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
        local = sorted_coords[start:end] - lo
        sampled_valid[order[start:end]] = block[local[:, 0], local[:, 1], local[:, 2]]
        start = end

    labels[np.flatnonzero(valid)] = sampled_valid
    return labels


def summarize_skeleton_ownership(
    dataset: h5py.Dataset,
    nerl_rows: list[dict[str, object]],
    *,
    skeleton_path: Path,
    id_key: str,
    position_key: str,
    skeleton_coord_order: str,
    dataset_axis_order: str,
    max_nodes_per_gt: int,
    false_merge_threshold: float,
    min_owner_nodes: int,
    min_owner_fraction: float,
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nerl_by_gt = {int(row["gt_segment_id"]): float(row["nerl"]) for row in nerl_rows}
    gt_erl_by_gt = {int(row["gt_segment_id"]): float(row["gt_erl"]) for row in nerl_rows}
    pred_erl_by_gt = {int(row["gt_segment_id"]): float(row["pred_erl"]) for row in nerl_rows}
    gt_ids = set(nerl_by_gt)

    coords_by_gt = load_skeleton_coords(
        skeleton_path,
        gt_ids=gt_ids,
        id_key=id_key,
        position_key=position_key,
        skeleton_coord_order=skeleton_coord_order,
        dataset_axis_order=dataset_axis_order,
        max_nodes_per_gt=max_nodes_per_gt,
    )

    gt_rows: list[dict[str, object]] = []
    owners: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for i, gt_id in enumerate(sorted(gt_ids), start=1):
        coords = coords_by_gt.get(gt_id)
        if coords is None or coords.size == 0:
            gt_rows.append(
                {
                    "gt_segment_id": gt_id,
                    "nerl": nerl_by_gt[gt_id],
                    "pred_erl": pred_erl_by_gt[gt_id],
                    "gt_erl": gt_erl_by_gt[gt_id],
                    "sampled_nodes": 0,
                    "num_pred_segments": 0,
                    "dominant_segment_id": "",
                    "dominant_fraction": np.nan,
                    "top_pred_segments": "",
                }
            )
            continue

        labels = sample_dataset_at_coords(dataset, coords)
        labels, counts = np.unique(labels[labels != 0], return_counts=True)
        order = np.argsort(counts)[::-1]
        labels = labels[order].astype(np.int64, copy=False)
        counts = counts[order].astype(np.int64, copy=False)
        total = int(counts.sum())
        dominant = int(labels[0]) if labels.size else ""
        dominant_fraction = float(counts[0] / total) if total and labels.size else np.nan

        for label, count in zip(labels, counts):
            fraction = float(count / total) if total else 0.0
            if int(count) >= min_owner_nodes and fraction >= min_owner_fraction:
                owners[int(label)].append((gt_id, int(count), fraction, nerl_by_gt[gt_id]))

        gt_rows.append(
            {
                "gt_segment_id": gt_id,
                "nerl": nerl_by_gt[gt_id],
                "pred_erl": pred_erl_by_gt[gt_id],
                "gt_erl": gt_erl_by_gt[gt_id],
                "sampled_nodes": total,
                "num_pred_segments": int(labels.size),
                "dominant_segment_id": dominant,
                "dominant_fraction": dominant_fraction,
                "top_pred_segments": fmt_top(zip(labels, counts), top_k),
            }
        )
        if i == 1 or i == len(gt_ids) or i % 25 == 0:
            print(f"  skeleton ownership sampling {i}/{len(gt_ids)}", flush=True)

    segment_rows: list[dict[str, object]] = []
    for segment_id, owner_list in owners.items():
        owner_list = sorted(owner_list, key=lambda x: (x[3], -x[1]))
        bad = [x for x in owner_list if x[3] <= false_merge_threshold]
        segment_rows.append(
            {
                "pred_segment_id": int(segment_id),
                "num_gt_skeletons": len(owner_list),
                "num_false_merge_suspect_gt": len(bad),
                "total_sampled_nodes": int(sum(x[1] for x in owner_list)),
                "min_nerl": float(min(x[3] for x in owner_list)),
                "top_gt_by_low_nerl": ";".join(
                    f"{gt}:{nerl:.4g}:{count}" for gt, count, _frac, nerl in owner_list[:top_k]
                ),
                "owner_gt_ids": ";".join(str(gt) for gt, _count, _frac, _nerl in owner_list),
            }
        )
    segment_rows.sort(
        key=lambda r: (
            -int(r["num_false_merge_suspect_gt"]),
            -int(r["num_gt_skeletons"]),
            float(r["min_nerl"]),
        )
    )
    gt_rows.sort(key=lambda r: float(r["nerl"]))
    return gt_rows, segment_rows


def write_report(
    path: Path,
    *,
    seg_path: Path,
    nerl_path: Path,
    dataset: h5py.Dataset,
    nerl_rows: list[dict[str, object]],
    z_rows: list[dict[str, object]],
    gt_sample_rows: list[dict[str, object]] | None,
    segment_rows: list[dict[str, object]] | None,
    false_merge_threshold: float,
    output_files: list[Path],
) -> None:
    ratios = np.asarray([float(row["nerl"]) for row in nerl_rows], dtype=np.float64)
    false_merge_count = int(np.sum(ratios <= false_merge_threshold))
    zeroish_count = int(np.sum(ratios <= 1.0e-6))
    q = np.nanquantile(ratios, [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    z_rank = sorted(
        z_rows,
        key=lambda r: (
            -float(r["birth_fraction"]),
            -float(r["death_fraction"]),
            float(r["same_label_fraction"]),
        ),
    )
    merge_segments = (
        [r for r in segment_rows or [] if int(r["num_gt_skeletons"]) >= 2]
        if segment_rows is not None
        else []
    )

    lines = [
        "# NISB Error Analysis",
        "",
        f"- segmentation: `{seg_path}`",
        f"- nerl per-GT: `{nerl_path}`",
        f"- dataset: `/main` shape={tuple(dataset.shape)} dtype={dataset.dtype} "
        f"chunks={dataset.chunks} compression={dataset.compression}",
        "",
        "## NERL Per-GT Summary",
        "",
        f"- GT skeletons: {len(nerl_rows)}",
        f"- nERL <= {false_merge_threshold:g}: {false_merge_count}",
        f"- nERL <= 1e-6: {zeroish_count}",
        "- nERL quantiles " f"[0,1,5,10,25,50,75,90,100]%: {', '.join(f'{x:.4g}' for x in q)}",
        "",
        "Lowest nERL GT skeletons:",
        "",
        "| rank | gt | nerl | pred_erl | gt_erl |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in nerl_rows[:10]:
        lines.append(
            f"| {row['rank_low_to_high']} | {row['gt_segment_id']} | "
            f"{float(row['nerl']):.4g} | {float(row['pred_erl']):.4g} | "
            f"{float(row['gt_erl']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Z Discontinuity Scan",
            "",
            "Highest birth/death label-churn z pairs:",
            "",
            "| z->z+1 | same-label frac | birth frac | death frac | large births | large deaths |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in z_rank[:10]:
        lines.append(
            f"| {row['z']}->{row['z_next']} | {float(row['same_label_fraction']):.4f} | "
            f"{float(row['birth_fraction']):.4f} | {float(row['death_fraction']):.4f} | "
            f"{row['num_large_birth_labels']} | {row['num_large_death_labels']} |"
        )

    if segment_rows is not None:
        lines.extend(
            [
                "",
                "## Skeleton Ownership",
                "",
                f"- predicted segments owning >=2 GT skeletons: {len(merge_segments)}",
                "",
                "| pred segment | GT skeletons | bad GTs | min nERL | low-nERL GT:score:nodes |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in merge_segments[:15]:
            lines.append(
                f"| {row['pred_segment_id']} | {row['num_gt_skeletons']} | "
                f"{row['num_false_merge_suspect_gt']} | {float(row['min_nerl']):.4g} | "
                f"{row['top_gt_by_low_nerl']} |"
            )

    if gt_sample_rows is not None:
        lines.extend(
            [
                "",
                "Lowest-nERL GT skeleton sampled segment owners:",
                "",
                "| gt | nERL | sampled nodes | dominant pred segment | "
                "dominant frac | top pred segments |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in gt_sample_rows[:15]:
            lines.append(
                f"| {row['gt_segment_id']} | {float(row['nerl']):.4g} | "
                f"{row['sampled_nodes']} | {row['dominant_segment_id']} | "
                f"{float(row['dominant_fraction']):.4f} | {row['top_pred_segments']} |"
            )

    lines.extend(["", "## Output Files", ""])
    for out in output_files:
        lines.append(f"- `{out}`")
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seg", type=Path, default=DEFAULT_SEG, help="decoded segmentation H5")
    parser.add_argument("--nerl", type=Path, default=DEFAULT_NERL, help="NERL per-GT NPZ")
    parser.add_argument("--dataset", default="main", help="H5 dataset key")
    parser.add_argument("--out-dir", type=Path, default=RESULT_DIR / "err_analysis")
    parser.add_argument(
        "--false-merge-threshold",
        type=float,
        default=0.01,
        help="nERL <= this is counted as a severe false-merge suspect",
    )
    parser.add_argument("--z-axis", type=int, default=-1, help="axis treated as z")
    parser.add_argument("--z-stride", type=int, default=10, help="sample every Nth z pair")
    parser.add_argument(
        "--max-z-pairs",
        type=int,
        default=None,
        help="cap z-pair checks by evenly subsampling the z-stride sequence",
    )
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument(
        "--large-area",
        type=int,
        default=10_000,
        help="2D label area threshold for large births/deaths and area jumps",
    )
    parser.add_argument(
        "--area-jump-factor",
        type=float,
        default=4.0,
        help="flag same label when adjacent-slice area changes by this factor",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=DEFAULT_SKELETON,
        help="optional NetworkX skeleton pickle for GT->pred segment ownership",
    )
    parser.add_argument("--skip-skeleton", action="store_true")
    parser.add_argument("--skeleton-id-key", default="id")
    parser.add_argument("--skeleton-position-key", default="index_position")
    parser.add_argument(
        "--skeleton-coord-order",
        default="xyz",
        help="axis order of skeleton index_position coordinates",
    )
    parser.add_argument(
        "--dataset-axis-order",
        default="xyz",
        help="axis order of the segmentation H5 dataset dimensions",
    )
    parser.add_argument(
        "--max-nodes-per-gt",
        type=int,
        default=512,
        help="deterministic per-GT skeleton node sample; <=0 keeps all nodes",
    )
    parser.add_argument(
        "--min-owner-nodes",
        type=int,
        default=5,
        help="minimum sampled nodes for assigning a predicted segment to a GT skeleton",
    )
    parser.add_argument(
        "--min-owner-fraction",
        type=float,
        default=0.01,
        help="minimum sampled-node fraction for assigning a predicted segment to a GT skeleton",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"seg:  {args.seg}")
    print(f"nerl: {args.nerl}")
    print(f"out:  {out_dir}")

    nerl_rows = load_nerl_rows(args.nerl, args.false_merge_threshold)
    per_gt_path = out_dir / "per_gt_nerl.csv"
    write_csv(
        per_gt_path,
        nerl_rows,
        [
            "rank_low_to_high",
            "gt_segment_id",
            "pred_erl",
            "gt_erl",
            "nerl",
            "false_merge_suspect",
        ],
    )
    ratios = np.asarray([float(row["nerl"]) for row in nerl_rows], dtype=np.float64)
    print(
        f"NERL: {len(nerl_rows)} GT skeletons; "
        f"{int(np.sum(ratios <= args.false_merge_threshold))} <= {args.false_merge_threshold:g}; "
        f"{int(np.sum(ratios <= 1.0e-6))} <= 1e-6"
    )

    output_files = [per_gt_path]
    gt_sample_rows = None
    segment_rows = None
    with h5py.File(args.seg, "r") as f:
        dataset = f[args.dataset]
        if dataset.ndim != 3:
            raise ValueError(f"Expected 3D segmentation dataset, got shape {dataset.shape}")
        z_axis = normalize_axis(args.z_axis, dataset.ndim)
        print(
            f"H5 /{args.dataset}: shape={dataset.shape} dtype={dataset.dtype} "
            f"chunks={dataset.chunks} compression={dataset.compression}"
        )

        print("Scanning z discontinuities...", flush=True)
        z_rows = scan_z_discontinuities(
            dataset,
            z_axis=z_axis,
            z_stride=args.z_stride,
            max_z_pairs=args.max_z_pairs,
            ignore_label=args.ignore_label,
            large_area=args.large_area,
            area_jump_factor=args.area_jump_factor,
            top_k=args.top_k,
        )
        z_path = out_dir / "z_discontinuities.csv"
        write_csv(
            z_path,
            z_rows,
            [
                "z",
                "z_next",
                "union_nonzero_voxels",
                "same_label_fraction",
                "changed_fraction_both_nonzero",
                "birth_voxels",
                "death_voxels",
                "birth_fraction",
                "death_fraction",
                "num_labels_z",
                "num_labels_z_next",
                "num_large_birth_labels",
                "num_large_death_labels",
                "top_birth_labels",
                "top_death_labels",
                "top_area_jump_labels",
            ],
        )
        output_files.append(z_path)

        if not args.skip_skeleton:
            if args.skeleton.exists():
                print(f"Sampling skeleton ownership from {args.skeleton}...", flush=True)
                gt_sample_rows, segment_rows = summarize_skeleton_ownership(
                    dataset,
                    nerl_rows,
                    skeleton_path=args.skeleton,
                    id_key=args.skeleton_id_key,
                    position_key=args.skeleton_position_key,
                    skeleton_coord_order=args.skeleton_coord_order,
                    dataset_axis_order=args.dataset_axis_order,
                    max_nodes_per_gt=args.max_nodes_per_gt,
                    false_merge_threshold=args.false_merge_threshold,
                    min_owner_nodes=args.min_owner_nodes,
                    min_owner_fraction=args.min_owner_fraction,
                    top_k=args.top_k,
                )
                gt_sample_path = out_dir / "gt_segment_samples.csv"
                write_csv(
                    gt_sample_path,
                    gt_sample_rows,
                    [
                        "gt_segment_id",
                        "nerl",
                        "pred_erl",
                        "gt_erl",
                        "sampled_nodes",
                        "num_pred_segments",
                        "dominant_segment_id",
                        "dominant_fraction",
                        "top_pred_segments",
                    ],
                )
                segment_path = out_dir / "segment_skeleton_ownership.csv"
                write_csv(
                    segment_path,
                    segment_rows,
                    [
                        "pred_segment_id",
                        "num_gt_skeletons",
                        "num_false_merge_suspect_gt",
                        "total_sampled_nodes",
                        "min_nerl",
                        "top_gt_by_low_nerl",
                        "owner_gt_ids",
                    ],
                )
                output_files.extend([gt_sample_path, segment_path])
            else:
                print(f"Skeleton file not found, skipping ownership analysis: {args.skeleton}")

        report_path = out_dir / "report.md"
        write_report(
            report_path,
            seg_path=args.seg,
            nerl_path=args.nerl,
            dataset=dataset,
            nerl_rows=nerl_rows,
            z_rows=z_rows,
            gt_sample_rows=gt_sample_rows,
            segment_rows=segment_rows,
            false_merge_threshold=args.false_merge_threshold,
            output_files=[*output_files, report_path],
        )
        output_files.append(report_path)

    print("\nWrote:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
