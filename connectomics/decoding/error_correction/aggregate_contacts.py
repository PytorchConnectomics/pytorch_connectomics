#!/usr/bin/env python3
"""Aggregate GT-free arm0_96 chunk contact rows into one exact segment RAG."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .artifacts import reject_evaluation_path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "dev" / "zebrafinch" / "arm096_error_correction" / "decoder_gtfree_v2"
DEFAULT_CHUNKS = ROOT / "contact_chunks"
DEFAULT_OUTPUT = ROOT / "contact_graph.npz"
KEY_PATTERN = re.compile(r"^z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)\.npz$")


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def key_tuple(path: Path) -> tuple[int, int, int]:
    match = KEY_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid contact chunk filename: {path}")
    return tuple(int(match.group(axis)) for axis in ("z", "y", "x"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-chunks", type=int, default=10_626)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    reject_evaluation_path(args.chunks)
    reject_evaluation_path(args.output)

    paths = sorted(args.chunks.glob("z*_y*_x*.npz"), key=key_tuple)
    if not paths:
        raise FileNotFoundError(f"no contact chunks in {args.chunks}")
    if not args.allow_partial and len(paths) != args.expected_chunks:
        raise RuntimeError(f"found {len(paths):,}/{args.expected_chunks:,} contact chunks")

    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "left",
            "right",
            "count",
            "centroid_nm_zyx",
            "axis_count_zyx",
            "normal_sum_zyx",
            "area_nm2",
            "affinity_sum",
            "affinity_sq_sum",
            "affinity_max",
            "affinity_ge_count",
        )
    }
    affinity_thresholds = None
    inventory_parts: dict[str, list[np.ndarray]] = {
        "segment_label": [],
        "touches_volume_boundary": [],
        "touches_keep_boundary": [],
    }
    boundary_schema: bool | None = None
    for number, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("gt_free") is not True or metadata.get("key") != path.stem:
                raise ValueError(f"{path}: invalid GT-free metadata")
            if int(metadata.get("schema", 0)) < 2:
                raise ValueError(f"{path}: contact chunk lacks affinity evidence")
            has_boundary = all(name in data for name in inventory_parts)
            if boundary_schema is None:
                boundary_schema = has_boundary
            elif boundary_schema != has_boundary:
                raise ValueError("contact chunks mix boundary-inventory schemas")
            if has_boundary:
                inventory_count = len(data["segment_label"])
                for name in inventory_parts:
                    values = np.asarray(data[name])
                    if len(values) != inventory_count:
                        raise ValueError(f"{path}: inconsistent {name} row count")
                    inventory_parts[name].append(values)
            count = np.asarray(data["count"], dtype=np.uint64)
            thresholds = np.asarray(data["affinity_thresholds"], dtype=np.float32)
            if affinity_thresholds is None:
                affinity_thresholds = thresholds
            elif not np.array_equal(affinity_thresholds, thresholds):
                raise ValueError(f"{path}: affinity thresholds differ across chunks")
            for name in parts:
                values = np.asarray(data[name])
                if len(values) != len(count):
                    raise ValueError(f"{path}: inconsistent {name} row count")
                parts[name].append(values)
        if number % 1_000 == 0:
            log(f"loaded {number:,}/{len(paths):,} chunks")

    arrays = {
        name: np.concatenate(values, axis=0) if values else np.zeros(0)
        for name, values in parts.items()
    }
    left = arrays["left"].astype(np.uint64, copy=False)
    right = arrays["right"].astype(np.uint64, copy=False)
    if np.any(left >= right):
        raise ValueError("contact rows are not canonical left < right pairs")
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    starts = np.r_[
        0,
        np.flatnonzero((left[1:] != left[:-1]) | (right[1:] != right[:-1])) + 1,
    ]
    count = arrays["count"][order].astype(np.uint64, copy=False)
    total_count = np.add.reduceat(count, starts)
    weighted_centroid = arrays["centroid_nm_zyx"][order] * count[:, None]
    centroid = np.add.reduceat(weighted_centroid, starts, axis=0) / total_count[:, None]
    axis_count = np.add.reduceat(arrays["axis_count_zyx"][order], starts, axis=0)
    normal_sum = np.add.reduceat(arrays["normal_sum_zyx"][order], starts, axis=0)
    area_nm2 = np.add.reduceat(arrays["area_nm2"][order], starts)
    affinity_sum = np.add.reduceat(arrays["affinity_sum"][order], starts)
    affinity_sq_sum = np.add.reduceat(arrays["affinity_sq_sum"][order], starts)
    affinity_max = np.maximum.reduceat(arrays["affinity_max"][order], starts)
    affinity_ge_count = np.add.reduceat(arrays["affinity_ge_count"][order], starts, axis=0)

    threshold_08 = np.flatnonzero(np.isclose(affinity_thresholds, 0.8))
    if len(threshold_08) != 1:
        raise ValueError("affinity thresholds must contain exactly one 0.8 entry")
    ordered_affinity_ge = arrays["affinity_ge_count"][order]
    ordered_affinity_sum = arrays["affinity_sum"][order]
    fraction_08 = ordered_affinity_ge[:, threshold_08[0]] / count
    affinity_mean = ordered_affinity_sum / count
    group = np.zeros(len(order), dtype=np.int64)
    group[starts] = 1
    group = np.cumsum(group) - 1
    best_fraction = np.maximum.reduceat(fraction_08, starts)
    eligible_best = fraction_08 == best_fraction[group]
    best_mean = np.maximum.reduceat(np.where(eligible_best, affinity_mean, -1.0), starts)
    eligible_best &= affinity_mean == best_mean[group]
    best_support = np.maximum.reduceat(np.where(eligible_best, count, 0), starts)
    eligible_best &= count == best_support[group]
    best_patch = np.minimum.reduceat(
        np.where(eligible_best, np.arange(len(order)), len(order)), starts
    )
    if np.any(best_patch >= len(order)):
        raise AssertionError("failed to choose a representative contact patch")

    boundary_output: dict[str, np.ndarray] = {}
    if boundary_schema:
        segment_label = np.concatenate(inventory_parts["segment_label"]).astype(
            np.uint64, copy=False
        )
        boundary_order = np.argsort(segment_label, kind="stable")
        segment_label = segment_label[boundary_order]
        boundary_starts = np.r_[0, np.flatnonzero(segment_label[1:] != segment_label[:-1]) + 1]
        boundary_output = {
            "segment_label": segment_label[boundary_starts],
            "touches_volume_boundary": np.logical_or.reduceat(
                np.concatenate(inventory_parts["touches_volume_boundary"])[boundary_order],
                boundary_starts,
            ),
            "touches_keep_boundary": np.logical_or.reduceat(
                np.concatenate(inventory_parts["touches_keep_boundary"])[boundary_order],
                boundary_starts,
            ),
            "complete_boundary_inventory": np.asarray(
                not args.allow_partial and len(paths) == args.expected_chunks
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        left=left[starts],
        right=right[starts],
        count=total_count,
        centroid_nm_zyx=centroid.astype(np.float32),
        axis_count_zyx=axis_count.astype(np.uint64),
        normal_sum_zyx=normal_sum.astype(np.int64),
        area_nm2=area_nm2.astype(np.float64),
        affinity_sum=affinity_sum.astype(np.float64),
        affinity_sq_sum=affinity_sq_sum.astype(np.float64),
        affinity_max=affinity_max.astype(np.float32),
        affinity_ge_count=affinity_ge_count.astype(np.uint64),
        affinity_thresholds=affinity_thresholds,
        best_patch_count=count[best_patch],
        best_patch_centroid_nm_zyx=arrays["centroid_nm_zyx"][order][best_patch].astype(np.float32),
        best_patch_axis_count_zyx=arrays["axis_count_zyx"][order][best_patch].astype(np.uint64),
        best_patch_normal_sum_zyx=arrays["normal_sum_zyx"][order][best_patch].astype(np.int64),
        best_patch_area_nm2=arrays["area_nm2"][order][best_patch].astype(np.float64),
        best_patch_affinity_sum=ordered_affinity_sum[best_patch].astype(np.float64),
        best_patch_affinity_sq_sum=arrays["affinity_sq_sum"][order][best_patch].astype(np.float64),
        best_patch_affinity_max=arrays["affinity_max"][order][best_patch].astype(np.float32),
        best_patch_affinity_ge_count=ordered_affinity_ge[best_patch].astype(np.uint64),
        representative_patch_policy=np.asarray(
            "max fraction affinity>=0.8, then mean affinity, support, stable chunk order"
        ),
        chunk_rows=np.asarray(len(left), dtype=np.uint64),
        chunks=np.asarray(len(paths), dtype=np.uint32),
        complete=np.asarray(not args.allow_partial and len(paths) == args.expected_chunks),
        gt_free=np.asarray(True),
        **boundary_output,
    )
    log(
        f"aggregated chunks={len(paths):,} rows={len(left):,}",
        f"unique_pairs={len(starts):,} -> {args.output}",
        f"boundary_labels={len(boundary_output.get('segment_label', [])):,}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
