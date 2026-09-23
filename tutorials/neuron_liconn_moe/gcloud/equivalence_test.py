#!/usr/bin/env python3
"""Compare chunked mean-edge segmentations with the whole-volume reference."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from connectomics.metrics.segmentation_numpy import voi
from connectomics.data.io import read_hdf5


def voi_total(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"segmentation shapes differ: {reference.shape} vs {candidate.shape}")
    if mask is None:
        mask = np.ones(reference.shape, dtype=bool)
    if not np.any(mask):
        raise ValueError("VOI mask is empty; refusing to pass a degenerate comparison.")
    split, merge = voi(candidate[mask], reference[mask], ignore_reconstruction=[], ignore_groundtruth=[])
    return float(split + merge)


def boundary_and_interior(reference: np.ndarray, chunk_size_xyz: Iterable[int]) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    if reference.ndim != 3:
        raise ValueError(f"reference must be ZYX, got {reference.shape}")
    # Convert XYZ chunk sizes to ZYX array axes.
    sizes = list(int(v) for v in chunk_size_xyz)[::-1]
    dims = reference.shape
    planes = [[p for p in range(size, dims[axis], size)] for axis, size in enumerate(sizes)]
    unique_labels, inverse = np.unique(reference, return_inverse=True)
    labels = [int(v) for v in unique_labels if int(v) != 0]
    # ``find_objects`` treats label 0 as background. Preserve a real smallest
    # label when the reference has no background voxels.
    relabeled = (inverse + int(unique_labels[0] != 0)).reshape(reference.shape)
    boundary: set[int] = set()
    # find_objects computes every label's bounding box in one scan of the array.
    objects = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(relabeled)
    for label in labels:
        relabel_index = int(np.searchsorted(unique_labels, label))
        object_index = relabel_index + int(unique_labels[0] != 0) - 1
        box = objects[object_index] if 0 <= object_index < len(objects) else None
        if box is None:
            continue
        if any(any(box[axis].start < plane < box[axis].stop for plane in planes[axis])
               for axis in range(3)):
            boundary.add(label)
    interior = set(labels) - boundary
    if not boundary or not interior:
        raise ValueError(f"non-degeneracy failed: |B|={len(boundary)} and |I|={len(interior)}; both must be > 0")
    return np.isin(reference, list(boundary)), np.isin(reference, list(interior)), boundary, interior


def read_precomputed_zyx(cloudpath: str | Path) -> np.ndarray:
    """Read a CloudVolume segmentation (XYZ storage) back as a ZYX array."""
    from cloudvolume import CloudVolume

    volume = CloudVolume(str(cloudpath), progress=False, bounded=True)
    raw = np.asarray(volume[:, :, :])
    if raw.ndim != 4 or raw.shape[3] != 1:
        raise ValueError(f"CloudVolume readback must be XYZC with C=1, got {raw.shape}")
    return np.transpose(raw[..., 0], (2, 1, 0))


def compare(reference: np.ndarray, candidate: np.ndarray, chunk_size_xyz: Iterable[int]) -> dict[str, float | int]:
    nonzero = reference != 0
    bmask, imask, boundary, interior = boundary_and_interior(reference, chunk_size_xyz)
    total = voi_total(reference, candidate, nonzero)
    bvoi = voi_total(reference, candidate, bmask)
    i_voi = voi_total(reference, candidate, imask)
    if total > 0.01 or bvoi - i_voi > 0.01:
        raise AssertionError(f"equivalence failed: total={total:.6f}, B-I={bvoi - i_voi:.6f}")
    if int(candidate.max()) <= 0:
        raise AssertionError("segmentation is degenerate: max id is not > 0")
    return {"voi_total": total, "voi_boundary": bvoi, "voi_interior": i_voi,
            "boundary_labels": len(boundary), "interior_labels": len(interior)}


def plumbing(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    value = voi_total(reference, candidate, reference != 0)
    if value != 0.0:
        raise AssertionError(f"one-chunk plumbing failed: VOI_total={value:.6f}, expected 0")
    return {"voi_total": value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--chunked", type=str, action="append", required=True)
    parser.add_argument("--chunk-size", type=int, nargs=3, action="append", required=True)
    parser.add_argument("--integrity", action="store_true")
    parser.add_argument("--plumbing", action="store_true")
    parser.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    if not args.reference.exists():
        raise FileNotFoundError(f"whole-volume reference is absent: {args.reference}")
    reference = np.asarray(read_hdf5(str(args.reference), dataset="main"))
    if len(args.chunk_size) != len(args.chunked):
        raise ValueError("each --chunked candidate requires its own --chunk-size XYZ")
    def read_candidate(path_string: str) -> np.ndarray:
        path = Path(path_string)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return np.asarray(read_hdf5(str(path), dataset="main"))
        return read_precomputed_zyx(path_string if "://" in path_string else f"file://{path}")

    candidates = [read_candidate(path) for path in args.chunked]
    for index, (path, candidate) in enumerate(zip(args.chunked, candidates)):
        if args.plumbing:
            print(path, plumbing(reference, candidate))
        elif args.integrity:
            value = voi_total(reference, candidate, reference != 0)
            if value > 0.001:
                detail = f"; see diagnostic {args.diagnostic}" if args.diagnostic else ""
                raise AssertionError(f"integrity gate failed: VOI_total={value:.6f} > 0.001{detail}")
            print(path, {"voi_total": value})
        else:
            print(path, compare(reference, candidate, args.chunk_size[index]))
    nonzero = reference != 0
    if len(candidates) >= 2 and voi_total(candidates[0], candidates[1], nonzero) > 0.01:
        raise AssertionError("chunk-size invariance failed under the W != 0 mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
