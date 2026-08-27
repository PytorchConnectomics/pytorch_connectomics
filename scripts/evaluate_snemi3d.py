#!/usr/bin/env python3
"""Evaluate SNEMI3D segmentations with the official challenge crop convention.

The SNEMI3D Grand Challenge evaluator scores a centered crop containing about
one third of the full volume before computing the legacy adapted Rand error.
This script ports that crop exactly, evaluates HDF5/NumPy segmentations, and
writes one auditable TSV for comparisons across decoders and experiments.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

# Match scripts/main.py so the script can run from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.metrics.segmentation_numpy import adapted_rand  # noqa: E402

_SUPPORTED_SUFFIXES = {".h5", ".hdf5", ".npy"}
_RESULT_FIELDS = (
    "status",
    "adapted_rand_error",
    "precision",
    "recall",
    "foreground_instances",
    "foreground_fraction",
    "path",
    "dataset",
    "shape",
    "dtype",
    "crop",
    "crop_shape",
    "crop_voxels",
    "reason",
)


@dataclass(frozen=True)
class CropBounds:
    """Half-open ZYX crop bounds."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z0, self.z1),
            slice(self.y0, self.y1),
            slice(self.x0, self.x1),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def __str__(self) -> str:
        return f"z={self.z0}:{self.z1},y={self.y0}:{self.y1},x={self.x0}:{self.x1}"


def snemi3d_gc_crop(shape: Sequence[int]) -> CropBounds:
    """Return the exact crop used by ``lib/snemi3d-gc/evaluation.py``.

    The original implementation deliberately uses floating-point division and
    truncates all six bounds to integers at the end. Preserve that behavior so
    scores match the historical challenge evaluator.
    """

    if len(shape) != 3:
        raise ValueError(f"SNEMI3D evaluation expects a 3D ZYX volume, got shape {tuple(shape)}.")
    z, y, x = (int(v) for v in shape)
    if min(z, y, x) <= 0:
        raise ValueError(f"Volume dimensions must be positive, got {(z, y, x)}.")

    n_voxels = z * y * x
    new_z = z / 2
    new_x = new_y = int(math.sqrt(n_voxels * 0.33 / new_z))
    z0 = new_z // 2
    z1 = new_z / 2 + new_z
    y0 = y // 2 - new_y // 2
    y1 = y0 + new_y
    x0 = x // 2 - new_x // 2
    x1 = x0 + new_x
    bounds = CropBounds(*map(int, (z0, z1, y0, y1, x0, x1)))

    if (
        bounds.z0 < 0
        or bounds.y0 < 0
        or bounds.x0 < 0
        or bounds.z1 > z
        or bounds.y1 > y
        or bounds.x1 > x
        or min(bounds.shape) <= 0
    ):
        raise ValueError(f"Challenge crop {bounds} is invalid for shape {(z, y, x)}.")
    return bounds


def full_volume_crop(shape: Sequence[int]) -> CropBounds:
    if len(shape) != 3:
        raise ValueError(f"Full-volume evaluation expects a 3D ZYX volume, got {tuple(shape)}.")
    z, y, x = (int(v) for v in shape)
    return CropBounds(0, z, 0, y, 0, x)


def discover_inputs(values: Iterable[str]) -> list[Path]:
    """Resolve explicit files, recursive directories, and shell-style globs."""

    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES
            )
            continue

        matches = [Path(match) for match in glob.glob(value, recursive=True)]
        if not matches:
            raise FileNotFoundError(f"Input path or glob did not match anything: {value}")
        for match in matches:
            if match.is_dir():
                paths.extend(
                    candidate
                    for candidate in match.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES
                )
            elif match.is_file() and match.suffix.lower() in _SUPPORTED_SUFFIXES:
                paths.append(match)

    return sorted({path.resolve() for path in paths})


def _array_metadata(path: Path, dataset: str) -> tuple[tuple[int, ...], np.dtype]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                raise KeyError(f"Dataset {dataset!r} not found")
            array = handle[dataset]
            return tuple(int(v) for v in array.shape), np.dtype(array.dtype)
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        return tuple(int(v) for v in array.shape), np.dtype(array.dtype)
    raise ValueError(f"Unsupported input format {path.suffix!r}")


def _read_crop(path: Path, dataset: str, bounds: CropBounds) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            return np.asarray(handle[dataset][bounds.slices])
    if suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r")[bounds.slices])
    raise ValueError(f"Unsupported input format {path.suffix!r}")


def _empty_result(path: Path, dataset: str) -> dict[str, object]:
    return {
        "status": "skipped",
        "adapted_rand_error": "",
        "precision": "",
        "recall": "",
        "foreground_instances": "",
        "foreground_fraction": "",
        "path": str(path),
        "dataset": dataset,
        "shape": "",
        "dtype": "",
        "crop": "",
        "crop_shape": "",
        "crop_voxels": "",
        "reason": "",
    }


def evaluate_candidate(
    path: Path,
    *,
    dataset: str,
    expected_shape: tuple[int, int, int],
    bounds: CropBounds,
    ground_truth_crop: np.ndarray,
) -> dict[str, object]:
    """Evaluate one candidate, returning a TSV-ready result row."""

    row = _empty_result(path, dataset)
    try:
        shape, dtype = _array_metadata(path, dataset)
        row["shape"] = "x".join(str(v) for v in shape)
        row["dtype"] = str(dtype)
        if shape != expected_shape:
            row["reason"] = f"shape mismatch: expected {expected_shape}, got {shape}"
            return row
        if not np.issubdtype(dtype, np.integer):
            row["reason"] = f"not an integer segmentation: dtype={dtype}"
            return row

        segmentation = _read_crop(path, dataset, bounds)
        are, precision, recall = adapted_rand(segmentation, ground_truth_crop, all_stats=True)
        foreground = segmentation != 0
        row.update(
            {
                "status": "ok",
                "adapted_rand_error": f"{float(are):.12g}",
                "precision": f"{float(precision):.12g}",
                "recall": f"{float(recall):.12g}",
                "foreground_instances": int(np.unique(segmentation[foreground]).size),
                "foreground_fraction": f"{float(foreground.mean()):.12g}",
                "crop": str(bounds),
                "crop_shape": "x".join(str(v) for v in bounds.shape),
                "crop_voxels": int(np.prod(bounds.shape, dtype=np.int64)),
                "reason": "",
            }
        )
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def _sort_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(row: dict[str, object]) -> tuple[int, float, str]:
        if row["status"] == "ok":
            return (0, float(row["adapted_rand_error"]), str(row["path"]))
        return (1, math.inf, str(row["path"]))

    return sorted(rows, key=key)


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(_sort_results(rows))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Segmentation files, directories, or globs")
    parser.add_argument(
        "--ground-truth",
        default="datasets/SNEMI/test-labels.h5",
        help="SNEMI3D ground-truth HDF5/NumPy volume",
    )
    parser.add_argument("--dataset", default="main", help="Candidate HDF5 dataset key")
    parser.add_argument(
        "--ground-truth-dataset", default="main", help="Ground-truth HDF5 dataset key"
    )
    parser.add_argument(
        "--crop",
        choices=("challenge", "full"),
        default="challenge",
        help="Evaluation support; challenge is the historical SNEMI3D-GC crop",
    )
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Skip paths containing this substring; may be repeated",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Exit nonzero if any discovered file is skipped",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ground_truth_path = Path(args.ground_truth).expanduser().resolve()
    gt_shape, gt_dtype = _array_metadata(ground_truth_path, args.ground_truth_dataset)
    if len(gt_shape) != 3 or not np.issubdtype(gt_dtype, np.integer):
        raise ValueError(
            "Ground truth must be a 3D integer label volume, "
            f"got shape={gt_shape}, dtype={gt_dtype}."
        )

    if args.crop == "challenge":
        bounds = snemi3d_gc_crop(gt_shape)
    else:
        bounds = full_volume_crop(gt_shape)
    ground_truth_crop = _read_crop(ground_truth_path, args.ground_truth_dataset, bounds)
    candidates = discover_inputs(args.inputs)
    excludes = tuple(args.exclude_name)

    rows: list[dict[str, object]] = []
    for path in candidates:
        if path == ground_truth_path:
            continue
        if excludes and any(value in str(path) for value in excludes):
            row = _empty_result(path, args.dataset)
            row["reason"] = f"excluded by name filter: {excludes}"
        else:
            row = evaluate_candidate(
                path,
                dataset=args.dataset,
                expected_shape=gt_shape,
                bounds=bounds,
                ground_truth_crop=ground_truth_crop,
            )
        rows.append(row)
        if row["status"] == "ok":
            print(f"{float(row['adapted_rand_error']):.6f}\t{path}")
        else:
            print(f"SKIP\t{path}\t{row['reason']}")

    output_path = Path(args.output).expanduser().resolve()
    write_results(output_path, rows)
    n_ok = sum(row["status"] == "ok" for row in rows)
    n_skipped = len(rows) - n_ok
    print(f"Wrote {n_ok} scores and {n_skipped} skipped entries to {output_path}")
    print(f"Crop: {bounds} ({bounds.shape}, {int(np.prod(bounds.shape))} voxels)")
    if n_ok == 0:
        return 1
    if args.fail_on_skip and n_skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
