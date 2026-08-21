"""Build and validate a whole-volume ABISS segment-size inventory."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import reject_evaluation_path, sha256_file

SIZE_DTYPE: np.dtype[Any] = np.dtype([("label", "<u8"), ("size", "<u8")])


def load_size_inventory(path: str | Path, *, require_report: bool = True) -> np.memmap:
    path = Path(path)
    reject_evaluation_path(path)
    report_path = path.with_suffix(path.suffix + ".json")
    if require_report:
        if not report_path.is_file():
            raise FileNotFoundError(f"size inventory report is missing: {report_path}")
        report = json.loads(report_path.read_text())
        if report.get("gt_free") is not True or report.get("sha256") != sha256_file(path):
            raise ValueError(f"size inventory provenance check failed: {path}")
    sizes: np.memmap = np.memmap(path, dtype=SIZE_DTYPE, mode="r")
    if len(sizes) and not np.all(sizes["label"][:-1] < sizes["label"][1:]):
        raise ValueError("size inventory labels must be unique and sorted")
    return sizes


def aggregate_size_files(
    pattern: str, output: Path, *, expected_files: int | None = None
) -> dict[str, object]:
    reject_evaluation_path(pattern)
    reject_evaluation_path(output)
    paths = [Path(value) for value in sorted(glob.glob(pattern, recursive=True))]
    paths = [path for path in paths if path.resolve() != output.resolve()]
    if not paths:
        raise FileNotFoundError(f"size glob matched no files: {pattern}")
    if expected_files is not None and len(paths) != expected_files:
        raise RuntimeError(f"size glob matched {len(paths):,}/{expected_files:,} files")
    parts = [np.asarray(np.memmap(path, dtype=SIZE_DTYPE, mode="r")) for path in paths]
    rows = np.concatenate(parts)
    rows = rows[rows["label"] != 0]
    order = np.argsort(rows["label"], kind="stable")
    labels = rows["label"][order]
    values = rows["size"][order]
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    result: np.ndarray = np.empty(len(starts), dtype=SIZE_DTYPE)
    result["label"] = labels[starts]
    result["size"] = np.add.reduceat(values, starts)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.tofile(output)
    report = {
        "schema": 1,
        "source_glob": pattern,
        "source_files": len(paths),
        "source_rows": len(rows),
        "segments": len(result),
        "voxels": int(result["size"].sum()),
        "sha256": sha256_file(output),
        "gt_free": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-files", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_size_files(args.input_glob, args.output, expected_files=args.expected_files),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
