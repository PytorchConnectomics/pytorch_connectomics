#!/usr/bin/env python3
"""Fail-closed preflight for the frozen j0126 reproduction inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import h5py

EXPECTED_ABISS_COMMIT = "452efa5f87f9d3cb241891ee44010d966a33b316"
EXPECTED_BBOX_XYZ = (0, 0, 0, 10664, 10912, 5700)
EXPECTED_INDEX_SHAPE_ZYX = (5700, 12288, 12288)
EXPECTED_COVERAGE_SHAPE_ZYX = (5700, 10913, 10664)
EXPECTED_KEEP_MASK_SHAPE_ZYX = (5700, 10912, 10664)
EXPECTED_NUCLEUS_SHAPE_ZYX = (1425, 1365, 1333)
EXPECTED_GRID_ZYX = (6, 11, 11)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_chunk(index_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = index_dir / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_spatial = tuple(
        int(stop) - int(start) for start, stop in zip(entry["start_zyx"], entry["stop_zyx"])
    )
    with h5py.File(path, "r", locking=False) as handle:
        if "main" not in handle:
            raise ValueError(f"{path}: missing dataset 'main'")
        dataset = handle["main"]
        expected_shape = (3, *expected_spatial)
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(f"{path}: shape {dataset.shape}, expected {expected_shape}")
        if str(dataset.dtype) != "float16":
            raise ValueError(f"{path}: dtype {dataset.dtype}, expected float16")
    return {"path": str(path), "bytes": path.stat().st_size}


def _load_zarr_v3_shape(path: Path) -> tuple[int, ...]:
    metadata = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    return tuple(int(value) for value in metadata["shape"])


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_tracked_status(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _cmake_cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        typed_key, value = line.split("=", 1)
        key = typed_key.split(":", 1)[0]
        values[key] = value
    return values


def _dynamic_dependencies(path: Path) -> list[str]:
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies = []
    for line in result.stdout.splitlines():
        if "(NEEDED)" in line and "[" in line and "]" in line:
            dependencies.append(line.split("[", 1)[1].split("]", 1)[0])
    return dependencies


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "cloud-volume",
        "h5py",
        "networkx",
        "numpy",
        "PyYAML",
        "scipy",
        "zarr",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinity-chunks", type=Path, required=True)
    parser.add_argument("--affinity-index", type=Path, required=True)
    parser.add_argument("--keep-mask", type=Path, required=True)
    parser.add_argument("--nucleus", type=Path, required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--abiss-home", type=Path, required=True)
    parser.add_argument("--pytc-prefix", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--min-free-tib", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for path in (
        args.affinity_chunks,
        args.affinity_index,
        args.keep_mask,
        args.nucleus,
        args.skeletons,
        args.abiss_home,
        args.pytc_prefix,
        args.storage_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    abiss_head = _git_head(args.abiss_home)
    if abiss_head != EXPECTED_ABISS_COMMIT:
        raise RuntimeError(f"ABISS is {abiss_head}; frozen result requires {EXPECTED_ABISS_COMMIT}")
    tracked_status = _git_tracked_status(args.abiss_home)
    if tracked_status:
        raise RuntimeError(
            "ABISS has tracked modifications; frozen replay requires a clean checkout:\n"
            + tracked_status
        )
    for relative in ("build/ws", "build/agg", "scripts/nucleus_competition.py"):
        path = args.abiss_home / relative
        if not path.is_file():
            raise FileNotFoundError(f"ABISS runtime is incomplete: {path}")

    expected_toolchain = {
        "CMAKE_C_COMPILER": str(args.pytc_prefix / "bin" / "x86_64-conda-linux-gnu-cc"),
        "CMAKE_CXX_COMPILER": str(args.pytc_prefix / "bin" / "x86_64-conda-linux-gnu-c++"),
        "Boost_DIR": str(args.pytc_prefix / "lib" / "cmake" / "Boost-1.82.0"),
        "TBB_DIR": str(args.pytc_prefix / "lib" / "cmake" / "TBB"),
        "EXTRACT_SIZE": "ON",
    }
    cache_path = args.abiss_home / "build" / "CMakeCache.txt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"ABISS build is missing its CMake cache: {cache_path}")
    cache = _cmake_cache(cache_path)
    mismatches = {
        key: {"actual": cache.get(key), "expected": expected}
        for key, expected in expected_toolchain.items()
        if cache.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"ABISS toolchain mismatch: {mismatches}")
    binary_dependencies = {
        name: _dynamic_dependencies(args.abiss_home / "build" / name) for name in ("ws", "agg")
    }
    for name, dependencies in binary_dependencies.items():
        if "libtbb.so.12" not in dependencies:
            raise RuntimeError(f"ABISS {name} requires {dependencies}; expected libtbb.so.12")
        incompatible = [
            dependency
            for dependency in dependencies
            if dependency == "libtbb.so.2" or ".so.1.85.0" in dependency
        ]
        if incompatible:
            raise RuntimeError(
                f"ABISS {name} has incompatible runtime dependencies: {incompatible}"
            )

    index = json.loads(args.affinity_index.read_text(encoding="utf-8"))
    chunks = index.get("chunks", [])
    if int(index.get("world_size", -1)) != 726 or len(chunks) != 726:
        raise RuntimeError(
            f"Affinity index has world_size={index.get('world_size')} and "
            f"{len(chunks)} chunks; expected 726"
        )
    keys = {tuple(int(value) for value in entry["index_zyx"]) for entry in chunks}
    expected_keys = {
        (z, y, x)
        for z in range(EXPECTED_GRID_ZYX[0])
        for y in range(EXPECTED_GRID_ZYX[1])
        for x in range(EXPECTED_GRID_ZYX[2])
    }
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise RuntimeError(f"Affinity grid mismatch; missing={missing[:5]}, extra={extra[:5]}")
    final_shape = tuple(int(value) for value in index["final_shape"])
    if final_shape != EXPECTED_INDEX_SHAPE_ZYX:
        raise RuntimeError(
            f"Affinity index shape is {final_shape}, expected {EXPECTED_INDEX_SHAPE_ZYX}"
        )
    covered_shape = tuple(
        max(int(entry["stop_zyx"][axis]) for entry in chunks) for axis in range(3)
    )
    if covered_shape != EXPECTED_COVERAGE_SHAPE_ZYX:
        raise RuntimeError(
            f"Affinity covered shape is {covered_shape}, " f"expected {EXPECTED_COVERAGE_SHAPE_ZYX}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        checked = list(
            executor.map(
                lambda entry: _check_chunk(args.affinity_index.parent, entry),
                chunks,
            )
        )

    keep_shape = _load_zarr_v3_shape(args.keep_mask)
    if keep_shape != EXPECTED_KEEP_MASK_SHAPE_ZYX:
        raise RuntimeError(
            f"Keep-mask shape is {keep_shape}, expected {EXPECTED_KEEP_MASK_SHAPE_ZYX}"
        )

    with h5py.File(args.nucleus, "r", locking=False) as handle:
        nucleus = handle["main"]
        nucleus_shape = tuple(int(value) for value in nucleus.shape)
        nucleus_dtype = str(nucleus.dtype)
    if nucleus_shape != EXPECTED_NUCLEUS_SHAPE_ZYX or nucleus_dtype != "uint16":
        raise RuntimeError(
            f"Nucleus volume is {nucleus_shape} {nucleus_dtype}; expected "
            f"{EXPECTED_NUCLEUS_SHAPE_ZYX} uint16"
        )

    with h5py.File(args.skeletons, "r", locking=False) as handle:
        skeleton_count = len(handle)
        skeleton_nodes = sum(int(handle[key]["vertices"].shape[0]) for key in handle)
    if skeleton_count != 50:
        raise RuntimeError(f"Skeleton file has {skeleton_count} groups; expected 50")

    usage = shutil.disk_usage(args.storage_root)
    free_tib = usage.free / 2**40
    if free_tib < args.min_free_tib:
        raise RuntimeError(
            f"Only {free_tib:.2f} TiB free at {args.storage_root}; "
            f"require at least {args.min_free_tib:.2f} TiB"
        )

    report = {
        "status": "pass",
        "abiss_commit": abiss_head,
        "abiss_binaries_sha256": {
            name: _sha256(args.abiss_home / "build" / name) for name in ("ws", "agg")
        },
        "abiss_toolchain": {
            "cmake": {key: cache[key] for key in expected_toolchain},
            "dynamic_dependencies": binary_dependencies,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "bbox_xyz": list(EXPECTED_BBOX_XYZ),
        "affinity": {
            "chunks": len(checked),
            "grid_zyx": list(EXPECTED_GRID_ZYX),
            "index_shape_zyx": list(final_shape),
            "covered_shape_zyx": list(covered_shape),
            "bytes": sum(item["bytes"] for item in checked),
            "index_sha256": _sha256(args.affinity_index),
        },
        "keep_mask_shape_zyx": list(keep_shape),
        "nucleus": {"shape_zyx": list(nucleus_shape), "dtype": nucleus_dtype},
        "skeletons": {"count": skeleton_count, "nodes": skeleton_nodes},
        "storage": {"root": str(args.storage_root), "free_tib": free_tib},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
