#!/usr/bin/env python3
"""Capture checkpoint goldens from the unmodified zebra-finch development tools.

This program is intentionally standalone.  It must be run before lifting or
refactoring ``dev/zebrafinch/nucleus_{competitive_split,anchor_merge,
shell_contamination}.py``.  It records their source hashes and CLI contracts,
and samples the real ``worst3`` nucleus-mask ROI from the before/after
CloudVolume layers.  The resulting NPZ contains real per-mask-voxel labels,
not generated or reconstructed observations.

The emitted ``MANIFEST.sha256`` covers the captured input evidence and every
golden.  Tests verify the manifest in a clean checkout without depending on
the gitignored ``dev/`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCALE_ZYX = np.asarray([4, 8, 8], dtype=np.int64)
LEGACY_SCRIPTS = (
    "nucleus_competitive_split.py",
    "nucleus_anchor_merge.py",
    "nucleus_shell_contamination.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_range(lo: int, hi: int, scale: int) -> tuple[int, int]:
    return (
        max(0, (lo - scale // 2 + scale - 1) // scale),
        (hi - scale // 2 + scale - 1) // scale,
    )


def _sample_cloudvolume(
    cloudpath: str,
    mask: np.ndarray,
    bbox_xyz: tuple[int, int, int, int, int, int],
    mask_origin_zyx: tuple[int, int, int],
) -> np.ndarray:
    from cloudvolume import CloudVolume

    x0, y0, z0, x1, y1, z1 = bbox_xyz
    zm0, ym0, xm0 = mask_origin_zyx
    zs = np.arange(zm0, zm0 + mask.shape[0]) * SCALE_ZYX[0] + SCALE_ZYX[0] // 2
    ys = np.arange(ym0, ym0 + mask.shape[1]) * SCALE_ZYX[1] + SCALE_ZYX[1] // 2
    xs = np.arange(xm0, xm0 + mask.shape[2]) * SCALE_ZYX[2] + SCALE_ZYX[2] // 2
    volume = CloudVolume(
        cloudpath,
        mip=0,
        progress=False,
        fill_missing=True,
        use_https=True,
        parallel=False,
    )
    chunk = volume.meta.chunk_size(0)
    cx, cy, cz = (int(value) for value in ((252, 252, 252) if chunk is None else chunk))
    sampled = np.zeros(mask.shape, dtype=np.uint64)
    for zb in range((z0 // cz) * cz, z1, cz):
        kz = np.nonzero((zs >= zb) & (zs < min(zb + cz, z1)))[0]
        if not kz.size:
            continue
        for xb in range((x0 // cx) * cx, x1, cx):
            kx = np.nonzero((xs >= xb) & (xs < min(xb + cx, x1)))[0]
            if not kx.size:
                continue
            block = np.asarray(
                volume[xb : min(xb + cx, x1), ys[0] : ys[-1] + 1, zb : min(zb + cz, z1)]
            )[..., 0]
            sampled[np.ix_(kz, np.arange(len(ys)), kx)] = block[
                np.ix_(xs[kx] - xb, ys - ys[0], zs[kz] - zb)
            ].transpose(2, 1, 0)
    return sampled


def _metrics(nucleus_ids: np.ndarray, segment_ids: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for nucleus in sorted(int(value) for value in np.unique(nucleus_ids) if value):
        values = segment_ids[nucleus_ids == nucleus]
        labels, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        total = int(counts.sum())
        result[str(nucleus)] = {
            "total_mask_voxels": total,
            "dominant_segment": str(int(labels[order[0]])),
            "dominance": float(counts[order[0]] / total),
            "misplaced_mask_voxels": int(total - counts[order[0]]),
            "segments": [
                {"segment": str(int(labels[index])), "voxels": int(counts[index])}
                for index in order
            ],
        }
    return result


def _shared_count(metrics: dict[str, Any], nuclei: tuple[int, ...], min_share: float) -> int:
    owners: dict[str, set[int]] = {}
    for nucleus in nuclei:
        row = metrics[str(nucleus)]
        for segment in row["segments"]:
            if (
                segment["segment"] != "0"
                and segment["voxels"] >= min_share * row["total_mask_voxels"]
            ):
                owners.setdefault(segment["segment"], set()).add(nucleus)
    return sum(len(values) > 1 for values in owners.values())


def _write_manifest(root: Path, paths: list[Path]) -> None:
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(paths)]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuclei", type=Path, required=True)
    parser.add_argument("--before-seg", required=True)
    parser.add_argument("--after-seg", required=True)
    parser.add_argument("--bbox", type=int, nargs=6, required=True)
    parser.add_argument("--focus-nuclei", type=int, nargs="+", required=True)
    parser.add_argument("--min-share", type=float, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    roi = output / "worst3_roi"
    cli = output / "legacy_cli"
    roi.mkdir(parents=True, exist_ok=True)
    cli.mkdir(parents=True, exist_ok=True)
    legacy = repo / "dev" / "zebrafinch"

    source_hashes = {name: sha256_file(legacy / name) for name in LEGACY_SCRIPTS}
    for name in LEGACY_SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(legacy / name), "--help"],
            cwd=repo,
            env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"},
            check=True,
            capture_output=True,
            text=True,
        )
        (cli / f"{Path(name).stem}.help.txt").write_text(completed.stdout)

    x0, y0, z0, x1, y1, z1 = (int(value) for value in args.bbox)
    zm0, zm1 = mask_range(z0, z1, int(SCALE_ZYX[0]))
    ym0, ym1 = mask_range(y0, y1, int(SCALE_ZYX[1]))
    xm0, xm1 = mask_range(x0, x1, int(SCALE_ZYX[2]))
    with h5py.File(args.nuclei, "r") as handle:
        nucleus = np.asarray(handle["main"][zm0:zm1, ym0:ym1, xm0:xm1])
    origin = (zm0, ym0, xm0)
    before = _sample_cloudvolume(args.before_seg, nucleus, tuple(args.bbox), origin)
    after = _sample_cloudvolume(args.after_seg, nucleus, tuple(args.bbox), origin)
    hit = nucleus != 0
    np.savez_compressed(
        roi / "anchor_samples.npz",
        nucleus_id=nucleus[hit].astype(np.uint16),
        before_segment_id=before[hit],
        after_segment_id=after[hit],
    )

    before_metrics = _metrics(nucleus, before)
    after_metrics = _metrics(nucleus, after)
    focus = tuple(sorted(args.focus_nuclei))
    gate = {
        "schema_version": "1.0",
        "focus_nuclei": list(focus),
        "min_share": args.min_share,
        "misplaced_mask_voxels": {
            "before": sum(row["misplaced_mask_voxels"] for row in before_metrics.values()),
            "after": sum(row["misplaced_mask_voxels"] for row in after_metrics.values()),
        },
        "shared_segments": {
            "before": _shared_count(before_metrics, focus, args.min_share),
            "after": _shared_count(after_metrics, focus, args.min_share),
        },
        "dominance": {
            "before": [before_metrics[str(value)]["dominance"] for value in focus],
            "after": [after_metrics[str(value)]["dominance"] for value in focus],
        },
    }
    (roi / "expected_metrics.json").write_text(json.dumps(gate, indent=2) + "\n")
    metadata = {
        "schema_version": "1.0",
        "captured_before_refactor": True,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "legacy_source_sha256": source_hashes,
        "inputs": {
            "nuclei": str(args.nuclei),
            "before_seg": args.before_seg,
            "after_seg": args.after_seg,
            "bbox_xyz": args.bbox,
            "mask_origin_zyx": list(origin),
            "mask_shape_zyx": list(nucleus.shape),
        },
    }
    (roi / "capture_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    emitted = [
        path for path in output.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"
    ]
    _write_manifest(output, emitted)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
