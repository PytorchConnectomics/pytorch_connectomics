"""Apply frozen branch unions and optional inter-object erosion chunk-wise."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from connectomics.data.processing.segment import seg_erosion_instance

from .artifacts import load_frozen_merge_roots, reject_evaluation_path, relabel_sorted
from .skeletonize import grid_chunks


def _cloudpath(value: str | Path) -> str:
    text = str(value)
    return text if "://" in text else Path(text).expanduser().resolve().as_uri()


def _local_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("file://"):
        return Path(unquote(urlparse(text).path))
    if "://" in text:
        raise ValueError("error-correction output must be a local file:// precomputed layer")
    return Path(text).expanduser().resolve()


def _open_volume(path: str | Path, **kwargs):
    from cloudvolume import CloudVolume

    return CloudVolume(_cloudpath(path), mip=0, progress=False, **kwargs)


def prepare_output(
    source_path: str | Path,
    output_path: str | Path,
    chunk_size_xyz: tuple[int, int, int],
) -> None:
    """Create an empty local precomputed segmentation matching the source bounds."""

    reject_evaluation_path(source_path)
    reject_evaluation_path(output_path)
    source = _open_volume(source_path, fill_missing=True, bounded=False)
    destination_path = _local_path(output_path)
    info_path = destination_path / "info"
    if info_path.is_file():
        destination = _open_volume(output_path, fill_missing=True, bounded=False)
        if tuple(int(value) for value in destination.shape[:3]) != tuple(
            int(value) for value in source.shape[:3]
        ):
            raise ValueError("existing output shape differs from the source segmentation")
        return
    info = copy.deepcopy(source.info)
    scale = info["scales"][0]
    scale["chunk_sizes"] = [list(int(value) for value in chunk_size_xyz)]
    info["type"] = "segmentation"
    info["data_type"] = "uint64"
    from cloudvolume import CloudVolume

    destination_path.mkdir(parents=True, exist_ok=True)
    destination = CloudVolume(
        destination_path.as_uri(),
        info=info,
        mip=0,
        compress=False,
        progress=False,
        fill_missing=True,
        bounded=False,
    )
    destination.commit_info()
    destination.provenance.description = (
        "GT-free morphology error correction: frozen branch unions followed by "
        "inter-object erosion"
    )
    destination.commit_provenance()


def _factorized_erosion(merged: np.ndarray, radius_zyx: tuple[int, int, int]) -> np.ndarray:
    """Run label-boundary erosion without truncating uint64 segment identities."""

    values, inverse = np.unique(merged, return_inverse=True)
    codes: np.ndarray = np.arange(1, len(values) + 1, dtype=np.uint32)
    zero = np.flatnonzero(values == 0)
    if len(zero):
        codes[zero[0]] = 0
    dense = codes[inverse].reshape(merged.shape)
    eroded = seg_erosion_instance(dense, tsz_h=radius_zyx)
    decode: np.ndarray = np.zeros(int(codes.max(initial=0)) + 1, dtype=np.uint64)
    decode[codes] = values
    return decode[eroded]


def correct_block(
    labels_zyx: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    erosion_radius_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Union first, then erase only boundaries between remaining distinct objects."""

    merged = relabel_sorted(labels_zyx, source_labels, target_labels)
    if not any(erosion_radius_zyx):
        return merged
    return _factorized_erosion(merged, erosion_radius_zyx)


def process_task(
    *,
    source_path: str | Path,
    output_path: str | Path,
    proposals_path: Path,
    proposal_report_path: Path,
    workdir: Path,
    core_xyz: tuple[int, int, int],
    erosion_radius_zyx: tuple[int, int, int],
    task_id: int,
    num_tasks: int,
    overwrite: bool,
    max_owned_chunks: int | None = None,
) -> dict[str, object]:
    reject_evaluation_path(source_path)
    reject_evaluation_path(output_path)
    source_labels, target_labels, proposal_sha256 = load_frozen_merge_roots(
        proposals_path, proposal_report_path
    )
    source = _open_volume(source_path, fill_missing=True, bounded=False)
    output = _open_volume(output_path, fill_missing=True, bounded=False)
    shape_xyz = np.asarray(source.shape[:3], dtype=np.int64)
    offset_xyz = np.asarray(source.voxel_offset, dtype=np.int64)
    chunks = list(grid_chunks(shape_xyz, np.asarray(core_xyz, dtype=np.int64)))
    owned = [item for index, item in enumerate(chunks) if index % num_tasks == task_id]
    if max_owned_chunks is not None:
        owned = owned[:max_owned_chunks]
    done_dir = workdir / "postprocess_chunks"
    done_dir.mkdir(parents=True, exist_ok=True)
    halo_xyz = np.asarray(erosion_radius_zyx[::-1], dtype=np.int64)
    written = skipped = 0
    for key, _, local_lo, local_hi, _ in owned:
        marker = done_dir / f"{key}.json"
        if marker.is_file() and not overwrite:
            completed = json.loads(marker.read_text())
            expected = {
                "key": key,
                "proposal_sha256": proposal_sha256,
                "erosion_radius_zyx": list(erosion_radius_zyx),
            }
            if any(completed.get(name) != value for name, value in expected.items()):
                raise ValueError(f"stale postprocess marker requires --overwrite: {marker}")
            skipped += 1
            continue
        lo = local_lo + offset_xyz
        hi = local_hi + offset_xyz
        read_lo = np.maximum(offset_xyz, lo - halo_xyz)
        read_hi = np.minimum(offset_xyz + shape_xyz, hi + halo_xyz)
        block = np.asarray(
            source[
                int(read_lo[0]) : int(read_hi[0]),
                int(read_lo[1]) : int(read_hi[1]),
                int(read_lo[2]) : int(read_hi[2]),
            ]
        )
        if block.ndim == 4:
            block = block[..., 0]
        corrected = correct_block(
            np.transpose(block, (2, 1, 0)),
            source_labels,
            target_labels,
            erosion_radius_zyx,
        )
        core_lo = (lo - read_lo)[::-1]
        core_hi = core_lo + (hi - lo)[::-1]
        core = corrected[
            int(core_lo[0]) : int(core_hi[0]),
            int(core_lo[1]) : int(core_hi[1]),
            int(core_lo[2]) : int(core_hi[2]),
        ]
        output[
            int(lo[0]) : int(hi[0]),
            int(lo[1]) : int(hi[1]),
            int(lo[2]) : int(hi[2]),
        ] = np.transpose(core, (2, 1, 0))[..., None]
        marker.write_text(
            json.dumps(
                {
                    "key": key,
                    "proposal_sha256": proposal_sha256,
                    "erosion_radius_zyx": list(erosion_radius_zyx),
                    "task_id": task_id,
                    "gt_free": True,
                },
                indent=2,
            )
            + "\n"
        )
        written += 1
    return {
        "task_id": task_id,
        "num_tasks": num_tasks,
        "owned": len(owned),
        "written": written,
        "skipped": skipped,
        "proposal_sha256": proposal_sha256,
    }


def verify_output(
    source_path: str | Path,
    output_path: str | Path,
    workdir: Path,
    core_xyz: tuple[int, int, int],
    proposals_path: Path,
    proposal_report_path: Path,
    erosion_radius_zyx: tuple[int, int, int],
) -> dict[str, object]:
    source = _open_volume(source_path, fill_missing=True, bounded=False)
    output = _open_volume(output_path, fill_missing=True, bounded=False)
    if tuple(source.shape[:3]) != tuple(output.shape[:3]):
        raise ValueError("corrected output shape differs from source")
    chunks = list(grid_chunks(np.asarray(source.shape[:3]), np.asarray(core_xyz)))
    markers = sorted((workdir / "postprocess_chunks").glob("z*_y*_x*.json"))
    if len(markers) != len(chunks):
        raise RuntimeError(f"postprocess is incomplete: {len(markers):,}/{len(chunks):,} chunks")
    _, _, proposal_sha256 = load_frozen_merge_roots(proposals_path, proposal_report_path)
    expected_keys = {item[0] for item in chunks}
    marker_keys = set()
    for marker in markers:
        completed = json.loads(marker.read_text())
        marker_keys.add(completed.get("key"))
        if completed.get("proposal_sha256") != proposal_sha256 or completed.get(
            "erosion_radius_zyx"
        ) != list(erosion_radius_zyx):
            raise ValueError(f"postprocess marker provenance differs: {marker}")
    if marker_keys != expected_keys:
        raise RuntimeError("postprocess marker keys do not match the source chunk grid")
    manifest = {
        "schema": 1,
        "source": str(source_path),
        "output": str(output_path),
        "chunks": len(chunks),
        "proposal_sha256": proposal_sha256,
        "operation_order": ["frozen branch unions", "inter-object erosion"],
        "erosion_radius_zyx": list(erosion_radius_zyx),
        "gt_free": True,
    }
    path = workdir / "error_correction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--proposal-report", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--core-xyz", type=int, nargs=3, default=[512, 512, 256])
    parser.add_argument("--erosion-radius-zyx", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-owned-chunks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    core = tuple(args.core_xyz)
    radius = tuple(args.erosion_radius_zyx)
    if args.command == "prepare":
        prepare_output(args.source, args.output, core)
        return 0
    if args.proposals is None or args.proposal_report is None:
        parser.error("run/verify require --proposals and --proposal-report")
    if args.command == "run":
        result: dict[str, object] = process_task(
            source_path=args.source,
            output_path=args.output,
            proposals_path=args.proposals,
            proposal_report_path=args.proposal_report,
            workdir=args.workdir,
            core_xyz=core,
            erosion_radius_zyx=radius,
            task_id=args.task_id,
            num_tasks=args.num_tasks,
            overwrite=args.overwrite,
            max_owned_chunks=args.max_owned_chunks,
        )
    else:
        result = verify_output(
            args.source,
            args.output,
            args.workdir,
            core,
            args.proposals,
            args.proposal_report,
            radius,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
