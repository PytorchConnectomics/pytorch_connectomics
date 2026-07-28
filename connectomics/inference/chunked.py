"""Chunked raw-prediction inference for large lazy volumes."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..chunked.chunk_grid import ChunkRef, build_chunk_grid
from .artifact import (
    build_prediction_artifact_metadata,
    write_prediction_artifact,
)
from .chunk_grid import (
    resolve_chunk_shape,
    resolve_global_prediction_crop,
    resolve_h5_spatial_chunks,
    validate_chunked_output_format,
)
from .lazy import get_lazy_image_reference_shape, lazy_predict_region
from .output import apply_prediction_transform, apply_storage_dtype_transform

logger = logging.getLogger(__name__)


def is_chunked_inference_enabled(cfg: Any) -> bool:
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return False
    strategy = str(getattr(inference_cfg, "strategy", "whole_volume")).lower()
    chunking_cfg = getattr(inference_cfg, "chunking", None)
    return strategy == "chunked" or bool(getattr(chunking_cfg, "enabled", False))


def _resolve_distributed_rank() -> tuple[int, int]:
    """Return (rank, world_size). (0, 1) when torch.distributed isn't initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def _per_chunk_dir(output_path: Path) -> Path:
    """Sibling directory holding per-chunk h5 files for distributed chunked inference."""
    return output_path.with_suffix(output_path.suffix + ".chunks")


def _chunk_file_path(chunks_dir: Path, chunk: ChunkRef) -> Path:
    return chunks_dir / f"chunk_{chunk.key}.h5"


def _precomputed_marker_path(chunks_dir: Path, chunk: ChunkRef) -> Path:
    """Resume marker for a chunk already written into the precomputed layer.

    The layer itself has no per-chunk file to stat, so completion is tracked here to
    keep the same "skip finished chunks on re-run" behaviour as the HDF5 path.
    """
    return chunks_dir / f"chunk_{chunk.key}.done"


def _open_precomputed_layer(
    layer_dir: Path,
    *,
    volume_size_xyz: Sequence[int],
    num_channels: int,
    data_type: str,
    resolution_xyz: Sequence[int],
    chunk_size_xyz: Sequence[int],
) -> Any:
    """Open the output precomputed layer, creating its ``info`` exactly once.

    Several ranks reach this concurrently, so creation is guarded by an O_EXCL lock
    file: the winner commits ``info`` and the losers wait for it to appear. Once
    ``info`` exists every rank just opens the layer; the voxel writes themselves are
    disjoint and storage-chunk aligned, so they need no coordination.
    """
    from cloudvolume import CloudVolume

    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_uri = "file://" + str(layer_dir.resolve())
    info_path = layer_dir / "info"

    if not info_path.exists():
        lock_path = layer_dir / ".info.lock"
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            fd = None
        if fd is not None:
            try:
                info = CloudVolume.create_new_info(
                    num_channels=int(num_channels),
                    layer_type="image",
                    data_type=str(data_type),
                    encoding="raw",
                    resolution=[int(v) for v in resolution_xyz],
                    voxel_offset=[0, 0, 0],
                    volume_size=[int(v) for v in volume_size_xyz],
                    chunk_size=[int(v) for v in chunk_size_xyz],
                )
                cv = CloudVolume(layer_uri, info=info, fill_missing=True, compress=True)
                cv.commit_info()
                logger.info(
                    "Created precomputed output layer %s (size_xyz=%s, channels=%d, "
                    "dtype=%s, resolution_xyz=%s, chunk_xyz=%s)",
                    layer_uri,
                    list(volume_size_xyz),
                    int(num_channels),
                    data_type,
                    list(resolution_xyz),
                    list(chunk_size_xyz),
                )
            finally:
                os.close(fd)
        else:
            for _ in range(600):  # ~60 s; the writer only has one small file to commit
                if info_path.exists():
                    break
                time.sleep(0.1)
            if not info_path.exists():
                raise RuntimeError(f"Timed out waiting for precomputed info at {info_path}")

    return CloudVolume(layer_uri, fill_missing=True, compress=True, progress=False)


def _validate_precomputed_alignment(
    chunk_shape_zyx: Sequence[int], chunk_size_xyz: Sequence[int]
) -> None:
    """Fail fast if inference chunks do not tile the layer's storage chunks.

    Ranks write disjoint inference chunks concurrently. If an inference chunk does not
    land on storage-chunk boundaries, two ranks can touch the same storage chunk and
    race, so this is a correctness requirement rather than a tuning knob.
    """
    chunk_shape_xyz = [int(v) for v in reversed(list(chunk_shape_zyx))]
    bad = [
        f"{axis}: inference chunk {chunk_shape_xyz[i]} is not a multiple of "
        f"storage chunk {int(chunk_size_xyz[i])}"
        for i, axis in enumerate("xyz")
        if int(chunk_size_xyz[i]) <= 0 or chunk_shape_xyz[i] % int(chunk_size_xyz[i]) != 0
    ]
    if bad:
        raise ValueError(
            "chunking.precomputed_chunk_size must divide the inference chunk on every "
            "axis so concurrent chunk writes never straddle a storage chunk. "
            + "; ".join(bad)
        )


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _resolve_external_chunk_shard(cfg: Any) -> tuple[int, int] | None:
    chunking_cfg = getattr(getattr(cfg, "inference", None), "chunking", None)
    if chunking_cfg is None:
        return None
    shard_id = getattr(chunking_cfg, "shard_id", None)
    num_shards = getattr(chunking_cfg, "num_shards", None)
    if shard_id is None and num_shards is None:
        return None
    if shard_id is None or num_shards is None:
        raise ValueError("Both inference.chunking.shard_id and num_shards must be set together.")
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    if num_shards <= 0:
        raise ValueError(f"inference.chunking.num_shards must be positive, got {num_shards}.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"inference.chunking.shard_id={shard_id} out of range for num_shards={num_shards}."
        )
    return shard_id, num_shards


def _resolve_inference_roi(
    cfg: Any,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """ROI (image geometry) in INPUT voxel coords (ZYX) restricting the chunk grid.

    Returns ``((z0, y0, x0), (z1, y1, x1))`` or ``None`` when unset. Accepts 3 ints
    (size from origin 0) or 6 ints (explicit start/stop). Chunks that don't overlap
    this box are pure padding in an over-sized volume and are skipped.
    """
    chunking_cfg = getattr(getattr(cfg, "inference", None), "chunking", None)
    roi = getattr(chunking_cfg, "roi", None) if chunking_cfg is not None else None
    if roi is None:
        return None
    vals = [int(v) for v in roi]
    if len(vals) == 3:
        start, stop = (0, 0, 0), (vals[0], vals[1], vals[2])
    elif len(vals) == 6:
        start, stop = (vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5])
    else:
        raise ValueError(
            f"inference.chunking.roi must have 3 (size) or 6 (start/stop) ints ZYX, got {roi!r}."
        )
    if any(stop[axis] <= start[axis] for axis in range(3)):
        raise ValueError(f"inference.chunking.roi stop must exceed start on every axis, got {roi!r}.")
    return start, stop


def _filter_chunks_to_roi(chunks, roi, crop_before):
    """Drop chunks whose core (in INPUT coords) lies entirely outside ``roi``."""
    (rz0, ry0, rx0), (rz1, ry1, rx1) = roi

    def _overlaps(ch) -> bool:
        cs = tuple(ch.start[axis] + crop_before[axis] for axis in range(3))
        ce = tuple(ch.stop[axis] + crop_before[axis] for axis in range(3))
        return (
            cs[0] < rz1 and ce[0] > rz0
            and cs[1] < ry1 and ce[1] > ry0
            and cs[2] < rx1 and ce[2] > rx0
        )

    return [ch for ch in chunks if _overlaps(ch)]


def is_external_chunk_sharding_enabled(cfg: Any) -> bool:
    return _resolve_external_chunk_shard(cfg) is not None


def _write_chunk_index(
    *,
    output_path: Path,
    chunks_dir: Path,
    chunks: list[ChunkRef],
    input_shape: tuple[int, int, int],
    final_shape: tuple[int, int, int],
    crop_pad: tuple[tuple[int, int], ...],
    chunk_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    checkpoint_path: str | Path | None,
    world_size: int,
) -> Path:
    index = {
        "input_shape": list(input_shape),
        "final_shape": list(final_shape),
        "chunk_shape": list(chunk_shape),
        "halo": list(halo),
        "crop_pad": [list(pair) for pair in crop_pad],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "world_size": world_size,
        "chunks": [
            {
                "key": chunk.key,
                "index_zyx": list(chunk.index),
                "start_zyx": list(chunk.start),
                "stop_zyx": list(chunk.stop),
                "path": str(_chunk_file_path(chunks_dir, chunk).relative_to(output_path.parent)),
            }
            for chunk in chunks
        ],
    }
    index_path = output_path.with_suffix(output_path.suffix + ".index.json")
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2)
    return index_path


def _stitch_chunk_prediction_files(
    *,
    cfg: Any,
    image_path: str,
    output_path: Path,
    chunks_dir: Path,
    chunks: list[ChunkRef],
    input_shape: tuple[int, int, int],
    final_shape: tuple[int, int, int],
    crop_pad: tuple[tuple[int, int], ...],
    chunk_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    compression,
    h5_spatial_chunks: tuple[int, int, int],
    checkpoint_path: str | Path | None,
    requested_head: str | None,
    qc_streaming_callback: Any = None,
) -> Path:
    """Stitch per-rank chunk artifacts into the canonical CZYX raw prediction H5."""
    import h5py

    if not chunks:
        raise ValueError("Cannot stitch chunked predictions: no chunks were generated.")

    first_chunk_path = _chunk_file_path(chunks_dir, chunks[0])
    if not first_chunk_path.exists():
        raise FileNotFoundError(f"Missing first chunk prediction file: {first_chunk_path}")

    with h5py.File(first_chunk_path, "r") as handle:
        first_dset = handle["main"]
        channel_count = int(first_dset.shape[0])
        output_dtype = first_dset.dtype

    transform_cfg = getattr(cfg.inference, "prediction_transform", None)

    def write_chunks(dataset) -> None:
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_path = _chunk_file_path(chunks_dir, chunk)
            if not chunk_path.exists():
                raise FileNotFoundError(
                    f"Missing chunk prediction file {chunk_idx}/{len(chunks)}: {chunk_path}"
                )

            expected_spatial = tuple(
                int(chunk.stop[axis]) - int(chunk.start[axis]) for axis in range(3)
            )
            with h5py.File(chunk_path, "r") as handle:
                source = handle["main"]
                if int(source.shape[0]) != channel_count:
                    raise ValueError(
                        f"Chunk {chunk.key} channel mismatch: "
                        f"{source.shape[0]} vs {channel_count}"
                    )
                if tuple(int(v) for v in source.shape[-3:]) != expected_spatial:
                    raise ValueError(
                        f"Chunk {chunk.key} spatial shape mismatch: "
                        f"{tuple(source.shape[-3:])} vs {expected_spatial}"
                    )

                # Stream by z slabs so stitching never materializes multi-GB chunks.
                slab_depth = max(1, int(h5_spatial_chunks[0]))
                for local_z0 in range(0, expected_spatial[0], slab_depth):
                    local_z1 = min(local_z0 + slab_depth, expected_spatial[0])
                    global_z0 = int(chunk.start[0]) + local_z0
                    global_z1 = int(chunk.start[0]) + local_z1
                    slab = source[
                        (
                            slice(None),
                            slice(local_z0, local_z1),
                            slice(None),
                            slice(None),
                        )
                    ]
                    dataset[
                        (
                            slice(None),
                            slice(global_z0, global_z1),
                            slice(int(chunk.start[1]), int(chunk.stop[1])),
                            slice(int(chunk.start[2]), int(chunk.stop[2])),
                        )
                    ] = slab
                    if qc_streaming_callback is not None:
                        qc_streaming_callback.update(slab, z_offset=global_z0, z_axis=1)

    write_prediction_artifact(
        output_path,
        metadata=build_prediction_artifact_metadata(
            cfg,
            image_path=str(image_path),
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            output_head=requested_head,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            chunk_shape=chunk_shape,
            halo=halo,
            intensity_scale=(
                float(getattr(transform_cfg, "intensity_scale", -1.0))
                if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                else None
            ),
            intensity_dtype=(
                str(getattr(transform_cfg, "intensity_dtype", output_dtype))
                if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                else str(output_dtype)
            ),
            extra={
                "compression": str(compression),
                "chunk_stitch_source": str(chunks_dir),
            },
        ),
        compression=compression,
        shape=(channel_count, *final_shape),
        dtype=output_dtype,
        chunks=(channel_count, *h5_spatial_chunks),
        writer=write_chunks,
    )
    return output_path


def _run_chunked_prediction_per_rank(
    *,
    cfg: Any,
    forward_fn,
    image_path: str,
    output_path: Path,
    checkpoint_path: str | Path | None,
    mask_path: str | None,
    mask_align_to_image: bool,
    requested_head: str | None,
    device: torch.device | str,
    chunks: list[ChunkRef],
    input_shape: tuple[int, int, int],
    final_shape: tuple[int, int, int],
    crop_pad: tuple[tuple[int, int], ...],
    crop_before: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    compression,
    h5_spatial_chunks: tuple[int, int, int],
    rank: int,
    world_size: int,
    qc_streaming_callback: Any = None,
    stitch_output: bool = True,
    use_distributed_barrier: bool = True,
) -> Path:
    """Per-rank chunked raw inference. Each rank writes its own per-chunk h5 files.

    Layout: output_path.chunks/chunk_{key}.h5 per chunk, plus a rank-0 index.json
    listing chunk metadata (for downstream stitching/decoding).
    """
    chunks_dir = _per_chunk_dir(output_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    my_chunks = [(idx, chunk) for idx, chunk in enumerate(chunks) if idx % world_size == rank]
    logger.info(
        "Per-rank chunked raw prediction: rank=%d/%d, total_chunks=%d, my_chunks=%d, "
        "chunk_shape=%s, halo=%s",
        rank,
        world_size,
        len(chunks),
        len(my_chunks),
        chunk_shape,
        halo,
    )

    transform_cfg = getattr(cfg.inference, "prediction_transform", None)

    # Optional: stream chunks straight into a CloudVolume precomputed layer instead of
    # per-chunk HDF5 + stitching, so ABISS/Seuron can read inference output directly.
    chunking_cfg = cfg.inference.chunking
    precomputed_out = bool(getattr(chunking_cfg, "precomputed", False))
    precomputed_cv: Any = None
    precomputed_dir = output_path.with_suffix("")
    if precomputed_out:
        pc_resolution = getattr(chunking_cfg, "precomputed_resolution", None)
        if not pc_resolution:
            raise ValueError(
                "inference.chunking.precomputed requires "
                "inference.chunking.precomputed_resolution (XYZ nm)."
            )
        pc_chunk_xyz = list(getattr(chunking_cfg, "precomputed_chunk_size", [128, 128, 64]))
        _validate_precomputed_alignment(chunk_shape, pc_chunk_xyz)

    for local_pos, (chunk_idx, chunk) in enumerate(my_chunks, start=1):
        chunk_path = (
            _precomputed_marker_path(chunks_dir, chunk)
            if precomputed_out
            else _chunk_file_path(chunks_dir, chunk)
        )
        if chunk_path.exists():
            logger.info(
                "[rank %d] chunk %d/%d %s: already exists, skipping",
                rank,
                chunk_idx,
                len(chunks),
                chunk.key,
            )
            continue

        pred_core_start = tuple(chunk.start[axis] + crop_before[axis] for axis in range(3))
        pred_core_stop = tuple(chunk.stop[axis] + crop_before[axis] for axis in range(3))
        read_start = tuple(max(0, pred_core_start[axis] - halo[axis]) for axis in range(3))
        read_stop = tuple(
            min(input_shape[axis], pred_core_stop[axis] + halo[axis]) for axis in range(3)
        )
        read_shape = tuple(read_stop[axis] - read_start[axis] for axis in range(3))
        core_shape = tuple(pred_core_stop[axis] - pred_core_start[axis] for axis in range(3))

        logger.info(
            "[rank %d] chunk %d/%d (%d/%d local) %s: core_shape=%s read_shape=%s "
            "core=%s:%s read=%s:%s",
            rank,
            chunk_idx,
            len(chunks),
            local_pos,
            len(my_chunks),
            chunk.key,
            core_shape,
            read_shape,
            pred_core_start,
            pred_core_stop,
            read_start,
            read_stop,
        )

        pred_tensor = lazy_predict_region(
            cfg,
            forward_fn,
            image_path,
            region_start=read_start,
            region_stop=read_stop,
            mask_path=mask_path,
            mask_align_to_image=mask_align_to_image,
            device=device,
            requested_head=requested_head,
        )
        pred = pred_tensor.detach().cpu().numpy()[0]
        del pred_tensor

        local_core_slices = tuple(
            slice(
                pred_core_start[axis] - read_start[axis],
                pred_core_stop[axis] - read_start[axis],
            )
            for axis in range(3)
        )
        core_pred = pred[(slice(None), *local_core_slices)]
        core_pred = apply_prediction_transform(cfg, core_pred)
        core_pred = apply_storage_dtype_transform(cfg, core_pred)

        channel_count = int(core_pred.shape[0])

        if precomputed_out:
            if precomputed_cv is None:
                precomputed_cv = _open_precomputed_layer(
                    precomputed_dir,
                    volume_size_xyz=tuple(reversed(final_shape)),
                    num_channels=channel_count,
                    data_type=str(core_pred.dtype),
                    resolution_xyz=pc_resolution,
                    chunk_size_xyz=pc_chunk_xyz,
                )
            # (C, Z, Y, X) -> CloudVolume's (X, Y, Z, C)
            z0, y0, x0 = (int(chunk.start[axis]) for axis in range(3))
            block = np.transpose(core_pred, (3, 2, 1, 0))
            x1, y1, z1 = (x0 + block.shape[0], y0 + block.shape[1], z0 + block.shape[2])
            precomputed_cv[x0:x1, y0:y1, z0:z1, :] = block
            chunk_path.write_text(
                json.dumps(
                    {
                        "chunk_key": chunk.key,
                        "chunk_start_zyx": list(chunk.start),
                        "chunk_stop_zyx": list(chunk.stop),
                        "written_xyz": [[x0, y0, z0], [x1, y1, z1]],
                    }
                )
            )
            logger.info(
                "[rank %d] chunk %d/%d %s -> precomputed [%d:%d, %d:%d, %d:%d]",
                rank, chunk_idx, len(chunks), chunk.key, x0, x1, y0, y1, z0, z1,
            )
            del pred, core_pred, block
            continue

        chunk_h5_spatial_chunks = tuple(
            max(1, min(int(h5_spatial_chunks[axis]), int(core_shape[axis]))) for axis in range(3)
        )
        write_prediction_artifact(
            chunk_path,
            core_pred,
            metadata=build_prediction_artifact_metadata(
                cfg,
                image_path=str(image_path),
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                output_head=requested_head,
                input_shape=read_shape,
                final_shape=core_shape,
                chunk_shape=core_shape,
                halo=halo,
                intensity_scale=(
                    float(getattr(transform_cfg, "intensity_scale", -1.0))
                    if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                    else None
                ),
                intensity_dtype=(
                    str(getattr(transform_cfg, "intensity_dtype", core_pred.dtype))
                    if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                    else str(core_pred.dtype)
                ),
                extra={
                    "compression": str(compression),
                    "chunk_key": chunk.key,
                    "chunk_index_zyx": list(chunk.index),
                    "chunk_start_zyx": list(chunk.start),
                    "chunk_stop_zyx": list(chunk.stop),
                    "chunk_read_start_zyx": list(read_start),
                    "chunk_read_stop_zyx": list(read_stop),
                    "chunk_read_shape_zyx": list(read_shape),
                },
            ),
            compression=compression,
            chunks=(channel_count, *chunk_h5_spatial_chunks),
        )

        del pred, core_pred

    if use_distributed_barrier:
        _distributed_barrier()

    if precomputed_out:
        # The layer *is* the output: every chunk wrote its own disjoint region, so there
        # is nothing to stitch and no whole-volume artifact to assemble.
        if rank == 0:
            logger.info(
                "Chunked raw prediction wrote %d chunks into precomputed layer %s",
                len(chunks),
                precomputed_dir,
            )
        return precomputed_dir

    if rank == 0:
        index_path = _write_chunk_index(
            output_path=output_path,
            chunks_dir=chunks_dir,
            chunks=chunks,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            chunk_shape=chunk_shape,
            halo=halo,
            checkpoint_path=checkpoint_path,
            world_size=world_size,
        )
        logger.info(
            "Chunked raw prediction shard wrote chunk metadata for %d chunks to %s; index=%s",
            len(chunks),
            chunks_dir,
            index_path,
        )
        if not stitch_output:
            return chunks_dir

        logger.info("Stitching %d per-rank chunks into %s", len(chunks), output_path)
        _stitch_chunk_prediction_files(
            cfg=cfg,
            image_path=image_path,
            output_path=output_path,
            chunks_dir=chunks_dir,
            chunks=chunks,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            chunk_shape=chunk_shape,
            halo=halo,
            compression=compression,
            h5_spatial_chunks=h5_spatial_chunks,
            checkpoint_path=checkpoint_path,
            requested_head=requested_head,
            qc_streaming_callback=qc_streaming_callback,
        )
        logger.info("Stitched chunked raw prediction wrote %s", output_path)

    return output_path if rank == 0 else chunks_dir


def run_chunked_prediction_inference(
    cfg: Any,
    forward_fn,
    image_path: str,
    *,
    output_path: str | Path,
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
    mask_path: str | None = None,
    mask_align_to_image: bool = False,
    requested_head: str | None = None,
    qc_streaming_callback: Any = None,
) -> Path:
    """Run chunked lazy inference and stream raw predictions into one HDF5 volume."""
    validate_chunked_output_format(cfg)
    chunking_cfg = cfg.inference.chunking
    reference_shape = get_lazy_image_reference_shape(cfg, image_path, mode="test")
    input_shape = tuple(int(v) for v in reference_shape[-3:])
    crop_pad = resolve_global_prediction_crop(cfg)
    crop_before = tuple(int(crop_pad[axis][0]) for axis in range(3))
    crop_after = tuple(int(crop_pad[axis][1]) for axis in range(3))
    final_shape = tuple(
        input_shape[axis] - crop_before[axis] - crop_after[axis] for axis in range(3)
    )
    if any(size <= 0 for size in final_shape):
        raise ValueError(
            f"Chunked inference crop {crop_pad} is too large for input shape {input_shape}."
        )

    chunk_shape = resolve_chunk_shape(cfg, final_shape)
    halo = tuple(int(v) for v in getattr(chunking_cfg, "halo", [0, 0, 0]))
    chunks = build_chunk_grid(final_shape, chunk_shape)
    roi = _resolve_inference_roi(cfg)
    if roi is not None:
        n_all = len(chunks)
        chunks = _filter_chunks_to_roi(chunks, roi, crop_before)
        if not chunks:
            raise ValueError(
                f"inference.chunking.roi={roi} excludes every chunk (final_shape={final_shape})."
            )
        logger.info(
            "Inference ROI %s (input ZYX voxels): kept %d/%d chunks, skipped %d pure-padding.",
            roi, len(chunks), n_all, n_all - len(chunks),
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = getattr(cfg.inference, "save_compression", "gzip")
    compression = None if compression in (None, "", "none") else compression
    h5_spatial_chunks = resolve_h5_spatial_chunks(final_shape)

    external_shard = _resolve_external_chunk_shard(cfg)
    if external_shard is not None:
        shard_id, num_shards = external_shard
        return _run_chunked_prediction_per_rank(
            cfg=cfg,
            forward_fn=forward_fn,
            image_path=image_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            mask_path=mask_path,
            mask_align_to_image=mask_align_to_image,
            requested_head=requested_head,
            device=device,
            chunks=chunks,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            crop_before=crop_before,
            chunk_shape=chunk_shape,
            halo=halo,
            compression=compression,
            h5_spatial_chunks=h5_spatial_chunks,
            rank=shard_id,
            world_size=num_shards,
            qc_streaming_callback=None,
            stitch_output=False,
            use_distributed_barrier=False,
        )

    rank, world_size = _resolve_distributed_rank()
    if world_size > 1:
        return _run_chunked_prediction_per_rank(
            cfg=cfg,
            forward_fn=forward_fn,
            image_path=image_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            mask_path=mask_path,
            mask_align_to_image=mask_align_to_image,
            requested_head=requested_head,
            device=device,
            chunks=chunks,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            crop_before=crop_before,
            chunk_shape=chunk_shape,
            halo=halo,
            compression=compression,
            h5_spatial_chunks=h5_spatial_chunks,
            rank=rank,
            world_size=world_size,
            qc_streaming_callback=qc_streaming_callback,
        )

    logger.info(
        "Chunked raw prediction inference: input_shape=%s, final_shape=%s, "
        "chunk_shape=%s, halo=%s, chunks=%d",
        input_shape,
        final_shape,
        chunk_shape,
        halo,
        len(chunks),
    )

    def iter_core_predictions():
        for chunk_idx, chunk in enumerate(chunks, start=1):
            pred_core_start = tuple(chunk.start[axis] + crop_before[axis] for axis in range(3))
            pred_core_stop = tuple(chunk.stop[axis] + crop_before[axis] for axis in range(3))
            read_start = tuple(max(0, pred_core_start[axis] - halo[axis]) for axis in range(3))
            read_stop = tuple(
                min(input_shape[axis], pred_core_stop[axis] + halo[axis]) for axis in range(3)
            )

            logger.info(
                "Raw prediction chunk %d/%d %s: core=%s:%s read=%s:%s",
                chunk_idx,
                len(chunks),
                chunk.key,
                pred_core_start,
                pred_core_stop,
                read_start,
                read_stop,
            )
            pred_tensor = lazy_predict_region(
                cfg,
                forward_fn,
                image_path,
                region_start=read_start,
                region_stop=read_stop,
                mask_path=mask_path,
                mask_align_to_image=mask_align_to_image,
                device=device,
                requested_head=requested_head,
            )
            pred = pred_tensor.detach().cpu().numpy()[0]
            del pred_tensor

            local_core_slices = tuple(
                slice(
                    pred_core_start[axis] - read_start[axis],
                    pred_core_stop[axis] - read_start[axis],
                )
                for axis in range(3)
            )
            core_pred = pred[(slice(None), *local_core_slices)]
            core_pred = apply_prediction_transform(cfg, core_pred)
            core_pred = apply_storage_dtype_transform(cfg, core_pred)
            yield chunk, core_pred
            del pred, core_pred

    prediction_iter = iter_core_predictions()
    first_chunk, first_core_pred = next(prediction_iter)
    channel_count = int(first_core_pred.shape[0])
    transform_cfg = getattr(cfg.inference, "prediction_transform", None)

    def _stream_qc(chunk_obj, arr) -> None:
        if qc_streaming_callback is None:
            return
        z0 = int(chunk_obj.slices[0].start)
        qc_streaming_callback.update(arr, z_offset=z0, z_axis=1)

    def write_chunks(
        dataset,
        initial_chunk=first_chunk,
        initial_core_pred=first_core_pred,
    ) -> None:
        dataset[(slice(None), *initial_chunk.slices)] = initial_core_pred
        _stream_qc(initial_chunk, initial_core_pred)
        for chunk, core_pred in prediction_iter:
            dataset[(slice(None), *chunk.slices)] = core_pred
            _stream_qc(chunk, core_pred)

    write_prediction_artifact(
        output_path,
        metadata=build_prediction_artifact_metadata(
            cfg,
            image_path=str(image_path),
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            output_head=requested_head,
            input_shape=input_shape,
            final_shape=final_shape,
            crop_pad=crop_pad,
            chunk_shape=chunk_shape,
            halo=halo,
            intensity_scale=(
                float(getattr(transform_cfg, "intensity_scale", -1.0))
                if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                else None
            ),
            intensity_dtype=(
                str(getattr(transform_cfg, "intensity_dtype", first_core_pred.dtype))
                if transform_cfg is not None and bool(getattr(transform_cfg, "enabled", False))
                else str(first_core_pred.dtype)
            ),
            extra={"compression": str(compression)},
        ),
        compression=compression,
        shape=(channel_count, *final_shape),
        dtype=first_core_pred.dtype,
        chunks=(channel_count, *h5_spatial_chunks),
        writer=write_chunks,
    )
    del first_core_pred

    logger.info("Chunked raw prediction inference wrote %s.", output_path)
    return output_path


__all__ = [
    "ChunkRef",
    "is_external_chunk_sharding_enabled",
    "is_chunked_inference_enabled",
    "run_chunked_prediction_inference",
]
