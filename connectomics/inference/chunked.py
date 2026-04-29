"""Chunked raw-prediction inference for large lazy volumes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..data.processing.affinity import (
    compute_affinity_crop_pad,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
)
from ..utils.channel_slices import resolve_channel_indices
from .artifact import (
    PredictionArtifactMetadata,
    write_prediction_artifact,
)
from .lazy import get_lazy_image_reference_shape, lazy_predict_region
from .output import apply_prediction_transform, apply_storage_dtype_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRef:
    index: tuple[int, int, int]
    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def key(self) -> str:
        z, y, x = self.index
        return f"z{z}_y{y}_x{x}"

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(self.start[axis], self.stop[axis]) for axis in range(3))


def is_chunked_inference_enabled(cfg: Any) -> bool:
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return False
    strategy = str(getattr(inference_cfg, "strategy", "whole_volume")).lower()
    chunking_cfg = getattr(inference_cfg, "chunking", None)
    return strategy == "chunked" or bool(getattr(chunking_cfg, "enabled", False))


def _build_chunk_grid(volume_shape: Sequence[int], chunk_shape: Sequence[int]) -> list[ChunkRef]:
    volume = tuple(int(v) for v in volume_shape)
    chunk = tuple(int(v) for v in chunk_shape)
    counts = tuple((volume[axis] + chunk[axis] - 1) // chunk[axis] for axis in range(3))
    result: list[ChunkRef] = []
    for index in product(*(range(count) for count in counts)):
        start = tuple(index[axis] * chunk[axis] for axis in range(3))
        stop = tuple(min(start[axis] + chunk[axis], volume[axis]) for axis in range(3))
        result.append(ChunkRef(index=tuple(int(v) for v in index), start=start, stop=stop))
    return result


def _normalize_crop_pad(value: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if value in (None, [], ()):
        return ((0, 0), (0, 0), (0, 0))
    values = [int(v) for v in value]
    if len(values) == 3:
        return tuple((v, v) for v in values)  # type: ignore[return-value]
    if len(values) == 6:
        return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    raise ValueError(f"inference.crop_pad must have length 3 or 6, got {value!r}")


def _resolve_selected_affinity_offsets(cfg: Any) -> list[tuple[int, int, int]]:
    groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not groups:
        return []

    label_channels = max(end for (start, end), _offsets in groups)
    channel_offsets: list[tuple[int, int, int] | None] = [None] * label_channels
    for (start, end), offsets in groups:
        for channel, offset in zip(range(start, end), offsets):
            channel_offsets[channel] = offset

    select_channel = getattr(getattr(cfg, "inference", None), "select_channel", None)
    if select_channel is not None:
        selected = resolve_channel_indices(
            select_channel,
            num_channels=len(channel_offsets),
            context="inference.select_channel",
        )
        channel_offsets = [channel_offsets[idx] for idx in selected]

    return [offset for offset in channel_offsets if offset is not None]


def _resolve_global_prediction_crop(
    cfg: Any,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    user_crop = _normalize_crop_pad(getattr(getattr(cfg, "inference", None), "crop_pad", None))
    affinity_mode = resolve_affinity_mode_from_cfg(cfg)
    if affinity_mode is None:
        affinity_crop = ((0, 0), (0, 0), (0, 0))
    else:
        offsets = _resolve_selected_affinity_offsets(cfg)
        affinity_crop = (
            compute_affinity_crop_pad(offsets, affinity_mode=affinity_mode)
            if offsets
            else ((0, 0), (0, 0), (0, 0))
        )
    return tuple(
        (
            int(user_crop[axis][0]) + int(affinity_crop[axis][0]),
            int(user_crop[axis][1]) + int(affinity_crop[axis][1]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _validate_chunked_output_contract(cfg: Any) -> None:
    save_cfg = getattr(getattr(cfg, "inference", None), "save_prediction", None)
    formats = [str(fmt).lower() for fmt in getattr(save_cfg, "output_formats", ["h5"])]
    unsupported_formats = [fmt for fmt in formats if fmt not in {"h5", "hdf5"}]
    if unsupported_formats:
        raise ValueError(
            "Chunked inference writes a single streamed HDF5 output only; "
            f"unsupported save_prediction.output_formats={unsupported_formats}."
        )


def _resolve_chunk_shape(cfg: Any, final_shape: Sequence[int]) -> tuple[int, int, int]:
    chunking_cfg = cfg.inference.chunking
    chunk_size = tuple(int(v) for v in chunking_cfg.chunk_size)
    axes = str(getattr(chunking_cfg, "axes", "all")).lower()
    if axes == "z":
        return (chunk_size[0], int(final_shape[1]), int(final_shape[2]))
    if axes != "all":
        raise ValueError("inference.chunking.axes must be 'all' or 'z'")
    return tuple(min(chunk_size[axis], int(final_shape[axis])) for axis in range(3))


def _resolve_h5_spatial_chunks(spatial_shape: Sequence[int]) -> tuple[int, int, int]:
    preferred = (64, 64, 64)
    return tuple(min(int(spatial_shape[axis]), preferred[axis]) for axis in range(3))


def _resolve_chunk_output_mode(cfg: Any) -> str:
    chunking_cfg = cfg.inference.chunking
    mode = str(getattr(chunking_cfg, "output_mode", "decoded")).lower()
    if mode not in {"decoded", "raw_prediction"}:
        raise ValueError("inference.chunking.output_mode must be 'decoded' or 'raw_prediction'.")
    return mode


def _resolve_distributed_rank() -> tuple[int, int]:
    """Return (rank, world_size). (0, 1) when torch.distributed isn't initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def _per_chunk_dir(output_path: Path) -> Path:
    """Sibling directory holding per-chunk h5 files for distributed chunked inference."""
    return output_path.with_suffix(output_path.suffix + ".chunks")


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
        "input_shape=%s, final_shape=%s, chunk_shape=%s, halo=%s",
        rank,
        world_size,
        len(chunks),
        len(my_chunks),
        input_shape,
        final_shape,
        chunk_shape,
        halo,
    )

    transform_cfg = getattr(cfg.inference, "prediction_transform", None)

    for local_pos, (chunk_idx, chunk) in enumerate(my_chunks, start=1):
        chunk_path = chunks_dir / f"chunk_{chunk.key}.h5"
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

        logger.info(
            "[rank %d] chunk %d/%d (%d/%d local) %s: core=%s:%s read=%s:%s",
            rank,
            chunk_idx,
            len(chunks),
            local_pos,
            len(my_chunks),
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

        channel_count = int(core_pred.shape[0])
        write_prediction_artifact(
            chunk_path,
            core_pred,
            metadata=PredictionArtifactMetadata(
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
                },
            ),
            compression=compression,
            chunks=(channel_count, *h5_spatial_chunks),
        )

        del pred, core_pred

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
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
                    "path": str(
                        (chunks_dir / f"chunk_{chunk.key}.h5").relative_to(output_path.parent)
                    ),
                }
                for chunk in chunks
            ],
        }
        index_path = output_path.with_suffix(output_path.suffix + ".index.json")
        with open(index_path, "w") as fh:
            json.dump(index, fh, indent=2)
        logger.info(
            "Per-rank chunked raw prediction wrote %d chunks to %s; index=%s",
            len(chunks),
            chunks_dir,
            index_path,
        )

    return chunks_dir


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
) -> Path:
    """Run chunked lazy inference and stream raw predictions into one HDF5 volume."""
    _validate_chunked_output_contract(cfg)
    chunking_cfg = cfg.inference.chunking
    reference_shape = get_lazy_image_reference_shape(cfg, image_path, mode="test")
    input_shape = tuple(int(v) for v in reference_shape[-3:])
    crop_pad = _resolve_global_prediction_crop(cfg)
    crop_before = tuple(int(crop_pad[axis][0]) for axis in range(3))
    crop_after = tuple(int(crop_pad[axis][1]) for axis in range(3))
    final_shape = tuple(
        input_shape[axis] - crop_before[axis] - crop_after[axis] for axis in range(3)
    )
    if any(size <= 0 for size in final_shape):
        raise ValueError(
            f"Chunked inference crop {crop_pad} is too large for input shape {input_shape}."
        )

    chunk_shape = _resolve_chunk_shape(cfg, final_shape)
    halo = tuple(int(v) for v in getattr(chunking_cfg, "halo", [0, 0, 0]))
    chunks = _build_chunk_grid(final_shape, chunk_shape)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = getattr(getattr(cfg.inference, "save_prediction", None), "compression", "gzip")
    compression = None if compression in (None, "", "none") else compression
    h5_spatial_chunks = _resolve_h5_spatial_chunks(final_shape)

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

    def write_chunks(dataset) -> None:
        dataset[(slice(None), *first_chunk.slices)] = first_core_pred
        for chunk, core_pred in prediction_iter:
            dataset[(slice(None), *chunk.slices)] = core_pred

    write_prediction_artifact(
        output_path,
        metadata=PredictionArtifactMetadata(
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
    "is_chunked_inference_enabled",
    "run_chunked_prediction_inference",
]
