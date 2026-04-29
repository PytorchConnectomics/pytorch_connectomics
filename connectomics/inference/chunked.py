"""Chunked inference+decoding for large affinity volumes."""

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
from .lazy import get_lazy_image_reference_shape, lazy_predict_region
from .output import apply_save_prediction_transform

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


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent.setdefault(value, value)
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


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


def _resolve_decode_affinity_cc_kwargs(cfg: Any) -> dict[str, Any]:
    # Import lazily: top-level decoding imports tuning helpers, which import
    # Lightning, which imports inference. Keeping this out of module import
    # avoids an inference<->training cycle.
    from ..decoding.pipeline import normalize_decode_modes

    steps = [
        step
        for step in normalize_decode_modes(getattr(cfg, "decoding", None) or [])
        if step.enabled
    ]
    if len(steps) != 1 or steps[0].name != "decode_affinity_cc":
        raise ValueError(
            "Chunked inference currently supports exactly one active decoding step: "
            "decode_affinity_cc."
        )
    return dict(steps[0].kwargs)


def _validate_chunked_output_contract(cfg: Any) -> None:
    save_cfg = getattr(getattr(cfg, "inference", None), "save_prediction", None)
    formats = [str(fmt).lower() for fmt in getattr(save_cfg, "output_formats", ["h5"])]
    unsupported_formats = [fmt for fmt in formats if fmt not in {"h5", "hdf5"}]
    if unsupported_formats:
        raise ValueError(
            "Chunked inference writes a single streamed HDF5 output only; "
            f"unsupported save_prediction.output_formats={unsupported_formats}."
        )

    post_cfg = getattr(getattr(cfg, "inference", None), "postprocessing", None)
    if post_cfg is None or not bool(getattr(post_cfg, "enabled", False)):
        return
    if getattr(post_cfg, "output_transpose", None):
        raise ValueError("Chunked inference does not support postprocessing.output_transpose yet.")
    binary_cfg = getattr(post_cfg, "binary", None) or {}
    if isinstance(binary_cfg, dict) and bool(binary_cfg.get("enabled", False)):
        raise ValueError("Chunked inference does not support binary postprocessing yet.")
    instance_cc3d_cfg = getattr(post_cfg, "instance_cc3d", None) or {}
    if isinstance(instance_cc3d_cfg, dict) and bool(instance_cc3d_cfg.get("enabled", False)):
        raise ValueError("Chunked inference does not support instance_cc3d postprocessing yet.")


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


def _take_face(array: np.ndarray, axis: int, side: str) -> np.ndarray:
    index = 0 if side == "neg" else -1
    return np.take(array, index, axis=axis)


def _extract_positive_seam_affinity(
    pred: np.ndarray,
    *,
    axis: int,
    core_start: Sequence[int],
    core_stop: Sequence[int],
    read_start: Sequence[int],
    read_stop: Sequence[int],
    threshold: float,
    edge_offset: int,
) -> np.ndarray:
    storage_global = int(core_stop[axis]) - 1 + int(edge_offset)
    if not (int(read_start[axis]) <= storage_global < int(read_stop[axis])):
        raise ValueError(
            "Chunk halo is too small for boundary stitching: "
            f"axis={axis}, storage_global={storage_global}, "
            f"read_start={tuple(read_start)}, read_stop={tuple(read_stop)}."
        )

    slices: list[slice | int] = []
    for dim in range(3):
        if dim == axis:
            slices.append(storage_global - int(read_start[dim]))
        else:
            slices.append(
                slice(
                    int(core_start[dim]) - int(read_start[dim]),
                    int(core_stop[dim]) - int(read_start[dim]),
                )
            )
    return pred[axis][tuple(slices)] > float(threshold)


def _union_face_pairs(
    uf: UnionFind,
    src_face: np.ndarray,
    dst_face: np.ndarray,
    seam_affinity: np.ndarray,
    *,
    min_contact: int,
) -> int:
    mask = seam_affinity & (src_face > 0) & (dst_face > 0) & (src_face != dst_face)
    if not mask.any():
        return 0
    pairs = np.stack([src_face[mask].ravel(), dst_face[mask].ravel()], axis=1).astype(
        np.int64, copy=False
    )
    if pairs.size == 0:
        return 0
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    kept = unique_pairs[counts >= int(min_contact)]
    for a, b in kept:
        uf.union(int(a), int(b))
    return int(len(kept))


def _remap_chunk(seg: np.ndarray, uf: UnionFind) -> np.ndarray:
    unique_values = np.unique(seg)
    mapped_values = np.array(
        [0 if int(value) == 0 else uf.find(int(value)) for value in unique_values],
        dtype=np.uint64,
    )
    remapped = mapped_values[np.searchsorted(unique_values, seg)]
    if remapped.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError("Chunked output label IDs exceed uint32 range.")
    return remapped.astype(np.uint32, copy=False)


def _write_h5(path: Path, data: np.ndarray, *, dataset: str = "main") -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=data, compression="gzip")


def _read_h5(path: Path, *, dataset: str = "main") -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset])


def run_chunked_prediction_inference(
    cfg: Any,
    forward_fn,
    image_path: str,
    *,
    output_path: str | Path,
    device: torch.device | str,
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

    logger.info(
        "Chunked raw prediction inference: input_shape=%s, final_shape=%s, "
        "chunk_shape=%s, halo=%s, chunks=%d",
        input_shape,
        final_shape,
        chunk_shape,
        halo,
        len(chunks),
    )

    import h5py

    with h5py.File(output_path, "w") as handle:
        dataset = None
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
            pred = pred_tensor.detach().cpu().float().numpy()[0]
            del pred_tensor

            local_core_slices = tuple(
                slice(
                    pred_core_start[axis] - read_start[axis],
                    pred_core_stop[axis] - read_start[axis],
                )
                for axis in range(3)
            )
            core_pred = pred[(slice(None), *local_core_slices)]
            core_pred = apply_save_prediction_transform(cfg, core_pred)

            if dataset is None:
                channel_count = int(core_pred.shape[0])
                dataset = handle.create_dataset(
                    "main",
                    shape=(channel_count, *final_shape),
                    dtype=core_pred.dtype,
                    chunks=(channel_count, *h5_spatial_chunks),
                    compression=compression,
                )
                dataset.attrs["kind"] = "raw_prediction"
                dataset.attrs["image_path"] = str(image_path)
                dataset.attrs["input_shape"] = json.dumps(list(input_shape))
                dataset.attrs["final_shape"] = json.dumps(list(final_shape))
                dataset.attrs["crop_pad"] = json.dumps([list(pair) for pair in crop_pad])
                dataset.attrs["chunk_shape"] = json.dumps(list(chunk_shape))
                dataset.attrs["halo"] = json.dumps(list(halo))
                save_cfg = getattr(cfg.inference, "save_prediction", None)
                if save_cfg is not None:
                    dataset.attrs["intensity_scale"] = float(
                        getattr(save_cfg, "intensity_scale", -1.0)
                    )
                    dataset.attrs["intensity_dtype"] = str(
                        getattr(save_cfg, "intensity_dtype", core_pred.dtype)
                    )

            dataset[(slice(None), *chunk.slices)] = core_pred
            del pred, core_pred

    logger.info("Chunked raw prediction inference wrote %s.", output_path)
    return output_path


def run_chunked_affinity_cc_inference(
    cfg: Any,
    forward_fn,
    image_path: str,
    *,
    output_path: str | Path,
    device: torch.device | str,
    mask_path: str | None = None,
    mask_align_to_image: bool = False,
    requested_head: str | None = None,
) -> Path:
    """Run chunked lazy inference, decode chunks, stitch boundaries, and write HDF5."""
    from ..decoding.decoders.segmentation import decode_affinity_cc

    _validate_chunked_output_contract(cfg)
    output_mode = _resolve_chunk_output_mode(cfg)
    if output_mode != "decoded":
        raise ValueError(
            "run_chunked_affinity_cc_inference requires "
            "inference.chunking.output_mode='decoded'."
        )
    chunking_cfg = cfg.inference.chunking
    decode_kwargs = _resolve_decode_affinity_cc_kwargs(cfg)
    threshold = float(
        getattr(chunking_cfg.stitching, "threshold", None)
        if getattr(chunking_cfg.stitching, "threshold", None) is not None
        else decode_kwargs.get("threshold", 0.5)
    )
    edge_offset = int(
        getattr(chunking_cfg.stitching, "edge_offset", None)
        if getattr(chunking_cfg.stitching, "edge_offset", None) is not None
        else decode_kwargs.get("edge_offset", 0)
    )
    min_contact = int(getattr(chunking_cfg.stitching, "min_contact", 1))
    method = str(getattr(chunking_cfg.stitching, "method", "affinity_cc_boundary_union"))
    if method != "affinity_cc_boundary_union":
        raise ValueError(
            "Chunked affinity decoding currently supports stitching.method="
            "'affinity_cc_boundary_union' only."
        )

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
    temp_root = (
        Path(chunking_cfg.temp_dir)
        if getattr(chunking_cfg, "temp_dir", "")
        else output_path.parent / f"{output_path.stem}_chunks"
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Chunked affinity inference: input_shape=%s, final_shape=%s, chunk_shape=%s, "
        "halo=%s, chunks=%d, threshold=%.4f, edge_offset=%d",
        input_shape,
        final_shape,
        chunk_shape,
        halo,
        len(chunks),
        threshold,
        edge_offset,
    )

    uf = UnionFind()
    pending_positive_faces: dict[
        tuple[int, tuple[int, int, int]], tuple[np.ndarray, np.ndarray]
    ] = {}
    chunk_paths: dict[str, Path] = {}
    next_offset = 0
    total_boundary_unions = 0

    for chunk_idx, chunk in enumerate(chunks, start=1):
        pred_core_start = tuple(chunk.start[axis] + crop_before[axis] for axis in range(3))
        pred_core_stop = tuple(chunk.stop[axis] + crop_before[axis] for axis in range(3))
        read_start = tuple(max(0, pred_core_start[axis] - halo[axis]) for axis in range(3))
        read_stop = tuple(
            min(input_shape[axis], pred_core_stop[axis] + halo[axis]) for axis in range(3)
        )

        logger.info(
            "Chunk %d/%d %s: core=%s:%s read=%s:%s",
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
        pred = pred_tensor.detach().cpu().float().numpy()[0]
        del pred_tensor

        decoded = decode_affinity_cc(pred, **decode_kwargs)
        local_core_slices = tuple(
            slice(
                pred_core_start[axis] - read_start[axis],
                pred_core_stop[axis] - read_start[axis],
            )
            for axis in range(3)
        )
        core_seg = np.asarray(decoded[local_core_slices], dtype=np.uint64)
        local_max = int(core_seg.max(initial=0))
        if local_max > 0:
            core_seg[core_seg > 0] += next_offset
            next_offset += local_max
        if next_offset > np.iinfo(np.uint32).max:
            raise ValueError("Chunked output label IDs exceed uint32 range.")
        core_seg_u32 = core_seg.astype(np.uint32, copy=False)

        for axis in range(3):
            if chunk.index[axis] > 0:
                neighbor_index = list(chunk.index)
                neighbor_index[axis] -= 1
                key = (axis, tuple(neighbor_index))
                pending = pending_positive_faces.pop(key, None)
                if pending is None:
                    raise RuntimeError(f"Missing pending positive face for border {key}.")
                src_face, seam_affinity = pending
                dst_face = _take_face(core_seg_u32, axis, "neg")
                total_boundary_unions += _union_face_pairs(
                    uf,
                    src_face,
                    dst_face,
                    seam_affinity,
                    min_contact=min_contact,
                )

            if chunk.stop[axis] < final_shape[axis]:
                src_face = _take_face(core_seg_u32, axis, "pos").copy()
                seam_affinity = _extract_positive_seam_affinity(
                    pred,
                    axis=axis,
                    core_start=pred_core_start,
                    core_stop=pred_core_stop,
                    read_start=read_start,
                    read_stop=read_stop,
                    threshold=threshold,
                    edge_offset=edge_offset,
                ).copy()
                pending_positive_faces[(axis, chunk.index)] = (src_face, seam_affinity)

        chunk_path = temp_root / f"{chunk.key}.h5"
        _write_h5(chunk_path, core_seg_u32)
        chunk_paths[chunk.key] = chunk_path
        del pred, decoded, core_seg, core_seg_u32

    if pending_positive_faces:
        raise RuntimeError(f"Unconsumed chunk boundary faces remain: {len(pending_positive_faces)}")

    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = getattr(getattr(cfg.inference, "save_prediction", None), "compression", "gzip")
    compression = None if compression in (None, "", "none") else compression
    h5_chunks = _resolve_h5_spatial_chunks(final_shape)
    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset(
            "main",
            shape=final_shape,
            dtype=np.uint32,
            chunks=h5_chunks,
            compression=compression,
        )
        for chunk in chunks:
            seg = _read_h5(chunk_paths[chunk.key])
            dataset[chunk.slices] = _remap_chunk(seg, uf)

    metadata = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "input_shape": list(input_shape),
        "final_shape": list(final_shape),
        "crop_pad": [list(pair) for pair in crop_pad],
        "chunk_shape": list(chunk_shape),
        "halo": list(halo),
        "chunks": len(chunks),
        "boundary_unions": int(total_boundary_unions),
        "decode_kwargs": decode_kwargs,
    }
    with (temp_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info(
        "Chunked affinity inference wrote %s (%d boundary union pairs).",
        output_path,
        total_boundary_unions,
    )
    return output_path


__all__ = [
    "ChunkRef",
    "UnionFind",
    "is_chunked_inference_enabled",
    "run_chunked_affinity_cc_inference",
    "run_chunked_prediction_inference",
]
