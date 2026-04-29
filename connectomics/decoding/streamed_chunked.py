"""Streamed chunked decoding for decoder-aware large-volume inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..inference.chunked import (
    _build_chunk_grid,
    _resolve_chunk_output_mode,
    _resolve_chunk_shape,
    _resolve_global_prediction_crop,
    _resolve_h5_spatial_chunks,
    _validate_chunked_output_contract,
)
from ..inference.lazy import get_lazy_image_reference_shape, lazy_predict_region
from ..inference.output import apply_prediction_transform
from .decoders.segmentation import decode_affinity_cc
from .pipeline import normalize_decode_modes, resolve_decode_modes_from_cfg

logger = logging.getLogger(__name__)


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


def _resolve_decode_affinity_cc_kwargs(cfg: Any) -> dict[str, Any]:
    steps = [
        step
        for step in normalize_decode_modes(resolve_decode_modes_from_cfg(cfg) or [])
        if step.enabled
    ]
    if len(steps) != 1 or steps[0].name != "decode_affinity_cc":
        raise ValueError(
            "Chunked decoding currently supports exactly one active decoding step: "
            "decode_affinity_cc."
        )
    return dict(steps[0].kwargs)


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


def run_chunked_affinity_cc_inference(
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
    """Run chunked lazy inference, decode chunks, stitch boundaries, and write HDF5."""
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
        pred = pred_tensor.detach().cpu().numpy()[0]
        del pred_tensor
        pred = apply_prediction_transform(cfg, pred)

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
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
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
    "UnionFind",
    "_union_face_pairs",
    "run_chunked_affinity_cc_inference",
]
