"""Raw prediction artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionArtifactMetadata:
    """Metadata stored with raw prediction artifacts."""

    kind: str = "raw_prediction"
    layout: str = "CZYX"
    image_path: str | None = None
    checkpoint_path: str | None = None
    output_head: str | None = None
    input_shape: tuple[int, ...] | None = None
    final_shape: tuple[int, ...] | None = None
    crop_pad: tuple[tuple[int, int], ...] | None = None
    transpose: tuple[int, ...] | None = None
    model_architecture: str | None = None
    model_output_identity: str | None = None
    decode_after_inference: bool | None = None
    chunk_shape: tuple[int, ...] | None = None
    halo: tuple[int, ...] | None = None
    channel_order: tuple[str, ...] | None = None
    activation: str | None = None
    intensity_scale: float | None = None
    intensity_dtype: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _cfg_get(obj: Any, path: str, default: Any = None) -> Any:
    node = obj
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(part, default)
        else:
            node = getattr(node, part, default)
    return node


def _tuple_or_none(value: Sequence[Any] | None) -> tuple[int, ...] | None:
    if value in (None, [], ()):
        return None
    return tuple(int(v) for v in value)


def _model_output_identity(cfg: Any, output_head: str | None) -> str | None:
    parts: list[str] = []
    if output_head:
        parts.append(f"head={output_head}")
    else:
        primary_head = _cfg_get(cfg, "model.primary_head")
        if primary_head:
            parts.append(f"primary_head={primary_head}")

    select_channel = _cfg_get(cfg, "inference.select_channel")
    if select_channel is not None:
        parts.append(f"select_channel={select_channel}")

    return ";".join(parts) if parts else None


def build_prediction_artifact_metadata(
    cfg: Any,
    *,
    image_path: str | None = None,
    checkpoint_path: str | None = None,
    output_head: str | None = None,
    input_shape: Sequence[int] | None = None,
    final_shape: Sequence[int] | None = None,
    crop_pad: Sequence[Sequence[int]] | None = None,
    chunk_shape: Sequence[int] | None = None,
    halo: Sequence[int] | None = None,
    intensity_scale: float | None = None,
    intensity_dtype: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> PredictionArtifactMetadata:
    """Build standard metadata for raw prediction artifacts."""
    transform_cfg = _cfg_get(cfg, "inference.prediction_transform")
    transform_enabled = bool(getattr(transform_cfg, "enabled", False))
    if intensity_scale is None and transform_enabled:
        intensity_scale = float(getattr(transform_cfg, "intensity_scale", -1.0))
    if intensity_dtype is None and transform_enabled:
        intensity_dtype = getattr(transform_cfg, "intensity_dtype", None)

    return PredictionArtifactMetadata(
        image_path=image_path,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        output_head=output_head,
        input_shape=_tuple_or_none(input_shape),
        final_shape=_tuple_or_none(final_shape),
        crop_pad=(
            tuple((int(pair[0]), int(pair[1])) for pair in crop_pad)
            if crop_pad is not None
            else None
        ),
        transpose=_tuple_or_none(_cfg_get(cfg, "data.data_transform.val_transpose")),
        model_architecture=_cfg_get(cfg, "model.arch.type"),
        model_output_identity=_model_output_identity(cfg, output_head),
        decode_after_inference=bool(_cfg_get(cfg, "inference.decode_after_inference", True)),
        chunk_shape=_tuple_or_none(chunk_shape),
        halo=_tuple_or_none(halo),
        intensity_scale=intensity_scale,
        intensity_dtype=intensity_dtype,
        extra=extra or {},
    )


def _json_attr(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return json.dumps(value)
    if isinstance(value, list):
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def write_prediction_artifact_attrs(dataset: Any, metadata: PredictionArtifactMetadata) -> None:
    """Write standard metadata attrs onto an HDF5 dataset."""
    attrs = asdict(metadata)
    extra = attrs.pop("extra", {}) or {}
    for key, value in {**attrs, **dict(extra)}.items():
        if value is not None:
            dataset.attrs[key] = _json_attr(value)


def write_prediction_artifact(
    path: str | Path,
    data: np.ndarray | None = None,
    *,
    metadata: PredictionArtifactMetadata | None = None,
    dataset: str = "main",
    compression: str | None = "gzip",
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | str | type | None = None,
    chunks: tuple[int, ...] | None = None,
    writer: Callable[[Any], None] | None = None,
) -> Path:
    """Write one raw prediction artifact.

    The canonical per-volume layout is ``(C, Z, Y, X)`` after inference-time
    crop and channel selection.
    """
    import h5py

    arr = None if data is None else np.asarray(data)
    if arr is None:
        if shape is None or dtype is None:
            raise ValueError("Streaming prediction artifacts require shape and dtype.")
        artifact_shape = tuple(int(v) for v in shape)
        artifact_dtype = np.dtype(dtype)
    else:
        if arr.ndim != 4:
            raise ValueError(f"Prediction artifacts must use CZYX layout, got shape {arr.shape}")
        artifact_shape = tuple(int(v) for v in arr.shape)
        artifact_dtype = arr.dtype

    if len(artifact_shape) != 4:
        raise ValueError(f"Prediction artifacts must use CZYX layout, got shape {artifact_shape}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        if arr is None:
            dset = handle.create_dataset(
                dataset,
                shape=artifact_shape,
                dtype=artifact_dtype,
                chunks=chunks,
                compression=compression,
            )
        else:
            dset = handle.create_dataset(
                dataset,
                data=arr,
                chunks=chunks,
                compression=compression,
            )
        write_prediction_artifact_attrs(
            dset,
            metadata
            or PredictionArtifactMetadata(
                final_shape=tuple(int(v) for v in artifact_shape[-3:]),
                intensity_dtype=str(artifact_dtype),
            ),
        )
        if writer is not None:
            writer(dset)
    return output_path


def read_prediction_artifact(
    path: str | Path,
    *,
    dataset: str = "main",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Read a raw prediction artifact and optional HDF5 attrs."""
    import h5py

    with h5py.File(path, "r") as handle:
        dset = handle[dataset]
        data = np.asarray(dset)
        metadata = dict(dset.attrs)

    if data.ndim != 4:
        raise ValueError(f"Prediction artifact must use CZYX layout, got shape {data.shape}")
    if return_metadata:
        return data, metadata
    return data


__all__ = [
    "PredictionArtifactMetadata",
    "build_prediction_artifact_metadata",
    "read_prediction_artifact",
    "write_prediction_artifact",
    "write_prediction_artifact_attrs",
]
