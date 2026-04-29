"""Raw prediction artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

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
    chunk_shape: tuple[int, ...] | None = None
    halo: tuple[int, ...] | None = None
    channel_order: tuple[str, ...] | None = None
    activation: str | None = None
    intensity_scale: float | None = None
    intensity_dtype: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


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
    data: np.ndarray,
    *,
    metadata: PredictionArtifactMetadata | None = None,
    dataset: str = "main",
    compression: str | None = "gzip",
) -> Path:
    """Write one raw prediction artifact.

    The canonical per-volume layout is ``(C, Z, Y, X)`` after inference-time
    crop and channel selection.
    """
    import h5py

    arr = np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"Prediction artifacts must use CZYX layout, got shape {arr.shape}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        dset = handle.create_dataset(dataset, data=arr, compression=compression)
        write_prediction_artifact_attrs(
            dset,
            metadata
            or PredictionArtifactMetadata(
                final_shape=tuple(int(v) for v in arr.shape[-3:]),
                intensity_dtype=str(arr.dtype),
            ),
        )
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
    "read_prediction_artifact",
    "write_prediction_artifact",
    "write_prediction_artifact_attrs",
]
