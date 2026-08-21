"""Bounded-read volume adapters and sparse checkpoint output artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .schema import ArtifactRef, BoundingBox


class VolumeReader(Protocol):
    spatial_shape_zyx: tuple[int, int, int]
    dtype: np.dtype
    channel_axis: int | None

    def read(self, bbox: BoundingBox) -> np.ndarray: ...

    def close(self) -> None: ...


class ArrayVolume:
    def __init__(self, array: np.ndarray, channel_axis: int | None = None) -> None:
        self.array = array
        self.channel_axis = channel_axis
        self.dtype = np.dtype(array.dtype)
        if channel_axis is None:
            if array.ndim != 3:
                raise ValueError("scalar volumes must be three-dimensional z/y/x arrays")
        elif channel_axis not in (0, -1, 3):
            raise ValueError("channel_axis must be 0 or -1")
        spatial = (
            array.shape
            if channel_axis is None
            else (array.shape[1:] if channel_axis == 0 else array.shape[:3])
        )
        self.spatial_shape_zyx = tuple(int(value) for value in spatial)

    def read(self, bbox: BoundingBox) -> np.ndarray:
        slices = bbox.as_slices()
        if self.channel_axis is None:
            return np.asarray(self.array[slices])
        if self.channel_axis == 0:
            return np.asarray(self.array[(slice(None),) + slices])
        return np.asarray(self.array[slices + (slice(None),)])

    def close(self) -> None:
        return None


class H5Volume(ArrayVolume):
    def __init__(self, path: Path, dataset: str, channel_axis: int | None) -> None:
        import h5py

        self._handle = h5py.File(path, "r")
        if dataset not in self._handle:
            self._handle.close()
            raise KeyError(f"dataset {dataset!r} not found in {path}")
        super().__init__(self._handle[dataset], channel_axis=channel_axis)

    def close(self) -> None:
        self._handle.close()


class CloudVolumeReader:
    """CloudVolume XYZ[C] storage exposed as canonical ZYX or CZYX."""

    def __init__(self, uri: str, channels: bool) -> None:
        from cloudvolume import CloudVolume

        self.volume = CloudVolume(uri, mip=0, progress=False, fill_missing=True, bounded=False)
        self.channel_axis = 0 if channels else None
        self.dtype = np.dtype(self.volume.dtype)
        self.spatial_shape_zyx = tuple(int(value) for value in self.volume.shape[:3][::-1])

    def read(self, bbox: BoundingBox) -> np.ndarray:
        z, y, x = bbox.as_slices()
        raw = np.asarray(self.volume[x, y, z])
        if raw.ndim == 4:
            converted = raw.transpose(3, 2, 1, 0)
            return converted if self.channel_axis == 0 else converted[0]
        return raw.transpose(2, 1, 0)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _ChunkRecord:
    path: Path
    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]


class ChunkedH5Volume:
    """Read a chunked inference artifact without stitching the full CZYX volume."""

    def __init__(self, path: Path, dataset: str, channel_axis: int | None) -> None:
        import h5py

        if channel_axis not in (0,):
            raise ValueError("chunked affinity artifacts require channel_axis=0")
        index_path = path.parent / path.name.replace(".h5.chunks", ".h5.index.json")
        if not index_path.is_file():
            raise FileNotFoundError(f"chunk index is missing: {index_path}")
        raw = json.loads(index_path.read_text())
        records = []
        for item in raw.get("chunks", []):
            chunk_path = path / Path(item["path"]).name
            records.append(
                _ChunkRecord(
                    chunk_path,
                    tuple(int(value) for value in item["start_zyx"]),
                    tuple(int(value) for value in item["stop_zyx"]),
                )
            )
        if not records:
            raise ValueError(f"chunk index contains no records: {index_path}")
        first = records[0].path
        with h5py.File(first, "r") as handle:
            key = dataset if dataset in handle else next(iter(handle))
            array = handle[key]
            if array.ndim != 4:
                raise ValueError(f"chunked affinity arrays must be CZYX, got {array.shape}")
            self.channels = int(array.shape[0])
            self.dtype = np.dtype(array.dtype)
        self.records = tuple(records)
        self.dataset = dataset
        self.channel_axis = 0
        self.spatial_shape_zyx = tuple(
            max(record.stop_zyx[axis] for record in records) for axis in range(3)
        )

    def read(self, bbox: BoundingBox) -> np.ndarray:
        import h5py

        output = np.zeros((self.channels,) + bbox.shape, dtype=self.dtype)
        for record in self.records:
            start = tuple(max(a, b) for a, b in zip(bbox.start_zyx, record.start_zyx))
            stop = tuple(min(a, b) for a, b in zip(bbox.stop_zyx, record.stop_zyx))
            if any(high <= low for low, high in zip(start, stop)):
                continue
            source = tuple(
                slice(low - origin, high - origin)
                for low, high, origin in zip(start, stop, record.start_zyx)
            )
            target = tuple(
                slice(low - origin, high - origin)
                for low, high, origin in zip(start, stop, bbox.start_zyx)
            )
            with h5py.File(record.path, "r") as handle:
                key = self.dataset if self.dataset in handle else next(iter(handle))
                output[(slice(None),) + target] = handle[key][(slice(None),) + source]
        return output

    def close(self) -> None:
        return None


def open_volume(
    uri: str,
    *,
    dataset: str = "main",
    channel_axis: int | None = None,
) -> VolumeReader:
    if uri.startswith(("file://", "gs://", "s3://")):
        bare = uri.removeprefix("file://")
        if uri.startswith("file://") and bare.endswith(".h5.chunks"):
            return ChunkedH5Volume(Path(bare), dataset, channel_axis)
        if not uri.startswith("file://") or Path(bare).is_dir():
            return CloudVolumeReader(uri, channels=channel_axis is not None)
        uri = bare
    path = Path(uri)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return ArrayVolume(np.load(path, mmap_mode="r"), channel_axis=channel_axis)
    if suffix == ".npz":
        archive = np.load(path)
        key = dataset if dataset in archive else archive.files[0]
        return ArrayVolume(archive[key], channel_axis=channel_axis)
    if suffix in (".h5", ".hdf5", ".hdf"):
        return H5Volume(path, dataset, channel_axis)
    if suffix == ".zarr" or path.is_dir():
        import zarr

        root = zarr.open(str(path), mode="r")
        array = root[dataset] if hasattr(root, "keys") and dataset in root else root
        return ArrayVolume(array, channel_axis=channel_axis)
    raise ValueError(f"unsupported checkpoint volume URI {uri!r}")


def full_bbox(reader: VolumeReader) -> BoundingBox:
    return BoundingBox((0, 0, 0), reader.spatial_shape_zyx)


def _channel_count(reader: VolumeReader) -> int:
    if reader.channel_axis is None:
        return 1
    shape = getattr(reader, "array", None)
    if shape is not None:
        return int(shape.shape[reader.channel_axis])
    volume = getattr(reader, "volume", None)
    if volume is not None:
        return int(volume.shape[3])
    return int(getattr(reader, "channels", 1))


def iter_bounded_tiles(
    reader: VolumeReader,
    bbox: BoundingBox,
    max_read_bytes: int,
) -> Iterator[BoundingBox]:
    if max_read_bytes <= 0:
        raise ValueError("max_read_bytes must be positive")
    bytes_per_voxel = reader.dtype.itemsize * _channel_count(reader)
    max_voxels = max(1, max_read_bytes // max(bytes_per_voxel, 1))
    z_len, y_len, x_len = bbox.shape
    x_step = min(x_len, max_voxels)
    y_step = min(y_len, max(1, max_voxels // x_step))
    z_step = min(z_len, max(1, max_voxels // (x_step * y_step)))
    z0, y0, x0 = bbox.start_zyx
    z1, y1, x1 = bbox.stop_zyx
    for z in range(z0, z1, z_step):
        for y in range(y0, y1, y_step):
            for x in range(x0, x1, x_step):
                yield BoundingBox(
                    (z, y, x),
                    (min(z + z_step, z1), min(y + y_step, y1), min(x + x_step, x1)),
                )


def read_bbox_chunked(reader: VolumeReader, bbox: BoundingBox, max_read_bytes: int) -> np.ndarray:
    if any(stop > size for stop, size in zip(bbox.stop_zyx, reader.spatial_shape_zyx)):
        raise ValueError(f"bbox {bbox} exceeds volume shape {reader.spatial_shape_zyx}")
    if reader.channel_axis is None:
        output = np.empty(bbox.shape, dtype=reader.dtype)
    else:
        channels = _channel_count(reader)
        output = (
            np.empty((channels,) + bbox.shape, dtype=reader.dtype)
            if reader.channel_axis == 0
            else np.empty(bbox.shape + (channels,), dtype=reader.dtype)
        )
    base = np.asarray(bbox.start_zyx)
    for tile in iter_bounded_tiles(reader, bbox, max_read_bytes):
        local_start = np.asarray(tile.start_zyx) - base
        local_stop = np.asarray(tile.stop_zyx) - base
        spatial_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(local_start, local_stop))
        target = spatial_slices
        if reader.channel_axis == 0:
            target = (slice(None),) + target
        elif reader.channel_axis is not None:
            target = target + (slice(None),)
        block = reader.read(tile)
        if block.nbytes > max_read_bytes:
            raise RuntimeError(
                f"volume reader returned {block.nbytes} bytes, exceeding budget {max_read_bytes}"
            )
        output[target] = block
    return output


def hash_volume(reader: VolumeReader, bbox: BoundingBox, max_read_bytes: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": bbox.shape, "dtype": str(reader.dtype), "channel_axis": reader.channel_axis},
            sort_keys=True,
        ).encode()
    )
    for tile in iter_bounded_tiles(reader, bbox, max_read_bytes):
        block = np.ascontiguousarray(reader.read(tile))
        if block.nbytes > max_read_bytes:
            raise RuntimeError(
                f"volume reader returned {block.nbytes} bytes, exceeding budget {max_read_bytes}"
            )
        digest.update(block.view(np.uint8))
    return digest.hexdigest()


def volume_artifact_ref(
    role: str,
    uri: str,
    reader: VolumeReader,
    bbox: BoundingBox,
    max_read_bytes: int,
    dataset: str | None,
) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        uri=uri,
        sha256=hash_volume(reader, bbox, max_read_bytes),
        dataset=dataset,
    )


def write_segmentation_delta(
    path: str | Path,
    before: np.ndarray,
    after: np.ndarray,
    scope: BoundingBox,
    abstention_mask: np.ndarray,
) -> int:
    changed = before != after
    indices = np.argwhere(changed).astype(np.int32)
    values = after[changed]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        scope_start_zyx=np.asarray(scope.start_zyx, dtype=np.int64),
        scope_stop_zyx=np.asarray(scope.stop_zyx, dtype=np.int64),
        changed_indices_zyx=indices,
        changed_values=values,
        abstention_mask=np.asarray(abstention_mask, dtype=bool),
    )
    return int(changed.sum())


def apply_segmentation_delta(
    base_roi: np.ndarray, path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    output = np.array(base_roi, copy=True)
    with np.load(path) as data:
        indices = data["changed_indices_zyx"]
        if indices.size:
            output[tuple(indices.T)] = data["changed_values"]
        abstention = np.asarray(data["abstention_mask"], dtype=bool)
    return output, abstention


__all__ = [
    "ArrayVolume",
    "ChunkedH5Volume",
    "VolumeReader",
    "apply_segmentation_delta",
    "full_bbox",
    "hash_volume",
    "iter_bounded_tiles",
    "open_volume",
    "read_bbox_chunked",
    "volume_artifact_ref",
    "write_segmentation_delta",
]
