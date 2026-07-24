"""Prepare and run vendored ABISS chunked decoding on large volumes.

This module bridges the tutorial config in ``tutorials/waterz_decoding_large_abiss.yaml``
to the vendored ABISS shell pipeline by:
- converting saved affinity H5 (C, Z, Y, X) into local precomputed/CloudVolume
- initializing WS/SEG precomputed outputs
- writing the ABISS JSON param file expected by ``lib/abiss/scripts/init.sh``
- running the watershed / remap / mean-edge agglomeration stages with the
  missing environment variables wired in for local execution
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

import h5py
import numpy as np
import yaml  # type: ignore[import-untyped]

STAGES_ALL = (
    "watershed",
    "remap_watershed",
    "agglomerate_mean_edge",
    "remap_agglomeration",
)


def _open_cloudvolume(cloudpath: str, **kwargs: Any) -> Any:
    """Import CloudVolume only when an execution path actually needs it."""

    from cloudvolume import CloudVolume

    return CloudVolume(cloudpath, **kwargs)


def _create_cloudvolume_info(**kwargs: Any) -> dict[str, Any]:
    """Build CloudVolume metadata without importing the package during dry resolve."""

    from cloudvolume import CloudVolume

    return CloudVolume.create_new_info(**kwargs)


@dataclass(frozen=True)
class OutputLayerSpec:
    """Metadata needed to initialize fresh local ABISS output layers."""

    resolution_xyz: tuple[int, int, int]
    chunk_size_xyz: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedConfig:
    """Everything required to plan or execute an already-prepared ABISS run."""

    workdir: Path
    secrets_dir: Path
    param_path: Path
    abiss_home: Path
    param_payload: Mapping[str, Any]
    root_tag: str
    top_mip: int
    stage_overlap: str = "0"
    stage_meta: str = ""
    stages: tuple[str, ...] = STAGES_ALL
    runtime_secrets_dir: Path | None = None
    output_layer_spec: OutputLayerSpec | None = None
    write_param: bool = True
    extra_env: Mapping[str, str] | None = None

    @property
    def seg_cloudpath(self) -> str:
        try:
            return str(self.param_payload["SEG_PATH"])
        except KeyError as exc:
            raise ValueError("ABISS param payload is missing required SEG_PATH.") from exc

    @property
    def effective_secrets_dir(self) -> Path:
        """Directory ABISS sees as ``SECRETS`` and therefore reads ``param`` from."""

        return self.runtime_secrets_dir or self.secrets_dir


@dataclass(frozen=True)
class StagePlan:
    """One subprocess invocation in an ABISS-large run."""

    stage: str
    argv: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class RunResult:
    """Resolved execution plan and, after execution, the discovered output volume."""

    prepared: PreparedConfig
    stage_plans: tuple[StagePlan, ...]
    executed: bool
    segmentation: Any | None = None

    @property
    def seg_cloudpath(self) -> str:
        return self.prepared.seg_cloudpath


@dataclass
class LargeWorkflowConfig:
    """Legacy H5-to-precomputed preparation settings used by the tutorial CLI."""

    workdir: Path
    secrets_dir: Path
    param_path: Path
    abiss_home: Path
    root_tag: str
    top_mip: int
    stage_overlap: str
    stage_meta: str
    aff_cloudpath: str
    ws_cloudpath: str
    seg_cloudpath: str
    scratch_cloudpath: str
    chunkmap_cloudpath: str
    bbox_xyz: list[int]
    chunk_size_xyz: list[int]
    source_h5: Path
    source_dataset: str
    source_num_channels: int
    copy_block_shape_xyz: list[int]
    aff_chunk_size_xyz: list[int]
    seg_chunk_size_xyz: list[int]
    resolution_xyz: list[int]
    param_payload: Dict[str, Any]

    def execution_config(
        self,
        *,
        stages: Sequence[str] = STAGES_ALL,
        write_param: bool = True,
    ) -> PreparedConfig:
        return PreparedConfig(
            workdir=self.workdir,
            secrets_dir=self.secrets_dir,
            param_path=self.param_path,
            abiss_home=self.abiss_home,
            param_payload=dict(self.param_payload),
            root_tag=self.root_tag,
            top_mip=self.top_mip,
            stage_overlap=self.stage_overlap,
            stage_meta=self.stage_meta,
            stages=tuple(stages),
            write_param=write_param,
        )


def _normalize_cloudpath(value: str | Path) -> str:
    s = str(value)
    if "://" in s:
        return s
    # Path.as_uri() emits a well-formed file URI on every platform
    # (file:///home/x on POSIX, file:///C:/x on Windows); manual "file://" +
    # str(path) produced a malformed file://C:\x on Windows.
    return Path(s).resolve().as_uri()


def _is_local_cloudpath(cloudpath: str) -> bool:
    return cloudpath.startswith("file://") or "://" not in cloudpath


def _cloudpath_to_local_path(cloudpath: str) -> Path:
    if cloudpath.startswith("file://"):
        parsed = urlparse(cloudpath)
        # url2pathname is platform-aware: on Windows it turns "/C:/x" into
        # "C:\\x" (plain Path("/C:/x") yields the invalid "\\C:\\x"); on POSIX
        # it is an unquoting identity. Handles percent-decoding too.
        return Path(url2pathname(parsed.path))
    return Path(cloudpath)


def _maybe_int_list(values: Sequence[Any], *, expected_len: int, name: str) -> list[int]:
    if len(values) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {values}.")
    return [int(v) for v in values]


def _compute_top_mip(bbox_xyz: Sequence[int], chunk_size_xyz: Sequence[int]) -> int:
    size_xyz = [int(bbox_xyz[i + 3]) - int(bbox_xyz[i]) for i in range(3)]
    dims = [max(1, math.ceil(size_xyz[i] / int(chunk_size_xyz[i]))) for i in range(3)]
    mip = 0
    while dims != [1, 1, 1]:
        dims = [(d + 1) // 2 for d in dims]
        mip += 1
    return mip


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _discover_h5_dataset_and_channels(
    source_h5: Path, configured_dataset: str | None
) -> tuple[str, int]:
    with h5py.File(source_h5, "r") as f:
        if configured_dataset:
            ds = f[configured_dataset]
            if ds.ndim != 4:
                raise ValueError(
                    f"Expected source affinity H5 dataset in CZYX order, got {ds.shape}."
                )
            return configured_dataset, int(ds.shape[0])

        datasets: list[str] = []

        def _visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        f.visititems(_visit)
        if len(datasets) != 1:
            raise ValueError(
                f"source_dataset was not set and {source_h5} contains "
                f"{len(datasets)} datasets: {datasets}"
            )
        ds = f[datasets[0]]
        if ds.ndim != 4:
            raise ValueError(f"Expected source affinity H5 dataset in CZYX order, got {ds.shape}.")
        return datasets[0], int(ds.shape[0])


def _info_exists(cloudpath: str) -> bool:
    if not _is_local_cloudpath(cloudpath):
        return False
    local = _cloudpath_to_local_path(cloudpath)
    return (local / "info").exists()


def _create_precomputed_layer(
    *,
    cloudpath: str,
    volume_size_xyz: Sequence[int],
    voxel_offset_xyz: Sequence[int],
    resolution_xyz: Sequence[int],
    chunk_size_xyz: Sequence[int],
    num_channels: int,
    layer_type: str,
    data_type: str,
    encoding: str,
    compress: bool = False,
) -> None:
    if _info_exists(cloudpath):
        return

    info = _create_cloudvolume_info(
        num_channels=num_channels,
        layer_type=layer_type,
        data_type=data_type,
        encoding=encoding,
        resolution=list(resolution_xyz),
        voxel_offset=list(voxel_offset_xyz),
        volume_size=list(volume_size_xyz),
        chunk_size=list(chunk_size_xyz),
        max_mip=0,
    )
    vol = _open_cloudvolume(cloudpath, info=info, bounded=True, progress=False, compress=compress)
    vol.commit_info()
    vol.commit_provenance()


def _validate_existing_layer(
    *,
    cloudpath: str,
    volume_size_xyz: Sequence[int],
    voxel_offset_xyz: Sequence[int],
    expected_channels: int,
    expected_chunk_size_xyz: Sequence[int] | None = None,
    expected_resolution_xyz: Sequence[int] | None = None,
    expected_layer_type: str | None = None,
    expected_dtype: str | None = None,
) -> None:
    vol = _open_cloudvolume(cloudpath, progress=False)
    shape = tuple(int(v) for v in vol.shape[:3])
    if shape != tuple(int(v) for v in volume_size_xyz):
        raise ValueError(
            f"Existing precomputed layer {cloudpath} has shape {shape}, "
            f"expected {tuple(volume_size_xyz)}."
        )
    offset = tuple(int(v) for v in vol.voxel_offset[:3])
    if offset != tuple(int(v) for v in voxel_offset_xyz):
        raise ValueError(
            f"Existing precomputed layer {cloudpath} has voxel_offset {offset}, "
            f"expected {tuple(voxel_offset_xyz)}."
        )
    channels = int(getattr(vol, "num_channels", expected_channels))
    if channels != int(expected_channels):
        raise ValueError(
            f"Existing precomputed layer {cloudpath} has {channels} channels, "
            f"expected {expected_channels}."
        )
    if expected_resolution_xyz is not None:
        resolution = tuple(int(v) for v in vol.resolution[:3])
        expected_resolution = tuple(int(v) for v in expected_resolution_xyz)
        if resolution != expected_resolution:
            raise ValueError(
                f"Existing precomputed layer {cloudpath} has resolution {resolution}, "
                f"expected {expected_resolution}."
            )
    if expected_layer_type is not None and str(vol.layer_type) != expected_layer_type:
        raise ValueError(
            f"Existing precomputed layer {cloudpath} has layer_type {vol.layer_type!r}, "
            f"expected {expected_layer_type!r}."
        )
    if expected_dtype is not None and np.dtype(vol.dtype) != np.dtype(expected_dtype):
        raise ValueError(
            f"Existing precomputed layer {cloudpath} has dtype {vol.dtype}, "
            f"expected {np.dtype(expected_dtype)}."
        )
    if expected_chunk_size_xyz is not None:
        scales = vol.info.get("scales", [])
        chunk_sizes = scales[0].get("chunk_sizes", []) if scales else []
        if not chunk_sizes:
            raise ValueError(
                f"Existing precomputed layer {cloudpath} is missing chunk_sizes metadata."
            )
        chunk_size = tuple(int(v) for v in chunk_sizes[0])
        if chunk_size != tuple(int(v) for v in expected_chunk_size_xyz):
            raise ValueError(
                f"Existing precomputed layer {cloudpath} has chunk_size {chunk_size}, "
                f"expected {tuple(int(v) for v in expected_chunk_size_xyz)}."
            )


def _validate_abiss_upload_alignment(
    *,
    bbox_xyz: Sequence[int],
    voxel_offset_xyz: Sequence[int],
    logical_chunk_size_xyz: Sequence[int],
    storage_chunk_size_xyz: Sequence[int],
) -> None:
    """Fail fast if ABISS logical chunk uploads will require non-aligned writes.

    ABISS writes whole logical chunks back into the WS/SEG CloudVolume layers.
    CloudVolume rejects non-aligned writes by default, and enabling them would be
    unsafe here because ABISS uploads chunk outputs in parallel.
    """
    axis_names = "xyz"
    bad_axes: list[str] = []
    for axis, axis_name in enumerate(axis_names):
        start = int(bbox_xyz[axis])
        stop = int(bbox_xyz[axis + 3])
        logical = int(logical_chunk_size_xyz[axis])
        storage = int(storage_chunk_size_xyz[axis])
        offset = int(voxel_offset_xyz[axis])

        boundary = start + logical
        while boundary < stop:
            if (boundary - offset) % storage != 0:
                bad_axes.append(
                    f"{axis_name}: boundary {boundary} is not aligned to storage chunk {storage}"
                )
                break
            boundary += logical

    if bad_axes:
        raise ValueError(
            "ABISS logical chunk uploads would require non-aligned CloudVolume writes for "
            f"seg_chunk_size_xyz={list(int(v) for v in storage_chunk_size_xyz)} and "
            f"param.CHUNK_SIZE={list(int(v) for v in logical_chunk_size_xyz)}. "
            "This is unsafe because ABISS uploads chunk outputs in parallel. "
            f"Misaligned axes: {', '.join(bad_axes)}. "
            "Use seg_chunk_size_xyz values that align with every internal CHUNK_SIZE boundary "
            "(for the bundled tutorial, setting the Z chunk size to 80 fixes the issue)."
        )


def _prepare_segmentation_output_layers(
    *,
    ws_cloudpath: str,
    seg_cloudpath: str,
    bbox_xyz: Sequence[int],
    resolution_xyz: Sequence[int],
    chunk_size_xyz: Sequence[int],
) -> None:
    voxel_offset = [int(value) for value in bbox_xyz[:3]]
    volume_size = [int(bbox_xyz[axis + 3]) - int(bbox_xyz[axis]) for axis in range(3)]

    print(f"Preparing watershed output layer at {ws_cloudpath}")
    _create_precomputed_layer(
        cloudpath=ws_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        resolution_xyz=resolution_xyz,
        chunk_size_xyz=chunk_size_xyz,
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        compress=False,
    )
    _validate_existing_layer(
        cloudpath=ws_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        expected_channels=1,
        expected_chunk_size_xyz=chunk_size_xyz,
        expected_resolution_xyz=resolution_xyz,
        expected_layer_type="segmentation",
        expected_dtype="uint64",
    )

    print(f"Preparing final segmentation output layer at {seg_cloudpath}")
    _create_precomputed_layer(
        cloudpath=seg_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        resolution_xyz=resolution_xyz,
        chunk_size_xyz=chunk_size_xyz,
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        compress=False,
    )
    _validate_existing_layer(
        cloudpath=seg_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        expected_channels=1,
        expected_chunk_size_xyz=chunk_size_xyz,
        expected_resolution_xyz=resolution_xyz,
        expected_layer_type="segmentation",
        expected_dtype="uint64",
    )

    size_map_cloudpath = seg_cloudpath.rstrip("/") + "/size_map"
    print(f"Preparing ABISS size-map output layer at {size_map_cloudpath}")
    _create_precomputed_layer(
        cloudpath=size_map_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        resolution_xyz=resolution_xyz,
        chunk_size_xyz=chunk_size_xyz,
        num_channels=1,
        layer_type="image",
        data_type="uint8",
        encoding="raw",
        compress=False,
    )
    _validate_existing_layer(
        cloudpath=size_map_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        expected_channels=1,
        expected_chunk_size_xyz=chunk_size_xyz,
        expected_resolution_xyz=resolution_xyz,
        expected_layer_type="image",
        expected_dtype="uint8",
    )


def _copy_affinity_h5_to_precomputed(
    *,
    source_h5: Path,
    dataset: str,
    cloudpath: str,
    bbox_xyz: Sequence[int],
    block_shape_xyz: Sequence[int],
    volume_chunk_size_xyz: Sequence[int],
    resolution_xyz: Sequence[int],
) -> None:
    if _info_exists(cloudpath):
        return

    local = _cloudpath_to_local_path(cloudpath)
    _ensure_dir(local)

    with h5py.File(source_h5, "r") as f:
        ds = f[dataset]
        if ds.ndim != 4:
            raise ValueError(f"Expected source affinities in CZYX order, got shape {ds.shape}.")
        channels, zdim, ydim, xdim = (int(v) for v in ds.shape)
        x0, y0, z0, x1, y1, z1 = [int(v) for v in bbox_xyz]
        if not (0 <= x0 <= x1 <= xdim and 0 <= y0 <= y1 <= ydim and 0 <= z0 <= z1 <= zdim):
            raise ValueError(
                f"BBOX {bbox_xyz} is out of range for source affinity shape {ds.shape} (CZYX)."
            )

        _create_precomputed_layer(
            cloudpath=cloudpath,
            volume_size_xyz=[x1 - x0, y1 - y0, z1 - z0],
            voxel_offset_xyz=[x0, y0, z0],
            resolution_xyz=resolution_xyz,
            chunk_size_xyz=[max(1, int(v)) for v in volume_chunk_size_xyz],
            num_channels=channels,
            layer_type="image",
            data_type=np.dtype(ds.dtype).name,
            encoding="raw",
            compress=False,
        )

        vol = _open_cloudvolume(cloudpath, bounded=True, progress=False, compress=False)

        bx, by, bz = [max(1, int(v)) for v in block_shape_xyz]
        total_blocks_x = math.ceil((x1 - x0) / bx)
        total_blocks_y = math.ceil((y1 - y0) / by)
        total_blocks_z = math.ceil((z1 - z0) / bz)
        total_blocks = total_blocks_x * total_blocks_y * total_blocks_z
        done = 0

        for z in range(z0, z1, bz):
            zz = min(z + bz, z1)
            for y in range(y0, y1, by):
                yy = min(y + by, y1)
                for x in range(x0, x1, bx):
                    xx = min(x + bx, x1)
                    block = np.asarray(ds[:, z:zz, y:yy, x:xx])
                    block_xyzc = np.transpose(block, (3, 2, 1, 0))
                    vol[x:xx, y:yy, z:zz] = block_xyzc
                    done += 1
                    if done == 1 or done == total_blocks or done % 10 == 0:
                        print(
                            f"Affinity copy: {done}/{total_blocks} blocks "
                            f"({x}:{xx}, {y}:{yy}, {z}:{zz})"
                        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_config(config_path: Path) -> LargeWorkflowConfig:
    cfg = _load_yaml(config_path)
    ab = dict(cfg.get("abiss_large", {}))
    if not ab:
        raise ValueError(f"Config {config_path} has no abiss_large section.")

    abiss_home = Path(ab["abiss_home"]).resolve()
    workdir = Path(ab["workdir"]).resolve()
    secrets_dir = Path(ab["secrets_dir"]).resolve()
    param_path = Path(ab.get("param_path", secrets_dir / "param")).resolve()
    root_tag = str(ab.get("root_tag", "0_0_0_0"))
    source_h5 = Path(ab["source_affinity_h5"]).resolve()
    configured_source_dataset = ab.get("source_dataset")
    source_dataset, source_num_channels = _discover_h5_dataset_and_channels(
        source_h5,
        str(configured_source_dataset) if configured_source_dataset is not None else None,
    )
    copy_block_shape_xyz = _maybe_int_list(
        ab.get("copy_block_shape_xyz", [512, 512, 64]),
        expected_len=3,
        name="copy_block_shape_xyz",
    )
    aff_chunk_size_xyz = _maybe_int_list(
        ab.get("aff_chunk_size_xyz", copy_block_shape_xyz),
        expected_len=3,
        name="aff_chunk_size_xyz",
    )
    seg_chunk_size_xyz = _maybe_int_list(
        ab.get("seg_chunk_size_xyz", copy_block_shape_xyz),
        expected_len=3,
        name="seg_chunk_size_xyz",
    )
    resolution_xyz = _maybe_int_list(
        ab.get("resolution_xyz", [1, 1, 1]),
        expected_len=3,
        name="resolution_xyz",
    )

    param = dict(ab.get("param", {}))

    with h5py.File(source_h5, "r") as f:
        ds = f[source_dataset]
        _, zdim, ydim, xdim = (int(v) for v in ds.shape)

    bbox_xyz = param.get("BBOX") or [0, 0, 0, xdim, ydim, zdim]
    bbox_xyz = _maybe_int_list(bbox_xyz, expected_len=6, name="param.BBOX")
    chunk_size_xyz = _maybe_int_list(
        param.get("CHUNK_SIZE", [xdim, ydim, min(zdim, 80)]),
        expected_len=3,
        name="param.CHUNK_SIZE",
    )

    top_mip = int(ab.get("top_mip", _compute_top_mip(bbox_xyz, chunk_size_xyz)))
    computed_top_mip = _compute_top_mip(bbox_xyz, chunk_size_xyz)
    if top_mip != computed_top_mip:
        raise ValueError(
            f"Configured top_mip={top_mip} does not match computed top_mip={computed_top_mip} "
            f"for BBOX={bbox_xyz} and CHUNK_SIZE={chunk_size_xyz}."
        )

    if root_tag == "0_0_0_0" and top_mip > 0:
        root_tag = f"{top_mip}_0_0_0"

    outputs_root = workdir.parent
    aff_cloudpath = _normalize_cloudpath(
        param.get("AFF_PATH", outputs_root / "precomputed" / "affinity")
    )
    ws_cloudpath = _normalize_cloudpath(param.get("WS_PATH", outputs_root / "precomputed" / "ws"))
    seg_cloudpath = _normalize_cloudpath(
        param.get("SEG_PATH", outputs_root / "precomputed" / "seg")
    )
    scratch_cloudpath = _normalize_cloudpath(param.get("SCRATCH_PATH", outputs_root / "scratch"))
    chunkmap_cloudpath = _normalize_cloudpath(
        param.get("CHUNKMAP_OUTPUT", outputs_root / "chunkmap")
    )

    stage_overlap = str(ab.get("overlap_mode", 0))
    stage_meta = " ".join(str(v) for v in ab.get("meta_dirs", []))

    copy_helper = Path(__file__).resolve().parents[2] / "scripts" / "copy_uri.py"
    upload_cmd = param.get("UPLOAD_CMD", f"{sys.executable} {copy_helper}")
    download_cmd = param.get("DOWNLOAD_CMD", f"{sys.executable} {copy_helper}")

    payload = dict(param)
    payload.update(
        {
            "NAME": str(param.get("NAME", config_path.stem)),
            "AFF_PATH": aff_cloudpath,
            "WS_PATH": ws_cloudpath,
            "SEG_PATH": seg_cloudpath,
            "SCRATCH_PATH": scratch_cloudpath,
            "CHUNKMAP_INPUT": chunkmap_cloudpath,
            "CHUNKMAP_OUTPUT": chunkmap_cloudpath,
            "UPLOAD_CMD": upload_cmd,
            "DOWNLOAD_CMD": download_cmd,
            "AFF_RESOLUTION": int(param.get("AFF_RESOLUTION", 0)),
            "AFF_CHANNELS": int(param.get("AFF_CHANNELS", source_num_channels)),
            "BBOX": bbox_xyz,
            "CHUNK_SIZE": chunk_size_xyz,
        }
    )
    payload.setdefault("WS_HIGH_THRESHOLD", 0.9)
    payload.setdefault("WS_LOW_THRESHOLD", 0.1)
    payload.setdefault("WS_SIZE_THRESHOLD", 400)
    payload.setdefault("WS_DUST_THRESHOLD", payload["WS_SIZE_THRESHOLD"])
    payload.setdefault("AGG_THRESHOLD", 0.2)
    payload.setdefault("PARANOID", False)
    payload.setdefault("CHUNKED_AGG_OUTPUT", False)

    return LargeWorkflowConfig(
        workdir=workdir,
        secrets_dir=secrets_dir,
        param_path=param_path,
        abiss_home=abiss_home,
        root_tag=root_tag,
        top_mip=top_mip,
        stage_overlap=stage_overlap,
        stage_meta=stage_meta,
        aff_cloudpath=aff_cloudpath,
        ws_cloudpath=ws_cloudpath,
        seg_cloudpath=seg_cloudpath,
        scratch_cloudpath=scratch_cloudpath,
        chunkmap_cloudpath=chunkmap_cloudpath,
        bbox_xyz=bbox_xyz,
        chunk_size_xyz=chunk_size_xyz,
        source_h5=source_h5,
        source_dataset=source_dataset,
        source_num_channels=source_num_channels,
        copy_block_shape_xyz=copy_block_shape_xyz,
        aff_chunk_size_xyz=aff_chunk_size_xyz,
        seg_chunk_size_xyz=seg_chunk_size_xyz,
        resolution_xyz=resolution_xyz,
        param_payload=payload,
    )


_prepare_config = prepare_config


def prepare(cfg: LargeWorkflowConfig, *, write_param: bool = True) -> None:
    _ensure_dir(cfg.workdir)
    _ensure_dir(cfg.secrets_dir)
    _ensure_parent(cfg.param_path)
    if _is_local_cloudpath(cfg.scratch_cloudpath):
        _ensure_dir(_cloudpath_to_local_path(cfg.scratch_cloudpath))
    if _is_local_cloudpath(cfg.chunkmap_cloudpath):
        _ensure_dir(_cloudpath_to_local_path(cfg.chunkmap_cloudpath))

    bbox = cfg.bbox_xyz
    voxel_offset = bbox[:3]
    volume_size = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]

    _validate_abiss_upload_alignment(
        bbox_xyz=bbox,
        voxel_offset_xyz=voxel_offset,
        logical_chunk_size_xyz=cfg.chunk_size_xyz,
        storage_chunk_size_xyz=cfg.seg_chunk_size_xyz,
    )

    print(f"Preparing affinity precomputed at {cfg.aff_cloudpath}")
    _copy_affinity_h5_to_precomputed(
        source_h5=cfg.source_h5,
        dataset=cfg.source_dataset,
        cloudpath=cfg.aff_cloudpath,
        bbox_xyz=bbox,
        block_shape_xyz=cfg.copy_block_shape_xyz,
        volume_chunk_size_xyz=cfg.aff_chunk_size_xyz,
        resolution_xyz=cfg.resolution_xyz,
    )
    _validate_existing_layer(
        cloudpath=cfg.aff_cloudpath,
        volume_size_xyz=volume_size,
        voxel_offset_xyz=voxel_offset,
        expected_channels=cfg.source_num_channels,
        expected_chunk_size_xyz=cfg.aff_chunk_size_xyz,
        expected_resolution_xyz=cfg.resolution_xyz,
        expected_layer_type="image",
    )

    _prepare_segmentation_output_layers(
        ws_cloudpath=cfg.ws_cloudpath,
        seg_cloudpath=cfg.seg_cloudpath,
        bbox_xyz=bbox,
        resolution_xyz=cfg.resolution_xyz,
        chunk_size_xyz=cfg.seg_chunk_size_xyz,
    )

    if write_param:
        _write_param(cfg.param_path, cfg.param_payload)


def _stage_command(cfg: PreparedConfig, stage: str) -> tuple[list[str], dict[str, str]]:
    scripts_dir = cfg.abiss_home / "scripts"
    env = os.environ.copy()
    if cfg.extra_env:
        env.update({str(key): str(value) for key, value in cfg.extra_env.items()})
    env.update(
        {
            "WORKER_HOME": str(cfg.abiss_home),
            "SECRETS": str(cfg.effective_secrets_dir),
            "OVERLAP": str(cfg.stage_overlap),
            "META": str(cfg.stage_meta),
        }
    )

    python_bin_dir = str(Path(sys.executable).resolve().parent)
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        python_bin_dir if not existing_path else python_bin_dir + os.pathsep + existing_path
    )

    if stage == "watershed":
        env["STAGE"] = "ws"
        cmd = ["bash", str(scripts_dir / "run_batch.sh"), "ws", str(cfg.top_mip), cfg.root_tag]
    elif stage == "remap_watershed":
        env["STAGE"] = "ws"
        cmd = ["bash", str(scripts_dir / "remap_batch.sh"), "ws", str(cfg.top_mip), cfg.root_tag]
    elif stage == "agglomerate_mean_edge":
        env["STAGE"] = "agg"
        cmd = ["bash", str(scripts_dir / "run_batch.sh"), "me", str(cfg.top_mip), cfg.root_tag]
    elif stage == "remap_agglomeration":
        env["STAGE"] = "agg"
        cmd = ["bash", str(scripts_dir / "remap_batch.sh"), "agg", str(cfg.top_mip), cfg.root_tag]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return cmd, env


def _stage_plan(cfg: PreparedConfig, stage: str) -> StagePlan:
    cmd, env = _stage_command(cfg, stage)
    env.setdefault("AIRFLOW_TMP_DIR", str(cfg.workdir / ".airflow"))
    return StagePlan(stage=stage, argv=tuple(cmd), env=env)


def _write_param(param_path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(param_path)
    with param_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote ABISS param JSON: {param_path}")


def _prepare_execution_outputs(cfg: PreparedConfig) -> None:
    spec = cfg.output_layer_spec
    if spec is None:
        return

    try:
        bbox = _maybe_int_list(cfg.param_payload["BBOX"], expected_len=6, name="BBOX")
        logical_chunk_size = _maybe_int_list(
            cfg.param_payload["CHUNK_SIZE"], expected_len=3, name="CHUNK_SIZE"
        )
        ws_cloudpath = str(cfg.param_payload["WS_PATH"])
        seg_cloudpath = str(cfg.param_payload["SEG_PATH"])
    except KeyError as exc:
        raise ValueError(f"ABISS output initialization is missing {exc.args[0]}.") from exc

    _validate_abiss_upload_alignment(
        bbox_xyz=bbox,
        voxel_offset_xyz=bbox[:3],
        logical_chunk_size_xyz=logical_chunk_size,
        storage_chunk_size_xyz=spec.chunk_size_xyz,
    )
    _prepare_segmentation_output_layers(
        ws_cloudpath=ws_cloudpath,
        seg_cloudpath=seg_cloudpath,
        bbox_xyz=bbox,
        resolution_xyz=spec.resolution_xyz,
        chunk_size_xyz=spec.chunk_size_xyz,
    )

    for key in ("SCRATCH_PATH", "CHUNKMAP_OUTPUT"):
        cloudpath = cfg.param_payload.get(key)
        if cloudpath is not None and _is_local_cloudpath(str(cloudpath)):
            _ensure_dir(_cloudpath_to_local_path(str(cloudpath)))


def _discover_segmentation(cloudpath: str) -> Any:
    """Open the result layer without materializing the potentially huge volume."""

    return _open_cloudvolume(cloudpath, progress=False)


def _prepare_runtime_secrets_view(cfg: PreparedConfig) -> None:
    """Expose source credentials beside the per-run ABISS param without copying them."""

    runtime_dir = cfg.runtime_secrets_dir
    if runtime_dir is None:
        return

    _ensure_dir(runtime_dir)
    if not cfg.secrets_dir.is_dir():
        return

    for source in cfg.secrets_dir.iterdir():
        # ABISS derives config.sh from the current param. Reusing either source file
        # would risk pointing the fresh replay at a historical namespace.
        if source.name in {"config.sh", "param"}:
            continue
        destination = runtime_dir / source.name
        source_resolved = source.resolve()
        if destination.is_symlink():
            if destination.resolve() == source_resolved:
                continue
            raise RuntimeError(
                f"Runtime credential link {destination} points somewhere other than "
                f"the configured source {source_resolved}."
            )
        if destination.exists():
            raise RuntimeError(
                f"Runtime credential path already exists and is not a managed link: {destination}"
            )
        destination.symlink_to(source_resolved, target_is_directory=source.is_dir())


def _execute_stage(cfg: PreparedConfig, plan: StagePlan) -> None:
    airflow_tmp_dir = cfg.workdir / ".airflow"
    airflow_tmp_dir.mkdir(parents=True, exist_ok=True)
    for lock_file in airflow_tmp_dir.glob(".cpulock_*"):
        lock_file.unlink(missing_ok=True)
    print(f"Running stage {plan.stage}: {' '.join(plan.argv)}")
    subprocess.run(
        list(plan.argv),
        cwd=str(cfg.workdir),
        env=dict(plan.env),
        check=True,
    )


def run_abiss_large(prepared: PreparedConfig, *, execute: bool = False) -> RunResult:
    """Resolve or execute the canonical four-stage ABISS-large invocation.

    Dry resolution is the default and performs no filesystem or subprocess I/O.
    Execution writes the canonical sorted param JSON, runs the requested stages in
    order, and opens the final segmentation layer for result discovery.
    """

    if not execute:
        plans = tuple(_stage_plan(prepared, stage) for stage in prepared.stages)
        return RunResult(prepared=prepared, stage_plans=plans, executed=False)

    payload = dict(prepared.param_payload)
    copy_helper = Path(__file__).resolve().parents[2] / "scripts" / "copy_uri.py"
    copy_command = f"{sys.executable} {copy_helper}"
    payload.setdefault("UPLOAD_CMD", copy_command)
    payload.setdefault("DOWNLOAD_CMD", copy_command)
    prepared = replace(prepared, param_payload=payload)
    plans = tuple(_stage_plan(prepared, stage) for stage in prepared.stages)

    _ensure_dir(prepared.workdir)
    _prepare_execution_outputs(prepared)
    _prepare_runtime_secrets_view(prepared)
    if prepared.write_param:
        _write_param(prepared.param_path, prepared.param_payload)
        if prepared.runtime_secrets_dir is not None:
            runtime_param_path = prepared.runtime_secrets_dir / "param"
            _write_param(runtime_param_path, prepared.param_payload)
    for plan in plans:
        _execute_stage(prepared, plan)

    segmentation = _discover_segmentation(prepared.seg_cloudpath)
    return RunResult(
        prepared=prepared,
        stage_plans=plans,
        executed=True,
        segmentation=segmentation,
    )


def run_stage(cfg: PreparedConfig, stage: str) -> None:
    _execute_stage(cfg, _stage_plan(cfg, stage))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config file with abiss_large section")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare precomputed layers and param JSON, then exit",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume prepare step was already run",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES_ALL,
        default=STAGES_ALL,
        help="ABISS stages to run after preparation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = prepare_config(Path(args.config).resolve())

    print(f"ABISS home:    {cfg.abiss_home}")
    print(f"Workdir:       {cfg.workdir}")
    print(f"Root tag:      {cfg.root_tag}")
    print(f"Top mip:       {cfg.top_mip}")
    print(f"BBOX xyz:      {cfg.bbox_xyz}")
    print(f"Chunk size xyz:{cfg.chunk_size_xyz}")
    print(f"Affinity src:  {cfg.source_h5}:{cfg.source_dataset}")
    print(f"Affinity dst:  {cfg.aff_cloudpath}")
    print(f"WS dst:        {cfg.ws_cloudpath}")
    print(f"SEG dst:       {cfg.seg_cloudpath}")

    if not args.skip_prepare:
        prepare(cfg, write_param=args.prepare_only)
    if args.prepare_only:
        return 0

    result = run_abiss_large(
        cfg.execution_config(
            stages=args.stages,
            write_param=not args.skip_prepare,
        ),
        execute=True,
    )

    print("ABISS large decode completed.")
    print(f"Final segmentation layer: {result.seg_cloudpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
