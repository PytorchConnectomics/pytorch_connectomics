#!/usr/bin/env python3
"""Create no-copy train/test views of the MICrONS Pinky neuron labels.

The Zenodo files store image, segmentation, and validity mask datasets in one
HDF5 container.  This script creates small HDF5 virtual-dataset (VDS) files so
PyTC can address each array independently without copying the source voxels.
Each view is cropped to the source file's valid-mask bounding box.  The split is
by source volume, with approximately 80/20 of valid annotated voxels assigned
to train/test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

DEFAULT_ROOT = Path("/projects/weilab/dataset/microns/train/pinky")
RESOLUTION_NM = (40.0, 4.0, 4.0)

# 804,658,862 / 994,661,430 valid voxels (80.9%).  All training views can
# supply the [16,160,160] model patch plus [10,10,10] trailing target context.
TRAIN_VOLUMES = (
    "pinky_stitched_vol19-vol34_realigned.h5",
    "pinky_stitched_vol40-vol41.h5",
    "pinky_vol201.h5",
    "pinky_vol401.h5",
    "pinky_vol502.h5",
)

# 190,002,568 / 994,661,430 valid voxels (19.1%).  The shallow volumes are
# useful held-out cases but cannot supply the training target context in Z.
TEST_VOLUMES = (
    "pinky_vol101.h5",
    "pinky_vol102.h5",
    "pinky_vol103.h5",
    "pinky_vol104.h5",
    "pinky_vol501.h5",
    "pinky_vol503.h5",
)

EXCLUDED_VOLUMES = {
    "pinky_stitched_vol19-vol34.h5": (
        "Superseded by the overlapping realigned 19-34 volume; including both "
        "would duplicate tissue and permit train/test leakage."
    )
}

SOURCE_DATASETS = {
    "image": "volumes/image",
    "label": "volumes/segmentation",
    "mask": "volumes/mask",
}

Shape3 = tuple[int, int, int]


def _shape3(values: Iterable[int], *, name: str) -> Shape3:
    shape = tuple(int(value) for value in values)
    if len(shape) != 3:
        raise ValueError(f"{name} must be 3D, found shape {shape}")
    return shape[0], shape[1], shape[2]


@dataclass(frozen=True)
class VolumeInfo:
    source: Path
    image_shape: Shape3
    label_shape: Shape3
    roi_start: Shape3
    roi_stop: Shape3
    valid_voxels: int
    mask_density: float

    @property
    def roi_shape(self) -> Shape3:
        return _shape3(
            (stop - start for start, stop in zip(self.roi_start, self.roi_stop)),
            name="valid-mask ROI",
        )


def _mask_extent(mask: h5py.Dataset) -> tuple[Shape3, Shape3, int]:
    """Return the nonzero mask bounding box and voxel count without loading it whole."""
    lo = np.asarray(mask.shape, dtype=np.int64)
    hi = np.zeros(mask.ndim, dtype=np.int64)
    count = 0

    for z in range(mask.shape[0]):
        section = np.asarray(mask[z]) > 0
        section_count = int(np.count_nonzero(section))
        count += section_count
        if section_count == 0:
            continue
        ys = np.flatnonzero(np.any(section, axis=1))
        xs = np.flatnonzero(np.any(section, axis=0))
        lo = np.minimum(lo, (z, int(ys[0]), int(xs[0])))
        hi = np.maximum(hi, (z + 1, int(ys[-1]) + 1, int(xs[-1]) + 1))

    if count == 0:
        raise ValueError(f"Validity mask {mask.name} is empty")
    return _shape3(lo, name="mask start"), _shape3(hi, name="mask stop"), count


def inspect_volume(source: Path) -> VolumeInfo:
    with h5py.File(source, "r") as handle:
        missing = [path for path in SOURCE_DATASETS.values() if path not in handle]
        if missing:
            raise KeyError(f"{source} is missing required datasets: {missing}")

        image = handle[SOURCE_DATASETS["image"]]
        label = handle[SOURCE_DATASETS["label"]]
        mask = handle[SOURCE_DATASETS["mask"]]
        image_shape = _shape3(image.shape, name=f"{source} image")
        label_shape = _shape3(label.shape, name=f"{source} label")
        if tuple(mask.shape) != image_shape:
            raise ValueError(f"{source}: image/mask shape mismatch {image_shape} vs {mask.shape}")

        resolution = tuple(float(v) for v in image.attrs.get("resolution", ()))
        if resolution != RESOLUTION_NM:
            raise ValueError(
                f"{source}: expected resolution {RESOLUTION_NM} nm, found {resolution}"
            )

        roi_start, roi_stop, valid_voxels = _mask_extent(mask)
        roi_shape = tuple(stop - start for start, stop in zip(roi_start, roi_stop))
        if label_shape not in (image_shape, roi_shape):
            raise ValueError(
                f"{source}: label shape {label_shape} matches neither image {image_shape} "
                f"nor valid-mask ROI {roi_shape}"
            )

        roi_voxels = int(np.prod(roi_shape))
        return VolumeInfo(
            source=source,
            image_shape=image_shape,
            label_shape=label_shape,
            roi_start=roi_start,
            roi_stop=roi_stop,
            valid_voxels=valid_voxels,
            mask_density=valid_voxels / roi_voxels,
        )


def _source_selection(info: VolumeInfo, kind: str) -> tuple[slice, slice, slice]:
    if kind == "label" and info.label_shape == info.roi_shape:
        return (
            slice(0, info.label_shape[0]),
            slice(0, info.label_shape[1]),
            slice(0, info.label_shape[2]),
        )
    return (
        slice(info.roi_start[0], info.roi_stop[0]),
        slice(info.roi_start[1], info.roi_stop[1]),
        slice(info.roi_start[2], info.roi_stop[2]),
    )


def _validate_existing_view(path: Path, info: VolumeInfo, kind: str) -> None:
    with h5py.File(path, "r") as handle:
        if "main" not in handle:
            raise ValueError(f"Existing view {path} has no 'main' dataset")
        dataset = handle["main"]
        if tuple(dataset.shape) != info.roi_shape:
            raise ValueError(
                f"Existing view {path} has shape {dataset.shape}, expected {info.roi_shape}"
            )
        if Path(str(dataset.attrs.get("source_file", ""))) != info.source:
            raise ValueError(f"Existing view {path} points at a different source file")
        if str(dataset.attrs.get("source_kind", "")) != kind:
            raise ValueError(f"Existing view {path} has the wrong source kind")


def create_view(path: Path, info: VolumeInfo, kind: str) -> None:
    """Create one cropped VDS, or verify an identical existing view."""
    if path.exists():
        _validate_existing_view(path, info, kind)
        return

    source_key = SOURCE_DATASETS[kind]
    with h5py.File(info.source, "r") as source_handle:
        source_dataset = source_handle[source_key]
        layout = h5py.VirtualLayout(shape=info.roi_shape, dtype=source_dataset.dtype)
        virtual_source = h5py.VirtualSource(
            str(info.source), source_key, shape=source_dataset.shape
        )
        layout[...] = virtual_source[_source_selection(info, kind)]

        with h5py.File(path, "x", libver="latest") as output_handle:
            output = output_handle.create_virtual_dataset("main", layout, fillvalue=0)
            output.attrs["resolution"] = np.asarray(RESOLUTION_NM, dtype=np.float64)
            output.attrs["source_file"] = str(info.source)
            output.attrs["source_dataset"] = source_key
            output.attrs["source_kind"] = kind
            output.attrs["source_roi_start"] = np.asarray(info.roi_start, dtype=np.int64)
            output.attrs["source_roi_stop"] = np.asarray(info.roi_stop, dtype=np.int64)


def _volume_manifest(info: VolumeInfo) -> dict[str, object]:
    return {
        "source": info.source.name,
        "source_shape": list(info.image_shape),
        "label_shape": list(info.label_shape),
        "roi_start": list(info.roi_start),
        "roi_stop": list(info.roi_stop),
        "view_shape": list(info.roi_shape),
        "valid_voxels": info.valid_voxels,
        "mask_density_in_view": info.mask_density,
    }


def _write_manifest(root: Path, split_info: dict[str, list[VolumeInfo]]) -> None:
    train_valid = sum(info.valid_voxels for info in split_info["train"])
    test_valid = sum(info.valid_voxels for info in split_info["test"])
    total_valid = train_valid + test_valid
    manifest = {
        "dataset": "MICrONS Pinky",
        "record": "https://zenodo.org/records/5760218",
        "doi": "10.5281/zenodo.5760218",
        "license": "CC-BY-4.0",
        "task": "3D neuron instance segmentation",
        "resolution_nm_zyx": list(RESOLUTION_NM),
        "source_datasets": SOURCE_DATASETS,
        "split_basis": "source volumes, balanced by nonzero validity-mask voxels",
        "train_valid_fraction": train_valid / total_valid,
        "test_valid_fraction": test_valid / total_valid,
        "excluded": EXCLUDED_VOLUMES,
        "train": [_volume_manifest(info) for info in split_info["train"]],
        "test": [_volume_manifest(info) for info in split_info["test"]],
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path = root / "split" / "manifest.json"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing manifest differs from the requested split: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def prepare(root: Path) -> dict[str, list[VolumeInfo]]:
    split_names: dict[str, Iterable[str]] = {
        "train": TRAIN_VOLUMES,
        "test": TEST_VOLUMES,
    }
    split_info: dict[str, list[VolumeInfo]] = {"train": [], "test": []}

    for split, names in split_names.items():
        output_dir = root / "split" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            info = inspect_volume(source)
            split_info[split].append(info)
            for kind in SOURCE_DATASETS:
                create_view(output_dir / f"{source.stem}_{kind}.h5", info, kind)

    _write_manifest(root, split_info)
    return split_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    split_info = prepare(args.root.resolve())
    train_valid = sum(info.valid_voxels for info in split_info["train"])
    test_valid = sum(info.valid_voxels for info in split_info["test"])
    total = train_valid + test_valid
    print(f"Prepared {len(split_info['train'])} train volumes: {train_valid:,} valid voxels")
    print(f"Prepared {len(split_info['test'])} test volumes:  {test_valid:,} valid voxels")
    print(f"Split: {train_valid / total:.1%} train / {test_valid / total:.1%} test")
    print(f"Manifest: {args.root.resolve() / 'split' / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
