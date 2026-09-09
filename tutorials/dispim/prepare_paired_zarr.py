#!/usr/bin/env python3
"""Stack the two simulated diSPIM views into channel-first training arrays.

The source simulator stores ``image_blur_A`` and ``image_blur_B`` as sibling
ZYX arrays.  PyTC interprets a YAML list of image paths as separate samples,
so dual-view training needs a single CZYX array.  This utility adds non-
destructive ``image_train``, ``image_val``, ``label_train``, ``label_val``,
and test ``image`` arrays to the existing matched Zarr groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import zarr


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2].parent / "dispim_sim" / "data"


def _create_array(group, name: str, *, shape, chunks, dtype, overwrite: bool):
    if name in group:
        existing = group[name]
        if not overwrite:
            if tuple(existing.shape) != tuple(shape):
                raise ValueError(
                    f"Existing {name} has shape {existing.shape}, expected {tuple(shape)}"
                )
            print(f"keep {name}: shape={existing.shape}")
            return existing, False
        del group[name]
    array = group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return array, True


def _copy_views(destination, sources, z_start: int, z_stop: int) -> None:
    z_chunk = int(destination.chunks[1])
    for channel, source in enumerate(sources):
        for out_start in range(0, z_stop - z_start, z_chunk):
            out_stop = min(out_start + z_chunk, z_stop - z_start)
            destination[channel, out_start:out_stop] = source[
                z_start + out_start : z_start + out_stop
            ]


def _copy_label(destination, label: np.ndarray, z_start: int, z_stop: int) -> None:
    z_chunk = int(destination.chunks[0])
    for out_start in range(0, z_stop - z_start, z_chunk):
        out_stop = min(out_start + z_chunk, z_stop - z_start)
        destination[out_start:out_stop] = label[z_start + out_start : z_start + out_stop]


def prepare(data_dir: Path, split_z: int, overwrite: bool) -> None:
    train_path = data_dir / "snemi_train_dispim_sim_matched.zarr"
    test_path = data_dir / "snemi_test_dispim_sim_matched.zarr"
    label_path = data_dir / "seg" / "train-labels.tif"

    train = zarr.open_group(train_path, mode="a")
    test = zarr.open_group(test_path, mode="a")
    train_views = [train["image_blur_A"], train["image_blur_B"]]
    test_views = [test["image_blur_A"], test["image_blur_B"]]
    if train_views[0].shape != train_views[1].shape:
        raise ValueError("Training views do not have identical ZYX shapes")
    if test_views[0].shape != test_views[1].shape:
        raise ValueError("Test views do not have identical ZYX shapes")

    label = tifffile.imread(label_path)
    if tuple(label.shape) != tuple(train_views[0].shape):
        raise ValueError(
            f"Image/label mismatch: {train_views[0].shape} versus {label.shape}"
        )
    depth, height, width = (int(v) for v in label.shape)
    if not 1 <= split_z < depth:
        raise ValueError(f"split_z must lie in [1,{depth - 1}], got {split_z}")

    image_chunks = (1, *tuple(int(v) for v in train_views[0].chunks))
    label_chunks = tuple(int(v) for v in train_views[0].chunks)
    train_specs = (
        ("image_train", (2, split_z, height, width), 0, split_z, True),
        ("image_val", (2, depth - split_z, height, width), split_z, depth, True),
        ("label_train", (split_z, height, width), 0, split_z, False),
        ("label_val", (depth - split_z, height, width), split_z, depth, False),
    )
    for name, shape, z_start, z_stop, is_image in train_specs:
        array, created = _create_array(
            train,
            name,
            shape=shape,
            chunks=image_chunks if is_image else label_chunks,
            dtype="float32" if is_image else label.dtype,
            overwrite=overwrite,
        )
        if created:
            if is_image:
                _copy_views(array, train_views, z_start, z_stop)
            else:
                _copy_label(array, label, z_start, z_stop)
            print(f"wrote {train_path}/{name}: shape={array.shape}")

    test_shape = (2, *tuple(int(v) for v in test_views[0].shape))
    test_chunks = (1, *tuple(int(v) for v in test_views[0].chunks))
    test_image, created = _create_array(
        test,
        "image",
        shape=test_shape,
        chunks=test_chunks,
        dtype="float32",
        overwrite=overwrite,
    )
    if created:
        _copy_views(test_image, test_views, 0, test_shape[1])
        print(f"wrote {test_path}/image: shape={test_image.shape}")

    metadata = {
        "axes": "CZYX",
        "channel_order": ["image_blur_A", "image_blur_B"],
        "view_angles_deg_simulator_xy": [45.0, 135.0],
        "relative_view_angle_deg": 90.0,
    }
    for name in ("image_train", "image_val"):
        train[name].attrs.update(metadata)
    test["image"].attrs.update(metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--split-z", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.split_z, args.overwrite)


if __name__ == "__main__":
    main()
