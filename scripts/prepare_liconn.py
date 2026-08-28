#!/usr/bin/env python3
"""Stream the final-proofread LICONN precomputed volume into BANIS Zarr splits.

The source Neuroglancer arrays are XYZ, while PyTC's lazy Zarr datasets and the
tube decoder consume the arrays in their stored order. This script writes ZYX
arrays so the first three BANIS affinity channels are Z, Y, X at inference.

The defaults reproduce Mansour's proofread crop without loading the full uint64
segmentation into memory:

    full ZYX: [795, 4870, 3825]
    crop ZYX: [140:555, 240:4530, 240:3585] -> [415, 4290, 3345]
    split:    train Z [0:270], validation Z [270:415]

The 270/145 split deliberately leaves at least 138 validation slices, which is
the BANIS+ 128-voxel patch plus its 10-voxel trailing target context.

Image and segmentation may live at different scales. ``--segmentation-resolution-xyz``
selects a coarser segmentation scale, which is then nearest-upsampled by the exact
integer factor implied by the two resolutions. This exports the 9x9x12 nm image with
the 18x18x24 nm proofread segmentation as GT (the proofread segmentation has no 9 nm
scale), at the cost of a label whose boundaries stay quantized to the coarse grid.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

DEFAULT_IMAGE = "/projects/weilab/dataset/liconn/ffn/ExPID82_1/image_230130b"
DEFAULT_SEGMENTATION = (
    "/projects/weilab/dataset/liconn/ffn/ExPID82_1/" "segmentation/231030_agg_240123"
)


def _read_info(path: Path) -> dict:
    with (path / "info").open() as handle:
        return json.load(handle)


def _exact_scale_index(info: dict, resolution_xyz: Sequence[float]) -> int:
    target = np.asarray(resolution_xyz, dtype=np.float64)
    matches = [
        index
        for index, scale in enumerate(info["scales"])
        if np.allclose(
            np.asarray(scale["resolution"], dtype=np.float64),
            target,
            rtol=0.0,
            atol=1.0e-6,
        )
    ]
    if len(matches) != 1:
        available = [scale["resolution"] for scale in info["scales"]]
        raise ValueError(
            f"Expected exactly one scale at XYZ resolution {list(target)}, "
            f"found {len(matches)}; available={available}"
        )
    return matches[0]


def _upsample_factor_xyz(
    image_resolution_xyz: Sequence[float],
    segmentation_resolution_xyz: Sequence[float],
    image_size_xyz: Sequence[int],
    segmentation_size_xyz: Sequence[int],
) -> tuple[int, ...]:
    """Per-axis integer factor mapping the segmentation grid onto the image grid.

    Requires the resolution ratio to be a whole number on every axis and the source
    shapes to agree with it under Neuroglancer's ceil-halving convention, so a fine
    voxel never reads outside the coarse array.
    """
    factors = []
    for axis, (fine_res, coarse_res) in enumerate(
        zip(image_resolution_xyz, segmentation_resolution_xyz)
    ):
        ratio = float(coarse_res) / float(fine_res)
        factor = int(round(ratio))
        if factor < 1 or abs(ratio - factor) > 1.0e-6:
            raise ValueError(
                f"Segmentation/image resolution ratio on axis {axis} is {ratio}, "
                "which is not a positive integer; nearest upsampling needs a whole factor."
            )
        factors.append(factor)

    for axis, (fine_size, coarse_size, factor) in enumerate(
        zip(image_size_xyz, segmentation_size_xyz, factors)
    ):
        expected = -(-int(fine_size) // factor)
        if expected != int(coarse_size):
            raise ValueError(
                f"Axis {axis}: image size {fine_size} at factor {factor} implies a "
                f"segmentation size of {expected}, but the source reports {coarse_size}."
            )
    return tuple(factors)


def _iter_blocks(
    shape: Sequence[int], block_shape: Sequence[int]
) -> Iterator[tuple[slice, slice, slice]]:
    ranges = [range(0, int(size), int(block)) for size, block in zip(shape, block_shape)]
    for starts in product(*ranges):
        slices = tuple(
            slice(start, min(start + int(block), int(size)))
            for start, block, size in zip(starts, block_shape, shape)
        )
        yield slices[0], slices[1], slices[2]


def _validate_crop(
    source_shape_zyx: Sequence[int],
    crop_start_zyx: Sequence[int],
    crop_stop_zyx: Sequence[int],
    split_z: int,
) -> tuple[int, int, int]:
    if not (len(source_shape_zyx) == len(crop_start_zyx) == len(crop_stop_zyx) == 3):
        raise ValueError("Source shape and crop coordinates must each have three axes.")

    crop_shape = []
    for axis, (size, start, stop) in enumerate(
        zip(source_shape_zyx, crop_start_zyx, crop_stop_zyx)
    ):
        if not 0 <= int(start) < int(stop) <= int(size):
            raise ValueError(f"Invalid crop on axis {axis}: [{start}:{stop}] for size {size}.")
        crop_shape.append(int(stop) - int(start))

    if not 1 <= int(split_z) < crop_shape[0]:
        raise ValueError(
            f"split_z={split_z} must be in [1, {crop_shape[0] - 1}] "
            f"for crop shape {tuple(crop_shape)}."
        )
    return crop_shape[0], crop_shape[1], crop_shape[2]


def _open_precomputed(path: Path, scale_index: int):
    try:
        import tensorstore as ts
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorstore is required to read Neuroglancer precomputed volumes."
        ) from exc

    return ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path)},
            "scale_index": int(scale_index),
        },
        read=True,
    ).result()


def _read_zyx_block(
    store,
    block_zyx: tuple[slice, slice, slice],
    source_offset_zyx: Sequence[int],
) -> np.ndarray:
    z_slice, y_slice, x_slice = (
        slice(
            int(block.start) + int(offset),
            int(block.stop) + int(offset),
        )
        for block, offset in zip(block_zyx, source_offset_zyx)
    )
    array_xyz = store[x_slice, y_slice, z_slice, 0].read().result()
    return np.asarray(array_xyz).transpose(2, 1, 0)


def _read_zyx_block_upsampled(
    store,
    block_zyx: tuple[slice, slice, slice],
    source_offset_zyx: Sequence[int],
    factor_zyx: Sequence[int],
) -> np.ndarray:
    """Read a fine-grid block out of a coarser store by integer nearest upsampling.

    Fine index ``f`` maps to coarse index ``f // factor``. The covering coarse range is
    read once, repeated ``factor`` times per axis, then trimmed back to the requested
    fine block, so block boundaries need not align to the coarse grid.
    """
    coarse_slices = []
    trims = []
    for block, offset, factor in zip(block_zyx, source_offset_zyx, factor_zyx):
        fine_start = int(block.start) + int(offset)
        fine_stop = int(block.stop) + int(offset)
        coarse_start = fine_start // int(factor)
        coarse_stop = -(-fine_stop // int(factor))
        coarse_slices.append(slice(coarse_start, coarse_stop))
        trims.append((fine_start - coarse_start * int(factor), fine_stop - fine_start))

    z_slice, y_slice, x_slice = coarse_slices
    array_xyz = store[x_slice, y_slice, z_slice, 0].read().result()
    block_array = np.asarray(array_xyz).transpose(2, 1, 0)
    for axis, factor in enumerate(factor_zyx):
        if int(factor) != 1:
            block_array = np.repeat(block_array, int(factor), axis=axis)
    return block_array[
        trims[0][0] : trims[0][0] + trims[0][1],
        trims[1][0] : trims[1][0] + trims[1][1],
        trims[2][0] : trims[2][0] + trims[2][1],
    ]


def _describe_plan(
    image_path: Path,
    segmentation_path: Path,
    dataset_root: Path,
    resolution_xyz: Sequence[float],
    source_shape_zyx: Sequence[int],
    crop_start_zyx: Sequence[int],
    crop_stop_zyx: Sequence[int],
    crop_shape_zyx: Sequence[int],
    split_z: int,
    segmentation_resolution_xyz: Sequence[float] | None = None,
    factor_zyx: Sequence[int] | None = None,
) -> None:
    print(f"Image source:       {image_path}")
    print(f"Segmentation source:{segmentation_path}")
    print(f"Resolution XYZ:     {list(resolution_xyz)} nm")
    if segmentation_resolution_xyz is not None:
        print(
            f"Segmentation XYZ:   {list(segmentation_resolution_xyz)} nm "
            f"(nearest-upsampled by {list(factor_zyx)} ZYX)"
        )
    print(f"Source shape ZYX:   {tuple(source_shape_zyx)}")
    print(
        "Crop ZYX:           "
        f"{tuple(crop_start_zyx)} -> {tuple(crop_stop_zyx)} "
        f"shape={tuple(crop_shape_zyx)}"
    )
    print(f"Train shape ZYX:    {(split_z, *crop_shape_zyx[1:])}")
    print(f"Val shape ZYX:      {(crop_shape_zyx[0] - split_z, *crop_shape_zyx[1:])}")
    print(f"Output root:        {dataset_root}")


def _create_split_arrays(
    root,
    split_name: str,
    shape: Sequence[int],
    image_dtype,
    segmentation_dtype,
    chunk_shape: Sequence[int],
    compressor,
    metadata: dict,
):
    group = root.require_group(split_name).require_group("data.zarr")
    chunks = tuple(min(int(chunk), int(size)) for chunk, size in zip(chunk_shape, shape))
    image = group.create_dataset(
        "img",
        shape=tuple(shape),
        chunks=chunks,
        dtype=image_dtype,
        compressor=compressor,
        fill_value=0,
        overwrite=False,
    )
    segmentation = group.create_dataset(
        "seg",
        shape=tuple(shape),
        chunks=chunks,
        dtype=segmentation_dtype,
        compressor=compressor,
        fill_value=0,
        overwrite=False,
    )
    group.attrs.update(metadata)
    group.attrs["split"] = split_name
    for array, role in ((image, "image"), (segmentation, "segmentation")):
        array.attrs["axes"] = ["z", "y", "x"]
        array.attrs["resolution_nm_zyx"] = metadata["resolution_nm_zyx"]
        array.attrs["role"] = role
    return image, segmentation


def prepare(args: argparse.Namespace) -> None:
    try:
        import numcodecs
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "zarr and numcodecs are required to prepare the LICONN dataset."
        ) from exc

    image_path = Path(args.image).resolve()
    segmentation_path = Path(args.segmentation).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    image_info = _read_info(image_path)
    segmentation_info = _read_info(segmentation_path)
    segmentation_resolution_xyz = args.segmentation_resolution_xyz or args.resolution_xyz
    image_scale_index = _exact_scale_index(image_info, args.resolution_xyz)
    segmentation_scale_index = _exact_scale_index(segmentation_info, segmentation_resolution_xyz)
    image_scale = image_info["scales"][image_scale_index]
    segmentation_scale = segmentation_info["scales"][segmentation_scale_index]
    factor_xyz = _upsample_factor_xyz(
        args.resolution_xyz,
        segmentation_resolution_xyz,
        image_scale["size"],
        segmentation_scale["size"],
    )
    factor_zyx = tuple(reversed(factor_xyz))

    source_shape_zyx = tuple(reversed([int(value) for value in image_scale["size"]]))
    crop_shape_zyx = _validate_crop(
        source_shape_zyx,
        args.crop_start_zyx,
        args.crop_stop_zyx,
        args.split_z,
    )
    _describe_plan(
        image_path,
        segmentation_path,
        dataset_root,
        args.resolution_xyz,
        source_shape_zyx,
        args.crop_start_zyx,
        args.crop_stop_zyx,
        crop_shape_zyx,
        args.split_z,
        segmentation_resolution_xyz,
        factor_zyx,
    )
    if args.dry_run:
        return

    if dataset_root.exists():
        raise FileExistsError(f"Output already exists: {dataset_root}. Refusing to overwrite it.")
    staging_root = dataset_root.with_name(f"{dataset_root.name}.partial")
    if staging_root.exists():
        raise FileExistsError(
            f"Staging output already exists: {staging_root}. "
            "Inspect or remove it before retrying."
        )
    staging_root.parent.mkdir(parents=True, exist_ok=True)

    image_store = _open_precomputed(image_path, image_scale_index)
    segmentation_store = _open_precomputed(segmentation_path, segmentation_scale_index)
    image_dtype = np.dtype(image_info["data_type"])
    segmentation_dtype = np.dtype(segmentation_info["data_type"])
    compressor = numcodecs.Blosc(
        cname="lz4",
        clevel=5,
        shuffle=numcodecs.Blosc.SHUFFLE,
    )
    resolution_zyx = list(reversed([float(value) for value in args.resolution_xyz]))
    metadata = {
        "axes": ["z", "y", "x"],
        "resolution_nm_zyx": resolution_zyx,
        "source_resolution_nm_xyz": [float(value) for value in args.resolution_xyz],
        "source_image": str(image_path),
        "source_segmentation": str(segmentation_path),
        "source_crop_start_zyx": [int(value) for value in args.crop_start_zyx],
        "source_crop_stop_zyx": [int(value) for value in args.crop_stop_zyx],
        "split_z_in_crop": int(args.split_z),
        "segmentation_source_resolution_nm_xyz": [
            float(value) for value in segmentation_resolution_xyz
        ],
        "segmentation_upsample_factor_zyx": [int(value) for value in factor_zyx],
    }

    # PyTC currently targets Zarr v2 stores. Explicitly select v2 so the
    # numcodecs compressor and DirectoryStore layout remain compatible even
    # when the environment has zarr-python 3 installed.
    root = zarr.open_group(str(staging_root), mode="w", zarr_format=2)
    train_shape = (int(args.split_z), *crop_shape_zyx[1:])
    val_shape = (crop_shape_zyx[0] - int(args.split_z), *crop_shape_zyx[1:])
    outputs = {
        "train": _create_split_arrays(
            root,
            "train",
            train_shape,
            image_dtype,
            segmentation_dtype,
            args.chunk_shape,
            compressor,
            metadata,
        ),
        "val": _create_split_arrays(
            root,
            "val",
            val_shape,
            image_dtype,
            segmentation_dtype,
            args.chunk_shape,
            compressor,
            metadata,
        ),
    }

    total_blocks = sum(
        sum(1 for _ in _iter_blocks(shape, args.block_shape)) for shape in (train_shape, val_shape)
    )
    completed = 0
    for split_name, local_shape, source_z_offset in (
        ("train", train_shape, 0),
        ("val", val_shape, int(args.split_z)),
    ):
        image_output, segmentation_output = outputs[split_name]
        source_offset = (
            int(args.crop_start_zyx[0]) + source_z_offset,
            int(args.crop_start_zyx[1]),
            int(args.crop_start_zyx[2]),
        )
        for block in _iter_blocks(local_shape, args.block_shape):
            image_block = _read_zyx_block(image_store, block, source_offset)
            if factor_zyx == (1, 1, 1):
                segmentation_block = _read_zyx_block(segmentation_store, block, source_offset)
            else:
                segmentation_block = _read_zyx_block_upsampled(
                    segmentation_store, block, source_offset, factor_zyx
                )
            image_output[block] = image_block
            segmentation_output[block] = segmentation_block
            if args.verify_writes:
                if not np.array_equal(np.asarray(image_output[block]), image_block):
                    raise RuntimeError(f"Image verification failed at {split_name} {block}.")
                if not np.array_equal(np.asarray(segmentation_output[block]), segmentation_block):
                    raise RuntimeError(f"Segmentation verification failed at {split_name} {block}.")
            completed += 1
            print(
                f"[{completed}/{total_blocks}] {split_name} "
                f"z={block[0]} y={block[1]} x={block[2]}",
                flush=True,
            )

    staging_root.rename(dataset_root)
    print(f"Prepared and verified LICONN dataset: {dataset_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--segmentation", default=DEFAULT_SEGMENTATION)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--resolution-xyz",
        type=float,
        nargs=3,
        default=[18.0, 18.0, 24.0],
    )
    parser.add_argument(
        "--segmentation-resolution-xyz",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Segmentation scale to read, when it differs from --resolution-xyz. "
            "Must be a whole-number multiple of it; the labels are nearest-upsampled "
            "onto the image grid. Defaults to --resolution-xyz."
        ),
    )
    parser.add_argument(
        "--crop-start-zyx",
        type=int,
        nargs=3,
        default=[140, 240, 240],
    )
    parser.add_argument(
        "--crop-stop-zyx",
        type=int,
        nargs=3,
        default=[555, 4530, 3585],
    )
    parser.add_argument("--split-z", type=int, default=270)
    parser.add_argument(
        "--chunk-shape",
        type=int,
        nargs=3,
        default=[32, 256, 256],
    )
    parser.add_argument(
        "--block-shape",
        type=int,
        nargs=3,
        default=[32, 512, 512],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-verify-writes",
        dest="verify_writes",
        action="store_false",
        help="Skip immediate read-back verification of every output block.",
    )
    parser.set_defaults(verify_writes=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
