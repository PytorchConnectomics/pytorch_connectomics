"""Prepare registered simulator A/B views and integer labels without changing sources.

Run with an environment that already provides Zarr v3 (e.g. dispim_sim/.venv).
The simulator resamples foreground using align_corners=False; labels use the
same voxel-center coordinates with nearest-neighbor sampling of integer IDs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def label_grid(labels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Source labels must be a ZYX integer array")
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Output shape must have three positive dimensions")
    indices = [
        np.minimum(((np.arange(m) + 0.5) * n / m).astype(np.int64), n - 1)
        for n, m in zip(labels.shape, shape)
    ]
    return labels[np.ix_(*indices)]


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path: Path, array: np.ndarray) -> None:
    with h5py.File(path, "x") as handle:
        handle.create_dataset("main", data=array, compression="gzip", compression_opts=1)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    input_dir: Path, labels_root: Path, output: Path, split: float = 0.8,
    spacing_nm: int = 20, slab_depth: int = 8,
) -> dict:
    import zarr

    if not 0 < split < 1:
        raise ValueError("Training split must be between zero and one")
    if spacing_nm not in (20, 10, 5) or slab_depth < 1:
        raise ValueError("Expected spacing 20, 10 or 5 nm and positive slab depth")
    suffix = "" if spacing_nm == 20 else f"-{spacing_nm}nm"
    output.mkdir(parents=True, exist_ok=False)
    source_labels = zarr.open_group(str(labels_root), mode="r")
    provenance = {
        "channel_order": ["image_blur_A", "image_blur_B"],
        "label_sampling": "voxel-center nearest: floor((output_index + 0.5) * N / M)",
        "input_axes": "CZYX",
        "label_axes": "ZYX",
        "sources": {},
        "status": "preparing",
    }
    for name, key in (("train", "segments"), ("test", "test_segments")):
        group_path = input_dir / f"snemi-{name}{suffix}.zarr"
        group = zarr.open_group(str(group_path), mode="r")
        attrs = dict(group.attrs)
        if (
            attrs.get("simulation_mode") != "whole_volume"
            or attrs.get("axes") != "ZYX"
            or attrs.get("expansion_deformation_applied") is not False
            or attrs.get("sectioning", {}).get("active") is not False
            or attrs.get("tiling", {}).get("active") is not False
        ):
            raise ValueError("Expected registered whole-volume simulation without deformation")
        if attrs["source"].split("::")[-1].lstrip("/") != key:
            raise ValueError("Simulator source label selector does not match split")
        views = [group[k] for k in provenance["channel_order"]]
        shape = views[0].shape
        if len(shape) != 3 or any(v.shape != shape for v in views):
            raise ValueError("Both views must be finite, registered ZYX arrays of equal shape")
        labels = np.asarray(source_labels[key][:])
        experiment = json.loads(attrs["experiment_params"])
        expansion = float(experiment["expansion_factor"])
        spacing = np.asarray(attrs["voxel_size_um_zyx"]) / expansion * 1000
        source_spacing = np.asarray(attrs["source_voxel_size_um_zyx"])
        expected = tuple(np.rint(np.asarray(labels.shape) * source_spacing * 1000 / spacing).astype(int))
        if shape != expected or not np.allclose(spacing, [spacing_nm] * 3):
            raise ValueError(f"{spacing_nm} nm shape/spacing mismatch: {shape}, {expected}, {spacing}")
        boundary = int(shape[0] * split)
        if not 0 < boundary < shape[0]:
            raise ValueError("Empty train/validation slab")
        parts = [("train", 0, boundary), ("val", boundary, shape[0])] if name == "train" else [
            ("test", 0, shape[0])
        ]
        indices = [
            np.minimum(((np.arange(m) + 0.5) * n / m).astype(np.int64), n - 1)
            for n, m in zip(labels.shape, shape)
        ]
        label_digest = hashlib.sha256()
        view_digests = [hashlib.sha256(), hashlib.sha256()]
        foreground = False
        with ExitStack() as stack:
            targets = []
            for part, lo, hi in parts:
                image_file = stack.enter_context(h5py.File(output / f"{part}-views.h5", "x"))
                label_file = stack.enter_context(h5py.File(output / f"{part}-labels.h5", "x"))
                spatial = (hi - lo, *shape[1:])
                chunks = tuple(min(n, c) for n, c in zip(spatial, (slab_depth, 128, 128)))
                image = image_file.create_dataset(
                    "main", shape=(2, *spatial), dtype=views[0].dtype,
                    chunks=(1, *chunks), compression="gzip", compression_opts=1,
                )
                label = label_file.create_dataset(
                    "main", shape=spatial, dtype=labels.dtype, chunks=chunks,
                    compression="gzip", compression_opts=1,
                )
                targets.append((lo, hi, image, label))
            # Stream Z slabs: 5 nm pairs need several GB, but preparation does
            # not need a full paired array or an expanded label grid in RAM.
            for z0 in range(0, shape[0], slab_depth):
                z1 = min(z0 + slab_depth, shape[0])
                mapped = labels[np.ix_(indices[0][z0:z1], indices[1], indices[2])]
                foreground = foreground or bool(np.any(mapped))
                label_digest.update(np.ascontiguousarray(mapped).tobytes())
                for lo, hi, _, target in targets:
                    left, right = max(z0, lo), min(z1, hi)
                    if left < right:
                        target[left - lo:right - lo] = mapped[left - z0:right - z0]
                for channel, view in enumerate(views):
                    slab = np.asarray(view[z0:z1])
                    if not np.isfinite(slab).all():
                        raise ValueError("Simulation views must be finite")
                    view_digests[channel].update(np.ascontiguousarray(slab).tobytes())
                    for lo, hi, target, _ in targets:
                        left, right = max(z0, lo), min(z1, hi)
                        if left < right:
                            target[channel, left - lo:right - lo] = slab[left - z0:right - z0]
        if not foreground:
            raise ValueError("Ground truth contains no foreground")
        record = {
            "gcs": f"gs://donglai/dispim/snemi/snemi-{name}{suffix}.zarr",
            "metadata": attrs,
            "source_label_path": str(labels_root / key),
            "source_label_sha256": array_hash(labels),
            "source_label_shape": list(labels.shape),
            "mapped_label_sha256": label_digest.hexdigest(),
            "shape": list(shape),
            "effective_spacing_nm_zyx": spacing.tolist(),
            "view_sha256": [digest.hexdigest() for digest in view_digests],
        }
        if name == "train":
            record["train_z"] = [0, boundary]
            record["validation_z"] = [boundary, shape[0]]
        provenance["sources"][name] = record
    provenance["status"] = "complete"
    provenance["files"] = {
        p.name: file_hash(p) for p in sorted(output.glob("*.h5"))
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-nm", type=int, choices=(20, 10, 5), default=20)
    args = parser.parse_args()
    result = prepare(args.input_dir, args.labels_root, args.output, spacing_nm=args.spacing_nm)
    print(json.dumps({k: v["shape"] for k, v in result["sources"].items()}))
