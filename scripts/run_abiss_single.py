#!/usr/bin/env python3
"""Single-volume ABISS runner invoked by ``decode_abiss``.

This script is ABISS-only and errors out when ABISS watershed is unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Match scripts/main.py behavior so this script works when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.data.io import read_hdf5, write_hdf5  # noqa: E402

_ABISS_TAG = "0_0_0_0"


def _parse_threshold(value: str | float) -> tuple[float, bool]:
    """Parse a threshold value that may be absolute or percentile.

    Accepted formats:
      - ``0.8`` or ``"0.8"``  → absolute value 0.8
      - ``"80%"``              → 80th percentile

    Returns:
        (numeric_value, is_percentile) tuple.
    """
    if isinstance(value, (int, float)):
        return float(value), False

    s = str(value).strip()

    # "80%" style → 80th percentile
    if s.endswith("%"):
        try:
            return float(s[:-1]), True
        except ValueError:
            raise ValueError(f"Invalid percentile threshold: '{value}'. Use e.g. '80%' or '95.5%'.")

    # Plain numeric string
    return float(s), False


def _resolve_threshold(
    value: str | float,
    affinities: np.ndarray,
    name: str,
) -> float:
    """Resolve a threshold to an absolute value, computing percentile if needed.

    Args:
        value: Threshold — absolute float or percentile string (e.g. ``"p80"``).
        affinities: Affinity array to compute percentiles from (any shape).
        name: Parameter name for error messages.

    Returns:
        Absolute threshold value.
    """
    numeric, is_pct = _parse_threshold(value)
    if not is_pct:
        return numeric

    if not (0 <= numeric <= 100):
        raise ValueError(f"`{name}` percentile must be in [0, 100], got {numeric}.")

    absolute = float(np.percentile(affinities, numeric))
    print(f"  {name}: percentile {numeric}% → absolute {absolute:.6f}")
    return absolute


def _read_array(path: Path, dataset: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return np.asarray(read_hdf5(str(path), dataset=dataset))
    if suffix == ".npy":
        return np.asarray(np.load(path))
    raise ValueError(f"Unsupported input format '{path.suffix}'. Use .h5/.hdf5 or .npy.")


def _write_array(path: Path, array: np.ndarray, dataset: str) -> None:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        write_hdf5(str(path), np.asarray(array), dataset=dataset)
        return
    if suffix == ".npy":
        np.save(path, np.asarray(array))
        return
    raise ValueError(f"Unsupported output format '{path.suffix}'. Use .h5/.hdf5 or .npy.")


def _parse_int_list(value: str, *, expected_len: Optional[int] = None, name: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(
            f"`{name}` expects exactly {expected_len} comma-separated integers, got {len(out)}."
        )
    return out


def _resolve_ws_binary(abiss_home: Path, ws_binary: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []

    if ws_binary:
        candidates.append(Path(ws_binary).expanduser())

    # Try explicit ABISS home first.
    candidates.append(abiss_home / "build" / "ws")

    # Also try PATH.
    path_ws = shutil.which("ws")
    if path_ws:
        candidates.append(Path(path_ws))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _select_affinity_channels(
    predictions: np.ndarray, channels: Optional[Iterable[int]]
) -> np.ndarray:
    if channels is None:
        if predictions.shape[0] >= 3:
            return predictions[:3]
        return predictions[:1]

    idx = list(int(c) for c in channels)
    if len(idx) == 0:
        raise ValueError("`--channels` must contain at least one channel index.")
    return predictions[np.asarray(idx, dtype=np.int64)]


def _to_abiss_affinity(
    predictions_czyx: np.ndarray, channels: Optional[Iterable[int]]
) -> np.ndarray:
    selected = np.asarray(_select_affinity_channels(predictions_czyx, channels), dtype=np.float32)
    if selected.ndim != 4:
        raise ValueError(f"Expected selected affinity predictions to be 4D, got {selected.shape}.")

    n_channels = selected.shape[0]

    if n_channels == 1:
        # Match ABISS cutout conversion for a single probability map.
        pmap_xyz = np.transpose(selected[0], (2, 1, 0))
        affinities = [np.minimum(np.roll(pmap_xyz, shift=1, axis=ax), pmap_xyz) for ax in range(3)]
        aff_xyzc = np.stack(affinities, axis=-1)
    else:
        if n_channels < 3:
            raise ValueError(
                "ABISS watershed requires 1 channel (probability) or "
                f">=3 affinity channels; got {n_channels}."
            )
        aff_xyzc = np.transpose(selected[:3], (3, 2, 1, 0))

    return np.asfortranarray(aff_xyzc.astype(np.float32, copy=False))


def _write_affinity_with_halo(
    path: Path, aff_xyzc: np.ndarray, halo: int = 1
) -> tuple[int, int, int]:
    """Write ABISS affinity mmap with symmetric XYZ halo and return written XYZ shape."""
    if halo < 0:
        raise ValueError(f"`halo` must be >= 0, got {halo}.")

    if aff_xyzc.ndim != 4:
        raise ValueError(f"Expected affinity volume (X, Y, Z, C), got shape {aff_xyzc.shape}.")

    xdim, ydim, zdim, n_channels = aff_xyzc.shape
    if n_channels != 3:
        raise ValueError(f"ABISS ws expects 3 affinity channels, got {n_channels}.")

    x_out = xdim + 2 * halo
    y_out = ydim + 2 * halo
    z_out = zdim + 2 * halo
    mmap_shape = (x_out, y_out, z_out, n_channels)

    # ABISS ws writes seg_* from interior voxels, so we provide a 1-voxel halo.
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=mmap_shape, order="F")
    mm[...] = 0
    if halo == 0:
        mm[...] = aff_xyzc
    else:
        mm[halo : halo + xdim, halo : halo + ydim, halo : halo + zdim, :] = aff_xyzc
    mm.flush()
    del mm

    return (x_out, y_out, z_out)


def _read_segmentation_xyz(
    path: Path, xyz_shape: tuple[int, int, int], halo: int = 1
) -> np.ndarray:
    """Read ABISS segmentation mmap, supporting cropped and uncropped writer variants."""
    itemsize = np.dtype(np.uint64).itemsize
    n_expected = int(np.prod(xyz_shape, dtype=np.int64))
    bytes_expected = n_expected * itemsize
    file_size = path.stat().st_size
    bytes_with_halo: Optional[int] = None
    xyz_with_halo: Optional[tuple[int, int, int]] = None

    if file_size == bytes_expected:
        return np.array(
            np.memmap(path, dtype=np.uint64, mode="r", shape=xyz_shape, order="F"),
            copy=True,
        )

    if halo > 0:
        xyz_with_halo = tuple(int(v + 2 * halo) for v in xyz_shape)
        n_with_halo = int(np.prod(xyz_with_halo, dtype=np.int64))
        bytes_with_halo = n_with_halo * itemsize
        if file_size == bytes_with_halo:
            seg_with_halo = np.memmap(
                path,
                dtype=np.uint64,
                mode="r",
                shape=xyz_with_halo,
                order="F",
            )
            return np.array(
                seg_with_halo[halo:-halo, halo:-halo, halo:-halo],
                copy=True,
            )

    if bytes_with_halo is None or xyz_with_halo is None:
        raise ValueError(
            "Unexpected ABISS segmentation file size. "
            f"Expected {bytes_expected} bytes for shape {xyz_shape}, "
            f"got {file_size} bytes at {path}."
        )
    raise ValueError(
        "Unexpected ABISS segmentation file size. "
        f"Expected {bytes_expected} bytes for shape {xyz_shape} "
        f"(or {bytes_with_halo} bytes for shape {xyz_with_halo} with halo), "
        f"got {file_size} bytes at {path}."
    )


def _write_abiss_param_file(
    path: Path, xyz_shape: tuple[int, int, int], boundary_flags: list[int], offset: int
) -> None:
    xdim, ydim, zdim = xyz_shape
    flags = " ".join(str(int(v)) for v in boundary_flags)
    path.write_text(f"{xdim} {ydim} {zdim}\n{flags}\n{int(offset)}\n", encoding="utf-8")


def _run_abiss_ws(
    predictions_czyx: np.ndarray,
    ws_binary: Path,
    ws_high_threshold: float,
    ws_low_threshold: float,
    ws_size_threshold: int,
    ws_dust_threshold: int,
    boundary_flags: list[int],
    offset: int,
    channels: Optional[list[int]] = None,
    workdir: Optional[Path] = None,
    keep_workdir: bool = False,
    ws_merge_threshold: Optional[float] = None,
    ws_merge_thresholds: Optional[list[float]] = None,
) -> "np.ndarray | dict[float, np.ndarray]":
    if ws_low_threshold > ws_high_threshold:
        raise ValueError(
            "Expected ws_low_threshold <= ws_high_threshold, got "
            f"{ws_low_threshold} > {ws_high_threshold}."
        )

    aff_xyzc = _to_abiss_affinity(predictions_czyx, channels=channels)
    output_xyz_shape = tuple(int(v) for v in aff_xyzc.shape[:3])

    if workdir is not None:
        ws_dir = workdir.resolve()
        ws_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="abiss_single_", dir=tempfile.gettempdir())
        ws_dir = Path(temp_ctx.name).resolve()

    try:
        aff_raw = ws_dir / "aff.raw"
        param_txt = ws_dir / "param.txt"
        seg_raw = ws_dir / f"seg_{_ABISS_TAG}.data"

        ws_xyz_shape = _write_affinity_with_halo(aff_raw, aff_xyzc, halo=1)
        _write_abiss_param_file(param_txt, ws_xyz_shape, boundary_flags, offset)

        cmd = [
            str(ws_binary),
            str(param_txt),
            str(aff_raw),
            str(ws_high_threshold),
            str(ws_low_threshold),
            str(int(ws_size_threshold)),
            str(int(ws_dust_threshold)),
            _ABISS_TAG,
        ]

        # Batch mode: pass multiple merge thresholds as argv[8..N].
        # The C++ binary computes watershed + region graph once, then
        # deep-copies and repeats the merge step for each threshold,
        # writing indexed output files (seg_{TAG}_{i}.data).
        use_batch = ws_merge_thresholds is not None and len(ws_merge_thresholds) > 1
        if use_batch:
            for mt in ws_merge_thresholds:
                cmd.append(str(mt))
        elif ws_merge_threshold is not None:
            cmd.append(str(ws_merge_threshold))

        subprocess.run(cmd, cwd=str(ws_dir), check=True)

        if use_batch:
            results: dict[float, np.ndarray] = {}
            for i, mt in enumerate(ws_merge_thresholds):
                seg_file = ws_dir / f"seg_{_ABISS_TAG}_{i}.data"
                if not seg_file.exists():
                    raise FileNotFoundError(
                        f"ABISS batch mode did not produce expected output: {seg_file}. "
                        f"Ensure the ws binary at {ws_binary} supports multi-threshold mode."
                    )
                seg_xyz = _read_segmentation_xyz(seg_file, output_xyz_shape, halo=1)
                results[round(mt, 10)] = np.transpose(seg_xyz, (2, 1, 0))
            return results

        if not seg_raw.exists():
            raise FileNotFoundError(f"ABISS watershed did not produce expected output: {seg_raw}")

        seg_xyz = _read_segmentation_xyz(seg_raw, output_xyz_shape, halo=1)
        # ABISS stores X,Y,Z. Convert back to Z,Y,X expected by this repository.
        return np.transpose(seg_xyz, (2, 1, 0))
    finally:
        if temp_ctx is not None and not keep_workdir:
            temp_ctx.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input predictions file (.h5/.hdf5/.npy)")
    parser.add_argument("--output", required=True, help="Output segmentation file (.h5/.hdf5/.npy)")
    parser.add_argument("--input-dataset", default="main", help="Dataset key for HDF5 input")
    parser.add_argument("--output-dataset", default="main", help="Dataset key for HDF5 output")

    parser.add_argument(
        "--abiss-home",
        default=str(Path(__file__).resolve().parent.parent / "lib" / "abiss"),
        help="Path to ABISS checkout containing build/ws.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Optional comma-separated channel indices to pass into ABISS affinity conversion.",
    )
    parser.add_argument(
        "--ws-binary",
        default=None,
        help="Optional path to ABISS ws binary. Overrides --abiss-home discovery.",
    )
    parser.add_argument(
        "--ws-high-threshold",
        type=str,
        default="0.5",
        help=(
            "ABISS watershed high threshold. Absolute (e.g. 0.8) or "
            "percentile (e.g. p80 or 80%%)."
        ),
    )
    parser.add_argument(
        "--ws-low-threshold",
        type=str,
        default="0.5",
        help="ABISS watershed low threshold. Absolute (e.g. 0.01) or percentile (e.g. p1 or 1%%).",
    )
    parser.add_argument(
        "--ws-size-threshold",
        type=int,
        default=0,
        help="ABISS watershed size threshold.",
    )
    parser.add_argument(
        "--ws-dust-threshold",
        type=int,
        default=None,
        help="ABISS watershed dust threshold. Default: --ws-size-threshold.",
    )
    parser.add_argument(
        "--ws-merge-threshold",
        type=str,
        default=None,
        help=(
            "Affinity threshold for size-based merging. Absolute or "
            "percentile. Default: --ws-low-threshold."
        ),
    )
    parser.add_argument(
        "--ws-merge-thresholds",
        type=str,
        default=None,
        help="Comma-separated merge thresholds for batch mode. Runs watershed once, "
        "merges with each threshold. Writes {output}_mt{i}.{ext} per threshold.",
    )
    parser.add_argument(
        "--abiss-boundary-flags",
        default="1,1,1,1,1,1",
        help="Six boundary flags for ABISS param.txt as comma-separated ints.",
    )
    parser.add_argument(
        "--abiss-offset",
        type=int,
        default=0,
        help="Segment id offset written to ABISS param.txt.",
    )
    parser.add_argument(
        "--abiss-workdir",
        default=None,
        help="Optional persistent workspace path for ABISS intermediate files.",
    )
    parser.add_argument(
        "--keep-abiss-workdir",
        action="store_true",
        help="Keep temporary ABISS workdir for debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    predictions = _read_array(input_path, dataset=args.input_dataset)
    if predictions.ndim == 5 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 4:
        raise ValueError(
            f"Expected predictions with shape (C, Z, Y, X); got shape {predictions.shape}."
        )

    ws_dust = args.ws_size_threshold if args.ws_dust_threshold is None else args.ws_dust_threshold
    channels = _parse_int_list(args.channels, name="channels") if args.channels else None
    boundary_flags = _parse_int_list(
        args.abiss_boundary_flags,
        expected_len=6,
        name="abiss-boundary-flags",
    )
    abiss_home = Path(args.abiss_home).expanduser().resolve()
    ws_binary = _resolve_ws_binary(abiss_home, args.ws_binary)
    if ws_binary is None:
        raise FileNotFoundError(
            "ABISS ws binary was not found. "
            "Set --ws-binary or --abiss-home to a valid ABISS checkout."
        )

    # Determine batch vs single merge threshold mode.
    batch_merge_thresholds: Optional[list[float]] = None
    if args.ws_merge_thresholds is not None:
        batch_merge_thresholds = [float(x.strip()) for x in args.ws_merge_thresholds.split(",")]

    # Resolve percentile thresholds against the affinity data.
    all_thresh_strs = [args.ws_high_threshold, args.ws_low_threshold]
    if args.ws_merge_threshold is not None:
        all_thresh_strs.append(args.ws_merge_threshold)

    needs_percentile = any(_parse_threshold(v)[1] for v in all_thresh_strs)
    if needs_percentile:
        aff_for_pct = _to_abiss_affinity(predictions, channels=channels)
        print(
            f"Resolving percentile thresholds (affinity range: "
            f"[{aff_for_pct.min():.6f}, {aff_for_pct.max():.6f}]):"
        )
    else:
        aff_for_pct = np.empty(0)  # unused, _resolve_threshold won't compute percentiles

    ws_high = _resolve_threshold(args.ws_high_threshold, aff_for_pct, "ws_high_threshold")
    ws_low = _resolve_threshold(args.ws_low_threshold, aff_for_pct, "ws_low_threshold")
    ws_merge: Optional[float] = None
    if args.ws_merge_threshold is not None:
        ws_merge = _resolve_threshold(args.ws_merge_threshold, aff_for_pct, "ws_merge_threshold")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch_merge_thresholds is not None and len(batch_merge_thresholds) > 1:
        # Batch mode: run watershed once, merge with each threshold
        result = _run_abiss_ws(
            predictions_czyx=predictions,
            ws_binary=ws_binary,
            ws_high_threshold=ws_high,
            ws_low_threshold=ws_low,
            ws_size_threshold=int(args.ws_size_threshold),
            ws_dust_threshold=int(ws_dust),
            boundary_flags=boundary_flags,
            offset=int(args.abiss_offset),
            channels=channels,
            workdir=Path(args.abiss_workdir).resolve() if args.abiss_workdir else None,
            keep_workdir=bool(args.keep_abiss_workdir),
            ws_merge_thresholds=batch_merge_thresholds,
        )
        stem = output_path.stem
        ext = output_path.suffix
        parent = output_path.parent
        for i, mt in enumerate(batch_merge_thresholds):
            mt_path = parent / f"{stem}_mt{i}{ext}"
            _write_array(mt_path, result[round(mt, 10)], dataset=args.output_dataset)
        return 0

    segmentation = _run_abiss_ws(
        predictions_czyx=predictions,
        ws_binary=ws_binary,
        ws_high_threshold=ws_high,
        ws_low_threshold=ws_low,
        ws_size_threshold=int(args.ws_size_threshold),
        ws_dust_threshold=int(ws_dust),
        boundary_flags=boundary_flags,
        offset=int(args.abiss_offset),
        channels=channels,
        workdir=Path(args.abiss_workdir).resolve() if args.abiss_workdir else None,
        keep_workdir=bool(args.keep_abiss_workdir),
        ws_merge_threshold=ws_merge,
    )

    _write_array(output_path, segmentation, dataset=args.output_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
