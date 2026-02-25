#!/usr/bin/env python3
"""
Apply a Python function to input array/volume data and save the result.

Examples:
    # Apply seg_erosion_instance on an HDF5 segmentation volume
    python scripts/apply_volume_function.py \
        --input seg.h5 \
        --output seg_eroded.h5 \
        --function connectomics.data.process.segment:seg_erosion_instance \
        --kwargs-json '{"tsz_h": 1}'

    # Using a repo-relative file path callable spec
    python scripts/apply_volume_function.py \
        --input seg.npy \
        --output seg_eroded.npy \
        --function connectomics/data/process/segment.py:seg_erosion_instance \
        --kwargs-json '{"tsz_h": 2}'
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running the script directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.data.io.io import read_volume, save_volume


def _infer_output_format(path: str) -> str:
    if path.endswith(".nii.gz"):
        return "nii.gz"

    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"h5", "hdf5"}:
        return "h5"
    if suffix in {"tif", "tiff"}:
        return suffix
    if suffix in {"nii"}:
        return suffix
    if suffix in {"png"}:
        return "png"
    if suffix in {"npy", "npz"}:
        return suffix
    raise ValueError(
        f"Cannot infer output format from {path!r}. Use a supported extension or --output-format."
    )


def _load_array(path: str, dataset: str | None = None, npz_key: str | None = None) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.load(path)
    if lower.endswith(".npz"):
        with np.load(path) as data:
            key = npz_key or (data.files[0] if data.files else None)
            if key is None:
                raise ValueError(f"No arrays found in NPZ file: {path}")
            if key not in data.files:
                raise KeyError(f"NPZ key {key!r} not found in {path}. Available: {data.files}")
            return data[key]
    return read_volume(path, dataset=dataset)


def _save_array(
    path: str,
    array: np.ndarray,
    *,
    dataset: str,
    npz_key: str,
    output_format: str | None,
) -> None:
    fmt = output_format or _infer_output_format(path)
    if fmt == "npy":
        np.save(path, array)
        return
    if fmt == "npz":
        np.savez_compressed(path, **{npz_key: array})
        return
    save_volume(path, array, dataset=dataset, file_format=fmt)


def _load_callable_from_file(file_path: str, attr_name: str):
    module_path = Path(file_path)
    if not module_path.is_absolute():
        cwd_path = Path.cwd() / module_path
        repo_path = Path(__file__).resolve().parent.parent / module_path
        if cwd_path.exists():
            module_path = cwd_path
        elif repo_path.exists():
            module_path = repo_path

    if not module_path.exists():
        raise FileNotFoundError(f"Function file not found: {file_path}")

    module_name = f"_apply_fn_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"{attr_name!r} not found in {module_path}") from exc


def _resolve_callable(spec: str):
    """
    Supported callable specs:
      - package.module:function
      - package.module.function
      - path/to/file.py:function
    """
    if ":" in spec:
        left, attr = spec.rsplit(":", 1)
        if left.endswith(".py"):
            return _load_callable_from_file(left, attr)
        module = importlib.import_module(left)
        return getattr(module, attr)

    if spec.endswith(".py"):
        raise ValueError("File-based function spec must use 'path/to/file.py:function_name'")

    if "." not in spec:
        raise ValueError(
            "Function spec must be 'package.module:function', "
            "'package.module.function', or 'path/to/file.py:function'."
        )

    module_name, attr = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _parse_json(value: str, name: str, expected_type: type) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {name}: {exc}") from exc

    if not isinstance(parsed, expected_type):
        raise TypeError(f"{name} must decode to {expected_type.__name__}, got {type(parsed).__name__}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load input data, apply a Python function, and save the output."
    )
    parser.add_argument("--input", required=True, help="Input file path or pattern (for PNG/TIFF stacks).")
    parser.add_argument("--output", required=True, help="Output file path (or PNG output directory).")
    parser.add_argument(
        "--function",
        required=True,
        help=(
            "Callable spec: 'package.module:function', 'package.module.function', "
            "or 'path/to/file.py:function'."
        ),
    )
    parser.add_argument(
        "--args-json",
        default="[]",
        help="JSON list of positional args passed after the input array. Default: []",
    )
    parser.add_argument(
        "--kwargs-json",
        default="{}",
        help="JSON object of keyword args for the function. Default: {}",
    )
    parser.add_argument(
        "--input-dataset",
        default="main",
        help="HDF5 dataset key for input .h5/.hdf5 files (default: main).",
    )
    parser.add_argument(
        "--output-dataset",
        default="main",
        help="HDF5 dataset key for output .h5/.hdf5 files (default: main).",
    )
    parser.add_argument(
        "--input-npz-key",
        default=None,
        help="Array key for input .npz (default: first key).",
    )
    parser.add_argument(
        "--output-npz-key",
        default="main",
        help="Array key for output .npz (default: main).",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        choices=["h5", "tif", "tiff", "png", "nii", "nii.gz", "npy", "npz"],
        help="Override inferred output format.",
    )
    parser.add_argument(
        "--copy-input",
        action="store_true",
        help="Pass a copy of the loaded array to the function (useful for in-place functions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fn = _resolve_callable(args.function)
    fn_args = _parse_json(args.args_json, "--args-json", list)
    fn_kwargs = _parse_json(args.kwargs_json, "--kwargs-json", dict)

    input_array = _load_array(args.input, dataset=args.input_dataset, npz_key=args.input_npz_key)
    work_array = input_array.copy() if args.copy_input else input_array

    print(f"Loaded input: {args.input}")
    print(f"  shape={input_array.shape}, dtype={input_array.dtype}")
    print(f"Applying: {args.function}")
    if fn_args:
        print(f"  args={fn_args}")
    if fn_kwargs:
        print(f"  kwargs={fn_kwargs}")

    result = fn(work_array, *fn_args, **fn_kwargs)
    if result is None:
        result = work_array

    if not isinstance(result, np.ndarray):
        result = np.asarray(result)

    output_dir = args.output if (args.output_format == "png" or args.output.endswith(".png")) else os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    _save_array(
        args.output,
        result,
        dataset=args.output_dataset,
        npz_key=args.output_npz_key,
        output_format=args.output_format,
    )

    print(f"Saved output: {args.output}")
    print(f"  shape={result.shape}, dtype={result.dtype}")


if __name__ == "__main__":
    main()
