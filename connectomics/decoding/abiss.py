"""ABISS external wrapper decoder.

This module exposes a decoder that bridges prediction tensors to an external
ABISS (or ABISS-compatible) command-line pipeline.

The wrapper writes predictions to temporary files, runs a user-specified
command, then reads back an instance-label segmentation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional, Sequence
import os
import subprocess

import numpy as np

from connectomics.data.io import read_hdf5, write_hdf5

from .utils import cast2dtype

__all__ = ["decode_abiss"]


def _format_command(
    command: str | Sequence[str],
    mapping: Mapping[str, str],
) -> tuple[str | List[str], bool]:
    """Format command placeholders for shell/list execution."""
    if isinstance(command, str):
        return command.format(**mapping), True
    if isinstance(command, Sequence):
        return [str(part).format(**mapping) for part in command], False
    raise TypeError(f"`command` must be str or sequence[str], got {type(command).__name__}.")


def _load_output(output_h5: Path, output_npy: Path, output_dataset: str) -> np.ndarray:
    """Load decoded segmentation output from file."""
    if output_h5.exists():
        seg = read_hdf5(str(output_h5), dataset=output_dataset)
    elif output_npy.exists():
        seg = np.load(output_npy)
    else:
        raise FileNotFoundError(
            "decode_abiss did not produce output file. "
            f"Expected one of: {output_h5}, {output_npy}"
        )

    seg = np.asarray(seg)
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(
            "decode_abiss output must be 3D label volume (Z, Y, X) "
            f"or singleton-channel 4D; got shape {seg.shape}."
        )

    if not np.issubdtype(seg.dtype, np.integer):
        seg = np.rint(seg).astype(np.uint64, copy=False)

    return cast2dtype(seg)


def decode_abiss(
    predictions: np.ndarray,
    command: str | Sequence[str],
    *,
    input_dataset: str = "main",
    output_dataset: str = "main",
    channels: Optional[Sequence[int]] = None,
    workdir: Optional[str] = None,
    keep_workspace: bool = False,
    timeout_sec: Optional[int] = None,
    env: Optional[Dict[str, Any]] = None,
    check: bool = True,
) -> np.ndarray:
    """Decode instance segmentation with an external ABISS command.

    Args:
        predictions: Model output, typically shape ``(C, Z, Y, X)``.
        command: External command to execute. Supports placeholders:
            - ``{workspace}``: working directory path
            - ``{input_h5}``, ``{input_npy}``: prediction file paths
            - ``{output_h5}``, ``{output_npy}``: expected output file paths
            - ``{input_dataset}``, ``{output_dataset}``: HDF5 dataset names
        input_dataset: Dataset name when writing input HDF5.
        output_dataset: Dataset name when reading output HDF5.
        channels: Optional channel indices to select before saving input.
        workdir: Optional fixed workspace directory. If None, uses temp dir.
        keep_workspace: Keep temp workspace when using auto temp dir.
        timeout_sec: Optional subprocess timeout in seconds.
        env: Optional extra environment variables for subprocess.
        check: Raise on non-zero return code if True.

    Returns:
        3D instance label volume ``(Z, Y, X)``.
    """
    pred = np.asarray(predictions)
    if pred.ndim not in (3, 4):
        raise ValueError(
            f"decode_abiss expects 3D/4D predictions, got shape {pred.shape}."
        )

    if channels is not None:
        if pred.ndim != 4:
            raise ValueError("`channels` can only be used for 4D predictions (C, Z, Y, X).")
        pred = pred[np.asarray(channels)]

    if workdir is not None:
        workspace_path = Path(workdir).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = TemporaryDirectory(prefix="decode_abiss_")
        workspace_path = Path(temp_ctx.name).resolve()

    try:
        input_h5 = workspace_path / "predictions.h5"
        input_npy = workspace_path / "predictions.npy"
        output_h5 = workspace_path / "segmentation.h5"
        output_npy = workspace_path / "segmentation.npy"

        # Save both formats so external command can choose the easiest input.
        write_hdf5(str(input_h5), pred, dataset=input_dataset)
        np.save(input_npy, pred)

        mapping = {
            "workspace": str(workspace_path),
            "input_h5": str(input_h5),
            "input_npy": str(input_npy),
            "output_h5": str(output_h5),
            "output_npy": str(output_npy),
            "input_dataset": input_dataset,
            "output_dataset": output_dataset,
        }
        cmd, use_shell = _format_command(command, mapping)

        proc_env = os.environ.copy()
        if env:
            proc_env.update({str(k): str(v) for k, v in env.items()})

        subprocess.run(
            cmd,
            shell=use_shell,
            env=proc_env,
            cwd=str(workspace_path),
            check=check,
            timeout=timeout_sec,
        )

        return _load_output(output_h5=output_h5, output_npy=output_npy, output_dataset=output_dataset)
    finally:
        if temp_ctx is not None and not keep_workspace:
            temp_ctx.cleanup()
