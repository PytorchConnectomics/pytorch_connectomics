"""ABISS external wrapper decoder.

This module exposes a decoder that bridges prediction tensors to an external
ABISS (or ABISS-compatible) command-line pipeline.

The wrapper writes predictions to temporary files, runs a user-specified
command, then reads back an instance-label segmentation.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from connectomics.data.io import read_hdf5, write_hdf5

from ..utils import cast2dtype

__all__ = ["decode_abiss"]


def _format_command(
    command: str | Sequence[str],
    mapping: Mapping[str, str],
) -> tuple[str | List[str], bool]:
    """Format command placeholders for shell/list execution."""
    if isinstance(command, str):
        return command.format_map(_SafeFormatMapping(mapping)), True
    if isinstance(command, Sequence):
        safe = _SafeFormatMapping(mapping)
        return [str(part).format_map(safe) for part in command], False
    raise TypeError(f"`command` must be str or sequence[str], got {type(command).__name__}.")


class _SafeFormatMapping(dict):
    """Mapping that leaves unknown placeholders untouched during str.format_map."""

    def __init__(self, mapping: Mapping[str, str]):
        super().__init__(mapping)

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _build_cli_suffix(cli_args: Dict[str, Any]) -> List[str]:
    """Convert a dict of parameters into CLI ``--key value`` tokens.

    Underscores in keys are converted to hyphens (``ws_high_threshold`` →
    ``--ws-high-threshold``).  List/tuple values are joined with commas.
    """
    tokens: List[str] = []
    for key, value in cli_args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
        elif isinstance(value, (list, tuple)):
            tokens.append(flag)
            tokens.append(",".join(str(v) for v in value))
        elif value is not None:
            tokens.append(flag)
            tokens.append(str(value))
    return tokens


def _append_cli_args(
    cmd: str | List[str],
    cli_tokens: List[str],
    use_shell: bool,
) -> str | List[str]:
    """Append CLI argument tokens to an already-formatted command."""
    if not cli_tokens:
        return cmd
    if use_shell:
        suffix = " " + " ".join(shlex.quote(t) for t in cli_tokens)
        return cmd + suffix
    return list(cmd) + cli_tokens


def _resolve_python_script_path(
    cmd: str | List[str],
    launch_cwd: Path,
    search_roots: Sequence[Path],
) -> str | List[str]:
    """Resolve relative Python script path to absolute path when possible."""

    def _patch_tokens(tokens: List[str]) -> bool:
        for idx in range(len(tokens) - 1):
            interpreter = Path(tokens[idx]).name.lower()
            if not interpreter.startswith("python"):
                continue

            script = tokens[idx + 1]
            if script.startswith("-"):
                continue

            candidate = Path(script).expanduser()
            if candidate.is_absolute():
                continue

            for root in search_roots:
                resolved = (root / candidate).resolve()
                if resolved.exists():
                    tokens[idx + 1] = str(resolved)
                    return True
        return False

    if isinstance(cmd, str):
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return cmd
        if _patch_tokens(tokens):
            return shlex.join(tokens)
        return cmd

    tokens = list(cmd)
    _patch_tokens(tokens)
    return tokens


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
    cli_args: Optional[Dict[str, Any]] = None,
) -> "np.ndarray | Dict[float, np.ndarray]":
    """Decode instance segmentation with an external ABISS command.

    Args:
        predictions: Model output, typically shape ``(C, Z, Y, X)``.
        command: External command to execute. Supports placeholders:
            - ``{workspace}``: working directory path
            - ``{input_h5}``, ``{input_npy}``: prediction file paths
            - ``{output_h5}``, ``{output_npy}``: expected output file paths
            - ``{input_dataset}``, ``{output_dataset}``: HDF5 dataset names
            - ``{python_exe}``: current Python interpreter path
            - Any key from *cli_args* (e.g. ``{ws_high_threshold}``)
        input_dataset: Dataset name when writing input HDF5.
        output_dataset: Dataset name when reading output HDF5.
        channels: Optional channel indices to select before saving input.
        workdir: Optional fixed workspace directory. If None, uses temp dir.
        keep_workspace: Keep temp workspace when using auto temp dir.
        timeout_sec: Optional subprocess timeout in seconds.
        env: Optional extra environment variables for subprocess.
        check: Raise on non-zero return code if True.
        cli_args: Optional dict of extra parameters appended to the command
            as ``--key value`` CLI flags.  Underscores in keys are converted
            to hyphens (e.g. ``ws_high_threshold`` → ``--ws-high-threshold``).
            Values are also available as placeholders in the command template.

            **Batch mode**: when *cli_args* contains a key
            ``ws_merge_thresholds`` whose value is a list of floats, the
            external script is invoked once with all thresholds and this
            function returns ``Dict[float, np.ndarray]`` instead of a
            single array.

    Returns:
        3D instance label volume ``(Z, Y, X)`` — or a dict mapping each
        merge threshold to its label volume when batch mode is active.
    """
    pred = np.asarray(predictions)
    if pred.ndim not in (3, 4):
        raise ValueError(f"decode_abiss expects 3D/4D predictions, got shape {pred.shape}.")

    if channels is not None:
        if pred.ndim != 4:
            raise ValueError("`channels` can only be used for 4D predictions (C, Z, Y, X).")
        pred = pred[np.asarray(channels)]

    # Detect batch merge-threshold mode.
    batch_mt: Optional[list] = None
    if cli_args and "ws_merge_thresholds" in cli_args:
        batch_mt = list(cli_args["ws_merge_thresholds"])

    if workdir is not None:
        workspace_path = Path(workdir).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = TemporaryDirectory(prefix="decode_abiss_")
        workspace_path = Path(temp_ctx.name).resolve()
    launch_cwd = Path.cwd().resolve()
    package_root = Path(__file__).resolve().parents[2]

    search_roots: List[Path] = []
    for root in (
        launch_cwd,
        Path(os.environ["HYDRA_ORIG_CWD"]).resolve() if "HYDRA_ORIG_CWD" in os.environ else None,
        (
            Path(os.environ["HYDRA_ORIGINAL_CWD"]).resolve()
            if "HYDRA_ORIGINAL_CWD" in os.environ
            else None
        ),
        package_root,
    ):
        if root is None:
            continue
        if root not in search_roots:
            search_roots.append(root)

    try:
        input_h5 = workspace_path / "predictions.h5"
        input_npy = workspace_path / "predictions.npy"
        output_h5 = workspace_path / "segmentation.h5"
        output_npy = workspace_path / "segmentation.npy"

        # Save both formats so external command can choose the easiest input.
        write_hdf5(str(input_h5), pred, dataset=input_dataset)
        np.save(input_npy, pred)

        mapping: Dict[str, str] = {
            "workspace": str(workspace_path),
            "input_h5": str(input_h5),
            "input_npy": str(input_npy),
            "output_h5": str(output_h5),
            "output_npy": str(output_npy),
            "input_dataset": input_dataset,
            "output_dataset": output_dataset,
            "python_exe": sys.executable,
        }
        # Merge cli_args into placeholders so templates can reference them.
        if cli_args:
            mapping.update(
                {k: str(v) for k, v in cli_args.items() if not isinstance(v, (list, tuple))}
            )

        cmd, use_shell = _format_command(command, mapping)
        cmd = _resolve_python_script_path(cmd, launch_cwd, search_roots)

        # Auto-append cli_args as --key value flags to the command.
        if cli_args:
            cli_tokens = _build_cli_suffix(cli_args)
            cmd = _append_cli_args(cmd, cli_tokens, use_shell)

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

        # Batch mode: read multiple output files written by run_abiss_single.
        if batch_mt:
            stem = output_h5.stem  # "segmentation"
            ext = output_h5.suffix  # ".h5"
            results: Dict[float, np.ndarray] = {}
            for i, mt in enumerate(batch_mt):
                mt_h5 = workspace_path / f"{stem}_mt{i}{ext}"
                mt_npy = workspace_path / f"{stem}_mt{i}.npy"
                results[round(mt, 10)] = _load_output(
                    output_h5=mt_h5,
                    output_npy=mt_npy,
                    output_dataset=output_dataset,
                )
            return results

        return _load_output(
            output_h5=output_h5, output_npy=output_npy, output_dataset=output_dataset
        )
    finally:
        if temp_ctx is not None and not keep_workspace:
            temp_ctx.cleanup()
