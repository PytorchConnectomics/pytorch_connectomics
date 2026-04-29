"""Checkpoint-derived runtime output dispatch helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..training.lightning.runtime import setup_run_directory
from .output_naming import tta_cache_suffix


def get_output_base_from_checkpoint(checkpoint_path: str) -> Path:
    """Determine the output base directory from checkpoint path."""
    ckpt_path = Path(checkpoint_path)
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    for parent in ckpt_path.parents:
        if timestamp_pattern.match(parent.name):
            return parent

    return ckpt_path.parent.parent / ckpt_path.stem


def extract_step_from_checkpoint(checkpoint_path: str) -> str:
    """Extract ``step=<N>`` from a checkpoint filename."""
    match = re.search(r"step=(\d+)", Path(checkpoint_path).stem)
    return f"step={match.group(1)}" if match else ""


def configure_checkpoint_output_paths(args: Any, cfg: Config) -> tuple[Path | None, str | None]:
    """Resolve mode-specific output directories derived from a checkpoint path."""
    if args.mode not in ["test", "tune", "tune-test"] or not args.checkpoint:
        return None, None

    output_base = get_output_base_from_checkpoint(args.checkpoint)
    output_base.mkdir(parents=True, exist_ok=True)
    step_suffix = extract_step_from_checkpoint(args.checkpoint)

    if args.mode in ["tune", "tune-test"]:
        if step_suffix:
            results_folder_name = f"results_{step_suffix}"
            tuning_folder_name = f"tuning_{step_suffix}"
        else:
            results_folder_name = "results"
            tuning_folder_name = "tuning"

        save_pred_cfg = cfg.inference.save_prediction
        save_pred_cfg.output_path = str(output_base / results_folder_name)
        save_pred_cfg.cache_suffix = tta_cache_suffix(cfg, checkpoint_path=args.checkpoint)

        if args.mode == "tune-test":
            print(f"Test output: {save_pred_cfg.output_path}")
            print(f"Test cache suffix: {save_pred_cfg.cache_suffix}")

        return output_base, str(output_base / tuning_folder_name)

    results_folder_name = "results"
    if step_suffix:
        results_folder_name = f"results_{step_suffix}"
        print(f"Using checkpoint {step_suffix} - output will be saved to: {results_folder_name}")

    cfg.inference.save_prediction.output_path = str(output_base / results_folder_name)
    return output_base, cfg.inference.save_prediction.output_path


def setup_runtime_directories(args: Any, cfg: Config) -> tuple[Path, Path]:
    """Create the run directory and return ``(run_dir, output_base)``."""
    output_base, dirpath = configure_checkpoint_output_paths(args, cfg)
    if output_base is not None and dirpath is not None:
        run_dir = setup_run_directory(args.mode, cfg, dirpath)
        print(f"Output base: {output_base}")
        return run_dir, output_base

    resume_checkpoint_path = None
    if args.mode == "train" and args.checkpoint and args.external_prefix is None:
        resume_checkpoint_path = args.checkpoint

    run_dir = setup_run_directory(
        args.mode,
        cfg,
        cfg.monitor.checkpoint.dirpath,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    return run_dir, run_dir.parent


__all__ = [
    "configure_checkpoint_output_paths",
    "extract_step_from_checkpoint",
    "get_output_base_from_checkpoint",
    "setup_runtime_directories",
]
