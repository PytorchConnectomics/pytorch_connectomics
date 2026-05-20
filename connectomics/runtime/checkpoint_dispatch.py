"""Checkpoint-derived runtime output dispatch helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..training.lightning.runtime import setup_run_directory
from .output_naming import format_checkpoint_dir_suffix, intermediate_prediction_cache_suffix


def get_output_base_from_checkpoint(checkpoint_path: str) -> Path:
    """Determine the output base directory from checkpoint path."""
    ckpt_path = Path(checkpoint_path)
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    for parent in ckpt_path.parents:
        if timestamp_pattern.match(parent.name):
            return parent

    return ckpt_path.parent.parent / ckpt_path.stem


def get_checkpoint_test_output_dir(checkpoint_path: str | Path | None) -> Path | None:
    """Return the checkpoint-derived ``test_<ckpt_stem>`` output directory."""
    if checkpoint_path is None:
        return None
    checkpoint_text = str(checkpoint_path).strip()
    if not checkpoint_text:
        return None
    output_base = get_output_base_from_checkpoint(checkpoint_text)
    ckpt_tag = format_checkpoint_dir_suffix(checkpoint_text)
    return output_base / f"test_{ckpt_tag}"


def extract_step_from_checkpoint(checkpoint_path: str) -> str:
    """Extract ``step=<N>`` from a checkpoint filename (legacy helper)."""
    match = re.search(r"step=(\d+)", Path(checkpoint_path).stem)
    return f"step={match.group(1)}" if match else ""


def configure_checkpoint_output_paths(args: Any, cfg: Config) -> tuple[Path | None, str | None]:
    """Resolve mode-specific output directories derived from a checkpoint path.

    Layout (formalised):

    ``<ckpt_run_dir>/test_<ckpt_stem>/``     for ``--mode test``.
    ``<ckpt_run_dir>/tune_<ckpt_stem>/``     for ``--mode tune``.
    Both for ``--mode tune-test``, with predictions cached under the
    test directory.

    The ``<ckpt_stem>`` is the sanitised stem of the checkpoint
    filename (see ``format_checkpoint_dir_suffix``). Examples:
    ``test_step=00050000``, ``test_last``, ``test_epoch=001-train_loss=0.1234``.
    Step-less checkpoints (``last.ckpt``) reuse the same directory
    across re-trains by design — the same physical ``last.ckpt`` is
    being tested.
    """
    if args.mode not in ["test", "tune", "tune-test"] or not args.checkpoint:
        return None, None

    output_base = get_output_base_from_checkpoint(args.checkpoint)
    output_base.mkdir(parents=True, exist_ok=True)
    ckpt_tag = format_checkpoint_dir_suffix(args.checkpoint)

    test_dir = output_base / f"test_{ckpt_tag}"
    tune_dir = output_base / f"tune_{ckpt_tag}"
    tune_prediction_dir = tune_dir / "predictions"

    if args.mode == "test":
        cfg.inference.save_path = str(test_dir)
        return output_base, str(test_dir)

    # tune or tune-test
    if getattr(cfg, "tune", None) is not None:
        cfg.tune.save_path = str(tune_dir)
        cfg.tune.save_predictions_path = str(tune_prediction_dir)
    cfg.inference.save_path = str(test_dir if args.mode == "tune-test" else tune_prediction_dir)
    cfg.inference.save_cache_suffix = intermediate_prediction_cache_suffix(
        cfg, checkpoint_path=args.checkpoint
    )

    if args.mode == "tune-test":
        print(f"Tuning prediction output: {tune_prediction_dir}")
        print(f"Test output: {cfg.inference.save_path}")
        print(f"Test cache suffix: {cfg.inference.save_cache_suffix}")

    return output_base, str(tune_dir)


def setup_runtime_directories(args: Any, cfg: Config) -> tuple[Path, Path]:
    """Create the run directory and return ``(run_dir, output_base)``."""
    output_base, save_path = configure_checkpoint_output_paths(args, cfg)
    if output_base is not None and save_path is not None:
        run_dir = setup_run_directory(args.mode, cfg, save_path)
        print(f"Output base: {output_base}")
        return run_dir, output_base

    resume_checkpoint_path = None
    if args.mode == "train" and args.checkpoint and args.external_prefix is None:
        resume_checkpoint_path = args.checkpoint

    run_dir = setup_run_directory(
        args.mode,
        cfg,
        cfg.save_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    return run_dir, run_dir.parent


__all__ = [
    "configure_checkpoint_output_paths",
    "extract_step_from_checkpoint",
    "get_checkpoint_test_output_dir",
    "get_output_base_from_checkpoint",
    "setup_runtime_directories",
]
