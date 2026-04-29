"""Pre-flight validation checks for training runs."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np
import torch


def preflight_check(cfg) -> list:
    """Run pre-flight checks before training."""
    issues = []

    def _is_virtual_data_path(path_value: str) -> bool:
        return isinstance(path_value, str) and path_value.startswith("random://")

    def _iter_data_paths(path_value):
        if path_value is None:
            return []
        if isinstance(path_value, list):
            return path_value
        return [path_value]

    def _validate_training_paths(path_value, kind: str) -> None:
        for raw_path in _iter_data_paths(path_value):
            if _is_virtual_data_path(raw_path):
                continue
            if "*" in raw_path or "?" in raw_path:
                if not glob(raw_path):
                    issues.append(f"ERROR: Training {kind} pattern matched no files: {raw_path}")
            elif not Path(raw_path).exists():
                issues.append(f"ERROR: Training {kind} not found: {raw_path}")

    _validate_training_paths(cfg.data.train.image, "image")
    _validate_training_paths(cfg.data.train.label, "label")

    if cfg.system.num_gpus > 0 and not torch.cuda.is_available():
        issues.append(f"ERROR: {cfg.system.num_gpus} GPU(s) requested but CUDA not available")

    if cfg.system.num_gpus > torch.cuda.device_count():
        issues.append(
            f"ERROR: {cfg.system.num_gpus} GPU(s) requested but only "
            f"{torch.cuda.device_count()} available"
        )

    if torch.cuda.is_available() and cfg.system.num_gpus > 0:
        try:
            patch_volume = np.prod(cfg.data.dataloader.patch_size)
            estimated_gb = (
                cfg.data.dataloader.batch_size * patch_volume * cfg.model.in_channels * 4 * 10 / 1e9
            )
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            if estimated_gb > available_gb * 0.8:
                issues.append(
                    "WARNING: Estimated memory ({estimated:.1f}GB) may exceed available "
                    "({available:.1f}GB)".format(estimated=estimated_gb, available=available_gb)
                )
                issues.append("   Tip: Consider reducing batch_size or patch_size")
        except Exception:
            pass

    if cfg.data.dataloader.patch_size:
        patch_size = cfg.data.dataloader.patch_size
        if min(patch_size) < 16:
            issues.append(
                f"WARNING: Very small patch size: {patch_size} (may not capture enough context)"
            )
        if max(patch_size) > 256:
            issues.append(f"WARNING: Very large patch size: {patch_size} (may cause GPU OOM)")

    optimizer_cfg = getattr(getattr(cfg, "optimization", None), "optimizer", None)
    lr = getattr(optimizer_cfg, "lr", None)
    if lr is not None:
        if lr > 1e-2:
            issues.append(f"WARNING: Learning rate very high: {lr} (may cause instability)")
        if lr < 1e-6:
            issues.append(f"WARNING: Learning rate very low: {lr} (training may be very slow)")

    return issues


def print_preflight_issues(issues: list) -> None:
    """Print preflight check issues and optionally stop interactive runs."""
    if not issues:
        return

    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK WARNINGS")
    print("=" * 60)
    for issue in issues:
        print(f"  {issue}")
    print("=" * 60 + "\n")

    import sys

    is_non_interactive = not sys.stdin.isatty() or os.environ.get("SLURM_JOB_ID") is not None
    if is_non_interactive:
        print("Non-interactive environment detected. Continuing automatically...\n")
        return

    try:
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Aborted by user")
            raise SystemExit(1)
    except KeyboardInterrupt as exc:
        print("\nAborted by user")
        raise SystemExit(1) from exc


__all__ = ["preflight_check", "print_preflight_issues"]
