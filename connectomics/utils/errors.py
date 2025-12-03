"""
Improved error messages for PyTorch Connectomics.

Provides helpful, actionable error messages for common issues.
"""

import os
import torch
from pathlib import Path
from typing import Optional


class ConnectomicsError(Exception):
    """Base class for PyTorch Connectomics errors with helpful messages."""

    def __init__(self, message: str, suggestions: Optional[list] = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with suggestions."""
        msg = f"\n{'=' * 60}\n"
        msg += f"❌ ERROR: {self.message}\n"
        msg += f"{'=' * 60}\n"

        if self.suggestions:
            msg += "\n💡 Suggested solutions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"  {i}. {suggestion}\n"

        msg += "\n📚 Documentation: https://connectomics.readthedocs.io"
        msg += "\n💬 Get help: https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w"
        msg += "\n🐛 Report bug: https://github.com/zudi-lin/pytorch_connectomics/issues\n"

        return msg


class DataNotFoundError(ConnectomicsError):
    """Error when data file is not found."""

    def __init__(self, path: str):
        message = f"Data file not found: {path}"
        suggestions = [
            f"Check if the file exists: ls {Path(path).parent}",
            f"Use absolute path instead: {Path(path).resolve()}",
            "Download tutorial data: python -m connectomics.utils.download lucchi",
            "See QUICKSTART.md for data download instructions",
        ]
        super().__init__(message, suggestions)


class CUDAOutOfMemoryError(ConnectomicsError):
    """Error when GPU runs out of memory."""

    def __init__(self, original_error: Exception):
        message = "GPU out of memory"
        suggestions = [
            "Reduce batch size: data.batch_size=1",
            "Use mixed precision: optimization.precision='16-mixed'",
            "Reduce patch size: data.patch_size=[64, 64, 64]",
            "Enable gradient accumulation: optimization.accumulate_grad_batches=4",
            "Use CPU-only: system.num_gpus=0",
        ]
        super().__init__(message, suggestions)


class ConfigurationError(ConnectomicsError):
    """Error in configuration file."""

    def __init__(self, field: str, issue: str):
        message = f"Configuration error in '{field}': {issue}"
        suggestions = [
            "Check YAML syntax (spaces, not tabs)",
            "Compare with example configs in tutorials/",
            "Validate config: python -c \"from connectomics.config import load_config; load_config('config.yaml')\"",
            "See .claude/CLAUDE.md for configuration documentation",
        ]
        super().__init__(message, suggestions)


class ModelLoadError(ConnectomicsError):
    """Error loading model checkpoint."""

    def __init__(self, checkpoint_path: str, reason: str):
        message = f"Could not load checkpoint '{checkpoint_path}': {reason}"
        suggestions = [
            f"Check if file exists: ls {checkpoint_path}",
            "Find available checkpoints: find outputs/ -name '*.ckpt'",
            "Try loading without strict matching: strict=False",
            "Re-train from scratch if checkpoint is corrupted",
        ]
        super().__init__(message, suggestions)


class DependencyError(ConnectomicsError):
    """Error with missing dependencies."""

    def __init__(self, package: str, feature: str):
        message = f"Missing dependency '{package}' required for {feature}"
        suggestions = [
            f"Install missing package: pip install {package}",
            f"Install full version: pip install -e .[full]",
            f"Check installation: python -c 'import {package}'",
        ]
        super().__init__(message, suggestions)


def handle_cuda_error(error: Exception) -> ConnectomicsError:
    """Convert CUDA errors to helpful messages."""
    error_str = str(error).lower()

    if "out of memory" in error_str:
        return CUDAOutOfMemoryError(error)
    elif "cuda" in error_str and "not available" in error_str:
        return ConnectomicsError(
            "CUDA not available",
            suggestions=[
                "Check GPU: nvidia-smi",
                "Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121",
                "Load CUDA module (HPC): module load cuda/12.1",
                "Use CPU-only: system.num_gpus=0",
            ],
        )
    elif "cudnn" in error_str:
        return ConnectomicsError(
            "cuDNN error",
            suggestions=[
                "Install cuDNN: conda install -c conda-forge cudnn",
                "Load cuDNN module (HPC): module load cudnn/8.9.0",
                "Verify cuDNN: python -c 'import torch; print(torch.backends.cudnn.is_available())'",
            ],
        )
    else:
        return ConnectomicsError(f"CUDA error: {error}", suggestions=[])


def handle_file_error(error: Exception, file_path: str) -> ConnectomicsError:
    """Convert file errors to helpful messages."""
    error_str = str(error).lower()

    if "no such file" in error_str or "not found" in error_str:
        return DataNotFoundError(file_path)
    elif "permission denied" in error_str:
        return ConnectomicsError(
            f"Permission denied: {file_path}",
            suggestions=[
                f"Check file permissions: ls -l {file_path}",
                f"Make readable: chmod +r {file_path}",
                "Run as correct user",
            ],
        )
    elif "truncated" in error_str or "corrupted" in error_str:
        return ConnectomicsError(
            f"Corrupted file: {file_path}",
            suggestions=[
                "Re-download the data file",
                f"Check file integrity: h5ls {file_path}",
                "Verify download completed successfully",
            ],
        )
    else:
        return ConnectomicsError(f"File error: {error}", suggestions=[])


def handle_training_error(error: Exception) -> ConnectomicsError:
    """Convert training errors to helpful messages."""
    error_str = str(error).lower()

    if "nan" in error_str or "inf" in error_str:
        return ConnectomicsError(
            "Training produced NaN/inf values",
            suggestions=[
                "Reduce learning rate: optimizer.lr=1e-5",
                "Enable gradient clipping: optimization.gradient_clip_val=1.0",
                "Use FP32 precision: optimization.precision='32'",
                "Enable anomaly detection: monitor.detect_anomaly=true",
                "Check data for NaN/inf values",
            ],
        )
    elif "dimension" in error_str or "shape" in error_str:
        return ConnectomicsError(
            f"Shape mismatch: {error}",
            suggestions=[
                "Check patch_size vs volume size",
                "Verify in_channels and out_channels match your data",
                "Check data shape: (batch, channels, depth, height, width)",
                "Enable debug mode: --fast-dev-run",
            ],
        )
    elif "dataloader" in error_str and "killed" in error_str:
        return ConnectomicsError(
            "DataLoader worker killed (out of memory)",
            suggestions=[
                "Reduce num_workers: system.num_workers=2",
                "Disable workers: system.num_workers=0",
                "Reduce batch size: data.batch_size=1",
                "Check system memory: free -h",
            ],
        )
    else:
        return ConnectomicsError(f"Training error: {error}", suggestions=[])


def preflight_check(cfg) -> list:
    """
    Run pre-flight checks before training.

    Args:
        cfg: Configuration object

    Returns:
        List of issues found (empty if all good)
    """
    issues = []

    # Check data files exist (supports glob patterns and lists)
    if cfg.data.train_image:
        from glob import glob

        # Handle list of files
        if isinstance(cfg.data.train_image, list):
            for img_path in cfg.data.train_image:
                if "*" in img_path or "?" in img_path:
                    matched_files = glob(img_path)
                    if not matched_files:
                        issues.append(f"❌ Training image pattern matched no files: {img_path}")
                elif not Path(img_path).exists():
                    issues.append(f"❌ Training image not found: {img_path}")
        # Handle single file/pattern
        else:
            # Check if pattern contains wildcards
            if "*" in cfg.data.train_image or "?" in cfg.data.train_image:
                # Expand glob pattern
                matched_files = glob(cfg.data.train_image)
                if not matched_files:
                    issues.append(
                        f"❌ Training image pattern matched no files: {cfg.data.train_image}"
                    )
            elif not Path(cfg.data.train_image).exists():
                issues.append(f"❌ Training image not found: {cfg.data.train_image}")

    if cfg.data.train_label:
        from glob import glob

        # Handle list of files
        if isinstance(cfg.data.train_label, list):
            for lbl_path in cfg.data.train_label:
                if "*" in lbl_path or "?" in lbl_path:
                    matched_files = glob(lbl_path)
                    if not matched_files:
                        issues.append(f"❌ Training label pattern matched no files: {lbl_path}")
                elif not Path(lbl_path).exists():
                    issues.append(f"❌ Training label not found: {lbl_path}")
        # Handle single file/pattern
        else:
            # Check if pattern contains wildcards
            if "*" in cfg.data.train_label or "?" in cfg.data.train_label:
                # Expand glob pattern
                matched_files = glob(cfg.data.train_label)
                if not matched_files:
                    issues.append(
                        f"❌ Training label pattern matched no files: {cfg.data.train_label}"
                    )
            elif not Path(cfg.data.train_label).exists():
                issues.append(f"❌ Training label not found: {cfg.data.train_label}")

    # Check GPU availability
    if cfg.system.training.num_gpus > 0 and not torch.cuda.is_available():
        issues.append(f"❌ {cfg.system.training.num_gpus} GPU(s) requested but CUDA not available")

    # Check GPU count
    if cfg.system.training.num_gpus > torch.cuda.device_count():
        issues.append(
            f"❌ {cfg.system.training.num_gpus} GPU(s) requested but only {torch.cuda.device_count()} available"
        )

    # Estimate memory requirements
    if torch.cuda.is_available() and cfg.system.training.num_gpus > 0:
        try:
            # Rough estimate: batch_size * patch_size * channels * 4 bytes * 10 (model overhead)
            import numpy as np

            patch_volume = np.prod(cfg.data.patch_size)
            estimated_gb = (
                cfg.system.training.batch_size * patch_volume * cfg.model.in_channels * 4 * 10 / 1e9
            )

            available_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            if estimated_gb > available_gb * 0.8:  # Leave 20% headroom
                issues.append(
                    f"⚠️  Estimated memory ({estimated_gb:.1f}GB) may exceed available ({available_gb:.1f}GB)"
                )
                issues.append(f"   💡 Consider reducing batch_size or patch_size")
        except Exception:
            pass  # Skip memory estimation if it fails

    # Check patch size vs expected volume size
    if cfg.data.patch_size:
        patch_size = cfg.data.patch_size
        if min(patch_size) < 16:
            issues.append(
                f"⚠️  Very small patch size: {patch_size} (may not capture enough context)"
            )
        if max(patch_size) > 256:
            issues.append(f"⚠️  Very large patch size: {patch_size} (may cause GPU OOM)")

    # Check learning rate
    if hasattr(cfg, "optimizer") and hasattr(cfg.optimizer, "lr"):
        lr = cfg.optimizer.get("lr", 1e-4)
        if lr > 1e-2:
            issues.append(f"⚠️  Learning rate very high: {lr} (may cause instability)")
        if lr < 1e-6:
            issues.append(f"⚠️  Learning rate very low: {lr} (training may be very slow)")

    return issues


def print_preflight_issues(issues: list):
    """Print preflight check issues."""
    if not issues:
        return

    print("\n" + "=" * 60)
    print("⚠️  PRE-FLIGHT CHECK WARNINGS")
    print("=" * 60)
    for issue in issues:
        print(f"  {issue}")
    print("=" * 60 + "\n")

    # Check if running in non-interactive environment (SLURM, cluster, etc.)
    import sys

    is_non_interactive = not sys.stdin.isatty() or os.environ.get("SLURM_JOB_ID") is not None

    if is_non_interactive:
        print("Non-interactive environment detected. Continuing automatically...\n")
        return

    # Ask user if they want to continue
    try:
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("❌ Aborted by user")
            exit(1)
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        exit(1)
