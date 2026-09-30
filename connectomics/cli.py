"""Installed command-line entry point."""

from __future__ import annotations

import logging
import sys
from importlib.resources import as_file, files

from .runtime.cli import parse_args, setup_config
from .runtime.dispatch import dispatch_runtime, prepare_cli_args, suppress_nonzero_rank_stdout
from .runtime.torch_safe_globals import register_torch_safe_globals


def main() -> None:
    """Resolve the command-line configuration and dispatch its runtime stage."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    register_torch_safe_globals()
    suppress_nonzero_rank_stdout()
    args = parse_args()
    demo_resource = files("connectomics.config").joinpath("demo/minimal.yaml")
    with as_file(demo_resource) as demo_config:
        prepare_cli_args(args, demo_config)
        mode_labels = {
            "train": "Training",
            "test": "Testing",
            "tune": "Parameter Tuning",
            "tune-test": "Parameter Tuning + Testing",
        }
        mode_label = mode_labels.get(args.mode, args.mode.capitalize())
        print("\n" + "=" * 60)
        print(f"PyTorch Connectomics | {mode_label}")
        print("=" * 60)
        cfg = setup_config(args)
        dispatch_runtime(args, cfg)


__all__ = ["main"]
