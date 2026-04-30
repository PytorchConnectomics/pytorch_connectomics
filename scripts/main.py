#!/usr/bin/env python3
"""PyTorch Connectomics command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for direct script execution.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from connectomics.runtime.cli import parse_args, setup_config  # noqa: E402
from connectomics.runtime.dispatch import (  # noqa: E402
    dispatch_runtime,
    prepare_cli_args,
    suppress_nonzero_rank_stdout,
)
from connectomics.runtime.torch_safe_globals import register_torch_safe_globals  # noqa: E402

register_torch_safe_globals()


def main() -> None:
    """Parse CLI options, resolve config, and dispatch the requested runtime mode."""
    suppress_nonzero_rank_stdout()
    args = parse_args()
    prepare_cli_args(args, REPO_ROOT)

    print("\n" + "=" * 60)
    print("PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)

    dispatch_runtime(args, cfg)


if __name__ == "__main__":
    main()
