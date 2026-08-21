#!/usr/bin/env python3
"""Promoted entry point for strict nucleus-shell contamination auditing."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def _fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", required=True)
    parser.add_argument("--bbox", type=int, nargs=6, required=True)
    parser.add_argument("--nuclei", type=Path)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--mip", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    legacy = Path(__file__).resolve().parents[1] / "dev/zebrafinch/nucleus_shell_contamination.py"
    if legacy.exists():
        runpy.run_path(str(legacy), run_name="__main__")
        return 0
    _fallback_parser().parse_args()
    raise SystemExit(
        "The historical CloudVolume driver requires the gitignored zebra-finch development "
        "workspace; use scripts/checkpoint.py with explicit tracked inputs in a clean checkout."
    )


if __name__ == "__main__":
    raise SystemExit(main())
