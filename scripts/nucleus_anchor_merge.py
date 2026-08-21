#!/usr/bin/env python3
"""Promoted entry point for the exclusion-then-anchor legacy utility."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def _fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="w2ctl")
    parser.add_argument("--in-name", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--min-nucleus-share", type=float, default=0.02)
    parser.add_argument("--block", type=int, default=252)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    legacy = Path(__file__).resolve().parents[1] / "dev/zebrafinch/nucleus_anchor_merge.py"
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
