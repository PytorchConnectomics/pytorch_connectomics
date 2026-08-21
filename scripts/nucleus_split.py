#!/usr/bin/env python3
"""Promoted entry point for the nucleus competitive-split utility.

The reusable watershed kernel lives in
``connectomics.decoding.checkpoint.kernels``.  When the historical development
driver is present this wrapper executes it verbatim, preserving its established
data-backend CLI.  ``--help`` remains available in a clean checkout.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def _fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="w2ctl")
    parser.add_argument("--param", type=Path)
    parser.add_argument("--seg-override")
    parser.add_argument("--bbox", type=int, nargs=6)
    parser.add_argument("--nuclei-h5", type=Path)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--min-nucleus-share", type=float, default=0.02)
    parser.add_argument("--block", type=int, default=252)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    legacy = Path(__file__).resolve().parents[1] / "dev/zebrafinch/nucleus_competitive_split.py"
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
