#!/usr/bin/env python3
"""Run YAML-configured whole-volume morphology error correction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.decoding.error_correction.workflow import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
