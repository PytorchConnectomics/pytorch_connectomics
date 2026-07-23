#!/usr/bin/env python3
"""Prepare and run vendored ABISS chunked decoding on large volumes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.runtime.abiss_large import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
