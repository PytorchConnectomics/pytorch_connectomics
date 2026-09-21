#!/usr/bin/env python3
"""Build a volume keep-mask through the importable data package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.data.keep_mask import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
