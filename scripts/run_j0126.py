#!/usr/bin/env python3
"""Run the shared playbook for tutorials/neuron_j0126/params.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.runtime.volume_pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(default_tutorial="neuron_j0126", job_prefix="j0126"))
