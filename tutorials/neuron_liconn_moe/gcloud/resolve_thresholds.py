#!/usr/bin/env python3
"""Resolve ABISS percentile thresholds once on the canonical affinity artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_abiss_volume import _read_array, _resolve_threshold


def resolve(source: Path) -> dict[str, float | str]:
    """Resolve exact percentiles; callers must provision RAM for one artifact."""
    # aff_canon.h5 is already the ABISS XYZC probability tensor serialized as
    # CZYX. Applying the whole-volume layout conversion here would shift it a
    # second time and produce thresholds for a different artifact.
    affinity = _read_array(source, "main")
    return {
        "source": str(source.resolve()),
        "ws_high_percentile": "94%",
        "ws_low_percentile": "20%",
        "ws_high": _resolve_threshold("94%", affinity, "ws_high_threshold"),
        "ws_low": _resolve_threshold("20%", affinity, "ws_low_threshold"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = resolve(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
