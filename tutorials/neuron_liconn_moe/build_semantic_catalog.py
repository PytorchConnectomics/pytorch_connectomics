"""Generate initial semantic artifacts for an existing LICONN MOE analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectomics.evaluation.semantic import (  # noqa: E402
    build_semantic_catalog,
    write_semantic_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-uri", required=True)
    parser.add_argument("--large-volume-um3", type=float, default=1.0)
    parser.add_argument("--title", default="ExPID96 S1 · mip1 eb2 · initial semantic candidates")
    args = parser.parse_args()
    source = args.volume_dir / "error_analysis/error_analysis.json"
    metadata = json.loads(source.read_text())["metadata"]
    catalog = build_semantic_catalog(
        source,
        args.volume_dir / Path(metadata["segmentation"]).name,
        args.volume_dir / "label_sizes.npz",
        layer_uri=args.layer_uri,
        large_volume_um3=args.large_volume_um3,
    )
    write_semantic_artifacts(catalog, args.output, title=args.title)
    print(json.dumps(catalog["summary"], indent=2))


if __name__ == "__main__":
    main()
