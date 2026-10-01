"""Re-key ABISS's nucleus-competition manifest by final segment label for the EC firewall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import reject_evaluation_path
from .postprocess import _open_volume

EMPTY = {"qualified_segment_labels": {}, "qualified_segment_owners": {}}


def firewall_from_abiss(manifest_path: Path, segmentation: str | Path) -> dict:
    payload = json.loads(manifest_path.read_text())
    if "repairs" not in payload:
        return payload
    seg = _open_volume(segmentation, fill_missing=True, bounded=False)
    labels: dict[str, dict[str, str]] = {}
    owners: dict[str, list[str]] = {}
    for repair in payload["repairs"]:
        territory = np.load(manifest_path.parent / repair["territory_file"])["territory"]
        f = int(repair["factor"])  # territory is (x, y, z), pooled by f on every axis
        x0, y0, z0, x1, y1, z1 = (int(v) for v in repair["bbox_xyz"])
        block = np.asarray(seg[x0:x1, y0:y1, z0:z1])[f // 2 :: f, f // 2 :: f, f // 2 :: f, 0]
        nucleus_of = {
            str(label): str(nucleus) for nucleus, label in repair["anchor_labels"].items()
        }
        parent = str(repair["parent_id"])
        for marker, anchor_label in repair["marker_labels"].items():
            cells = np.argwhere(territory == int(marker))
            cells = cells[(cells < block.shape[:3]).all(axis=1)]
            values = block[cells[:, 0], cells[:, 1], cells[:, 2]]
            values = values[values != 0]
            if values.size == 0:
                continue
            unique, counts = np.unique(values, return_counts=True)
            nucleus = nucleus_of[str(anchor_label)]
            labels.setdefault(parent, {})[nucleus] = str(int(unique[np.argmax(counts)]))
            owners.setdefault(parent, []).append(nucleus)
    return {"qualified_segment_labels": labels, "qualified_segment_owners": owners}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest.is_file():
        reject_evaluation_path(args.manifest)
        firewall = firewall_from_abiss(args.manifest, args.segmentation)
    else:
        print(f"nucleus firewall: no manifest at {args.manifest}; writing an empty firewall")
        firewall = EMPTY
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(firewall, indent=1))
    owners = firewall.get("qualified_segment_owners", {})
    print(f"nucleus firewall: {len(owners)} qualified segments -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
