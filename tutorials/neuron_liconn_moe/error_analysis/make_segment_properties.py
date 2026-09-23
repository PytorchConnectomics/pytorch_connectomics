#!/usr/bin/env python3
"""Emit a Neuroglancer `segment_properties/` sidecar for the published seg layer.

An 83 MB catalog is a record, not something a viewer renders. Neuroglancer's
segment-properties format is the mechanism that actually puts per-segment values
in front of a reviewer: numbers become sortable and colourable, tags become
filters, so "show me every axon-calibre segment with a blunt free end" is a click
rather than a query someone has to write.

The five-class semantic catalog (`semantic/semantic_segmentation.json`, when
present) leads: `class:<name>` tags and the first word of each label, so the
viewer shows axon / dendrite / glia / blood_vessel / unclassified directly.

**Everything else here is a proxy measurement.** Every tag is therefore prefixed
`proxy:` and every numeric property is named `proxy_*`, so a value cannot appear
in the viewer's UI stripped of that label. Nothing in this file is a
segmentation-quality number; see `report.md` for the statement of limits.

Writes `segment_properties/info` locally. Uploading it, and adding the
`"segment_properties"` key to the layer's own `info`, are separate deliberate
steps — patching a published layer is not something this script does silently.

    python dev/liconn_moe/make_segment_properties.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402


def number(identifier: str, description: str, values: list[float]) -> dict:
    return {
        "id": identifier,
        "type": "number",
        "data_type": "float32",
        "description": description,
        "values": [round(float(v), 5) for v in values],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=V.ANALYSIS)
    parser.add_argument("--links", type=Path, default=V.LINKS)
    parser.add_argument("--output", type=Path, default=V.OUT / "segment_properties" / "info")
    parser.add_argument("--semantic", type=Path,
                        default=V.RUN / "semantic" / "semantic_segmentation.json")
    args = parser.parse_args()

    classes: dict[str, str] = {}
    class_tags: list[str] = []
    if args.semantic.exists():
        catalog = json.loads(args.semantic.read_text())
        classes = {r["id"]: r["semantic_class"] for r in catalog["segments"]}
        class_tags = [f"class:{c['class']}" for c in catalog["summary"]["categories"]]

    payload = json.loads(args.analysis.read_text())
    segments = sorted(payload["segments"], key=lambda s: int(s["id"]))

    proposed: set[str] = set()
    if args.links.exists():
        links = json.loads(args.links.read_text())
        for group in links.get("groups", []):
            proposed.update(group)

    ids = [s["id"] for s in segments]
    tag_list = class_tags + [
        "proxy:axon_like",
        "proxy:dendrite_like",
        "proxy:ambiguous_caliber",
        "proxy:complete",
        "proxy:broken_one_end",
        "proxy:broken_multi_end",
        "proxy:isolated_fragment",
        "proxy:axon_with_terminal",
        "proxy:terminal_only",
        "proxy:beaded",
        "proxy:shaft_only",
        "proxy:has_bouton_head_end",
        "proxy:all_ends_blunt",
        "proxy:tube_blind_spot",
        "proxy:in_link_proposal",
    ]
    tag_index = {name: i for i, name in enumerate(tag_list)}

    tags: list[list[int]] = []
    for segment in segments:
        continuity = segment["continuity"]
        shapes = [end.get("shape") for end in segment["free_ends"]]
        current = [
            tag_index[f"proxy:{segment['semantic_type']}"]
            for _ in (0,)
            if f"proxy:{segment['semantic_type']}" in tag_index
        ]
        if segment["id"] in classes:
            current.append(tag_index[f"class:{classes[segment['id']]}"])
        key = f"proxy:{segment['completeness_class']}"
        if key in tag_index:
            current.append(tag_index[key])
        pieces = segment.get("pieces") or {}
        composition = pieces.get("composition")
        if composition and f"proxy:{composition}" in tag_index:
            current.append(tag_index[f"proxy:{composition}"])
        if any(shape == "bouton_head" for shape in shapes):
            current.append(tag_index["proxy:has_bouton_head_end"])
        if shapes and all(shape == "blunt" for shape in shapes):
            current.append(tag_index["proxy:all_ends_blunt"])
        if segment.get("tube_blind_spot"):
            current.append(tag_index["proxy:tube_blind_spot"])
        if segment["id"] in proposed:
            current.append(tag_index["proxy:in_link_proposal"])
        # Neuroglancer requires the per-segment tag indices to be sorted.
        tags.append(sorted(set(current)))

    def continuity_value(segment: dict, field: str, default: float = 0.0) -> float:
        value = segment["continuity"].get(field)
        return default if value is None else float(value)

    info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ids,
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": [
                        f"{s['id']} "
                        + (f"{classes[s['id']]} · " if s["id"] in classes else "")
                        + f"{s['semantic_type'].replace('_like', '')}"
                        f"/{s['completeness_class'].replace('_', '-')}"
                        for s in segments
                    ],
                },
                {
                    "id": "description",
                    "type": "description",
                    "values": [
                        "PROXY geometry only, not a quality measurement. "
                        f"caliber {continuity_value(s, 'caliber_radius_um'):.3f} um, "
                        f"skeleton {continuity_value(s, 'retained_length_um'):.2f} um, "
                        f"{s['free_end_count']} free end(s)"
                        + (f", pieces: {(s.get('pieces') or {}).get('composition')}"
                           if (s.get("pieces") or {}).get("composition") else "")
                        + (
                            " incl. bouton head"
                            if any(e.get("shape") == "bouton_head" for e in s["free_ends"])
                            else ""
                        )
                        for s in segments
                    ],
                },
                {"id": "tags", "type": "tags", "tags": tag_list, "values": tags},
                number("proxy_free_ends", "unexplained skeleton terminals (candidates)",
                       [s["free_end_count"] for s in segments]),
                number("proxy_bouton_head_ends", "free ends with a terminal swelling",
                       [sum(1 for e in s["free_ends"] if e.get("shape") == "bouton_head")
                        for s in segments]),
                number("proxy_terminal_swellings",
                       "swelling pieces that contain a skeleton terminal",
                       [(s.get("pieces") or {}).get("terminal_swelling_count", 0)
                        for s in segments]),
                number("proxy_interior_swellings",
                       "swelling pieces with no terminal (en-passant-like)",
                       [(s.get("pieces") or {}).get("interior_swelling_count", 0)
                        for s in segments]),
                number("proxy_piece_count", "pieces the skeleton was cut into",
                       [(s.get("pieces") or {}).get("piece_count", 0) for s in segments]),
                number("proxy_caliber_um", "median skeleton radius (um)",
                       [continuity_value(s, "caliber_radius_um") for s in segments]),
                number("proxy_skeleton_length_um", "retained skeleton length (um)",
                       [continuity_value(s, "retained_length_um") for s in segments]),
                number("proxy_branch_per_um", "branch points per um of skeleton",
                       [continuity_value(s, "branch_points_per_um") for s in segments]),
                number("proxy_voxels", "voxel count",
                       [s["voxel_count"] for s in segments]),
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, allow_nan=False))
    temporary.replace(args.output)
    size = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size:.1f} MB) for {len(ids)} segments")
    print(f"tags: {', '.join(tag_list)}")
    counts = {name: sum(1 for t in tags if tag_index[name] in t) for name in tag_list}
    for name, count in counts.items():
        print(f"  {name:32s} {count:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
