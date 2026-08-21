#!/usr/bin/env python3
"""Extract the physical-volume border flags needed by the GT-free junction decoder.

The complete contact graph is 12 GB.  The resolver needs border flags only for labels in its
exhaustive junction scope, so this writes a small, sorted, explicitly GT-free projection.
Keep-mask contact is retained for audit but is deliberately not used as a volume-border test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .resolve import reject_evaluation_path, sha256_file

DEV = Path(__file__).resolve().parent
DEFAULT_CONTACTS = DEV / "matchguard_contacts_full" / "contact_graph.npz"
DEFAULT_CANDIDATES = (
    DEV / "matchguard_error_correction" / "decoder_gtfree" / "contact_merge_candidates.npz"
)
DEFAULT_JUNCTIONS = (
    DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v1" / "junction_features_raw.npz"
)
DEFAULT_OUTPUT = (
    DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v2" / "boundary_inventory.npz"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--junctions", type=Path, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.contacts, args.candidates, args.junctions, args.output):
        reject_evaluation_path(path)

    with np.load(args.candidates, allow_pickle=False) as candidate:
        if not bool(candidate["gt_free"].item()):
            raise ValueError("candidate table is not explicitly GT-free")
        left = np.asarray(candidate["left"])
        right = np.asarray(candidate["right"])
    with np.load(args.junctions, allow_pickle=False) as junction:
        if not bool(junction["gt_free"].item()) or not np.all(junction["source"] == 1):
            raise ValueError("junction table is not a decoder-only GT-free scope")
        rows = np.asarray(junction["row"], dtype=np.int64)
    wanted = np.unique(np.concatenate((left[rows], right[rows])))

    with np.load(args.contacts, allow_pickle=False) as contact:
        if not bool(contact["gt_free"].item()) or not bool(contact["complete"].item()):
            raise ValueError("contact graph is not complete and explicitly GT-free")
        if not bool(contact["complete_boundary_inventory"].item()):
            raise ValueError("contact graph boundary inventory is incomplete")
        labels = np.asarray(contact["segment_label"], dtype=np.uint64)
        if not np.all(labels[:-1] < labels[1:]):
            raise ValueError("contact graph boundary labels must be unique and sorted")
        index = np.searchsorted(labels, wanted)
        valid = (index < len(labels)) & (labels[np.minimum(index, len(labels) - 1)] == wanted)
        if not np.all(valid):
            raise ValueError(f"boundary inventory lacks {int(np.count_nonzero(~valid))} labels")
        volume = np.asarray(contact["touches_volume_boundary"], dtype=bool)[index]
        keep = np.asarray(contact["touches_keep_boundary"], dtype=bool)[index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        segment_label=wanted,
        touches_volume_boundary=volume,
        touches_keep_boundary=keep,
        candidate_sha256=np.asarray(sha256_file(args.candidates)),
        junction_sha256=np.asarray(sha256_file(args.junctions)),
        selection=np.asarray("all labels in exhaustive decoder junction scope"),
        complete=np.asarray(True),
        gt_free=np.asarray(True),
    )
    print(
        f"boundary inventory -> {args.output} labels={len(wanted):,} "
        f"physical_border={int(np.count_nonzero(volume)):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
