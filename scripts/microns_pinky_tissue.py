#!/usr/bin/env python3
"""Map the black (resin/out-of-section) area of MICrONS Pinky mip-0 without decoding it.

The mip-0 export stores one PNG per 4096 x 4096 tile and writes a 2-byte
`<row>_<col>.txt` placeholder wherever the tile is entirely black, so tile
occupancy is a directory listing.  Within an occupied tile the tissue fraction
is recovered from the compressed size alone: PNG spends no bytes on flat black,
so `size / max(size in that section)` tracks the true non-zero fraction to
~0.8% mean / 3% max (see --verify).  Scanning all 2176 sections takes ~3 s.

    python scripts/microns_pinky_tissue.py                   # scan + report + write JSON
    python scripts/microns_pinky_tissue.py --verify 1000      # decode one section, check the estimate

The JSON holds a (Z, row, col) tissue-fraction grid; threshold it to decide
which tiles inference should visit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/projects/weilab/dataset/microns/test/pinky/mip0")
TILE = 4096


def scan(root: Path) -> tuple[list[str], np.ndarray]:
    """Return (section names, per-tile PNG byte size as a (Z, row, col) array)."""
    zs = sorted(d.name for d in os.scandir(root) if d.is_dir())
    sizes: list[dict[tuple[int, int], int]] = []
    nrow = ncol = 0
    for z in zs:
        tiles = {}
        for e in os.scandir(root / z):
            if not e.name.endswith(".png"):
                continue                       # `.txt` placeholder == all-black tile
            r, c = (int(v) for v in e.name[:-4].split("_"))
            tiles[(r, c)] = e.stat().st_size
            nrow, ncol = max(nrow, r + 1), max(ncol, c + 1)
        sizes.append(tiles)

    out = np.zeros((len(zs), nrow, ncol), np.int64)
    for i, tiles in enumerate(sizes):
        for (r, c), s in tiles.items():
            out[i, r, c] = s
    return zs, out


def tissue_fraction(sizes: np.ndarray) -> np.ndarray:
    """Per-tile non-zero fraction, normalised by the largest tile of each section."""
    full = sizes.reshape(len(sizes), -1).max(1).astype(np.float64)
    full[full == 0] = 1.0                      # fully blank section
    return np.minimum(1.0, sizes / full[:, None, None])


def verify(root: Path, z: str, frac: np.ndarray) -> None:
    from PIL import Image                      # only needed for the check

    err, true_sum, est_sum = [], 0.0, 0.0
    for e in os.scandir(root / z):
        if not e.name.endswith(".png"):
            continue
        r, c = (int(v) for v in e.name[:-4].split("_"))
        true = float((np.asarray(Image.open(e.path)) > 0).mean())
        err.append(abs(true - frac[r, c]))
        true_sum, est_sum = true_sum + true, est_sum + frac[r, c]
    err = np.array(err)
    print(f"verify z={z}: {len(err)} tiles decoded, mean|err|={err.mean():.4f} "
          f"max|err|={err.max():.3f}; section tissue {true_sum:.1f} tiles true "
          f"vs {est_sum:.1f} estimated")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="mip-0 tile tree")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output (default <root>_tissue_tiles.json)")
    ap.add_argument("--min-frac", type=float, default=0.01,
                    help="tiles below this tissue fraction are treated as black")
    ap.add_argument("--verify", metavar="Z", default=None,
                    help="decode this section and report the estimator error")
    args = ap.parse_args()

    zs, sizes = scan(args.root)
    frac = tissue_fraction(sizes)
    keep = frac >= args.min_frac

    blank_z = [z for z, k in zip(zs, keep.any((1, 2))) if not k]
    zi = np.flatnonzero(keep.any((1, 2)))
    rows, cols = np.flatnonzero(keep.any((0, 2))), np.flatnonzero(keep.any((0, 1)))
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]

    print(f"sections: {len(zs)} total, {len(zi)} with tissue, {len(blank_z)} blank")
    if blank_z:
        print(f"  blank: {' '.join(blank_z)}")
    print(f"tile grid {frac.shape[1]} x {frac.shape[2]}; tissue tiles span "
          f"rows {r0}..{r1}, cols {c0}..{c1}")
    print(f"crop (z, y, x): [{zs[zi[0]]}..{zs[zi[-1]]}] "
          f"[{r0 * TILE}, {(r1 + 1) * TILE}) [{c0 * TILE}, {(c1 + 1) * TILE}) "
          f"-> {(r1 - r0 + 1) * TILE} x {(c1 - c0 + 1) * TILE} in yx")

    box = frac[np.ix_(zi, rows, cols)]
    print(f"inside that crop: {100 * (box < args.min_frac).mean():.1f}% of tiles are all black, "
          f"{100 * (1 - box.mean()):.1f}% of voxels are black "
          f"({100 * ((box >= args.min_frac) & (box < 0.99)).mean():.1f}% of tiles are partial)")

    if args.verify is not None:
        verify(args.root, args.verify, frac[zs.index(args.verify)])

    out = args.out or args.root.parent / f"{args.root.name}_tissue_tiles.json"
    out.write_text(json.dumps({
        "root": str(args.root),
        "tile_size": [TILE, TILE],
        "z": zs,
        "crop_tile_bbox": [int(r0), int(r1) + 1, int(c0), int(c1) + 1],
        "tissue_frac": np.round(frac, 3).tolist(),
    }))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
