"""Score MICrONS Pinky test volumes and report the false-merge-free ceiling.

The framework's test-mode evaluation reports adapted-Rand / VOI for the decoded
segmentation. This adds the oracle-merge readout used elsewhere in the lab: every
predicted fragment is relabelled to its majority-overlap GT id, which heals all
splits an agglomerator could ever fix and leaves only the error baked into the
fragments themselves (false merges + boundary loss). The gap between the two
columns is the headroom a better decode/agglomeration can buy.

Usage (cwd must be the repo root so ``connectomics`` imports):
    python scripts/pinky_test_ceiling.py --results-dir <test_step=NNN dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from connectomics.metrics.oracle import oracle_merge_segmentation
from connectomics.metrics.segmentation_numpy import adapted_rand, voi

GT_DIR = Path("/projects/weilab/dataset/mito/microns/pinky/split/test")


def read_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        key = "main" if "main" in handle else sorted(handle.keys())[0]
        return handle[key][:]


def score(seg: np.ndarray, gt: np.ndarray) -> dict:
    are, prec, rec = adapted_rand(seg, gt, all_stats=True)
    split, merge = voi(seg, gt)
    ids = np.unique(seg)
    return {
        "are": float(are),
        "are_precision": float(prec),
        "are_recall": float(rec),
        "voi_split": float(split),
        "voi_merge": float(merge),
        "voi": float(split + merge),
        "n_pred": int((ids != 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", default=GT_DIR, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    volume_dirs = sorted(p for p in args.results_dir.iterdir() if p.is_dir())
    if not volume_dirs:
        raise SystemExit(f"no per-volume subdirectories under {args.results_dir}")

    rows = []
    for volume_dir in volume_dirs:
        decoded = sorted(volume_dir.glob("decoded_*.h5"))
        if not decoded:
            print(f"[skip] {volume_dir.name}: no decoded_*.h5")
            continue
        if len(decoded) > 1:
            print(f"[warn] {volume_dir.name}: {len(decoded)} decoded artifacts, using {decoded[-1].name}")
        seg = read_h5(decoded[-1])
        # Result dirs are named after the image stem (``pinky_vol101_image``);
        # the GT file drops that suffix (``pinky_vol101_label.h5``).
        stem = volume_dir.name[: -len("_image")] if volume_dir.name.endswith("_image") else volume_dir.name
        gt = read_h5(args.gt_dir / f"{stem}_label.h5")
        seg = np.squeeze(seg)
        if seg.shape != gt.shape:
            print(f"[skip] {volume_dir.name}: shape mismatch seg {seg.shape} vs gt {gt.shape}")
            continue

        base = score(seg, gt)
        oracle = score(oracle_merge_segmentation(seg, gt), gt)
        row = {
            "volume": volume_dir.name,
            "artifact": decoded[-1].name,
            "voxels": int(gt.size),
            "n_gt": int((np.unique(gt) != 0).sum()),
            "base": base,
            "oracle_merge": oracle,
        }
        rows.append(row)
        print(
            f"{row['volume']:<16} gt={row['n_gt']:>5} frag={base['n_pred']:>7} | "
            f"ARE {base['are']:.4f} -> oracle {oracle['are']:.4f} | "
            f"VOI {base['voi']:.4f} (s {base['voi_split']:.4f} / m {base['voi_merge']:.4f})"
            f" -> oracle {oracle['voi']:.4f} (s {oracle['voi_split']:.4f} / m {oracle['voi_merge']:.4f})",
            flush=True,
        )

    if not rows:
        raise SystemExit("no volumes scored")

    weights = np.array([r["voxels"] for r in rows], dtype=np.float64)
    weights /= weights.sum()

    def agg(kind: str, key: str) -> tuple[float, float]:
        values = np.array([r[kind][key] for r in rows], dtype=np.float64)
        return float(values.mean()), float((values * weights).sum())

    summary = {}
    print("\n" + "=" * 78)
    print(f"{'metric':<22}{'mean':>12}{'voxel-weighted':>18}")
    for kind in ("base", "oracle_merge"):
        for key in ("are", "voi", "voi_split", "voi_merge"):
            mean, weighted = agg(kind, key)
            summary[f"{kind}_{key}"] = {"mean": mean, "voxel_weighted": weighted}
            print(f"{kind + '/' + key:<22}{mean:>12.4f}{weighted:>18.4f}")

    out = args.out or args.results_dir / "test_ceiling_report.json"
    out.write_text(json.dumps({"volumes": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
