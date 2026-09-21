"""Run ABISS watershed + agglomeration on the cached Pinky test affinities.

The affinity-CC decode leaks through single surviving edges; ABISS builds a
watershed over-segmentation and then agglomerates on a region graph, which is
the standard remedy. Inference is already cached, so this is decode-only.

Two conventions have to line up with ABISS's ``ws`` binary and neither is
self-evident from the code, so both are switches here and are meant to be
picked by measurement (``--channels``, ``--roll``):

* channel order — this repo stores channel ``c`` as the edge along array axis
  ``c`` (z, y, x); ``ws`` reads an (X, Y, Z, C) Fortran volume and treats
  channel 0 as the X edge, so the triplet normally has to be reversed.
* edge storage — the ``banis`` target stores edge (i, i+1) at voxel i;
  zwatershed-family code stores it at voxel i+1. ``--roll 1`` selects the
  runner's ``--edge-storage source``, which shifts by one voxel.

Usage (PYTHONPATH must include the repo root):
    python scripts/pinky_abiss_decode.py --results-dir <test_step=NNN dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from connectomics.metrics.oracle import oracle_merge_segmentation
from pinky_test_ceiling import GT_DIR, read_h5, score

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
AFF_NAME = "raw_x1_head-aff_ch0-1-2.h5"


def load_abiss_runner(runner_path: Path):
    spec = importlib.util.spec_from_file_location("run_abiss_volume", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def volume_stem(name: str) -> str:
    return name[: -len("_image")] if name.endswith("_image") else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", default=GT_DIR, type=Path)
    parser.add_argument("--volumes", default="", type=str, help="comma-separated stems; default all")
    parser.add_argument("--channels", default="2,1,0", type=str)
    parser.add_argument("--roll", default=1, type=int, choices=(0, 1),
                        help="1 selects the runner's --edge-storage source (one-voxel shift)")
    parser.add_argument("--ws-high", default=0.90, type=float)
    parser.add_argument("--ws-low", default=0.25, type=float)
    parser.add_argument("--ws-size", default=400, type=int)
    parser.add_argument("--ws-dust", default=200, type=int)
    parser.add_argument("--merge-thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8", type=str)
    parser.add_argument("--merge-function", default=None, type=str)
    parser.add_argument("--runner", default=REPO / "scripts" / "run_abiss_volume.py", type=Path)
    parser.add_argument("--ws-binary", default=REPO / "lib" / "abiss" / "build" / "ws", type=Path)
    parser.add_argument("--save-seg", action="store_true", help="write the best-scoring segmentation per volume")
    parser.add_argument("--tag", default="abiss", type=str, help="label for the output json")
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    runner = load_abiss_runner(args.runner)
    channels = [int(c) for c in args.channels.split(",")]
    merge_thresholds = [float(v) for v in args.merge_thresholds.split(",")]
    wanted = {v for v in args.volumes.split(",") if v}

    volume_dirs = sorted(
        p for p in args.results_dir.iterdir()
        if p.is_dir() and (p / AFF_NAME).exists()
        and (not wanted or volume_stem(p.name) in wanted or p.name in wanted)
    )
    if not volume_dirs:
        raise SystemExit(f"no cached affinities under {args.results_dir}")

    print(f"channels={channels} roll={args.roll} ws_high={args.ws_high} ws_low={args.ws_low} "
          f"size={args.ws_size} dust={args.ws_dust} merge_fn={args.merge_function} "
          f"merge_thresholds={merge_thresholds}")

    rows = []
    for volume_dir in volume_dirs:
        stem = volume_stem(volume_dir.name)
        aff = read_h5(volume_dir / AFF_NAME).astype(np.float32)
        gt = read_h5(args.gt_dir / f"{stem}_label.h5")
        edge_storage = "source" if args.roll else "destination"

        start = time.time()
        segs = runner._run_abiss_ws(
            predictions_czyx=aff,
            ws_binary=Path(args.ws_binary),
            ws_high_threshold=args.ws_high,
            ws_low_threshold=args.ws_low,
            ws_size_threshold=args.ws_size,
            ws_dust_threshold=args.ws_dust,
            boundary_flags=[1, 1, 1, 1, 1, 1],
            offset=0,
            channels=channels,
            ws_merge_thresholds=merge_thresholds,
            ws_merge_function=args.merge_function,
            edge_storage=edge_storage,
        )
        if not isinstance(segs, dict):
            segs = {merge_thresholds[0]: segs}
        elapsed = time.time() - start

        print(f"\n=== {stem}  aff {aff.shape}  gt ids {int((np.unique(gt) != 0).sum())} "
              f"({elapsed:.1f}s for {len(segs)} merge thresholds) ===")
        print(f"{'mt':>6}{'frag':>9}{'ARE':>9}{'VOI':>9}{'VOIs':>9}{'VOIm':>9}"
              f"{'oARE':>9}{'oVOI':>9}{'oVOIs':>9}{'oVOIm':>9}")
        best = None
        for mt in sorted(segs):
            seg = np.ascontiguousarray(segs[mt])
            if seg.shape != gt.shape:
                print(f"[skip] mt={mt}: shape {seg.shape} vs gt {gt.shape}")
                continue
            base = score(seg, gt)
            oracle = score(oracle_merge_segmentation(seg, gt), gt)
            rows.append({"volume": stem, "merge_threshold": mt, "voxels": int(gt.size),
                         "channels": channels, "roll": args.roll,
                         "base": base, "oracle_merge": oracle})
            print(f"{mt:>6.2f}{base['n_pred']:>9}{base['are']:>9.4f}{base['voi']:>9.4f}"
                  f"{base['voi_split']:>9.4f}{base['voi_merge']:>9.4f}"
                  f"{oracle['are']:>9.4f}{oracle['voi']:>9.4f}"
                  f"{oracle['voi_split']:>9.4f}{oracle['voi_merge']:>9.4f}", flush=True)
            if best is None or base["voi"] < best[1]["voi"]:
                best = (mt, base, seg)
        if args.save_seg and best is not None:
            from connectomics.data.io import write_hdf5
            out_h5 = volume_dir / f"decoded_{args.tag}_mt{best[0]:.2f}.h5"
            write_hdf5(str(out_h5), best[2].astype(np.uint32), dataset="main")
            print(f"  saved {out_h5.name} (mt {best[0]:.2f}, VOI {best[1]['voi']:.4f})")
        del aff, gt, segs

    if not rows:
        raise SystemExit("no volumes scored")

    print("\n" + "=" * 78)
    print("voxel-weighted means across volumes")
    print(f"{'mt':>6}{'ARE':>9}{'VOI':>9}{'VOIs':>9}{'VOIm':>9}{'oARE':>9}{'oVOI':>9}")
    summary = {}
    for mt in merge_thresholds:
        subset = [r for r in rows if r["merge_threshold"] == mt]
        if not subset:
            continue
        weights = np.array([r["voxels"] for r in subset], dtype=np.float64)
        weights /= weights.sum()

        def wavg(kind: str, key: str) -> float:
            return float((np.array([r[kind][key] for r in subset]) * weights).sum())

        summary[f"{mt:.2f}"] = {
            "are": wavg("base", "are"), "voi": wavg("base", "voi"),
            "voi_split": wavg("base", "voi_split"), "voi_merge": wavg("base", "voi_merge"),
            "oracle_are": wavg("oracle_merge", "are"), "oracle_voi": wavg("oracle_merge", "voi"),
            "n_volumes": len(subset),
        }
        s = summary[f"{mt:.2f}"]
        print(f"{mt:>6.2f}{s['are']:>9.4f}{s['voi']:>9.4f}{s['voi_split']:>9.4f}"
              f"{s['voi_merge']:>9.4f}{s['oracle_are']:>9.4f}{s['oracle_voi']:>9.4f}")

    out = args.out or args.results_dir / f"{args.tag}_report.json"
    out.write_text(json.dumps({"config": vars(args) | {"runner": str(args.runner),
                                                       "ws_binary": str(args.ws_binary),
                                                       "results_dir": str(args.results_dir),
                                                       "gt_dir": str(args.gt_dir),
                                                       "out": str(out)},
                               "rows": rows, "summary": summary}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
