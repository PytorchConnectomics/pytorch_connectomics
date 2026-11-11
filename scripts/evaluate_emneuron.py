#!/usr/bin/env python3
"""Decode EMNeuron affinities with ABISS and score them with SegNeuron's metrics.

Metrics equal the calls in SegNeuron's ``Postprocess/FRMC_post.py``::

    skimage.metrics.adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
    skimage.metrics.variation_of_information(gt, seg, ignore_labels=(0,))  # (split, merge)

computed here from one contingency table (``--check-skimage`` asserts the equality once):
voxels where the ground truth is 0 (membrane) are ignored, and VI = split + merge in bits.

``vi_old`` is the VI of the SegNeuron paper's tables. They were scored with scikit-image
0.17.2, which drops the GT-0 counts but divides the contingency table by the total voxel
count, so every VI term is scaled by the GT foreground fraction (``gt_fg_frac``):
``vi_old = vi * count(gt != 0) / gt.size`` (SegNeuron issue #8). ARE is the same in both.
Compare ``vi_old``, not ``vi``, with the published numbers.

Decoder: ABISS ws through ``decode_abiss`` on ABISS-layout affinities (channels x, y, z;
edge (i-1, i) stored at i), with no shift or reorder. One watershed per (high, low,
size, dust, merge function) runs every merge threshold in batch mode. ``ws_size``
0 keeps the watershed basins (decoded once: the merge threshold has no effect); a large
value turns the merge into single linkage at the merge threshold. Voxels ABISS leaves at 0 take the label of the nearest labeled voxel
(otherwise they would score as one giant segment); their fraction inside the GT
foreground is reported.

Writes one TSV row per (volume, GT file, parameter set).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy import ndimage

# Match scripts/main.py so the script can run from any working directory.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from connectomics.decoding.decoders.abiss import decode_abiss  # noqa: E402

ABISS_CMD = (
    "{python_exe} -m connectomics.decoding.abiss_runner --abiss-home {abiss_home} "
    "--input {input_h5} --output {output_h5}"
)
FIELDS = [
    "volume",
    "gt",
    "decoder",
    "params",
    "vi",
    "vi_split",
    "vi_merge",
    "are",
    "are_precision",
    "are_recall",
    "vi_old",
    "gt_fg_frac",
    "n_seg",
    "unlabeled_fg",
    "seconds",
]


def read_any(path: str) -> np.ndarray:
    if path.endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    with h5py.File(path, "r") as f:
        return f[list(f.keys())[0]][...]


def scores(gt: np.ndarray, seg: np.ndarray) -> dict[str, float]:
    """skimage adapted_rand_error / variation_of_information with GT label 0 ignored."""
    fg = gt > 0
    _, gi = np.unique(gt[fg], return_inverse=True)
    _, si = np.unique(seg[fg], return_inverse=True)
    _, nij = np.unique(gi.astype(np.int64) * (int(si.max()) + 1) + si, return_counts=True)
    a = np.bincount(gi).astype(np.float64)  # GT segment sizes
    b = np.bincount(si).astype(np.float64)  # predicted segment sizes
    nij = nij.astype(np.float64)
    n = float(fg.sum())
    sum_p2 = (nij**2).sum() - n
    precision = sum_p2 / ((a**2).sum() - n)  # skimage's naming
    recall = sum_p2 / ((b**2).sum() - n)
    pij, pa, pb = nij / n, a / n, b / n
    h_joint = -(pij * np.log2(pij)).sum()
    h_gt = -(pa * np.log2(pa)).sum()
    h_seg = -(pb * np.log2(pb)).sum()
    return {
        "vi": 2 * h_joint - h_gt - h_seg,
        "vi_split": h_joint - h_gt,
        "vi_merge": h_joint - h_seg,
        "are": 1.0 - 2.0 * precision * recall / (precision + recall),
        "are_precision": precision,
        "are_recall": recall,
        "vi_old": (2 * h_joint - h_gt - h_seg) * n / gt.size,  # skimage 0.17.2 normalization
        "gt_fg_frac": n / gt.size,
    }


def check_skimage(gt: np.ndarray, seg: np.ndarray) -> None:
    from skimage.metrics import adapted_rand_error, variation_of_information

    are, precision, recall = adapted_rand_error(gt, seg, ignore_labels=(0,))
    split, merge = variation_of_information(gt, seg, ignore_labels=(0,))
    ref = {
        "vi_split": split,
        "vi_merge": merge,
        "are": are,
        "are_precision": precision,
        "are_recall": recall,
    }
    mine = scores(gt, seg)
    for key, value in ref.items():
        if not np.isclose(mine[key], value, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"metric {key}: {mine[key]!r} != skimage {value!r}")
    print("skimage check ok:", {k: round(float(v), 6) for k, v in ref.items()}, flush=True)


def fill_unlabeled(seg: np.ndarray) -> np.ndarray:
    """Give every 0 voxel the label of its nearest labeled voxel."""
    zero = seg == 0
    if not zero.any():
        return seg
    if zero.all():
        return np.ones_like(seg)
    idx = ndimage.distance_transform_edt(zero, return_distances=False, return_indices=True)
    return seg[tuple(idx)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--volume", required=True, help="name written to the TSV")
    ap.add_argument("--pred", required=True, help="affinity h5/npy, (C, Z, Y, X)")
    ap.add_argument("--gt", required=True, nargs="+", help="GT label volume(s) on the same grid")
    ap.add_argument("--out", required=True, help="TSV to append to")
    ap.add_argument("--abiss-home", default=os.environ.get("PYTC_ABISS_HOME"))
    ap.add_argument(
        "--channels",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="input channels holding x, y, z affinities (default 0 1 2: ABISS layout)",
    )
    ap.add_argument(
        "--foreground",
        default=None,
        help="optional (Z, Y, X) foreground probability; affinity becomes min(aff, fg)",
    )
    ap.add_argument("--ws-high", type=float, nargs="+", default=[0.9, 0.98, 0.995])
    ap.add_argument("--ws-low", type=float, nargs="+", default=[0.05])
    ap.add_argument("--ws-size", type=int, nargs="+", default=[0, 300, 1000, 3000, 1000000000])
    ap.add_argument("--ws-dust", type=int, nargs="+", default=[0])
    ap.add_argument("--ws-merge-function", nargs="+", default=["max", "mean"])
    ap.add_argument(
        "--ws-merge-thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    ap.add_argument(
        "--check-skimage", action="store_true", help="assert equality with skimage once"
    )
    args = ap.parse_args()
    if not args.abiss_home:
        ap.error("set --abiss-home or PYTC_ABISS_HOME to an ABISS checkout with build/ws")

    aff = read_any(args.pred)
    if aff.ndim != 4 or aff.shape[0] < 3:
        raise ValueError(f"{args.pred}: expected (C>=3, Z, Y, X) affinities, got {aff.shape}")
    aff = aff[list(args.channels)].astype(np.float32)
    if args.foreground:
        aff = np.minimum(aff, read_any(args.foreground).astype(np.float32)[None])

    gts = {}
    for path in args.gt:
        gt = read_any(path)
        gt = np.where(gt < 0, 0, gt).astype(np.int64)
        if gt.shape != aff.shape[1:]:
            raise ValueError(f"{path}: GT {gt.shape} vs prediction {aff.shape[1:]}")
        gts[Path(path).stem] = gt

    new = not Path(args.out).exists()
    checked = not args.check_skimage
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    grid = itertools.product(
        args.ws_high, args.ws_low, args.ws_size, args.ws_dust, args.ws_merge_function
    )
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        for high, low, size, dust, merge_function in grid:
            start = time.time()
            cli_args = {
                "ws_high_threshold": high,
                "ws_low_threshold": low,
                "ws_size_threshold": size,
                "ws_dust_threshold": dust,
                "ws_merge_function": merge_function,
            }
            if size == 0:
                # No region merges below a zero size threshold: one result for every merge
                # threshold, recorded with merge=None.
                cli_args["ws_merge_threshold"] = args.ws_merge_thresholds[0]
                segs = {
                    None: decode_abiss(
                        aff, ABISS_CMD, abiss_home=args.abiss_home, env=env, cli_args=cli_args
                    )
                }
            else:
                cli_args["ws_merge_thresholds"] = list(args.ws_merge_thresholds)
                segs = decode_abiss(
                    aff, ABISS_CMD, abiss_home=args.abiss_home, env=env, cli_args=cli_args
                )
            seconds = (time.time() - start) / len(segs)
            for merge, seg in sorted(segs.items()):
                params = {
                    "high": high,
                    "low": low,
                    "size": size,
                    "dust": dust,
                    "merge_function": merge_function,
                    "merge": merge,
                }
                filled = fill_unlabeled(seg)
                for gt_name, gt in gts.items():
                    if not checked:
                        check_skimage(gt, filled)
                        checked = True
                    row = dict(
                        volume=args.volume,
                        gt=gt_name,
                        decoder="abiss",
                        params=json.dumps(params, sort_keys=True),
                        n_seg=int(len(np.unique(filled))),
                        unlabeled_fg=float((seg[gt > 0] == 0).mean()),
                        seconds=seconds,
                        **scores(gt, filled),
                    )
                    writer.writerow(
                        {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
                    )
                    f.flush()
            print(args.volume, params, f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
