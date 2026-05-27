"""Analyze waterz segmentation vs ground truth on SNEMI test set.

Examines:
1. Segment size distributions (prediction vs GT)
2. Over/under-segmentation analysis via IoU
3. False split / false merge error counts
4. Oracle study — ARE improvement per GT segment
5. Effect of size-based dust removal on adapted rand error

Usage:
    python crackit/snemi/analyze_seg.py <pred.h5> [gt.h5]
    python crackit/snemi/analyze_seg.py  # uses default paths

See also:
    test_merge_dust.py — compares Python vs C++ merge_dust implementations
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomics.metrics.segmentation_numpy import adapted_rand

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "outputs/neuron_snemi/20260310_174424/results"
PRED_H5 = RESULTS_DIR / "test-input_x1_ch0-1-2_ckpt-epoch=095-train_loss_total_epoch=1.0395_prediction_waterz_0.1-0.999-xyz-800-0.3-600-aff85_his256-0.5.h5"
GT_H5 = ROOT / "datasets/SNEMI/test-labels.h5"


def load_vol(path, dataset="main"):
    with h5py.File(path, "r") as f:
        return f[dataset][()]


def segment_size_stats(seg, name="seg"):
    """Print segment size distribution statistics."""
    ids, counts = np.unique(seg, return_counts=True)
    fg_mask = ids > 0
    ids, counts = ids[fg_mask], counts[fg_mask]

    print(f"\n{'='*60}")
    print(f"  {name}: {len(ids)} segments")
    print(f"  Total foreground voxels: {counts.sum():,}")
    print(f"  Size stats: min={counts.min()}, median={int(np.median(counts))}, "
          f"mean={counts.mean():.0f}, max={counts.max()}")

    thresholds = [10, 50, 100, 200, 500, 1000, 5000]
    print(f"\n  Size distribution:")
    prev = 0
    for t in thresholds:
        n = np.sum((counts >= prev) & (counts < t))
        vol = counts[(counts >= prev) & (counts < t)].sum()
        print(f"    [{prev:>5d}, {t:>5d}): {n:>5d} segs, {vol:>10,} voxels")
        prev = t
    n = np.sum(counts >= prev)
    vol = counts[counts >= prev].sum()
    print(f"    [{prev:>5d},   inf): {n:>5d} segs, {vol:>10,} voxels")

    return ids, counts


def overlap_analysis(pred, gt, bb_gt=None, split_iou=0.5):
    """Analyze over/under-segmentation using seg_to_iou(gt, pred) only.

    Only runs seg_to_iou in one direction (GT→Pred, ~333 segments) which
    is fast.  Derives both false splits and false merges from this single
    pass:

    - **False merge**: multiple GT segments whose best pred match is the
      same pred ID.
    - **False split**: GT segment whose best pred match has IoU < split_iou
      (no single pred covers the majority → the GT was fragmented).
    """
    import time
    from connectomics.data.processing.bbox import compute_bbox_all
    from connectomics.data.processing.iou import seg_to_iou

    print(f"\n{'='*60}")

    # Precompute GT bounding boxes
    if bb_gt is None:
        t0 = time.time()
        bb_gt = compute_bbox_all(gt, do_count=True)
        print(f"  compute_bbox_all(gt): {time.time()-t0:.1f}s")

    # seg_to_iou(gt, pred): for each GT, find best-matching pred
    t0 = time.time()
    gt2pred = seg_to_iou(gt, pred, bb0=bb_gt)
    gt2pred = gt2pred[gt2pred[:, 0] > 0]
    n_gt = len(gt2pred)
    print(f"  seg_to_iou(gt→pred, {n_gt} segs): {time.time()-t0:.1f}s")

    # Compute IoU for each GT's best match
    denom = (gt2pred[:, 2] + gt2pred[:, 3] - gt2pred[:, 4]).astype(np.float64)
    denom[denom == 0] = 1
    iou_vals = gt2pred[:, 4].astype(np.float64) / denom

    # --- IoU distribution ---
    print(f"\n  GT → best Pred match (IoU distribution):")
    print(f"    min={iou_vals.min():.3f}, median={np.median(iou_vals):.3f}, "
          f"mean={iou_vals.mean():.3f}, max={iou_vals.max():.3f}")
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        n = np.sum(iou_vals >= t)
        print(f"    IoU >= {t:.1f}: {n:>4d} / {n_gt}  ({100*n/max(n_gt,1):.1f}%)")

    # --- False merges: multiple GT → same pred ---
    pred_to_gts = defaultdict(list)
    for i in range(n_gt):
        p_id = int(gt2pred[i, 1])
        if p_id > 0:
            pred_to_gts[p_id].append(
                (int(gt2pred[i, 0]), float(iou_vals[i]), int(gt2pred[i, 2]))
            )

    merge_groups = []
    merged_gt_ids = set()
    for p_id, gt_list in pred_to_gts.items():
        if len(gt_list) > 1:
            merge_groups.append((p_id, gt_list))
            merged_gt_ids.update(g for g, _, _ in gt_list)

    print(f"\n  FALSE MERGES (multiple GT → one pred):")
    print(f"    Merge groups:       {len(merge_groups)}")
    print(f"    GT segments affected: {len(merged_gt_ids)}")
    for i, (p_id, gt_list) in enumerate(
        sorted(merge_groups, key=lambda x: -len(x[1]))[:10]
    ):
        gt_list = sorted(gt_list, key=lambda x: -x[1])  # sort by IoU descending
        gt_info = ", ".join(
            f"GT {g}({sz:,}px, iou={iou:.2f})" for g, iou, sz in gt_list
        )
        print(f"    [{i+1}] pred {p_id} ← {gt_info}")
    if len(merge_groups) > 10:
        print(f"    ... and {len(merge_groups) - 10} more groups")

    # --- False splits: GT with best-match IoU < split_iou ---
    split_mask = iou_vals < split_iou
    split_gt = gt2pred[split_mask]
    split_ious = iou_vals[split_mask]

    print(f"\n  FALSE SPLITS (GT best-match IoU < {split_iou}):")
    print(f"    Split GT segments:  {len(split_gt)} / {n_gt}")
    if len(split_gt) > 0:
        print(f"    IoU of splits: min={split_ious.min():.3f}, "
              f"median={np.median(split_ious):.3f}, max={split_ious.max():.3f}")
        # Show worst splits (lowest IoU)
        order = np.argsort(split_ious)
        for i in range(min(10, len(order))):
            idx = order[i]
            g_id = int(split_gt[idx, 0])
            g_sz = int(split_gt[idx, 2])
            best_p = int(split_gt[idx, 1])
            print(f"    [{i+1}] GT {g_id} ({g_sz:,}px) → best pred {best_p}, IoU={split_ious[idx]:.3f}")
        if len(order) > 10:
            print(f"    ... and {len(order) - 10} more")

    print(f"{'='*60}")

    return bb_gt


def count_split_merge_errors(pred, gt):
    """Count false splits and false merges via overlap matrix.

    Builds the full (gt_id, pred_id) → overlap_count co-occurrence
    matrix in one ``np.unique`` pass, then analyses it in both
    directions:

    * **False merges** — multiple GT segments whose best pred match is
      the same pred segment (those GT neurons got merged).
    * **False splits** — multiple pred segments whose best GT match is
      the same GT segment (that GT neuron got fragmented).

    Parameters
    ----------
    pred, gt : ndarray, shape (Z, Y, X)
        Predicted and ground-truth segmentations (0 = background).

    Returns
    -------
    dict with keys:
        n_gt, n_pred, false_split_gt, false_merge_gt,
        split_details, merge_groups
    """
    fg = (pred > 0) & (gt > 0)
    if not fg.any():
        return {"n_gt": 0, "n_pred": 0, "false_split_gt": 0,
                "false_merge_gt": 0, "split_details": [], "merge_groups": []}

    gt_fg = gt[fg].ravel()
    pred_fg = pred[fg].ravel()

    # One-pass co-occurrence: (gt_id, pred_id) → overlap count
    pairs = np.stack([gt_fg, pred_fg], axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)

    # Segment sizes
    gt_ids, gt_counts = np.unique(gt[gt > 0], return_counts=True)
    gt_size = dict(zip(gt_ids.tolist(), gt_counts.tolist()))
    pred_ids, pred_counts = np.unique(pred[pred > 0], return_counts=True)
    pred_size = dict(zip(pred_ids.tolist(), pred_counts.tolist()))

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    # --- Direction 1: GT→Pred (for false merges) ---
    # For each GT, find best-matching pred (by overlap)
    gt_best_pred = {}  # gt_id → (pred_id, overlap)
    for i in range(len(unique_pairs)):
        g, p = int(unique_pairs[i, 0]), int(unique_pairs[i, 1])
        c = int(pair_counts[i])
        if g not in gt_best_pred or c > gt_best_pred[g][1]:
            gt_best_pred[g] = (p, c)

    # False merges: multiple GT → same pred
    pred_to_gts = defaultdict(list)
    for g_id, (p_id, ovl) in gt_best_pred.items():
        g_sz = gt_size.get(g_id, 0)
        p_sz = pred_size.get(p_id, 0)
        denom = g_sz + p_sz - ovl
        iou = ovl / max(denom, 1)
        pred_to_gts[p_id].append((g_id, iou, g_sz))

    merge_groups = []
    merged_gt_ids = set()
    for p_id, gt_list in pred_to_gts.items():
        if len(gt_list) > 1:
            merge_groups.append((p_id, gt_list))
            merged_gt_ids.update(g for g, _, _ in gt_list)

    # --- Direction 2: Pred→GT (for false splits) ---
    # For each pred, find best-matching GT (by overlap)
    pred_best_gt = {}  # pred_id → (gt_id, overlap)
    for i in range(len(unique_pairs)):
        g, p = int(unique_pairs[i, 0]), int(unique_pairs[i, 1])
        c = int(pair_counts[i])
        if p not in pred_best_gt or c > pred_best_gt[p][1]:
            pred_best_gt[p] = (g, c)

    # False splits: multiple pred → same GT
    gt_to_preds = defaultdict(list)
    for p_id, (g_id, ovl) in pred_best_gt.items():
        gt_to_preds[g_id].append((p_id, pred_size.get(p_id, 0)))

    split_details = []
    for g_id, pred_list in gt_to_preds.items():
        if len(pred_list) > 1:
            split_details.append((g_id, len(pred_list), pred_list))

    # --- Print report ---
    print(f"\n{'='*60}")
    print(f"  Split/Merge Error Analysis")
    print(f"  GT segments:          {n_gt}")
    print(f"  Pred segments:        {n_pred}")

    # False merges
    print(f"\n  FALSE MERGES (multiple GT → one pred):")
    print(f"    Merge groups:       {len(merge_groups)}")
    print(f"    GT segments affected: {len(merged_gt_ids)}")
    for i, (p_id, gt_list) in enumerate(
        sorted(merge_groups, key=lambda x: -len(x[1]))[:10]
    ):
        gt_list = sorted(gt_list, key=lambda x: -x[1])  # sort by IoU descending
        gt_info = ", ".join(
            f"GT {g}({sz:,}px, iou={iou:.2f})" for g, iou, sz in gt_list
        )
        print(f"    [{i+1}] pred {p_id} ← {gt_info}")
    if len(merge_groups) > 10:
        print(f"    ... and {len(merge_groups) - 10} more groups")

    # False splits
    print(f"\n  FALSE SPLITS (one GT → multiple pred):")
    print(f"    Split GT segments:  {len(split_details)}")
    if split_details:
        n_frags = [n for _, n, _ in split_details]
        print(f"    Fragments per GT: min={min(n_frags)}, "
              f"median={int(np.median(n_frags))}, max={max(n_frags)}")
        total_extra = sum(n - 1 for n in n_frags)
        print(f"    Total extra fragments: {total_extra}")
        for i, (g_id, n, _) in enumerate(
            sorted(split_details, key=lambda x: -x[1])[:10]
        ):
            print(f"    [{i+1}] GT {g_id} → {n} pred fragments")
        if len(split_details) > 10:
            print(f"    ... and {len(split_details) - 10} more")

    print(f"{'='*60}")

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "false_split_gt": len(split_details),
        "false_merge_gt": len(merged_gt_ids),
        "split_details": split_details,
        "merge_groups": merge_groups,
    }


def oracle_study(pred, gt, top_k=20):
    """Oracle study: ARE improvement if each GT segment is perfectly predicted.

    Uses incremental contingency table update — builds the table once,
    then for each GT segment updates only the affected row.  O(nnz_per_row)
    per GT instead of O(volume).
    """
    from connectomics.metrics.segmentation_numpy import adapted_rand_oracle

    gt_ids, gt_counts = np.unique(gt[gt > 0], return_counts=True)
    gt_size = dict(zip(gt_ids.tolist(), gt_counts.tolist()))

    print(f"\n{'='*60}")
    print(f"  Oracle Study ({len(gt_ids)} GT segments, incremental)")

    results, are_base = adapted_rand_oracle(pred, gt, gt_ids=gt_ids)

    print(f"  Baseline ARE: {are_base:.6f}")
    print(f"  {'#':>4s}  {'GT':>6s}  {'Size':>10s}  {'ARE':>10s}  {'delta':>10s}")
    print(f"  {'-'*46}")

    for rank, (g_id, are_o, delta) in enumerate(results[:top_k]):
        sz = gt_size.get(g_id, 0)
        print(f"  {rank+1:>3d}.  {g_id:>6d}  {sz:>10,}  {are_o:>10.6f}  {delta:>+10.6f}")
    if len(results) > top_k:
        print(f"  ...  {len(results) - top_k} more")

    print(f"{'='*60}")
    return results


def dust_removal_sweep(pred, gt):
    """Sweep dust size thresholds and measure adapted rand error."""
    from skimage.morphology import remove_small_objects

    print(f"\n{'='*60}")
    print(f"  Dust removal sweep (set small segs to background):")
    print(f"  {'Threshold':>10s}  {'Segments':>8s}  {'ARE':>8s}  {'Prec':>8s}  {'Rec':>8s}")
    print(f"  {'-'*46}")

    are, prec, rec = adapted_rand(pred, gt, all_stats=True)
    n_segs = len(np.unique(pred)) - 1
    print(f"  {'0 (base)':>10s}  {n_segs:>8d}  {are:>8.4f}  {prec:>8.4f}  {rec:>8.4f}")

    best_are = are
    best_thresh = 0

    for thresh in [50, 100, 200, 500, 1000, 2000, 5000]:
        cleaned = remove_small_objects(pred, min_size=thresh)
        n_segs = len(np.unique(cleaned)) - 1
        are_t, prec_t, rec_t = adapted_rand(cleaned, gt, all_stats=True)
        marker = " *" if are_t < best_are else ""
        print(f"  {thresh:>10d}  {n_segs:>8d}  {are_t:>8.4f}  {prec_t:>8.4f}  {rec_t:>8.4f}{marker}")
        if are_t < best_are:
            best_are = are_t
            best_thresh = thresh

    print(f"\n  Best: threshold={best_thresh}, ARE={best_are:.4f}")
    return best_thresh, best_are


def main():
    parser = argparse.ArgumentParser(description="Analyze segmentation vs ground truth")
    parser.add_argument("pred", nargs="?", default=str(PRED_H5), help="Prediction H5 file")
    parser.add_argument("gt", nargs="?", default=str(GT_H5), help="Ground truth H5 file")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    print(f"Loading prediction: {pred_path}")
    pred = load_vol(pred_path)
    print(f"Loading ground truth: {gt_path}")
    gt = load_vol(gt_path)

    print(f"\nPrediction: shape={pred.shape}, dtype={pred.dtype}")
    print(f"Ground truth: shape={gt.shape}, dtype={gt.dtype}")

    if pred.shape != gt.shape:
        common = tuple(min(p, g) for p, g in zip(pred.shape, gt.shape))
        pred = pred[: common[0], : common[1], : common[2]]
        gt = gt[: common[0], : common[1], : common[2]]
        print(f"Cropped to common shape: {common}")

    # 1. Size distributions
    segment_size_stats(pred, "Prediction")
    segment_size_stats(gt, "Ground Truth")

    # 2. Baseline metric
    print(f"\n{'='*60}")
    are, prec, rec = adapted_rand(pred, gt, all_stats=True)
    print(f"  Baseline Adapted Rand Error: {are:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}")

    # 3. Overlap analysis (only seg_to_iou(gt, pred) — fast)
    bb_gt = overlap_analysis(pred, gt)

    # 4. Split/merge error counts (direct overlap matrix)
    error_info = count_split_merge_errors(pred, gt)

    # 5. Oracle study
    oracle_study(pred, gt)

    # 6. Simple dust removal
    best_dust_thresh, best_dust_are = dust_removal_sweep(pred, gt)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Baseline ARE:          {are:.4f}  (Prec={prec:.4f} Rec={rec:.4f})")
    print(f"  Best dust removal:     {best_dust_are:.4f}  (thresh={best_dust_thresh})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
