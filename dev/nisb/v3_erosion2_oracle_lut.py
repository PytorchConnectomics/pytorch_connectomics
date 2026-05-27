"""LUT-level oracle probes for NISB v3 erosion2.

This script samples a decoded segmentation at every ERL skeleton node once,
saves the prediction and GT/skeleton LUTs, then evaluates missing-node fill
oracles directly from those LUTs. The expensive H5 sampling step is skipped on
later runs unless ``--force`` is set.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connectomics.metrics.nerl import (
    NerlGraphOptions,
    extract_nerl_score_outputs,
    import_em_erl,
    load_nerl_graph,
)

RUN_DIR = Path("outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101")
DEFAULT_SEG = RUN_DIR / "decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66.h5"
DEFAULT_SKELETON = Path("/projects/weilab/dataset/nisb/base/test/seed101/skeleton.pkl")
DEFAULT_OUT_DIR = RUN_DIR / "seg_fusion/oracle_lut"

GRAPH_FIELDS = (
    "skeleton_id",
    "skeleton_len",
    "node_skeleton_index",
    "node_coords_zyx",
    "edge_u",
    "edge_v",
    "edge_len",
    "edge_ptr",
)


@dataclass(frozen=True)
class ScoreSummary:
    name: str
    nerl: float
    pred_erl: float
    gt_erl: float
    num_skeletons: int
    missing_nodes: int
    total_nodes: int
    missing_ratio: float
    skeletons_with_missing: int
    omitted_edges: int
    omitted_edge_len: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seg", type=Path, default=DEFAULT_SEG)
    ap.add_argument("--dataset", default="main")
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKELETON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--chunk-num", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--merge-threshold", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def tag_from_seg(path: Path) -> str:
    tag = path.stem
    for old, new in (
        ("decoded_x1_", ""),
        ("affinity_cc_numba-0-", "cc"),
    ):
        tag = tag.replace(old, new)
    return tag


def load_graph(skeleton: Path):
    graph, voxel_coords = load_nerl_graph(
        skeleton,
        NerlGraphOptions(
            skeleton_id_attribute="id",
            skeleton_position_attribute="index_position",
            skeleton_edge_length_attribute="edge_length",
            skeleton_position_order="xyz",
        ),
    )
    node_positions = np.asarray(
        graph.node_coords_zyx if voxel_coords else graph.get_nodes_position(None),
        dtype=np.int64,
    )
    return graph, node_positions


def cache_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"{tag}_node_luts.npz"


def build_or_load_cache(args: argparse.Namespace, tag: str):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(out_dir, tag)
    ERLGraph, _, compute_segment_lut = import_em_erl()

    if path.exists() and not args.force:
        data = np.load(path, allow_pickle=False)
        graph = ERLGraph(**{name: data[name] for name in GRAPH_FIELDS})
        return path, graph, {key: data[key] for key in data.files}

    graph, node_positions = load_graph(args.skeleton)
    pred_lut, _mask_segment_id = compute_segment_lut(
        str(args.seg),
        node_positions,
        chunk_num=int(args.chunk_num),
        data_type=np.uint32,
        segment_dataset=args.dataset,
        num_workers=int(args.num_workers),
    )
    pred_lut = np.asarray(pred_lut, dtype=np.uint64)
    node_skeleton_index = np.asarray(graph.node_skeleton_index, dtype=np.int64)
    gt_lut_raw = np.asarray(graph.skeleton_id[node_skeleton_index], dtype=np.uint64)

    # Keep an explicit collision-safe GT label space for oracle edits. Raw GT
    # IDs can overlap low predicted segment IDs, which would create artificial
    # merges if mixed into a prediction LUT.
    gt_label_offset = np.uint64(int(pred_lut.max(initial=0)) + 1)
    gt_lut_safe = gt_label_offset + node_skeleton_index.astype(np.uint64)

    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.uint8),
        pred_lut=pred_lut,
        gt_lut_raw=gt_lut_raw,
        gt_lut_safe=gt_lut_safe,
        gt_label_offset=np.asarray(gt_label_offset, dtype=np.uint64),
        node_positions_zyx=node_positions.astype(np.int64, copy=False),
        source_seg=np.asarray(str(args.seg)),
        source_skeleton=np.asarray(str(args.skeleton)),
        source_dataset=np.asarray(str(args.dataset)),
        **{name: getattr(graph, name) for name in GRAPH_FIELDS},
    )
    data = np.load(path, allow_pickle=False)
    return path, graph, {key: data[key] for key in data.files}


def score_lut(
    graph, lut: np.ndarray, *, name: str, merge_threshold: int
) -> tuple[ScoreSummary, np.ndarray]:
    _, compute_erl_score, _ = import_em_erl()
    lut = np.asarray(lut)
    score = compute_erl_score(
        graph,
        lut,
        mask_segment_id=None,
        merge_threshold=int(merge_threshold),
    )
    score.compute_erl()
    pred_erl, gt_erl, num_skeletons, per_gt_erl = extract_nerl_score_outputs(score)
    missing_mask = lut == 0
    total_nodes = int(lut.size)
    missing_nodes = int(np.count_nonzero(missing_mask))
    node_skeleton_index = np.asarray(graph.node_skeleton_index, dtype=np.int64)
    per_missing = np.bincount(
        node_skeleton_index,
        weights=missing_mask.astype(np.int64),
        minlength=len(graph.skeleton_id),
    ).astype(np.int64)

    edge_u = np.asarray(graph.edge_u, dtype=np.int64)
    edge_v = np.asarray(graph.edge_v, dtype=np.int64)
    edge_omitted = (lut[edge_u] == 0) | (lut[edge_v] == 0)
    summary = ScoreSummary(
        name=name,
        nerl=float(pred_erl / gt_erl if gt_erl > 0 else np.nan),
        pred_erl=float(pred_erl),
        gt_erl=float(gt_erl),
        num_skeletons=int(num_skeletons),
        missing_nodes=missing_nodes,
        total_nodes=total_nodes,
        missing_ratio=float(0.0 if total_nodes == 0 else missing_nodes / total_nodes),
        skeletons_with_missing=int(np.count_nonzero(per_missing)),
        omitted_edges=int(np.count_nonzero(edge_omitted)),
        omitted_edge_len=float(np.asarray(graph.edge_len, dtype=np.float64)[edge_omitted].sum()),
    )
    return summary, per_gt_erl


def fill_missing_with_dominant_pred(
    graph, pred_lut: np.ndarray, gt_lut_safe: np.ndarray
) -> np.ndarray:
    filled = np.asarray(pred_lut, dtype=np.uint64).copy()
    node_skeleton_index = np.asarray(graph.node_skeleton_index, dtype=np.int64)
    for skel_index in range(len(graph.skeleton_id)):
        skel_mask = node_skeleton_index == skel_index
        missing = skel_mask & (filled == 0)
        if not np.any(missing):
            continue
        nonzero = filled[skel_mask & (filled != 0)]
        if nonzero.size:
            labels, counts = np.unique(nonzero, return_counts=True)
            fill_label = labels[np.argmax(counts)]
        else:
            fill_label = gt_lut_safe[missing][0]
        filled[missing] = fill_label
    return filled


def write_summary_csv(path: Path, rows: list[ScoreSummary]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ScoreSummary.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_per_gt_missing_csv(
    path: Path,
    graph,
    pred_lut: np.ndarray,
    per_gt_by_name: dict[str, np.ndarray],
) -> None:
    node_skeleton_index = np.asarray(graph.node_skeleton_index, dtype=np.int64)
    per_total = np.bincount(node_skeleton_index, minlength=len(graph.skeleton_id)).astype(np.int64)
    per_missing = np.bincount(
        node_skeleton_index,
        weights=(np.asarray(pred_lut) == 0).astype(np.int64),
        minlength=len(graph.skeleton_id),
    ).astype(np.int64)
    fields = [
        "gt_segment_id",
        "total_nodes",
        "missing_nodes",
        "missing_node_ratio",
    ]
    for name in per_gt_by_name:
        fields.extend([f"{name}_pred_erl", f"{name}_gt_erl", f"{name}_nerl"])

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, gt_id in enumerate(graph.skeleton_id):
            row: dict[str, object] = {
                "gt_segment_id": int(gt_id),
                "total_nodes": int(per_total[i]),
                "missing_nodes": int(per_missing[i]),
                "missing_node_ratio": float(
                    0.0 if per_total[i] == 0 else per_missing[i] / per_total[i]
                ),
            }
            for name, per_gt in per_gt_by_name.items():
                pred_erl = float(per_gt[i, 0])
                gt_erl = float(per_gt[i, 1])
                row[f"{name}_pred_erl"] = pred_erl
                row[f"{name}_gt_erl"] = gt_erl
                row[f"{name}_nerl"] = float(pred_erl / gt_erl if gt_erl > 0 else np.nan)
            writer.writerow(row)


def write_report(path: Path, cache: Path, rows: list[ScoreSummary]) -> None:
    lines = [
        "# v3_erosion2 Oracle LUT Results",
        "",
        f"LUT cache: `{cache}`",
        "",
        "| Variant | NERL | Pred ERL | Missing nodes | Omitted edges | Omitted edge len |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.nerl:.6f} | {row.pred_erl:.6f} | "
            f"{row.missing_nodes}/{row.total_nodes} ({row.missing_ratio:.6f}) | "
            f"{row.omitted_edges} | {row.omitted_edge_len:.3f} |"
        )
    lines.extend(
        [
            "",
            "Oracle variants:",
            "",
            "- `zero_to_gt_safe`: only prediction-0 skeleton nodes are replaced with",
            "  collision-safe GT/skeleton labels. This is the literal GT-identity",
            "  LUT edit and avoids accidental label collisions with predicted IDs.",
            "- `zero_to_dominant_pred`: prediction-0 skeleton nodes are replaced",
            "  with the dominant nonzero predicted label on the same GT skeleton,",
            "  falling back to the collision-safe GT label for fully missing",
            "  skeletons. This is closer to a nearest-label fill oracle.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    tag = args.tag or tag_from_seg(args.seg)
    cache, graph, data = build_or_load_cache(args, tag)

    pred_lut = np.asarray(data["pred_lut"], dtype=np.uint64)
    gt_lut_safe = np.asarray(data["gt_lut_safe"], dtype=np.uint64)

    luts = {
        "baseline": pred_lut,
        "zero_to_gt_safe": np.where(pred_lut == 0, gt_lut_safe, pred_lut),
        "zero_to_dominant_pred": fill_missing_with_dominant_pred(graph, pred_lut, gt_lut_safe),
    }

    summaries: list[ScoreSummary] = []
    per_gt_by_name: dict[str, np.ndarray] = {}
    for name, lut in luts.items():
        summary, per_gt = score_lut(
            graph,
            lut,
            name=name,
            merge_threshold=int(args.merge_threshold),
        )
        summaries.append(summary)
        per_gt_by_name[name] = per_gt

    summary_csv = args.out_dir / f"{tag}_oracle_summary.csv"
    per_gt_csv = args.out_dir / f"{tag}_oracle_per_gt.csv"
    report_md = args.out_dir / f"{tag}_oracle_report.md"
    write_summary_csv(summary_csv, summaries)
    write_per_gt_missing_csv(per_gt_csv, graph, pred_lut, per_gt_by_name)
    write_report(report_md, cache, summaries)

    for row in summaries:
        print(
            f"{row.name}: nerl={row.nerl:.6f} pred_erl={row.pred_erl:.6f} "
            f"missing={row.missing_nodes}/{row.total_nodes} "
            f"omitted_edges={row.omitted_edges}",
            flush=True,
        )
    print(f"cache: {cache}", flush=True)
    print(f"summary: {summary_csv}", flush=True)
    print(f"per_gt: {per_gt_csv}", flush=True)
    print(f"report: {report_md}", flush=True)


if __name__ == "__main__":
    main()
