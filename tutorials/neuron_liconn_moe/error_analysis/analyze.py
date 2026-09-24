#!/usr/bin/env python3
"""Error analysis for the eb2 segmentation: semantic type, and done or not.

Two independent readings of every cohort segment are produced and then compared,
because they disagree in a way that is the point of the exercise:

profile
    `connectomics.metrics.unsupervised.morphology` + `.classification`, the
    existing no-GT catalog vocabulary. Its `broken_axon_candidate` can only fire
    on something already elongated >= 3, thinner than 0.35 um and at least 2 um
    long, so a short or bouton-shaped split fragment cannot be called broken by
    it at all -- it lands in `small_interior_fragment_candidate` or
    `unclassified` and its free ends go unrecorded.
skeleton
    `connectomics.metrics.unsupervised.continuity`, which decides caliber class
    from local skeleton radius and reads completeness off skeleton terminals
    against the crop faces. Neither step assumes the object is a tube.

The cross-tabulation of the two is written out as `tube_blind_spot`: segments the
skeleton calls axon-calibre and broken while the profile classifier declines to
call them axons. That population is the answer to "many broken axon terminals
which are not tubes".

Nothing here is a score. There is no ground truth for any moe volume: a free end
is a geometric candidate for a false split, not a proven one, and a `done`
segment can still be an end-to-end false merge.

    python dev/liconn_moe/analyze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

from connectomics.metrics.unsupervised.classification import (  # noqa: E402
    SegmentClassificationConfig,
    classify_segment,
)
from connectomics.metrics.unsupervised.continuity import (  # noqa: E402
    COMPLETENESS_CLASSES,
    SEMANTIC_TYPES,
    TERMINAL_SHAPES,
    ContinuityConfig,
    analyze_continuity,
)
from connectomics.metrics.unsupervised.pieces import (  # noqa: E402
    COMPOSITION_CLASSES,
    PieceConfig,
    decompose_segment,
)
from connectomics.metrics.unsupervised.morphology import (  # noqa: E402
    MORPHOLOGY_CLASSES,
    MorphologyConfig,
    analyze_morphology,
)

SCHEMA_NAME = "pytc.nogt.error_analysis"
SCHEMA_VERSION = "1.0.0"


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    """Encode unavailable measurements as null; the catalog carries no NaN."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_skeletons(path: Path) -> tuple[dict[int, tuple], dict]:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        vertex_offset = np.asarray(data["vertex_offset"], dtype=np.int64)
        edge_offset = np.asarray(data["edge_offset"], dtype=np.int64)
        vertices = np.asarray(data["vertices_um_zyx"], dtype=np.float64)
        radii = np.asarray(data["radii_um"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.int64)
        metadata = json.loads(str(data["metadata"]))
    graphs = {}
    for index, label in enumerate(labels.tolist()):
        v0, v1 = vertex_offset[index], vertex_offset[index + 1]
        e0, e1 = edge_offset[index], edge_offset[index + 1]
        graphs[label] = (vertices[v0:v1], edges[e0:e1], radii[v0:v1])
    return graphs, metadata


def free_end_payload(end) -> dict:
    return {
        "vertex_index": end.vertex_index,
        "position_um_zyx": [round(v, 5) for v in end.position_um_zyx],
        "outward_tangent_zyx": [round(v, 5) for v in end.outward_tangent_zyx],
        "radius_um": round(end.radius_um, 5),
        "distance_to_face_um": round(end.distance_to_face_um, 5),
        "nearest_face": end.nearest_face,
        "shape": end.shape,
        "head_radius_um": round(end.head_radius_um, 5) if end.head_radius_um else None,
        "shaft_radius_um": round(end.shaft_radius_um, 5) if end.shaft_radius_um else None,
        "head_ratio": round(end.head_ratio, 4) if end.head_ratio else None,
        "shaft_source": end.shaft_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeletons", type=Path, default=V.SKELETONS)
    parser.add_argument("--output", type=Path, default=V.ANALYSIS)
    parser.add_argument("--min-voxels", type=int, default=V.MIN_VOXELS)
    parser.add_argument(
        "--face-margin-um",
        type=float,
        default=0.05,
        help="slack added to a terminal's own radius before calling its end censored",
    )
    parser.add_argument(
        "--axon-max-radius-um",
        type=float,
        default=None,
        help="caliber at or below which a segment is axon_like; set it from "
        "caliber_survey.py, not from the EM-scale default",
    )
    parser.add_argument("--dendrite-min-radius-um", type=float, default=None)
    parser.add_argument("--min-branch-length-um", type=float, default=None,
                        help="skeleton length below which branch density is ignored")
    parser.add_argument("--min-branch-points", type=int, default=None)
    parser.add_argument(
        "--head-window-um",
        type=float,
        default=None,
        help="tip-side window for the terminal radius profile. Must fit inside a "
        "typical terminal branch (p50 0.16 um here) and hold >=3 skeleton samples "
        "(spacing bottoms out near 0.029 um, the voxel grid).",
    )
    parser.add_argument("--shaft-window-um", type=float, default=None)
    parser.add_argument(
        "--piece-length-um",
        type=float,
        default=0.4,
        help="target piece length for the piecewise decomposition",
    )
    parser.add_argument(
        "--reference-caliber-um",
        type=float,
        default=None,
        help="caliber a piece is thick or thin relative to. Defaults to the cohort's "
        "own median caliber, which is the typical axon here; a per-segment reference "
        "would hide a segment that is entirely one swelling.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()

    graphs, skeleton_metadata = load_skeletons(args.skeletons)
    log(f"loaded {len(graphs)} skeletons from {args.skeletons}")

    morphology_config = MorphologyConfig(
        voxel_size_um=V.SPACING_UM_ZYX, min_voxels=args.min_voxels
    )
    with h5py.File(V.SEG, "r") as handle:
        seg = np.asarray(handle["main"])
    log(f"loaded {seg.shape} {seg.dtype}; running profile morphology")
    analysis = analyze_morphology(seg, morphology_config)
    del seg
    log(f"profile morphology measured {len(analysis.records)} labels")

    gates = {
        name: value
        for name, value in (
            ("axon_max_radius_um", args.axon_max_radius_um),
            ("dendrite_min_radius_um", args.dendrite_min_radius_um),
            ("min_branch_length_um", args.min_branch_length_um),
            ("min_branch_points", args.min_branch_points),
            ("head_window_um", args.head_window_um),
            ("shaft_window_um", args.shaft_window_um),
        )
        if value is not None
    }
    continuity_config = ContinuityConfig(
        volume_extent_um=V.EXTENT_UM_ZYX, face_margin_um=args.face_margin_um, **gates
    )
    log(f"terminal profile windows: head {continuity_config.head_window_um} um, "
        f"shaft {continuity_config.shaft_window_um} um")
    log(f"continuity gates: axon<={continuity_config.axon_max_radius_um} "
        f"dendrite>={continuity_config.dendrite_min_radius_um} um, branch density needs "
        f">={continuity_config.min_branch_length_um} um and "
        f">={continuity_config.min_branch_points} branch points")
    classification_config = SegmentClassificationConfig(
        crumbs_max_voxels_exclusive=args.min_voxels
    )

    # The piece reference is the cohort's median caliber -- the typical axon on
    # this volume -- so a segment that is entirely a swelling still reads as one.
    reference_caliber = args.reference_caliber_um
    if reference_caliber is None:
        # Straight from the skeletons: per label, the length-weighted median
        # radius; then the median across labels. Running analyze_continuity twice
        # would double the cost of the whole pass for one scalar.
        measured = []
        for vertices, edges, radii in graphs.values():
            if not len(edges):
                continue
            span = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            weight = np.zeros(len(vertices))
            np.add.at(weight, edges[:, 0], span / 2)
            np.add.at(weight, edges[:, 1], span / 2)
            used = weight > 0
            if not np.any(used):
                continue
            order = np.argsort(radii[used], kind="stable")
            cumulative = np.cumsum(weight[used][order])
            pick = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
            measured.append(float(radii[used][order[min(pick, len(order) - 1)]]))
        reference_caliber = float(np.median(measured)) if measured else 0.08
    piece_config = PieceConfig(piece_length_um=args.piece_length_um)
    log(f"piece decomposition: target {args.piece_length_um} um, "
        f"reference caliber {reference_caliber:.4f} um")

    segments = []
    semantic = Counter()
    completeness = Counter()
    profile = Counter()
    cross = Counter()
    composition_counts = Counter()
    unclassified_composition = Counter()
    blind_spot = []
    free_end_total = 0
    shape_totals = Counter()
    free_end_shapes = Counter()
    for index, record in enumerate(analysis.records):
        label = int(record.label)
        graph = graphs.get(label)
        if graph is None:
            continuity = None
        else:
            continuity = analyze_continuity(
                label, graph[0], graph[1], graph[2], record.voxel_count, continuity_config
            )
        decision = classify_segment(
            record.morphology_class,
            record.voxel_count,
            arbor=continuity.arbor if continuity is not None else None,
            config=classification_config,
        )
        semantic_type = continuity.semantic_type if continuity else "unmeasured"
        completeness_class = continuity.completeness_class if continuity else "unmeasured"
        semantic[semantic_type] += 1
        completeness[completeness_class] += 1
        profile[record.morphology_class] += 1
        cross[(record.morphology_class, semantic_type, completeness_class)] += 1
        decomposition = None
        if graph is not None and continuity is not None and continuity.measured:
            retained = graph[1][list(continuity.arbor.retained_edge_indices)]
            decomposition = decompose_segment(
                label, graph[0], retained, graph[2],
                reference_caliber_um=reference_caliber, config=piece_config,
            )
            composition_counts[decomposition.composition] += 1
            if record.morphology_class == "unclassified":
                unclassified_composition[decomposition.composition] += 1
        ends = list(continuity.free_ends) if continuity else []
        free_end_total += len(ends)
        if continuity:
            shape_totals.update(continuity.terminal_shape_counts)
        free_end_shapes.update(end.shape for end in ends)

        # The population the tube-shaped rule cannot see: skeleton says
        # axon-calibre with an unexplained end, profile declines to call it an axon.
        is_blind = (
            semantic_type == "axon_like"
            and completeness_class in ("broken_one_end", "broken_multi_end", "isolated_fragment")
            and record.morphology_class
            not in ("good_axon_candidate", "broken_axon_candidate", "suspicious_axon_candidate")
        )
        if is_blind:
            blind_spot.append(label)

        segments.append(
            {
                "id": str(label),
                "voxel_count": int(record.voxel_count),
                "volume_um3": jsonable(record.volume_um3),
                "semantic_type": semantic_type,
                "done": bool(continuity.done) if continuity else False,
                "completeness_class": completeness_class,
                "free_end_count": len(ends),
                "free_ends": [free_end_payload(end) for end in ends],
                "continuity": jsonable(
                    {
                        "measured": continuity.measured,
                        "semantic_basis": continuity.semantic_basis,
                        "caliber_radius_um": continuity.caliber_radius_um,
                        "caliber_source": continuity.caliber_source,
                        "branch_points_per_um": continuity.branch_points_per_um,
                        "branch_density_usable": continuity.branch_density_usable,
                        "retained_length_um": continuity.retained_length_um,
                        "terminal_count": continuity.terminal_count,
                        "censored_end_count": continuity.censored_end_count,
                        "bouton_head_end_count": continuity.bouton_head_end_count,
                        "terminal_shape_counts": continuity.terminal_shape_counts,
                    }
                    if continuity
                    else {"measured": False}
                ),
                "pieces": jsonable(
                    {
                        "composition": decomposition.composition,
                        "reference_caliber_um": decomposition.reference_caliber_um,
                        "piece_count": decomposition.piece_count,
                        "kind_counts": decomposition.kind_counts,
                        "terminal_swelling_count": decomposition.terminal_swelling_count,
                        "interior_swelling_count": decomposition.interior_swelling_count,
                        "shaft_length_um": decomposition.shaft_length_um,
                        "swelling_length_um": decomposition.swelling_length_um,
                        "pieces": [asdict(piece) for piece in decomposition.pieces],
                    }
                )
                if decomposition is not None
                else None,
                "arbor": jsonable(asdict(continuity.arbor))
                if continuity is not None and continuity.arbor is not None
                else None,
                "profile": jsonable(asdict(record)),
                "classification": jsonable(asdict(decision)),
                "tube_blind_spot": is_blind,
                "correctness": {
                    "status": "unverified",
                    "is_correct": None,
                    "probability_correct": None,
                },
                "review": None,
            }
        )
        if (index + 1) % 2000 == 0 or index + 1 == len(analysis.records):
            log(f"combined {index + 1} / {len(analysis.records)} segments")

    incomplete = ("broken_one_end", "broken_multi_end", "isolated_fragment")
    broken = {key: 0 for key in SEMANTIC_TYPES}
    for (_, semantic_type, completeness_class), count in cross.items():
        if completeness_class in incomplete:
            broken[semantic_type] += count
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "volume": V.NAME,
            "model": "eb2 (1 GPU, effective batch 2) IST-LICONN banis+ 200k, cross-sample",
            "segmentation": str(V.SEG),
            "segmentation_sha256": skeleton_metadata["segmentation_sha256"],
            "shape_zyx": list(V.SHAPE_ZYX),
            "spacing_nm_zyx": list(V.SPACING_NM_ZYX),
            "extent_um_zyx": [round(v, 4) for v in V.EXTENT_UM_ZYX],
            "coordinate_convention": (
                "vertices and positions are (voxel_index_zyx + 0.5) * spacing_um_zyx"
            ),
            "cohort": f"labels with >= {args.min_voxels} voxels",
            "cohort_label_count": len(analysis.records),
            "coverage": jsonable(analysis.summary["coverage"]),
            "skeletonization": skeleton_metadata,
            "morphology_config": jsonable(asdict(morphology_config)),
            "continuity_config": {
                **jsonable(asdict(continuity_config)),
                "arbor": jsonable(asdict(continuity_config.arbor)),
            },
            "classification_config": {
                "crumbs_max_voxels_exclusive": classification_config.crumbs_max_voxels_exclusive,
                "backbone": jsonable(asdict(classification_config.backbone)),
            },
            "analysis_script_sha256": sha256(Path(__file__)),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        },
        "vocabularies": {
            "semantic_type": list(SEMANTIC_TYPES),
            "completeness_class": list(COMPLETENESS_CLASSES),
            "terminal_shape": list(TERMINAL_SHAPES),
            "piece_composition": list(COMPOSITION_CLASSES),
            "morphology_class": list(MORPHOLOGY_CLASSES),
        },
        "summary": {
            "radius_bins_um": jsonable(analysis.summary["radius_bins"]),
            "semantic_type_counts": dict(semantic),
            "completeness_class_counts": dict(completeness),
            "morphology_class_counts": dict(profile),
            "broken_by_semantic_type": broken,
            "free_end_total": free_end_total,
            "terminal_shape_counts_all": dict(shape_totals),
            "free_end_shape_counts": dict(free_end_shapes),
            "bouton_head_free_ends": free_end_shapes.get("bouton_head", 0),
            "tube_blind_spot_count": len(blind_spot),
            "piece_composition_counts": dict(composition_counts),
            "unclassified_piece_composition": dict(unclassified_composition),
            "piece_reference_caliber_um": reference_caliber,
            "cross_tab": [
                {
                    "morphology_class": key[0],
                    "semantic_type": key[1],
                    "completeness_class": key[2],
                    "count": count,
                }
                for key, count in sorted(cross.items(), key=lambda item: -item[1])
            ],
        },
        "limitations": [
            "No ground truth exists for any moe volume; no field here is a score.",
            "A free end is a geometric candidate for a false split, not a proven one; "
            "a real axon terminal also ends inside the crop.",
            "`done` does not certify a segment: an end-to-end false merge is `done`.",
            "Caliber gates and the face margin are provisional and grid-dependent.",
            "TEASAR forest topology is imposed by the skeletonizer, not biological.",
        ],
        "segments": segments,
    }
    temporary = args.output.with_suffix(".TMP.json")
    temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
    temporary.replace(args.output)

    log(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print("\nsemantic type")
    for key in SEMANTIC_TYPES:
        print(f"  {key:20s} {semantic.get(key, 0):7d}   broken {broken.get(key, 0):7d}")
    print("completeness")
    for key in COMPLETENESS_CLASSES:
        print(f"  {key:20s} {completeness.get(key, 0):7d}")
    print(f"free ends total        {free_end_total:7d}")
    print("free end shape")
    for key in TERMINAL_SHAPES:
        print(f"  {key:20s} {free_end_shapes.get(key, 0):7d}")
    print("piece composition")
    for key in COMPOSITION_CLASSES:
        print(f"  {key:22s} {composition_counts.get(key, 0):7d}"
              f"   of which were 'unclassified': {unclassified_composition.get(key, 0):6d}")
    print(f"tube blind spot        {len(blind_spot):7d}  "
          f"(axon-calibre and broken, but not an axon to the profile rule)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
