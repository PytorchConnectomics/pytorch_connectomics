"""Build a volume-weighted, provisional semantic catalog from an existing analysis.

This reporting adapter preserves every positive label, including objects below
the morphology cutoff. Biological classes remain candidates until human review.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..metrics.unsupervised.classification import classify_semantic_candidate

__all__ = ["build_semantic_catalog", "summarize_semantic_records", "write_semantic_artifacts"]

_CATEGORIES = {
    "blood_vessel": "Blood vessel (review required)",
    "glia": "Glia (review required)",
    "dendrite": "Dendrite candidate (neurons with soma included)",
    "axon": "Axon candidate (terminals included)",
    "unclassified": "Unclassified",
}


# Below this skeleton length a thick object is a blob: a detached axon terminal
# as easily as a dendrite piece. On ExPID96 S1 this is 18 of 128 thick calls,
# 1.3% of their volume, each ~0.2 um3 -- bouton-sized, far below a soma.
_MIN_DENDRITE_LENGTH_UM = 2.0
# An axon call needs 3 um of skeleton OR 0.1 um3 (a thin process can be long with
# little volume). Short AND small pieces -- often false splits -- are penalized as
# unclassified, so what reads as `axon` is a substantial piece. Pooled over the 12
# eb2 volumes the median axon call was 0.037 um3 and 1.66 um long.
_MIN_AXON_LENGTH_UM = 3.0
_MIN_AXON_VOLUME_UM3 = 0.1


def _digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_semantic_records(
    records: list[dict[str, Any]], shape: list[int] | tuple[int, ...], voxel_volume: float
) -> dict[str, Any]:
    """Summarize initial candidate assignments with foreground and ROI denominators."""
    total = int(np.prod(shape))
    foreground = sum(r["voxel_count"] for r in records)
    if not 0 < foreground <= total:
        raise ValueError("Foreground must be positive and within the ROI")
    categories = []
    for key, title in _CATEGORIES.items():
        selected = [r for r in records if r["semantic_class"] == key]
        count = sum(r["voxel_count"] for r in selected)
        categories.append(
            {
                "class": key,
                "label": title,
                "segment_count": len(selected),
                "voxel_count": count,
                "volume_um3": count * voxel_volume,
                "percent_foreground": 100 * count / foreground,
                "percent_roi": 100 * count / total,
                "assessment_status": (
                    "not_assessed"
                    if key in ("blood_vessel", "glia")
                    else "provisional_geometry"
                ),
            }
        )
    return {
        "denominator": "all voxels in the full segmentation array; no tissue mask",
        "foreground_definition": "segmentation label > 0",
        "total_voxels": total,
        "foreground_voxels": foreground,
        "background_voxels": total - foreground,
        "foreground_percent": 100 * foreground / total,
        "foreground_volume_um3": foreground * voxel_volume,
        "segment_count": len(records),
        "categories": categories,
        "pie_chart": {
            "value_field": "voxel_count",
            "denominator": "foreground",
            "category_field": "semantic_class",
            "hide_zero_categories": True,
        },
    }


def build_semantic_catalog(
    analysis_path: str | Path,
    segmentation_path: str | Path,
    label_sizes_path: str | Path,
    *,
    layer_uri: str,
    large_volume_um3: float = 1.0,
) -> dict[str, Any]:
    """Verify an analyzed segmentation and build an initial all-label catalog.

    Paths and dataset layout are supplied by the caller. The analysis supplies
    physical spacing and caliber gates. This function performs no publication.
    """
    if not np.isfinite(large_volume_um3) or large_volume_um3 <= 0:
        raise ValueError("large_volume_um3 must be finite and positive")
    source = Path(analysis_path)
    analysis = json.loads(source.read_text())
    metadata = analysis["metadata"]
    segmentation = Path(segmentation_path)
    if _digest(segmentation) != metadata["segmentation_sha256"]:
        raise ValueError("Segmentation differs from the analyzed source")
    spacing = np.asarray(metadata["spacing_nm_zyx"]) / 1000
    voxel_volume = float(np.prod(spacing))
    measured = {r["id"]: r for r in analysis["segments"]}
    with np.load(label_sizes_path) as sizes:
        counts = {str(int(i)): int(n) for i, n in zip(sizes["ids"], sizes["counts"])}
    # Independently verify the cached histogram and include background explicitly.
    histogram: np.ndarray = np.zeros(max(map(int, counts)) + 1, dtype=np.int64)
    with h5py.File(segmentation) as handle:
        volume = handle["main"]
        shape = list(volume.shape)
        if shape != metadata["shape_zyx"]:
            raise ValueError("Shape differs from analysis")
        for start in range(0, shape[0], 8):
            ids, frequency = np.unique(volume[start : start + 8], return_counts=True)
            if int(ids.max()) >= len(histogram):
                raise ValueError("Uncatalogued label in segmentation")
            histogram[ids.astype(np.int64)] += frequency
    actual = {str(int(i)): int(histogram[i]) for i in np.flatnonzero(histogram) if i}
    if actual != counts:
        raise ValueError("Cached label histogram differs from segmentation")
    if any(r["voxel_count"] != counts[key] for key, r in measured.items()):
        raise ValueError("Analysis voxel counts differ from segmentation")
    records: list[dict[str, Any]] = []
    for label, count in counts.items():
        source_record = measured.get(label)
        profile = (source_record or {}).get("profile") or {}
        continuity = (source_record or {}).get("continuity") or {}
        pieces = (source_record or {}).get("pieces") or {}
        if source_record is None:
            category, basis = "unclassified", "below_analysis_cutoff"
        else:
            category, basis = classify_semantic_candidate(
                source_record.get("semantic_type"),
                continuity.get("caliber_radius_um"),
                pieces.get("shaft_length_um", 0),
                axon_max_radius_um=metadata["continuity_config"]["axon_max_radius_um"],
                dendrite_min_radius_um=metadata["continuity_config"]["dendrite_min_radius_um"],
                skeleton_length_um=continuity.get("retained_length_um"),
                min_dendrite_length_um=_MIN_DENDRITE_LENGTH_UM,
                volume_um3=count * voxel_volume,
                min_axon_length_um=_MIN_AXON_LENGTH_UM,
                min_axon_volume_um3=_MIN_AXON_VOLUME_UM3,
            )
        large = count * voxel_volume >= large_volume_um3
        records.append(
            {
                "id": label,
                "voxel_count": count,
                "volume_um3": count * voxel_volume,
                "semantic_class": category,
                "automatic_class": category,
                "classification_basis": basis,
                "status": "unreviewed_candidate",
                "review": None,
                "correctness": None,
                "large_object": large,
                "below_analysis_cutoff": source_record is None,
                "review_priority": "high" if large else "normal",
                "review_reason": "large_foreground_contribution" if large else None,
                "centroid_um_zyx": profile.get("centroid_um_zyx"),
                "bbox_zyx": profile.get("bbox_zyx"),
                "evidence": {
                    "profile_class": profile.get("morphology_class"),
                    "local_caliber_um": continuity.get("caliber_radius_um"),
                    "original_semantic_type": (source_record or {}).get("semantic_type"),
                    "piece_composition": pieces.get("composition"),
                    "shaft_length_um": pieces.get("shaft_length_um"),
                    "terminal_swelling_count": pieces.get("terminal_swelling_count"),
                },
            }
        )
    records.sort(key=lambda r: (-r["voxel_count"], int(r["id"])))
    summary = summarize_semantic_records(records, shape, voxel_volume)
    queue = [
        dict(r, percent_foreground=100 * r["voxel_count"] / summary["foreground_voxels"])
        for r in records
        if r["large_object"]
    ]
    summary["large_object_review"] = {
        "threshold_volume_um3": large_volume_um3,
        "segment_count": len(queue),
        "percent_foreground": sum(r["percent_foreground"] for r in queue),
        "top_20_ids": [r["id"] for r in queue[:20]],
    }
    catalog = {
        "schema_name": "pytc.initial_semantic_segmentation",
        "schema_version": "1.0.0",
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "segmentation_layer": layer_uri,
            "segmentation_sha256": metadata["segmentation_sha256"],
            "analysis_sha256": _digest(source),
            "generator_sha256": _digest(__file__),
            "shape_zyx": shape,
            "spacing_nm_zyx": metadata["spacing_nm_zyx"],
            "coordinate_convention": metadata["coordinate_convention"],
            "source_analysis": str(source.resolve()),
            "caliber_gates_um": {
                "axon_max": metadata["continuity_config"]["axon_max_radius_um"],
                "dendrite_min": metadata["continuity_config"]["dendrite_min_radius_um"],
            },
            "min_dendrite_skeleton_length_um": _MIN_DENDRITE_LENGTH_UM,
            "min_axon_skeleton_length_um": _MIN_AXON_LENGTH_UM,
            "min_axon_volume_um3": _MIN_AXON_VOLUME_UM3,
            "validation": {
                "segmentation_sha256_matches_analysis": True,
                "all_positive_label_counts_verified_against_hdf5": True,
                "all_analyzed_label_counts_match": True,
            },
        },
        "limitations": [
            "Initial geometry-based candidates, not measured biological tissue fractions.",
            "Vessel and glia classes are not assessed; zero assignments do not mean absent.",
            "Neuronal somata belong to dendrite, but no soma rule exists: a soma is "
            "dendrite only if its caliber clears the dendrite gate, else unclassified.",
            "All foreground labels are included; sub-cutoff labels remain unclassified.",
            "Dense branching alone does not distinguish dendrites, glia, axons or false merges.",
            "Axon terminals are big blobs: a thick object with a skeleton under "
            f"{_MIN_DENDRITE_LENGTH_UM} um stays unclassified, not dendrite.",
            f"Axon calls need >= {_MIN_AXON_LENGTH_UM} um of skeleton or >= "
            f"{_MIN_AXON_VOLUME_UM3} um3; short-and-small fragments stay unclassified.",
            "Foreground percentage is occupancy of the full crop, not of a tissue mask.",
            "No human review, correctness score or ERL is inferred.",
        ],
        "summary": summary,
        "segments": records,
    }
    return catalog


def write_semantic_artifacts(
    catalog: dict[str, Any],
    output_dir: str | Path,
    *,
    title: str = "Initial semantic candidates",
) -> dict[str, Path]:
    """Write the catalog, summary, large-object review queue, and volume pie chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = catalog["summary"]
    queue = [
        dict(r, percent_foreground=100 * r["voxel_count"] / summary["foreground_voxels"])
        for r in catalog["segments"]
        if r["large_object"]
    ]
    paths = {}
    for name, payload in [
        ("semantic_segmentation.json", catalog),
        ("semantic_summary.json", summary),
        ("semantic_review_queue.json", {"metadata": catalog["metadata"], "segments": queue}),
    ]:
        paths[name] = output_dir / name
        paths[name].write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visible = [r for r in summary["categories"] if r["voxel_count"]]
    colors = {"axon": "#4c91ca", "dendrite": "#e8a34a", "unclassified": "#a8adb5"}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.pie(
        [r["voxel_count"] for r in visible],
        labels=[r["label"] for r in visible],
        colors=[colors[r["class"]] for r in visible],
        autopct="%.1f%%",
        startangle=90,
    )
    ax.set_title(f"{title}\n" f"Foreground: {summary['foreground_percent']:.2f}% of full crop")
    fig.text(
        0.5,
        0.04,
        "Slices = foreground volume. Vessel / glia: not assessed.\n"
        f"{len(queue)} large objects queued for review; "
        f"{summary['large_object_review']['percent_foreground']:.1f}% of foreground.",
        ha="center",
        fontsize=10,
    )
    paths["semantic_composition.png"] = output_dir / "semantic_composition.png"
    fig.savefig(paths["semantic_composition.png"], dpi=160, bbox_inches="tight")
    plt.close(fig)
    return paths
