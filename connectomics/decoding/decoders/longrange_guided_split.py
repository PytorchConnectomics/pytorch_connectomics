"""Long-range guided false-merge correction for decoded segmentations."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
from scipy import ndimage

from ..utils import cast2dtype

__all__ = ["longrange_guided_split"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LabelStats:
    label: int
    voxels: int
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @property
    def bbox(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.lo, self.hi))  # type: ignore[return-value]

    @property
    def extents(self) -> tuple[int, int, int]:
        return tuple(b - a for a, b in zip(self.lo, self.hi))  # type: ignore[return-value]


@dataclass(frozen=True)
class _GuideSeed:
    guide_id: int
    guide_voxels: int
    bbox: tuple[slice, slice, slice]
    primary_id: int
    overlap_voxels: int
    dominant_primary_fraction: float


@dataclass(frozen=True)
class _Candidate:
    primary_id: int
    primary_voxels: int
    seeds: tuple[_GuideSeed, ...]
    score: float


def _as_3d_segmentation(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(
            "longrange_guided_split expects a 3D segmentation or singleton-channel "
            f"4D segmentation, got shape {arr.shape}."
        )
    return arr


def _load_h5_dataset(path: str, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if dataset in f:
            return f[dataset][...]
        keys = [key for key in f.keys() if hasattr(f[key], "shape")]
        if len(keys) == 1:
            return f[keys[0]][...]
        raise ValueError(
            f"{path}: expected dataset {dataset!r} or exactly one dataset, got {list(f.keys())}"
        )


def _iter_spatial_chunks(source: Any) -> Iterator[tuple[slice, slice, slice]]:
    shape = tuple(int(v) for v in source.shape)
    if hasattr(source, "iter_chunks") and getattr(source, "chunks", None):
        yield from source.iter_chunks()
        return
    yield tuple(slice(0, s) for s in shape)  # type: ignore[misc]


def _slice_start(slc: tuple[slice, slice, slice]) -> np.ndarray:
    return np.asarray([0 if s.start is None else int(s.start) for s in slc], dtype=np.int64)


def _update_stats(
    stats: dict[int, tuple[int, np.ndarray, np.ndarray]],
    label: int,
    count: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> None:
    if label in stats:
        old_count, old_lo, old_hi = stats[label]
        stats[label] = (
            old_count + count,
            np.minimum(old_lo, lo),
            np.maximum(old_hi, hi),
        )
    else:
        stats[label] = (count, lo.astype(np.int64), hi.astype(np.int64))


def _compute_label_stats(source: Any, *, ignore_label: int = 0) -> dict[int, _LabelStats]:
    raw_stats: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for slc in _iter_spatial_chunks(source):
        block = np.asarray(source[slc])
        labels, counts = np.unique(block, return_counts=True)
        start = _slice_start(slc)
        for label_value, count_value in zip(labels, counts):
            label = int(label_value)
            if label == ignore_label:
                continue
            mask = block == label_value
            coords = np.nonzero(mask)
            if coords[0].size == 0:
                continue
            lo = start + np.asarray([int(axis.min()) for axis in coords], dtype=np.int64)
            hi = start + np.asarray([int(axis.max()) + 1 for axis in coords], dtype=np.int64)
            _update_stats(raw_stats, label, int(count_value), lo, hi)

    return {
        label: _LabelStats(
            label=label,
            voxels=int(count),
            lo=tuple(int(v) for v in lo),
            hi=tuple(int(v) for v in hi),
        )
        for label, (count, lo, hi) in raw_stats.items()
    }


def _bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute bbox for an empty mask.")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def _pad_bbox(
    bbox: tuple[slice, slice, slice],
    shape: Sequence[int],
    pad: Sequence[int],
) -> tuple[slice, slice, slice]:
    padded = []
    for slc, dim, width in zip(bbox, shape, pad):
        start = max(0, int(slc.start or 0) - int(width))
        stop = min(int(dim), int(slc.stop or 0) + int(width))
        padded.append(slice(start, stop))
    return tuple(padded)  # type: ignore[return-value]


def _normalize_pad(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    pad = tuple(int(v) for v in value)
    if len(pad) != 3:
        raise ValueError(f"bbox_pad must have length 3, got {value!r}.")
    return pad  # type: ignore[return-value]


def _format_bbox(bbox: tuple[slice, slice, slice]) -> tuple[str, str]:
    start = ",".join(str(int(s.start or 0)) for s in bbox)
    stop = ",".join(str(int(s.stop or 0)) for s in bbox)
    return start, stop


def _select_large_guide_components(
    stats: dict[int, _LabelStats],
    *,
    min_seed_voxels: int,
    min_seed_axis_extent: int,
    max_seed_axis_extent: int,
) -> list[_LabelStats]:
    selected = []
    for item in stats.values():
        extents = item.extents
        if item.voxels < min_seed_voxels:
            continue
        if min(extents) < min_seed_axis_extent:
            continue
        if max_seed_axis_extent > 0 and max(extents) > max_seed_axis_extent:
            continue
        selected.append(item)
    selected.sort(key=lambda s: (-s.voxels, s.label))
    return selected


def _find_guide_parent(
    primary: np.ndarray,
    guide_source: Any,
    guide_stats: _LabelStats,
    *,
    min_seed_overlap_voxels: int,
    min_seed_guide_fraction: float,
    ignore_label: int,
) -> _GuideSeed | None:
    bbox = guide_stats.bbox
    primary_crop = primary[bbox]
    guide_crop = np.asarray(guide_source[bbox])
    seed_mask = guide_crop == guide_stats.label
    if not seed_mask.any():
        return None

    labels, counts = np.unique(primary_crop[seed_mask], return_counts=True)
    keep = labels != ignore_label
    labels = labels[keep]
    counts = counts[keep]
    if labels.size == 0:
        return None

    order = np.argsort(counts)[::-1]
    parent_id = int(labels[order[0]])
    overlap = int(counts[order[0]])
    fraction = overlap / float(guide_stats.voxels)
    if overlap < min_seed_overlap_voxels:
        return None
    if fraction < min_seed_guide_fraction:
        return None
    return _GuideSeed(
        guide_id=guide_stats.label,
        guide_voxels=guide_stats.voxels,
        bbox=bbox,
        primary_id=parent_id,
        overlap_voxels=overlap,
        dominant_primary_fraction=float(fraction),
    )


def _build_candidates(
    primary: np.ndarray,
    guide_source: Any,
    *,
    guide_stats: dict[int, _LabelStats],
    min_parent_voxels: int,
    min_seed_voxels: int,
    min_seed_axis_extent: int,
    max_seed_axis_extent: int,
    min_seed_overlap_voxels: int,
    min_seed_guide_fraction: float,
    min_seeds_in_parent: int,
    max_splits_per_parent: int,
    max_parents_per_volume: int,
    target_ids: Sequence[int] | None,
    ignore_label: int,
) -> tuple[list[_Candidate], list[dict[str, object]]]:
    primary_labels, primary_counts = np.unique(primary, return_counts=True)
    primary_sizes = {
        int(label): int(count)
        for label, count in zip(primary_labels, primary_counts)
        if int(label) != ignore_label
    }
    target_set = {int(v) for v in target_ids} if target_ids is not None else None

    seed_groups: dict[int, list[_GuideSeed]] = {}
    for component in _select_large_guide_components(
        guide_stats,
        min_seed_voxels=min_seed_voxels,
        min_seed_axis_extent=min_seed_axis_extent,
        max_seed_axis_extent=max_seed_axis_extent,
    ):
        seed = _find_guide_parent(
            primary,
            guide_source,
            component,
            min_seed_overlap_voxels=min_seed_overlap_voxels,
            min_seed_guide_fraction=min_seed_guide_fraction,
            ignore_label=ignore_label,
        )
        if seed is None:
            continue
        if target_set is not None and seed.primary_id not in target_set:
            continue
        seed_groups.setdefault(seed.primary_id, []).append(seed)

    rows: list[dict[str, object]] = []
    candidates: list[_Candidate] = []
    for parent_id, seeds in sorted(seed_groups.items()):
        parent_size = primary_sizes.get(parent_id, 0)
        seeds = sorted(seeds, key=lambda s: (-s.overlap_voxels, s.guide_id))
        retained_count = len(seeds)
        accepted = (
            parent_size >= min_parent_voxels
            and retained_count >= min_seeds_in_parent
            and retained_count <= max_splits_per_parent
        )
        reason = ""
        if parent_size < min_parent_voxels:
            reason = "parent_too_small"
        elif retained_count < min_seeds_in_parent:
            reason = "too_few_guide_seeds"
        elif retained_count > max_splits_per_parent:
            reason = "too_many_guide_seeds"

        total_overlap = sum(seed.overlap_voxels for seed in seeds)
        score = float(retained_count * 1_000_000 + total_overlap)
        if accepted:
            candidates.append(
                _Candidate(
                    primary_id=parent_id,
                    primary_voxels=parent_size,
                    seeds=tuple(seeds),
                    score=score,
                )
            )

        seed_bbox = _union_bboxes([seed.bbox for seed in seeds]) if seeds else None
        start, stop = _format_bbox(seed_bbox) if seed_bbox is not None else ("", "")
        rows.append(
            {
                "primary_id": parent_id,
                "primary_voxels": parent_size,
                "bbox_start": start,
                "bbox_stop": stop,
                "retained_seed_count": retained_count,
                "retained_seed_ids": ";".join(str(seed.guide_id) for seed in seeds),
                "retained_seed_overlaps": ";".join(str(seed.overlap_voxels) for seed in seeds),
                "dominant_primary_fraction": ";".join(
                    f"{seed.dominant_primary_fraction:.6g}" for seed in seeds
                ),
                "candidate_score": f"{score:.6g}",
                "decision": "candidate" if accepted else "reject",
                "reject_reason": reason,
            }
        )

    candidates.sort(key=lambda c: (-c.score, c.primary_id))
    if max_parents_per_volume > 0:
        candidates = candidates[:max_parents_per_volume]
    return candidates, rows


def _union_bboxes(bboxes: Sequence[tuple[slice, slice, slice]]) -> tuple[slice, slice, slice]:
    if not bboxes:
        raise ValueError("Cannot union an empty bbox list.")
    lo = np.asarray([[int(s.start or 0) for s in bbox] for bbox in bboxes], dtype=np.int64)
    hi = np.asarray([[int(s.stop or 0) for s in bbox] for bbox in bboxes], dtype=np.int64)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo.min(axis=0), hi.max(axis=0)))  # type: ignore[return-value]


def _assign_from_markers(parent_mask: np.ndarray, markers: np.ndarray) -> np.ndarray:
    seed_mask = markers != 0
    if not seed_mask.any():
        return np.zeros_like(markers, dtype=np.uint64)
    _, nearest = ndimage.distance_transform_edt(~seed_mask, return_indices=True)
    assigned = markers[tuple(nearest)].astype(np.uint64, copy=False)
    assigned[~parent_mask] = 0
    return assigned


def _prepare_markers(
    parent_mask: np.ndarray,
    guide_crop: np.ndarray,
    seeds: Sequence[_GuideSeed],
    *,
    erosion_iterations: int,
    min_marker_voxels: int,
) -> tuple[np.ndarray, list[int]]:
    markers = np.zeros(parent_mask.shape, dtype=np.uint64)
    retained_guide_ids: list[int] = []
    structure = ndimage.generate_binary_structure(3, 1)
    next_marker = 1
    for seed in seeds:
        marker_mask = parent_mask & (guide_crop == seed.guide_id)
        if erosion_iterations > 0 and marker_mask.any():
            eroded = ndimage.binary_erosion(
                marker_mask,
                structure=structure,
                iterations=erosion_iterations,
                border_value=0,
            )
            if int(eroded.sum()) >= min_marker_voxels:
                marker_mask = eroded
        if int(marker_mask.sum()) < min_marker_voxels:
            continue
        markers[marker_mask] = next_marker
        retained_guide_ids.append(seed.guide_id)
        next_marker += 1
    return markers, retained_guide_ids


def _resolve_new_label_start(
    value: int | str, primary: np.ndarray, guide_stats: dict[int, _LabelStats]
) -> int:
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError("new_label_start must be an integer or 'auto'.")
        guide_max = max(guide_stats.keys(), default=0)
        return max(int(primary.max()), guide_max) + 1
    return int(value)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_report_dir(
    report_dir: str,
    guide_seg_path: str,
    primary_affinity_path: str,
    guide_affinity_path: str,
) -> Path | None:
    if report_dir:
        return Path(report_dir).expanduser()
    for value in (guide_seg_path, primary_affinity_path, guide_affinity_path):
        if value:
            return Path(value).expanduser().parent
    return None


def _split_candidates(
    primary: np.ndarray,
    guide_source: Any,
    candidates: Sequence[_Candidate],
    *,
    guide_stats: dict[int, _LabelStats],
    assignment: str,
    bbox_pad: tuple[int, int, int],
    erosion_iterations: int,
    min_marker_voxels: int,
    min_child_voxels: int,
    new_label_start: int | str,
    dry_run: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if assignment not in {"nearest", "seeded_watershed"}:
        raise ValueError(
            "longrange_guided_split assignment currently supports 'nearest' "
            "and 'seeded_watershed'."
        )

    output = primary.astype(np.uint64, copy=True)
    next_label = _resolve_new_label_start(new_label_start, primary, guide_stats)
    decisions: list[dict[str, object]] = []

    for candidate in candidates:
        parent_id = candidate.primary_id
        decision: dict[str, object] = {
            "primary_id": parent_id,
            "accepted": False,
            "reject_reason": "",
            "bbox_start": "",
            "bbox_stop": "",
            "guide_seed_ids": ";".join(str(seed.guide_id) for seed in candidate.seeds),
            "output_child_ids": "",
            "child_voxels": "",
            "changed_voxels": 0,
            "assignment": assignment,
            "mean_cut_affinity_primary": "",
            "mean_cut_affinity_guide": "",
        }
        if dry_run:
            decision["reject_reason"] = "dry_run"
            decisions.append(decision)
            continue

        parent_mask_global = output == parent_id
        if not parent_mask_global.any():
            decision["reject_reason"] = "parent_not_found"
            decisions.append(decision)
            continue

        bbox = _pad_bbox(_bbox_from_mask(parent_mask_global), output.shape, bbox_pad)
        start, stop = _format_bbox(bbox)
        decision["bbox_start"] = start
        decision["bbox_stop"] = stop
        parent_crop = output[bbox]
        guide_crop = np.asarray(guide_source[bbox])
        parent_mask = parent_crop == parent_id
        markers, retained_guide_ids = _prepare_markers(
            parent_mask,
            guide_crop,
            candidate.seeds,
            erosion_iterations=erosion_iterations,
            min_marker_voxels=min_marker_voxels,
        )
        if len(retained_guide_ids) < 2:
            decision["reject_reason"] = "too_few_markers_after_filter"
            decisions.append(decision)
            continue

        assigned = _assign_from_markers(parent_mask, markers)
        marker_ids, child_counts = np.unique(assigned[assigned != 0], return_counts=True)
        valid = child_counts >= min_child_voxels
        marker_ids = marker_ids[valid]
        child_counts = child_counts[valid]
        if marker_ids.size < 2:
            decision["reject_reason"] = "too_few_children_after_assignment"
            decisions.append(decision)
            continue

        order = np.lexsort((marker_ids, -child_counts))
        marker_ids = marker_ids[order]
        child_counts = child_counts[order]
        output_ids: list[int] = []
        for idx, _marker_id in enumerate(marker_ids):
            if idx == 0:
                output_ids.append(parent_id)
            else:
                output_ids.append(next_label)
                next_label += 1

        updated = parent_crop.copy()
        for marker_id, output_id in zip(marker_ids, output_ids):
            updated[parent_mask & (assigned == marker_id)] = output_id

        changed = int(np.sum((updated != parent_crop) & parent_mask))
        output[bbox] = updated
        decision.update(
            {
                "accepted": True,
                "guide_seed_ids": ";".join(str(v) for v in retained_guide_ids),
                "output_child_ids": ";".join(str(v) for v in output_ids),
                "child_voxels": ";".join(str(int(v)) for v in child_counts),
                "changed_voxels": changed,
            }
        )
        decisions.append(decision)

    return cast2dtype(output), decisions


def longrange_guided_split(
    primary: np.ndarray,
    *,
    guide_seg: np.ndarray | None = None,
    guide_seg_path: str = "",
    guide_dataset: str = "main",
    guide_prediction_path: str = "",
    guide_decode: dict[str, Any] | None = None,
    primary_affinity_path: str = "",
    guide_affinity_path: str = "",
    candidate_mode: str = "auto",
    target_ids: Sequence[int] | None = None,
    dry_run: bool = False,
    report: bool = True,
    report_dir: str = "",
    tag: str = "lrgf",
    min_parent_voxels: int = 200_000,
    min_seed_voxels: int = 50_000,
    min_seed_axis_extent: int = 16,
    max_seed_axis_extent: int = 0,
    min_seed_overlap_voxels: int = 5_000,
    min_seed_guide_fraction: float = 0.25,
    min_seeds_in_parent: int = 2,
    max_splits_per_parent: int = 8,
    max_parents_per_volume: int = 0,
    bbox_pad: int | Sequence[int] = (0, 0, 0),
    assignment: str = "nearest",
    erosion_iterations: int = 0,
    min_marker_voxels: int = 1,
    min_child_voxels: int = 1,
    new_label_start: int | str = "auto",
    ignore_label: int = 0,
    **_unused: Any,
) -> np.ndarray:
    """Split primary labels that contain multiple large long-range guide components.

    The decoder is intended to run as a correction step after a primary
    segmentation decoder. Candidate discovery is bbox-first: large labels from
    the guide segmentation are mapped to their dominant primary parent, and only
    parents with multiple retained guide labels are split.
    """
    del guide_prediction_path, guide_decode, tag
    if candidate_mode not in {"auto", "sparse_overlap"}:
        raise ValueError("candidate_mode must be 'auto' or 'sparse_overlap'.")
    if candidate_mode == "sparse_overlap":
        logger.info("candidate_mode='sparse_overlap' requested; using bbox-first auto mode in v0.")

    primary_arr = _as_3d_segmentation(primary)
    pad = _normalize_pad(bbox_pad)

    if guide_seg is None:
        if not guide_seg_path:
            logger.info("longrange_guided_split: no guide segmentation; returning input.")
            return primary_arr.copy()
        with h5py.File(guide_seg_path, "r") as f:
            if guide_dataset in f:
                guide_source = f[guide_dataset]
            else:
                keys = [key for key in f.keys() if hasattr(f[key], "shape")]
                if len(keys) != 1:
                    raise ValueError(
                        f"{guide_seg_path}: expected dataset {guide_dataset!r} or exactly "
                        f"one dataset, got {list(f.keys())}"
                    )
                guide_source = f[keys[0]]
            return _run_longrange_guided_split(
                primary_arr,
                guide_source,
                guide_seg_path=guide_seg_path,
                primary_affinity_path=primary_affinity_path,
                guide_affinity_path=guide_affinity_path,
                target_ids=target_ids,
                dry_run=dry_run,
                report=report,
                report_dir=report_dir,
                min_parent_voxels=min_parent_voxels,
                min_seed_voxels=min_seed_voxels,
                min_seed_axis_extent=min_seed_axis_extent,
                max_seed_axis_extent=max_seed_axis_extent,
                min_seed_overlap_voxels=min_seed_overlap_voxels,
                min_seed_guide_fraction=min_seed_guide_fraction,
                min_seeds_in_parent=min_seeds_in_parent,
                max_splits_per_parent=max_splits_per_parent,
                max_parents_per_volume=max_parents_per_volume,
                bbox_pad=pad,
                assignment=assignment,
                erosion_iterations=erosion_iterations,
                min_marker_voxels=min_marker_voxels,
                min_child_voxels=min_child_voxels,
                new_label_start=new_label_start,
                ignore_label=ignore_label,
            )

    guide_arr = _as_3d_segmentation(guide_seg)
    return _run_longrange_guided_split(
        primary_arr,
        guide_arr,
        guide_seg_path=guide_seg_path,
        primary_affinity_path=primary_affinity_path,
        guide_affinity_path=guide_affinity_path,
        target_ids=target_ids,
        dry_run=dry_run,
        report=report,
        report_dir=report_dir,
        min_parent_voxels=min_parent_voxels,
        min_seed_voxels=min_seed_voxels,
        min_seed_axis_extent=min_seed_axis_extent,
        max_seed_axis_extent=max_seed_axis_extent,
        min_seed_overlap_voxels=min_seed_overlap_voxels,
        min_seed_guide_fraction=min_seed_guide_fraction,
        min_seeds_in_parent=min_seeds_in_parent,
        max_splits_per_parent=max_splits_per_parent,
        max_parents_per_volume=max_parents_per_volume,
        bbox_pad=pad,
        assignment=assignment,
        erosion_iterations=erosion_iterations,
        min_marker_voxels=min_marker_voxels,
        min_child_voxels=min_child_voxels,
        new_label_start=new_label_start,
        ignore_label=ignore_label,
    )


def _run_longrange_guided_split(
    primary: np.ndarray,
    guide_source: Any,
    *,
    guide_seg_path: str,
    primary_affinity_path: str,
    guide_affinity_path: str,
    target_ids: Sequence[int] | None,
    dry_run: bool,
    report: bool,
    report_dir: str,
    min_parent_voxels: int,
    min_seed_voxels: int,
    min_seed_axis_extent: int,
    max_seed_axis_extent: int,
    min_seed_overlap_voxels: int,
    min_seed_guide_fraction: float,
    min_seeds_in_parent: int,
    max_splits_per_parent: int,
    max_parents_per_volume: int,
    bbox_pad: tuple[int, int, int],
    assignment: str,
    erosion_iterations: int,
    min_marker_voxels: int,
    min_child_voxels: int,
    new_label_start: int | str,
    ignore_label: int,
) -> np.ndarray:
    if tuple(primary.shape) != tuple(guide_source.shape):
        raise ValueError(
            f"Guide segmentation shape {tuple(guide_source.shape)} does not match "
            f"primary shape {tuple(primary.shape)}."
        )

    guide_stats = _compute_label_stats(guide_source, ignore_label=ignore_label)
    candidates, candidate_rows = _build_candidates(
        primary,
        guide_source,
        guide_stats=guide_stats,
        min_parent_voxels=min_parent_voxels,
        min_seed_voxels=min_seed_voxels,
        min_seed_axis_extent=min_seed_axis_extent,
        max_seed_axis_extent=max_seed_axis_extent,
        min_seed_overlap_voxels=min_seed_overlap_voxels,
        min_seed_guide_fraction=min_seed_guide_fraction,
        min_seeds_in_parent=min_seeds_in_parent,
        max_splits_per_parent=max_splits_per_parent,
        max_parents_per_volume=max_parents_per_volume,
        target_ids=target_ids,
        ignore_label=ignore_label,
    )
    result, decision_rows = _split_candidates(
        primary,
        guide_source,
        candidates,
        guide_stats=guide_stats,
        assignment=assignment,
        bbox_pad=bbox_pad,
        erosion_iterations=erosion_iterations,
        min_marker_voxels=min_marker_voxels,
        min_child_voxels=min_child_voxels,
        new_label_start=new_label_start,
        dry_run=dry_run,
    )

    if report:
        out_dir = _resolve_report_dir(
            report_dir,
            guide_seg_path,
            primary_affinity_path,
            guide_affinity_path,
        )
        if out_dir is not None:
            _write_csv(
                out_dir / "guided_split_candidates.csv",
                candidate_rows,
                [
                    "primary_id",
                    "primary_voxels",
                    "bbox_start",
                    "bbox_stop",
                    "retained_seed_count",
                    "retained_seed_ids",
                    "retained_seed_overlaps",
                    "dominant_primary_fraction",
                    "candidate_score",
                    "decision",
                    "reject_reason",
                ],
            )
            _write_csv(
                out_dir / "guided_split_decisions.csv",
                decision_rows,
                [
                    "primary_id",
                    "accepted",
                    "reject_reason",
                    "bbox_start",
                    "bbox_stop",
                    "guide_seed_ids",
                    "output_child_ids",
                    "child_voxels",
                    "changed_voxels",
                    "assignment",
                    "mean_cut_affinity_primary",
                    "mean_cut_affinity_guide",
                ],
            )
    logger.info(
        "longrange_guided_split: %d candidates, %d accepted splits",
        len(candidates),
        sum(1 for row in decision_rows if bool(row["accepted"])),
    )
    return result
