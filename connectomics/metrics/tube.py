"""Ground-truth-free analysis for tube-like instance segmentations.

The metrics in this module describe whether predicted instances look like
substantial, complete, single tubes. They are diagnostic rankers, not proof of
biological correctness: real axons may terminate inside a volume, and a
geometrically clean tube can still join two different axons end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import cc3d
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import median_filter

from connectomics.data.processing.bbox import seg_stats

MIN_SPAN_FRAC = 0.25
MIN_SIZE = 20000
BORDER = 2

_FACE_NAMES = ("z0", "zmax", "y0", "ymax", "x0", "xmax")


@dataclass(frozen=True)
class TubeAnalysisConfig:
    """Thresholds for :func:`analyze_tubes`.

    A label enters the detailed analysis when it is either long or large.
    ``border_margin`` is the maximum voxel distance from a volume face, so a
    margin of two includes coordinates 0, 1, and 2. Parallel-strand fractions
    are estimated from every ``multi_component_slice_step`` slice;
    ``parallel_min_slices`` remains expressed in original z slices.
    """

    substantial_min_z_slices: int = 21
    substantial_min_voxels: int = 10000
    long_span_fraction: float = MIN_SPAN_FRAC
    decent_min_voxels: int = MIN_SIZE
    border_margin: int = BORDER
    border_patch_min_voxels: int = 10
    multi_component_min_voxels: int = 50
    multi_component_slice_step: int = 3
    parallel_min_slices: int = 15
    parallel_fraction_threshold: float = 0.30
    disconnected_component_min_voxels: int = 1000
    bump_min_slices: int = 40
    bump_relative_excess: float = 0.20
    bump_absolute_excess: int = 200
    bump_max_slices: int = 30
    bump_median_window: int = 31

    def __post_init__(self) -> None:
        positive_ints: dict[str, int] = {
            "substantial_min_z_slices": self.substantial_min_z_slices,
            "substantial_min_voxels": self.substantial_min_voxels,
            "decent_min_voxels": self.decent_min_voxels,
            "border_patch_min_voxels": self.border_patch_min_voxels,
            "multi_component_min_voxels": self.multi_component_min_voxels,
            "multi_component_slice_step": self.multi_component_slice_step,
            "parallel_min_slices": self.parallel_min_slices,
            "disconnected_component_min_voxels": self.disconnected_component_min_voxels,
            "bump_min_slices": self.bump_min_slices,
            "bump_absolute_excess": self.bump_absolute_excess,
            "bump_max_slices": self.bump_max_slices,
            "bump_median_window": self.bump_median_window,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.border_margin < 0:
            raise ValueError(f"border_margin must be non-negative, got {self.border_margin}")
        for name, fraction_value in (
            ("long_span_fraction", self.long_span_fraction),
            ("parallel_fraction_threshold", self.parallel_fraction_threshold),
        ):
            if not 0.0 <= fraction_value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {fraction_value}")
        if self.bump_relative_excess < 0.0:
            raise ValueError(
                "bump_relative_excess must be non-negative, " f"got {self.bump_relative_excess}"
            )


@dataclass(frozen=True)
class TubeRecord:
    """GT-free measurements for one substantial predicted instance."""

    label: int
    voxel_count: int
    z_min: int
    z_max: int
    z_span: int
    z_slice_count: int
    z_occupancy_fraction: float
    is_long_enough: bool
    is_decent: bool
    face_contacts: tuple[str, ...]
    border_end_count: int
    border_patch_count: int
    median_cross_section_area: float
    max_cross_section_area: int
    bump_count: int
    multi_component_sample_count: int
    evaluated_sample_count: int
    multi_component_fraction: float
    is_parallel: bool
    component_count_3d: int
    significant_component_count_3d: int
    is_disconnected: bool

    @property
    def face_count(self) -> int:
        """Number of distinct relaxed volume faces touched."""

        return len(self.face_contacts)

    @property
    def is_complete(self) -> bool:
        """Whether both z-directed tube ends reach a volume face."""

        return self.border_end_count >= 2

    @property
    def is_single_tube(self) -> bool:
        """Whether no persistent parallel strand or 3D disconnection is found."""

        return not self.is_parallel and not self.is_disconnected

    @property
    def is_valid_tube(self) -> bool:
        """Whether the instance is both complete and geometrically single."""

        return self.is_complete and self.is_single_tube


@dataclass(frozen=True)
class TubeAnalysisSummary:
    """Aggregate count- and volume-weighted tube statistics."""

    total_label_count: int
    total_foreground_voxels: int
    substantial_count: int
    substantial_voxels: int
    long_enough_count: int
    decent_count: int
    decent_voxels: int
    complete_count: int
    complete_voxels: int
    valid_count: int
    valid_voxels: int
    parallel_count: int
    disconnected_count: int
    bumped_count: int
    complete_fraction: float
    complete_volume_fraction: float
    valid_fraction: float
    valid_volume_fraction: float
    face_count_histogram: tuple[int, ...]
    border_end_histogram: tuple[int, ...]


@dataclass(frozen=True)
class TubeAnalysis:
    """Detailed and aggregate output from :func:`analyze_tubes`."""

    volume_shape: tuple[int, int, int]
    config: TubeAnalysisConfig
    total_label_count: int
    total_foreground_voxels: int
    tubes: tuple[TubeRecord, ...]

    @property
    def summary(self) -> TubeAnalysisSummary:
        """Compute aggregate statistics over the analyzed tubes."""

        decent = tuple(tube for tube in self.tubes if tube.is_decent)
        complete = tuple(tube for tube in decent if tube.is_complete)
        valid = tuple(tube for tube in decent if tube.is_valid_tube)
        substantial_voxels = sum(tube.voxel_count for tube in self.tubes)
        decent_voxels = sum(tube.voxel_count for tube in decent)
        complete_voxels = sum(tube.voxel_count for tube in complete)
        valid_voxels = sum(tube.voxel_count for tube in valid)
        face_histogram = tuple(
            sum(tube.face_count == face_count for tube in self.tubes) for face_count in range(7)
        )
        end_histogram = tuple(
            sum(tube.border_end_count == end_count for tube in self.tubes) for end_count in range(3)
        )
        return TubeAnalysisSummary(
            total_label_count=self.total_label_count,
            total_foreground_voxels=self.total_foreground_voxels,
            substantial_count=len(self.tubes),
            substantial_voxels=substantial_voxels,
            long_enough_count=sum(tube.is_long_enough for tube in self.tubes),
            decent_count=len(decent),
            decent_voxels=decent_voxels,
            complete_count=len(complete),
            complete_voxels=complete_voxels,
            valid_count=len(valid),
            valid_voxels=valid_voxels,
            parallel_count=sum(tube.is_parallel for tube in self.tubes),
            disconnected_count=sum(tube.is_disconnected for tube in self.tubes),
            bumped_count=sum(tube.bump_count > 0 for tube in self.tubes),
            complete_fraction=_safe_fraction(len(complete), len(decent)),
            complete_volume_fraction=_safe_fraction(complete_voxels, decent_voxels),
            valid_fraction=_safe_fraction(len(valid), len(decent)),
            valid_volume_fraction=_safe_fraction(valid_voxels, decent_voxels),
            face_count_histogram=face_histogram,
            border_end_histogram=end_histogram,
        )

    def largest_incomplete(self, limit: int = 8) -> tuple[TubeRecord, ...]:
        """Return the largest decent tubes whose two ends do not reach a border."""

        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        incomplete = (tube for tube in self.tubes if tube.is_decent and not tube.is_complete)
        return tuple(sorted(incomplete, key=lambda tube: tube.voxel_count, reverse=True)[:limit])


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_segmentation(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f"Tube analysis requires a 3D segmentation, got shape {seg.shape}")
    if not np.issubdtype(seg.dtype, np.integer):
        raise TypeError(f"Tube analysis requires integer labels, got dtype {seg.dtype}")
    if np.issubdtype(seg.dtype, np.signedinteger) and np.any(seg < 0):
        raise ValueError("Tube analysis requires non-negative labels")
    return seg


def _area_profiles(seg: np.ndarray, labels: set[int]) -> dict[int, dict[int, int]]:
    profiles: dict[int, dict[int, int]] = {label: {} for label in labels}
    for z in range(seg.shape[0]):
        slice_labels, counts = np.unique(seg[z], return_counts=True)
        for label, count in zip(slice_labels.tolist(), counts.tolist()):
            label = int(label)
            if label in labels:
                profiles[label][z] = int(count)
    return profiles


def _face_contacts(
    seg: np.ndarray,
    labels: set[int],
    margin: int,
) -> dict[int, tuple[str, ...]]:
    width = margin + 1
    slabs = (
        seg[:width],
        seg[-width:],
        seg[:, :width],
        seg[:, -width:],
        seg[:, :, :width],
        seg[:, :, -width:],
    )
    contacts: dict[int, set[str]] = {label: set() for label in labels}
    for face_name, slab in zip(_FACE_NAMES, slabs):
        for label in np.unique(slab).tolist():
            label = int(label)
            if label in labels:
                contacts[label].add(face_name)
    return {label: tuple(sorted(names)) for label, names in contacts.items()}


def _border_end_count(
    sub: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
    volume_shape: tuple[int, int, int],
    margin: int,
) -> int:
    z0, z1, y0, _y1, x0, _x1 = bounds
    zyx_ends = ((0, z0),) if z0 == z1 else ((0, z0), (-1, z1))
    count = 0
    for local_z, global_z in zyx_ends:
        ys, xs = np.where(sub[local_z])
        if len(ys) == 0:
            continue
        global_ys = ys + y0
        global_xs = xs + x0
        if (
            global_z <= margin
            or global_z >= volume_shape[0] - 1 - margin
            or int(global_ys.min()) <= margin
            or int(global_ys.max()) >= volume_shape[1] - 1 - margin
            or int(global_xs.min()) <= margin
            or int(global_xs.max()) >= volume_shape[2] - 1 - margin
        ):
            count += 1
    return count


def _border_patch_count(
    sub: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
    volume_shape: tuple[int, int, int],
    margin: int,
    min_voxels: int,
) -> int:
    z0, z1, y0, y1, x0, x1 = bounds
    zz = np.arange(z0, z1 + 1)[:, None, None]
    yy = np.arange(y0, y1 + 1)[None, :, None]
    xx = np.arange(x0, x1 + 1)[None, None, :]
    shell = (
        (zz <= margin)
        | (zz >= volume_shape[0] - 1 - margin)
        | (yy <= margin)
        | (yy >= volume_shape[1] - 1 - margin)
        | (xx <= margin)
        | (xx >= volume_shape[2] - 1 - margin)
    )
    contact = sub & shell
    if not contact.any():
        return 0
    components = cc3d.connected_components(contact, connectivity=26)
    component_sizes = np.bincount(components.ravel())
    return int(np.count_nonzero(component_sizes[1:] >= min_voxels))


def _multi_component_stats(
    sub: np.ndarray,
    min_voxels: int,
    step: int,
) -> tuple[int, int, float]:
    multi_count = 0
    evaluated_count = 0
    for section in sub[::step]:
        if not section.any():
            continue
        evaluated_count += 1
        components, count = cc_label(section)
        if count < 2:
            continue
        component_sizes = np.bincount(components.ravel())[1:]
        if np.count_nonzero(component_sizes >= min_voxels) >= 2:
            multi_count += 1
    return multi_count, evaluated_count, _safe_fraction(multi_count, evaluated_count)


def _component_stats(
    sub: np.ndarray,
    min_voxels: int,
) -> tuple[int, int]:
    components = cc3d.connected_components(sub, connectivity=26)
    component_sizes = np.bincount(components.ravel())[1:]
    return len(component_sizes), int(np.count_nonzero(component_sizes >= min_voxels))


def _area_bumps(
    profile: dict[int, int],
    config: TubeAnalysisConfig,
) -> tuple[tuple[int, int, int, int], ...]:
    if len(profile) < config.bump_median_window:
        return ()
    z_min, z_max = min(profile), max(profile)
    areas: np.ndarray = np.zeros(z_max - z_min + 1, dtype=np.float64)
    for z, area in profile.items():
        areas[z - z_min] = area
    baseline = median_filter(areas, size=config.bump_median_window, mode="nearest")
    flags = (
        (areas > (1.0 + config.bump_relative_excess) * baseline)
        & (areas - baseline > config.bump_absolute_excess)
        & (areas > 0)
    )
    bumps: list[tuple[int, int, int, int]] = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(flags) and flags[end + 1]:
            end += 1
        bump_z0, bump_z1 = z_min + index, z_min + end
        before, after = bump_z0 - 1, bump_z1 + 1
        if (
            bump_z1 - bump_z0 + 1 <= config.bump_max_slices
            and before in profile
            and after in profile
        ):
            bumps.append((bump_z0, bump_z1, before, after))
        index = end + 1
    return tuple(bumps)


def analyze_tubes(
    seg: np.ndarray,
    config: TubeAnalysisConfig | None = None,
    stats: Any = None,
) -> TubeAnalysis:
    """Analyze a 3D instance segmentation without ground truth.

    The detailed set includes every label meeting either the configured
    z-slice or voxel-count threshold. Completeness is reported separately for
    the stricter "decent" subset that is both large and spans a configured
    fraction of the volume in z.

    Args:
        seg: Three-dimensional non-negative integer instance segmentation.
        config: Analysis thresholds. Defaults preserve the LiCONN research
            operating points.
        stats: Optional cached ``seg_stats`` output or ``(bounds, sizes)``.

    Returns:
        Structured per-label and aggregate GT-free analysis.
    """

    seg = _validate_segmentation(seg)
    config = config or TubeAnalysisConfig()
    if stats is None:
        bounds, sizes, _ = seg_stats(seg)
    else:
        bounds, sizes = stats[:2]
    selected = {
        int(label)
        for label, bbox in bounds.items()
        if int(sizes[label]) >= config.substantial_min_voxels
        or bbox[1] - bbox[0] + 1 >= config.substantial_min_z_slices
    }
    profiles = _area_profiles(seg, selected)
    contacts = _face_contacts(seg, selected, config.border_margin)
    long_span = ceil(config.long_span_fraction * seg.shape[0])
    volume_shape = (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2]))
    records: list[TubeRecord] = []
    for label in sorted(selected):
        label_bounds = (
            int(bounds[label][0]),
            int(bounds[label][1]),
            int(bounds[label][2]),
            int(bounds[label][3]),
            int(bounds[label][4]),
            int(bounds[label][5]),
        )
        z0, z1, y0, y1, x0, x1 = label_bounds
        sub = seg[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] == label
        profile = profiles[label]
        z_span = z1 - z0 + 1
        z_slice_count = len(profile)
        multi_count, evaluated_count, multi_fraction = _multi_component_stats(
            sub,
            config.multi_component_min_voxels,
            config.multi_component_slice_step,
        )
        component_count, significant_component_count = _component_stats(
            sub,
            config.disconnected_component_min_voxels,
        )
        cross_section_areas = np.fromiter(profile.values(), dtype=np.int64)
        is_long_enough = z_span >= long_span
        records.append(
            TubeRecord(
                label=label,
                voxel_count=int(sizes[label]),
                z_min=z0,
                z_max=z1,
                z_span=z_span,
                z_slice_count=z_slice_count,
                z_occupancy_fraction=_safe_fraction(z_slice_count, z_span),
                is_long_enough=is_long_enough,
                is_decent=is_long_enough and int(sizes[label]) >= config.decent_min_voxels,
                face_contacts=contacts[label],
                border_end_count=_border_end_count(
                    sub,
                    label_bounds,
                    volume_shape,
                    config.border_margin,
                ),
                border_patch_count=_border_patch_count(
                    sub,
                    label_bounds,
                    volume_shape,
                    config.border_margin,
                    config.border_patch_min_voxels,
                ),
                median_cross_section_area=float(np.median(cross_section_areas)),
                max_cross_section_area=int(cross_section_areas.max()),
                bump_count=(
                    len(_area_bumps(profile, config))
                    if z_slice_count >= config.bump_min_slices
                    else 0
                ),
                multi_component_sample_count=multi_count,
                evaluated_sample_count=evaluated_count,
                multi_component_fraction=multi_fraction,
                is_parallel=(
                    multi_count
                    >= ceil(config.parallel_min_slices / config.multi_component_slice_step)
                    and multi_fraction > config.parallel_fraction_threshold
                ),
                component_count_3d=component_count,
                significant_component_count_3d=significant_component_count,
                is_disconnected=significant_component_count >= 2,
            )
        )
    total_foreground_voxels = sum(int(sizes[label]) for label in bounds)
    return TubeAnalysis(
        volume_shape=volume_shape,
        config=config,
        total_label_count=len(bounds),
        total_foreground_voxels=total_foreground_voxels,
        tubes=tuple(records),
    )


def format_tube_analysis(analysis: TubeAnalysis, top_incomplete: int = 8) -> str:
    """Format a concise human-readable tube analysis report."""

    if top_incomplete < 0:
        raise ValueError(f"top_incomplete must be non-negative, got {top_incomplete}")
    summary = analysis.summary
    lines = [
        (
            f"Tube analysis: {summary.total_label_count} labels, "
            f"{summary.substantial_count} substantial "
            f"(z>={analysis.config.substantial_min_z_slices} slices or "
            f"voxels>={analysis.config.substantial_min_voxels})"
        ),
        (
            f"  long enough: {summary.long_enough_count}/{summary.substantial_count}; "
            f"decent (long and voxels>={analysis.config.decent_min_voxels}): "
            f"{summary.decent_count}"
        ),
        (
            f"  COMPLETE (>=2 border ends): {summary.complete_count}/{summary.decent_count} "
            f"({100.0 * summary.complete_fraction:.1f}%, "
            f"volume {100.0 * summary.complete_volume_fraction:.1f}%) | "
            f"INCOMPLETE: {summary.decent_count - summary.complete_count}"
        ),
        (
            f"  VALID (complete and single): {summary.valid_count}/{summary.decent_count} "
            f"({100.0 * summary.valid_fraction:.1f}%, "
            f"volume {100.0 * summary.valid_volume_fraction:.1f}%)"
        ),
        (
            f"  quality flags among substantial tubes: parallel {summary.parallel_count}, "
            f"disconnected {summary.disconnected_count}, bumped {summary.bumped_count}"
        ),
        f"  distinct-face histogram: {dict(enumerate(summary.face_count_histogram))}",
    ]
    incomplete = analysis.largest_incomplete(top_incomplete)
    if incomplete:
        lines.append("  largest incomplete decent tubes:")
        for tube in incomplete:
            lines.append(
                f"    seg {tube.label}: voxels {tube.voxel_count}, "
                f"z{tube.z_min}-{tube.z_max} ({tube.z_span} span), "
                f"border ends {tube.border_end_count}, "
                f"parallel {tube.is_parallel}, disconnected {tube.is_disconnected}"
            )
    return "\n".join(lines)


def completeness_report(
    seg: np.ndarray,
    verbose_top: int = 8,
    stats: Any = None,
) -> tuple[int, int]:
    """Print and return the complete/decent tube counts.

    This preserves the original public completeness entry point while using
    the richer canonical tube analysis.
    """

    analysis = analyze_tubes(seg, stats=stats)
    print(format_tube_analysis(analysis, top_incomplete=verbose_top), flush=True)
    summary = analysis.summary
    return summary.complete_count, summary.decent_count


__all__ = [
    "BORDER",
    "MIN_SIZE",
    "MIN_SPAN_FRAC",
    "TubeAnalysis",
    "TubeAnalysisConfig",
    "TubeAnalysisSummary",
    "TubeRecord",
    "analyze_tubes",
    "completeness_report",
    "format_tube_analysis",
]
