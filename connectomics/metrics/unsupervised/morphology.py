"""Physical morphology diagnostics for instance labels without ground truth.

These measurements rank hypotheses; they cannot establish cell identity, detect
all false merges, or measure NERL. Profiles follow the dominant coordinate of
the physical PCA axis, so they also support lateral and oblique objects. They
are inexpensive approximations to centerlines, not medial-axis skeletons.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import cc3d
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label as label_components

__all__ = [
    "MORPHOLOGY_CLASSES",
    "MorphologyConfig",
    "MorphologyRecord",
    "MorphologyAnalysis",
    "analyze_morphology",
]

MORPHOLOGY_CLASSES = (
    "good_axon_candidate",
    "broken_axon_candidate",
    "suspicious_axon_candidate",
    "dendrite_like_candidate",
    "branched_process_candidate",
    "small_interior_fragment_candidate",
    "unclassified",
)
_FACE_NAMES = (("z0", "zmax"), ("y0", "ymax"), ("x0", "xmax"))
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MorphologyConfig:
    """Physical thresholds; ``voxel_size_um`` and all axes use z, y, x order.

    ``radius_bins_um`` contains finite lower edges; the last bin is open.
    ``profile_smoothing_um`` is the Gaussian sigma along the profile axis.
    Connected components use 26-neighborhoods in 3D and 8-neighborhoods in 2D.
    """

    voxel_size_um: tuple[float, float, float]
    min_voxels: int = 1000
    min_axon_length_um: float = 2.0
    max_axon_radius_um: float = 0.35
    min_elongation: float = 3.0
    border_margin_um: float = 0.08
    component_min_voxels: int = 100
    multi_component_min_area_um2: float = 0.005
    multi_component_fraction: float = 0.2
    profile_smoothing_um: float = 0.2
    radius_bins_um: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 1.0)

    def __post_init__(self) -> None:
        spacing = np.asarray(self.voxel_size_um, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
            raise ValueError("voxel_size_um must contain three finite positive values")
        object.__setattr__(self, "voxel_size_um", tuple(float(x) for x in spacing))
        for name in ("min_voxels", "component_min_voxels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "min_axon_length_um",
            "max_axon_radius_um",
            "min_elongation",
            "multi_component_min_area_um2",
            "multi_component_fraction",
            "border_margin_um",
            "profile_smoothing_um",
        ):
            value = float(getattr(self, name))
            lower_inclusive = name in ("border_margin_um", "profile_smoothing_um")
            if not np.isfinite(value) or (value < 0 if lower_inclusive else value <= 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if lower_inclusive else 'positive'}"
                )
            object.__setattr__(self, name, value)
        if self.min_elongation < 1:
            raise ValueError("min_elongation must be at least one")
        if self.multi_component_fraction > 1:
            raise ValueError("multi_component_fraction must be at most one")
        bins = np.asarray(self.radius_bins_um, dtype=float)
        if (
            bins.ndim != 1
            or bins.size == 0
            or bins[0] != 0
            or not np.all(np.isfinite(bins))
            or not np.all(np.diff(bins) > 0)
        ):
            raise ValueError(
                "radius_bins_um must have finite increasing lower edges starting at zero"
            )
        object.__setattr__(self, "radius_bins_um", tuple(float(x) for x in bins))


@dataclass(frozen=True)
class MorphologyRecord:
    """Measurements for one retained label; classes are deliberately tentative.

    Length sums component profile paths, including a half-section at each end.
    The chord, endpoints, and continuity describe its longest component.
    Tortuosity is total path length divided by the sum of component chords.
    Radius is the median equivalent radius of occupied, orientation-corrected
    sections of the whole label; branching can consequently inflate it.
    """

    label: int
    voxel_count: int
    volume_um3: float
    bbox_zyx: tuple[tuple[int, int], ...]
    centroid_um_zyx: tuple[float, float, float]
    principal_axis_zyx: tuple[float, float, float]
    axis_angles_deg_zyx: tuple[float, float, float]
    dominant_axis: int
    elongation: float
    radius_um: float
    length_um: float
    component_lengths_um: tuple[float, ...]
    chord_length_um: float
    tortuosity: float
    expected_crop_chord_um: float
    continuity_proxy: float
    component_count: int
    significant_component_count: int
    multistrand_fraction: float
    profile_area_cv: float
    boundary_end_count: int
    boundary_end_faces: tuple[str, ...]
    touches_boundary: bool
    suspicious_geometry: bool
    morphology_class: str


@dataclass(frozen=True)
class MorphologyAnalysis:
    records: tuple[MorphologyRecord, ...]
    summary: dict[str, Any]
    config: MorphologyConfig
    volume_shape: tuple[int, ...]


def _float_triplet(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _label_statistics(seg: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """Avoid max-ID-sized statistics arrays for sparse integer label spaces."""

    maximum = int(seg.max(initial=0))
    if maximum >= seg.size or maximum > 1_000_000:
        ids = np.unique(seg)
        if not ids.size or ids[0] != 0:
            ids = np.concatenate((np.zeros(1, dtype=seg.dtype), ids))
        compact = np.empty(seg.shape, dtype=np.uint32 if ids.size < 2**32 else np.uint64)
        # Slice-wise search avoids a volume-sized int64 inverse-index array.
        for z in range(seg.shape[0]):
            compact[z] = np.searchsorted(ids, seg[z])
        seg = compact
    else:
        ids = np.arange(maximum + 1)
    return seg, cc3d.statistics(seg), ids


def _pca(coords: tuple[np.ndarray, ...], spacing: np.ndarray) -> tuple:
    centered = [values.astype(float) * step for values, step in zip(coords, spacing)]
    center = np.array([values.mean() for values in centered])
    for values, mean in zip(centered, center):
        values -= mean
    covariance = np.array([[np.dot(a, b) for b in centered] for a in centered])
    covariance /= len(coords[0])
    # Uniform voxel support keeps even a one-voxel-wide tube finite and physical.
    covariance += np.diag(spacing**2 / 12)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0:
        direction = -direction
    return center, direction, dominant, float(np.sqrt(eigenvalues[-1] / eigenvalues[-2]))


def _profile_path(
    coords: tuple[np.ndarray, ...],
    origin: np.ndarray,
    spacing: np.ndarray,
    dominant: int,
    direction: np.ndarray,
    smoothing_um: float,
) -> tuple[float, float, np.ndarray]:
    section = coords[dominant]
    counts = np.bincount(section)
    occupied = np.flatnonzero(counts)
    points = np.column_stack(
        [
            (np.bincount(section, weights=values)[occupied] / counts[occupied] + start + 0.5) * step
            for values, start, step in zip(coords, origin, spacing)
        ]
    )
    # A 26-connected component has no missing dominant-axis sections. Keeping
    # runs explicit also prevents interpolation if this helper gets sparse data.
    runs = np.split(np.arange(len(occupied)), np.flatnonzero(np.diff(occupied) > 1) + 1)
    path_length = 0.0
    chord_length = 0.0
    endpoints = np.stack((points[0], points[-1]))
    for run in runs:
        curve = points[run].copy()
        if len(curve) > 1 and smoothing_um > 0:
            for axis in range(3):
                if axis != dominant:
                    curve[:, axis] = gaussian_filter1d(
                        curve[:, axis], smoothing_um / spacing[dominant], mode="nearest"
                    )
        half_section = spacing[dominant] / (2 * abs(direction[dominant]))
        if len(curve) > 1:
            first = curve[1] - curve[0]
            last = curve[-1] - curve[-2]
            first /= np.linalg.norm(first)
            last /= np.linalg.norm(last)
        else:
            first = last = direction
        start = curve[0] - first * half_section
        end = curve[-1] + last * half_section
        path_length += float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        path_length += 2 * half_section
        chord_length += float(np.linalg.norm(end - start))
    return path_length, chord_length, endpoints


def _crop_chord(center: np.ndarray, direction: np.ndarray, extent: np.ndarray) -> float:
    active = np.abs(direction) > 1e-10
    ends = np.stack((-center[active], extent[active] - center[active])) / direction[active]
    lower = np.min(ends, axis=0).max()
    upper = np.max(ends, axis=0).min()
    return float(max(0.0, upper - lower))


def _terminal_contacts(
    endpoints: np.ndarray,
    direction: np.ndarray,
    extent: np.ndarray,
    spacing: np.ndarray,
    margin: float,
) -> tuple[int, tuple[str, ...]]:
    """Require two separated, outward-directed ends on different crop faces.

    Contacts use terminal section centroids, not arbitrary side-wall voxels.
    Outward direction and distinct faces prevent a tube lying along one volume
    side from acquiring two fictitious ends. Ends closer than twice the contact
    tolerance cannot establish distinct terminal patches.
    """

    contacts: list[list[str]] = [[], []]
    tolerance = margin + spacing / 2
    for end_index, point in enumerate(endpoints):
        outward: np.ndarray = direction * (-1 if end_index == 0 else 1)
        for axis in range(3):
            if outward[axis] < -1e-6 and point[axis] <= tolerance[axis]:
                contacts[end_index].append(_FACE_NAMES[axis][0])
            if outward[axis] > 1e-6 and extent[axis] - point[axis] <= tolerance[axis]:
                contacts[end_index].append(_FACE_NAMES[axis][1])
    separated = np.linalg.norm(endpoints[1] - endpoints[0]) > 2 * float(tolerance.max())
    if separated:
        for first in contacts[0]:
            for last in contacts[1]:
                if first != last:
                    return 2, (first, last)
    faces = sorted(set(contacts[0] + contacts[1]))
    return (1 if faces else 0), tuple(faces[:1])


def _classify(
    config: MorphologyConfig,
    elongation: float,
    radius: float,
    length: float,
    boundary_ends: int,
    touches_boundary: bool,
    suspicious: bool,
    multistrand: float,
) -> str:
    elongated = elongation >= config.min_elongation
    thin = radius <= config.max_axon_radius_um
    substantial = length >= config.min_axon_length_um
    if elongated and thin and substantial:
        if suspicious:
            return "suspicious_axon_candidate"
        return "good_axon_candidate" if boundary_ends == 2 else "broken_axon_candidate"
    if elongated and not thin and substantial:
        return "dendrite_like_candidate"
    if not elongated and multistrand >= config.multi_component_fraction:
        # Dendritic arbors and glial processes can both have this geometry.
        return "branched_process_candidate"
    if not substantial and thin and not touches_boundary:
        return "small_interior_fragment_candidate"
    return "unclassified"


def _analyze_label(
    seg: np.ndarray, value: int, original_id: int, bbox: tuple, config: MorphologyConfig
) -> MorphologyRecord:
    mask = seg[bbox] == value
    coords = np.nonzero(mask)
    count = len(coords[0])
    origin = np.array([sl.start for sl in bbox])
    spacing = np.asarray(config.voxel_size_um)
    center, direction, dominant, elongation = _pca(coords, spacing)
    center += (origin + 0.5) * spacing
    areas = np.bincount(coords[dominant])
    occupied = np.flatnonzero(areas)
    voxel_area = float(np.prod(np.delete(spacing, dominant)) * abs(direction[dominant]))
    corrected_areas = areas[occupied] * voxel_area
    radius = float(np.sqrt(np.median(corrected_areas) / np.pi))
    multi_sections = 0
    for section in occupied:
        plane = np.take(mask, section, axis=dominant)
        components_2d, _ = label_components(plane, structure=np.ones((3, 3)))
        counts_2d = np.bincount(components_2d.ravel())[1:]
        multi_sections += int(
            np.count_nonzero(counts_2d * voxel_area >= config.multi_component_min_area_um2) >= 2
        )
    multistrand = float(multi_sections / len(occupied))
    component_labels, component_count = cc3d.connected_components(
        mask, connectivity=26, return_N=True
    )
    component_stats = cc3d.statistics(component_labels)
    component_sizes = component_stats["voxel_counts"][1:]
    significant_count = int(np.count_nonzero(component_sizes >= config.component_min_voxels))
    paths = []
    for component_id in range(1, component_count + 1):
        if component_count == 1:
            component_coords = coords
            component_origin = origin
        else:
            component_bbox = component_stats["bounding_boxes"][component_id]
            component_coords = np.nonzero(component_labels[component_bbox] == component_id)
            component_origin = origin + np.array([sl.start for sl in component_bbox])
        paths.append(
            _profile_path(
                component_coords,
                component_origin,
                spacing,
                dominant,
                direction,
                config.profile_smoothing_um,
            )
        )
    longest = max(paths, key=lambda path: path[0])
    length = sum(path[0] for path in paths)
    summed_chords = sum(path[1] for path in paths)
    crop_chord = _crop_chord(center, direction, np.asarray(seg.shape) * spacing)
    boundary_ends, faces = _terminal_contacts(
        longest[2], direction, np.asarray(seg.shape) * spacing, spacing, config.border_margin_um
    )
    touches_boundary = bool(
        np.any(origin * spacing <= config.border_margin_um)
        or np.any(
            (np.asarray(seg.shape) - np.array([sl.stop for sl in bbox])) * spacing
            <= config.border_margin_um
        )
    )
    suspicious = significant_count > 1 or multistrand >= config.multi_component_fraction
    return MorphologyRecord(
        label=original_id,
        voxel_count=count,
        volume_um3=float(count * np.prod(spacing)),
        bbox_zyx=tuple((int(sl.start), int(sl.stop)) for sl in bbox),
        centroid_um_zyx=_float_triplet(center),
        principal_axis_zyx=_float_triplet(direction),
        axis_angles_deg_zyx=_float_triplet(np.degrees(np.arccos(np.abs(direction).clip(0, 1)))),
        dominant_axis=dominant,
        elongation=elongation,
        radius_um=radius,
        length_um=float(length),
        component_lengths_um=tuple(float(path[0]) for path in paths),
        chord_length_um=float(longest[1]),
        tortuosity=float(length / summed_chords),
        expected_crop_chord_um=crop_chord,
        continuity_proxy=float(np.clip(longest[1] / crop_chord, 0, 1)) if crop_chord else 0.0,
        component_count=int(component_count),
        significant_component_count=significant_count,
        multistrand_fraction=multistrand,
        profile_area_cv=float(corrected_areas.std() / corrected_areas.mean()),
        boundary_end_count=boundary_ends,
        boundary_end_faces=faces,
        touches_boundary=touches_boundary,
        suspicious_geometry=suspicious,
        morphology_class=_classify(
            config,
            elongation,
            radius,
            longest[0],
            boundary_ends,
            touches_boundary,
            suspicious,
            multistrand,
        ),
    )


def _summary(
    records: tuple[MorphologyRecord, ...],
    config: MorphologyConfig,
    shape: tuple[int, ...],
    total_labels: int,
    foreground_voxels: int,
) -> dict[str, Any]:
    analyzed_voxels = sum(record.voxel_count for record in records)
    bins = []
    for index, lower in enumerate(config.radius_bins_um):
        upper = config.radius_bins_um[index + 1] if index + 1 < len(config.radius_bins_um) else None
        group = [
            r for r in records if r.radius_um >= lower and (upper is None or r.radius_um < upper)
        ]
        lengths = [length for record in group for length in record.component_lengths_um]
        total_length = sum(lengths)
        bins.append(
            {
                "lower_um": lower,
                "upper_um": upper,
                "count": len(group),
                "component_count": len(lengths),
                "total_length_um": total_length,
                "length_weighted_segment_length_um": (
                    sum(length**2 for length in lengths) / total_length if total_length else None
                ),
                "length_weighted_continuity": (
                    sum(r.length_um * r.continuity_proxy for r in group) / total_length
                    if total_length
                    else None
                ),
            }
        )
    angles = [r.axis_angles_deg_zyx[0] for r in records]
    return {
        "volume_shape_zyx": [int(x) for x in shape],
        "voxel_size_um": list(config.voxel_size_um),
        "config": asdict(config),
        "coverage": {
            "total_label_count": total_labels,
            "total_foreground_voxels": foreground_voxels,
            "analyzed_label_count": len(records),
            "analyzed_voxels": analyzed_voxels,
            "analyzed_label_fraction": len(records) / total_labels if total_labels else 0.0,
            "analyzed_foreground_fraction": (
                analyzed_voxels / foreground_voxels if foreground_voxels else 0.0
            ),
            "excluded_label_count": total_labels - len(records),
            "excluded_voxels": foreground_voxels - analyzed_voxels,
        },
        "class_counts": {
            name: sum(r.morphology_class == name for r in records) for name in MORPHOLOGY_CLASSES
        },
        "class_volume_um3": {
            name: sum(r.volume_um3 for r in records if r.morphology_class == name)
            for name in MORPHOLOGY_CLASSES
        },
        "suspicious_geometry_count": sum(r.suspicious_geometry for r in records),
        "radius_bins": bins,
        "angle_degrees_to_z": {
            "median": float(np.median(angles)) if angles else None,
            "p10": float(np.percentile(angles, 10)) if angles else None,
            "p90": float(np.percentile(angles, 90)) if angles else None,
        },
        "nerl": None,
        "nerl_unavailable_reason": "NERL requires reference skeletons and their physical lengths.",
        "interpretation": [
            "Classes are exclusive geometric candidates, not biological or error-rate estimates.",
            "Good axon candidates have two separated terminal contacts on different crop faces; "
            "clean geometry cannot exclude an end-to-end false merge.",
            "Broken candidates may be natural terminations or crop effects; small interior "
            "fragments are not confirmed detached spines.",
            "Branched process and thick/dendrite-like shapes do not establish semantic identity.",
            "Radius uses median section area times the dominant PCA component before sqrt(A/pi).",
            "Lengths are smoothed component profile-centroid paths, not skeleton lengths; "
            "branching, folds and parallel strands make these approximate.",
            "Continuity is the longest component chord / PCA line chord through the crop, "
            "clipped to [0,1]; it is not NERL.",
            "Radius-bin length-weighted segment length is sum(component_length^2) / "
            "sum(component_length), not expected run length against a reference.",
            "Radius bins include all analyzed morphology classes; angle percentiles are "
            "unweighted over analyzed labels, and PCA directions are undirected.",
        ],
    }


def analyze_morphology(seg: np.ndarray, config: MorphologyConfig) -> MorphologyAnalysis:
    """Analyze integer 3D labels with 0 as background, retaining original IDs.

    Labels smaller than ``min_voxels`` are counted for coverage but not measured.
    Processing is native resolution and label-local; no volume skeleton or
    reference labels are required. Negative labels and noninteger arrays raise.
    """

    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError("seg must be a 3D label array in z, y, x order")
    if not np.issubdtype(seg.dtype, np.integer) or np.issubdtype(seg.dtype, np.bool_):
        raise ValueError("seg must contain integer instance labels")
    if np.issubdtype(seg.dtype, np.signedinteger) and np.any(seg < 0):
        raise ValueError("seg must not contain negative labels")
    if not isinstance(config, MorphologyConfig):
        raise TypeError("config must be MorphologyConfig")
    if seg.size == 0:
        return MorphologyAnalysis((), _summary((), config, seg.shape, 0, 0), config, seg.shape)
    compact, statistics, original_ids = _label_statistics(seg)
    counts = statistics["voxel_counts"]
    nonzero_ids = np.flatnonzero(counts[1:]) + 1
    foreground = int(sum(int(counts[value]) for value in nonzero_ids))
    selected = nonzero_ids[counts[nonzero_ids] >= config.min_voxels]
    measured = []
    for index, value in enumerate(selected):
        measured.append(
            _analyze_label(
                compact,
                int(value),
                int(original_ids[value]),
                statistics["bounding_boxes"][value],
                config,
            )
        )
        if (index + 1) % 1000 == 0 or index + 1 == len(selected):
            _LOGGER.info("Morphology: measured %s / %s labels", index + 1, len(selected))
    records = tuple(measured)
    return MorphologyAnalysis(
        records,
        _summary(records, config, seg.shape, len(nonzero_ids), foreground),
        config,
        seg.shape,
    )
