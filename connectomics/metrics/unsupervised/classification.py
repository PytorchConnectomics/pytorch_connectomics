"""Combine no-GT profile and arbor candidates without asserting cell correctness.

This module owns the class decision only. Callers own segment identifiers,
semantic-review provenance, schemas, reports and serialization. A reviewed class
is a display override; it never changes the independently computed geometry class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from typing import Literal

from .arbor import ArborMetrics, BackboneConfig, is_dendrite_backbone_candidate
from .morphology import MORPHOLOGY_CLASSES

__all__ = [
    "SegmentClassificationConfig",
    "SegmentClassification",
    "classify_segment",
    "classify_semantic_candidate",
    "classify_end_quality",
]


def classify_semantic_candidate(
    semantic_type: str | None,
    caliber_radius_um: float | None,
    shaft_length_um: float,
    *,
    axon_max_radius_um: float,
    dendrite_min_radius_um: float,
    skeleton_length_um: float | None = None,
    min_dendrite_length_um: float = 2.0,
    volume_um3: float | None = None,
    min_axon_length_um: float = 3.0,
    min_axon_volume_um3: float = 0.1,
) -> tuple[str, str]:
    """Return a coarse semantic candidate and its geometric evidence basis.

    Caliber gates must be calibrated by the caller. Thin shaft evidence retains
    branched/swollen axon candidates without requiring whole-object elongation.
    Dense branching alone does not establish dendrite identity. Thick caliber on a
    short skeleton does not either: an axon terminal is a big blob, so a detached
    one reads thick. Axon candidates include their terminals; neuronal somata
    belong to dendrite. An axon call also needs `min_axon_length_um` of skeleton
    OR `min_axon_volume_um3` of volume: a thin process can be long with little
    volume. Short AND small is a fragment -- often a false split -- and stays
    unclassified. These rules do
    not identify vessels or glia/soma, or certify segmentation correctness.
    Inputs are measurements, independent of catalog schemas and segment IDs.
    """
    if not (
        isfinite(axon_max_radius_um)
        and isfinite(dendrite_min_radius_um)
        and 0 < axon_max_radius_um <= dendrite_min_radius_um
    ):
        raise ValueError("Caliber gates must be finite, positive and ordered")
    if caliber_radius_um is None:
        return "unclassified", "no_local_caliber"
    fragment = (skeleton_length_um is not None and skeleton_length_um < min_axon_length_um) and (
        volume_um3 is not None and volume_um3 < min_axon_volume_um3
    )
    if semantic_type == "axon_like":
        if fragment:
            return "unclassified", "short_and_small_fragment"
        return "axon", "local_axon_caliber"
    if caliber_radius_um <= axon_max_radius_um and shaft_length_um > 0:
        if fragment:
            return "unclassified", "short_and_small_fragment"
        return "axon", "thin_shaft_with_branches_or_swellings"
    if caliber_radius_um >= dendrite_min_radius_um:
        if skeleton_length_um is not None and skeleton_length_um < min_dendrite_length_um:
            return "unclassified", "thick_short_blob_terminal_or_dendrite"
        return "dendrite", "thick_backbone_candidate"
    if semantic_type == "dendrite_like":
        return "unclassified", "branch_density_without_thick_backbone"
    return "unclassified", "ambiguous_caliber"


def classify_end_quality(
    semantic_class: str,
    free_ends: list[dict],
    *,
    measured: bool,
    site_radius_um: float = 1.0,
) -> tuple[str, str, int | None]:
    """Starting completeness call from skeleton ends: (quality, basis, end sites).

    Complete means every end is explained, however branched the object: an end
    at the volume border (censored out of `free_ends`, or flagged `at_border` by
    segmentation evidence), a surface `spur`, or, for an axon, an axon terminal
    (flagged `terminal`, or a `bouton_head` tip). Unexplained ends within
    `site_radius_um` of each other are one end site -- skeletonization leaves
    hair-thin spurs in clusters, so twenty tips around one spot are one place
    the object stops. Any unexplained site is a false split candidate. Merges
    are never called here. Only axon and dendrite are judged by their ends.
    """
    if semantic_class not in ("axon", "dendrite"):
        return "unknown", "not_judged_by_ends", None
    if not measured:
        return "unknown", "no_skeleton", None
    def explained(end: dict) -> bool:
        if end.get("spur") or end.get("at_border"):
            return True
        return semantic_class == "axon" and (
            bool(end.get("terminal")) or end.get("shape") == "bouton_head"
        )

    positions = [e["position_um_zyx"] for e in free_ends if not explained(e)]
    parent = list(range(len(positions)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    limit = site_radius_um**2
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])) <= limit:
                parent[root(i)] = root(j)
    sites = len({root(i) for i in range(len(positions))})
    if sites == 0:
        return "complete", "all_ends_at_border_or_axon_terminal", 0
    return "false_split", "unexplained_end_inside_volume", sites


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class SegmentClassificationConfig:
    """Size and backbone gates; one voxel disables the crumbs size category.

    A crumb contains strictly fewer than ``crumbs_max_voxels_exclusive`` voxels.
    This threshold uses the caller's grid; adjust it when voxel size changes.
    Backbone thresholds use micrometers and remain provisional geometric gates.
    """

    crumbs_max_voxels_exclusive: int = 1000
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "crumbs_max_voxels_exclusive",
            _positive_integer(self.crumbs_max_voxels_exclusive, "crumbs_max_voxels_exclusive"),
        )
        if not isinstance(self.backbone, BackboneConfig):
            raise TypeError("backbone must be a BackboneConfig")


@dataclass(frozen=True)
class SegmentClassification:
    """Class decision suitable for ``dataclasses.asdict`` and JSON serialization.

    ``morphology_class`` is the profile-only candidate. ``automatic_class`` also
    incorporates any supplied backbone evidence. ``display_class`` applies size
    precedence, then an optional reviewed class, then the automatic candidate.
    No field certifies biological identity, a true split/merge, or correctness.
    """

    morphology_class: str
    automatic_class: str
    display_class: str
    automatic_basis: Literal["profile_geometry", "backbone_geometry"]
    backbone_assessed: bool
    backbone_candidate: bool
    display_basis: Literal["size", "user_annotation", "automatic"]


def classify_segment(
    morphology_class: str,
    voxel_count: int,
    *,
    arbor: ArborMetrics | None = None,
    reviewed_class: str | None = None,
    config: SegmentClassificationConfig | None = None,
) -> SegmentClassification:
    """Refine a profile candidate with optional physical backbone measurements.

    ``morphology_class`` must be a canonical value from ``MORPHOLOGY_CLASSES``;
    legacy schema translation belongs in the importing application. ``arbor`` is
    produced by ``unsupervised.arbor.analyze_arbor`` for the same segment.
    The caller chooses which segments warrant skeleton analysis.

    A passing backbone gate supplies ``dendrite_like_candidate`` evidence without
    requiring global PCA elongation. If it fails or is unavailable, the original
    profile candidate is retained, including an existing dendrite candidate.
    Pruning preserves the diameter and its caliber, so pruning thresholds alone
    cannot change this backbone decision. Trees do not establish neuron identity.

    ``reviewed_class`` may be a caller-defined semantic label (e.g. ``glia``),
    but cannot be ``crumbs``: crumbs is reserved for the size rule. Callers must
    retain the review's provenance separately. Even an annotated segment remains
    a crumb below the configured threshold. No segment IDs or dataset paths are
    involved in any decision.
    """
    count = _positive_integer(voxel_count, "voxel_count")
    if morphology_class not in MORPHOLOGY_CLASSES:
        raise ValueError(f"Unknown morphology_class: {morphology_class!r}")
    if reviewed_class is not None and (
        not isinstance(reviewed_class, str)
        or not reviewed_class.strip()
        or reviewed_class != reviewed_class.strip()
        or reviewed_class == "crumbs"
    ):
        raise ValueError("reviewed_class must be a nonempty class name other than 'crumbs'")
    if config is None:
        config = SegmentClassificationConfig()
    elif not isinstance(config, SegmentClassificationConfig):
        raise TypeError("config must be a SegmentClassificationConfig")
    if arbor is not None and not isinstance(arbor, ArborMetrics):
        raise TypeError("arbor must be ArborMetrics or None")
    passes = arbor is not None and is_dendrite_backbone_candidate(arbor, config.backbone)
    automatic = "dendrite_like_candidate" if passes else morphology_class
    display_basis: Literal["size", "user_annotation", "automatic"]
    if count < config.crumbs_max_voxels_exclusive:
        display, display_basis = "crumbs", "size"
    elif reviewed_class is not None:
        display, display_basis = reviewed_class, "user_annotation"
    else:
        display, display_basis = automatic, "automatic"
    return SegmentClassification(
        morphology_class=morphology_class,
        automatic_class=automatic,
        display_class=display,
        automatic_basis="backbone_geometry" if passes else "profile_geometry",
        backbone_assessed=arbor is not None,
        backbone_candidate=passes,
        display_basis=display_basis,
    )
