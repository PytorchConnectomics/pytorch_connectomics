"""Combine no-GT profile and arbor candidates without asserting cell correctness.

This module owns the class decision only. Callers own segment identifiers,
semantic-review provenance, schemas, reports and serialization. A reviewed class
is a display override; it never changes the independently computed geometry class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

from .arbor import ArborMetrics, BackboneConfig, is_dendrite_backbone_candidate
from .morphology import MORPHOLOGY_CLASSES

__all__ = ["SegmentClassificationConfig", "SegmentClassification", "classify_segment"]


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
