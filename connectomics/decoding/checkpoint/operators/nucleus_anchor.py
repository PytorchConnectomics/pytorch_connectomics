"""Nucleus-instance description, certification, and deterministic policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import (
    full_bbox,
    iter_bounded_tiles,
    open_volume,
    read_bbox_chunked,
    volume_artifact_ref,
)
from ..schema import (
    ActionSpec,
    AnchorTotals,
    ArtifactRef,
    BoundingBox,
    Certificate,
    CheckpointPlan,
    Condition,
    ConsolidateSameAnchorParams,
    ContactScope,
    ContactScopes,
    DescriptionBundle,
    Descriptor,
    EntityRef,
    ForbidMergeParams,
    Provenance,
    RebuildLocalRagParams,
    SplitByAnchorParams,
)
from ..serialization import content_hash, sha256_file, stable_id, write_json


@dataclass(frozen=True)
class NucleusAnchorConfig:
    checkpoint_id: str
    pass_id: str
    segmentation_uri: str
    nucleus_uri: str
    affinity_uri: str
    scope: BoundingBox
    min_share: float
    channel_indices: tuple[int, ...]
    contact_scopes_uri: str | None
    segmentation_dataset: str = "main"
    nucleus_dataset: str = "main"
    affinity_dataset: str = "main"
    affinity_channel_axis: int = 0
    affinity_convention: str = "probability"
    sigmoid_restore: float | None = None
    pooling_factor: int = 1
    nucleus_scale_zyx: tuple[int, int, int] = (1, 1, 1)
    max_read_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 0.0 < self.min_share <= 1.0:
            raise ValueError("min_share is required and must lie in (0, 1]")
        if not self.channel_indices:
            raise ValueError("channel_indices must be explicit and non-empty")
        if self.affinity_channel_axis not in (0, -1, 3):
            raise ValueError("affinity_channel_axis must be 0 or -1")
        if self.affinity_convention not in ("probability", "deepem", "banis"):
            raise ValueError("affinity_convention must be probability, deepem, or banis")
        if self.pooling_factor < 1 or self.max_read_bytes <= 0:
            raise ValueError("pooling_factor and max_read_bytes must be positive")
        if len(self.nucleus_scale_zyx) != 3 or any(value < 1 for value in self.nucleus_scale_zyx):
            raise ValueError("nucleus_scale_zyx must contain three positive integers")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], *, min_share: float) -> "NucleusAnchorConfig":
        allowed = {
            "checkpoint_id",
            "pass_id",
            "operator",
            "segmentation",
            "nuclei",
            "affinity",
            "scope_zyx",
            "contact_scopes",
            "channel_indices",
            "affinity_channel_axis",
            "affinity_convention",
            "sigmoid_restore",
            "pooling_factor",
            "nucleus_scale_zyx",
            "max_read_bytes",
        }
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"unknown nucleus-anchor specification fields: {sorted(unknown)}")
        if spec.get("operator") != "nucleus_anchor":
            raise ValueError("operator must be nucleus_anchor")

        def artifact(name: str) -> tuple[str, str]:
            value = spec.get(name)
            if not isinstance(value, Mapping) or not isinstance(value.get("uri"), str):
                raise ValueError(f"{name} must contain a uri")
            extra = set(value) - {"uri", "dataset"}
            if extra:
                raise ValueError(f"unknown {name} fields: {sorted(extra)}")
            return value["uri"], str(value.get("dataset", "main"))

        seg_uri, seg_dataset = artifact("segmentation")
        nuc_uri, nuc_dataset = artifact("nuclei")
        aff_uri, aff_dataset = artifact("affinity")
        scope = spec.get("scope_zyx")
        if not isinstance(scope, Sequence) or len(scope) != 6:
            raise ValueError("scope_zyx must be [z0, y0, x0, z1, y1, x1]")
        return cls(
            checkpoint_id=str(spec.get("checkpoint_id", "checkpoint")),
            pass_id=str(spec.get("pass_id", "nucleus_anchor")),
            segmentation_uri=seg_uri,
            segmentation_dataset=seg_dataset,
            nucleus_uri=nuc_uri,
            nucleus_dataset=nuc_dataset,
            affinity_uri=aff_uri,
            affinity_dataset=aff_dataset,
            scope=BoundingBox(tuple(int(v) for v in scope[:3]), tuple(int(v) for v in scope[3:])),
            min_share=float(min_share),
            channel_indices=tuple(int(v) for v in spec.get("channel_indices", ())),
            contact_scopes_uri=(
                str(spec["contact_scopes"]) if spec.get("contact_scopes") else None
            ),
            affinity_channel_axis=int(spec.get("affinity_channel_axis", 0)),
            affinity_convention=str(spec.get("affinity_convention", "probability")),
            sigmoid_restore=(
                float(spec["sigmoid_restore"]) if spec.get("sigmoid_restore") is not None else None
            ),
            pooling_factor=int(spec.get("pooling_factor", 1)),
            nucleus_scale_zyx=tuple(int(v) for v in spec.get("nucleus_scale_zyx", (1, 1, 1))),
            max_read_bytes=int(spec.get("max_read_bytes", 64 * 1024 * 1024)),
        )

    def as_hash_data(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "pass_id": self.pass_id,
            "segmentation_uri": self.segmentation_uri,
            "segmentation_dataset": self.segmentation_dataset,
            "nucleus_uri": self.nucleus_uri,
            "nucleus_dataset": self.nucleus_dataset,
            "affinity_uri": self.affinity_uri,
            "affinity_dataset": self.affinity_dataset,
            "scope": [*self.scope.start_zyx, *self.scope.stop_zyx],
            "min_share": self.min_share,
            "channel_indices": self.channel_indices,
            "contact_scopes_uri": self.contact_scopes_uri,
            "affinity_channel_axis": self.affinity_channel_axis,
            "affinity_convention": self.affinity_convention,
            "sigmoid_restore": self.sigmoid_restore,
            "pooling_factor": self.pooling_factor,
            "nucleus_scale_zyx": self.nucleus_scale_zyx,
            "max_read_bytes": self.max_read_bytes,
            "connectivity": 6,
            "tie_break": "lowest_anchor_id",
            "cost_channels": "min",
            "cost_pooling": "min",
            "mask_pooling": "max",
            "upsampling": "nearest",
        }


def load_contact_scopes(path: str | Path) -> ContactScopes:
    source = Path(path)
    raw = json.loads(source.read_text())
    allowed = {"schema_version", "artifact_id", "scopes"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown contact-scopes fields: {sorted(unknown)}")
    scopes = []
    for row in raw.get("scopes", []):
        bbox_values = row.get("bbox_zyx")
        bbox = None
        if bbox_values is not None:
            if len(bbox_values) != 6:
                raise ValueError("contact scope bbox_zyx must have six values")
            bbox = BoundingBox(tuple(bbox_values[:3]), tuple(bbox_values[3:]))
        scopes.append(
            ContactScope(
                seg_id=str(row["seg_id"]),
                anchor_ids=tuple(sorted((str(v) for v in row["anchor_ids"]), key=int)),
                provenance=str(row["provenance"]),
                gap_um=float(row["gap_um"]),
                eligibility=str(row["eligibility"]),  # type: ignore[arg-type]
                bbox=bbox,
            )
        )
    return ContactScopes(
        schema_version=str(raw.get("schema_version", "1.0")),
        artifact_id=str(raw.get("artifact_id") or stable_id("contact-scopes", scopes)),
        scopes=tuple(scopes),
    )


def _aligned_nuclei(config: NucleusAnchorConfig, reader: Any) -> np.ndarray:
    scale = np.asarray(config.nucleus_scale_zyx, dtype=np.int64)
    start = np.asarray(config.scope.start_zyx, dtype=np.int64)
    stop = np.asarray(config.scope.stop_zyx, dtype=np.int64)
    nucleus_start = start // scale
    nucleus_stop = (stop - 1) // scale + 1
    bbox = BoundingBox(tuple(nucleus_start), tuple(nucleus_stop))
    low_resolution = read_bbox_chunked(reader, bbox, config.max_read_bytes)
    coords = [
        np.arange(start[axis], stop[axis]) // scale[axis] - nucleus_start[axis] for axis in range(3)
    ]
    return np.asarray(low_resolution[np.ix_(*coords)])


def _component_contained(
    component: np.ndarray, scope: BoundingBox, volume_shape: tuple[int, int, int]
) -> bool:
    for axis in range(3):
        low_face = np.take(component, 0, axis=axis)
        high_face = np.take(component, -1, axis=axis)
        if scope.start_zyx[axis] > 0 and low_face.any():
            return False
        if scope.stop_zyx[axis] < volume_shape[axis] and high_face.any():
            return False
    return True


def _relative_slices(inner: BoundingBox, outer: BoundingBox) -> tuple[slice, slice, slice]:
    if any(
        inner_low < outer_low or inner_high > outer_high
        for inner_low, inner_high, outer_low, outer_high in zip(
            inner.start_zyx,
            inner.stop_zyx,
            outer.start_zyx,
            outer.stop_zyx,
        )
    ):
        raise ValueError(f"contact repair scope {inner} lies outside checkpoint scope {outer}")
    return tuple(
        slice(inner_low - outer_low, inner_high - outer_low)
        for inner_low, inner_high, outer_low in zip(
            inner.start_zyx, inner.stop_zyx, outer.start_zyx
        )
    )  # type: ignore[return-value]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class NucleusAnchorOperator:
    name = "nucleus_anchor"
    version = "1.0.0"

    def __init__(self, config: NucleusAnchorConfig) -> None:
        self.config = config
        self.configuration_hash = content_hash(config.as_hash_data())

    def _compute_anchor_totals(self, reader: Any) -> tuple[dict[str, int], str]:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "shape": reader.spatial_shape_zyx,
                    "dtype": str(reader.dtype),
                    "channel_axis": reader.channel_axis,
                },
                sort_keys=True,
            ).encode()
        )
        counts: dict[str, int] = {}
        for tile in iter_bounded_tiles(reader, full_bbox(reader), self.config.max_read_bytes):
            block = np.ascontiguousarray(reader.read(tile))
            if block.nbytes > self.config.max_read_bytes:
                raise RuntimeError("nucleus reader exceeded max_read_bytes")
            digest.update(block.view(np.uint8))
            labels, values = np.unique(block[block != 0], return_counts=True)
            for label, count in zip(labels.tolist(), values.tolist()):
                key = str(int(label))
                counts[key] = counts.get(key, 0) + int(count)
        return counts, digest.hexdigest()

    def describe(self, output_dir: Path) -> DescriptionBundle:
        config = self.config
        output_dir.mkdir(parents=True, exist_ok=True)
        segmentation_reader = open_volume(
            config.segmentation_uri, dataset=config.segmentation_dataset, channel_axis=None
        )
        nucleus_reader = open_volume(
            config.nucleus_uri, dataset=config.nucleus_dataset, channel_axis=None
        )
        affinity_reader = open_volume(
            config.affinity_uri,
            dataset=config.affinity_dataset,
            channel_axis=config.affinity_channel_axis,
        )
        try:
            segmentation_ref = volume_artifact_ref(
                "segmentation",
                config.segmentation_uri,
                segmentation_reader,
                config.scope,
                config.max_read_bytes,
                config.segmentation_dataset,
            )
            affinity_ref = volume_artifact_ref(
                "affinity",
                config.affinity_uri,
                affinity_reader,
                config.scope,
                config.max_read_bytes,
                config.affinity_dataset,
            )
            totals, nucleus_hash = self._compute_anchor_totals(nucleus_reader)
            nucleus_ref = ArtifactRef(
                "nuclei", config.nucleus_uri, nucleus_hash, config.nucleus_dataset
            )
            totals_record = AnchorTotals(
                schema_version="1.0",
                artifact_id=stable_id("anchor-totals", {"nucleus": nucleus_hash, "totals": totals}),
                nucleus_artifact_sha256=nucleus_hash,
                reference_shape_zyx=nucleus_reader.spatial_shape_zyx,
                totals=totals,
            )
            totals_path = output_dir / "anchor_totals.json"
            write_json(totals_path, totals_record)
            totals_ref = ArtifactRef(
                "anchor_totals", str(totals_path.resolve()), sha256_file(totals_path)
            )
            input_refs: list[ArtifactRef] = [segmentation_ref, nucleus_ref, affinity_ref]
            if config.contact_scopes_uri:
                contact_path = Path(config.contact_scopes_uri)
                input_refs.append(
                    ArtifactRef(
                        "contact_scopes", str(contact_path.resolve()), sha256_file(contact_path)
                    )
                )
            provenance = Provenance(
                input_artifacts=tuple(input_refs),
                configuration_hash=self.configuration_hash,
                timestamp_utc=_utc_now(),
                source_version=self.version,
            )
            segmentation = read_bbox_chunked(
                segmentation_reader, config.scope, config.max_read_bytes
            )
            segmentation_shape_zyx = segmentation_reader.spatial_shape_zyx
            nuclei = _aligned_nuclei(config, nucleus_reader)
        finally:
            segmentation_reader.close()
            nucleus_reader.close()
            affinity_reader.close()

        hit = nuclei != 0
        contact_scopes = (
            load_contact_scopes(config.contact_scopes_uri).scopes
            if config.contact_scopes_uri
            else ()
        )
        contact_by_identity = {
            (scope.seg_id, scope.anchor_ids): scope for scope in contact_scopes
        }
        overlaps: dict[int, dict[int, int]] = {}
        if hit.any():
            pairs = np.stack((segmentation[hit], nuclei[hit]), axis=1)
            unique, counts = np.unique(pairs, axis=0, return_counts=True)
            for (component, anchor), count in zip(unique.tolist(), counts.tolist()):
                if int(component) != 0:
                    overlaps.setdefault(int(component), {})[int(anchor)] = int(count)

        descriptors: list[Descriptor] = []
        certificates: list[Certificate] = []
        for component_id in sorted(overlaps):
            component_mask = segmentation == component_id
            counts = overlaps[component_id]
            fractions = {
                str(anchor): count / totals[str(anchor)] for anchor, count in sorted(counts.items())
            }
            qualifying = tuple(
                str(anchor)
                for anchor, count in sorted(counts.items())
                if count >= config.min_share * totals[str(anchor)]
            )
            contact = contact_by_identity.get((str(component_id), qualifying))
            repair_scope = contact.bbox if contact is not None and contact.bbox else config.scope
            repair_component = component_mask[_relative_slices(repair_scope, config.scope)]
            contained = _component_contained(
                repair_component, repair_scope, segmentation_shape_zyx
            )
            component = EntityRef(
                "component",
                str(component_id),
                bbox=repair_scope,
                source=config.segmentation_uri,
            )
            values: tuple[tuple[str, Any, str | None], ...] = (
                ("component.volume_voxels", int(component_mask.sum()), "voxels"),
                ("anchor.overlap_count", {str(k): v for k, v in sorted(counts.items())}, "voxels"),
                ("anchor.overlap_fraction", fractions, "fraction_of_anchor_mass"),
                ("anchor.distinct_ids", list(qualifying), None),
                ("anchor.distinct_count", len(qualifying), "anchors"),
                ("scope.containment", "contained" if contained else "partial", None),
            )
            component_descriptors = []
            for key, value, units in values:
                descriptor = Descriptor(
                    schema_version="1.0",
                    descriptor_id=stable_id(
                        "descriptor",
                        {
                            "subject": component_id,
                            "key": key,
                            "value": value,
                            "scope": repair_scope,
                        },
                    ),
                    subject=component,
                    key=key,
                    value=value,
                    units=units,
                    confidence=1.0,
                    operator_name=self.name,
                    operator_version=self.version,
                    provenance=provenance,
                    scope=repair_scope,
                )
                descriptors.append(descriptor)
                component_descriptors.append(descriptor)
            if len(qualifying) >= 2:
                certificate = Certificate(
                    schema_version="1.0",
                    certificate_id=stable_id(
                        "certificate", {"component": component_id, "anchors": qualifying}
                    ),
                    certificate_type="distinct_anchor_identity_conflict",
                    affected_component=component,
                    anchor_kind="nucleus_instance",
                    distinct_anchor_ids=qualifying,
                    supporting_descriptor_ids=tuple(d.descriptor_id for d in component_descriptors),
                    strength="hard",
                    operator_version=self.version,
                    provenance=provenance,
                )
                certificates.append(certificate)
        return DescriptionBundle(
            schema_version="1.0",
            checkpoint_id=config.checkpoint_id,
            pass_id=config.pass_id,
            operator_name=self.name,
            operator_version=self.version,
            input_artifacts=tuple(input_refs),
            anchor_totals_artifact=totals_ref,
            configuration_hash=self.configuration_hash,
            scope=config.scope,
            descriptors=tuple(descriptors),
            certificates=tuple(certificates),
        )

    def plan(self, description: DescriptionBundle) -> CheckpointPlan:
        if description.configuration_hash != self.configuration_hash:
            raise ValueError("description configuration hash does not match this operator")
        contact_scopes = (
            load_contact_scopes(self.config.contact_scopes_uri).scopes
            if self.config.contact_scopes_uri
            else ()
        )
        eligible = {
            (scope.seg_id, scope.anchor_ids): scope
            for scope in contact_scopes
            if scope.eligibility == "contact"
        }
        descriptor_by_component: dict[str, dict[str, Any]] = {}
        for descriptor in description.descriptors:
            descriptor_by_component.setdefault(descriptor.subject.stable_id, {})[
                descriptor.key
            ] = descriptor.value
        actions: list[ActionSpec] = []
        for certificate in description.certificates:
            component_id = certificate.affected_component.stable_id
            key = (component_id, certificate.distinct_anchor_ids)
            contact = eligible.get(key)
            if contact is None:
                continue
            repair_scope = contact.bbox or description.scope
            containment = descriptor_by_component[component_id]["scope.containment"]
            separation_claim = "contained" if containment == "contained" else "local_only"
            targets = (certificate.affected_component,) + tuple(
                EntityRef("anchor", anchor, bbox=repair_scope, source=self.config.nucleus_uri)
                for anchor in certificate.distinct_anchor_ids
            )
            common = {
                "schema_version": "1.0",
                "targets": targets,
                "preconditions": (Condition("anchor.distinct_count", "ge", 2),),
                "supporting_ids": (certificate.certificate_id,),
            }
            specs = (
                (
                    "split_by_anchor",
                    SplitByAnchorParams(
                        component_id,
                        certificate.distinct_anchor_ids,
                        repair_scope,
                        separation_claim,  # type: ignore[arg-type]
                        self.config.channel_indices,
                        self.config.affinity_channel_axis,
                        self.config.affinity_convention,  # type: ignore[arg-type]
                        self.config.sigmoid_restore,
                        self.config.pooling_factor,
                        self.config.nucleus_scale_zyx,
                        self.config.max_read_bytes,
                    ),
                    ("voxel-level refinement is confined to the parent component",),
                ),
                (
                    "consolidate_same_anchor",
                    ConsolidateSameAnchorParams(
                        component_id, certificate.distinct_anchor_ids, repair_scope
                    ),
                    ("same-anchor pieces are consolidated only after exclusion",),
                ),
                (
                    "forbid_merge",
                    ForbidMergeParams(component_id, certificate.distinct_anchor_ids),
                    ("distinct consolidated anchor territories have pairwise cannot-links",),
                ),
                (
                    "rebuild_local_rag",
                    RebuildLocalRagParams(repair_scope),
                    ("local graph state records the corrected partition",),
                ),
            )
            for operation, parameters, expected in specs:
                payload = {
                    "checkpoint": description.checkpoint_id,
                    "component": component_id,
                    "operation": operation,
                    "parameters": parameters,
                }
                actions.append(
                    ActionSpec(
                        action_id=stable_id("action", payload),
                        operation=operation,  # type: ignore[arg-type]
                        parameters=parameters,
                        expected_postconditions=expected,
                        status="pending",
                        failure_message=None,
                        **common,
                    )
                )
        provisional = CheckpointPlan(
            schema_version="1.0",
            plan_version="1.0",
            plan_id="pending",
            checkpoint_id=description.checkpoint_id,
            pass_id=description.pass_id,
            operator_name=self.name,
            operator_version=self.version,
            input_artifacts=description.input_artifacts,
            anchor_totals_artifact=description.anchor_totals_artifact,
            descriptors=description.descriptors,
            certificates=description.certificates,
            actions=tuple(actions),
            expected_invariants=(
                "repaired components contain at most one qualifying nucleus anchor per territory",
                "each qualifying anchor has one dominant output territory",
                "outside-scope partition is unchanged",
                "distinct territories have exported cannot-links",
                "consolidation follows resolved exclusion",
                "second checkpoint run is a no-op",
                "identical frozen applies have identical output hashes",
            ),
            configuration=self.config.as_hash_data(),
            configuration_hash=self.configuration_hash,
            scope=description.scope,
        )
        return CheckpointPlan(**{**provisional.__dict__, "plan_id": stable_id("plan", provisional)})


__all__ = ["NucleusAnchorConfig", "NucleusAnchorOperator", "load_contact_scopes"]
