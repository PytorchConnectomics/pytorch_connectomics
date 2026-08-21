"""Typed records for the segmentation checkpoint intervention protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence, Union

JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
EntityKind = Literal["fragment", "component", "contact", "anchor", "territory"]
Operation = Literal[
    "annotate",
    "split_by_anchor",
    "forbid_merge",
    "consolidate_same_anchor",
    "rebuild_local_rag",
    "hold_edge",
    "release_edge",
]
ExecutionStatus = Literal["pending", "executed", "skipped", "failed"]


def _require_namespaced(value: str, field_name: str) -> None:
    if value.count(".") < 1 or value.startswith(".") or value.endswith("."):
        raise ValueError(f"{field_name} must be a namespaced key, got {value!r}")


@dataclass(frozen=True)
class BoundingBox:
    """Half-open physical array bounds in canonical z, y, x order."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.start_zyx) != 3 or len(self.stop_zyx) != 3:
            raise ValueError("bounding boxes must have exactly three z/y/x coordinates")
        if any(lo < 0 or hi <= lo for lo, hi in zip(self.start_zyx, self.stop_zyx)):
            raise ValueError(f"invalid half-open bounding box {self.start_zyx}:{self.stop_zyx}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(hi - lo for lo, hi in zip(self.start_zyx, self.stop_zyx))

    def as_slices(self) -> tuple[slice, slice, slice]:
        result = tuple(slice(lo, hi) for lo, hi in zip(self.start_zyx, self.stop_zyx))
        return result  # type: ignore[return-value]


@dataclass(frozen=True)
class EntityRef:
    kind: EntityKind
    stable_id: str
    bbox: BoundingBox | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.stable_id:
            raise ValueError("entity stable_id must be non-empty")


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    uri: str
    sha256: str
    dataset: str | None = None

    def __post_init__(self) -> None:
        if not self.role or not self.uri:
            raise ValueError("artifact role and uri must be non-empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError(f"invalid sha256 for {self.role}: {self.sha256!r}")


@dataclass(frozen=True)
class Provenance:
    input_artifacts: tuple[ArtifactRef, ...]
    configuration_hash: str
    timestamp_utc: str
    source_version: str | None = None

    def __post_init__(self) -> None:
        if len(self.configuration_hash) != 64:
            raise ValueError("configuration_hash must be a sha256")
        if not self.timestamp_utc.endswith("Z"):
            raise ValueError("timestamp_utc must be an ISO-8601 UTC timestamp ending in Z")


@dataclass(frozen=True)
class Descriptor:
    schema_version: str
    descriptor_id: str
    subject: EntityRef
    key: str
    value: JSONValue
    operator_name: str
    operator_version: str
    provenance: Provenance
    scope: BoundingBox
    units: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_namespaced(self.key, "descriptor key")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("descriptor confidence must lie in [0, 1]")


@dataclass(frozen=True)
class Certificate:
    schema_version: str
    certificate_id: str
    certificate_type: Literal["distinct_anchor_identity_conflict"]
    affected_component: EntityRef
    anchor_kind: Literal["nucleus_instance"]
    distinct_anchor_ids: tuple[str, ...]
    supporting_descriptor_ids: tuple[str, ...]
    strength: Literal["hard"]
    operator_version: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if len(set(self.distinct_anchor_ids)) < 2:
            raise ValueError("identity-conflict certificates require at least two anchors")
        if tuple(sorted(set(self.distinct_anchor_ids), key=int)) != self.distinct_anchor_ids:
            raise ValueError("distinct_anchor_ids must be unique and numerically sorted")


@dataclass(frozen=True)
class Condition:
    field: str
    operator: Literal["eq", "ne", "gt", "ge", "lt", "le"]
    value: JSONScalar

    def __post_init__(self) -> None:
        _require_namespaced(self.field, "condition field")


@dataclass(frozen=True)
class AnnotateParams:
    target: EntityRef
    key: str
    value: JSONScalar
    note: str | None = None

    def __post_init__(self) -> None:
        _require_namespaced(self.key, "annotation key")


@dataclass(frozen=True)
class SplitByAnchorParams:
    component_id: str
    anchor_ids: tuple[str, ...]
    repair_scope: BoundingBox
    separation_claim: Literal["contained", "local_only"]
    channel_indices: tuple[int, ...]
    affinity_channel_axis: int
    affinity_convention: Literal["probability", "deepem", "banis"]
    sigmoid_restore: float | None
    pooling_factor: int
    nucleus_scale_zyx: tuple[int, int, int]
    max_read_bytes: int
    connectivity: Literal[6] = 6
    tie_break: Literal["lowest_anchor_id"] = "lowest_anchor_id"

    def __post_init__(self) -> None:
        if len(set(self.anchor_ids)) < 2:
            raise ValueError("split_by_anchor requires at least two distinct anchors")
        if not self.channel_indices or self.pooling_factor < 1 or self.max_read_bytes < 1:
            raise ValueError("split numerical configuration is incomplete")


@dataclass(frozen=True)
class ConsolidateSameAnchorParams:
    component_id: str
    anchor_ids: tuple[str, ...]
    repair_scope: BoundingBox


@dataclass(frozen=True)
class ForbidMergeParams:
    component_id: str
    anchor_ids: tuple[str, ...]
    binding: Literal["anchor_territory_sets"] = "anchor_territory_sets"

    def __post_init__(self) -> None:
        if len(set(self.anchor_ids)) < 2:
            raise ValueError("forbid_merge requires at least two distinct anchors")


@dataclass(frozen=True)
class RebuildLocalRagParams:
    repair_scope: BoundingBox
    mode: Literal["recompute", "invalidate"] = "recompute"


@dataclass(frozen=True)
class EdgeLifecycleParams:
    edge_ids: tuple[str, ...]


ActionParams = Union[
    AnnotateParams,
    SplitByAnchorParams,
    ConsolidateSameAnchorParams,
    ForbidMergeParams,
    RebuildLocalRagParams,
    EdgeLifecycleParams,
]


_PARAM_TYPES: dict[str, type[ActionParams]] = {
    "annotate": AnnotateParams,
    "split_by_anchor": SplitByAnchorParams,
    "consolidate_same_anchor": ConsolidateSameAnchorParams,
    "forbid_merge": ForbidMergeParams,
    "rebuild_local_rag": RebuildLocalRagParams,
    "hold_edge": EdgeLifecycleParams,
    "release_edge": EdgeLifecycleParams,
}


@dataclass(frozen=True)
class ActionSpec:
    schema_version: str
    action_id: str
    operation: Operation
    targets: tuple[EntityRef, ...]
    parameters: ActionParams
    preconditions: tuple[Condition, ...]
    expected_postconditions: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    status: ExecutionStatus = "pending"
    failure_message: str | None = None

    def __post_init__(self) -> None:
        expected = _PARAM_TYPES[self.operation]
        if not isinstance(self.parameters, expected):
            raise TypeError(
                f"{self.operation} parameters must be {expected.__name__}, "
                f"got {type(self.parameters).__name__}"
            )
        if self.operation in ("hold_edge", "release_edge") and self.status != "pending":
            raise ValueError("placeholder edge lifecycle actions cannot be pre-executed")
        if self.status == "failed" and not self.failure_message:
            raise ValueError("failed actions require a failure_message")


@dataclass(frozen=True)
class CheckpointPlan:
    schema_version: str
    plan_version: str
    plan_id: str
    checkpoint_id: str
    pass_id: str
    operator_name: str
    operator_version: str
    input_artifacts: tuple[ArtifactRef, ...]
    anchor_totals_artifact: ArtifactRef
    descriptors: tuple[Descriptor, ...]
    certificates: tuple[Certificate, ...]
    actions: tuple[ActionSpec, ...]
    expected_invariants: tuple[str, ...]
    configuration: Mapping[str, JSONValue]
    configuration_hash: str
    scope: BoundingBox

    def __post_init__(self) -> None:
        if self.operator_name != "nucleus_anchor":
            raise ValueError(f"unsupported operator {self.operator_name!r}")
        if len(self.configuration_hash) != 64:
            raise ValueError("configuration_hash must be a sha256")
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("action IDs must be unique")


@dataclass(frozen=True)
class ActionExecution:
    action_id: str
    operation: Operation
    status: ExecutionStatus
    elapsed_seconds: float
    affected_voxels: int
    territory_bindings: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.status == "pending":
            raise ValueError("execution records cannot remain pending")
        if self.elapsed_seconds < 0 or self.affected_voxels < 0:
            raise ValueError("timing and affected voxel counts must be non-negative")


@dataclass(frozen=True)
class VerificationOutcome:
    invariant: str
    passed: bool
    details: str


@dataclass(frozen=True)
class UnrepairedConflict:
    component_id: str
    anchor_ids: tuple[str, ...]
    reason: Literal["not_in_contact_scopes", "local_only", "failed_action"]


@dataclass(frozen=True)
class CheckpointResult:
    schema_version: str
    result_id: str
    plan_id: str
    action_executions: tuple[ActionExecution, ...]
    verification: tuple[VerificationOutcome, ...]
    output_artifacts: tuple[ArtifactRef, ...]
    summary_statistics: Mapping[str, JSONValue]
    repaired_components: tuple[str, ...]
    certified_unrepaired: tuple[UnrepairedConflict, ...]
    annotations: tuple[AnnotateParams, ...] = ()

    def __post_init__(self) -> None:
        overlap = set(self.repaired_components) & {
            conflict.component_id for conflict in self.certified_unrepaired
        }
        if overlap:
            raise ValueError(
                f"components cannot be both repaired and unrepaired: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class AnchorTotals:
    schema_version: str
    artifact_id: str
    nucleus_artifact_sha256: str
    reference_shape_zyx: tuple[int, int, int]
    totals: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(int(key) <= 0 or value <= 0 for key, value in self.totals.items()):
            raise ValueError("anchor totals must contain positive numeric IDs and counts")


@dataclass(frozen=True)
class ContactScope:
    seg_id: str
    anchor_ids: tuple[str, ...]
    provenance: str
    gap_um: float
    eligibility: Literal["contact", "bridge"]
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class ContactScopes:
    schema_version: str
    artifact_id: str
    scopes: tuple[ContactScope, ...]

    def __post_init__(self) -> None:
        keys = [(scope.seg_id, scope.anchor_ids) for scope in self.scopes]
        if len(keys) != len(set(keys)):
            raise ValueError("contact scope entries must be unique")


@dataclass(frozen=True)
class DescriptionBundle:
    schema_version: str
    checkpoint_id: str
    pass_id: str
    operator_name: str
    operator_version: str
    input_artifacts: tuple[ArtifactRef, ...]
    anchor_totals_artifact: ArtifactRef
    configuration_hash: str
    scope: BoundingBox
    descriptors: tuple[Descriptor, ...]
    certificates: tuple[Certificate, ...]


def evaluate_condition(condition: Condition, descriptors: Mapping[str, JSONValue]) -> bool:
    """Evaluate the deliberately small, non-executable condition language."""

    if condition.field not in descriptors:
        return False
    left = descriptors[condition.field]
    right = condition.value
    operations = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }
    try:
        return bool(operations[condition.operator](left, right))
    except TypeError:
        return False


__all__ = [
    "ActionExecution",
    "ActionSpec",
    "AnchorTotals",
    "AnnotateParams",
    "ArtifactRef",
    "BoundingBox",
    "Certificate",
    "CheckpointPlan",
    "CheckpointResult",
    "Condition",
    "ConsolidateSameAnchorParams",
    "ContactScope",
    "ContactScopes",
    "Descriptor",
    "DescriptionBundle",
    "EdgeLifecycleParams",
    "EntityRef",
    "ForbidMergeParams",
    "Provenance",
    "RebuildLocalRagParams",
    "SplitByAnchorParams",
    "UnrepairedConflict",
    "VerificationOutcome",
    "evaluate_condition",
]
