"""Deterministic executor for validated checkpoint action plans."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import (
    full_bbox,
    hash_volume,
    open_volume,
    read_bbox_chunked,
    write_segmentation_delta,
)
from .kernels import (
    competitive_split_component,
    consolidate_same_anchor,
    local_rag,
    pairwise_cannot_links,
)
from .schema import (
    ActionExecution,
    AnnotateParams,
    ArtifactRef,
    BoundingBox,
    CheckpointPlan,
    CheckpointResult,
    ConsolidateSameAnchorParams,
    ForbidMergeParams,
    RebuildLocalRagParams,
    SplitByAnchorParams,
    UnrepairedConflict,
    evaluate_condition,
)
from .serialization import (
    append_jsonl,
    content_hash,
    read_json,
    sha256_file,
    stable_id,
    write_json,
)


class PreconditionError(RuntimeError):
    """Raised before mutation when frozen plan inputs no longer match."""


def _artifact(plan: CheckpointPlan, role: str) -> ArtifactRef:
    found = [artifact for artifact in plan.input_artifacts if artifact.role == role]
    if len(found) != 1:
        raise PreconditionError(f"plan requires exactly one {role} artifact")
    return found[0]


def _local_slices(inner: BoundingBox, outer: BoundingBox) -> tuple[slice, slice, slice]:
    if any(
        lo < outer_lo or hi > outer_hi
        for lo, hi, outer_lo, outer_hi in zip(
            inner.start_zyx, inner.stop_zyx, outer.start_zyx, outer.stop_zyx
        )
    ):
        raise PreconditionError(f"action scope {inner} lies outside plan scope {outer}")
    return tuple(
        slice(lo - outer_lo, hi - outer_lo)
        for lo, hi, outer_lo in zip(inner.start_zyx, inner.stop_zyx, outer.start_zyx)
    )  # type: ignore[return-value]


def _aligned_nuclei(plan: CheckpointPlan, reader: Any, max_read_bytes: int) -> np.ndarray:
    scale = np.asarray(plan.configuration["nucleus_scale_zyx"], dtype=np.int64)
    start = np.asarray(plan.scope.start_zyx)
    stop = np.asarray(plan.scope.stop_zyx)
    low = start // scale
    high = (stop - 1) // scale + 1
    bbox = BoundingBox(tuple(low), tuple(high))
    block = read_bbox_chunked(reader, bbox, max_read_bytes)
    coordinates = [np.arange(start[i], stop[i]) // scale[i] - low[i] for i in range(3)]
    return np.asarray(block[np.ix_(*coordinates)])


def _validate_inputs(
    plan: CheckpointPlan,
    segmentation_reader: Any,
    nucleus_reader: Any,
    affinity_reader: Any,
    max_read_bytes: int,
) -> None:
    if content_hash(plan.configuration) != plan.configuration_hash:
        raise PreconditionError("plan configuration snapshot no longer matches configuration_hash")
    expected = {
        "segmentation": hash_volume(segmentation_reader, plan.scope, max_read_bytes),
        "nuclei": hash_volume(nucleus_reader, full_bbox(nucleus_reader), max_read_bytes),
        "affinity": hash_volume(affinity_reader, plan.scope, max_read_bytes),
    }
    for role, actual in expected.items():
        planned = _artifact(plan, role).sha256
        if actual != planned:
            raise PreconditionError(
                f"{role} hash mismatch: plan has {planned}, current input has {actual}"
            )
    totals_path = Path(plan.anchor_totals_artifact.uri)
    if not totals_path.exists() or sha256_file(totals_path) != plan.anchor_totals_artifact.sha256:
        raise PreconditionError("required anchor totals artifact is missing or changed")
    totals = read_json(totals_path)
    if totals.nucleus_artifact_sha256 != expected["nuclei"]:
        raise PreconditionError("anchor totals were computed from a different nucleus volume")


def _segmentation_sha256(labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"shape": labels.shape, "dtype": str(labels.dtype)}, sort_keys=True).encode()
    )
    digest.update(np.ascontiguousarray(labels).view(np.uint8))
    return digest.hexdigest()


def execute_plan(plan: CheckpointPlan, output_dir: str | Path) -> CheckpointResult:
    """Validate every frozen precondition, then apply actions in plan order."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_actions = [
        action for action in plan.actions if isinstance(action.parameters, SplitByAnchorParams)
    ]
    max_read_bytes = (
        split_actions[0].parameters.max_read_bytes
        if split_actions
        else int(plan.configuration.get("max_read_bytes", 64 * 1024 * 1024))
    )
    segmentation_artifact = _artifact(plan, "segmentation")
    nucleus_artifact = _artifact(plan, "nuclei")
    affinity_artifact = _artifact(plan, "affinity")
    segmentation_reader = open_volume(
        segmentation_artifact.uri,
        dataset=segmentation_artifact.dataset or "main",
        channel_axis=None,
    )
    nucleus_reader = open_volume(
        nucleus_artifact.uri, dataset=nucleus_artifact.dataset or "main", channel_axis=None
    )
    affinity_axis = int(plan.configuration["affinity_channel_axis"])
    affinity_reader = open_volume(
        affinity_artifact.uri,
        dataset=affinity_artifact.dataset or "main",
        channel_axis=affinity_axis,
    )
    try:
        _validate_inputs(plan, segmentation_reader, nucleus_reader, affinity_reader, max_read_bytes)
        original = read_bbox_chunked(segmentation_reader, plan.scope, max_read_bytes)
        nuclei = _aligned_nuclei(plan, nucleus_reader, max_read_bytes)
        affinity = read_bbox_chunked(affinity_reader, plan.scope, max_read_bytes)
    finally:
        segmentation_reader.close()
        nucleus_reader.close()
        affinity_reader.close()

    current = np.asarray(original, dtype=np.uint64).copy()
    abstention = np.zeros(current.shape, dtype=bool)
    descriptor_values: dict[str, dict[str, Any]] = {}
    for descriptor in plan.descriptors:
        descriptor_values.setdefault(descriptor.subject.stable_id, {})[
            descriptor.key
        ] = descriptor.value
    split_state: dict[str, dict[str, Any]] = {}
    constraints: list[dict[str, Any]] = []
    rag_records: list[dict[str, Any]] = []
    annotations: list[AnnotateParams] = []
    executions: list[ActionExecution] = []
    execution_log = output / "execution.jsonl"
    if execution_log.exists():
        raise FileExistsError(f"append-only execution log already exists: {execution_log}")

    for action in plan.actions:
        started = time.perf_counter()
        affected = 0
        bindings: Mapping[str, tuple[str, ...]] = {}
        status = "executed"
        failure: str | None = None
        component_id = action.targets[0].stable_id if action.targets else ""
        if not all(
            evaluate_condition(condition, descriptor_values.get(component_id, {}))
            for condition in action.preconditions
        ):
            raise PreconditionError(f"descriptor precondition failed for action {action.action_id}")
        try:
            if isinstance(action.parameters, AnnotateParams):
                annotations.append(action.parameters)
            elif isinstance(action.parameters, SplitByAnchorParams):
                params = action.parameters
                slices = _local_slices(params.repair_scope, plan.scope)
                affinity_slices = slices
                if affinity_axis == 0:
                    affinity_slices = (slice(None),) + affinity_slices
                else:
                    affinity_slices = affinity_slices + (slice(None),)
                split = competitive_split_component(
                    current[slices],
                    nuclei[slices],
                    affinity[affinity_slices],
                    component_id=int(params.component_id),
                    anchor_ids=tuple(int(value) for value in params.anchor_ids),
                    channel_indices=params.channel_indices,
                    channel_axis=params.affinity_channel_axis,
                    convention=params.affinity_convention,
                    sigmoid_restore=params.sigmoid_restore,
                    factor=params.pooling_factor,
                    contained=params.separation_claim == "contained",
                )
                current[slices] = split.labels
                abstention[slices] |= split.abstention_mask
                affected = split.affected_voxels
                split_state[params.component_id] = {
                    "piece_bindings": split.piece_bindings,
                    "territory_bindings": split.piece_bindings,
                    "assigned_voxels": split.assigned_voxels,
                    "parent_voxels": int((original[slices] == int(params.component_id)).sum()),
                    "abstained_voxels": int(split.abstention_mask.sum()),
                    "separation_claim": params.separation_claim,
                    "repair_scope": params.repair_scope,
                }
                bindings = {
                    anchor: tuple(str(piece) for piece in pieces)
                    for anchor, pieces in split.piece_bindings.items()
                }
            elif isinstance(action.parameters, ConsolidateSameAnchorParams):
                params = action.parameters
                state = split_state.get(params.component_id)
                if state is None:
                    raise PreconditionError("consolidation must follow split_by_anchor")
                slices = _local_slices(params.repair_scope, plan.scope)
                consolidation = consolidate_same_anchor(
                    current[slices], state["piece_bindings"], cannot_links=set()
                )
                current[slices] = consolidation.labels
                state["territory_bindings"] = consolidation.territory_bindings
                affected = consolidation.affected_voxels
                bindings = {
                    anchor: tuple(str(piece) for piece in pieces)
                    for anchor, pieces in consolidation.territory_bindings.items()
                }
            elif isinstance(action.parameters, ForbidMergeParams):
                params = action.parameters
                state = split_state.get(params.component_id)
                if (
                    state is None
                    or state["territory_bindings"] == state["piece_bindings"]
                    and any(len(pieces) > 1 for pieces in state["piece_bindings"].values())
                ):
                    raise PreconditionError("forbid_merge must follow completed consolidation")
                pairs = pairwise_cannot_links(state["territory_bindings"])
                constraint = {
                    "schema_version": "1.0",
                    "constraint_id": stable_id(
                        "cannot-link", {"component": params.component_id, "pairs": pairs}
                    ),
                    "operation": "forbid_merge",
                    "component_id": params.component_id,
                    "binding": "anchor_territory_sets",
                    "anchor_territories": {
                        anchor: [str(value) for value in values]
                        for anchor, values in state["territory_bindings"].items()
                    },
                    "pairs": [[str(a), str(b)] for a, b in pairs],
                    "supporting_ids": list(action.supporting_ids),
                    "consumer": None,
                    "consumer_status": "write_only_no_decoder_consumer",
                }
                constraints.append(constraint)
                bindings = {
                    anchor: tuple(str(value) for value in values)
                    for anchor, values in state["territory_bindings"].items()
                }
            elif isinstance(action.parameters, RebuildLocalRagParams):
                slices = _local_slices(action.parameters.repair_scope, plan.scope)
                edges = local_rag(current[slices])
                rag_records.append(
                    {
                        "action_id": action.action_id,
                        "scope_zyx": [
                            *action.parameters.repair_scope.start_zyx,
                            *action.parameters.repair_scope.stop_zyx,
                        ],
                        "mode": action.parameters.mode,
                        "edges": [[str(a), str(b)] for a, b in edges],
                        "graph_hash": content_hash(edges),
                    }
                )
            else:
                status = "skipped"
                failure = "operation is a typed placeholder and has no executor"
        except Exception as error:
            status = "failed"
            failure = str(error)
        execution = ActionExecution(
            action_id=action.action_id,
            operation=action.operation,
            status=status,  # type: ignore[arg-type]
            elapsed_seconds=time.perf_counter() - started,
            affected_voxels=affected,
            territory_bindings=bindings,
            failure_message=failure,
        )
        executions.append(execution)
        append_jsonl(execution_log, execution)
        if status == "failed":
            break

    delta_path = output / "segmentation_delta.npz"
    changed_voxels = write_segmentation_delta(
        delta_path, np.asarray(original), current, plan.scope, abstention
    )
    constraint_path = output / "cannot_links.json"
    write_json(
        constraint_path,
        {
            "schema_version": "1.0",
            "manifest_type": "segmentation_checkpoint_cannot_links",
            "consumer_status": "write_only_no_decoder_consumer",
            "constraints": constraints,
        },
    )
    rag_path = output / "local_rag.json"
    write_json(rag_path, {"schema_version": "1.0", "records": rag_records})
    artifacts = tuple(
        ArtifactRef(role, str(path.resolve()), sha256_file(path))
        for role, path in (
            ("segmentation_delta", delta_path),
            ("cannot_links", constraint_path),
            ("local_rag", rag_path),
            ("execution_log", execution_log),
        )
    )
    repaired = tuple(
        sorted(
            component
            for component, state in split_state.items()
            if state["separation_claim"] == "contained"
            and not any(
                execution.status == "failed"
                and execution.action_id
                in {
                    action.action_id
                    for action in plan.actions
                    if action.targets and action.targets[0].stable_id == component
                }
                for execution in executions
            )
        )
    )
    action_components = {
        action.targets[0].stable_id
        for action in plan.actions
        if action.targets and action.operation == "split_by_anchor"
    }
    certificate_components = {
        certificate.affected_component.stable_id: certificate for certificate in plan.certificates
    }
    unrepaired: list[UnrepairedConflict] = []
    for component, certificate in sorted(certificate_components.items()):
        if component in repaired:
            continue
        reason = "not_in_contact_scopes"
        if component in action_components:
            reason = (
                "failed_action"
                if any(execution.status == "failed" for execution in executions)
                else "local_only"
            )
        unrepaired.append(
            UnrepairedConflict(
                component,
                certificate.distinct_anchor_ids,
                reason,  # type: ignore[arg-type]
            )
        )
    stats: dict[str, Any] = {
        "changed_voxels": changed_voxels,
        "abstained_voxels": int(abstention.sum()),
        "output_segmentation_sha256": _segmentation_sha256(current),
        "constraint_count": len(constraints),
        "failed_actions": sum(execution.status == "failed" for execution in executions),
        "skipped_actions": sum(execution.status == "skipped" for execution in executions),
        "component_execution": {
            component: {
                "assigned_voxels": state["assigned_voxels"],
                "parent_voxels": state["parent_voxels"],
                "abstained_voxels": state["abstained_voxels"],
                "separation_claim": state["separation_claim"],
            }
            for component, state in split_state.items()
        },
    }
    provisional = CheckpointResult(
        schema_version="1.0",
        result_id="pending",
        plan_id=plan.plan_id,
        action_executions=tuple(executions),
        verification=(),
        output_artifacts=artifacts,
        summary_statistics=stats,
        repaired_components=repaired,
        certified_unrepaired=tuple(unrepaired),
        annotations=tuple(annotations),
    )
    return CheckpointResult(
        **{**provisional.__dict__, "result_id": stable_id("result", provisional)}
    )


__all__ = ["PreconditionError", "execute_plan"]
