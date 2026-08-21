"""Postcondition verification independent of training and decoder internals."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .actions import _aligned_nuclei, _artifact, _local_slices, execute_plan
from .io import apply_segmentation_delta, open_volume, read_bbox_chunked
from .schema import (
    CheckpointPlan,
    CheckpointResult,
    SplitByAnchorParams,
    VerificationOutcome,
)
from .serialization import content_hash, sha256_file, stable_id


def _outcome(name: str, passed: bool, details: str) -> VerificationOutcome:
    return VerificationOutcome(name, bool(passed), details)


def _output_artifact(result: CheckpointResult, role: str) -> Path:
    matches = [artifact for artifact in result.output_artifacts if artifact.role == role]
    if len(matches) != 1:
        raise ValueError(f"result requires exactly one {role} artifact")
    artifact = matches[0]
    path = Path(artifact.uri)
    if not path.exists() or sha256_file(path) != artifact.sha256:
        raise ValueError(f"result artifact {role} is missing or changed")
    return path


def verify_result(
    plan: CheckpointPlan,
    result: CheckpointResult,
    verification_dir: str | Path,
    *,
    replay_determinism: bool = True,
) -> CheckpointResult:
    if result.plan_id != plan.plan_id:
        raise ValueError("result does not belong to the supplied plan")
    output_dir = Path(verification_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    segmentation_reader = open_volume(
        segmentation_artifact.uri,
        dataset=segmentation_artifact.dataset or "main",
        channel_axis=None,
    )
    nucleus_reader = open_volume(
        nucleus_artifact.uri, dataset=nucleus_artifact.dataset or "main", channel_axis=None
    )
    try:
        original = read_bbox_chunked(segmentation_reader, plan.scope, max_read_bytes)
        nuclei = _aligned_nuclei(plan, nucleus_reader, max_read_bytes)
    finally:
        segmentation_reader.close()
        nucleus_reader.close()
    delta_path = _output_artifact(result, "segmentation_delta")
    corrected, abstention = apply_segmentation_delta(original, delta_path)
    constraints = json.loads(_output_artifact(result, "cannot_links").read_text())
    rag = json.loads(_output_artifact(result, "local_rag").read_text())
    totals_path = Path(plan.anchor_totals_artifact.uri)
    if not totals_path.exists() or sha256_file(totals_path) != plan.anchor_totals_artifact.sha256:
        raise ValueError("required anchor totals artifact is missing or changed")
    totals = json.loads(totals_path.read_text())
    totals_values = totals["totals"]
    certificate_by_component = {
        certificate.affected_component.stable_id: certificate for certificate in plan.certificates
    }
    outcomes: list[VerificationOutcome] = []

    expected_components = set(certificate_by_component)
    reported_components = set(result.repaired_components) | {
        conflict.component_id for conflict in result.certified_unrepaired
    }
    outcomes.append(
        _outcome(
            "certified_conflicts_explicitly_accounted",
            expected_components == reported_components,
            f"expected={sorted(expected_components)} reported={sorted(reported_components)}",
        )
    )

    contamination_failures: list[str] = []
    dominant_failures: list[str] = []
    idempotence_failures: list[str] = []
    containment_failures: list[str] = []
    repair_scopes = np.zeros(original.shape, dtype=bool)
    for action in split_actions:
        params = action.parameters
        slices = _local_slices(params.repair_scope, plan.scope)
        repair_scopes[slices] = True
        parent = original[slices] == int(params.component_id)
        local_output = corrected[slices]
        local_nuclei = nuclei[slices]
        if params.separation_claim == "local_only":
            boundary_parent = np.zeros(parent.shape, dtype=bool)
            boundary_parent[0] |= parent[0]
            boundary_parent[-1] |= parent[-1]
            boundary_parent[:, 0] |= parent[:, 0]
            boundary_parent[:, -1] |= parent[:, -1]
            boundary_parent[:, :, 0] |= parent[:, :, 0]
            boundary_parent[:, :, -1] |= parent[:, :, -1]
            unsafe = boundary_parent & (local_output != int(params.component_id))
            if unsafe.any():
                containment_failures.append(f"{params.component_id}:{int(unsafe.sum())}")
            continue
        label_owners: dict[int, list[str]] = {}
        for anchor in params.anchor_ids:
            anchor_mask = parent & (local_nuclei == int(anchor))
            labels, counts = np.unique(local_output[anchor_mask], return_counts=True)
            qualifying = [
                (int(label), int(count))
                for label, count in zip(labels.tolist(), counts.tolist())
                if count >= float(plan.configuration["min_share"]) * int(totals_values[anchor])
            ]
            for label, _count in qualifying:
                label_owners.setdefault(label, []).append(anchor)
            if not qualifying:
                dominant_failures.append(f"{params.component_id}:{anchor}:no qualifying territory")
            else:
                ordered = sorted(qualifying, key=lambda item: (-item[1], item[0]))
                if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
                    dominant_failures.append(f"{params.component_id}:{anchor}:tied dominance")
        shared = {label: owners for label, owners in label_owners.items() if len(owners) > 1}
        if shared:
            contamination_failures.append(f"{params.component_id}:{shared}")
            idempotence_failures.append(params.component_id)
    outcomes.append(
        _outcome(
            "repaired_components_have_no_qualifying_multi_anchor_territory",
            not contamination_failures,
            "none" if not contamination_failures else "; ".join(contamination_failures),
        )
    )
    outcomes.append(
        _outcome(
            "each_qualifying_anchor_has_one_dominant_territory",
            not dominant_failures,
            "none" if not dominant_failures else "; ".join(dominant_failures),
        )
    )
    outside_changed = (corrected != original) & ~repair_scopes
    outcomes.append(
        _outcome(
            "outside_repair_scope_partition_unchanged",
            not outside_changed.any(),
            f"changed_voxels={int(outside_changed.sum())}",
        )
    )
    outcomes.append(
        _outcome(
            "partial_scope_new_territories_do_not_touch_continuing_boundary",
            not containment_failures,
            "none" if not containment_failures else "; ".join(containment_failures),
        )
    )

    manifest_components = {
        entry["component_id"]: entry for entry in constraints.get("constraints", [])
    }
    constraint_failures: list[str] = []
    for component in result.repaired_components:
        certificate = certificate_by_component[component]
        entry = manifest_components.get(component)
        if entry is None:
            constraint_failures.append(f"{component}:missing")
            continue
        territories = entry["anchor_territories"]
        expected_pairs = {
            tuple(sorted((left, right)))
            for a, b in itertools.combinations(certificate.distinct_anchor_ids, 2)
            for left in territories.get(a, [])
            for right in territories.get(b, [])
        }
        actual_pairs = {tuple(sorted(pair)) for pair in entry["pairs"]}
        if expected_pairs != actual_pairs or not expected_pairs:
            constraint_failures.append(f"{component}:pair mismatch")
    outcomes.append(
        _outcome(
            "distinct_anchor_territories_have_canonical_cannot_links",
            not constraint_failures,
            "none" if not constraint_failures else "; ".join(constraint_failures),
        )
    )

    order_failures: list[str] = []
    for component in {action.targets[0].stable_id for action in plan.actions if action.targets}:
        operations = [
            action.operation
            for action in plan.actions
            if action.targets and action.targets[0].stable_id == component
        ]
        expected = [
            "split_by_anchor",
            "consolidate_same_anchor",
            "forbid_merge",
            "rebuild_local_rag",
        ]
        if operations and operations != expected:
            order_failures.append(f"{component}:{operations}")
    outcomes.append(
        _outcome(
            "exclusion_then_consolidation_then_constraint_order",
            not order_failures,
            "none" if not order_failures else "; ".join(order_failures),
        )
    )
    outcomes.append(
        _outcome(
            "second_checkpoint_would_plan_no_new_contained_repairs",
            not idempotence_failures,
            "none" if not idempotence_failures else "; ".join(idempotence_failures),
        )
    )

    budget_failures: list[str] = []
    component_stats = result.summary_statistics.get("component_execution", {})
    for component in result.repaired_components:
        stats: Any = component_stats[component]
        parent = int(stats["parent_voxels"])
        if int(stats["abstained_voxels"]) > 0.05 * parent:
            budget_failures.append(f"{component}:abstention")
        if any(int(count) < 0.01 * parent for count in stats["assigned_voxels"].values()):
            budget_failures.append(f"{component}:territory")
    outcomes.append(
        _outcome(
            "abstention_and_minimum_territory_caps",
            not budget_failures,
            "none" if not budget_failures else "; ".join(budget_failures),
        )
    )
    outcomes.append(
        _outcome(
            "local_rag_state_rebuilt",
            len(rag.get("records", []))
            == sum(action.operation == "rebuild_local_rag" for action in plan.actions),
            f"records={len(rag.get('records', []))}",
        )
    )
    outcomes.append(
        _outcome(
            "all_planned_actions_executed_without_failure",
            not any(execution.status == "failed" for execution in result.action_executions),
            f"failed={sum(execution.status == 'failed' for execution in result.action_executions)}",
        )
    )

    if replay_determinism:
        replay = execute_plan(plan, output_dir / "determinism_replay")
        same_hash = (
            replay.summary_statistics["output_segmentation_sha256"]
            == result.summary_statistics["output_segmentation_sha256"]
        )
        replay_constraints = json.loads(_output_artifact(replay, "cannot_links").read_text())
        same_constraints = content_hash(replay_constraints) == content_hash(constraints)
        outcomes.append(
            _outcome(
                "independent_apply_is_deterministic",
                same_hash and same_constraints,
                f"segmentation_hash_equal={same_hash} constraints_equal={same_constraints}",
            )
        )
    else:
        outcomes.append(
            _outcome(
                "independent_apply_is_deterministic",
                False,
                "NOT RUN: independent replay disabled",
            )
        )

    provisional = CheckpointResult(
        **{
            **result.__dict__,
            "result_id": "pending",
            "verification": tuple(outcomes),
        }
    )
    return CheckpointResult(
        **{**provisional.__dict__, "result_id": stable_id("result", provisional)}
    )


def verification_passed(result: CheckpointResult) -> bool:
    return bool(result.verification) and all(outcome.passed for outcome in result.verification)


__all__ = ["verification_passed", "verify_result"]
