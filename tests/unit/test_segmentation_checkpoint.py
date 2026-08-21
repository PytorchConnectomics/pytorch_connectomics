from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomics.decoding.checkpoint.actions import PreconditionError, execute_plan
from connectomics.decoding.checkpoint.engine import (
    apply_pass,
    describe_pass,
    load_description,
    load_plan,
    plan_pass,
    verify_pass,
)
from connectomics.decoding.checkpoint.io import ArrayVolume, read_bbox_chunked
from connectomics.decoding.checkpoint.kernels import affinity_cost, competitive_split_component
from connectomics.decoding.checkpoint.schema import BoundingBox, CheckpointPlan
from connectomics.decoding.checkpoint.serialization import canonical_json, read_json, write_json
from connectomics.decoding.checkpoint.verification import verification_passed

FIXTURE = Path(__file__).parents[1] / "fixtures/checkpoint"


def _line_case(*, one_anchor: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segmentation = np.zeros((3, 3, 9), dtype=np.uint64)
    segmentation[1, 1, 1:8] = 7
    nuclei = np.zeros_like(segmentation, dtype=np.uint16)
    nuclei[1, 1, 2] = 11
    if not one_anchor:
        nuclei[1, 1, 6] = 22
    affinity = np.ones((3,) + segmentation.shape, dtype=np.float32)
    return segmentation, nuclei, affinity


def _write_case(
    root: Path,
    segmentation: np.ndarray,
    nuclei: np.ndarray,
    affinity: np.ndarray,
    *,
    min_share: float = 0.5,
    contact: bool = True,
    scope: tuple[int, int, int, int, int, int] | None = None,
    anchor_ids: tuple[int, ...] = (11, 22),
    contact_bbox: tuple[int, int, int, int, int, int] | None = None,
    max_read_bytes: int = 1024,
) -> tuple[dict[str, object], float]:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "seg.npy", segmentation)
    np.save(root / "nuc.npy", nuclei)
    np.save(root / "aff.npy", affinity)
    rows = []
    if contact:
        row: dict[str, object] = {
            "seg_id": "7",
            "anchor_ids": list(anchor_ids),
            "provenance": "synthetic caller-supplied contact",
            "gap_um": 0.0,
            "eligibility": "contact",
        }
        if contact_bbox is not None:
            row["bbox_zyx"] = list(contact_bbox)
        rows.append(row)
    contacts = {"schema_version": "1.0", "artifact_id": "synthetic", "scopes": rows}
    (root / "contact_scopes.json").write_text(json.dumps(contacts))
    if scope is None:
        scope = (0, 0, 0, *segmentation.shape)
    spec: dict[str, object] = {
        "checkpoint_id": "synthetic",
        "pass_id": "nucleus_anchor",
        "operator": "nucleus_anchor",
        "segmentation": {"uri": str(root / "seg.npy")},
        "nuclei": {"uri": str(root / "nuc.npy")},
        "affinity": {"uri": str(root / "aff.npy")},
        "scope_zyx": list(scope),
        "contact_scopes": str(root / "contact_scopes.json"),
        "channel_indices": [0, 1, 2],
        "affinity_channel_axis": 0,
        "affinity_convention": "probability",
        "pooling_factor": 1,
        "nucleus_scale_zyx": [1, 1, 1],
        "max_read_bytes": max_read_bytes,
    }
    return spec, min_share


def _serialized_apply(
    root: Path, spec: dict[str, object], min_share: float
) -> tuple[CheckpointPlan, object]:
    description = describe_pass(spec, root / "run", min_share=min_share)
    description = load_description(root / "run/description.json")
    plan_pass(spec, description, root / "run/plan.json", min_share=min_share)
    plan = load_plan(root / "run/plan.json")
    result = apply_pass(plan, root / "apply")
    return plan, result


def _territory_masks(result: object, corrected: np.ndarray) -> dict[str, np.ndarray]:
    manifest_path = next(
        Path(artifact.uri)
        for artifact in result.output_artifacts
        if artifact.role == "cannot_links"
    )
    entry = json.loads(manifest_path.read_text())["constraints"][0]
    return {
        anchor: np.isin(corrected, np.asarray([int(value) for value in territories]))
        for anchor, territories in entry["anchor_territories"].items()
    }


def test_01_one_component_one_nucleus_describes_without_repair(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case(one_anchor=True)
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity, anchor_ids=(11,))
    description = describe_pass(spec, tmp_path / "run", min_share=threshold)
    assert {item.key for item in description.descriptors} >= {
        "component.volume_voxels",
        "anchor.distinct_count",
    }
    assert description.certificates == ()
    plan = plan_pass(spec, description, tmp_path / "plan.json", min_share=threshold)
    assert plan.actions == ()


def test_02_two_nuclei_certify_split_and_export_cannot_link(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    plan, result = _serialized_apply(tmp_path, spec, threshold)
    assert len(plan.certificates) == 1
    assert plan.certificates[0].strength == "hard"
    assert [action.operation for action in plan.actions] == [
        "split_by_anchor",
        "consolidate_same_anchor",
        "forbid_merge",
        "rebuild_local_rag",
    ]
    verified = verify_pass(plan, result, tmp_path / "apply")
    assert verification_passed(verified)
    manifest = json.loads(
        next(Path(a.uri) for a in result.output_artifacts if a.role == "cannot_links").read_text()
    )
    assert manifest["consumer_status"] == "write_only_no_decoder_consumer"
    assert len(manifest["constraints"][0]["pairs"]) == 1
    assert all(execution.elapsed_seconds >= 0 for execution in result.action_executions)
    assert all(execution.affected_voxels >= 0 for execution in result.action_executions)


def test_03_atomic_supervoxel_splits_through_serialized_boundary(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    _plan, result = _serialized_apply(tmp_path, spec, threshold)
    delta = next(Path(a.uri) for a in result.output_artifacts if a.role == "segmentation_delta")
    with np.load(delta) as data:
        corrected = segmentation.copy()
        indices = data["changed_indices_zyx"]
        corrected[tuple(indices.T)] = data["changed_values"]
        abstention = data["abstention_mask"]
    masks = _territory_masks(result, corrected)
    expected_11 = np.zeros_like(segmentation, dtype=bool)
    expected_11[1, 1, 1:5] = True
    expected_22 = np.zeros_like(segmentation, dtype=bool)
    expected_22[1, 1, 5:8] = True
    np.testing.assert_array_equal(masks["11"], expected_11)
    np.testing.assert_array_equal(masks["22"], expected_22)
    assert not abstention.any()
    assert int((masks["11"] & (nuclei == 0)).sum()) == 3
    assert int((masks["22"] & (nuclei == 0)).sum()) == 2


def test_04_consolidation_follows_exclusion_and_never_joins_distinct_anchors(
    tmp_path: Path,
) -> None:
    segmentation, nuclei, affinity = _line_case()
    segmentation[0, 0, 0] = 7
    nuclei[0, 0, 0] = 11
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    plan, result = _serialized_apply(tmp_path, spec, threshold)
    operations = [action.operation for action in plan.actions]
    assert operations.index("split_by_anchor") < operations.index("consolidate_same_anchor")
    assert operations.index("consolidate_same_anchor") < operations.index("forbid_merge")
    forbid = next(e for e in result.action_executions if e.operation == "forbid_merge")
    split = next(e for e in result.action_executions if e.operation == "split_by_anchor")
    assert len(split.territory_bindings["11"]) == 2
    assert len(forbid.territory_bindings["11"]) == 1
    assert set(forbid.territory_bindings) == {"11", "22"}
    assert forbid.territory_bindings["11"] != forbid.territory_bindings["22"]


def test_05_minority_contamination_uses_anchor_relative_mass(tmp_path: Path) -> None:
    segmentation = np.zeros((5, 5, 12), dtype=np.uint64)
    segmentation[2, 2, 1:8] = 7
    nuclei = np.zeros_like(segmentation, dtype=np.uint16)
    nuclei[2, 2, 2] = 11
    nuclei[2, 2, 6:8] = 22
    nuclei[1, 1, 1:9] = 22  # anchor 22 is dominant outside the fused component
    affinity = np.ones((3,) + segmentation.shape, dtype=np.float32)
    spec, _ = _write_case(tmp_path, segmentation, nuclei, affinity, min_share=0.2)
    description = describe_pass(spec, tmp_path / "run", min_share=0.2)
    assert len(description.certificates) == 1
    fractions = next(d.value for d in description.descriptors if d.key == "anchor.overlap_fraction")
    assert fractions["22"] == pytest.approx(0.2)


def test_06_subthreshold_overlap_does_not_certify(tmp_path: Path) -> None:
    segmentation = np.zeros((5, 5, 14), dtype=np.uint64)
    segmentation[2, 2, 1:8] = 7
    nuclei = np.zeros_like(segmentation, dtype=np.uint16)
    nuclei[2, 2, 2] = 11
    nuclei[2, 2, 6] = 22
    nuclei[1, 1, 1:10] = 22
    affinity = np.ones((3,) + segmentation.shape, dtype=np.float32)
    spec, _ = _write_case(tmp_path, segmentation, nuclei, affinity, min_share=0.2)
    description = describe_pass(spec, tmp_path / "run", min_share=0.2)
    assert description.certificates == ()


def test_07_scope_safety_and_partial_containment_downgrade(tmp_path: Path) -> None:
    segmentation = np.zeros((3, 3, 11), dtype=np.uint64)
    segmentation[1, 1, 1:10] = 7
    nuclei = np.zeros_like(segmentation, dtype=np.uint16)
    nuclei[1, 1, 3] = 11
    nuclei[1, 1, 6] = 22
    affinity = np.ones((3,) + segmentation.shape, dtype=np.float32)
    scope = (0, 0, 2, 3, 3, 8)
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity, scope=scope)
    plan, result = _serialized_apply(tmp_path, spec, threshold)
    split = next(
        action.parameters for action in plan.actions if action.operation == "split_by_anchor"
    )
    assert split.separation_claim == "local_only"
    assert result.repaired_components == ()
    assert result.certified_unrepaired[0].reason == "local_only"
    with np.load(
        next(Path(a.uri) for a in result.output_artifacts if a.role == "segmentation_delta")
    ) as d:
        assert np.all((d["changed_indices_zyx"][:, 2] >= 0) & (d["changed_indices_zyx"][:, 2] < 6))
    verified = verify_pass(plan, result, tmp_path / "apply")
    containment = next(
        outcome
        for outcome in verified.verification
        if outcome.invariant == "partial_scope_new_territories_do_not_touch_continuing_boundary"
    )
    assert containment.passed


def test_08_second_run_is_noop(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path / "first", segmentation, nuclei, affinity)
    _plan, result = _serialized_apply(tmp_path / "first", spec, threshold)
    delta = next(Path(a.uri) for a in result.output_artifacts if a.role == "segmentation_delta")
    corrected = segmentation.copy()
    with np.load(delta) as data:
        corrected[tuple(data["changed_indices_zyx"].T)] = data["changed_values"]
    spec2, threshold2 = _write_case(tmp_path / "second", corrected, nuclei, affinity)
    description2 = describe_pass(spec2, tmp_path / "second/run", min_share=threshold2)
    plan2 = plan_pass(spec2, description2, tmp_path / "second/plan.json", min_share=threshold2)
    assert description2.certificates == ()
    assert plan2.actions == ()


def test_09_plan_and_result_serialization_round_trip(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    plan, result = _serialized_apply(tmp_path, spec, threshold)
    write_json(tmp_path / "roundtrip_plan.json", plan)
    write_json(tmp_path / "roundtrip_result.json", result)
    assert read_json(tmp_path / "roundtrip_plan.json") == plan
    assert read_json(tmp_path / "roundtrip_result.json") == result


def test_10_plan_and_execution_are_deterministic_and_ties_choose_lowest_anchor(
    tmp_path: Path,
) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    first = describe_pass(spec, tmp_path / "d1", min_share=threshold)
    second = describe_pass(spec, tmp_path / "d2", min_share=threshold)
    plan1 = plan_pass(spec, first, tmp_path / "p1.json", min_share=threshold)
    plan2 = plan_pass(spec, second, tmp_path / "p2.json", min_share=threshold)
    assert canonical_json(plan1, exclude_volatile=True) == canonical_json(
        plan2, exclude_volatile=True
    )
    result1 = apply_pass(plan1, tmp_path / "a1")
    result2 = apply_pass(plan1, tmp_path / "a2")
    assert (
        result1.summary_statistics["output_segmentation_sha256"]
        == result2.summary_statistics["output_segmentation_sha256"]
    )
    delta = next(Path(a.uri) for a in result1.output_artifacts if a.role == "segmentation_delta")
    corrected = segmentation.copy()
    with np.load(delta) as data:
        corrected[tuple(data["changed_indices_zyx"].T)] = data["changed_values"]
    masks = _territory_masks(result1, corrected)
    assert masks["11"][1, 1, 4]  # exactly equidistant tie goes to lower anchor 11


def test_11_changed_input_fails_before_action_execution(tmp_path: Path) -> None:
    segmentation, nuclei, affinity = _line_case()
    spec, threshold = _write_case(tmp_path, segmentation, nuclei, affinity)
    description = describe_pass(spec, tmp_path / "run", min_share=threshold)
    plan = plan_pass(spec, description, tmp_path / "plan.json", min_share=threshold)
    segmentation[0, 0, 0] = 99
    np.save(tmp_path / "seg.npy", segmentation)
    with pytest.raises(PreconditionError, match="segmentation hash mismatch"):
        execute_plan(plan, tmp_path / "apply")
    assert not (tmp_path / "apply/execution.jsonl").exists()


def test_12_goldens_manifest_real_gate_legacy_contract_and_bounded_io(tmp_path: Path) -> None:
    for line in (FIXTURE / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((FIXTURE / relative).read_bytes()).hexdigest() == expected
    expected = json.loads((FIXTURE / "worst3_roi/expected_metrics.json").read_text())
    with np.load(FIXTURE / "worst3_roi/anchor_samples.npz") as data:
        nucleus = data["nucleus_id"]
        before = data["before_segment_id"]
        after = data["after_segment_id"]

    def metrics(values: np.ndarray) -> tuple[int, int, list[float]]:
        misplaced = 0
        dominance = []
        owners: dict[int, set[int]] = {}
        focus = set(expected["focus_nuclei"])
        for anchor in sorted(int(value) for value in np.unique(nucleus)):
            labels, counts = np.unique(values[nucleus == anchor], return_counts=True)
            misplaced += int(counts.sum() - counts.max())
            if anchor not in focus:
                continue
            dominance.append(float(counts.max() / counts.sum()))
            for label, count in zip(labels.tolist(), counts.tolist()):
                if label != 0 and count >= expected["min_share"] * counts.sum():
                    owners.setdefault(int(label), set()).add(anchor)
        shared = sum(len(anchor_ids) > 1 for anchor_ids in owners.values())
        return misplaced, shared, dominance

    before_metrics = metrics(before)
    after_metrics = metrics(after)
    assert before_metrics[0] == expected["misplaced_mask_voxels"]["before"] == 115_479
    assert after_metrics[0] == expected["misplaced_mask_voxels"]["after"] == 5_071
    assert before_metrics[1] == expected["shared_segments"]["before"] == 1
    assert after_metrics[1] == expected["shared_segments"]["after"] == 0
    np.testing.assert_allclose(before_metrics[2], expected["dominance"]["before"])
    np.testing.assert_allclose(after_metrics[2], expected["dominance"]["after"])
    assert min(after_metrics[2]) >= 0.99

    pairs = (
        ("scripts/nucleus_split.py", "nucleus_competitive_split.help.txt"),
        ("scripts/nucleus_anchor_merge.py", "nucleus_anchor_merge.help.txt"),
        ("scripts/nucleus_contamination.py", "nucleus_shell_contamination.help.txt"),
    )
    for script, golden in pairs:
        completed = subprocess.run(
            [sys.executable, script, "--help"], check=True, capture_output=True, text=True
        )
        option = re.compile(r"--[a-z][a-z0-9-]*")
        assert set(option.findall(completed.stdout)) == set(
            option.findall((FIXTURE / "legacy_cli" / golden).read_text())
        )

    class RecordingVolume(ArrayVolume):
        def __init__(self, array: np.ndarray) -> None:
            super().__init__(array)
            self.read_sizes: list[int] = []

        def read(self, bbox: BoundingBox) -> np.ndarray:
            block = super().read(bbox)
            self.read_sizes.append(block.nbytes)
            return block

    reader = RecordingVolume(np.zeros((10, 20, 30), dtype=np.uint64))
    result = read_bbox_chunked(reader, BoundingBox((0, 0, 0), (10, 20, 30)), 512)
    assert result.shape == (10, 20, 30)
    assert max(reader.read_sizes) <= 512


def test_pinned_affinity_channel_min_pooling_and_six_connected_axis_order() -> None:
    affinity = np.ones((4, 2, 2, 2), dtype=np.float32)
    affinity[0, 0, 0, 0] = 0.1
    affinity[3] = 0.0  # explicitly unselected channel cannot affect cost
    cost = affinity_cost(
        affinity,
        channel_indices=(0, 1, 2),
        channel_axis=0,
        convention="probability",
        sigmoid_restore=None,
        factor=2,
    )
    np.testing.assert_allclose(cost, np.asarray([[[0.9]]], dtype=np.float32))

    diagonal = np.zeros((3, 3, 3), dtype=np.uint64)
    diagonal[(0, 1, 2), (0, 1, 2), (0, 1, 2)] = 7
    nuclei = np.zeros_like(diagonal)
    nuclei[0, 0, 0] = 11
    nuclei[2, 2, 2] = 22
    affinities = np.ones((3,) + diagonal.shape, dtype=np.float32)
    split = competitive_split_component(
        diagonal,
        nuclei,
        affinities,
        component_id=7,
        anchor_ids=(11, 22),
        channel_indices=(0, 1, 2),
    )
    assert split.abstention_mask[1, 1, 1]
