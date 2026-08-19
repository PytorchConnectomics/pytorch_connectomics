"""Contracts for ABISS competitive nucleus growth and sparse RAG overlays."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import shlex
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pytest

from connectomics.runtime import abiss_chunk

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "lib" / "abiss" / "scripts"
COMPARISON_HARNESS = ROOT / "dev" / "zebrafinch" / "compare_nucleus_competition.py"
MIGRATION_TOOL = SCRIPTS / "migrate_nucleus_competition.py"
ACCEPTANCE_REPORT = ROOT / "dev" / "zebrafinch" / "nucleus_acceptance_report.py"
NATIVE96_RUN = ROOT / "dev" / "zebrafinch" / "wholevol_arm0_native96_nuc_matchguard"
WIN144_RUN = ROOT / "dev" / "zebrafinch" / "wholevol_arm096_nuc_competitive_v2"
NATIVE96_PUBLICATION = NATIVE96_RUN / "seg_arm0_native96_nuc_matchguard" / "nucleus_competition"
LEGACY_ORACLE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "abiss_nucleus_competition" / "legacy_emitted_labels.json"
)
REQUIRE_ZEBRAFINCH_ARTIFACTS = os.environ.get("PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS") == "1"


def _missing_zebrafinch_artifacts(*paths: Path) -> tuple[Path, ...]:
    return tuple(path for path in paths if not path.is_dir())


def _missing_zebrafinch_artifacts_reason(*paths: Path) -> str:
    missing = _missing_zebrafinch_artifacts(*paths)
    noun = "directory" if len(missing) == 1 else "directories"
    return f"missing gitignored zebrafinch artifact {noun}: " + ", ".join(map(str, missing))


def _skipif_missing_zebrafinch_artifacts(*paths: Path) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        bool(_missing_zebrafinch_artifacts(*paths)) and not REQUIRE_ZEBRAFINCH_ARTIFACTS,
        reason=_missing_zebrafinch_artifacts_reason(*paths),
    )


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"abiss_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_comparison_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "nucleus_competition_comparison", COMPARISON_HARNESS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tool(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_manifest(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ws_manifest = tmp_path / "watershed_manifest.json"
    ws_manifest.write_text(json.dumps({"abiss_build_id": "test", "provenance_sha": "test"}))
    run_dir = tmp_path / ".nuccomp-runs" / "test-run"
    run_dir.mkdir(parents=True)
    units_path = run_dir / "units.json"
    units_path.write_text(json.dumps({"plan_type": "test"}, sort_keys=True) + "\n")
    plan_digest = _sha256(units_path)
    manifest: dict[str, Any] = {
        "schema_version": "3.0",
        "required_capabilities": [],
        "manifest_type": "abiss_nucleus_competition",
        "plan_digest": plan_digest,
        "plan_file": str(units_path.relative_to(tmp_path)),
        "completion": {"state": "complete", "plan_digest": plan_digest},
        "fingerprints": {
            "watershed": {
                "manifest_file": str(ws_manifest),
                "manifest_sha256": _sha256(ws_manifest),
            }
        },
        **payload,
    }
    overlay = _load_script("nucleus_overlay")
    manifest.setdefault("identity", overlay.identity_declaration())
    manifest.setdefault("zero_repairs", not manifest.get("repairs"))
    manifest.setdefault(
        "reason",
        (
            "competitive_repairs_completed"
            if manifest.get("repairs")
            else "no_competitive_contact_units"
        ),
    )
    manifest.setdefault(
        "ledger",
        overlay.build_publication_ledger(
            manifest.get("repairs", []), manifest.get("qualified_segment_labels", {})
        ),
    )
    for repair in manifest.get("repairs", []):
        territory_file = repair.get("territory_file")
        if territory_file:
            repair.setdefault("territory_sha256", _sha256(tmp_path / territory_file))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "WS_PATH": "file:///ws",
        "WS_MANIFEST": str(ws_manifest),
        "NUC_COMPETITION_MANIFEST": str(manifest_path),
        "NUC_COMPETITION_PLAN_DIGEST": plan_digest,
    }


def _owner_id(nucleus_id: int) -> int:
    payload = f"nucleus-owner:{nucleus_id}".encode("ascii")
    return (1 << 60) + int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")


def _copy_publication_as_schema_1_2(source: Path, destination: Path) -> Path:
    """Copy a publication and rewind its manifest to the preserved schema-1.2 bytes.

    The reference publication is migrated in place once it goes into production, so a copy
    of it cannot be assumed to still be legacy. Migration preserves the original manifest
    beside the new one, and that is what these tests must migrate from.
    """

    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    if json.loads(manifest_path.read_text()).get("schema_version") == "1.2":
        return manifest_path
    backups = sorted(destination.glob("manifest.schema-1.2.*.json"))
    if not backups:
        raise AssertionError(f"no preserved schema-1.2 manifest to restore in {source}")
    shutil.copyfile(backups[-1], manifest_path)
    return manifest_path


@pytest.fixture
def migrated_native96_manifest(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    assert not _missing_zebrafinch_artifacts(
        NATIVE96_PUBLICATION
    ), _missing_zebrafinch_artifacts_reason(NATIVE96_PUBLICATION)
    publication = tmp_path / "nucleus_competition"
    manifest_path = _copy_publication_as_schema_1_2(NATIVE96_PUBLICATION, publication)
    migration = _load_tool("nucleus_competition_migration", MIGRATION_TOOL)
    return migration.migrate_manifest(manifest_path), manifest_path


class _ScalarVolume:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape + (1,)
        self.dtype = array.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        block = self.array[key[:3]]
        return block[..., np.newaxis]


class _AffinityVolume:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        return self.array[key[:3]]


def test_scan_select_and_contact_filter_are_nucleus_relative() -> None:
    competition = _load_script("nucleus_competition")
    shape = (40, 16, 12)
    segmentation: np.ndarray = np.zeros(shape, dtype=np.uint64)
    segmentation[2:38, 2:14, 2:10] = 7
    nuclei: np.ndarray = np.zeros(shape, dtype=np.uint32)
    nuclei[5:9, 5:9, 4:8] = 1
    nuclei[14:18, 7:11, 4:8] = 2
    nuclei[32:36, 7:11, 4:8] = 3
    seg_volume = _ScalarVolume(segmentation)
    nucleus_volume = _ScalarVolume(nuclei)

    stats = competition.scan_nucleus_geometry(
        nucleus_volume,
        (0, 0, 0, *shape),
        (1, 1, 1),
        (0, 0, 0),
        block_z=4,
    )
    histograms = competition.nucleus_segment_histograms(
        nucleus_volume,
        seg_volume,
        stats,
        (1, 1, 1),
        (0, 0, 0),
    )
    targets, _shares = competition.qualifying_targets(histograms, stats, 0.02)
    units, bridges = competition.contact_units(
        targets,
        stats,
        (1, 1, 1),
        (0, 0, 0),
        (1000.0, 1000.0, 1000.0),
        contact_um=8.0,
    )

    assert targets == {7: (1, 2, 3)}
    assert [unit["anchor_ids"] for unit in units] == [(1, 2)]
    assert bridges == [
        {
            "parent_id": "7",
            "groups": [[1, 2], [3]],
            "min_cross_gap_um": pytest.approx(13.0371960728),
        }
    ]


def test_histograms_sample_the_nearest_neighbour_footprint() -> None:
    competition = _load_script("nucleus_competition")
    low_shape = (3, 3, 3)
    high_shape = tuple(2 * size for size in low_shape)
    nuclei: np.ndarray = np.zeros(low_shape, dtype=np.uint32)
    nuclei[1, 1, 1] = 1
    nuclei[2, 1, 1] = 2

    segmentation: np.ndarray = np.full(high_shape, 7, dtype=np.uint64)
    # Both nucleus centers remain in segment 7. Segment 9 is visible only on
    # the low-x face of each 2x2x2 nearest-neighbour footprint.
    segmentation[1, 1:3, 1:3] = 9
    segmentation[3, 1:3, 1:3] = 9

    nucleus_volume = _ScalarVolume(nuclei)
    stats = competition.scan_nucleus_geometry(
        nucleus_volume,
        (0, 0, 0, *high_shape),
        (2, 2, 2),
        (0, 0, 0),
        block_z=2,
    )
    histograms = competition.nucleus_segment_histograms(
        nucleus_volume,
        _ScalarVolume(segmentation),
        stats,
        (2, 2, 2),
        (0, 0, 0),
    )
    targets, shares = competition.qualifying_targets(histograms, stats, 0.2)

    assert targets[9] == (1, 2)
    assert shares[9][1] == pytest.approx(0.5)
    assert shares[9][2] == pytest.approx(0.5)


def test_seeded_flood_refines_only_the_parent_component() -> None:
    pytest.importorskip("skimage")
    competition = _load_script("nucleus_competition")
    shape = (24, 16, 12)
    segmentation: np.ndarray = np.zeros(shape, dtype=np.uint64)
    segmentation[2:22, 2:14, 2:10] = 7
    nuclei: np.ndarray = np.zeros(shape, dtype=np.uint32)
    nuclei[5:9, 5:9, 4:8] = 1
    nuclei[14:18, 7:11, 4:8] = 2
    affinity = np.full(shape + (3,), 0.9, dtype=np.float32)
    affinity[11:13, :, :, :] = 0.1

    territory, counts, labels = competition.flood_unit(
        _AffinityVolume(affinity),
        _ScalarVolume(segmentation),
        _ScalarVolume(nuclei),
        {"parent_id": 7, "anchor_ids": (1, 2)},
        (2, 2, 2, 22, 14, 10),
        (1, 1, 1),
        (0, 0, 0),
        factor=2,
        affinity_channels=(0, 1, 2),
        slab_z=4,
    )

    assert set(np.unique(territory)) == {1, 2}
    assert counts[1] > 0 and counts[2] > 0
    assert labels[1] != labels[2]
    assert 7 not in labels.values()
    assert min(labels.values()) >= competition.NEW_ID_BASE


def test_sparse_overlay_is_consistent_for_partial_chunk_intersection(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")
    parent_id = 7
    winner_internal_id = (1 << 60) + 11
    owner_11 = _owner_id(11)
    owner_22 = _owner_id(22)
    new_internal_id = (1 << 60) + 23
    territory: np.ndarray = np.full((2, 2, 2), winner_internal_id, dtype=np.uint64)
    territory[1, :, :] = new_internal_id
    territory_path = tmp_path / "territory.npz"
    np.savez_compressed(
        territory_path,
        territory=territory,
        bbox_xyz=np.asarray([4, 4, 2, 8, 8, 6], dtype=np.int64),
        factor=np.asarray(2, dtype=np.int64),
    )
    params = _write_complete_manifest(
        tmp_path,
        {
            "qualified_segment_owners": {"7": [11, 22]},
            "qualified_segment_labels": {"7": {"11": str(owner_11), "22": str(owner_22)}},
            "repairs": [
                {
                    "parent_id": str(parent_id),
                    "bbox_xyz": [4, 4, 2, 8, 8, 6],
                    "factor": 2,
                    "territory_file": territory_path.name,
                    "territories": [
                        {
                            "anchor_id": "11",
                            "internal_territory_id": str(winner_internal_id),
                            "emitted_id": str(owner_11),
                        },
                        {
                            "anchor_id": "22",
                            "internal_territory_id": str(new_internal_id),
                            "emitted_id": str(owner_22),
                        },
                    ],
                }
            ],
        },
    )
    cutout: np.ndarray = np.full((4, 6, 6, 1), parent_id, dtype=np.uint64)

    result = overlay.apply_nucleus_competition(
        cutout,
        chunk_start_xyz=(6, 3, 1),
        global_params=params,
    )

    expected: np.ndarray = np.full((4, 6, 6), parent_id, dtype=np.uint64)
    expected[0:2, 1:5, 1:5] = owner_22
    np.testing.assert_array_equal(result[..., 0], expected)


def test_sparse_overlay_tags_every_competitive_territory_with_its_owner(
    tmp_path: Path,
) -> None:
    overlay = _load_script("nucleus_overlay")
    winner_internal_id = (1 << 60) + 11
    owner_11 = _owner_id(11)
    owner_22 = _owner_id(22)
    new_internal_id = (1 << 60) + 23
    territory: np.ndarray = np.full((2, 2, 2), winner_internal_id, dtype=np.uint64)
    territory[1, :, :] = new_internal_id
    np.savez_compressed(
        tmp_path / "territory.npz",
        territory=territory,
        bbox_xyz=np.asarray([0, 0, 0, 4, 4, 4], dtype=np.int64),
        factor=np.asarray(2, dtype=np.int64),
    )
    params = _write_complete_manifest(
        tmp_path,
        {
            "qualified_segment_owners": {"7": [11, 22]},
            "qualified_segment_labels": {"7": {"11": str(owner_11), "22": str(owner_22)}},
            "repairs": [
                {
                    "parent_id": "7",
                    "bbox_xyz": [0, 0, 0, 4, 4, 4],
                    "factor": 2,
                    "territory_file": "territory.npz",
                    "territories": [
                        {
                            "anchor_id": "11",
                            "internal_territory_id": str(winner_internal_id),
                            "emitted_id": str(owner_11),
                        },
                        {
                            "anchor_id": "22",
                            "internal_territory_id": str(new_internal_id),
                            "emitted_id": str(owner_22),
                        },
                    ],
                }
            ],
        },
    )
    segmentation = np.asfortranarray(np.full((4, 4, 4, 1), 7, dtype=np.uint64))
    sparse_nuclei = np.asfortranarray(np.zeros((4, 4, 4, 1), dtype=np.uint32))
    sparse_nuclei[0, 0, 0, 0] = 11
    sparse_nuclei[3, 0, 0, 0] = 22

    refined, ownership = overlay.apply_nucleus_competition_state(
        segmentation,
        sparse_nuclei,
        (0, 0, 0),
        params,
    )

    assert np.all(ownership[:2, ..., 0] == 11)
    assert np.all(ownership[2:, ..., 0] == 22)
    assert np.all(refined[:2, ..., 0] == owner_11)
    assert np.all(refined[2:, ..., 0] == owner_22)


def test_subshare_filter_only_changes_protected_owners(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")
    params = _write_complete_manifest(
        tmp_path,
        {
            "protected_nucleus_owners": [11],
            "qualified_segment_owners": {"7": [11]},
            "qualified_segment_labels": {"7": {"11": str(_owner_id(11))}},
            "repairs": [],
        },
    )
    segmentation = np.asfortranarray(np.asarray([7, 7, 9, 9], dtype=np.uint64).reshape(2, 2, 1, 1))
    sparse_nuclei = np.asfortranarray(
        np.asarray([11, 22, 11, 22], dtype=np.uint32).reshape(2, 2, 1, 1)
    )
    assert segmentation.flags.f_contiguous and not segmentation.flags.c_contiguous
    assert sparse_nuclei.flags.f_contiguous and not sparse_nuclei.flags.c_contiguous

    refined, ownership = overlay.apply_nucleus_competition_state(
        segmentation,
        sparse_nuclei,
        (0, 0, 0),
        params,
    )

    np.testing.assert_array_equal(ownership[..., 0].reshape(-1), [11, 22, 0, 22])
    np.testing.assert_array_equal(refined[..., 0].reshape(-1), [_owner_id(11), _owner_id(11), 9, 9])


def test_zero_repair_publication_still_runs_canonicalization_and_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = _load_script("nucleus_overlay")
    params = _write_complete_manifest(
        tmp_path,
        {
            "protected_nucleus_owners": [],
            "qualified_segment_owners": {"7": [11]},
            "qualified_segment_labels": {"7": {"11": str(_owner_id(11))}},
            "repairs": [],
        },
    )
    segmentation = np.asfortranarray(np.asarray([7, 7, 9, 9], dtype=np.uint64).reshape(2, 2, 1, 1))
    nuclei = np.asfortranarray(np.asarray([11, 0, 22, 0], dtype=np.uint32).reshape(2, 2, 1, 1))
    expected_segmentation = segmentation.copy()
    expected_segmentation[:1, :, :, 0] = _owner_id(11)
    expected_nuclei = nuclei.copy()
    calls = []
    original_filter = overlay.filter_sparse_nucleus_ownership
    original_canonicalize = overlay.canonicalize_qualified_segments

    def filter_sibling(*args: Any, **kwargs: Any) -> int:
        calls.append("filter")
        return original_filter(*args, **kwargs)

    def canonicalize_sibling(*args: Any, **kwargs: Any) -> int:
        calls.append("canonicalize")
        return original_canonicalize(*args, **kwargs)

    monkeypatch.setattr(overlay, "filter_sparse_nucleus_ownership", filter_sibling)
    monkeypatch.setattr(overlay, "canonicalize_qualified_segments", canonicalize_sibling)

    refined, ownership = overlay.apply_nucleus_competition_state(
        segmentation,
        nuclei,
        (0, 0, 0),
        params,
    )

    np.testing.assert_array_equal(refined, expected_segmentation)
    np.testing.assert_array_equal(ownership, expected_nuclei)
    assert calls == ["filter", "canonicalize"]
    manifest = json.loads(Path(params["NUC_COMPETITION_MANIFEST"]).read_text())
    assert manifest["completion"]["state"] == "complete"
    assert manifest["reason"] == "no_competitive_contact_units"


def test_one_stable_label_is_shared_by_every_segment_of_an_owner(tmp_path: Path) -> None:
    competition = _load_script("nucleus_competition")
    overlay = _load_script("nucleus_overlay")
    protected, labels = competition.qualified_owner_labels(
        {7: {11: 0.5}, 9: {11: 0.3}, 10: {11: 0.2, 22: 0.4}},
        [{"anchor_ids": (11, 22)}],
    )
    owner_label = labels[7][11]
    assert protected == [11, 22]
    assert labels[9][11] == owner_label
    assert labels[10][11] == owner_label
    assert labels[10][22] != owner_label
    params = _write_complete_manifest(
        tmp_path,
        {
            "protected_nucleus_owners": [11],
            "qualified_segment_owners": {"7": [11], "9": [11]},
            "qualified_segment_labels": {
                "7": {"11": str(owner_label)},
                "9": {"11": str(owner_label)},
            },
            "repairs": [],
        },
    )
    segmentation = np.asarray([7, 9], dtype=np.uint64).reshape(2, 1, 1, 1)
    nuclei = np.asarray([11, 11], dtype=np.uint32).reshape(2, 1, 1, 1)

    refined, _ = overlay.apply_nucleus_competition_state(
        segmentation,
        nuclei,
        (0, 0, 0),
        params,
    )

    assert np.all(refined[..., 0] == owner_label)


def test_missing_competition_manifest_fails_closed(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")

    with pytest.raises(FileNotFoundError, match="run the competitive_nucleus_growth stage"):
        overlay.apply_nucleus_competition(
            np.zeros((2, 2, 2, 1), dtype=np.uint64),
            (0, 0, 0),
            {
                "WS_PATH": "file:///ws",
                "NUC_COMPETITION_MANIFEST": str(tmp_path / "missing.json"),
            },
        )


def test_legacy_gate2_fixtures_reproduce_completed_run_labels(tmp_path: Path) -> None:
    comparison = _load_comparison_harness()
    fixture = json.loads(LEGACY_ORACLE_FIXTURE.read_text())

    for case in fixture["cases"]:
        case_dir = tmp_path / case["name"]
        case_dir.mkdir()
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(json.dumps(case["manifest"]))
        np.savez_compressed(
            case_dir / "territory.npz",
            territory=np.asarray(case["territory"], dtype=np.int32),
            bbox_xyz=np.asarray([0, 0, 0, 8, 4, 4], dtype=np.int64),
            factor=np.asarray(4, dtype=np.int64),
        )
        parent_id = int(case["manifest"]["repairs"][0]["parent_id"])
        base: np.ndarray = np.full((8, 4, 4), parent_id, dtype=np.uint64)
        expected = base.copy()
        expected[:4, :, :] = int(case["expected_x_labels"][0])
        expected[4:, :, :] = int(case["expected_x_labels"][1])

        emitted = comparison.apply_emitted_labels(base, (0, 0, 0), manifest_path)

        np.testing.assert_array_equal(emitted, expected, err_msg=case["name"])


def test_schema_version_boundary_is_explicit(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")
    params = _write_complete_manifest(
        tmp_path,
        {
            "protected_nucleus_owners": [],
            "qualified_segment_owners": {},
            "qualified_segment_labels": {},
            "repairs": [],
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = overlay.load_validated_manifest(manifest_path, params)
    assert manifest["schema_version"] == "3.0"
    assert manifest["required_capabilities"] == []

    missing_capabilities = dict(manifest)
    missing_capabilities.pop("required_capabilities")
    manifest_path.write_text(json.dumps(missing_capabilities))
    with pytest.raises(ValueError, match="lacks required_capabilities"):
        overlay.load_validated_manifest(manifest_path, params)

    withdrawn = dict(manifest)
    withdrawn["schema_version"] = "2.0"
    manifest_path.write_text(json.dumps(withdrawn))
    with pytest.raises(ValueError, match="withdrawn pre-release contract") as error:
        overlay.load_validated_manifest(manifest_path, params)
    assert "inverted the parent-id rule" in str(error.value)
    assert "migrate_nucleus_competition.py" not in str(error.value)

    legacy = dict(manifest)
    legacy["schema_version"] = "1.2"
    manifest_path.write_text(json.dumps(legacy))
    # The overlay resolves the tool from its own directory, so the expected command is
    # built the same way rather than hardcoding a repo-relative path that only holds in
    # this checkout.
    command = (
        f"python {shlex.quote(str(MIGRATION_TOOL))} --manifest "
        f"{shlex.quote(str(manifest_path.resolve()))}"
    )

    with pytest.raises(ValueError, match="without recomputation") as error:
        overlay.load_validated_manifest(manifest_path, {})
    assert command in str(error.value)


def test_schema_1_0_rejection_does_not_claim_opaque_ids_are_reproducible(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_type": "abiss_nucleus_competition",
                "repairs": [],
            }
        )
    )

    with pytest.raises(ValueError, match="no lossless production migration"):
        overlay.load_validated_manifest(manifest_path, {})


def test_schema_1_2_migration_is_accepted_without_recomputing_territories(
    tmp_path: Path,
) -> None:
    overlay = _load_script("nucleus_overlay")
    migration = _load_tool("nucleus_competition_migration_tiny", MIGRATION_TOOL)
    owner_11 = _owner_id(11)
    owner_22 = _owner_id(22)
    manifest_path = tmp_path / "manifest.json"
    territory_path = tmp_path / "territory.npz"
    np.savez_compressed(
        territory_path,
        territory=np.asarray([[[1]], [[2]], [[0]]], dtype=np.int32),
        bbox_xyz=np.asarray([0, 0, 0, 3, 1, 1], dtype=np.int64),
        factor=np.asarray(1, dtype=np.int64),
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "manifest_type": "abiss_nucleus_competition",
                "base_watershed": "file:///unused/in-test",
                "nucleus_instances": "/nuclei.h5",
                "qualified_segment_labels": {"7": {"11": str(owner_11), "22": str(owner_22)}},
                "repairs": [
                    {
                        "parent_id": "7",
                        "anchor_ids": ["11", "22"],
                        "bbox_xyz": [0, 0, 0, 3, 1, 1],
                        "factor": 1,
                        "territory_file": territory_path.name,
                        "marker_labels": {"1": str(owner_11), "2": str(owner_22)},
                        "marker_nucleus_ids": {"1": "11", "2": "22"},
                        "pooled_voxels": {"11": 1, "22": 1},
                    }
                ],
            }
        )
    )
    ws_manifest = tmp_path / "watershed_manifest.json"
    ws_manifest.write_text(json.dumps({"abiss_build_id": "test", "provenance_sha": "test"}))
    territory_sha = _sha256(territory_path)

    migrated = migration.migrate_manifest(manifest_path, ws_manifest)

    assert _sha256(territory_path) == territory_sha
    assert migrated["schema_version"] == "3.0"
    assert migrated["required_capabilities"] == []
    assert migrated["migration"]["territory_recomputed"] is False
    assert migrated["repairs"][0]["territory_encoding"] == "marker_index"
    params = {
        "WS_PATH": "file:///unused/in-test",
        "WS_MANIFEST": str(ws_manifest),
        "NUC_COMPETITION_MANIFEST": str(manifest_path),
        "NUC_COMPETITION_PLAN_DIGEST": migrated["plan_digest"],
    }
    overlay.load_validated_manifest(manifest_path, params)
    base = np.asarray([7, 7, 7, 99], dtype=np.uint64).reshape(4, 1, 1, 1)
    emitted = overlay.apply_nucleus_competition(base, (0, 0, 0), params)
    np.testing.assert_array_equal(emitted[..., 0].reshape(-1), [owner_11, owner_22, 7, 99])
    minted = {int(value) for value in migrated["ledger"]["emitted_id_space"]["minted_ids"]}
    assert set(int(value) for value in np.unique(emitted)) <= minted | {7, 99}


@_skipif_missing_zebrafinch_artifacts(NATIVE96_PUBLICATION)
def test_native96_mint_is_nucleus_scoped_across_units_and_canonicalization(
    migrated_native96_manifest: tuple[dict[str, Any], Path],
) -> None:
    manifest, _path = migrated_native96_manifest
    mint = manifest["identity"]["mint"]
    observations = []
    parent_by_nucleus: dict[str, set[str]] = {}
    for repair in manifest["repairs"]:
        for item in repair["territories"]:
            observations.append((str(item["anchor_id"]), str(item["emitted_id"])))
            parent_by_nucleus.setdefault(str(item["anchor_id"]), set()).add(repair["parent_id"])
    for owner_labels in manifest["qualified_segment_labels"].values():
        observations.extend((str(owner), str(label)) for owner, label in owner_labels.items())

    nucleus_to_id: dict[str, set[str]] = {}
    id_to_nucleus: dict[str, set[str]] = {}
    for nucleus, emitted in observations:
        key = mint["key_template"].format(nucleus_id=int(nucleus)).encode("ascii")
        expected = int(mint["namespace_base"]) + int.from_bytes(
            hashlib.sha256(key).digest()[: int(mint["prefix_bytes"])], "big"
        )
        assert emitted == str(expected)
        nucleus_to_id.setdefault(nucleus, set()).add(emitted)
        id_to_nucleus.setdefault(emitted, set()).add(nucleus)
    assert all(len(values) == 1 for values in nucleus_to_id.values())
    assert all(len(values) == 1 for values in id_to_nucleus.values())
    assert len(parent_by_nucleus["373"]) == 2
    assert len(nucleus_to_id["373"]) == 1


@_skipif_missing_zebrafinch_artifacts(NATIVE96_PUBLICATION)
def test_native96_publication_ledger_closes_every_measured_retirement(
    migrated_native96_manifest: tuple[dict[str, Any], Path],
) -> None:
    manifest, _path = migrated_native96_manifest
    ledger = manifest["ledger"]
    competitive = [item for item in ledger["retirements"] if item["reason"] == "competitive_split"]
    canonical = [
        item for item in ledger["retirements"] if item["reason"] == "owner_canonicalization"
    ]
    assert len(competitive) == 8
    assert len(canonical) == 77
    assert len(ledger["consolidations"]) == 15
    assert all(len(item["sources"]) > 1 for item in ledger["consolidations"])
    assert max(len(item["sources"]) for item in ledger["consolidations"]) == 7
    declared_minted = set(ledger["emitted_id_space"]["minted_ids"])
    actual_minted = {
        str(item["emitted_id"]) for repair in manifest["repairs"] for item in repair["territories"]
    }
    actual_minted.update(
        str(label)
        for owners in manifest["qualified_segment_labels"].values()
        for label in owners.values()
    )
    assert declared_minted == actual_minted
    assert ledger["emitted_id_space"]["otherwise"] == "untouched_base_id"


@_skipif_missing_zebrafinch_artifacts(WIN144_RUN, NATIVE96_RUN)
def test_realization_gate_reproduces_the_two_published_run_oracles() -> None:
    assert not _missing_zebrafinch_artifacts(
        WIN144_RUN, NATIVE96_RUN
    ), _missing_zebrafinch_artifacts_reason(WIN144_RUN, NATIVE96_RUN)
    acceptance = _load_tool("nucleus_acceptance_realization", ACCEPTANCE_REPORT)

    win144 = acceptance.realization_gate(WIN144_RUN)
    native96 = acceptance.realization_gate(NATIVE96_RUN)

    assert (win144["realized_unit_count"], win144["unit_count"]) == (0, 9)
    assert (native96["realized_unit_count"], native96["unit_count"]) == (8, 8)
    assert (native96["realized_owner_count"], native96["owner_count"]) == (15, 15)


@_skipif_missing_zebrafinch_artifacts(NATIVE96_RUN)
def test_native96_realization_is_invariant_across_migration(tmp_path: Path) -> None:
    assert not _missing_zebrafinch_artifacts(NATIVE96_RUN), _missing_zebrafinch_artifacts_reason(
        NATIVE96_RUN
    )
    acceptance = _load_tool("nucleus_acceptance_migration_invariance", ACCEPTANCE_REPORT)
    migration = _load_tool("nucleus_competition_migration_invariance", MIGRATION_TOOL)
    run_dir = tmp_path / "native96_run"
    publication = run_dir / "nucleus_competition"
    manifest_path = _copy_publication_as_schema_1_2(NATIVE96_PUBLICATION, publication)
    shutil.copy2(
        NATIVE96_RUN / "nucleus_shell_contamination_tol0.json",
        run_dir / "nucleus_shell_contamination_tol0.json",
    )

    before = acceptance.realization_gate(run_dir)
    migrated = migration.migrate_manifest(manifest_path)
    after = acceptance.realization_gate(run_dir)

    assert migrated["schema_version"] == "3.0"
    assert migrated["required_capabilities"] == []
    fields = (
        "realized_unit_count",
        "unit_count",
        "realized_owner_count",
        "owner_count",
    )
    assert {field: after[field] for field in fields} == {field: before[field] for field in fields}


def test_final_aggregation_remap_replays_competition_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    param = {"bbox": [3, 5, 7, 9, 11, 13]}
    original: np.ndarray = np.full((6, 6, 6, 1), 7, dtype=np.uint64)
    winner_internal_id = (1 << 60) + 11
    owner_11 = _owner_id(11)
    owner_22 = _owner_id(22)
    new_internal_id = (1 << 60) + 23
    territory: np.ndarray = np.full((6, 6, 6), winner_internal_id, dtype=np.uint64)
    territory[3:, :, :] = new_internal_id
    np.savez_compressed(
        tmp_path / "territory.npz",
        territory=territory,
        bbox_xyz=np.asarray(param["bbox"], dtype=np.int64),
        factor=np.asarray(1, dtype=np.int64),
    )
    global_param = {
        **_write_complete_manifest(
            tmp_path,
            {
                "qualified_segment_owners": {"7": [11, 22]},
                "qualified_segment_labels": {"7": {"11": str(owner_11), "22": str(owner_22)}},
                "repairs": [
                    {
                        "parent_id": "7",
                        "bbox_xyz": param["bbox"],
                        "factor": 1,
                        "territory_file": "territory.npz",
                        "territories": [
                            {
                                "anchor_id": "11",
                                "internal_territory_id": str(winner_internal_id),
                                "emitted_id": str(owner_11),
                            },
                            {
                                "anchor_id": "22",
                                "internal_territory_id": str(new_internal_id),
                                "emitted_id": str(owner_22),
                            },
                        ],
                    }
                ],
            },
        ),
        "AFF_RESOLUTION": 0,
    }
    expected = original.copy()
    expected[:3, :, :, 0] = owner_11
    expected[3:, :, :, 0] = owner_22
    saved: dict[str, np.ndarray] = {}

    chunk_utils = ModuleType("chunk_utils")
    setattr(chunk_utils, "read_inputs", lambda path: param if path == "task.json" else global_param)
    cut_common = ModuleType("cut_chunk_common")
    setattr(cut_common, "load_data", lambda *args, **kwargs: object())
    setattr(cut_common, "cut_data", lambda *args, **kwargs: original.copy())
    setattr(
        cut_common,
        "save_raw_data",
        lambda name, value: saved.setdefault(name, value.copy()),
    )
    monkeypatch.setitem(sys.modules, "chunk_utils", chunk_utils)
    monkeypatch.setitem(sys.modules, "cut_chunk_common", cut_common)
    monkeypatch.delitem(sys.modules, "nucleus_overlay", raising=False)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("PARAM_JSON", "global.json")
    monkeypatch.setenv("WS_PATH", "file:///ws")
    monkeypatch.setattr(sys, "argv", ["cut_chunk_remap.py", "task.json"])

    runpy.run_path(str(SCRIPTS / "cut_chunk_remap.py"), run_name="__main__")

    np.testing.assert_array_equal(saved["seg.raw"], expected)


def test_emitted_mapping_is_exact_and_derived_from_declared_nucleus_identity() -> None:
    overlay = _load_script("nucleus_overlay")
    identity = overlay.identity_declaration()
    valid = [
        {
            "anchor_id": "11",
            "internal_territory_id": "101",
            "emitted_id": str(_owner_id(11)),
        },
        {
            "anchor_id": "22",
            "internal_territory_id": "102",
            "emitted_id": str(_owner_id(22)),
        },
    ]
    overlay.validate_emitted_mapping(identity, 7, {101, 102}, valid)

    with pytest.raises(ValueError, match="exact territory-id set"):
        overlay.validate_emitted_mapping(identity, 7, {101, 102, 103}, valid)
    with pytest.raises(ValueError, match="declared nucleus mint"):
        overlay.validate_emitted_mapping(
            identity,
            7,
            {101, 102},
            [
                {
                    "anchor_id": "11",
                    "internal_territory_id": "101",
                    "emitted_id": str(_owner_id(11)),
                },
                {
                    "anchor_id": "22",
                    "internal_territory_id": "102",
                    "emitted_id": str(_owner_id(11)),
                },
            ],
        )


def test_identity_rejection_names_field_expected_and_found() -> None:
    overlay = _load_script("nucleus_overlay")
    identity = overlay.identity_declaration()
    identity["mint"]["digest"] = "sha512"

    with pytest.raises(ValueError) as error:
        overlay.validate_publication_identity({"identity": identity})

    message = str(error.value)
    assert "identity.mint.digest" in message
    assert "expected 'sha256'" in message
    assert "found 'sha512'" in message


def test_missing_unanimous_seed_aborts_flood() -> None:
    pytest.importorskip("skimage")
    competition = _load_script("nucleus_competition")
    shape = (8, 8, 8)
    segmentation: np.ndarray = np.full(shape, 7, dtype=np.uint64)
    nuclei: np.ndarray = np.zeros(shape, dtype=np.uint32)
    nuclei[1:3, 1:3, 1:3] = 11
    affinity = np.full(shape + (3,), 0.9, dtype=np.float32)

    with pytest.raises(ValueError, match="22 has no unanimous pooled seed"):
        competition.flood_unit(
            _AffinityVolume(affinity),
            _ScalarVolume(segmentation),
            _ScalarVolume(nuclei),
            {"parent_id": 7, "anchor_ids": (11, 22)},
            (0, 0, 0, *shape),
            (1, 1, 1),
            (0, 0, 0),
            factor=1,
            affinity_channels=(0, 1, 2),
            slab_z=4,
        )


def test_invalid_nucleus_coordinates_abort() -> None:
    competition = _load_script("nucleus_competition")
    nucleus = _ScalarVolume(np.ones((2, 2, 2), dtype=np.uint32))

    with pytest.raises(ValueError, match="outside"):
        competition._aligned_nucleus_xyz(
            nucleus,
            (100, 100, 100, 102, 102, 102),
            (1, 1, 1),
            (0, 0, 0),
        )


def test_generated_territory_collision_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    competition = _load_script("nucleus_competition")
    monkeypatch.setattr(competition, "_stable_territory_id", lambda parent, anchor: 1 << 60)
    monkeypatch.setattr(competition, "_find_parent_box", lambda *args: (0, 0, 0, 4, 4, 4))
    stats: dict[int, dict[str, Any]] = {
        11: {
            "start_xyz": np.zeros(3, dtype=np.int64),
            "stop_xyz": np.ones(3, dtype=np.int64),
        },
        22: {
            "start_xyz": np.zeros(3, dtype=np.int64),
            "stop_xyz": np.ones(3, dtype=np.int64),
        },
    }
    settings = {
        "ratio_zyx": (1, 1, 1),
        "offset_zyx": (0, 0, 0),
        "margin_zyx": (0, 0, 0),
        "block_xyz": (4, 4, 4),
        "factor": 1,
    }

    with pytest.raises(RuntimeError, match="territory id collision"):
        competition._prepare_units(
            [
                {
                    "parent_id": 7,
                    "anchor_ids": (11, 22),
                    "min_gap_um": 0.0,
                    "max_gap_um": 0.0,
                }
            ],
            stats,
            {7: {11: 0.5, 22: 0.5}},
            object(),
            settings,
            (0, 0, 0, 4, 4, 4),
        )


def test_overlapping_scopes_for_one_parent_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    competition = _load_script("nucleus_competition")
    monkeypatch.setattr(competition, "_find_parent_box", lambda *args: (0, 0, 0, 4, 4, 4))
    stats: dict[int, dict[str, Any]] = {
        anchor: {
            "start_xyz": np.zeros(3, dtype=np.int64),
            "stop_xyz": np.ones(3, dtype=np.int64),
        }
        for anchor in (11, 22, 33, 44)
    }
    units = [
        {
            "parent_id": 7,
            "anchor_ids": anchors,
            "min_gap_um": 0.0,
            "max_gap_um": 0.0,
        }
        for anchors in ((11, 22), (33, 44))
    ]
    settings = {
        "ratio_zyx": (1, 1, 1),
        "offset_zyx": (0, 0, 0),
        "margin_zyx": (0, 0, 0),
        "block_xyz": (4, 4, 4),
        "factor": 1,
    }

    with pytest.raises(RuntimeError, match="scopes overlap"):
        competition._prepare_units(
            units,
            stats,
            {7: {anchor: 0.25 for anchor in stats}},
            object(),
            settings,
            (0, 0, 0, 4, 4, 4),
        )


def test_incompatible_watershed_manifest_aborts_before_overlay(tmp_path: Path) -> None:
    overlay = _load_script("nucleus_overlay")
    params = _write_complete_manifest(
        tmp_path,
        {
            "protected_nucleus_owners": [],
            "qualified_segment_owners": {},
            "qualified_segment_labels": {},
            "repairs": [],
        },
    )
    Path(params["WS_MANIFEST"]).write_text("{}")

    with pytest.raises(ValueError, match="another watershed identity") as error:
        overlay.apply_nucleus_competition(
            np.zeros((2, 2, 2, 1), dtype=np.uint64),
            (0, 0, 0),
            params,
        )
    assert "--watershed-manifest" in str(error.value)


def test_scan_flood_merge_is_plan_bound_and_preserves_last_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("skimage")
    competition = _load_script("nucleus_competition")
    shape = (24, 16, 12)
    segmentation: np.ndarray = np.zeros(shape, dtype=np.uint64)
    segmentation[2:22, 2:14, 2:10] = 7
    nuclei: np.ndarray = np.zeros(shape, dtype=np.uint32)
    nuclei[5:9, 5:9, 4:8] = 11
    nuclei[14:18, 7:11, 4:8] = 22
    affinities = np.full(shape + (3,), 0.9, dtype=np.float32)
    affinities[11:13, :, :, :] = 0.1
    affinity_volume = _AffinityVolume(affinities)
    watershed_volume = _ScalarVolume(segmentation)
    nucleus_volume = _ScalarVolume(nuclei)
    manifest_path = tmp_path / "competition" / "manifest.json"
    params = {
        "AFF_PATH": str(tmp_path / "affinity"),
        "WS_PATH": str(tmp_path / "watershed"),
        "NUC_PATH": str(tmp_path / "nuclei"),
        "NUC_COMPETITION_MANIFEST": str(manifest_path),
        "BBOX": [0, 0, 0, *shape],
        "NUC_RATIO": [1, 1, 1],
        "NUC_OFFSET": [0, 0, 0],
        "NUC_VOXEL_SIZE_ZYX_NM": [1000, 1000, 1000],
        "NUC_COMPETITION_MARGIN_ZYX": [2, 2, 2],
        "NUC_COMPETITION_FACTOR": 2,
        "NUC_COMPETITION_BLOCK_ZYX": [4, 8, 8],
        "NUC_COMPETITION_SLAB_Z": 4,
        "NUC_MAX_UNITS": 4,
    }
    param_path = tmp_path / "param.json"
    param_path.write_text(json.dumps(params, sort_keys=True) + "\n")
    param_sha = _sha256(param_path)
    fingerprints = {
        "param": {"sha256": param_sha},
        "nucleus": {"sha256": "nucleus"},
        "watershed": {"manifest_sha256": "watershed"},
        "affinity": {"index_sha256": "affinity", "chunk_count": 1},
        "code": {"abiss_build_id": "test", "python_sources_sha256": "test"},
    }
    monkeypatch.setattr(competition, "input_fingerprints", lambda *args: fingerprints)
    monkeypatch.setattr(
        competition,
        "_open_inputs",
        lambda _params, *, affinity: (
            affinity_volume if affinity else None,
            watershed_volume,
            nucleus_volume,
        ),
    )

    plan, plan_digest = competition.scan_stage(params, param_path, "test-run")
    assert len(plan["units"]) == 1
    assert competition.flood_stage(params, param_path, "test-run", 3) is None
    record = competition.flood_stage(params, param_path, "test-run", 0)
    assert record["plan_digest"] == plan_digest
    with np.load(
        manifest_path.parent / ".nuccomp-runs" / "test-run" / record["territory_file"],
        allow_pickle=False,
    ) as archive:
        assert archive["territory"].dtype == np.uint64
        assert {int(v) for v in np.unique(archive["territory"]) if v} == {
            int(item["internal_territory_id"]) for item in plan["units"][0]["territories"]
        }

    manifest = competition.merge_stage(params, param_path, "test-run")
    assert manifest["schema_version"] == "3.0"
    assert manifest["required_capabilities"] == []
    assert manifest["completion"] == {"state": "complete", "plan_digest": plan_digest}
    mapping = manifest["repairs"][0]["territories"]
    assert {int(item["emitted_id"]) for item in mapping} == {_owner_id(11), _owner_id(22)}
    assert all(int(item["emitted_id"]) != 7 for item in mapping)
    assert manifest["identity"] == competition.identity_declaration()
    assert manifest["reason"] == "competitive_repairs_completed"
    comparison = _load_comparison_harness()
    semantic_units = comparison.semantic_units(manifest_path)
    assert len(semantic_units) == 1
    assert semantic_units[0]["parent_id"] == 7
    assert {item["anchor_id"] for item in semantic_units[0]["anchors"]} == {11, 22}
    published = manifest_path.read_bytes()

    record_path = manifest_path.parent / ".nuccomp-runs" / "test-run" / "unit_00000.json"
    stale = json.loads(record_path.read_text())
    stale["plan_digest"] = "0" * 64
    record_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="another plan"):
        competition.merge_stage(params, param_path, "test-run")
    assert manifest_path.read_bytes() == published


def test_scan_fails_closed_above_nuc_max_units(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    competition = _load_script("nucleus_competition")
    params = {
        "AFF_PATH": str(tmp_path / "affinity"),
        "WS_PATH": str(tmp_path / "watershed"),
        "NUC_PATH": str(tmp_path / "nuclei"),
        "NUC_COMPETITION_MANIFEST": str(tmp_path / "competition" / "manifest.json"),
        "BBOX": [0, 0, 0, 4, 4, 4],
        "NUC_VOXEL_SIZE_ZYX_NM": [1000, 1000, 1000],
        "NUC_MAX_UNITS": 2,
    }
    param_path = tmp_path / "param.json"
    param_path.write_text(json.dumps(params))
    volume = _ScalarVolume(np.ones((4, 4, 4), dtype=np.uint64))
    units = [
        {
            "parent_id": index + 1,
            "anchor_ids": (2 * index + 1, 2 * index + 2),
            "min_gap_um": 0.0,
            "max_gap_um": 0.0,
        }
        for index in range(3)
    ]
    monkeypatch.setattr(competition, "input_fingerprints", lambda *args: {})
    monkeypatch.setattr(
        competition,
        "_open_inputs",
        lambda _params, *, affinity: (None, volume, volume),
    )
    monkeypatch.setattr(competition, "scan_nucleus_geometry", lambda *args, **kwargs: {})
    monkeypatch.setattr(competition, "nucleus_segment_histograms", lambda *args: {})
    monkeypatch.setattr(competition, "qualifying_targets", lambda *args: ({}, {}))
    monkeypatch.setattr(competition, "contact_units", lambda *args: (units, []))

    with pytest.raises(RuntimeError, match=r"3 units, exceeding NUC_MAX_UNITS=2"):
        competition.scan_stage(params, param_path, "capacity-test")

    assert competition.DEFAULT_MAX_UNITS == 64
    assert "array capacity 3/2 units observed/configured" in capsys.readouterr().out
    run_dir = tmp_path / "competition" / ".nuccomp-runs" / "capacity-test"
    assert not (run_dir / "units.json").exists()


def test_chunkmap_input_override_is_not_replaced_by_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_h5 = tmp_path / "affinity.h5"
    with h5py.File(source_h5, "w") as handle:
        handle.create_dataset("main", data=np.zeros((3, 2, 2, 2), dtype=np.uint8))
    input_chunkmap = (tmp_path / "reused" / "chunkmap").as_uri()
    output_chunkmap = (tmp_path / "fresh" / "chunkmap").as_uri()
    monkeypatch.setattr(
        abiss_chunk,
        "_load_yaml",
        lambda _path: {
            "abiss_chunk": {
                "abiss_home": str(tmp_path / "abiss"),
                "workdir": str(tmp_path / "work"),
                "secrets_dir": str(tmp_path / "secrets"),
                "source_affinity_h5": str(source_h5),
                "source_dataset": "main",
                "param": {
                    "BBOX": [0, 0, 0, 2, 2, 2],
                    "CHUNK_SIZE": [2, 2, 2],
                    "CHUNKMAP_INPUT": input_chunkmap,
                    "CHUNKMAP_OUTPUT": output_chunkmap,
                },
            }
        },
    )

    prepared = abiss_chunk.prepare_config(tmp_path / "config.yaml")

    assert prepared.param_payload["CHUNKMAP_INPUT"] == input_chunkmap
    assert prepared.param_payload["CHUNKMAP_OUTPUT"] == output_chunkmap
