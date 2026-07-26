"""Tests for resolving Seuron provenance into a fail-closed ABISS payload."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from connectomics.runtime.seuron_provenance import (
    CLASSIFIED_KEYS,
    GENERATED_OUTPUT,
    IGNORED_INFRA,
    OPTIONAL_INPUT,
    REQUIRED_INPUT,
    classify_and_map,
    load_provenance,
)

FIXTURE = Path(__file__).parent / "fixtures" / "seuron_provenance.json"
TUTORIAL = Path(__file__).parents[2] / "tutorials" / "seuron_provenance_replay.yaml"


def _fixture_payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write_provenance(tmp_path: Path, processing) -> Path:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"processing": processing}), encoding="utf-8")
    return path


def test_real_provenance_maps_only_whitelisted_keys(tmp_path):
    spec = load_provenance(FIXTURE)
    mapped = classify_and_map(spec.seg_block, name="replay", out_root=tmp_path)

    assert set(spec.seg_block) == CLASSIFIED_KEYS
    assert set(mapped.param) == REQUIRED_INPUT | OPTIONAL_INPUT | GENERATED_OUTPUT
    for key in REQUIRED_INPUT | OPTIONAL_INPUT:
        assert mapped.param[key] == spec.seg_block[key]
    assert not (set(mapped.param) & IGNORED_INFRA)


def test_unknown_segmentation_key_raises(tmp_path):
    seg_block = deepcopy(load_provenance(FIXTURE).seg_block)
    seg_block["NEW_SEURON_FIELD"] = "schema drift"

    with pytest.raises(ValueError, match="Unknown.*NEW_SEURON_FIELD"):
        classify_and_map(seg_block, name="replay", out_root=tmp_path)


def test_outputs_are_fresh_and_minted_under_run_root(tmp_path):
    seg_block = load_provenance(FIXTURE).seg_block
    mapped = classify_and_map(seg_block, name="bounded-replay", out_root=tmp_path)
    run_uri = (tmp_path.resolve() / "bounded-replay").as_uri()

    for key in GENERATED_OUTPUT - {"NAME"}:
        assert mapped.param[key].startswith(run_uri + "/")
        assert mapped.param[key] != seg_block[key]
    assert mapped.param["NAME"] == "bounded-replay"
    assert mapped.param["NAME"] != seg_block["NAME"]
    assert mapped.param["WS_PATH"].endswith("/precomputed/ws/bounded-replay")
    assert mapped.param["SEG_PATH"].endswith("/precomputed/seg/bounded-replay")
    assert mapped.param["CHUNKMAP_OUTPUT"].endswith("/scratch/bounded-replay/ws/chunkmap")


def test_load_provenance_rejects_zero_segmentation_blocks(tmp_path):
    processing = _fixture_payload()["processing"][1:]
    path = _write_provenance(tmp_path, processing)

    with pytest.raises(ValueError, match="exactly one.*found 0"):
        load_provenance(path)


def test_load_provenance_rejects_two_segmentation_blocks(tmp_path):
    processing = _fixture_payload()["processing"]
    processing.append(deepcopy(processing[0]))
    path = _write_provenance(tmp_path, processing)

    with pytest.raises(ValueError, match="exactly one.*found 2"):
        load_provenance(path)


def test_missing_required_input_raises(tmp_path):
    seg_block = deepcopy(load_provenance(FIXTURE).seg_block)
    del seg_block["CHUNK_SIZE"]

    with pytest.raises(ValueError, match="Missing required.*CHUNK_SIZE"):
        classify_and_map(seg_block, name="replay", out_root=tmp_path)


def test_image_path_is_optional_input_and_never_remapped(tmp_path):
    seg_block = load_provenance(FIXTURE).seg_block
    mapped = classify_and_map(seg_block, name="replay", out_root=tmp_path)

    assert mapped.param["IMAGE_PATH"] == seg_block["IMAGE_PATH"]

    without_image = dict(seg_block)
    del without_image["IMAGE_PATH"]
    mapped_without_image = classify_and_map(
        without_image, name="replay-no-image", out_root=tmp_path
    )
    assert "IMAGE_PATH" not in mapped_without_image.param


def test_affinity_override_takes_precedence(tmp_path):
    seg_block = load_provenance(FIXTURE).seg_block
    mapped = classify_and_map(
        seg_block,
        name="replay",
        out_root=tmp_path,
        aff_override="file:///mirror/affinity",
    )

    assert mapped.param["AFF_PATH"] == "file:///mirror/affinity"
    assert seg_block["AFF_PATH"].startswith("gs://")


def test_cli_overrides_yaml_values_without_writing(tmp_path):
    from scripts import run_seuron_provenance as cli

    out_root = tmp_path / "cli-output"
    args = cli.parse_args(
        [
            "--config",
            str(TUTORIAL),
            "--out-root",
            str(out_root),
            "--name",
            "cli-name",
            "--secrets-dir",
            str(tmp_path / "cli-secrets"),
            "--abiss-home",
            str(tmp_path / "cli-abiss"),
            "--exec-bbox",
            "0",
            "0",
            "0",
            "1024",
            "1024",
            "512",
            "--score-bbox",
            "256",
            "256",
            "128",
            "768",
            "768",
            "384",
            "--mirror",
            str(tmp_path / "lower-precedence-mirror"),
            "--aff-path",
            "file:///cli-affinity",
            "--mode",
            "resume",
        ]
    )

    resolved = cli.resolve_replay(args)

    assert resolved.out_root == out_root.resolve()
    assert resolved.name == "cli-name"
    assert resolved.secrets_dir == (tmp_path / "cli-secrets").resolve()
    assert resolved.abiss_home == (tmp_path / "cli-abiss").resolve()
    assert resolved.execution_bbox == (0, 0, 0, 1024, 1024, 512)
    assert resolved.score_bbox == (256, 256, 128, 768, 768, 384)
    assert resolved.param["AFF_PATH"] == "file:///cli-affinity"
    assert resolved.mode == "resume"
    assert not out_root.exists()


def test_resolve_cli_does_not_import_cloudvolume_or_create_state(tmp_path):
    home = tmp_path / "home"
    cloudvolume_root = home / ".cloudvolume"
    out_root = tmp_path / "outputs"
    import_guard = tmp_path / "import-guard"
    import_guard.mkdir()
    (import_guard / "cloudvolume.py").write_text(
        "raise AssertionError('cloudvolume imported during pure resolve')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"CLOUD_VOLUME_DIR": str(cloudvolume_root), "HOME": str(home)})
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(import_guard), env.get("PYTHONPATH")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TUTORIAL.parents[1] / "scripts" / "run_seuron_provenance.py"),
            "--config",
            str(TUTORIAL),
            "--out-root",
            str(out_root),
            "--secrets-dir",
            str(cloudvolume_root / "secrets"),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["execute"] is False
    assert not cloudvolume_root.exists()
    assert not out_root.exists()


def test_output_modes_enforce_fresh_resume_and_explicit_overwrite(tmp_path):
    from scripts import run_seuron_provenance as cli

    base = cli.resolve_replay(
        cli.parse_args(
            [
                "--config",
                str(TUTORIAL),
                "--out-root",
                str(tmp_path),
                "--name",
                "fresh-run",
            ]
        )
    )
    expected = {
        "provenance_sha": "provenance",
        "whitelisted_param_sha": "param",
        "abiss_build_id": "build",
        "execution_bbox": list(base.execution_bbox),
    }

    base.run_root.mkdir()
    (base.run_root / "artifact").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="empty output namespace"):
        cli._apply_output_mode(base, expected)

    resume = replace(base, name="resume-run", mode="resume")
    resume.run_root.mkdir()
    resume.manifest_path.write_text(json.dumps(expected), encoding="utf-8")
    cli._apply_output_mode(resume, expected)

    overwrite = replace(base, name="overwrite-run", mode="overwrite")
    overwrite.run_root.mkdir()
    (overwrite.run_root / "artifact").write_text("delete me", encoding="utf-8")
    cli._apply_output_mode(overwrite, expected)
    assert not overwrite.run_root.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provenance_sha", "different-provenance"),
        ("whitelisted_param_sha", "different-param"),
        ("abiss_build_id", "different-build"),
        ("execution_bbox", [0, 0, 0, 1, 1, 1]),
    ],
)
def test_resume_rejects_each_manifest_compatibility_field(tmp_path, field, replacement):
    from scripts import run_seuron_provenance as cli

    resolved = cli.resolve_replay(
        cli.parse_args(
            [
                "--config",
                str(TUTORIAL),
                "--out-root",
                str(tmp_path),
                "--name",
                "resume-run",
                "--mode",
                "resume",
            ]
        )
    )
    expected = {
        "provenance_sha": "provenance",
        "whitelisted_param_sha": "param",
        "abiss_build_id": "build",
        "execution_bbox": list(resolved.execution_bbox),
    }
    incompatible_manifest = deepcopy(expected)
    incompatible_manifest[field] = replacement
    resolved.run_root.mkdir()
    resolved.manifest_path.write_text(json.dumps(incompatible_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        cli._apply_output_mode(resolved, expected)


def test_affinity_preflight_uses_physical_resolution_and_full_bounds(monkeypatch, tmp_path):
    pytest.importorskip("cloudvolume")  # optional dep; _preflight_affinity opens a CloudVolume
    from scripts import run_seuron_provenance as cli

    affinity = tmp_path / "affinity"
    affinity.mkdir()
    resolved = cli.resolve_replay(
        cli.parse_args(
            [
                "--config",
                str(TUTORIAL),
                "--out-root",
                str(tmp_path / "output"),
                "--name",
                "preflight-run",
                "--aff-path",
                str(affinity),
            ]
        )
    )
    opens = []
    reads = []

    class FakeVolume:
        mip = 0
        resolution = (9, 9, 20)
        chunk_size = (64, 64, 64)
        bounds = SimpleNamespace(minpt=(0, 0, 0), maxpt=(10664, 10912, 5700))

        def __init__(self, path, **kwargs):
            opens.append((path, kwargs))

        def __getitem__(self, key):
            reads.append(key)
            return object()

    monkeypatch.setattr("cloudvolume.CloudVolume", FakeVolume)

    metadata = cli._preflight_affinity(resolved)

    assert opens[0][1]["mip"] == [9, 9, 20]
    assert metadata.resolution_xyz == (9, 9, 20)
    assert metadata.chunk_size_xyz == (64, 64, 64)
    assert len(reads) == 2

    FakeVolume.bounds = SimpleNamespace(minpt=(0, 0, 0), maxpt=(4000, 4000, 2500))
    with pytest.raises(ValueError, match="outside affinity bounds") as exc_info:
        cli._preflight_affinity(resolved)
    assert "credentials" not in str(exc_info.value)


def test_replay_execute_writes_copy_uri_transfer_commands(monkeypatch, tmp_path):
    from connectomics.runtime import abiss_large
    from scripts import run_seuron_provenance as cli

    resolved = cli.resolve_replay(
        cli.parse_args(
            [
                "--config",
                str(TUTORIAL),
                "--out-root",
                str(tmp_path / "output"),
                "--name",
                "execute-run",
                "--secrets-dir",
                str(tmp_path / "credentials"),
            ]
        )
    )
    prepared = replace(cli.prepare_execution(resolved), stages=("watershed",))
    calls = []

    def fake_run(argv, *, cwd, env, check):
        calls.append((argv, cwd, env, check))

    monkeypatch.setattr(abiss_large.subprocess, "run", fake_run)
    monkeypatch.setattr(abiss_large, "_discover_segmentation", lambda path: object())

    result = abiss_large.run_abiss_large(prepared, execute=True)

    copy_helper = (TUTORIAL.parents[1] / "scripts" / "copy_uri.py").resolve()
    copy_command = f"{sys.executable} {copy_helper}"
    assert "UPLOAD_CMD" not in resolved.param
    assert "DOWNLOAD_CMD" not in resolved.param
    assert "UPLOAD_CMD" not in prepared.param_payload
    assert "DOWNLOAD_CMD" not in prepared.param_payload
    assert result.prepared.param_payload["UPLOAD_CMD"] == copy_command
    assert result.prepared.param_payload["DOWNLOAD_CMD"] == copy_command
    assert len(calls) == 1

    param = json.loads(prepared.param_path.read_text(encoding="utf-8"))
    runtime_param = json.loads((prepared.runtime_secrets_dir / "param").read_text(encoding="utf-8"))
    assert param["UPLOAD_CMD"] == copy_command
    assert param["DOWNLOAD_CMD"] == copy_command
    assert runtime_param == param


def test_arbitrary_secrets_dir_gets_canonical_cloudvolume_view(monkeypatch, tmp_path):
    pytest.importorskip("cloudvolume")  # optional dep; the canonical view opens a CloudVolume
    from scripts import run_seuron_provenance as cli

    affinity = tmp_path / "affinity"
    affinity.mkdir()
    secrets_dir = tmp_path / "gcp-credentials"
    secrets_dir.mkdir()
    credential = secrets_dir / "google-secret.json"
    credential.write_text("{}\n", encoding="utf-8")
    resolved = cli.resolve_replay(
        cli.parse_args(
            [
                "--config",
                str(TUTORIAL),
                "--out-root",
                str(tmp_path / "output"),
                "--name",
                "custom-secrets",
                "--aff-path",
                str(affinity),
                "--secrets-dir",
                str(secrets_dir),
            ]
        )
    )
    observed = []

    class FakeVolume:
        mip = 0
        resolution = (9, 9, 20)
        chunk_size = (64, 64, 64)
        bounds = SimpleNamespace(minpt=(0, 0, 0), maxpt=(10664, 10912, 5700))

        def __init__(self, path, **kwargs):
            cloudvolume_root = Path(os.environ["CLOUD_VOLUME_DIR"])
            canonical = cloudvolume_root / "secrets" / "google-secret.json"
            observed.append((canonical.resolve(), canonical.read_text(encoding="utf-8")))

        def __getitem__(self, key):
            return object()

    monkeypatch.setattr("cloudvolume.CloudVolume", FakeVolume)

    cli._preflight_affinity(resolved)
    prepared = cli.prepare_execution(resolved)

    assert observed == [(credential.resolve(), "{}\n")]
    assert prepared.runtime_secrets_dir == resolved.run_root / ".cloudvolume" / "secrets"
    assert prepared.extra_env == {"CLOUD_VOLUME_DIR": str(resolved.run_root / ".cloudvolume")}


def test_igneous_tasks_preserve_provenance_order():
    spec = load_provenance(FIXTURE)

    assert [block["task"] for block in spec.igneous_blocks] == [
        "DownsampleTask",
        "MeshTask",
        "MultiResShardedMeshMergeTask",
        "DownsampleTask",
    ]


def test_malformed_processing_method_raises(tmp_path):
    path = _write_provenance(tmp_path, [{"method": None}])

    with pytest.raises(ValueError, match=r"processing\[0\]\.method"):
        load_provenance(path)


def test_to_dict_returns_detached_payload(tmp_path):
    mapped = classify_and_map(load_provenance(FIXTURE).seg_block, name="replay", out_root=tmp_path)

    payload = mapped.to_dict()
    payload["BBOX"][0] = 123
    assert mapped.param["BBOX"][0] == 0
