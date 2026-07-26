"""Tests for the runtime-owned ABISS-large executor."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

from connectomics.runtime import abiss_large


def _prepared(
    tmp_path: Path,
    *,
    stages: tuple[str, ...] = ("watershed", "remap_watershed"),
) -> abiss_large.PreparedConfig:
    return abiss_large.PreparedConfig(
        workdir=tmp_path / "run" / "work",
        secrets_dir=tmp_path / "run" / "secrets",
        param_path=tmp_path / "run" / "secrets" / "param",
        abiss_home=tmp_path / "lib" / "abiss",
        param_payload={
            "NAME": "fixture",
            "AFF_PATH": "file:///input/affinity",
            "SEG_PATH": "file:///output/segmentation",
            "BBOX": [0, 0, 0, 1024, 1024, 512],
            "CHUNK_SIZE": [512, 512, 256],
            "AGG_THRESHOLD": "0.3",
            "UPLOAD_CMD": "fixture-upload",
            "DOWNLOAD_CMD": "fixture-download",
        },
        root_tag="1_0_0_0",
        top_mip=1,
        stage_overlap="0",
        stage_meta="",
        stages=stages,
    )


def test_execute_false_returns_plan_without_writes_or_result_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry resolution performed I/O")

    monkeypatch.setattr(abiss_large, "_write_param", unexpected)
    monkeypatch.setattr(abiss_large, "_execute_stage", unexpected)
    monkeypatch.setattr(abiss_large, "_discover_segmentation", unexpected)

    result = abiss_large.run_abiss_large(prepared)

    assert result.executed is False
    assert result.segmentation is None
    assert result.seg_cloudpath == "file:///output/segmentation"
    assert [plan.stage for plan in result.stage_plans] == [
        "watershed",
        "remap_watershed",
    ]
    assert not (tmp_path / "run").exists()


def test_execute_writes_param_and_runs_stages_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path)
    calls: list[tuple[list[str], str, dict[str, str], bool]] = []
    discovered: list[str] = []
    output_marker = object()

    def fake_run(argv: list[str], *, cwd: str, env: dict[str, str], check: bool) -> None:
        calls.append((argv, cwd, env, check))

    def fake_discover(path: str) -> object:
        discovered.append(path)
        return output_marker

    monkeypatch.setattr(abiss_large.subprocess, "run", fake_run)
    monkeypatch.setattr(abiss_large, "_discover_segmentation", fake_discover)

    result = abiss_large.run_abiss_large(prepared, execute=True)

    with prepared.param_path.open("r", encoding="utf-8") as f:
        assert json.load(f) == dict(prepared.param_payload)
    assert prepared.param_path.read_text(encoding="utf-8").endswith("\n")

    scripts_dir = prepared.abiss_home / "scripts"
    assert [call[0] for call in calls] == [
        [
            "bash",
            str(scripts_dir / "run_batch.sh"),
            "ws",
            "1",
            "1_0_0_0",
        ],
        [
            "bash",
            str(scripts_dir / "remap_batch.sh"),
            "ws",
            "1",
            "1_0_0_0",
        ],
    ]
    assert all(call[1] == str(prepared.workdir) for call in calls)
    assert all(call[3] is True for call in calls)
    assert [call[2]["STAGE"] for call in calls] == ["ws", "ws"]
    for _, _, env, _ in calls:
        assert env["WORKER_HOME"] == str(prepared.abiss_home)
        assert env["SECRETS"] == str(prepared.secrets_dir)
        assert env["AIRFLOW_TMP_DIR"] == str(prepared.workdir / ".airflow")

    assert discovered == ["file:///output/segmentation"]
    assert result.executed is True
    assert result.segmentation is output_marker


def test_default_stage_order_preserves_four_stage_pipeline(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, stages=abiss_large.STAGES_ALL)

    result = abiss_large.run_abiss_large(prepared)

    assert [Path(plan.argv[1]).name for plan in result.stage_plans] == [
        "run_batch.sh",
        "remap_batch.sh",
        "run_batch.sh",
        "remap_batch.sh",
    ]
    assert [plan.argv[2] for plan in result.stage_plans] == ["ws", "ws", "me", "agg"]
    assert [plan.env["STAGE"] for plan in result.stage_plans] == ["ws", "ws", "agg", "agg"]


def test_nonzero_stage_exit_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    calls = 0

    def fake_run(argv: list[str], *, cwd: str, env: dict[str, str], check: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(17, argv)

    def unexpected_discovery(path: str) -> None:
        raise AssertionError(f"discovered result after failed stage: {path}")

    monkeypatch.setattr(abiss_large.subprocess, "run", fake_run)
    monkeypatch.setattr(abiss_large, "_discover_segmentation", unexpected_discovery)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        abiss_large.run_abiss_large(prepared, execute=True)

    assert exc_info.value.returncode == 17
    assert calls == 2


def test_result_discovery_opens_segmentation_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, stages=())
    opened: list[tuple[str, bool]] = []
    marker = object()

    def fake_cloud_volume(path: str, *, progress: bool) -> object:
        opened.append((path, progress))
        return marker

    monkeypatch.setattr(abiss_large, "_open_cloudvolume", fake_cloud_volume)

    result = abiss_large.run_abiss_large(prepared, execute=True)

    assert opened == [("file:///output/segmentation", False)]
    assert result.segmentation is marker


def test_runtime_secrets_stages_param_without_touching_credential_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = replace(
        _prepared(tmp_path, stages=("watershed",)),
        secrets_dir=tmp_path / "credentials" / "secrets",
        param_path=tmp_path / "run" / "param",
        runtime_secrets_dir=tmp_path / "run" / ".abiss_secrets",
    )
    prepared.secrets_dir.mkdir(parents=True)
    credential = prepared.secrets_dir / "s3-secret"
    credential.write_text("credential source\n", encoding="utf-8")
    (prepared.secrets_dir / "config.sh").write_text("stale config\n", encoding="utf-8")
    (prepared.secrets_dir / "param").write_text("historical param\n", encoding="utf-8")
    monkeypatch.setattr(abiss_large.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(abiss_large, "_discover_segmentation", lambda path: object())

    result = abiss_large.run_abiss_large(prepared, execute=True)

    assert json.loads(prepared.param_path.read_text()) == dict(prepared.param_payload)
    runtime_param = prepared.runtime_secrets_dir / "param"
    assert json.loads(runtime_param.read_text()) == dict(prepared.param_payload)
    runtime_credential = prepared.runtime_secrets_dir / "s3-secret"
    assert runtime_credential.is_symlink()
    assert runtime_credential.resolve() == credential.resolve()
    assert runtime_credential.read_text(encoding="utf-8") == "credential source\n"
    assert not (prepared.runtime_secrets_dir / "config.sh").exists()
    assert (prepared.secrets_dir / "param").read_text(encoding="utf-8") == "historical param\n"
    assert result.stage_plans[0].env["SECRETS"] == str(prepared.runtime_secrets_dir)


def test_custom_param_path_does_not_clobber_shared_secrets_param(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = replace(
        _prepared(tmp_path, stages=()),
        secrets_dir=tmp_path / "shared" / "secrets",
        param_path=tmp_path / "run" / "param",
    )
    prepared.secrets_dir.mkdir(parents=True)
    shared_param = prepared.secrets_dir / "param"
    shared_param.write_text("shared historical state\n", encoding="utf-8")
    monkeypatch.setattr(abiss_large, "_discover_segmentation", lambda path: object())

    abiss_large.run_abiss_large(prepared, execute=True)

    assert json.loads(prepared.param_path.read_text()) == dict(prepared.param_payload)
    assert shared_param.read_text(encoding="utf-8") == "shared historical state\n"


def test_write_param_false_preserves_prepared_param(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = replace(_prepared(tmp_path, stages=()), write_param=False)
    prepared.param_path.parent.mkdir(parents=True)
    prepared.param_path.write_text("already prepared\n", encoding="utf-8")
    monkeypatch.setattr(abiss_large, "_discover_segmentation", lambda path: object())

    abiss_large.run_abiss_large(prepared, execute=True)

    assert prepared.param_path.read_text(encoding="utf-8") == "already prepared\n"


def test_replay_output_initialization_uses_validated_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = dict(_prepared(tmp_path).param_payload)
    payload.update(
        {
            "WS_PATH": (tmp_path / "run" / "ws").as_uri(),
            "SCRATCH_PATH": (tmp_path / "run" / "scratch").as_uri(),
            "CHUNKMAP_OUTPUT": (tmp_path / "run" / "scratch" / "ws" / "chunkmap").as_uri(),
        }
    )
    prepared = replace(
        _prepared(tmp_path, stages=()),
        param_payload=payload,
        output_layer_spec=abiss_large.OutputLayerSpec(
            resolution_xyz=(9, 9, 20),
            chunk_size_xyz=(64, 64, 64),
        ),
        write_param=False,
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        abiss_large,
        "_prepare_segmentation_output_layers",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(abiss_large, "_discover_segmentation", lambda path: object())

    abiss_large.run_abiss_large(prepared, execute=True)

    assert calls == [
        {
            "ws_cloudpath": payload["WS_PATH"],
            "seg_cloudpath": payload["SEG_PATH"],
            "bbox_xyz": payload["BBOX"],
            "resolution_xyz": (9, 9, 20),
            "chunk_size_xyz": (64, 64, 64),
        }
    ]
    assert (tmp_path / "run" / "scratch" / "ws" / "chunkmap").is_dir()


def test_thin_cli_delegates_to_api_and_yaml_builds_same_execution_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scripts.run_abiss_large as cli

    source_h5 = tmp_path / "affinity.h5"
    with h5py.File(source_h5, "w") as f:
        f.create_dataset("main", data=np.zeros((3, 4, 6, 8), dtype=np.uint8))

    config_path = tmp_path / "abiss.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "abiss_large": {
                    "abiss_home": str(tmp_path / "abiss"),
                    "workdir": str(tmp_path / "outputs" / "work"),
                    "secrets_dir": str(tmp_path / "outputs" / "secrets"),
                    "source_affinity_h5": str(source_h5),
                    "source_dataset": "main",
                    "root_tag": "0_0_0_0",
                    "top_mip": 0,
                    "param": {
                        "NAME": "equivalence",
                        "BBOX": [0, 0, 0, 8, 6, 4],
                        "CHUNK_SIZE": [8, 6, 4],
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workflow = abiss_large.prepare_config(config_path)
    prepared = workflow.execution_config(stages=("watershed",), write_param=False)
    prepared_calls: list[abiss_large.PreparedConfig] = []

    def fake_prepare_config(path: Path) -> abiss_large.LargeWorkflowConfig:
        assert path == config_path.resolve()
        return workflow

    def fake_run(value: abiss_large.PreparedConfig, *, execute: bool) -> abiss_large.RunResult:
        assert execute is True
        prepared_calls.append(value)
        return abiss_large.RunResult(
            prepared=value,
            stage_plans=(),
            executed=True,
            segmentation=object(),
        )

    monkeypatch.setattr(abiss_large, "prepare_config", fake_prepare_config)
    monkeypatch.setattr(abiss_large, "run_abiss_large", fake_run)

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--skip-prepare",
            "--stages",
            "watershed",
        ]
    )

    assert cli.main is abiss_large.main
    assert exit_code == 0
    assert prepared_calls == [prepared]
    assert prepared.workdir == workflow.workdir
    assert prepared.secrets_dir == workflow.secrets_dir
    assert prepared.param_path == workflow.param_path
    assert prepared.abiss_home == workflow.abiss_home
    assert prepared.param_payload == workflow.param_payload
    assert prepared.seg_cloudpath == workflow.seg_cloudpath
    assert prepared.write_param is False
