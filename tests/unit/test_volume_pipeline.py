"""Contracts for shared local/Slurm execution and dataset-owned fitted values."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from connectomics.runtime import volume_pipeline as pipeline


def make_step(name="mask", *, done=False, **kwargs):
    return pipeline.Step(
        name, name, f"python {name}.py", lambda: pipeline.Status(done, "artifact status"), **kwargs
    )


@pytest.fixture
def subprocess_run(monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="123;cluster\n"))
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    return run


@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_pending_launches_and_complete_does_not(launcher, subprocess_run, capsys):
    steps = [make_step("pending"), make_step("complete", done=True)]
    assert pipeline.execute_steps(steps, launcher=launcher) == 0
    subprocess_run.assert_called_once()
    assert "python pending.py" in str(subprocess_run.call_args)
    assert "python complete.py" not in str(subprocess_run.call_args)
    assert "skipping (use --force to rerun)" in capsys.readouterr().out


def test_disabled_step_reports_reason_even_when_forced(subprocess_run, capsys):
    step = make_step(skip="mask.enabled is false")
    assert pipeline.execute_steps([step], forced=["mask"]) == 0
    subprocess_run.assert_not_called()
    assert "skipping (mask.enabled is false)" in capsys.readouterr().out


def test_forced_complete_step_runs(subprocess_run):
    assert pipeline.execute_steps([make_step(done=True)], forced=["mask"]) == 0
    subprocess_run.assert_called_once()


def test_slurm_arrays_and_afterok_chain(subprocess_run):
    subprocess_run.side_effect = [
        SimpleNamespace(stdout="101;cluster\n"),
        SimpleNamespace(stdout="102\n"),
    ]
    steps = [make_step("infer", array=3, resources="-p gpu -c 8"), make_step("abiss")]
    assert pipeline.execute_steps(steps, launcher="slurm", job_prefix="third") == 0
    first = subprocess_run.call_args_list[0].args[0]
    second = subprocess_run.call_args_list[1].args[0]
    assert first == [
        "sbatch",
        "--parsable",
        "--job-name=third-infer",
        "--array=0-2",
        "-p",
        "gpu",
        "-c",
        "8",
        "--wrap=python infer.py --shard-id $SLURM_ARRAY_TASK_ID --num-shards 3",
    ]
    assert second == [
        "sbatch",
        "--parsable",
        "--job-name=third-abiss",
        "--dependency=afterok:101",
        "--wrap=python abiss.py",
    ]


def test_local_arrays_are_sequential_shards(subprocess_run):
    assert pipeline.execute_steps([make_step(array=3)]) == 0
    assert [call.args[0] for call in subprocess_run.call_args_list] == [
        f"python mask.py --shard-id {index} --num-shards 3" for index in range(3)
    ]


@pytest.mark.parametrize("array", [0, 1])
def test_unsharded_slurm_keeps_original_command(array, subprocess_run):
    assert pipeline.execute_steps([make_step(array=array)], launcher="slurm") == 0
    args = subprocess_run.call_args.args[0]
    assert args[-1] == "--wrap=python mask.py"
    assert not any(arg.startswith("--array") for arg in args)


@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_prepare_callback_once_before_prepare_and_launch(launcher, subprocess_run):
    events = []

    def record_run(command, **kwargs):
        events.append(command)
        return SimpleNamespace(stdout="123")

    subprocess_run.side_effect = record_run
    prepare_fn = Mock(side_effect=lambda: events.append("callback"))
    step = make_step(prepare="python initialize.py", prepare_fn=prepare_fn)
    assert pipeline.execute_steps([step], launcher=launcher) == 0
    prepare_fn.assert_called_once_with()
    assert events[:2] == ["callback", "python initialize.py"]
    assert "python mask.py" in str(events[2])


@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_dry_run_never_calls_subprocess_or_preparation(launcher, subprocess_run, capsys):
    prepare_fn = Mock()
    step = make_step(array=3, prepare="python initialize.py", prepare_fn=prepare_fn)
    assert pipeline.execute_steps([step], launcher=launcher, dry_run=True) == 0
    subprocess_run.assert_not_called()
    prepare_fn.assert_not_called()
    output = capsys.readouterr().out
    assert "python initialize.py" in output
    assert "python mask.py" in output


@pytest.mark.parametrize("launcher, expected", [("local", 1), ("slurm", 0)])
def test_missing_inputs_block_local_only(tmp_path, launcher, expected, subprocess_run, capsys):
    step = make_step(inputs=[("tissue", tmp_path / "missing.zarr")])
    assert pipeline.execute_steps([step], launcher=launcher) == expected
    assert subprocess_run.call_count == (launcher == "slurm")
    assert ("BLOCKED" in capsys.readouterr().out) == (launcher == "local")


def test_preparation_can_create_a_required_input(tmp_path, subprocess_run):
    manifest = tmp_path / "nuclei.json"
    prepare_fn = Mock(side_effect=lambda: manifest.write_text("{}"))
    step = make_step(inputs=[("manifest", manifest)], prepare_fn=prepare_fn)
    assert pipeline.execute_steps([step]) == 0
    prepare_fn.assert_called_once_with()
    subprocess_run.assert_called_once()


def test_check_only_skips_all_execution(subprocess_run, capsys):
    prepare_fn = Mock()
    step = make_step(prepare="init", prepare_fn=prepare_fn, skip="disabled")
    assert pipeline.execute_steps([step], check=True) == 0
    prepare_fn.assert_not_called()
    subprocess_run.assert_not_called()
    assert "disabled: disabled" in capsys.readouterr().out


def test_selection_preserves_canonical_order(subprocess_run):
    steps = [make_step("first"), make_step("second"), make_step("third")]
    assert pipeline.execute_steps(steps, selected=["third", "first"]) == 0
    assert [call.args[0] for call in subprocess_run.call_args_list] == [
        "python first.py",
        "python third.py",
    ]


@pytest.mark.parametrize("option", ["selected", "forced"])
def test_unknown_step_fails_before_launch(option, subprocess_run):
    with pytest.raises(ValueError, match="unknown steps: typo; available: mask"):
        pipeline.execute_steps([make_step()], **{option: ["typo"]})
    subprocess_run.assert_not_called()


def test_unknown_launcher_fails_before_launch(subprocess_run):
    with pytest.raises(ValueError, match="unknown launcher"):
        pipeline.execute_steps([make_step()], launcher="typo")
    subprocess_run.assert_not_called()


@pytest.mark.parametrize(
    "params, block, gpus, expected",
    [
        (
            {
                "cluster": {"account": "weiss", "extra": "--qos=normal --constraint='a b'"},
                "inference": {
                    "slurm_partition": "gpu",
                    "cpus": 8,
                    "memory": "64G",
                    "time": "8:00:00",
                },
            },
            "inference",
            1,
            "-p gpu --gres=gpu:1 -c 8 --mem 64G -t 8:00:00 -A weiss "
            "--qos=normal '--constraint=a b'",
        ),
        (
            {
                "cluster": {},
                "abiss": {"slurm_partition": "", "cpus": 64, "memory": "250G", "time": "24:00:00"},
            },
            "abiss",
            0,
            "-c 64 --mem 250G -t 24:00:00",
        ),
        (
            {
                "cluster": {
                    "partition": "long",
                    "account": "ignored-by-original",
                    "extra": "--qos=ignored-by-original",
                    "abiss": {"cpus": 64, "memory": "200G", "time": "2-00:00:00"},
                }
            },
            "abiss",
            0,
            "-p long -c 64 --mem 200G -t 2-00:00:00",
        ),
        (
            {
                "cluster": {
                    "partition": "long",
                    "smoke": {
                        "partition": "short",
                        "cpus": 16,
                        "memory": "64G",
                        "time": "04:00:00",
                    },
                }
            },
            "smoke",
            0,
            "-p short -c 16 --mem 64G -t 04:00:00",
        ),
    ],
)
def test_both_resource_schemas_preserve_exact_flags(params, block, gpus, expected):
    assert pipeline.sbatch_resources(OmegaConf.create(params), block, gpus=gpus) == expected


def test_from_scratch_selects_its_flat_schema_when_both_exist():
    params = OmegaConf.create(
        {
            "pipeline": "cube_from_scratch",
            "cluster": {
                "partition": "long",
                "abiss": {"cpus": 8, "memory": "64G", "time": "1:00:00"},
            },
            "abiss": {"cpus": 64, "memory": "250G", "time": "24:00:00"},
        }
    )
    assert pipeline.sbatch_resources(params, "abiss") == "-c 64 --mem 250G -t 24:00:00"


@pytest.mark.parametrize("kind", ["absent", "null"])
def test_fitted_values_report_every_offending_key(kind):
    params = {} if kind == "absent" else dict.fromkeys(pipeline.FITTED_KEYS)
    with pytest.raises(ValueError) as error:
        pipeline.require_dataset_values({"param": params})
    assert f"{kind}:" in str(error.value)
    for key in pipeline.FITTED_KEYS:
        assert key in str(error.value)


def test_fitted_values_distinguish_absent_and_null():
    with pytest.raises(ValueError) as error:
        pipeline.require_dataset_values({"param": {"WS_HIGH_THRESHOLD": None}})
    assert "absent: WS_LOW_THRESHOLD, AGG_THRESHOLD, CHUNK_SIZE" in str(error.value)
    assert "null: WS_HIGH_THRESHOLD" in str(error.value)


def test_complete_fitted_values_pass():
    pipeline.require_dataset_values(
        {"param": dict(zip(pipeline.FITTED_KEYS, [0.95, 0.0, 0.8, [32, 32, 16]]))}
    )


def test_shared_fitted_defaults_cannot_be_inherited(tmp_path):
    base = tmp_path / "_base"
    base.mkdir()
    tutorial = tmp_path / "neuron_third"
    tutorial.mkdir()
    values = dict(zip(pipeline.FITTED_KEYS, [0.95, 0.0, 0.8, [32, 32, 16]]))
    OmegaConf.save(OmegaConf.create({"abiss_chunk": {"param": values}}), base / "abiss.yaml")
    config = tutorial / "2_abiss.yaml"
    config.write_text("_base_: ../_base/abiss.yaml\n")
    abiss = pipeline.load_workflow_yaml(config).abiss_chunk
    with pytest.raises(ValueError, match="inherited outside dataset") as error:
        pipeline.require_dataset_values(abiss, path=config)
    for key in pipeline.FITTED_KEYS:
        assert key in str(error.value)


def test_smoke_can_inherit_its_dataset_fitted_values(tmp_path):
    values = dict(zip(pipeline.FITTED_KEYS, [0.95, 0.0, 0.8, [32, 32, 16]]))
    OmegaConf.save(OmegaConf.create({"abiss_chunk": {"param": values}}), tmp_path / "2_abiss.yaml")
    smoke = tmp_path / "2_abiss_smoke.yaml"
    smoke.write_text("_base_: 2_abiss.yaml\n")
    pipeline.require_dataset_values(pipeline.load_workflow_yaml(smoke).abiss_chunk, path=smoke)


def test_params_and_recursive_workflow_loading(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("params:\n  paths:\n    output: /synthetic/out\n")
    recipe = tmp_path / "2_abiss.yaml"
    recipe.write_text(
        "_base_: params.yaml\nabiss_chunk:\n  workdir: ${params.paths.output}/abiss\n"
    )
    smoke = tmp_path / "2_abiss_smoke.yaml"
    smoke.write_text("_base_: 2_abiss.yaml\nabiss_chunk:\n  smoke: true\n")
    assert pipeline.load_params(params).paths.output == "/synthetic/out"
    resolved = pipeline.load_workflow_yaml(smoke)
    assert resolved.abiss_chunk.workdir == "/synthetic/out/abiss"
    assert resolved.abiss_chunk.smoke is True
    assert "params" not in resolved


def test_shards_require_all_sentinels(tmp_path):
    out = tmp_path / "keep.zarr"
    out.mkdir()
    assert pipeline.check_shards(out, 2).done is False
    Path(f"{out}.done.0").touch()
    assert pipeline.check_shards(out, 2).detail.startswith("1/2")
    Path(f"{out}.done.1").touch()
    assert pipeline.check_shards(out, 2).done is True


def test_layer_info_alone_does_not_mean_complete(tmp_path):
    (tmp_path / "info").write_text("{}")
    assert pipeline.check_layer(tmp_path, 2).done is False
    scale = tmp_path / "1_1_1"
    scale.mkdir()
    (scale / "first").touch()
    assert pipeline.check_layer(tmp_path, 2).done is False
    (scale / "second").touch()
    assert pipeline.check_layer(tmp_path, 2).done is True


def test_glob_inputs(tmp_path):
    pattern = tmp_path / "**" / "size.*.txt"
    assert pipeline.input_exists(pattern) is False
    child = tmp_path / "chunk"
    child.mkdir()
    (child / "size.0.txt").touch()
    assert pipeline.input_exists(pattern) is True


def test_generic_cli_uses_supplied_tutorial_and_default_job_prefix(
    tmp_path, monkeypatch, subprocess_run, capsys
):
    from connectomics.playbooks import cube_decode

    tutorial = tmp_path / "neuron_third"
    tutorial.mkdir()
    params = tutorial / "params.yaml"
    params.write_text("params:\n  pipeline: cube\n  cluster:\n    launcher: slurm\n")
    builder = Mock(return_value=[make_step()])
    monkeypatch.setattr(cube_decode, "build_steps", builder)
    assert pipeline.main(["--params", str(params)]) == 0
    assert builder.call_args.args[0].pipeline == "cube"
    assert builder.call_args.args[1] == tutorial
    assert "--job-name=third-mask" in subprocess_run.call_args.args[0]
    assert str(params) in capsys.readouterr().out


def test_tutorial_cli_defaults_and_local_override(tmp_path, monkeypatch, subprocess_run):
    from connectomics.playbooks import cube_decode

    (tmp_path / "params.yaml").write_text("params:\n  cluster:\n    launcher: slurm\n")
    monkeypatch.setattr(cube_decode, "build_steps", lambda params, tutorial: [make_step()])
    assert pipeline.main(["--local"], default_tutorial=tmp_path, job_prefix="original") == 0
    assert subprocess_run.call_args.args[0] == "python mask.py"


@pytest.mark.parametrize(
    "sequence", [["mask", "abiss"], ["score", "guard", "abiss", "smoke", "mask"]]
)
def test_cli_cannot_replace_or_reorder_sequence_in_params(tmp_path, subprocess_run, sequence):
    params = tmp_path / "params.yaml"
    OmegaConf.save(OmegaConf.create({"params": {"pipeline": sequence}}), params)
    with pytest.raises(ValueError, match="Unknown pipeline.*choose from cube, cube_from_scratch"):
        pipeline.main(["--params", str(params)])
    subprocess_run.assert_not_called()
