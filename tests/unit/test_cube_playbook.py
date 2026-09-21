"""Step fixtures captured before migration from untouched a692d2e3 drivers.

The capture harness used the same scenario() and serialize() below. Moritz's main
storage count was set to 2 to match the synthetic BBOX; its smoke count was left
at 3584. All completion predicates ran against real temporary files. Only config
loaders were replaced. Fixtures are immutable evidence, never regenerated here.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
import zarr
from omegaconf import OmegaConf

from connectomics.playbooks import cube_decode
from connectomics.runtime import volume_pipeline

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests/fixtures/playbook_baseline"


def scenario(dataset, root, state):
    resources = {"cpus": 3, "memory": "17G", "time": "02:03:04", "slurm_partition": "cpu"}
    params = {
        "paths": {"dataset_root": str(root / "data"), "output_root": str(root / "out")},
        "cluster": {
            "launcher": "slurm",
            "account": "lab",
            "extra": "--qos=normal",
            "partition": "medium",
        },
        "data": {
            "raw_em": str(root / "em.zarr/main"),
            "dense_images": str(root / "images"),
            "dense_labels": str(root / "labels"),
            "tissue_mask": str(root / "tissue.zarr"),
            "keep_mask": str(root / "keep.zarr"),
            "nucleus_volume": "",
            "gt_skeletons": str(root / "skeletons"),
            "skeletons": str(root / "skeletons"),
            "ffn_node_lut": str(root / "ffn_node_lut.npz"),
        },
        "download": {
            **resources,
            "enabled": state != "disabled",
            "checkpoint_url": "https://example.org/model.ckpt",
            "train_zip": "https://example.org/train.zip",
            "em_source": "gs://synthetic/em",
            "em_bbox": [],
            "slab": 1,
            "tile_xy": 4,
            "num_shards": 2,
        },
        "mask": {**resources, "enabled": state != "disabled", "num_shards": 2},
        "train": {**resources, "enabled": state != "disabled", "num_gpus": 2},
        "inference": {**resources, "num_shards": 3},
        "abiss": dict(resources),
        "error_correction": dict(resources),
    }
    for block in ("mask", "smoke", "abiss", "score"):
        params["cluster"][block] = {"cpus": 5, "memory": "23G", "time": "03:04:05"}
    params["cluster"]["mask"]["partition"] = "short"
    param = {
        "SEG_PATH": f"file://{root}/seg",
        "BBOX": [0, 0, 0, 128, 128, 64],
        "CHUNK_SIZE": [128, 128, 32],
        "WS_HIGH_THRESHOLD": 0.91,
        "WS_LOW_THRESHOLD": 0.03,
        "AGG_THRESHOLD": 0.7,
    }
    abiss = {
        "abiss_chunk": {
            "param": param,
            "workdir": str(root / "abiss/run"),
            "source_affinity_h5": str(root / "affinity.h5"),
            "abiss_home": str(root / "abiss_home"),
            "seg_chunk_size_xyz": [128, 128, 32],
        }
    }
    smoke = {
        "abiss_chunk": {
            **abiss["abiss_chunk"],
            "param": {
                **param,
                "SEG_PATH": f"file://{root}/smoke_seg",
                "BBOX": [0, 0, 0, 1024, 1024, 1792],
            },
        }
    }
    ec = {
        "error_correction": {
            "workdir": str(root / "ec"),
            "output_segmentation": str(root / "ec_seg"),
            "nucleus_manifest": str(root / "nuclei.json"),
            "segmentation": str(root / "seg"),
            "affinity_chunks": str(root / "affinity.chunks"),
            "keep_mask": str(root / "keep.zarr"),
            "size_glob": str(root / "sizes/*.txt"),
        }
    }
    workflows = {
        "2_abiss.yaml": abiss,
        "3_abiss.yaml": abiss,
        "2_abiss_smoke.yaml": smoke,
        "4_error_correction.yaml": ec,
    }
    pytc = {
        "1_train.yaml": {"save_path": str(root / "train")},
        "2_infer.yaml": {"save_path": str(root / "infer"), "decoding": {"save_suffix": "aff"}},
    }
    if state == "complete":

        def touch(path, text=""):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        for name in (
            "images",
            "labels",
            "tissue.zarr",
            "keep.zarr",
            "data/ckpt/model.ckpt",
            "seg/info",
            "ec/error_correction_manifest.json",
            "out/reports/guard_seg.json",
            "out/reports/nerl.json",
            "train/run/checkpoints/model.ckpt",
            "abiss/scratch/done/0",
        ):
            touch(root / name)
        for name in ("tissue.zarr", "keep.zarr"):
            for shard in range(2):
                touch(root / f"{name}.done.{shard}")
        zarr.open_group(str(root / "em.zarr"), mode="w").create_array(
            "main", shape=(2, 4, 4), dtype="uint8"
        )
        touch(root / "em.zarr.progress.0", "0 1")
        touch(root / "infer/vol_aff.h5.index.json", json.dumps({"chunks": [{"path": "chunk0.h5"}]}))
        touch(root / "infer/chunk0.h5")
        for layer, count in (("seg", 2), ("smoke_seg", 8 * 8 * 56)):
            for i in range(count):
                touch(root / layer / "128_128_32" / str(i))
    return OmegaConf.create(params), workflows, pytc


def serialize(steps, root):
    result = []
    for step in steps:
        status = step.status()
        result.append(
            {
                "name": step.name,
                "title": step.title,
                "command": step.command,
                "resources": step.resources,
                "array": getattr(step, "array", 0),
                "prepare": getattr(step, "prepare", ""),
                "prepare_fn_present": getattr(step, "prepare_fn", None) is not None,
                "inputs": [(label, str(path)) for label, path in getattr(step, "inputs", [])],
                "skip": getattr(step, "skip", ""),
                "status.done": status.done,
                "status.detail": status.detail,
            }
        )
    return json.loads(json.dumps(result).replace(str(root), "<ROOT>").replace(str(REPO), "<REPO>"))


@pytest.mark.parametrize(
    "dataset,pipeline",
    [
        ("j0126", "cube_from_scratch"),
        ("moritz_l4", "cube"),
    ],
)
@pytest.mark.parametrize("state", ["pending", "complete", "disabled"])
def test_original_step_baselines(tmp_path, monkeypatch, dataset, pipeline, state):
    params, workflows, pytc = scenario(dataset, tmp_path, state)
    params.pipeline = pipeline
    monkeypatch.setattr(
        cube_decode, "load_workflow_yaml", lambda p: OmegaConf.create(workflows[p.name])
    )
    monkeypatch.setattr(cube_decode, "load_pytc_config", lambda p: OmegaConf.create(pytc[p.name]))
    steps = cube_decode.build_steps(params, REPO / f"tutorials/neuron_{dataset}")
    if dataset == "j0126":
        assert [step.name for step in steps] == list(cube_decode.CUBE_FROM_SCRATCH)
        by_name = {step.name: step for step in steps}
        assert by_name["ec"].chain
        assert [entry[0] for entry in by_name["ec"].chain] == list(cube_decode.EC_STAGES)
        assert "--vds" in by_name["abiss"].command
        assert "evaluate_j0126.py" in by_name["eval"].command
        assert bool(by_name["train"].skip) == (state == "disabled")
        assert by_name["abiss"].status().done == (state == "complete")
    else:
        baseline = json.loads((FIXTURES / f"{dataset}.json").read_text())
        assert serialize(steps, tmp_path) == baseline[state]


@pytest.mark.parametrize("pipeline", ["cube", "cube_from_scratch"])
@pytest.mark.parametrize("failure", ["absent", "null"])
def test_fitted_values_on_both_sequences(tmp_path, monkeypatch, pipeline, failure):
    dataset = "moritz_l4" if pipeline == "cube" else "j0126"
    params, workflows, pytc = scenario(dataset, tmp_path, "pending")
    params.pipeline = pipeline
    keys = ("WS_HIGH_THRESHOLD", "WS_LOW_THRESHOLD", "AGG_THRESHOLD", "CHUNK_SIZE")
    for workflow in workflows.values():
        if "abiss_chunk" in workflow:
            for key in keys:
                if failure == "absent":
                    workflow["abiss_chunk"]["param"].pop(key, None)
                else:
                    workflow["abiss_chunk"]["param"][key] = None
    monkeypatch.setattr(
        cube_decode, "load_workflow_yaml", lambda p: OmegaConf.create(workflows[p.name])
    )
    monkeypatch.setattr(cube_decode, "load_pytc_config", lambda p: OmegaConf.create(pytc[p.name]))
    with pytest.raises(ValueError) as error:
        cube_decode.build_steps(params, REPO / f"tutorials/neuron_{dataset}")
    for key in keys:
        assert key in str(error.value)
    assert failure in str(error.value).lower()


def third_tutorial(root):
    tutorial = root / "neuron_third"
    tutorial.mkdir()
    params = {
        "pipeline": "cube",
        "paths": {
            "repository": str(REPO),
            "dataset_root": str(root / "third_data"),
            "output_root": str(root / "third_output"),
        },
        "data": {
            "affinity_precomputed": str(root / "third_affinity"),
            "gt_skeletons": str(root / "third_skeletons"),
            "keep_mask": str(root / "third_keep.h5"),
            "mask_ratio": [2, 4, 4],
            "masks": {
                "strategy": "downsampled_sources",
                "ratio_zyx": [2, 4, 4],
                "border_start_zyx": [0, 9, 9],
                "sources": [
                    {
                        "path": str(root / "third_vessels.h5"),
                        "dataset": "main",
                        "polarity": "exclude",
                    }
                ],
            },
        },
        "frame": {
            "volume_shape_zyx": [64, 256, 384],
            "volume_origin_global_zyx": [71, 13, 17],
            "resolution_xyz_nm": [8, 9, 25],
        },
        "cluster": {
            "launcher": "slurm",
            "partition": "cpu",
            **{
                key: {"cpus": 2, "memory": "8G", "time": "00:20:00"}
                for key in ("mask", "smoke", "abiss", "score")
            },
        },
    }
    (tutorial / "params.yaml").write_text(yaml.safe_dump({"params": params}))
    abiss = {
        "_base_": [str(REPO / "tutorials/_base/abiss.yaml"), "params.yaml"],
        "abiss_chunk": {
            "seg_chunk_size_xyz": [128, 128, 32],
            "aff_chunk_size_xyz": [128, 128, 32],
            "param": {
                "NAME": "third",
                "AFF_PATH": "file://${params.data.affinity_precomputed}",
                "BBOX": [0, 0, 0, 384, 256, 64],
                "CHUNK_SIZE": [128, 128, 32],
                "WS_HIGH_THRESHOLD": 0.93,
                "WS_LOW_THRESHOLD": 0.02,
                "AGG_THRESHOLD": 0.72,
                "WS_SIZE_THRESHOLD": 100,
                "WS_DUST_THRESHOLD": 50,
            },
        },
    }
    (tutorial / "2_abiss.yaml").write_text(yaml.safe_dump(abiss))
    return tutorial


def test_config_only_onboarding(tmp_path, monkeypatch, capsys):
    tutorial = third_tutorial(tmp_path)
    params = volume_pipeline.load_params(tutorial / "params.yaml")
    steps = cube_decode.build_steps(params, tutorial)
    assert tuple(s.name for s in steps) == cube_decode.CUBE
    assert {p.name for p in tutorial.iterdir()} == {"params.yaml", "2_abiss.yaml"}
    for step in steps:
        assert str(tmp_path) in step.command
        assert "neuron_moritz_l4" not in step.command
        assert "neuron_j0126" not in step.command
    mask, smoke, abiss, guard, score = steps
    assert str(tutorial / "params.yaml") in mask.command
    assert str(tmp_path / "third_keep.h5") in mask.command
    assert str(tutorial / "2_abiss.yaml") in abiss.command
    assert str(tmp_path / "third_output") in guard.command
    assert str(tmp_path / "third_skeletons") in score.command
    assert "--global-origin-zyx 71 13 17" in score.command
    assert "--resolution-xyz-nm 8 9 25" in score.command
    assert "--volume-shape-zyx 64 256 384" in score.command
    generated = tmp_path / "third_output/abiss_smoke/config.yaml"
    with monkeypatch.context() as m:

        def forbidden(*args, **kwargs):
            pytest.fail("dry-run invoked subprocess")

        m.setattr(volume_pipeline.subprocess, "run", forbidden)
        assert volume_pipeline.execute_steps(steps, launcher="slurm", dry_run=True) == 0
    assert not generated.exists()
    smoke.prepare_fn()
    resolved = cube_decode._load_abiss(generated)
    original = cube_decode._load_abiss(tutorial / "2_abiss.yaml")
    for key in ("CHUNK_SIZE", "WS_HIGH_THRESHOLD", "WS_LOW_THRESHOLD", "AGG_THRESHOLD"):
        assert resolved.param[key] == original.param[key]
    assert resolved.param.AFF_PATH == f"file://{tmp_path}/third_affinity"
    assert "third_output/abiss_smoke" in resolved.param.SEG_PATH
    assert resolved.param.AFF_KEEP_MASK == str(tmp_path / "third_keep.h5") + "::main"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/validate_tutorial_configs.py"),
            "--glob",
            "neuron_third/*.yaml",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 1 canonical tutorial configs successfully; skipped 1" in result.stdout


def test_unknown_sequence(tmp_path):
    params = OmegaConf.create({"pipeline": "cube_typo"})
    with pytest.raises(ValueError, match="cube_from_scratch"):
        cube_decode.build_steps(params, tmp_path)


@pytest.mark.parametrize(
    "storage,expected",
    [({}, 8), ({"copy_block_shape_xyz": [256, 256, 32]}, 64)],
)
def test_storage_completion_uses_executor_defaults(storage, expected):
    abiss = OmegaConf.create({"param": {"BBOX": [0, 0, 0, 1024, 1024, 128]}, **storage})
    assert cube_decode._storage_chunks(abiss) == expected


@pytest.mark.parametrize("h5_source", [False, True])
def test_derived_smoke_preserves_fit_and_isolates_outputs(tmp_path, h5_source):
    param = {
        "BBOX": [10, 20, 30, 5610, 8554, 3336],
        "CHUNK_SIZE": [2816, 2944, 32],
        "WS_HIGH_THRESHOLD": 0.93,
        "WS_LOW_THRESHOLD": 0.02,
        "AGG_THRESHOLD": 0.72,
        "AFF_PATH": f"file://{tmp_path}/full_affinity",
        "CHUNKMAP_INPUT": f"file://{tmp_path}/full_chunkmap",
        "NUC_COMPETITION_MANIFEST": str(tmp_path / "full_manifest.json"),
    }
    abiss = OmegaConf.create({"param": param, "top_mip": 7, "root_tag": "7_0_0_0"})
    if h5_source:
        abiss.source_affinity_h5 = str(tmp_path / "input.h5")
    smoke = cube_decode._smoke_config(abiss, tmp_path)
    assert list(smoke.param.BBOX) == [10, 20, 30, 1034, 1044, 1822]
    for key in volume_pipeline.FITTED_KEYS:
        assert smoke.param[key] == abiss.param[key]
    for key in ("WS_PATH", "SEG_PATH", "SCRATCH_PATH", "CHUNKMAP_OUTPUT", "CHUNKMAP_INPUT"):
        assert f"{tmp_path}/abiss_smoke/" in smoke.param[key]
    assert f"{tmp_path}/abiss_smoke/" in smoke.param.NUC_COMPETITION_MANIFEST
    assert (smoke.param.AFF_PATH == abiss.param.AFF_PATH) is (not h5_source)
    assert "top_mip" not in smoke and "root_tag" not in smoke
    assert list(abiss.param.BBOX) == param["BBOX"]


def test_scorer_uses_supplied_frame(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from connectomics.metrics import nerl

    monkeypatch.setattr(nerl, "import_em_erl", lambda: (SimpleNamespace, None, None))
    path = REPO / "scripts/score_moritz_l4_nerl.py"
    spec = importlib.util.spec_from_file_location("cube_scorer", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    skeletons = tmp_path / "skeletons"
    skeletons.mkdir()
    (skeletons / "cell1.nml").write_text(
        '<things><thing id="1"><nodes>'
        '<node id="1" x="13" y="11" z="118"/>'
        '<node id="2" x="14" y="12" z="119"/>'
        '<node id="3" x="13" y="13" z="118"/>'
        '</nodes><edges><edge source="1" target="2"/>'
        '<edge source="2" target="3"/></edges></thing></things>'
    )
    default_graph = scorer.build_graph(skeletons)
    np.testing.assert_array_equal(
        default_graph.node_coords_zyx, [[0, 11, 13], [1, 12, 14], [0, 13, 13]]
    )
    np.testing.assert_allclose(default_graph.edge_len, [np.sqrt(28**2 + 2 * 11.24**2)] * 2)

    graph = scorer.build_graph(
        skeletons,
        volume_shape_zyx=(3, 3, 3),
        global_origin_zyx=(118, 11, 13),
        resolution_zyx_nm=(20, 10, 5),
    )
    np.testing.assert_array_equal(graph.node_coords_zyx, [[0, 0, 0], [1, 1, 1], [0, 2, 0]])
    np.testing.assert_allclose(graph.edge_len, [np.sqrt(525), np.sqrt(525)])
    np.testing.assert_allclose(graph.skeleton_len, [2 * np.sqrt(525)])
    np.testing.assert_array_equal(graph.edge_u, [0, 1])
    np.testing.assert_array_equal(graph.edge_v, [1, 2])

    cropped = scorer.build_graph(
        skeletons,
        volume_shape_zyx=(1, 3, 3),
        global_origin_zyx=(118, 11, 13),
        resolution_zyx_nm=(20, 10, 5),
    )
    np.testing.assert_array_equal(cropped.node_coords_zyx, [[0, 0, 0], [0, 2, 0]])
    assert len(cropped.edge_len) == 0
