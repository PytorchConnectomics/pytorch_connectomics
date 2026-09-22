"""Contracts shared by the local and cloud LICONN volume workflows."""

import importlib.util
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

TUTORIAL = Path(__file__).parents[2] / "tutorials" / "neuron_liconn_moe"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, TUTORIAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    for key in os.environ:
        if key.startswith(("MOE_", "LICONN_MOE_")) or key == "ABISS_HOME":
            monkeypatch.delenv(key)
    return monkeypatch


def test_cloud_roots_keep_model_and_family_scoping(clean_env, tmp_path):
    for key, value in {
        "LICONN_MOE_REPO": tmp_path / "repo",
        "LICONN_MOE_SRC_ZARR": tmp_path / "src",
        "LICONN_MOE_PREPARED": tmp_path / "prepared",
        "MOE_OUT_ROOT": tmp_path / "out",
        "MOE_CKPT": tmp_path / "20260921_000000/checkpoints/model.ckpt",
        "LICONN_MOE_GCS_BUCKET": "test-bucket",
        "MOE_GCS_KIND": "mip1_eb2",
    }.items():
        clean_env.setenv(key, str(value))
    volumes = load_script("volumes")
    for name, family in [
        ("ExPID96_S1_40XW002_18x", "expid96"),
        ("ExPID99_32x_2_cerebellum", "expid99"),
        ("ExPID108_32x_Cortex_L1_01", "expid108"),
        ("ExPID71_Hippocampus_300nm_40XW01", "expid71"),
    ]:
        assert volumes.source_zarr(name) == tmp_path / "src" / f"{name}.zarr"
        assert volumes.prepared_h5(name) == tmp_path / "prepared" / f"{name}.h5"
        assert volumes.work_dir(name) == tmp_path / "out" / name
        assert volumes.gcs_folder(name) == f"gs://test-bucket/liconn/moe/{family}/mip1_eb2"
    assert volumes.TEST_OUT == tmp_path / "20260921_000000/test_model"
    legacy = next(iter(volumes.LEGACY))
    assert volumes.layer_name(legacy, 0.6079).endswith("mt0608")


@pytest.mark.parametrize("grid", ["train", "mip0"])
@pytest.mark.parametrize("auto", [False, True])
def test_preparation_preserves_grid_and_auto_recipes(clean_env, tmp_path, grid, auto):
    clean_env.setenv("MOE_GRID", grid)
    clean_env.setenv("LICONN_MOE_ROOT", str(tmp_path))
    volumes = load_script("volumes")
    name = "test-volume"
    volumes.VOLUMES[name] = {"auto": True} if auto else {"factor": (1, 2, 2)}
    group = zarr.open_group(str(volumes.source_zarr(name)), mode="w")
    group.create_dataset("0", shape=(12, 20, 20), dtype="uint8")
    group.attrs["spacing_nm_zyx"] = [30.0, 9.0, 9.0]
    plan = volumes.plan(name)
    if grid == "mip0":
        assert plan["shape"] == (12, 20, 20)
        assert plan["prepare_args"] == ["--factor", "1", "1", "1"]
        assert volumes.PREPARED == tmp_path / "prepared_mip0_grid"
    else:
        assert plan["shape"] == (12, 10, 10)
        assert volumes.PREPARED == tmp_path / "prepared_train_grid"
    # The automatic recipe must not manufacture extra Z planes.
    assert plan["spacing_zyx"][0] == 30.0
    assert plan["grid_deviation"][0] > 0


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("large", [False, True])
def test_watershed_binary_selection(clean_env, tmp_path, cloud, large):
    # The script adjusts sys.path for its sibling import; keep test state isolated.
    clean_env.syspath_prepend(str(TUTORIAL))
    sweep = load_script("sweep_merge_threshold")
    clean_env.setattr(sweep.V, "REPO", tmp_path / "repo")
    home = tmp_path / "cloud-abiss" if cloud else sweep.V.REPO / "lib/abiss"
    if cloud:
        clean_env.setenv("ABISS_HOME", str(home))
    binary = home / ("build64/ws64" if large else "build/ws")
    binary.parent.mkdir(parents=True)
    binary.touch()
    affinity = tmp_path / "affinity.h5"
    with h5py.File(affinity, "w") as handle:
        # Chunked empty datasets exercise the size guard without allocating a volume.
        shape = (3, 2048, 2048, 512) if large else (3, 8, 8, 8)
        handle.create_dataset("main", shape=shape, chunks=(1, 8, 8, 8), dtype=np.float16)
    assert sweep.resolve_ws_binary(affinity) == binary
    binary.unlink()
    with pytest.raises(SystemExit, match="REQUIRES ws64" if large else "not found"):
        sweep.resolve_ws_binary(affinity)
