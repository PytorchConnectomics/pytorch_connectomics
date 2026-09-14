"""Keep-mask numeric, source movement, and CLI contracts."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
import zarr

from connectomics.data import keep_mask


def _spec(tmp_path, **kwargs):
    return keep_mask.KeepMaskSpec(
        strategy="downsampled_sources",
        out=tmp_path / "keep.h5",
        volume_shape_zyx=(5, 7, 9),
        ratio_zyx=(2, 2, 2),
        border_start_zyx=(0, 3, 1),
        **kwargs,
    )


def _h5(path, data, dataset="main"):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=data)


def test_sources_polarity_border_and_crop(tmp_path):
    vessel = np.zeros((4, 5, 6), np.uint8)
    vessel[1, 3, 3] = 255
    tissue = np.ones_like(vessel)
    tissue[2, 3, 4] = 0
    _h5(tmp_path / "vessel.h5", vessel)
    _h5(tmp_path / "tissue.h5", tissue, "tissue")
    spec = _spec(
        tmp_path,
        sources=[
            {"path": tmp_path / "vessel.h5", "dataset": "main", "polarity": "exclude"},
            keep_mask.MaskSource(tmp_path / "tissue.h5", "tissue", "keep"),
        ],
    )

    keep_mask.build(spec)

    expected = np.ones((3, 4, 5), np.uint8)
    expected[:, :2, :] = 0
    expected[:, :, :1] = 0
    expected[1, 3, 3] = 0
    expected[2, 3, 4] = 0
    with h5py.File(spec.out) as f:
        np.testing.assert_array_equal(f["main"][:], expected)
        np.testing.assert_array_equal(f["main"].attrs["downsample_factors_zyx"], (2, 2, 2))
        assert f["main"].attrs["axis_order"] == "ZYX"


@pytest.mark.parametrize(
    "polarity,shape,message",
    [
        ("other", (3, 4, 5), "polarity"),
        ("exclude", (2, 4, 5), "does not cover"),
    ],
)
def test_source_errors_fail_before_output(tmp_path, polarity, shape, message):
    source = tmp_path / "source.h5"
    _h5(source, np.ones(shape))
    spec = _spec(tmp_path, sources=[keep_mask.MaskSource(source, polarity=polarity)])
    with pytest.raises(ValueError, match=message):
        keep_mask.build(spec)
    assert not spec.out.exists()


def test_unknown_strategy_and_nonpositive_ratio(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        keep_mask.build(replace(_spec(tmp_path), strategy="typo"))
    with pytest.raises(ValueError, match="ratio_zyx"):
        keep_mask.build(replace(_spec(tmp_path), ratio_zyx=(0, 2, 2)))


def test_j0126_functions_and_constants_are_verbatim_movement():
    repo = Path(__file__).resolve().parents[2]
    original = subprocess.run(
        ["git", "show", "a692d2e3:scripts/build_j0126_keep_mask.py"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    moved = Path(inspect.getfile(keep_mask)).read_text()
    moved_nodes = {
        node.name: node for node in ast.parse(moved).body if isinstance(node, ast.FunctionDef)
    }
    for node in ast.parse(original).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.get_source_segment(moved, moved_nodes[node.name]) == ast.get_source_segment(
                original, node
            ), node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assert getattr(keep_mask, target.id) == ast.literal_eval(node.value), target.id


def test_ffn_init_and_tissue_shard_write_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(keep_mask, "TISSUE_SHAPE_ZYX", (2, 2, 3))
    values = np.zeros((3, 2, 2, 6), dtype=np.uint8)
    values[1, 1, 1, 3] = 129

    class Volume:
        shape = values.shape
        num_channels = 6

        def __getitem__(self, item):
            return values[item]

    monkeypatch.setitem(
        sys.modules, "cloudvolume", SimpleNamespace(CloudVolume=lambda *a, **k: Volume())
    )
    spec = replace(
        _spec(tmp_path), strategy="ffn_tissue_border", out=tmp_path / "tissue.zarr", z_slab=1
    )
    argv = sys.argv
    # The original writer uses the zarr-2 compressor API. Do not change it during relocation.
    with zarr.config.set({"default_zarr_format": 2}):
        keep_mask.build(spec, stage="tissue", init=True)
        assert not Path(f"{spec.out}.done.1").exists()
        out = zarr.open(str(spec.out), mode="r")
        np.testing.assert_array_equal(out[:], np.ones((2, 2, 3), dtype=np.uint8))
        keep_mask.build(spec, stage="tissue", shard_id=1, num_shards=2)
    assert sys.argv is argv
    expected = np.ones((2, 2, 3), dtype=np.uint8)
    expected[1, 1, 1] = 0
    np.testing.assert_array_equal(zarr.open(str(spec.out), mode="r")[:], expected)
    assert Path(f"{spec.out}.done.1").read_text() == "done\n"
    assert not Path(f"{spec.out}.done.0").exists()


def test_ffn_keep_init_and_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(keep_mask, "KEEP_SHAPE_ZYX", (2, 4, 4))
    monkeypatch.setattr(keep_mask, "KEEP_CHUNKS", (1, 2, 2))
    monkeypatch.setattr(keep_mask, "CELL", 4)
    tissue = tmp_path / "tissue.zarr"
    em = tmp_path / "em.zarr"
    zarr.open(str(tissue), mode="w", shape=(2, 2, 2), dtype="uint8")[:] = 1
    raw = np.full((2, 4, 4), 100, dtype=np.uint8)
    raw[:, 0, :] = 0
    zarr.open(str(em), mode="w", shape=raw.shape, dtype="uint8")[:] = raw
    spec = replace(
        _spec(tmp_path),
        strategy="ffn_tissue_border",
        out=tmp_path / "keep.zarr",
        tissue=tissue,
        em=em,
        border_offset=0,
    )
    keep_mask.build(spec, stage="keep", init=True)
    keep_mask.build(spec, stage="keep")
    expected = np.ones_like(raw)
    expected[:, 0, :] = 0
    np.testing.assert_array_equal(zarr.open(str(spec.out), mode="r")[:], expected)
    assert Path(f"{spec.out}.done.0").read_text() == "done\n"


def test_ffn_adapter_restores_argv_on_failure(tmp_path):
    argv = sys.argv
    spec = replace(_spec(tmp_path), strategy="ffn_tissue_border", out=tmp_path / "missing.zarr")
    with pytest.raises(SystemExit, match="run --init first"):
        keep_mask.build(spec, stage="tissue")
    assert sys.argv is argv


def test_config_only_mask_cli(tmp_path):
    source = tmp_path / "mask.h5"
    _h5(source, np.zeros((3, 4, 5), dtype=np.uint8))
    params = {
        "params": {
            "data": {
                "keep_mask": str(tmp_path / "configured.h5"),
                "masks": {
                    "strategy": "downsampled_sources",
                    "sources": [{"path": str(source), "dataset": "main", "polarity": "exclude"}],
                    "ratio_zyx": [2, 2, 2],
                    "border_start_zyx": [0, 3, 1],
                },
            },
            "frame": {"volume_shape_zyx": [5, 7, 9], "volume_origin_global_zyx": [1, 2, 3]},
        }
    }
    config = tmp_path / "params.yaml"
    config.write_text(yaml.safe_dump(params))
    assert keep_mask.cli_main(["--params", str(config)]) == 0
    with h5py.File(tmp_path / "configured.h5") as f:
        assert f["main"].shape == (3, 4, 5)
        np.testing.assert_array_equal(f["main"].attrs["global_offset_zyx"], [1, 2, 3])


def test_moritz_wrapper_preserves_flags(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(keep_mask, "build", captured.append)
    assert (
        keep_mask.moritz_main(
            ["--bv", str(tmp_path / "vessel.h5"), "--out", str(tmp_path / "out.h5")]
        )
        == 0
    )
    assert captured[0].out == tmp_path / "out.h5"
    assert captured[0].sources[0].path == tmp_path / "vessel.h5"
    assert captured[0].ratio_zyx == (4, 8, 8)
    assert captured[0].volume_shape_zyx == (3306, 8534, 5599)
    assert captured[0].border_start_zyx == (0, 103, 103)
