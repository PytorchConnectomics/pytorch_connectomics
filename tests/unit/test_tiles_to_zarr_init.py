"""`tiles_to_zarr --stage init` must keep writing OME-NGFF 0.4, i.e. zarr v2.

The stage raised under zarr 3 (`Group.create_dataset` requires `shape=`, and a
numcodecs compressor is rejected for a v3 array), so the array creation moved to
`create_array`. That switch is only safe with `zarr_format=2` pinned: zarr 3
defaults a fresh group to v3, which writes `zarr.json` instead of the
`.zgroup`/`.zattrs`/`.zarray` trio that the NGFF 0.4 metadata written here -- and
the already-converted volumes on disk -- depend on.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "tiles_to_zarr.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tiles_to_zarr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_args(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=str(output), levels=2, chunk=[8, 8, 8],
        resolution=[10.0, 10.0, 10.0], force=True,
    )


@pytest.fixture
def initialized(tmp_path: Path):
    module = _load_script()
    output = tmp_path / "out.zarr"
    module.stage_init(_init_args(output), {"depth": 16, "height": 16, "width": 16})
    return output


def test_init_writes_zarr_v2_ngff_group(initialized: Path) -> None:
    assert json.loads((initialized / ".zgroup").read_text())["zarr_format"] == 2
    multiscales = json.loads((initialized / ".zattrs").read_text())["multiscales"]
    assert multiscales[0]["version"] == "0.4"
    assert [d["path"] for d in multiscales[0]["datasets"]] == ["0", "1"]


def test_init_array_metadata_matches_the_converted_volumes(initialized: Path) -> None:
    meta = json.loads((initialized / "0" / ".zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["dtype"] == "|u1" and meta["fill_value"] == 0
    assert meta["shape"] == [16, 16, 16] and meta["chunks"] == [8, 8, 8]
    assert meta["compressor"]["id"] == "blosc"
    assert meta["compressor"]["cname"] == "zstd" and meta["compressor"]["clevel"] == 3
    # Level 1 is the 2x downsample.
    assert json.loads((initialized / "1" / ".zarray").read_text())["shape"] == [8, 8, 8]


def test_init_output_is_writable_the_way_the_later_stages_open_it(initialized: Path) -> None:
    """`stage_base`/`stage_pyramid` reopen with `zarr.open(..., mode="r+")["0"]`."""
    block = np.random.default_rng(0).integers(0, 255, size=(8, 8, 8), dtype=np.uint8)
    zarr.open(str(initialized), mode="r+")["0"][0:8, 0:8, 0:8] = block

    stored = zarr.open(str(initialized), mode="r")["0"]
    np.testing.assert_array_equal(stored[0:8, 0:8, 0:8], block)
    assert stored[8:16, 8:16, 8:16].max() == 0  # untouched blocks keep fill_value
    assert (initialized / "0" / "0.0.0").exists()  # v2 dot-separated chunk keys
