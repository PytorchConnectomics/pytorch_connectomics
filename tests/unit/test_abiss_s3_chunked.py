"""Offline contracts for the S3 chunked ABISS workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from scripts.run_abiss_volume import _shift_to_destination_storage, _to_abiss_affinity
from tutorials.neuron_liconn_ist.merge_fn_sweep import EPS, uncompress
from tutorials.neuron_liconn_moe.gcloud import equivalence_test as eq
from tutorials.neuron_liconn_moe.gcloud import make_prob_affinity as canon
from tutorials.neuron_liconn_moe.gcloud.resolve_thresholds import resolve as resolve_thresholds
from tutorials.neuron_liconn_moe.gcloud.resolve_chunked import (
    compare_task_keys,
    read_task_flag_keys,
    task_flag_keys,
    artifact_bbox,
    preflight,
    resolve,
    resolve_variant,
    write_variant_config,
    write_affinity_diagnostic,
)


def base(**overrides):
    value = {"merge_criterion": "mean", "CHUNK_SIZE": [256, 256, 128],
             "seg_chunk_size_xyz": [128, 128, 128], "BBOX": [0, 0, 0, 650, 650, 503],
             "AFF_CHANNELS": [0, 1, 2]}
    value.update(overrides)
    return value


def test_criterion_and_alignment_guards():
    for criterion in ("max", "rlme", "cs"):
        with pytest.raises(ValueError, match="available chunked stages") as error:
            resolve(base(merge_criterion=criterion))
        assert criterion in str(error.value)
        assert "agglomerate_mean_edge" in str(error.value)
    assert resolve(base())["merge_criterion"] == "mean"
    with pytest.raises(ValueError, match="axis Z"):
        resolve(base(seg_chunk_size_xyz=[128, 128, 96]))
    with pytest.raises(ValueError, match=r"only \[1, 1, 7\]"):
        resolve(base(CHUNK_SIZE=[2048, 2048, 80], seg_chunk_size_xyz=[512, 512, 80],
                    BBOX=[0, 0, 0, 2048, 2048, 503]))
    with pytest.raises(ValueError, match="AFF_CHANNELS"):
        resolve(base(AFF_CHANNELS=[2, 1, 0]))


def test_layout_equivalence_and_zero_faces(tmp_path: Path):
    rng = np.random.default_rng(4)
    compressed = rng.uniform(0.01, 0.8, size=(3, 7, 9, 11)).astype(np.float32)
    left = _to_abiss_affinity(uncompress(compressed, 0.2), [2, 1, 0], "source")
    right = _to_abiss_affinity(canon_array(compressed), [0, 1, 2], "destination")
    np.testing.assert_array_equal(left, right)
    source = tmp_path / "source.h5"
    output = tmp_path / "canon.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("only", data=compressed)
    canon.convert(source, output, "only", slab_z=3)
    with h5py.File(output, "r") as handle:
        data = handle["main"][:]
    np.testing.assert_array_equal(data, canon_array(compressed))
    assert np.all(data[2, 0] == 0)       # Z destination face
    assert np.all(data[1, :, 0, :] == 0) # Y destination face
    assert np.all(data[0, :, :, 0] == 0) # X destination face


def canon_array(compressed):
    # This is the canonical value used by the layout test, expressed only in
    # terms of the imported production helpers.
    source = np.transpose(compressed[[2, 1, 0]], (3, 2, 1, 0))
    canonical = np.transpose(uncompress(_shift_to_destination_storage(source), 0.2), (3, 2, 1, 0))
    canonical[0, :, :, 0] = 0
    canonical[1, :, 0, :] = 0
    canonical[2, 0, :, :] = 0
    return canonical


def test_global_shift_differs_from_per_chunk_shift():
    rng = np.random.default_rng(5)
    source = rng.uniform(0.1, 0.9, size=(8, 5, 6, 3)).astype(np.float32)
    global_shift = _shift_to_destination_storage(source)
    split = 3
    per_chunk = np.concatenate((_shift_to_destination_storage(source[:, :, :split]),
                                _shift_to_destination_storage(source[:, :, split:])), axis=2)
    changed = np.any(global_shift != per_chunk, axis=-1)
    assert np.all(changed[:, :, split])
    assert not np.any(changed[:, :, :split - 1])


def test_masks_and_seam_metric():
    reference = np.zeros((1, 1, 8), dtype=np.uint32)
    reference[0, 0, :4] = 1  # ends at plane 4, so interior
    reference[0, 0, 3:7] = 2  # crosses plane 4, so boundary
    candidate = reference.copy()
    candidate[0, 0, 4] = 99
    bmask, imask, boundary, interior = eq.boundary_and_interior(reference, [4, 1, 1])
    assert boundary == {2} and interior == {1}
    assert eq.voi_total(reference, candidate, imask) == 0
    assert eq.voi_total(reference, candidate, bmask) > 0
    with pytest.raises(ValueError, match="empty"):
        eq.voi_total(reference, candidate, np.zeros(reference.shape, dtype=bool))
    all_boundary = np.ones((1, 1, 8), dtype=np.uint32)
    with pytest.raises(ValueError, match=r"\|B\|=1.*\|I\|=0"):
        eq.boundary_and_interior(all_boundary, [4, 1, 1])

    no_background = np.array([[[3, 3, 4, 4, 4, 5, 6, 6]]], dtype=np.uint32)
    _, _, boundary, interior = eq.boundary_and_interior(no_background, [4, 1, 1])
    assert boundary == {4} and interior == {3, 5, 6}


def test_ordering_guards_and_key_sets(tmp_path: Path, monkeypatch):
    import tutorials.neuron_liconn_moe.gcloud.resolve_chunked as resolver
    monkeypatch.setattr(resolver.Path, "is_file", lambda self: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError):
        resolve(base(merge_criterion="max"))
    assert not list(tmp_path.iterdir())
    monkeypatch.undo()
    flags = tmp_path / "scratch" / "done"
    flags.mkdir(parents=True)
    (flags / "run_watershed_atomic_chunk_0.txt").touch()
    (flags / "run_watershed_atomic_chunk_2.txt").touch()
    expected = task_flag_keys("run", [("watershed", "atomic", "chunk_0"),
                                      ("watershed", "atomic", "chunk_1")])
    assert read_task_flag_keys(tmp_path / "scratch") == {
        "run_watershed_atomic_chunk_0", "run_watershed_atomic_chunk_2"
    }
    with pytest.raises(ValueError, match="missing=.*chunk_1.*extra=.*chunk_2"):
        compare_task_keys(expected, tmp_path / "scratch")


def test_preflight_and_prepare_config_for_each_variant(tmp_path: Path):
    source = tmp_path / "source.h5"
    output = tmp_path / "aff_canon.h5"
    with h5py.File(source, "w") as handle:
        data = np.full((3, 2, 4, 4), 0.5, dtype=np.float32)
        data[:, 1, 2, 2] = 0.98  # production conversion must tolerate saturation
        handle.create_dataset("only", data=data)
    canon.convert(source, output, "only", slab_z=1)
    bbox = artifact_bbox(output)
    config = base(affinity_h5=str(output), BBOX=bbox,
                  diagnostic_path=str(tmp_path / "diagnostic.json"))
    checked = preflight(config)
    assert checked["affinity_h5"] == str(output.resolve())
    assert checked["saturated_fraction"] >= 0
    write_affinity_diagnostic(output, tmp_path / "diagnostic.json")
    template = Path("tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml")
    from connectomics.runtime.abiss_chunk import prepare_config
    for variant in ("one", "A", "B"):
        path = write_variant_config(template, tmp_path / f"{variant}.yaml",
                                    run_prefix=tmp_path / "run", affinity_h5=output,
                                    variant=variant, ws_high=0.8, ws_low=0.2,
                                    bbox=bbox, criterion="mean")
        prepared = prepare_config(path)
        assert prepared.param_payload["CHUNK_SIZE"]
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["abiss_chunk"]["param"]["AFF_PATH"] == f"file://{tmp_path / 'run' / f'chunked_{variant}' / 'aff'}"
    one_path = tmp_path / "one.yaml"
    write_variant_config(template, one_path, run_prefix=tmp_path / "run", affinity_h5=output,
                         variant="one", ws_high=0.8, ws_low=0.2, bbox=bbox)
    import yaml
    assert yaml.safe_load(one_path.read_text(encoding="utf-8"))["abiss_chunk"]["param"]["CHUNK_SIZE"] == [4, 4, 2]


def test_preflight_rejects_shape_dtype_and_missing_file(tmp_path: Path):
    missing = base(affinity_h5=str(tmp_path / "missing.h5"))
    with pytest.raises(FileNotFoundError):
        preflight(missing)
    shape_path = tmp_path / "wrong_shape.h5"
    with h5py.File(shape_path, "w") as handle:
        handle.create_dataset("main", data=np.full((3, 2, 4, 3), 0.5, dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        preflight(base(affinity_h5=str(shape_path), BBOX=[0, 0, 0, 4, 4, 2]))

    dtype_path = tmp_path / "wrong_dtype.h5"
    with h5py.File(dtype_path, "w") as handle:
        # Shape exactly matches BBOX; this failure is dtype-only.
        handle.create_dataset("main", data=np.full((3, 2, 4, 4), 0.5, dtype=np.float16))
    with pytest.raises(ValueError, match="float32"):
        preflight(base(affinity_h5=str(dtype_path), BBOX=[0, 0, 0, 4, 4, 2]))


def test_diagnostic_records_resolved_ws_thresholds(tmp_path: Path):
    path = tmp_path / "canon.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=np.full((3, 2, 3, 4), 0.5, dtype=np.float32))
    diagnostic = write_affinity_diagnostic(path, tmp_path / "diagnostic.json",
                                           ws_high=0.94, ws_low=0.20)
    assert diagnostic["ws_high"] == 0.94
    assert diagnostic["ws_low"] == 0.20


def _run_volume_with_stubs(tmp_path: Path, *, stages: str, decode: str = "") -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    log = tmp_path / "calls.log"
    affinity = tmp_path / "missing-affinity.h5"
    python_stub = bindir / "python"
    python_stub.write_text(
        """#!/bin/bash
set -eu
echo "python $*" >> "$STUB_LOG"
if [[ "${1:-}" == -c ]]; then
  if [[ "$2" == *json.load* ]]; then
    [[ "$2" == *ws_high* ]] && echo 0.94 || echo 0.20
  else
    echo "$STUB_AFF"
  fi
elif [[ "${1:-}" == - && "${2:-}" == "$STUB_AFF" ]]; then
  echo "h5py" >> "$STUB_LOG"
  echo main
elif [[ "${1:-}" == - && "${2:-}" == 4 ]]; then
  echo "2 2 2"
elif [[ "${1:-}" == - ]]; then
  echo "0 0 0 4 4 2"
elif [[ "${1:-}" == *resolve_thresholds.py ]]; then
  output=""
  while (($#)); do [[ "$1" == --output ]] && output="$2"; shift; done
  printf '{"ws_high": 0.94, "ws_low": 0.20}\n' > "$output"
elif [[ "${1:-}" == *artifact_bbox* || "${1:-}" == *variant_specs* ]]; then
  echo "0 0 0 4 4 2"
fi
""",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    for name in ("nvidia-smi", "date"):
        tool = bindir / name
        if name == "date":
            body = "#!/bin/bash\necho date >> \"$STUB_LOG\"\n"
        else:
            body = "#!/bin/bash\necho \"%s $*\" >> \"$STUB_LOG\"\n" % name
        tool.write_text(body, encoding="utf-8")
        tool.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "STUB_LOG": str(log),
        "STUB_AFF": str(affinity),
        "WORK": str(tmp_path / "work"),
        "REPO": str(Path(__file__).resolve().parents[2]),
        "MOE_OUT_ROOT": str(tmp_path / "out"),
        "STAGES": stages,
        "DECODE": decode,
        "S3_ACCEPT": "0",
        "ABISS_HOME": str(tmp_path / "abiss"),
    })
    script = Path("tutorials/neuron_liconn_moe/gcloud/run_volume.sh")
    result = subprocess.run(["bash", str(script), "TestVolume"], env=env, check=False,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(result.stdout)
    return log.read_text(encoding="utf-8")


def test_run_volume_stub_preserves_inference_skip_and_sweep_paths(tmp_path: Path):
    calls = _run_volume_with_stubs(tmp_path / "inference", stages="gpu")
    assert "scripts/main.py" in calls
    assert f"{tmp_path / 'inference' / 'missing-affinity.h5'}" not in calls

    calls = _run_volume_with_stubs(tmp_path / "cpu", stages="cpu", decode="whole")
    assert "h5py" in calls
    assert calls.index("h5py") < calls.index("run_abiss_volume.py")
    assert "--input-dataset main" in calls

    calls = _run_volume_with_stubs(tmp_path / "sweep", stages="cpu")
    assert "sweep_merge_threshold.py" in calls


def test_percentile_thresholds_are_resolved_from_canonical_artifact(tmp_path: Path):
    path = tmp_path / "canon.h5"
    values = np.linspace(0.01, 0.99, 3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=values)
    resolved = resolve_thresholds(path)
    aff = _to_abiss_affinity(values, [0, 1, 2], "destination")
    assert resolved["ws_high"] == pytest.approx(float(np.percentile(aff, 94)))
    assert resolved["ws_low"] == pytest.approx(float(np.percentile(aff, 20)))


def test_precomputed_readback_returns_zyx(monkeypatch):
    import sys
    import types

    class FakeCloudVolume:
        def __init__(self, path, **kwargs):
            assert path.startswith("file://")

        def __getitem__(self, item):
            return np.arange(24, dtype=np.uint32).reshape(4, 3, 2, 1)

    monkeypatch.setitem(sys.modules, "cloudvolume", types.SimpleNamespace(CloudVolume=FakeCloudVolume))
    result = eq.read_precomputed_zyx("file:///tmp/seg")
    assert result.shape == (2, 3, 4)
    assert result[0, 0, 0] == 0 and result[1, 2, 3] == 23


def test_one_chunk_plumbing_mode_requires_exact_identity():
    reference = np.array([[[0, 1], [1, 2]]], dtype=np.uint32)
    assert eq.plumbing(reference, reference.copy())["voi_total"] == 0.0
    changed = reference.copy()
    changed[0, 0, 1] = 7
    with pytest.raises(AssertionError, match="expected 0"):
        eq.plumbing(reference, changed)


def test_pins():
    assert EPS == 1e-7
    mapped = 1 / (1 + np.exp(-np.log(0.47 / 0.53) / 0.2))
    assert mapped == pytest.approx(0.35417863, abs=1e-8)
