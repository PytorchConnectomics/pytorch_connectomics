import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from connectomics.config import load_config
from connectomics.decoding.error_correction.artifacts import (
    load_frozen_merge_roots,
    sha256_file,
)
from connectomics.decoding.error_correction.postprocess import correct_block
from connectomics.decoding.error_correction.sizes import (
    SIZE_DTYPE,
    aggregate_size_files,
    load_size_inventory,
)
from connectomics.decoding.error_correction.workflow import (
    ErrorCorrectionConfig,
    stage_commands,
)
from connectomics.utils.yaml_config import load_yaml_with_bases_and_params


def _write_frozen_proposal(tmp_path: Path) -> tuple[Path, Path]:
    proposals = tmp_path / "frozen.npz"
    np.savez_compressed(
        proposals,
        left=np.asarray([20, 30], dtype=np.uint64),
        right=np.asarray([10, 20], dtype=np.uint64),
        gt_free=np.asarray(True),
        frozen_before_evaluation=np.asarray(True),
    )
    report = tmp_path / "frozen.json"
    report.write_text(json.dumps({"proposal_sha256": sha256_file(proposals)}))
    return proposals, report


def test_frozen_proposal_builds_transitive_minimum_roots(tmp_path: Path):
    proposals, report = _write_frozen_proposal(tmp_path)

    labels, roots, _ = load_frozen_merge_roots(proposals, report)

    np.testing.assert_array_equal(labels, [10, 20, 30])
    np.testing.assert_array_equal(roots, [10, 10, 10])


def test_frozen_proposal_rejects_report_hash_drift(tmp_path: Path):
    proposals, report = _write_frozen_proposal(tmp_path)
    report.write_text(json.dumps({"proposal_sha256": "0" * 64}))

    with pytest.raises(ValueError, match="hash differs"):
        load_frozen_merge_roots(proposals, report)


def test_union_is_applied_before_inter_object_erosion():
    segmentation = np.full((3, 3, 4), np.uint64(2**53 + 10), dtype=np.uint64)
    segmentation[:, :, 2:] = np.uint64(2**53 + 20)
    source = np.asarray([2**53 + 10, 2**53 + 20], dtype=np.uint64)
    roots = np.asarray([2**53 + 10, 2**53 + 10], dtype=np.uint64)

    without_union = correct_block(
        segmentation, np.zeros(0, np.uint64), np.zeros(0, np.uint64), (1, 1, 1)
    )
    with_union = correct_block(segmentation, source, roots, (1, 1, 1))

    assert np.count_nonzero(without_union == 0) > 0
    assert np.count_nonzero(with_union == 0) == 0
    assert np.all(with_union == np.uint64(2**53 + 10))


def test_size_inventory_aggregates_duplicate_labels(tmp_path: Path):
    first = np.asarray([(1, 3), (2, 4)], dtype=SIZE_DTYPE)
    second = np.asarray([(1, 5), (3, 6)], dtype=SIZE_DTYPE)
    first.tofile(tmp_path / "seg_size_a.data")
    second.tofile(tmp_path / "seg_size_b.data")
    output = tmp_path / "all.data"

    report = aggregate_size_files(str(tmp_path / "seg_size_*.data"), output)
    result = load_size_inventory(output)

    assert report["gt_free"] is True
    np.testing.assert_array_equal(result["label"], [1, 2, 3])
    np.testing.assert_array_equal(result["size"], [8, 4, 6])


def test_tutorial_config_has_exhaustive_gt_free_scope():
    config_path = Path("tutorials/neuron_j0126/4_error_correction.yaml")
    config = ErrorCorrectionConfig.load(config_path)

    commands = stage_commands(
        config,
        "junction_scope",
        task_id=0,
        num_tasks=80,
        overwrite=False,
        max_owned_chunks=None,
    )
    command = " ".join(commands[0]).lower()
    assert "decoder-scope" in command
    assert "--min-affinity 0.05" in command
    assert "test_50" not in command
    assert "oracle" not in command
    assert "ffn" not in command


def test_tutorial_params_are_inherited_and_fully_resolved():
    affinity = load_config("tutorials/neuron_j0126/2_infer.yaml")
    abiss = load_yaml_with_bases_and_params(Path("tutorials/neuron_j0126/3_abiss.yaml"))
    correction = load_yaml_with_bases_and_params(Path("tutorials/neuron_j0126/4_error_correction.yaml"))

    assert affinity.save_path.endswith("outputs/neuron_j0126/affinity")
    assert affinity.test.data.test.image.endswith("j0126_em.zarr/main")
    # No input may point outside what step 0 downloads or the pipeline writes.
    for key in ("segmentation", "affinity_chunks", "keep_mask", "nucleus_manifest", "size_glob"):
        value = correction["error_correction"][key]
        assert "dev/zebrafinch" not in value, (key, value)
    assert "params" not in abiss
    assert "${params" not in str(abiss)
    assert "${params" not in str(correction)
    assert abiss["abiss_chunk"]["workdir"].endswith("outputs/neuron_j0126/abiss/run")
    assert correction["error_correction"]["workdir"].endswith(
        "outputs/neuron_j0126/error_correction_v7"
    )


def test_config_rejects_unknown_and_evaluation_inputs(tmp_path: Path):
    original = yaml.safe_load(Path("tutorials/neuron_j0126/4_error_correction.yaml").read_text())
    # The copy keeps `_base_: params.yaml`, which resolves next to the config being
    # loaded, so params.yaml has to travel with it into tmp_path.
    shutil.copy("tutorials/neuron_j0126/params.yaml", tmp_path / "params.yaml")
    original["error_correction"]["surprise"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match="unknown error_correction keys"):
        ErrorCorrectionConfig.load(path)

    original["error_correction"].pop("surprise")
    original["error_correction"]["segmentation"] = "/data/test_50/seg"
    path.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match="evaluation/GT path"):
        ErrorCorrectionConfig.load(path)
