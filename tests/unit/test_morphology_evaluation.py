"""Contracts for physical morphology evaluation without ground truth."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from connectomics.config import Config, load_config
from connectomics.evaluation import EvaluationContext, run_evaluation_stage
from connectomics.evaluation.morphology import write_morphology_artifacts
from connectomics.runtime.cache_resolver import _try_cache_only_intermediate_eval


def _config() -> Config:
    cfg = Config()
    cfg.evaluation.enabled = True
    cfg.evaluation.metrics = ["morphology"]
    cfg.evaluation.morphology.min_voxels = 4
    cfg.evaluation.morphology.component_min_voxels = 2
    cfg.data.test.resolution = [100.0, 40.0, 40.0]
    return cfg


def _segmentation() -> np.ndarray:
    seg: np.ndarray = np.zeros((12, 12, 12), dtype=np.uint64)
    seg[:, 4:7, 4:7] = 2**53 + 7
    return seg


def _context(cfg: Config, **kwargs) -> EvaluationContext:
    return EvaluationContext(
        cfg=cfg,
        evaluation_cfg=cfg.evaluation,
        inference_cfg=cfg.inference,
        **kwargs,
    )


def test_morphology_stage_runs_without_gt_and_converts_nm_resolution():
    cfg = _config()
    captured = {}
    result = run_evaluation_stage(
        _context(cfg, metrics_sink=lambda metrics: captured.update(metrics)),
        _segmentation(),
        None,
        filenames=["sample"],
        batch_idx=0,
    )

    assert result.computed is True
    analysis = captured["_morphology_analysis"]
    assert analysis.config.voxel_size_um == pytest.approx((0.1, 0.04, 0.04))
    assert captured["morphology_coverage"]["analyzed_label_count"] == 1
    assert captured["morphology_nerl"] is None
    assert "nerl" not in captured


def test_morphology_explicit_spacing_overrides_data_resolution():
    cfg = _config()
    cfg.evaluation.morphology.voxel_size_um = [0.2, 0.3, 0.4]
    captured = {}

    run_evaluation_stage(
        _context(cfg, metrics_sink=lambda metrics: captured.update(metrics)),
        _segmentation(),
        None,
        filenames=["sample"],
        batch_idx=0,
    )

    assert captured["_morphology_analysis"].config.voxel_size_um == (0.2, 0.3, 0.4)


def test_morphology_missing_spacing_fails_instead_of_reporting_unscaled_lengths():
    cfg = _config()
    cfg.data.test.resolution = None

    with pytest.raises(ValueError, match="voxel_size_um.*data.test.resolution"):
        run_evaluation_stage(
            _context(cfg),
            _segmentation(),
            None,
            filenames=["sample"],
            batch_idx=0,
        )


def test_morphology_stage_writes_json_csv_and_readable_report(tmp_path):
    cfg = _config()

    run_evaluation_stage(
        _context(cfg, output_path=tmp_path),
        _segmentation(),
        None,
        filenames=["sample"],
        batch_idx=0,
    )

    output_dir = tmp_path / "sample"
    summary = json.loads((output_dir / "eval_prediction_x1_morphology_summary.json").read_text())
    assert summary["config"]["voxel_size_um"] == [0.1, 0.04, 0.04]
    assert summary["volume_shape"] == [12, 12, 12]
    assert summary["summary"]["coverage"]["analyzed_label_count"] == 1
    assert summary["summary"]["nerl"] is None
    with (output_dir / "eval_prediction_x1_morphology_instances.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["label"] == str(2**53 + 7)
    assert json.loads(rows[0]["bbox_zyx"]) == [[0, 12], [4, 7], [4, 7]]
    assert len(json.loads(rows[0]["component_lengths_um"])) == 1
    report = (output_dir / "eval_prediction_x1.txt").read_text()
    assert "Ground-Truth-Free Morphology Analysis" in report
    assert "NERL requires ground truth" in report
    assert "eval_prediction_x1_morphology_instances.csv" in report


def test_morphology_empty_artifacts_keep_csv_header_and_json_nulls(tmp_path):
    cfg = _config()
    captured = {}
    run_evaluation_stage(
        _context(cfg, metrics_sink=lambda metrics: captured.update(metrics)),
        np.zeros((8, 8, 8), dtype=np.uint16),
        None,
        filenames=["empty"],
        batch_idx=0,
    )

    paths = write_morphology_artifacts(captured["_morphology_analysis"], tmp_path)

    with paths["instances"].open() as handle:
        reader = csv.DictReader(handle)
        assert "label" in reader.fieldnames
        assert list(reader) == []
    summary = json.loads(paths["summary"].read_text())
    assert summary["summary"]["nerl"] is None
    assert "NaN" not in paths["summary"].read_text()


def test_morphology_config_loads_known_fields_and_rejects_unknown_fields(tmp_path):
    path = tmp_path / "morphology.yaml"
    path.write_text(
        "evaluation:\n"
        "  enabled: true\n"
        "  metrics: [morphology]\n"
        "  morphology:\n"
        "    voxel_size_um: [0.1, 0.04, 0.04]\n"
        "    radius_bins_um: [0, 0.2, 0.5]\n"
        "    min_voxels: 20\n"
    )
    cfg = load_config(path)
    assert cfg.evaluation.morphology.voxel_size_um == [0.1, 0.04, 0.04]
    assert cfg.evaluation.morphology.radius_bins_um == [0, 0.2, 0.5]
    assert cfg.evaluation.morphology.min_voxels == 20

    path.write_text(path.read_text() + "    unknown_morphology_field: 1\n")
    with pytest.raises(Exception, match="unknown_morphology_field"):
        load_config(path)


def test_cached_predictions_dispatch_morphology_without_label_files(tmp_path, monkeypatch):
    cfg = _config()
    cfg.decoding.save_path = str(tmp_path / "results")
    path = tmp_path / "prediction.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=_segmentation())

    monkeypatch.setattr(
        "connectomics.decoding.run_decoding_stage",
        lambda _cfg, prediction: SimpleNamespace(has_decoding_config=False, decoded=prediction),
    )
    monkeypatch.setattr(
        "connectomics.inference.output.apply_prediction_transform",
        lambda _cfg, prediction: prediction,
    )

    assert _try_cache_only_intermediate_eval(cfg, [path], ["sample"], checkpoint_path=None) is True
    summary_path = tmp_path / "results" / "sample" / "eval_prediction_x1_morphology_summary.json"
    assert json.loads(summary_path.read_text())["summary"]["coverage"]["analyzed_label_count"] == 1
