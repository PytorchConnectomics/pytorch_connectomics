"""Tests for the per-mode + per-volume output path resolution logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from connectomics.config import Config
from connectomics.config.schema.stages import TuneConfig
from connectomics.runtime.checkpoint_dispatch import configure_checkpoint_output_paths
from connectomics.runtime.output_naming import (
    final_prediction_output_tag,
    intermediate_decode_step_output_tag,
    intermediate_prediction_cache_suffix,
    is_raw_cache_suffix,
    raw_cache_suffix,
    resolve_dataset_volume_stems,
    resolve_volume_save_dir,
)


def _make_args(checkpoint: str, mode: str = "test"):
    return argparse.Namespace(mode=mode, checkpoint=checkpoint)


def test_test_mode_save_path_derives_from_checkpoint(tmp_path):
    """In test mode, configure_checkpoint_output_paths writes a results_step=<NNN> dir
    under the checkpoint's run directory."""
    run_dir = tmp_path / "outputs" / "exp" / "20260101_000000"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "step=00200000.ckpt"
    ckpt_path.touch()

    cfg = Config()
    args = _make_args(str(ckpt_path), mode="test")

    configure_checkpoint_output_paths(args, cfg)

    assert cfg.inference.save_path == str(run_dir / "results_step=00200000")


def test_tune_mode_save_paths_derive_from_checkpoint(tmp_path):
    """Tune mode populates both tune.save_path and tune.save_predictions_path."""
    run_dir = tmp_path / "outputs" / "exp" / "20260101_000000"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "step=00050000.ckpt"
    ckpt_path.touch()

    cfg = Config()
    cfg.tune = TuneConfig()
    args = _make_args(str(ckpt_path), mode="tune")

    configure_checkpoint_output_paths(args, cfg)

    assert cfg.tune.save_path == str(run_dir / "tuning_step=00050000")
    assert cfg.tune.save_predictions_path == str(
        run_dir / "tuning_step=00050000" / "predictions"
    )
    assert cfg.inference.save_path == str(run_dir / "results_step=00050000")


def test_resolve_dataset_volume_stems_explicit_string():
    cfg = Config()
    cfg.data.test.path = "/data"
    cfg.data.test.image = "/data/seed101/img.h5"
    cfg.data.test.name = "alpha"

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["alpha"]


def test_resolve_dataset_volume_stems_explicit_list():
    cfg = Config()
    cfg.data.test.path = "/data"
    cfg.data.test.image = ["/data/a/img.h5", "/data/b/img.h5"]
    cfg.data.test.name = ["alpha", "beta"]

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["alpha", "beta"]


def test_resolve_dataset_volume_stems_uninformative_stem_falls_back_to_parent(tmp_path):
    """When the filename stem is uninformative (e.g. `img`), use the parent dir."""
    base = tmp_path / "datasets"
    (base / "seed0").mkdir(parents=True)
    (base / "seed0" / "img.h5").touch()
    (base / "seed1").mkdir(parents=True)
    (base / "seed1" / "img.h5").touch()

    cfg = Config()
    cfg.data.test.path = ""
    cfg.data.test.image = [
        str(base / "seed0" / "img.h5"),
        str(base / "seed1" / "img.h5"),
    ]

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["seed0", "seed1"]


def test_resolve_dataset_volume_stems_with_path_equal_to_volume_dir(tmp_path):
    """Regression for review_v2 finding 1: resolver and writer must agree
    when ``data.test.path`` is the volume identity directory and
    ``data.test.image`` is inside a container."""
    base = tmp_path / "nisb" / "seed101"
    (base / "data.zarr").mkdir(parents=True)
    (base / "data.zarr" / "img").touch()

    cfg = Config()
    cfg.data.test.path = str(base)
    cfg.data.test.image = str(base / "data.zarr" / "img")

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["seed101"]


def test_resolve_dataset_volume_stems_skips_zarr_container_parent(tmp_path):
    """``data.zarr/img`` paths must resolve to the seed dir, not ``data.zarr``."""
    from connectomics.runtime.output_naming import _stem_from_image_path

    # Direct helper test (regression for review_v1 finding 2).
    assert _stem_from_image_path("/data/seed101/data.zarr/img") == "seed101"
    assert _stem_from_image_path("/data/seed102/data.zarr/img") == "seed102"
    assert _stem_from_image_path("/data/seed101/data.n5/img") == "seed101"
    assert _stem_from_image_path("/data/seed101/dataset.ome.zarr/img") == "seed101"

    # End-to-end through resolve_dataset_volume_stems.
    base = tmp_path / "nisb"
    (base / "seed101" / "data.zarr").mkdir(parents=True)
    (base / "seed102" / "data.zarr").mkdir(parents=True)
    (base / "seed101" / "data.zarr" / "img").touch()
    (base / "seed102" / "data.zarr" / "img").touch()

    cfg = Config()
    cfg.data.test.path = ""
    cfg.data.test.image = [
        str(base / "seed101" / "data.zarr" / "img"),
        str(base / "seed102" / "data.zarr" / "img"),
    ]

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["seed101", "seed102"]


def test_resolve_dataset_volume_stems_decode_only_uses_load_path_parent():
    """In decode-only flow, volume stem comes from the load_prediction_path's parent."""
    cfg = Config()
    cfg.decoding.load_prediction_path = (
        "/scratch/results_step=00200000/seed101/raw_x1.h5"
    )

    stems = resolve_dataset_volume_stems(cfg, mode="test")
    assert stems == ["seed101"]


def test_resolve_dataset_volume_stems_tune_uses_val():
    """In tune mode, the resolver reads data.val.image (not data.test.image)."""
    cfg = Config()
    cfg.data.val.image = "/data/seedX/img.h5"
    cfg.data.val.name = "tune-alpha"

    stems = resolve_dataset_volume_stems(cfg, mode="tune")
    assert stems == ["tune-alpha"]


def test_resolve_volume_save_dir_joins_inference_save_path(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path / "results_step=00200000")

    out = resolve_volume_save_dir(cfg, mode="test", volume_stem="seed101")
    assert out == Path(str(tmp_path / "results_step=00200000")) / "seed101"


def test_raw_cache_suffix_format():
    """raw_cache_suffix returns the canonical per-volume artifact filename."""
    cfg = Config()
    suffix = raw_cache_suffix(cfg)
    assert suffix == "raw_x1.h5"
    assert is_raw_cache_suffix(suffix)


def test_raw_cache_suffix_with_channel_selector():
    """Channel selector flows into the raw filename; checkpoint does not."""
    cfg = Config()
    cfg.inference.model.select_channel = [0, 1, 2]
    suffix = raw_cache_suffix(cfg, checkpoint_path="/x/step=00050000.ckpt")
    # No `_ckpt-` token; channel encoded in filename.
    assert "ckpt" not in suffix
    assert suffix.endswith(".h5")
    assert "ch0-1-2" in suffix


def test_intermediate_prediction_cache_suffix_chunked():
    """Chunked-raw mode appends a token but keeps the raw_ prefix."""
    cfg = Config()
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [1000, 1000, 1350]
    suffix = intermediate_prediction_cache_suffix(cfg)
    assert suffix.startswith("raw_x1")
    assert "chunked-raw" in suffix
    assert suffix.endswith(".h5")


def test_final_prediction_output_tag_drops_ckpt_and_stem():
    """Per-volume layout: filename has no ``_ckpt-`` and no leading volume stem."""
    cfg = Config()
    cfg.decoding.steps = [{"name": "decode_affinity_cc", "kwargs": {"threshold": 0.75, "backend": "numba", "edge_offset": 0}}]
    tag = final_prediction_output_tag(cfg, checkpoint_path="/x/step=00050000.ckpt")
    assert tag.startswith("decoded_x1")
    assert "ckpt" not in tag
    assert tag.endswith(".h5")


def test_final_prediction_output_tag_uses_prediction_label_when_no_decoder():
    cfg = Config()
    tag = final_prediction_output_tag(cfg)
    assert tag.startswith("prediction_x1")
    assert tag.endswith(".h5")


def test_intermediate_decode_step_output_tag_format():
    cfg = Config()
    step = {"name": "decode_affinity_cc", "kwargs": {"threshold": 0.5}}
    tag = intermediate_decode_step_output_tag(cfg, step)
    assert tag.startswith("decoded_x1")
    assert tag.endswith(".h5")
    assert "affinity_cc" in tag
