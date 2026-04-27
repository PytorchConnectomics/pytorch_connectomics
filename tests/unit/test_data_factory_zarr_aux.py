from pathlib import Path

import numpy as np

from connectomics.config import Config
from connectomics.data.io import save_volume
from connectomics.training.lightning.data_factory import _maybe_precompute_label_aux


def test_maybe_precompute_label_aux_reuses_existing_zarr_cache(monkeypatch, tmp_path: Path):
    label_path = tmp_path / "data.zarr" / "seg"
    aux_path = tmp_path / "data.zarr" / "seg_skeleton"
    save_volume(str(label_path), np.zeros((2, 2, 2), dtype=np.uint16), file_format="zarr")
    save_volume(str(aux_path), np.ones((2, 2, 2), dtype=np.uint16), file_format="zarr")

    cfg = Config()
    cfg.data.label_transform.targets = [{"name": "skeleton_aware_edt", "kwargs": {}}]
    cfg.data.train.label_aux_type = "skeleton"

    def fail(*args, **kwargs):
        raise AssertionError("existing zarr cache should be reused")

    monkeypatch.setattr(
        "connectomics.data.processing.distance.precompute_skeleton_volume",
        fail,
    )

    paths = _maybe_precompute_label_aux(
        cfg,
        cfg.data.train,
        [str(label_path)],
        split_name="train",
    )

    assert paths == [str(aux_path)]


def test_maybe_precompute_label_aux_uses_configured_cache_dir(monkeypatch, tmp_path: Path):
    label_path = tmp_path / "labels.h5"
    save_volume(str(label_path), np.zeros((2, 2, 2), dtype=np.uint16), file_format="h5")

    cache_dir = tmp_path / "cache"
    expected_aux = cache_dir / "labels_skeleton.h5"

    cfg = Config()
    cfg.data.label_transform.targets = [{"name": "skeleton_aware_edt", "kwargs": {}}]
    cfg.data.label_transform.cache_dir = str(cache_dir)
    cfg.data.train.label_aux_type = "skeleton"

    def fake_precompute(label_path_arg, output_path_arg, resolution):
        assert label_path_arg == str(label_path)
        assert output_path_arg == str(expected_aux)
        expected_aux.parent.mkdir(parents=True, exist_ok=True)
        save_volume(output_path_arg, np.ones((2, 2, 2), dtype=np.uint16), file_format="h5")
        return output_path_arg

    monkeypatch.setattr(
        "connectomics.data.processing.distance.precompute_skeleton_volume",
        fake_precompute,
    )

    paths = _maybe_precompute_label_aux(
        cfg,
        cfg.data.train,
        [str(label_path)],
        split_name="train",
    )

    assert paths == [str(expected_aux)]
