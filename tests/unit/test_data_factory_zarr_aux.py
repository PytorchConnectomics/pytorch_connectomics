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
