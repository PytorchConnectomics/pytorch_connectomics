from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from connectomics.config import Config
from connectomics.inference.stage import run_prediction_inference


class _DummyManager:
    def __init__(self, cfg: Config, prediction: torch.Tensor):
        self.cfg = cfg
        self.prediction = prediction
        self.observed = {}

    def predict_with_tta(
        self,
        images,
        *,
        mask=None,
        mask_align_to_image=False,
        requested_head=None,
    ):
        self.observed = {
            "images_shape": tuple(images.shape),
            "mask": mask,
            "mask_align_to_image": mask_align_to_image,
            "requested_head": requested_head,
        }
        return self.prediction


def test_run_prediction_inference_writes_raw_artifact_metadata(tmp_path):
    cfg = Config()
    cfg.model.arch.type = "mednext"
    cfg.data.data_transform.val_transpose = [2, 1, 0]
    cfg.inference.save_prediction.storage_dtype = "float16"
    cfg.inference.select_channel = [0, 1]

    prediction = torch.arange(1 * 2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    manager = _DummyManager(cfg, prediction)
    output_path = tmp_path / "raw_prediction.h5"

    returned = run_prediction_inference(
        manager,
        torch.zeros(1, 1, 3, 4, 5),
        requested_head="affinity",
        output_path=output_path,
        image_path="input.h5",
        checkpoint_path="checkpoint.ckpt",
        input_shape=(3, 4, 5),
    )

    assert returned is prediction
    assert manager.observed["requested_head"] == "affinity"

    with h5py.File(output_path, "r") as handle:
        dataset = handle["main"]
        assert dataset.shape == (2, 3, 4, 5)
        assert dataset.dtype == np.dtype("float16")
        assert dataset.attrs["image_path"] == "input.h5"
        assert dataset.attrs["checkpoint_path"] == "checkpoint.ckpt"
        assert dataset.attrs["output_head"] == "affinity"
        assert dataset.attrs["model_architecture"] == "mednext"
        assert dataset.attrs["model_output_identity"] == "head=affinity;select_channel=[0, 1]"
        assert bool(dataset.attrs["decode_after_inference"]) is True
        assert json.loads(dataset.attrs["transpose"]) == [2, 1, 0]
        assert json.loads(dataset.attrs["final_shape"]) == [3, 4, 5]
