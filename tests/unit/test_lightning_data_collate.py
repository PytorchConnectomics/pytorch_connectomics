from __future__ import annotations

import torch

from connectomics.config import Config
from connectomics.inference.output import resolve_output_filenames
from connectomics.training.lightning.data import ConnectomicsDataModule, collate_dict


def test_test_dataloader_preserves_variable_shape_tensors():
    datamodule = ConnectomicsDataModule(
        train_data_dicts=[],
        test_data_dicts=[
            {"image": torch.zeros((1, 51, 516, 516), dtype=torch.float32)},
            {"image": torch.zeros((1, 50, 256, 256), dtype=torch.float32)},
        ],
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="test")
    batch = next(iter(datamodule.test_dataloader()))

    assert isinstance(batch["image"], list)
    assert [tuple(t.shape) for t in batch["image"]] == [
        (1, 51, 516, 516),
        (1, 50, 256, 256),
    ]


def test_train_collate_still_stacks_equal_shape_tensors():
    batch = collate_dict(
        [
            {"image": torch.zeros((1, 4, 4, 4), dtype=torch.float32)},
            {"image": torch.ones((1, 4, 4, 4), dtype=torch.float32)},
        ]
    )

    assert tuple(batch["image"].shape) == (2, 1, 4, 4, 4)


def test_resolve_output_filenames_supports_list_collated_images():
    cfg = Config()
    batch = {
        "image": [
            torch.zeros((1, 4, 4, 4), dtype=torch.float32),
            torch.zeros((1, 5, 6, 6), dtype=torch.float32),
        ],
        "image_meta_dict": [
            {"filename_or_obj": "/tmp/input_a.h5"},
            {"filename_or_obj": "/tmp/input_b.h5"},
        ],
    }

    assert resolve_output_filenames(cfg, batch, global_step=11) == ["input_a", "input_b"]
