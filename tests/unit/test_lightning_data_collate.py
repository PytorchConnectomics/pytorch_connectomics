from __future__ import annotations

import torch
import pytest

from connectomics.config import Config
from connectomics.inference.output import resolve_output_filenames
from connectomics.training.lightning.data import (
    ConnectomicsDataModule,
    DistributedEvaluationSampler,
    collate_dict,
)


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


def test_resolve_output_filenames_supports_string_image_paths_without_meta():
    cfg = Config()
    batch = {
        "image": [
            "/tmp/input_c.h5",
            "/tmp/input_d.tif",
        ],
    }

    assert resolve_output_filenames(cfg, batch, global_step=7) == ["input_c", "input_d"]


def test_distributed_evaluation_sampler_partitions_without_duplicates():
    dataset = list(range(10))

    rank0 = list(DistributedEvaluationSampler(dataset, rank=0, world_size=4))
    rank1 = list(DistributedEvaluationSampler(dataset, rank=1, world_size=4))
    rank2 = list(DistributedEvaluationSampler(dataset, rank=2, world_size=4))
    rank3 = list(DistributedEvaluationSampler(dataset, rank=3, world_size=4))

    combined = rank0 + rank1 + rank2 + rank3

    assert sorted(combined) == list(range(10))
    assert len(set(combined)) == 10


def test_test_dataloader_reuses_single_volume_for_distributed_tta_sharding(monkeypatch):
    monkeypatch.setattr(
        "connectomics.training.lightning.data._is_distributed_evaluation_active",
        lambda: True,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)

    datamodule = ConnectomicsDataModule(
        train_data_dicts=[],
        test_data_dicts=[{"image": torch.zeros((1, 8, 8, 8), dtype=torch.float32)}],
        batch_size=1,
        num_workers=0,
        distributed_tta_sharding=True,
    )

    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    assert not isinstance(dataloader.sampler, DistributedEvaluationSampler)
    assert len(dataloader) == 1


def test_test_dataloader_rejects_multi_volume_distributed_tta_sharding(monkeypatch):
    monkeypatch.setattr(
        "connectomics.training.lightning.data._is_distributed_evaluation_active",
        lambda: True,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 2)

    datamodule = ConnectomicsDataModule(
        train_data_dicts=[],
        test_data_dicts=[
            {"image": torch.zeros((1, 8, 8, 8), dtype=torch.float32)},
            {"image": torch.zeros((1, 8, 8, 8), dtype=torch.float32)},
        ],
        batch_size=1,
        num_workers=0,
        distributed_tta_sharding=True,
    )

    datamodule.setup(stage="test")

    with pytest.raises(RuntimeError, match="Distributed TTA sharding requires a single test sample"):
        datamodule.test_dataloader()
