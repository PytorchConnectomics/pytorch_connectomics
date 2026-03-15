"""
PyTorch Lightning DataModules for connectomics datasets.

Provides ConnectomicsDataModule (MONAI transform-based) and
SimpleDataModule (wraps pre-built dataloaders).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import CacheDataset, Dataset
from monai.transforms import Compose
from torch.utils.data import DataLoader, Sampler


class ConnectomicsDataModule(pl.LightningDataModule):
    """
    Lightning DataModule using MONAI Dataset/CacheDataset.

    Used as a fallback when pre-loaded cache is not enabled.
    Transforms (including loading and cropping) are applied on-the-fly.

    Args:
        train_data_dicts: Training data dictionaries.
        val_data_dicts: Validation data dictionaries.
        test_data_dicts: Test data dictionaries.
        transforms: Dict of Compose for 'train'/'val'/'test'.
        dataset_type: 'standard' or 'cached'.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        cache_rate: Cache rate for CacheDataset.
        val_steps_per_epoch: Override validation dataset length.
        seed: Random seed for validation reseeding.
        distributed_tta_sharding: Keep test samples replicated on all ranks so
            TTA passes can be partitioned inside inference rather than by sampler.
        **dataset_kwargs: Extra args (iter_num, sample_size, etc.).
    """

    def __init__(
        self,
        train_data_dicts: List[Dict[str, Any]],
        val_data_dicts: Optional[List[Dict[str, Any]]] = None,
        test_data_dicts: Optional[List[Dict[str, Any]]] = None,
        transforms: Optional[Dict[str, Compose]] = None,
        dataset_type: str = "standard",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        cache_rate: float = 1.0,
        val_steps_per_epoch: Optional[int] = None,
        seed: int = 0,
        distributed_tta_sharding: bool = False,
        **dataset_kwargs,
    ):
        super().__init__()
        self.train_data_dicts = train_data_dicts
        self.val_data_dicts = val_data_dicts
        self.test_data_dicts = test_data_dicts
        self.skip_validation = not val_data_dicts or len(val_data_dicts) == 0
        self.transforms = transforms or {}
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cache_rate = cache_rate
        self.val_steps_per_epoch = val_steps_per_epoch
        self.seed = seed
        self.distributed_tta_sharding = distributed_tta_sharding
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            if self.train_data_dicts:
                self.train_dataset = self._create_dataset(
                    self.train_data_dicts,
                    self.transforms.get("train"),
                    "train",
                )
            if self.val_data_dicts:
                self.val_dataset = self._create_dataset(
                    self.val_data_dicts,
                    self.transforms.get("val"),
                    "val",
                )
        if stage in ("test", None):
            if self.test_data_dicts:
                self.test_dataset = self._create_dataset(
                    self.test_data_dicts,
                    self.transforms.get("test"),
                    "test",
                )

    def _create_dataset(self, data_dicts, transforms, mode):
        iter_num = self.dataset_kwargs.get("iter_num", -1)
        if mode == "val" and self.val_steps_per_epoch is not None:
            iter_num = self.val_steps_per_epoch

        if self.dataset_type == "cached":
            ds = CacheDataset(
                data=data_dicts,
                transform=transforms,
                cache_rate=self.cache_rate,
            )
        else:
            ds = Dataset(data=data_dicts, transform=transforms)

        if iter_num and iter_num > 0:
            ds = _IterNumDataset(ds, iter_num)
        return ds

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.skip_validation:
            return []
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        sampler = None
        if self.test_dataset is not None and _distributed_world_size() > 1:
            if self.distributed_tta_sharding:
                if len(self.test_dataset) != 1:
                    raise RuntimeError(
                        "Distributed TTA sharding requires a single test sample per rank. "
                        "Disable inference.test_time_augmentation.distributed_sharding for "
                        "multi-volume test datasets."
                    )
            else:
                sampler = DistributedEvaluationSampler(self.test_dataset)
        return self._create_dataloader(
            self.test_dataset,
            shuffle=False,
            collate_fn=collate_dict_list,
            sampler=sampler,
        )

    def _create_dataloader(self, dataset, shuffle, collate_fn=None, sampler=None):
        if dataset is None:
            return None
        if collate_fn is None:
            collate_fn = collate_dict
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            collate_fn=collate_fn,
        )


class SimpleDataModule(pl.LightningDataModule):
    """Wraps pre-built train/val/test dataloaders."""

    def __init__(
        self,
        train_loader=None,
        val_loader=None,
        test_loader=None,
    ):
        super().__init__()
        self._train = train_loader
        self._val = val_loader
        self._test = test_loader

    def train_dataloader(self):
        return self._train

    def val_dataloader(self):
        return self._val if self._val is not None else []

    def test_dataloader(self):
        return self._test if self._test is not None else []


class _IterNumDataset(torch.utils.data.Dataset):
    """Wraps a dataset to override __len__ with iter_num.

    Uses modulo indexing: when iter_num > len(dataset), indices wrap around
    to the beginning of the underlying dataset.
    """

    def __init__(self, dataset, iter_num: int):
        self.dataset = dataset
        self._len = iter_num

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


def _is_distributed_evaluation_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _distributed_world_size() -> int:
    if not _is_distributed_evaluation_active():
        return 1
    return int(torch.distributed.get_world_size())


class DistributedEvaluationSampler(Sampler[int]):
    """Shard evaluation samples across DDP ranks without padding or duplication."""

    def __init__(
        self,
        dataset,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if rank is None or world_size is None:
            if not _is_distributed_evaluation_active():
                raise RuntimeError(
                    "DistributedEvaluationSampler requires an initialized distributed process "
                    "group or explicit rank/world_size."
                )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got {rank}/{world_size}.")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.indices = list(range(len(dataset)))[self.rank :: self.world_size]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_dict(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stack dict-of-arrays batch into dict-of-tensors."""
    if not batch:
        return {}
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.stack([torch.from_numpy(v) for v in values])
        else:
            result[key] = values
    return result


def collate_dict_list(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Preserve per-sample values as lists for variable-shape test volumes."""
    if not batch:
        return {}

    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], np.ndarray):
            result[key] = [torch.from_numpy(v) for v in values]
        else:
            result[key] = values
    return result


__all__ = [
    "ConnectomicsDataModule",
    "DistributedEvaluationSampler",
    "SimpleDataModule",
    "collate_dict",
    "collate_dict_list",
]
