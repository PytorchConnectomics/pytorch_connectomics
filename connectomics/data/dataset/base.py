"""Shared base class for patch-sampling datasets."""

from __future__ import annotations

import logging
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from .crop_sampling import center_crop_position, random_crop_position

logger = logging.getLogger(__name__)


class PatchDataset(torch.utils.data.Dataset):
    """
    Abstract base for datasets that sample random patches from volumes.

    Subclasses must implement:
        _crop_volumes(vol_idx, pos) -> dict with "image" and optional "label"/"mask"
        _has_labels(vol_idx) -> bool

    Subclasses must populate ``self.volume_sizes`` during __init__.

    Provides:
        - __getitem__ with foreground-aware retry loop
        - set_epoch / get_sampling_fingerprint for validation reseeding
        - Shared crop position sampling via crop_sampling.py
    """

    def __init__(
        self,
        patch_size: Tuple[int, ...],
        iter_num: int = 500,
        transforms: Optional[Compose] = None,
        mode: str = "train",
        max_attempts: int = 10,
        foreground_threshold: float = 0.0,
    ):
        super().__init__()

        ndim = len(patch_size)
        if ndim not in (2, 3):
            raise ValueError(f"patch_size must be 2D or 3D, got {ndim}D")
        self.patch_size = ensure_tuple_rep(patch_size, ndim)
        self.iter_num = iter_num
        self.transforms = transforms
        self.mode = mode
        self.max_attempts = max_attempts
        self.foreground_threshold = foreground_threshold

        # Validation reseeding support
        self.base_seed = 0
        self.current_epoch = 0

        # Subclass must populate this during __init__
        self.volume_sizes: List[Tuple[int, ...]] = []

    @property
    def num_volumes(self) -> int:
        return len(self.volume_sizes)

    @abstractmethod
    def _crop_volumes(self, vol_idx: int, pos: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Crop image/label/mask from volume at given position.

        Returns dict with "image" (required), "label" and "mask" (optional, None if absent).
        Values are numpy arrays with channel dim: (C, ...).
        """
        ...

    @abstractmethod
    def _has_labels(self, vol_idx: int) -> bool:
        """Whether the volume at vol_idx has associated labels."""
        ...

    def __len__(self) -> int:
        return self.iter_num

    def __getitem__(self, index: int) -> Dict[str, Any]:
        vol_idx = random.randint(0, self.num_volumes - 1)

        use_fg = (
            self.mode == "train"
            and self._has_labels(vol_idx)
            and self.foreground_threshold > 0
        )

        data = None
        if use_fg:
            for _ in range(self.max_attempts):
                pos = self._get_random_crop_position(vol_idx)
                data = self._crop_volumes(vol_idx, pos)

                label = data.get("label")
                if label is None:
                    break  # no label => no foreground filtering needed

                fg_frac = float((label > 0).sum()) / float(label.size)

                # Reject if mask is present but entirely zero in this crop
                mask = data.get("mask")
                if mask is not None and not (mask > 0).any():
                    continue

                if fg_frac >= self.foreground_threshold:
                    break
            # If loop exhausted without break, data holds the last attempt

        if data is None:
            # Either use_fg was False, or max_attempts==0
            pos = (
                self._get_random_crop_position(vol_idx)
                if self.mode == "train"
                else self._get_center_crop_position(vol_idx)
            )
            data = self._crop_volumes(vol_idx, pos)

        # Remove None values so downstream code doesn't see phantom entries
        data = {k: v for k, v in data.items() if v is not None}

        if self.transforms:
            data = self.transforms(data)

        return data

    # -- Crop position helpers (overridable by subclasses) --

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        return random_crop_position(
            self.volume_sizes[vol_idx], self.patch_size, rng=random
        )

    def _get_center_crop_position(self, vol_idx: int) -> Tuple[int, ...]:
        return center_crop_position(self.volume_sizes[vol_idx], self.patch_size)

    # -- Validation reseeding --

    def set_epoch(self, epoch: int, base_seed: int = 0):
        """Set epoch for deterministic validation reseeding."""
        if self.mode == "val":
            self.base_seed = base_seed
            self.current_epoch = epoch
            effective_seed = self.base_seed + epoch
            random.seed(effective_seed)
            logger.debug(
                "[Validation] epoch=%s, effective_seed=%s, dataset=%s@%s",
                epoch, effective_seed, type(self).__name__, id(self),
            )

    def get_sampling_fingerprint(self, num_samples: int = 5) -> str:
        """Generate fingerprint of validation sampling for verification."""
        if self.mode != "val":
            return "N/A (training mode)"
        state = random.getstate()
        try:
            samples = []
            for _ in range(num_samples):
                vol_idx = random.randint(0, self.num_volumes - 1)
                pos = self._get_random_crop_position(vol_idx)
                samples.append((vol_idx, pos))
            return ", ".join(f"v{v}@{p}" for v, p in samples)
        finally:
            random.setstate(state)


__all__ = ["PatchDataset"]
