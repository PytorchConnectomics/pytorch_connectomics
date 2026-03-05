from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class NNUNetConfig:
    """nnUNet pretrained model loading configuration."""

    checkpoint: Optional[str] = None
    plans: Optional[str] = None
    dataset: Optional[str] = None
    device: str = "cuda"
    spatial_dims: Optional[int] = None  # Explicitly set 2 or 3 when auto-detection is ambiguous
