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
