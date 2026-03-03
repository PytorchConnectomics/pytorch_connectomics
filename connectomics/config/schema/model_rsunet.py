from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RSUNetConfig:
    """RSUNet architecture-specific configuration."""

    width: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    norm: str = "batch"
    activation: str = "relu"
    num_groups: int = 8
    down_factors: Optional[List[List[int]]] = None
    depth_2d: int = 0
    kernel_2d: List[int] = field(default_factory=lambda: [1, 3, 3])
    act_negative_slope: float = 0.01
    act_init: float = 0.25
