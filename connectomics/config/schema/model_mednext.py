from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MedNeXtConfig:
    """MedNeXt architecture-specific configuration."""

    size: str = "S"
    base_channels: int = 32
    exp_r: Any = 4
    kernel_size: int = 3
    do_res: bool = True
    do_res_up_down: bool = True
    block_counts: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2, 2, 2, 2])
    checkpoint_style: Optional[str] = None
    norm: str = "group"
    dim: str = "3d"
    grn: bool = False
