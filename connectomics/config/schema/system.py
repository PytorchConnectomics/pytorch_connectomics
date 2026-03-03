from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemConfig:
    """System configuration for hardware, workers, and reproducibility."""

    profile: Optional[str] = None
    num_gpus: Optional[int] = None
    num_workers: Optional[int] = None
    seed: Optional[int] = None
