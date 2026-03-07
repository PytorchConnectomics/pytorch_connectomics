from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemConfig:
    """System configuration for hardware, workers, and reproducibility."""

    profile: Optional[str] = None
    num_gpus: int = 1
    num_workers: int = 8
    seed: int = 42
