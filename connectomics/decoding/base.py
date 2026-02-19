"""Core decoding type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import numpy as np


class DecodeFunction(Protocol):
    """Callable decoder signature used by the registry."""

    def __call__(self, predictions: np.ndarray, **kwargs: Any) -> np.ndarray: ...


@dataclass
class DecodeStep:
    """Single step in a decoding pipeline."""

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

