"""Core decoding type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

import numpy as np


class DecodeFunction(Protocol):
    """Callable decoder signature used by the registry."""

    def __call__(self, predictions: np.ndarray, **kwargs: Any) -> np.ndarray: ...


class GraphOp(Protocol):
    """Callable decoder-graph operation signature used by the registry."""

    def __call__(self, inputs: List[np.ndarray], **kwargs: Any) -> np.ndarray: ...


@dataclass
class DecodeStep:
    """Single step in a decoding pipeline."""

    enabled: bool = True
    name: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeNode:
    """Single named node in a decoding graph."""

    enabled: bool = True
    name: str = ""
    op: str = ""
    inputs: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeGraph:
    """Decoder graph with one declared output node."""

    nodes: List[DecodeNode] = field(default_factory=list)
    output: str = ""
