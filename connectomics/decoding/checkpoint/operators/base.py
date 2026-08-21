"""Operator protocol for independent checkpoint passes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schema import CheckpointPlan, DescriptionBundle


class CheckpointOperator(Protocol):
    name: str
    version: str

    def describe(self, output_dir: Path) -> DescriptionBundle: ...

    def plan(self, description: DescriptionBundle) -> CheckpointPlan: ...


__all__ = ["CheckpointOperator"]
