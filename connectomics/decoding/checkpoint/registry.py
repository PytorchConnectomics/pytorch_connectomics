"""Explicit factory for independent checkpoint passes."""

from __future__ import annotations

from typing import Any, Mapping

from .operators.base import CheckpointOperator
from .operators.nucleus_anchor import NucleusAnchorConfig, NucleusAnchorOperator


def create_operator(spec: Mapping[str, Any], *, min_share: float) -> CheckpointOperator:
    name = spec.get("operator")
    if name != "nucleus_anchor":
        raise ValueError(f"unknown or unimplemented checkpoint operator {name!r}")
    return NucleusAnchorOperator(NucleusAnchorConfig.from_spec(spec, min_share=min_share))


def list_operators() -> tuple[str, ...]:
    return ("nucleus_anchor",)


__all__ = ["create_operator", "list_operators"]
