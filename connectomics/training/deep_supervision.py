"""Compatibility shim for legacy deep supervision imports.

Loss orchestration logic now lives in :mod:`connectomics.training.loss_orchestrator`.
This module re-exports the legacy symbols to avoid breaking downstream imports.
"""

from .loss_orchestrator import DeepSupervisionHandler, LossOrchestrator, match_target_to_output

__all__ = ["DeepSupervisionHandler", "LossOrchestrator", "match_target_to_output"]
