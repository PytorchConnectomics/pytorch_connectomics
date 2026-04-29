"""Standalone decoding stage helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pipeline import apply_decode_mode, resolve_decode_modes_from_cfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodingStageResult:
    """Output from a decoding stage execution."""

    decoded: np.ndarray
    postprocessed: np.ndarray
    has_decoding_config: bool
    duration_s: float


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def apply_decoding_postprocessing(cfg: Any, data: np.ndarray) -> np.ndarray:
    """Apply postprocessing configured under ``cfg.inference.postprocessing``.

    The postprocessing transforms operate on decoded prediction arrays, so the
    implementation lives with decoding even though the config section remains
    under inference for now.
    """
    inference_cfg = _cfg_get(cfg, "inference", None)
    postprocessing = _cfg_get(inference_cfg, "postprocessing", None)
    if not _cfg_get(postprocessing, "enabled", False):
        return data

    output = data
    binary_config = _cfg_get(postprocessing, "binary", None)
    if binary_config is not None and _cfg_get(binary_config, "enabled", False):
        from .postprocessing import apply_binary_postprocessing

        if output.ndim in (4, 5):
            batch_size = output.shape[0]
        elif output.ndim == 3:
            batch_size = 1
            output = output[np.newaxis, ...]
        elif output.ndim == 2:
            batch_size = 1
            output = output[np.newaxis, np.newaxis, ...]
        else:
            batch_size = 1

        results = []
        for batch_idx in range(batch_size):
            sample = output[batch_idx]
            if sample.ndim == 4:
                foreground_prob = sample[0]
            elif sample.ndim == 3:
                foreground_prob = sample
            elif sample.ndim == 2:
                foreground_prob = sample[np.newaxis, ...]
            else:
                foreground_prob = sample

            processed = apply_binary_postprocessing(foreground_prob, binary_config)
            if sample.ndim == 4:
                processed = processed[np.newaxis, ...]
            elif sample.ndim == 2:
                processed = processed[np.newaxis, np.newaxis, ...]
            results.append(processed)

        output = np.stack(results, axis=0)

    instance_cc3d_config = _cfg_get(postprocessing, "instance_cc3d", None)
    if instance_cc3d_config is not None:
        import cc3d

        cc3d_cfg = dict(instance_cc3d_config) if hasattr(instance_cc3d_config, "items") else {}
        connectivity = cc3d_cfg.get("connectivity", 6)
        min_size = cc3d_cfg.get("min_size", 0)

        spatial = output
        had_batch = False
        if spatial.ndim == 4:
            had_batch = True
            spatial = spatial[0]

        relabeled = cc3d.connected_components(spatial.astype(np.uint32), connectivity=connectivity)
        if min_size > 0:
            relabeled = cc3d.dust(relabeled, threshold=min_size, connectivity=connectivity)

        output = relabeled[np.newaxis, ...] if had_batch else relabeled
        logger.info(
            "Instance cc3d postprocessing: connectivity=%d, min_size=%d, instances=%d",
            connectivity,
            min_size,
            len(np.unique(relabeled)) - 1,
        )

    output_transpose = _cfg_get(postprocessing, "output_transpose", [])
    if output_transpose:
        try:
            output = np.transpose(output, axes=output_transpose)
        except Exception as exc:
            logger.warning(
                "Transpose failed with axes %s: %s. Keeping original shape.",
                output_transpose,
                exc,
            )

    return output


def run_decoding_stage(cfg: Any, predictions: np.ndarray) -> DecodingStageResult:
    """Decode raw predictions and apply decoded-output postprocessing."""
    start = time.time()
    has_decoding_cfg = bool(resolve_decode_modes_from_cfg(cfg))
    decoded = apply_decode_mode(cfg, predictions)
    postprocessed = apply_decoding_postprocessing(cfg, decoded) if has_decoding_cfg else decoded
    return DecodingStageResult(
        decoded=decoded,
        postprocessed=postprocessed,
        has_decoding_config=has_decoding_cfg,
        duration_s=time.time() - start,
    )


__all__ = [
    "DecodingStageResult",
    "apply_decoding_postprocessing",
    "run_decoding_stage",
]
