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
    """Apply postprocessing configured under ``cfg.decoding.postprocessing``."""
    decoding_cfg = _cfg_get(cfg, "decoding", None)
    postprocessing = _cfg_get(decoding_cfg, "postprocessing", None)
    if not _cfg_get(postprocessing, "enabled", False):
        return data

    output = data
    binary_config = _cfg_get(postprocessing, "binary", None)
    if binary_config is not None and _cfg_get(binary_config, "enabled", False):
        from .postprocess import apply_binary_postprocessing

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


def _maybe_apply_affinity_mask(cfg: Any, predictions: np.ndarray) -> np.ndarray:
    """Zero affinity channels at masked-out voxels per ``decoding.affinity_mask_path``.

    The mask file must contain a single 3D ``uint8`` dataset matching the spatial
    shape of ``predictions[0]`` (the affinity volume). 0 = drop, 1 = keep. Mask is
    broadcast across the channel dimension; voxels where ``mask==0`` get zeroed
    so that connected-components decoders sever all outgoing edges there.
    """
    decoding_cfg = _cfg_get(cfg, "decoding", None)
    mask_path = _cfg_get(decoding_cfg, "affinity_mask_path", "") or ""
    if not mask_path:
        return predictions
    if predictions.ndim != 4:
        raise ValueError(
            "affinity_mask_path requires 4D predictions (C, *spatial); "
            f"got shape {predictions.shape}"
        )
    import h5py

    with h5py.File(mask_path, "r") as f:
        if "main" in f:
            mask = f["main"][...]
        else:
            keys = list(f.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"affinity_mask_path={mask_path}: expected dataset 'main' or "
                    f"a single dataset, got {keys}"
                )
            mask = f[keys[0]][...]
    if mask.shape != predictions.shape[1:]:
        raise ValueError(
            f"affinity_mask shape {mask.shape} != predictions spatial shape "
            f"{predictions.shape[1:]}"
        )
    zero_idx = (mask == 0) if mask.dtype != np.bool_ else ~mask
    n_zero = int(zero_idx.sum())
    n_total = int(zero_idx.size)
    logger.info(
        "Applying affinity mask %s: zeroing %d/%d voxels (%.2f%%) across %d channels",
        mask_path,
        n_zero,
        n_total,
        100.0 * n_zero / max(n_total, 1),
        predictions.shape[0],
    )
    for c in range(predictions.shape[0]):
        predictions[c][zero_idx] = 0
    return predictions


def run_decoding_stage(
    cfg: Any,
    predictions: np.ndarray,
    *,
    on_step_complete: Any = None,
) -> DecodingStageResult:
    """Decode raw predictions and apply decoded-output postprocessing.

    ``on_step_complete``, if provided, is forwarded to the per-step decoder
    pipeline so callers can write intermediate per-step outputs.
    """
    start = time.time()
    if predictions.ndim == 5 and predictions.shape[0] == 1:
        predictions = predictions[0]
    has_decoding_cfg = bool(resolve_decode_modes_from_cfg(cfg))
    if has_decoding_cfg:
        predictions = _maybe_apply_affinity_mask(cfg, predictions)
    decoded = apply_decode_mode(cfg, predictions, on_step_complete=on_step_complete)
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
