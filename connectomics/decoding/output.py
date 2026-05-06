"""Output writing for decoded segmentation artifacts.

The decoded artifact is the Stage-3 output (instance label map) and must
not pass through ``cfg.inference.save_inference.dtype``: that field
configures the Stage-2 raw-prediction artifact (probability/affinity
field) and silently truncates uint32 instance IDs above 65504 when
applied to label volumes via float16.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import DictConfig

from ..config import Config

logger = logging.getLogger(__name__)


def _resolve_decoded_output_dir(cfg: Config | DictConfig) -> str | None:
    """Output directory for decoded artifacts.

    Priority: ``cfg.decoding.output_path`` →
    ``cfg.inference.save_prediction.output_path`` →
    parent of ``cfg.decoding.input_prediction_path``.
    """
    decoding_cfg = getattr(cfg, "decoding", None)
    if decoding_cfg is not None:
        path = getattr(decoding_cfg, "output_path", "")
        if path:
            return str(path)

    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is not None:
        save_pred_cfg = getattr(inference_cfg, "save_prediction", None)
        if save_pred_cfg is not None:
            path = getattr(save_pred_cfg, "output_path", None)
            if path:
                return str(path)

    if decoding_cfg is not None:
        input_path = getattr(decoding_cfg, "input_prediction_path", "")
        if input_path:
            return str(Path(input_path).expanduser().parent)

    return None


def write_decoded_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str,
) -> None:
    """Persist decoded segmentation arrays to disk as HDF5.

    Saves integer label volumes as-is. Does not consult inference-stage
    storage dtype config; instance IDs are preserved at their native
    dtype.
    """
    output_dir_value = _resolve_decoded_output_dir(cfg)
    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import write_hdf5

    if predictions.ndim >= 4:
        actual_batch_size = predictions.shape[0]
    elif predictions.ndim == 3:
        if len(filenames) > 0 and predictions.shape[0] == len(filenames):
            actual_batch_size = predictions.shape[0]
        else:
            actual_batch_size = 1
            predictions = predictions[np.newaxis, ...]
    else:
        actual_batch_size = 1
        predictions = predictions[np.newaxis, ...]

    if len(filenames) != actual_batch_size:
        logger.warning(
            f"write_decoded_outputs - filename count ({len(filenames)}) "
            f"does not match batch size ({actual_batch_size}). Using first "
            f"{min(len(filenames), actual_batch_size)} filenames."
        )

    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            logger.warning(
                f"write_decoded_outputs - no filename for batch index {idx}, skipping"
            )
            continue

        sample = np.squeeze(predictions[idx])
        out_path = output_dir / f"{filenames[idx]}_{suffix}.h5"
        write_hdf5(out_path, sample, dataset="main")
        logger.info(f"Saved HDF5: {out_path.name}")


__all__ = ["write_decoded_outputs"]
