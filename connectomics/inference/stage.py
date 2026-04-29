"""Inference stage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .artifact import build_prediction_artifact_metadata, write_prediction_artifact
from .manager import InferenceManager
from .output import apply_prediction_transform, apply_storage_dtype_transform


def _prediction_tensor_to_czyx(predictions: torch.Tensor) -> np.ndarray:
    """Convert a single-volume prediction tensor to artifact CZYX layout."""
    arr = predictions.detach().cpu().numpy()
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(
                "run_prediction_inference can write one artifact per call; "
                f"got batch size {arr.shape[0]}."
            )
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Prediction artifact expects CZYX data, got shape {arr.shape}.")
    return arr


def _normalize_compression(value: Any) -> str | None:
    return None if value in (None, "", "none") else str(value)


def run_prediction_inference(
    manager: InferenceManager,
    images: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    mask_align_to_image: bool = False,
    requested_head: Optional[str] = None,
    output_path: str | Path | None = None,
    image_path: str | None = None,
    checkpoint_path: str | Path | None = None,
    input_shape: Sequence[int] | None = None,
    crop_pad: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Run model prediction without decoding or evaluation.

    When ``output_path`` is provided, the single-volume prediction is also
    written as the canonical raw prediction artifact.
    """
    predictions = manager.predict_with_tta(
        images,
        mask=mask,
        mask_align_to_image=mask_align_to_image,
        requested_head=requested_head,
    )

    if output_path is not None:
        data = _prediction_tensor_to_czyx(predictions)
        data = apply_prediction_transform(manager.cfg, data)
        data = apply_storage_dtype_transform(manager.cfg, data)
        compression = _normalize_compression(
            getattr(getattr(manager.cfg.inference, "save_prediction", None), "compression", "gzip")
        )
        write_prediction_artifact(
            output_path,
            data,
            metadata=build_prediction_artifact_metadata(
                manager.cfg,
                image_path=image_path,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                output_head=requested_head,
                input_shape=input_shape,
                final_shape=data.shape[-3:],
                crop_pad=crop_pad,
                intensity_dtype=str(data.dtype),
                extra={"compression": str(compression)},
            ),
            compression=compression,
        )

    return predictions


__all__ = ["run_prediction_inference"]
