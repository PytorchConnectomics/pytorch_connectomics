"""
Shared cross-cutting utilities for PyTorch Connectomics.

This package contains only genuinely cross-cutting helpers used by 2+ packages:
- channel_slices.py: Channel selector parsing and resolution (inference, data, training)
- label_overlap.py: Label overlap computation (metrics, decoding)
- errors.py: Pre-flight validation checks (scripts)
"""

from .channel_slices import (
    infer_min_required_channels,
    normalize_channel_range_selector,
    normalize_channel_selector,
    resolve_channel_index,
    resolve_channel_indices,
    resolve_channel_range,
)
from .label_overlap import compute_label_overlap

__all__ = [
    "infer_min_required_channels",
    "normalize_channel_range_selector",
    "normalize_channel_selector",
    "resolve_channel_index",
    "resolve_channel_indices",
    "resolve_channel_range",
    "compute_label_overlap",
]
