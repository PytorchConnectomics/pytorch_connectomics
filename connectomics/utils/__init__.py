"""
Shared cross-cutting utilities for PyTorch Connectomics.

This package contains only genuinely cross-cutting helpers used by 2+ packages:
- channel_slices.py: Channel selector parsing and resolution (inference, data, training)
- label_overlap.py: Label overlap computation (metrics, decoding)
- model_outputs.py: Model-output head selection helpers (config, training, inference)
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
from .model_outputs import (
    get_model_head_names,
    get_total_model_head_channels,
    resolve_configured_output_channels,
    resolve_configured_output_head,
    resolve_head_target_slice,
    resolve_output_channels,
    resolve_output_head,
    resolve_output_heads,
    select_output_tensor,
    unwrap_main_output,
)

__all__ = [
    "infer_min_required_channels",
    "normalize_channel_range_selector",
    "normalize_channel_selector",
    "resolve_channel_index",
    "resolve_channel_indices",
    "resolve_channel_range",
    "compute_label_overlap",
    "get_model_head_names",
    "get_total_model_head_channels",
    "resolve_configured_output_channels",
    "resolve_configured_output_head",
    "resolve_head_target_slice",
    "resolve_output_channels",
    "resolve_output_head",
    "resolve_output_heads",
    "select_output_tensor",
    "unwrap_main_output",
]
