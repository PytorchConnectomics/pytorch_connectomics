"""Pre-decoding QC helpers (outlier detection, affinity masking)."""

from .affinity import (
    AffinityQCAccumulator,
    AffinityQCParams,
    AffinityQCReport,
    apply_affinity_mask_from_spec,
    begin_streaming_qc,
    build_affinity_mask,
    finish_streaming_qc,
    render_markdown_report,
    run_affinity_qc,
    scan_prediction,
)

__all__ = [
    "AffinityQCAccumulator",
    "AffinityQCParams",
    "AffinityQCReport",
    "apply_affinity_mask_from_spec",
    "begin_streaming_qc",
    "build_affinity_mask",
    "finish_streaming_qc",
    "render_markdown_report",
    "run_affinity_qc",
    "scan_prediction",
]
