"""Pre-decoding QC helpers (outlier detection, affinity masking)."""

from .affinity import (
    AffinityQCAccumulator,
    AffinityQCParams,
    AffinityQCReport,
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
    "begin_streaming_qc",
    "build_affinity_mask",
    "finish_streaming_qc",
    "render_markdown_report",
    "run_affinity_qc",
    "scan_prediction",
]
