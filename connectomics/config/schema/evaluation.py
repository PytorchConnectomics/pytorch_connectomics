from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    enabled: bool = False
    metrics: Optional[List[str]] = None
    prediction_threshold: float = 0.5
    instance_iou_threshold: float = 0.5
    nerl_graph: Any = None
    nerl_mask: Any = None
    nerl_resolution: Optional[List[float]] = None
    nerl_merge_threshold: int = 1
    nerl_chunk_num: int = 1
    nerl_skeleton_id_attribute: str = "id"
    nerl_skeleton_position_attribute: str = "index_position"
    nerl_skeleton_edge_length_attribute: str = "edge_length"
    nerl_skeleton_position_order: str = "xyz"
    nerl_prediction_position_order: Optional[str] = None
