"""Explicit evaluation runtime context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.pipeline.dict_utils import cfg_get


@dataclass
class EvaluationContext:
    """Runtime inputs required by evaluation helpers.

    The context is intentionally independent of ``ConnectomicsModule``. Lightning
    code owns translation from module state into this explicit contract.
    """

    cfg: Any
    evaluation_cfg: Any = None
    inference_cfg: Any = None
    device: Any = "cpu"
    enabled: bool | None = None
    checkpoint_path: str | Path | None = None
    output_path: str | Path | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    log_fn: Callable[..., None] | None = None
    metrics_sink: Callable[[dict[str, Any]], None] | None = None
    distributed_single_volume_sharding: bool = False

    def cfg_value(self, cfg_obj: Any, name: str, default: Any = None) -> Any:
        return cfg_get(cfg_obj, name, default)

    @property
    def is_enabled(self) -> bool:
        if self.enabled is not None:
            return bool(self.enabled)
        return bool(self.cfg_value(self.evaluation_cfg, "enabled", False))

    @property
    def requested_metrics(self) -> set[str]:
        metrics = self.cfg_value(self.evaluation_cfg, "metrics", None)
        if metrics is None:
            return set()
        if isinstance(metrics, str):
            return {metrics.lower()}
        return {str(metric).lower() for metric in metrics}

    def metric_requested(self, metric_name: str) -> bool:
        return metric_name.lower() in self.requested_metrics

    def metric(self, metric_name: str) -> Any:
        key = metric_name.lower()
        return self.metrics.get(key, self.metrics.get(f"test_{key}"))

    def resolved_output_path(self) -> str | Path | None:
        if self.output_path is not None:
            return self.output_path
        save_prediction_cfg = self.cfg_value(self.inference_cfg, "save_prediction", None)
        return self.cfg_value(save_prediction_cfg, "output_path", None)

    def log_metric(self, name: str, value: Any, **kwargs: Any) -> None:
        if self.log_fn is None:
            return
        self.log_fn(name, value, **kwargs)

    def persist_metrics(self, metrics_dict: dict[str, Any]) -> bool:
        if self.metrics_sink is None:
            return False
        self.metrics_sink(metrics_dict)
        return True


__all__ = ["EvaluationContext"]
