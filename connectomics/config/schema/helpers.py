from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .root import Config


def configure_edge_mode(
    cfg: "Config", mode: str = "seg-all", thickness: int = 1, processing_mode: str = "3d"
) -> None:
    """Configure edge detection mode for instance segmentation."""
    cfg.data.label_transform.edge_mode.mode = mode
    cfg.data.label_transform.edge_mode.thickness = thickness
    cfg.data.label_transform.edge_mode.processing_mode = processing_mode


def configure_instance_segmentation(cfg: "Config", boundary_thickness: int = 5) -> None:
    """Configure for instance segmentation with boundary detection."""
    configure_edge_mode(cfg, mode="seg-all", thickness=boundary_thickness, processing_mode="3d")
    cfg.data.label_transform.boundary_thickness = boundary_thickness

