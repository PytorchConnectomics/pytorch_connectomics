from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DualViewMedNeXtConfig:
    """Fusion controls for two-view MedNeXt models.

    ``siamese_mean`` shares the complete encoder and averages corresponding
    multiscale features. ``partial_gated`` keeps the stem, first encoder stage,
    and first downsampling block view-specific, then shares the deeper encoder.
    """

    design: str = "siamese_mean"
    view_angles_deg: List[float] = field(default_factory=lambda: [-45.0, 45.0])
    angle_conditioning: bool = False
    gate_reduction: int = 4

