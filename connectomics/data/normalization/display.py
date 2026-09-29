"""Fixed display curve for aligned-intensity codes (browser views, figures).

Display is a view of the stored codes, never a second copy: decode to aligned
intensity, put the noise floor at black, and emphasize the bright end with a
gamma above 1 (no histogram equalization). The constants are in aligned units,
so one curve and one Neuroglancer shader serve every volume.
"""

from __future__ import annotations

import numpy as np

from .codec import CODEC_KNEE, CODEC_R_MAX, decode_aligned

__all__ = [
    "DISPLAY_BLACK",
    "DISPLAY_WHITE",
    "DISPLAY_GAMMA",
    "display_from_codes",
    "neuroglancer_shader",
    "display_attrs",
]

DISPLAY_BLACK = 0.75
DISPLAY_WHITE = 4.5
DISPLAY_GAMMA = 1.5


def display_from_codes(codes: np.ndarray) -> np.ndarray:
    """Stored uint8 codes -> uint8 display brightness through the fixed curve."""
    aligned = decode_aligned(np.arange(256))
    level = np.clip((aligned - DISPLAY_BLACK) / (DISPLAY_WHITE - DISPLAY_BLACK), 0.0, 1.0)
    lut = np.rint(255.0 * level**DISPLAY_GAMMA).astype(np.uint8)
    return lut[np.asarray(codes, dtype=np.uint8)]


def neuroglancer_shader() -> str:
    """Neuroglancer/Mindglancer image shader applying the display curve to the codes.

    Black, white and gamma are ``#uicontrol`` sliders defaulting to the fixed
    curve, so viewers can adjust the view without touching the stored data or
    the model input. Display settings are not part of the training contract.
    """
    sqrt_knee = float(np.sqrt(CODEC_KNEE))
    span = float(np.sqrt(CODEC_R_MAX + CODEC_KNEE)) - sqrt_knee
    # The black and white slider ranges overlap, so the width is guarded.
    return (
        f"#uicontrol float black slider(min=0, max=2, default={DISPLAY_BLACK})\n"
        f"#uicontrol float white slider(min=1, max=16, default={DISPLAY_WHITE})\n"
        f"#uicontrol float gamma slider(min=0.5, max=3, default={DISPLAY_GAMMA})\n"
        "void main() {\n"
        f"  float s = {sqrt_knee:.6f} + toNormalized(getDataValue()) * {span:.6f};\n"
        f"  float r = s * s - {CODEC_KNEE:.6f};  // decode to aligned intensity\n"
        "  float width = max(white - black, 0.000001);\n"
        "  emitGrayscale(pow(clamp((r - black) / width, 0.0, 1.0), gamma));\n"
        "}\n"
    )


def display_attrs() -> dict:
    """JSON-ready display description, for volume metadata."""
    return {
        "display_black_aligned": DISPLAY_BLACK,
        "display_white_aligned": DISPLAY_WHITE,
        "display_gamma": DISPLAY_GAMMA,
        "neuroglancer_shader": neuroglancer_shader(),
    }
