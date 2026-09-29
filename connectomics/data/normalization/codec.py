"""The fixed uint8 code table for aligned intensity, shared by every volume.

Aligned intensity ``r`` (0 = fitted offset, 1 = tissue median; see
``alignment``) is stored as

    r_clip = clip(r, 0, R_MAX)
    code   = round(255 * (sqrt(r_clip + A) - sqrt(A)) / (sqrt(R_MAX + A) - sqrt(A)))

The square root follows shot noise, so each code step is a near-constant
fraction of the local noise; the knee ``A`` only sets the curvature near zero.
Negative ``r`` (below the fitted offset) is *not* representable in v1: it all
becomes code 0, together with small positive values that round to 0. Use
``quantization_report`` to measure that loss before accepting v1 for a volume.

Because the table is identical for all volumes, a code means the same aligned
intensity everywhere and numeric decoding needs no per-volume metadata. Safe
interpretation still needs the codec version: ``require_codec`` rejects any
version this module does not implement.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CODEC_NAME",
    "CODEC_KNEE",
    "CODEC_R_MIN",
    "CODEC_R_MAX",
    "U16_SCALE",
    "encode_aligned",
    "decode_aligned",
    "codes_to_u16",
    "require_codec",
    "quantization_report",
    "codec_attrs",
]

CODEC_NAME = "sqrt_aligned_u8_v1"
CODEC_KNEE = 0.25
CODEC_R_MIN = 0.0
CODEC_R_MAX = 32.0
# Optional uint16 interoperability cache: u16 = round(U16_SCALE * decoded r),
# read back with ``normalize: divide-2000``. It keeps all 256 codes distinct;
# it is not a lossless copy of the raw volume.
U16_SCALE = 2000.0

_SQRT_KNEE = float(np.sqrt(CODEC_KNEE))
_SQRT_SPAN = float(np.sqrt(CODEC_R_MAX + CODEC_KNEE)) - _SQRT_KNEE


def encode_aligned(aligned: np.ndarray) -> np.ndarray:
    """Aligned intensity -> uint8 code. ``r`` outside [0, R_MAX] saturates."""
    clipped = np.clip(np.asarray(aligned, dtype=np.float32), CODEC_R_MIN, CODEC_R_MAX)
    codes = (np.sqrt(clipped + np.float32(CODEC_KNEE)) - _SQRT_KNEE) * (255.0 / _SQRT_SPAN)
    return np.clip(np.rint(codes), 0.0, 255.0).astype(np.uint8)


def decode_aligned(codes: np.ndarray) -> np.ndarray:
    """uint8 code -> aligned intensity in [0, R_MAX], float32.

    Decode stored codes *before* any interpolation: averaging square-root codes
    and decoding afterwards biases the result low.
    """
    root = _SQRT_KNEE + np.asarray(codes, dtype=np.float32) * np.float32(_SQRT_SPAN / 255.0)
    return (root * root - np.float32(CODEC_KNEE)).astype(np.float32, copy=False)


def codes_to_u16(codes: np.ndarray) -> np.ndarray:
    """uint8 codes -> optional uint16 cache at the fixed scale ``U16_SCALE``."""
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise TypeError(f"codes_to_u16 expects uint8 codes, got {codes.dtype}.")
    return np.rint(decode_aligned(codes) * np.float32(U16_SCALE)).astype(np.uint16)


def require_codec(attrs: dict) -> None:
    """Raise unless ``attrs`` declare the codec this module implements."""
    name = attrs.get("codec")
    if name != CODEC_NAME:
        raise ValueError(
            f"Unsupported intensity codec {name!r}; this build decodes only {CODEC_NAME!r}."
        )
    for key, expected in (("codec_knee", CODEC_KNEE), ("codec_r_max", CODEC_R_MAX)):
        if key in attrs and float(attrs[key]) != expected:
            raise ValueError(f"Codec {name!r} declares {key}={attrs[key]}, expected {expected}.")


def quantization_report(aligned: np.ndarray, noise_sd: np.ndarray | float | None = None) -> dict:
    """Clipping, endpoint occupancy and round-trip error of v1 on aligned values.

    ``clipped_low`` / ``clipped_high`` count values strictly outside [0, R_MAX]
    (information v1 cannot store); ``code0`` / ``code255`` count endpoint
    occupancy, which also includes in-range values that round to an endpoint.
    With ``noise_sd`` (local noise SD in aligned units, scalar or per pixel)
    the RMS error is also reported relative to noise.
    """
    aligned = np.asarray(aligned, dtype=np.float32)
    codes = encode_aligned(aligned)
    error = decode_aligned(codes) - aligned
    in_range = (aligned >= CODEC_R_MIN) & (aligned <= CODEC_R_MAX)
    report = {
        "codec": CODEC_NAME,
        "pixels": int(aligned.size),
        "clipped_low": float(np.mean(aligned < CODEC_R_MIN)),
        "clipped_high": float(np.mean(aligned > CODEC_R_MAX)),
        "code0": float(np.mean(codes == 0)),
        "code255": float(np.mean(codes == 255)),
        "codes_used": int(np.unique(codes).size),
        "rms_error_in_range": float(np.sqrt(np.mean(error[in_range] ** 2))) if in_range.any()
        else 0.0,
        "bias_in_range": float(np.mean(error[in_range])) if in_range.any() else 0.0,
    }
    if noise_sd is not None:
        noise_sd = np.broadcast_to(np.asarray(noise_sd, dtype=np.float32), aligned.shape)
        ratio = error[in_range] / noise_sd[in_range]
        report["rms_error_over_noise"] = float(np.sqrt(np.mean(ratio**2))) if ratio.size else 0.0
    return report


def codec_attrs() -> dict:
    """JSON-ready description of the code table, for volume metadata."""
    return {
        "codec": CODEC_NAME,
        "codec_formula": "code = round(255 * (sqrt(clip(r, 0, r_max) + knee) - sqrt(knee)) / "
        "(sqrt(r_max + knee) - sqrt(knee)))",
        "codec_knee": CODEC_KNEE,
        "codec_r_min": CODEC_R_MIN,
        "codec_r_max": CODEC_R_MAX,
    }
