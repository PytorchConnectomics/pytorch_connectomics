"""Image intensity normalization: raw microscope values -> one shared uint8 profile.

``alignment`` maps each volume to aligned intensity (0 = no light, 1 = tissue
median, flat field divided out); ``codec`` is the fixed uint8 code table that
stores it (decoded for training); ``display`` is the fixed browser view of the
same codes.
"""

from .alignment import alignment_attrs, fit_alignment, flat_field_image, to_aligned
from .codec import (
    CODEC_NAME,
    codes_to_u16,
    decode_aligned,
    encode_aligned,
    quantization_report,
    require_codec,
)
from .display import display_from_codes, neuroglancer_shader
from .noise import estimate_noise_model

__all__ = [
    "CODEC_NAME",
    "alignment_attrs",
    "codes_to_u16",
    "decode_aligned",
    "display_from_codes",
    "encode_aligned",
    "estimate_noise_model",
    "fit_alignment",
    "flat_field_image",
    "neuroglancer_shader",
    "quantization_report",
    "require_codec",
    "to_aligned",
]
