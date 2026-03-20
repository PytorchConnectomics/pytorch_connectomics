"""
Architecture module for connectomics models.

Provides:
- Registry system for architecture management
- Base model interface with deep supervision support
- MONAI model wrappers (BasicUNet, UNet, UNETR, SwinUNETR)
- Future: MedNeXt model wrappers

Usage:
    from connectomics.models.architectures import (
        register_architecture,
        list_architectures,
        ConnectomicsModel,
    )

    # List available models
    print(list_architectures())

    # Register custom model
    @register_architecture('my_model')
    def build_my_model(cfg):
        return MyModel(cfg)
"""

import logging

# Import base model
from .base import ConnectomicsModel

# Import registry functions
from .registry import (
    get_architecture_builder,
    get_architecture_info,
    is_architecture_available,
    list_architectures,
    register_architecture,
    unregister_architecture,
)

# Import MONAI models to trigger registration
try:
    from . import monai_models  # noqa: F401

    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False

# Import MedNeXt models to trigger registration
try:
    from . import mednext_models  # noqa: F401

    _MEDNEXT_AVAILABLE = True
except ImportError:
    _MEDNEXT_AVAILABLE = False

# Import RSUNet models (always available - pure PyTorch)
from . import rsunet  # noqa: F401

# Import nnUNet models to trigger registration
try:
    from . import nnunet_models  # noqa: F401

    _NNUNET_AVAILABLE = True
except ImportError:
    _NNUNET_AVAILABLE = False


# Check what's available
def get_available_architectures() -> dict:
    """
    Get information about available architectures and their dependencies.

    Returns:
        Dictionary with:
            - 'monai': List of MONAI architectures (if available)
            - 'mednext': List of MedNeXt architectures (if available)
            - 'all': List of all registered architectures
    """
    all_archs = list_architectures()

    info = {
        "all": all_archs,
        "monai": [a for a in all_archs if a.startswith("monai_")] if _MONAI_AVAILABLE else [],
        "mednext": [a for a in all_archs if a.startswith("mednext")] if _MEDNEXT_AVAILABLE else [],
        "rsunet": [a for a in all_archs if a.startswith("rsunet")],
        "nnunet": [a for a in all_archs if a.startswith("nnunet")] if _NNUNET_AVAILABLE else [],
    }

    return info


def print_available_architectures():
    """Log a formatted list of available architectures."""
    _logger = logging.getLogger(__name__)
    info = get_available_architectures()

    lines = ["\n" + "=" * 60, "Available Architectures", "=" * 60]

    if info["monai"]:
        lines.append(f"\nMONAI Models ({len(info['monai'])}):")
        for arch in info["monai"]:
            lines.append(f"  - {arch}")
    else:
        lines.append("\nMONAI Models: Not available (install with: pip install monai)")

    if info["mednext"]:
        lines.append(f"\nMedNeXt Models ({len(info['mednext'])}):")
        for arch in info["mednext"]:
            lines.append(f"  - {arch}")
    else:
        lines.append("\nMedNeXt Models: Not available (see MEDNEXT.md for setup)")

    if info["rsunet"]:
        lines.append(f"\nRSUNet Models ({len(info['rsunet'])}):")
        for arch in info["rsunet"]:
            lines.append(f"  - {arch}")

    if info["nnunet"]:
        lines.append(f"\nnnUNet Models ({len(info['nnunet'])}):")
        for arch in info["nnunet"]:
            lines.append(f"  - {arch}")
    else:
        lines.append("\nnnUNet Models: Not available (install with: pip install nnunetv2)")

    lines.append(f"\nTotal: {len(info['all'])} architectures")
    lines.append("=" * 60)

    _logger.info("\n".join(lines))


__all__ = [
    # Registry
    "register_architecture",
    "get_architecture_builder",
    "list_architectures",
    "is_architecture_available",
    "unregister_architecture",
    "get_architecture_info",
    # Base model
    "ConnectomicsModel",
    # Utilities
    "get_available_architectures",
    "print_available_architectures",
]
