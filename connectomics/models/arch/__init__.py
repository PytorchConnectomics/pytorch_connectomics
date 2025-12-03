"""
Architecture module for connectomics models.

Provides:
- Registry system for architecture management
- Base model interface with deep supervision support
- MONAI model wrappers (BasicUNet, UNet, UNETR, SwinUNETR)
- Future: MedNeXt model wrappers

Usage:
    from connectomics.models.arch import (
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

# Import registry functions
from .registry import (
    register_architecture,
    get_architecture_builder,
    list_architectures,
    is_architecture_available,
    unregister_architecture,
    get_architecture_info,
)

# Import base model
from .base import ConnectomicsModel

# Import MONAI models to trigger registration
try:
    from . import monai_models

    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False

# Import MedNeXt models to trigger registration
try:
    from . import mednext_models

    _MEDNEXT_AVAILABLE = True
except ImportError:
    _MEDNEXT_AVAILABLE = False

# Import RSUNet models (always available - pure PyTorch)
try:
    from . import rsunet

    _RSUNET_AVAILABLE = True
except ImportError:
    _RSUNET_AVAILABLE = False

# Import nnUNet models to trigger registration
try:
    from . import nnunet_models

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
        "rsunet": [a for a in all_archs if a.startswith("rsunet")] if _RSUNET_AVAILABLE else [],
        "nnunet": [a for a in all_archs if a.startswith("nnunet")] if _NNUNET_AVAILABLE else [],
    }

    return info


def print_available_architectures():
    """Print a formatted list of available architectures."""
    info = get_available_architectures()

    print("\n" + "=" * 60)
    print("Available Architectures")
    print("=" * 60)

    if info["monai"]:
        print(f"\nMONAI Models ({len(info['monai'])}):")
        for arch in info["monai"]:
            print(f"  - {arch}")
    else:
        print("\nMONAI Models: Not available (install with: pip install monai)")

    if info["mednext"]:
        print(f"\nMedNeXt Models ({len(info['mednext'])}):")
        for arch in info["mednext"]:
            print(f"  - {arch}")
    else:
        print("\nMedNeXt Models: Not available (see MEDNEXT.md for setup)")

    if info["rsunet"]:
        print(f"\nRSUNet Models ({len(info['rsunet'])}):")
        for arch in info["rsunet"]:
            print(f"  - {arch}")

    if info["nnunet"]:
        print(f"\nnnUNet Models ({len(info['nnunet'])}):")
        for arch in info["nnunet"]:
            print(f"  - {arch}")
    else:
        print("\nnnUNet Models: Not available (install with: pip install nnunetv2)")

    print(f"\nTotal: {len(info['all'])} architectures")
    print("=" * 60 + "\n")


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
