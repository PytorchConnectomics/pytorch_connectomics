"""
Architecture registry for model management.

Provides centralized registration and discovery of model architectures.
Makes it easy to add new models and validate architecture names.
"""

from typing import Dict, Callable, List
import warnings


# Global registry
_ARCHITECTURE_REGISTRY: Dict[str, Callable] = {}


def register_architecture(name: str):
    """
    Decorator to register architecture builders.

    Example:
        @register_architecture('my_model')
        def build_my_model(cfg):
            return MyModel(...)

    Args:
        name: Unique name for the architecture

    Returns:
        Decorator function
    """

    def decorator(builder_fn: Callable) -> Callable:
        if name in _ARCHITECTURE_REGISTRY:
            warnings.warn(
                f"Architecture '{name}' already registered. Overwriting previous registration.",
                UserWarning,
            )
        _ARCHITECTURE_REGISTRY[name] = builder_fn
        return builder_fn

    return decorator


def get_architecture_builder(name: str) -> Callable:
    """
    Get builder function for architecture.

    Args:
        name: Architecture name

    Returns:
        Builder function that takes cfg and returns a model

    Raises:
        ValueError: If architecture not found
    """
    if name not in _ARCHITECTURE_REGISTRY:
        available = list_architectures()
        raise ValueError(
            f"Architecture '{name}' not found.\n"
            f"Available architectures: {available}\n"
            f"Register new architectures with @register_architecture decorator."
        )
    return _ARCHITECTURE_REGISTRY[name]


def list_architectures() -> List[str]:
    """
    List all registered architectures.

    Returns:
        Sorted list of architecture names
    """
    return sorted(_ARCHITECTURE_REGISTRY.keys())


def is_architecture_available(name: str) -> bool:
    """
    Check if architecture is available.

    Args:
        name: Architecture name

    Returns:
        True if architecture is registered
    """
    return name in _ARCHITECTURE_REGISTRY


def unregister_architecture(name: str) -> None:
    """
    Unregister an architecture (useful for testing).

    Args:
        name: Architecture name

    Raises:
        ValueError: If architecture not found
    """
    if name not in _ARCHITECTURE_REGISTRY:
        raise ValueError(f"Architecture '{name}' not registered.")
    del _ARCHITECTURE_REGISTRY[name]


def get_architecture_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all registered architectures.

    Returns:
        Dict mapping architecture names to their metadata
    """
    info = {}
    for name, builder in _ARCHITECTURE_REGISTRY.items():
        info[name] = {
            "name": name,
            "module": builder.__module__,
            "doc": builder.__doc__.strip() if builder.__doc__ else "No documentation",
        }
    return info


__all__ = [
    "register_architecture",
    "get_architecture_builder",
    "list_architectures",
    "is_architecture_available",
    "unregister_architecture",
    "get_architecture_info",
]
