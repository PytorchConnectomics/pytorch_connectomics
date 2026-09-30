"""PyTorch Connectomics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-connectomics")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
