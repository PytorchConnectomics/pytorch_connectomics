"""Path-related helpers for Lightning training modules."""

from __future__ import annotations

from glob import glob
from typing import List


def expand_file_paths(path_or_pattern) -> List[str]:
    """
    Expand file path inputs into a list of paths.

    Args:
        path_or_pattern: Single file path, glob pattern, or list of paths/patterns.

    Returns:
        List of expanded file paths, sorted for glob inputs.
    """
    if isinstance(path_or_pattern, list):
        return path_or_pattern

    if "*" in path_or_pattern or "?" in path_or_pattern:
        paths = sorted(glob(path_or_pattern))
        if not paths:
            raise FileNotFoundError(f"No files found matching pattern: {path_or_pattern}")
        return paths

    return [path_or_pattern]
