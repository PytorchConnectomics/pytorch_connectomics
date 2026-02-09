#!/usr/bin/env python3
"""Validate top-level tutorial YAML configs.

Checks:
1. Config can be loaded by the Hydra/OmegaConf loader.
2. Legacy keys that should not appear in top-level tutorials are absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import yaml

from connectomics.config import load_config


LEGACY_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    (("inference", "data"), "Use `test.data` instead of `inference.data`."),
    (
        ("data", "augmentation", "enabled"),
        "Use augmentation `preset` + per-transform `enabled` flags.",
    ),
    (
        ("inference", "test_time_augmentation", "act"),
        "Use `inference.test_time_augmentation.channel_activations`.",
    ),
]


def _has_path(data: Any, path: Tuple[str, ...]) -> bool:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_config_paths(glob_patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in glob_patterns:
        paths.extend(Path().glob(pattern))
    # Keep deterministic order and unique paths.
    return sorted(set(p for p in paths if p.is_file()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tutorial YAML configs.")
    parser.add_argument(
        "--glob",
        action="append",
        default=["tutorials/*.yaml"],
        help="Glob pattern to include (can be passed multiple times).",
    )
    args = parser.parse_args()

    config_paths = _iter_config_paths(args.glob)
    if not config_paths:
        print("No matching config files found.")
        return 1

    errors: List[str] = []
    for config_path in config_paths:
        raw = _load_yaml(config_path)

        for pattern, message in LEGACY_PATTERNS:
            if _has_path(raw, pattern):
                dotted = ".".join(pattern)
                errors.append(f"{config_path}: legacy key `{dotted}` found. {message}")

        try:
            load_config(config_path)
        except Exception as exc:  # pragma: no cover - exact exception type may vary.
            errors.append(f"{config_path}: failed to load ({type(exc).__name__}: {exc})")

    if errors:
        print("Tutorial config validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Validated {len(config_paths)} tutorial configs successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
