#!/usr/bin/env python3
"""Validate tutorial-tree YAML configs.

Checks:
1. Canonical tutorial configs can be loaded by the Hydra/OmegaConf loader.
2. Legacy keys that should not appear in top-level tutorials are absent.

Some large-volume workflow recipes live under ``tutorials/`` but are consumed
directly by workflow scripts instead of ``scripts/main.py --config``. These are
identified by a custom top-level root in ``CUSTOM_WORKFLOW_ROOTS`` and reported
separately rather than loaded through the structured Config schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import yaml

from connectomics.config import load_config
from connectomics.runtime.preflight import validate_runtime_coherence

LEGACY_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    (("inference", "data"), "Use `test.data` instead of `inference.data`."),
    (
        ("data", "augmentation", "enabled"),
        "Use per-transform `enabled` flags instead.",
    ),
    (
        ("inference", "test_time_augmentation", "act"),
        "Use `inference.model.channel_activations`.",
    ),
]

for _root in (
    ("inference",),
    ("default", "inference"),
    ("test", "inference"),
    ("tune", "inference"),
):
    _root_path = ".".join(_root)
    LEGACY_PATTERNS.extend(
        [
            (_root + ("head",), f"Use `{_root_path}.model.head`."),
            (_root + ("select_channel",), f"Use `{_root_path}.model.select_channel`."),
            (_root + ("crop_pad",), f"Use `{_root_path}.model.crop_pad`."),
            (_root + ("strategy",), f"Use `{_root_path}.execution.strategy`."),
            (_root + ("do_eval",), f"Use `{_root_path}.execution.do_eval`."),
            (_root + ("sliding_window",), f"Use `{_root_path}.window`."),
            (_root + ("save",), f"Use `{_root_path}.save_results` plus sibling save_* leaves."),
            (_root + ("save_prediction",), f"Use `{_root_path}.save_results`."),
            (_root + ("save_inference",), f"Use `{_root_path}.save_results`."),
            (_root + ("output_path",), f"Use `{_root_path}.save_path`."),
            (_root + ("cache_suffix",), f"Use `{_root_path}.save_cache_suffix`."),
            (_root + ("dtype",), f"Use `{_root_path}.save_dtype`."),
            (_root + ("backend",), f"Use `{_root_path}.save_backend`."),
            (_root + ("compression",), f"Use `{_root_path}.save_compression`."),
            (_root + ("chunks",), f"`{_root_path}.chunks` was deleted (unused field)."),
            (_root + ("write_mode",), f"`{_root_path}.write_mode` was deleted (unused field)."),
            (_root + ("tta_result_path",), f"Use `{_root_path}.load_tta_path`."),
        ]
    )

for _root in (
    ("decoding",),
    ("default", "decoding"),
    ("test", "decoding"),
    ("tune", "decoding"),
):
    _root_path = ".".join(_root)
    LEGACY_PATTERNS.extend(
        [
            (_root + ("save",), f"Use `{_root_path}.save_results` and `{_root_path}.save_intermediate`."),
            (_root + ("output_path",), f"Use `{_root_path}.save_path`."),
            (_root + ("output_suffix",), f"Use `{_root_path}.save_suffix`."),
            (_root + ("input_prediction_path",), f"Use `{_root_path}.load_prediction_path`."),
        ]
    )

# tune.output: sub-block hoisted to tune.save_*
for _root in (("tune",), ("default", "tune")):
    _root_path = ".".join(_root)
    LEGACY_PATTERNS.extend(
        [
            (_root + ("output", "output_dir"), f"Use `{_root_path}.save_path`."),
            (_root + ("output", "output_pred"), f"Use `{_root_path}.save_predictions_path`."),
            (_root + ("output", "cache_suffix"), f"Use `{_root_path}.save_cache_suffix`."),
            (_root + ("output", "save_all_trials"), f"Use `{_root_path}.save_all_trials`."),
            (_root + ("output", "save_best_segmentation"), f"Use `{_root_path}.save_best_segmentation`."),
            (_root + ("output", "save_study"), f"Use `{_root_path}.save_study`."),
            (_root + ("output", "visualizations"), f"Use `{_root_path}.save_visualizations`."),
            (_root + ("output", "report"), f"Use `{_root_path}.save_report`."),
            (_root + ("output",), f"`{_root_path}.output` was hoisted; use `{_root_path}.save_*` siblings."),
        ]
    )

# `data.train.name` is structurally allowed but advisory only — train mode does
# not write per-volume artifacts. Validator emits an info warning but does not
# fail. Implemented inline in the validator main loop.
ADVISORY_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    (("data", "train", "name"),
     "data.train.name has no effect; train mode writes no per-volume artifacts. "
     "Set `data.val.name` or `data.test.name` instead."),
    (("default", "data", "train", "name"),
     "default.data.train.name has no effect; set under val/test instead."),
]

CUSTOM_WORKFLOW_ROOTS = {"large_decode", "abiss_large"}


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
    advisories: List[str] = []
    canonical_count = 0
    custom_workflows: List[Path] = []
    for config_path in config_paths:
        raw = _load_yaml(config_path)
        if isinstance(raw, dict) and CUSTOM_WORKFLOW_ROOTS.intersection(raw):
            custom_workflows.append(config_path)
            continue

        canonical_count += 1
        for pattern, message in LEGACY_PATTERNS:
            if _has_path(raw, pattern):
                dotted = ".".join(pattern)
                errors.append(f"{config_path}: legacy key `{dotted}` found. {message}")

        for pattern, message in ADVISORY_PATTERNS:
            if _has_path(raw, pattern):
                dotted = ".".join(pattern)
                advisories.append(f"{config_path}: advisory key `{dotted}`. {message}")

        try:
            cfg = load_config(config_path)
            validate_runtime_coherence(cfg)
        except Exception as exc:  # pragma: no cover - exact exception type may vary.
            errors.append(f"{config_path}: failed to load ({type(exc).__name__}: {exc})")

    if advisories:
        print("Advisory warnings (non-fatal):")
        for adv in advisories:
            print(f"  - {adv}")

    if errors:
        print("Tutorial config validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"Validated {canonical_count} canonical tutorial configs successfully; "
        f"skipped {len(custom_workflows)} custom workflow YAMLs."
    )
    if custom_workflows:
        print("Custom workflows:")
        for path in custom_workflows:
            print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
