"""Convert pre-PR-8 Lightning checkpoints to current schema paths.

Some older checkpoints pickled config dataclasses under their previous module
locations (e.g. ``connectomics.config.schema.inference.DecodeModeConfig``,
since moved to ``connectomics.config.schema.decoding``). Pickle resolves
classes by module path at load time, so unpickling fails after the move.

This script aliases each legacy path to its current class, loads the
checkpoint, and re-saves it. ``torch.save`` serializes by each class's
current ``__module__``, so the rewritten file loads cleanly under the
current codebase (including ``weights_only=True``).

Usage::

    python scripts/checkpoint_conversion.py path/to/checkpoints/*.ckpt
    python scripts/checkpoint_conversion.py file.ckpt --no-backup
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
from dataclasses import is_dataclass
from pathlib import Path

import torch

# Pre-PR-8 checkpoints reference dataclasses from
# `connectomics.config.schema.inference` that have since moved into
# `decoding` / `evaluation`. Alias every dataclass from those modules
# under the legacy `inference` namespace so unpickling resolves.
_LEGACY_NAMESPACE = "connectomics.config.schema.inference"
_CURRENT_NAMESPACES = (
    "connectomics.config.schema.decoding",
    "connectomics.config.schema.evaluation",
)


def install_legacy_aliases() -> None:
    legacy = importlib.import_module(_LEGACY_NAMESPACE)
    for current_mod_name in _CURRENT_NAMESPACES:
        current = importlib.import_module(current_mod_name)
        for name, obj in current.__dict__.items():
            if inspect.isclass(obj) and is_dataclass(obj) and not hasattr(legacy, name):
                setattr(legacy, name, obj)


def convert_checkpoint(path: Path, *, backup: bool) -> None:
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(state, path)
    print(f"converted: {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="checkpoint files (.ckpt)")
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="overwrite in place without writing a .bak copy",
    )
    args = p.parse_args(argv)

    install_legacy_aliases()

    files = [Path(s) for s in args.paths]
    missing = [f for f in files if not f.is_file()]
    if missing:
        for f in missing:
            print(f"not a file: {f}", file=sys.stderr)
        return 1

    for f in files:
        convert_checkpoint(f, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
