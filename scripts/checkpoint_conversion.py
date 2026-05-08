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


_LEGACY_SHIM_CLASSES: dict[str, type] = {}


def _make_legacy_shim(name: str) -> type:
    """Return a placeholder class for a legacy schema name.

    Used only at unpickle time so ``find_class(legacy_module, name)`` succeeds
    for classes that were removed entirely (not just relocated). Instances are
    detected and stripped from the loaded state before re-saving so the
    rewritten checkpoint contains no references to the placeholder.
    """
    if name in _LEGACY_SHIM_CLASSES:
        return _LEGACY_SHIM_CLASSES[name]
    cls = type(
        name,
        (),
        {
            "__module__": _LEGACY_NAMESPACE,
            "_legacy_shim": True,
            "__setstate__": lambda self, state: self.__dict__.update(state),
        },
    )
    _LEGACY_SHIM_CLASSES[name] = cls
    return cls


def install_legacy_aliases() -> None:
    legacy = importlib.import_module(_LEGACY_NAMESPACE)
    for current_mod_name in _CURRENT_NAMESPACES:
        current = importlib.import_module(current_mod_name)
        for name, obj in current.__dict__.items():
            if inspect.isclass(obj) and is_dataclass(obj) and not hasattr(legacy, name):
                setattr(legacy, name, obj)
    if not getattr(legacy, "_shim_getattr_installed", False):
        original_getattr = legacy.__dict__.get("__getattr__")

        def _legacy_module_getattr(name: str):
            if original_getattr is not None:
                try:
                    return original_getattr(name)
                except AttributeError:
                    pass
            print(f"  [legacy shim] {_LEGACY_NAMESPACE}.{name}", file=sys.stderr)
            return _make_legacy_shim(name)

        legacy.__getattr__ = _legacy_module_getattr  # type: ignore[attr-defined]
        legacy._shim_getattr_installed = True  # type: ignore[attr-defined]


def _is_legacy_shim_instance(value: object) -> bool:
    return getattr(type(value), "_legacy_shim", False) is True


def scrub_legacy_state(obj: object, seen: set[int] | None = None) -> None:
    """Recursively delete attributes / dict entries holding shim instances."""
    if seen is None:
        seen = set()
    sid = id(obj)
    if sid in seen:
        return
    seen.add(sid)

    if isinstance(obj, dict):
        for k in [k for k, v in obj.items() if _is_legacy_shim_instance(v)]:
            del obj[k]
        for v in list(obj.values()):
            scrub_legacy_state(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            scrub_legacy_state(v, seen)
    elif hasattr(obj, "__dict__"):
        for k in [k for k, v in obj.__dict__.items() if _is_legacy_shim_instance(v)]:
            delattr(obj, k)
        for v in list(obj.__dict__.values()):
            scrub_legacy_state(v, seen)


def convert_checkpoint(path: Path, *, backup: bool) -> None:
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    state = torch.load(path, map_location="cpu", weights_only=False)
    scrub_legacy_state(state)
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
