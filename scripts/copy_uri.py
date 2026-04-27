#!/usr/bin/env python3
"""Copy files or directories between local paths and file:// URIs.

This is a small compatibility helper for the vendored ABISS shell scripts,
which expect upload/download command strings. For the local workflow here we
use it instead of `cp` or a missing `cloudfiles` CLI.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse, unquote


def _resolve_path(value: str) -> Path:
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path))
    return Path(value)


def _copy(src: Path, dst: Path, *, dst_raw: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")

    dst_is_dir_hint = dst_raw.endswith("/") or dst.name in {"", "."}

    if src.is_dir():
        if dst.exists() and dst.is_dir():
            target = dst / src.name
        elif dst_is_dir_hint:
            target = dst / src.name
        else:
            target = dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, target, dirs_exist_ok=True)
        return

    if dst.exists() and dst.is_dir():
        target = dst / src.name
    elif dst_is_dir_hint:
        target = dst / src.name
    else:
        target = dst
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()

    src = _resolve_path(args.src)
    dst = _resolve_path(args.dst)
    _copy(src, dst, dst_raw=args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
