"""Resume manifest for chunked workflows.

A simple JSON sidecar that records which chunks have been written. The
config dict at the top is checked on resume so that a re-run with mismatched
chunk shape / overlap / dtype refuses to silently continue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["ResumeManifest", "ManifestConfigMismatch"]


class ManifestConfigMismatch(ValueError):
    """Raised when a resumed manifest's recorded config disagrees with the
    current run."""


class ResumeManifest:
    """Append-only manifest of completed chunk keys, stored as JSON.

    The on-disk format is ``{"config": {...}, "completed": ["z0_y0_x0", ...]}``.
    Writes go through ``<path>.tmp`` and ``os.replace`` for crash-safety.
    """

    def __init__(self, path: str | Path, config: Mapping[str, Any]):
        self.path = Path(path)
        self.config = dict(config)
        self._completed: set[str] = set()

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        config: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> "ResumeManifest":
        path = Path(path)
        if overwrite and path.exists():
            path.unlink()
        if not path.exists():
            manifest = cls(path, config)
            manifest._write()
            return manifest

        with path.open("r") as f:
            payload = json.load(f)
        manifest = cls(path, config)
        manifest._completed = set(payload.get("completed", []))
        existing_cfg = payload.get("config", {})
        keys_to_check = ("chunk_shape", "overlap", "output_dtype", "output_shape")
        diffs: list[str] = []
        for key in keys_to_check:
            if key in manifest.config and key in existing_cfg and existing_cfg[key] != manifest.config[key]:
                diffs.append(f"{key}: existing={existing_cfg[key]} requested={manifest.config[key]}")
        if diffs:
            raise ManifestConfigMismatch(
                f"Resume manifest at {path} disagrees with requested config: "
                + "; ".join(diffs)
                + ". Re-run with overwrite=True or change the requested config."
            )
        return manifest

    @property
    def completed(self) -> set[str]:
        return set(self._completed)

    def mark_completed(self, chunk_key: str) -> None:
        if chunk_key in self._completed:
            return
        self._completed.add(chunk_key)
        self._write()

    def mark_many(self, chunk_keys: Iterable[str]) -> None:
        new = {k for k in chunk_keys if k not in self._completed}
        if not new:
            return
        self._completed.update(new)
        self._write()

    def _write(self) -> None:
        payload = {
            "config": self.config,
            "completed": sorted(self._completed),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
