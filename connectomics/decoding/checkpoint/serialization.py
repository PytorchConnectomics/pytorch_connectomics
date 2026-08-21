"""Safe, deterministic serialization for checkpoint protocol records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast

from . import schema

T = TypeVar("T")
_RECORD_TYPES = {
    cls.__name__: cls
    for cls in (
        schema.ActionExecution,
        schema.ActionSpec,
        schema.AnchorTotals,
        schema.AnnotateParams,
        schema.ArtifactRef,
        schema.BoundingBox,
        schema.Certificate,
        schema.CheckpointPlan,
        schema.CheckpointResult,
        schema.Condition,
        schema.ConsolidateSameAnchorParams,
        schema.ContactScope,
        schema.ContactScopes,
        schema.Descriptor,
        schema.DescriptionBundle,
        schema.EdgeLifecycleParams,
        schema.EntityRef,
        schema.ForbidMergeParams,
        schema.Provenance,
        schema.RebuildLocalRagParams,
        schema.SplitByAnchorParams,
        schema.UnrepairedConflict,
        schema.VerificationOutcome,
    )
}
_VOLATILE_KEYS = {
    "timestamp_utc",
    "elapsed_seconds",
    "status",
    "failure_message",
    "descriptor_id",
    "certificate_id",
    "action_id",
    "plan_id",
    "result_id",
    "artifact_id",
    # Artifact locations are deployment details. Content hashes and datasets are
    # the stable artifact identity used for plans and action IDs.
    "uri",
}


def to_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {
            field.name: to_data(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
        return {"_type": type(value).__name__, **result}
    if isinstance(value, tuple):
        return [to_data(item) for item in value]
    if isinstance(value, list):
        return [to_data(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_data(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def from_data(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(from_data(item) for item in value)
    if not isinstance(value, dict):
        return value
    type_name = value.get("_type")
    if type_name is None:
        return {str(key): from_data(item) for key, item in value.items()}
    if type_name not in _RECORD_TYPES:
        raise ValueError(f"unknown serialized checkpoint record type {type_name!r}")
    cls = _RECORD_TYPES[type_name]
    valid = {field.name for field in dataclasses.fields(cls)}
    unknown = set(value) - valid - {"_type"}
    if unknown:
        raise ValueError(f"unknown fields for {type_name}: {sorted(unknown)}")
    kwargs = {key: from_data(item) for key, item in value.items() if key != "_type"}
    return cls(**kwargs)


def canonical_json(value: Any, *, exclude_volatile: bool = False) -> str:
    data = to_data(value)

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, dict):
            return {
                key: clean(child)
                for key, child in sorted(item.items())
                if not (exclude_volatile and key in _VOLATILE_KEYS)
            }
        return item

    return json.dumps(clean(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any, *, exclude_volatile: bool = False) -> str:
    return hashlib.sha256(
        canonical_json(value, exclude_volatile=exclude_volatile).encode()
    ).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{content_hash(value, exclude_volatile=True)[:24]}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path, expected_type: type[T] | None = None) -> T:
    raw = json.loads(Path(path).read_text())
    value = from_data(raw)
    if expected_type is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"{path} contains {type(value).__name__}, expected {expected_type.__name__}"
        )
    return cast(T, value)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(to_data(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML as data only; YAML constructors stay disabled."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
    else:
        import yaml

        data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError("checkpoint pass specification must be a mapping")
    return data


__all__ = [
    "append_jsonl",
    "canonical_json",
    "content_hash",
    "from_data",
    "load_spec",
    "read_json",
    "sha256_file",
    "stable_id",
    "to_data",
    "write_json",
]
