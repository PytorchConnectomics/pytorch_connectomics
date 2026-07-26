"""Pure helpers for resolving a Seuron provenance record to local ABISS inputs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Tuple, Union

REQUIRED_INPUT_ORDER = (
    "AFF_PATH",
    "AFF_MIP",
    "AFF_RESOLUTION",
    "BBOX",
    "CHUNK_SIZE",
    "AGG_THRESHOLD",
    "WS_HIGH_THRESHOLD",
    "WS_LOW_THRESHOLD",
    "WS_SIZE_THRESHOLD",
    "WS_DUST_THRESHOLD",
)
OPTIONAL_INPUT_ORDER = ("IMAGE_PATH",)
GENERATED_OUTPUT_ORDER = (
    "NAME",
    "WS_PATH",
    "WS_PREFIX",
    "SEG_PATH",
    "SEG_PREFIX",
    "SCRATCH_PATH",
    "SCRATCH_PREFIX",
    "CHUNKMAP_OUTPUT",
    "NG_PREFIX",
)
IGNORED_INFRA_ORDER = (
    "WORKER_IMAGE",
    "REDIS_SERVER",
    "REDIS_DB",
    "STATSD_HOST",
    "STATSD_PORT",
    "MOUNT_PATH",
    "WORKSPACE_PATH",
    "SKIP_SKELETON",
)

REQUIRED_INPUT = frozenset(REQUIRED_INPUT_ORDER)
OPTIONAL_INPUT = frozenset(OPTIONAL_INPUT_ORDER)
GENERATED_OUTPUT = frozenset(GENERATED_OUTPUT_ORDER)
IGNORED_INFRA = frozenset(IGNORED_INFRA_ORDER)
CLASSIFIED_KEYS = REQUIRED_INPUT | OPTIONAL_INPUT | GENERATED_OUTPUT | IGNORED_INFRA

_SEGMENTATION_MARKERS = frozenset(
    {
        "AGG_THRESHOLD",
        "WS_HIGH_THRESHOLD",
        "WS_LOW_THRESHOLD",
        "WS_SIZE_THRESHOLD",
        "WS_DUST_THRESHOLD",
    }
)


@dataclass(frozen=True)
class ReplaySpec:
    """The one segmentation method and subsequent igneous methods, in source order."""

    seg_block: Dict[str, Any]
    igneous_blocks: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class AbissParam(Mapping[str, Any]):
    """Whitelisted ABISS-local parameter payload."""

    param: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.param[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.param)

    def __len__(self) -> int:
        return len(self.param)

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached, mutable copy suitable for execution overrides."""

        return deepcopy(self.param)


def _is_segmentation_method(method: Mapping[str, Any]) -> bool:
    return bool(set(method) & _SEGMENTATION_MARKERS)


def load_provenance(path: Union[str, Path]) -> ReplaySpec:
    """Load a provenance JSON file and isolate its single ABISS segmentation method."""

    provenance_path = Path(path)
    with provenance_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Provenance {provenance_path} must contain a JSON object.")
    processing = payload.get("processing")
    if not isinstance(processing, list):
        raise ValueError(f"Provenance {provenance_path} must contain a processing list.")

    segmentation: list[Dict[str, Any]] = []
    igneous: list[Dict[str, Any]] = []
    for index, block in enumerate(processing):
        if not isinstance(block, dict):
            raise ValueError(f"processing[{index}] must be an object.")
        method = block.get("method")
        if not isinstance(method, dict):
            raise ValueError(f"processing[{index}].method must be an object.")
        method_copy = deepcopy(method)
        if _is_segmentation_method(method_copy):
            segmentation.append(method_copy)
        else:
            igneous.append(method_copy)

    if len(segmentation) != 1:
        raise ValueError(
            "Expected exactly one ABISS segmentation block in provenance processing, "
            f"found {len(segmentation)}."
        )

    return ReplaySpec(seg_block=segmentation[0], igneous_blocks=tuple(igneous))


def _validate_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Replay name must be a non-empty path component, got {name!r}.")


def _directory_uri(path: Path) -> str:
    return path.as_uri().rstrip("/") + "/"


def classify_and_map(
    seg_block: Mapping[str, Any],
    *,
    name: str,
    out_root: Union[str, Path],
    aff_override: Union[str, Path, None] = None,
) -> AbissParam:
    """Validate every source key and mint an ABISS-local payload in a fresh namespace."""

    if not isinstance(seg_block, Mapping):
        raise TypeError("seg_block must be a mapping.")

    unknown = set(seg_block) - CLASSIFIED_KEYS
    if unknown:
        raise ValueError(f"Unknown Seuron segmentation keys: {sorted(unknown)}")

    missing = REQUIRED_INPUT - set(seg_block)
    if missing:
        raise ValueError(f"Missing required Seuron segmentation keys: {sorted(missing)}")

    _validate_name(name)
    run_root = Path(out_root).expanduser().resolve() / name
    precomputed_root = run_root / "precomputed"
    ws_prefix = precomputed_root / "ws"
    seg_prefix = precomputed_root / "seg"
    scratch_prefix = run_root / "scratch"
    scratch_path = scratch_prefix / name

    param: Dict[str, Any] = {key: deepcopy(seg_block[key]) for key in REQUIRED_INPUT_ORDER}
    for key in OPTIONAL_INPUT_ORDER:
        if key in seg_block:
            param[key] = deepcopy(seg_block[key])
    if aff_override is not None:
        param["AFF_PATH"] = str(aff_override)

    param.update(
        {
            "NAME": name,
            "WS_PATH": (ws_prefix / name).as_uri(),
            "WS_PREFIX": _directory_uri(ws_prefix),
            "SEG_PATH": (seg_prefix / name).as_uri(),
            "SEG_PREFIX": _directory_uri(seg_prefix),
            "SCRATCH_PATH": scratch_path.as_uri(),
            "SCRATCH_PREFIX": _directory_uri(scratch_prefix),
            # The local ABISS build defaults CHUNKMAP_INPUT to this exact path.
            # Keeping input and output aliased lets the agglomeration stages
            # consume the watershed chunk map without adding a non-provenance
            # compatibility key to the fail-closed payload.
            "CHUNKMAP_OUTPUT": (scratch_path / "ws" / "chunkmap").as_uri(),
            "NG_PREFIX": _directory_uri(precomputed_root),
        }
    )
    return AbissParam(param=param)


__all__ = [
    "AbissParam",
    "CLASSIFIED_KEYS",
    "GENERATED_OUTPUT",
    "IGNORED_INFRA",
    "OPTIONAL_INPUT",
    "REQUIRED_INPUT",
    "ReplaySpec",
    "classify_and_map",
    "load_provenance",
]
