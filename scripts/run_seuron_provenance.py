#!/usr/bin/env python3
"""Resolve or execute an ABISS-local replay of a Seuron provenance record.

Resolution is the default and is deliberately read-only. Pass ``--execute`` to
run ABISS, after affinity/credential preflight and output namespace checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse

import yaml  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from connectomics.runtime.abiss_large import (  # noqa: E402
    OutputLayerSpec,
    PreparedConfig,
    RunResult,
    run_abiss_large,
)
from connectomics.runtime.seuron_provenance import (  # noqa: E402
    GENERATED_OUTPUT,
    ReplaySpec,
    classify_and_map,
    load_provenance,
)

MANIFEST_COMPATIBILITY_KEYS = (
    "provenance_sha",
    "whitelisted_param_sha",
    "abiss_build_id",
    "execution_bbox",
)


@dataclass(frozen=True)
class ResolvedReplay:
    """Fully resolved, but not prepared or executed, replay configuration."""

    config_path: Path
    provenance_path: Path
    backend: str
    data: str
    out_root: Path
    name: str
    secrets_dir: Path
    abiss_home: Path
    mode: str
    param: dict[str, Any]
    execution_bbox: tuple[int, int, int, int, int, int]
    score_bbox: tuple[int, int, int, int, int, int] | None
    replay_spec: ReplaySpec

    @property
    def run_root(self) -> Path:
        return self.out_root / self.name

    @property
    def workdir(self) -> Path:
        return self.run_root / "work"

    @property
    def param_path(self) -> Path:
        return self.run_root / "param"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def chunk_grid(self) -> tuple[int, int, int]:
        chunk_size = self.param["CHUNK_SIZE"]
        return tuple(
            math.ceil(
                (self.execution_bbox[axis + 3] - self.execution_bbox[axis]) / chunk_size[axis]
            )
            for axis in range(3)
        )


@dataclass(frozen=True)
class AffinityMetadata:
    """Validated affinity metadata reused to initialize fresh output layers."""

    resolution_xyz: tuple[int, int, int]
    chunk_size_xyz: tuple[int, int, int]


def _default_secrets_dir() -> Path:
    cloudvolume_root = os.environ.get("CLOUD_VOLUME_DIR")
    if cloudvolume_root:
        return Path(cloudvolume_root).expanduser() / "secrets"
    return Path.home() / ".cloudvolume" / "secrets"


def _filesystem_path(value: str | os.PathLike[str], *, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _cloudpath(value: str | os.PathLike[str], *, base: Path, local_only: bool = False) -> str:
    text = os.path.expandvars(str(value)).strip()
    parsed = urlparse(text)
    if parsed.scheme:
        if local_only and parsed.scheme != "file":
            raise ValueError(f"--mirror must name a local path, got {text!r}.")
        return text
    return _filesystem_path(text, base=base).as_uri()


def _bbox(value: Sequence[Any], *, field: str) -> tuple[int, int, int, int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 6:
        raise ValueError(f"{field} must contain six XYZ coordinates, got {value!r}.")
    bbox = tuple(int(item) for item in value)
    if any(bbox[axis] >= bbox[axis + 3] for axis in range(3)):
        raise ValueError(f"{field} must have positive extent on every axis, got {list(bbox)}.")
    return bbox  # type: ignore[return-value]


def _contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return all(
        int(outer[axis]) <= int(inner[axis]) < int(inner[axis + 3]) <= int(outer[axis + 3])
        for axis in range(3)
    )


def _load_replay_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping.")
    replay = payload.get("seuron_replay")
    if not isinstance(replay, dict):
        raise ValueError(f"Config {config_path} has no seuron_replay mapping.")
    return dict(replay)


def _selected_mode(args: argparse.Namespace, replay: Mapping[str, Any]) -> str:
    mode = args.mode or replay.get("mode", "fresh")
    if args.overwrite:
        if args.mode not in (None, "overwrite"):
            raise ValueError("--overwrite conflicts with a non-overwrite --mode.")
        mode = "overwrite"
    mode = str(mode)
    if mode not in {"fresh", "resume", "overwrite"}:
        raise ValueError(f"Unsupported replay mode {mode!r}.")
    return mode


def resolve_replay(args: argparse.Namespace) -> ResolvedReplay:
    """Resolve YAML, CLI precedence, and provenance without touching outputs."""

    config_path = Path(args.config).expanduser().resolve()
    replay = _load_replay_yaml(config_path)
    config_base = config_path.parent
    cli_base = Path.cwd()

    backend = str(replay.get("backend", "abiss_local"))
    if backend != "abiss_local":
        raise ValueError(
            f"Backend {backend!r} is not implemented; Phase 1 supports only 'abiss_local'."
        )
    data = str(replay.get("data", "local"))
    if data != "local":
        raise ValueError(f"Data mode {data!r} is not implemented; Phase 1 supports only 'local'.")

    provenance_value = replay.get("provenance_path")
    if provenance_value is None:
        raise ValueError("seuron_replay.provenance_path is required.")
    provenance_path = _filesystem_path(provenance_value, base=config_base)

    out_root_value = args.out_root if args.out_root is not None else replay.get("out_root")
    if out_root_value is None:
        raise ValueError("seuron_replay.out_root or --out-root is required.")
    out_root = _filesystem_path(
        out_root_value,
        base=cli_base if args.out_root is not None else config_base,
    )

    name_value = args.name if args.name is not None else replay.get("name")
    if name_value is None:
        raise ValueError("seuron_replay.name or --name is required.")
    name = str(name_value)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Replay name must be one non-empty path component, got {name!r}.")

    secrets_value = args.secrets_dir
    if secrets_value is None:
        secrets_value = replay.get("secrets_dir", _default_secrets_dir())
    secrets_dir = _filesystem_path(
        secrets_value,
        base=cli_base if args.secrets_dir is not None else config_base,
    )

    abiss_value = args.abiss_home if args.abiss_home is not None else replay.get("abiss_home")
    if abiss_value is None:
        raise ValueError("seuron_replay.abiss_home or --abiss-home is required.")
    abiss_home = _filesystem_path(
        abiss_value,
        base=cli_base if args.abiss_home is not None else config_base,
    )

    aff_override = None
    if args.aff_path is not None:
        aff_override = _cloudpath(args.aff_path, base=cli_base)
    elif args.mirror is not None:
        aff_override = _cloudpath(args.mirror, base=cli_base, local_only=True)
    elif replay.get("aff_path") is not None:
        aff_override = _cloudpath(replay["aff_path"], base=config_base)
    elif replay.get("mirror") is not None:
        aff_override = _cloudpath(replay["mirror"], base=config_base, local_only=True)

    spec = load_provenance(provenance_path)
    mapped = classify_and_map(
        spec.seg_block,
        name=name,
        out_root=out_root,
        aff_override=aff_override,
    )
    param = dict(mapped.param)

    exec_value = args.exec_bbox if args.exec_bbox is not None else replay.get("exec_bbox")
    execution_bbox = _bbox(
        exec_value if exec_value is not None else param["BBOX"],
        field="execution bbox",
    )
    param["BBOX"] = list(execution_bbox)

    score_value = args.score_bbox if args.score_bbox is not None else replay.get("score_bbox")
    score_bbox = _bbox(score_value, field="score bbox") if score_value is not None else None
    if score_bbox is not None and not _contains(execution_bbox, score_bbox):
        raise ValueError(
            f"score bbox {list(score_bbox)} must be contained in execution bbox "
            f"{list(execution_bbox)}."
        )

    return ResolvedReplay(
        config_path=config_path,
        provenance_path=provenance_path,
        backend=backend,
        data=data,
        out_root=out_root,
        name=name,
        secrets_dir=secrets_dir,
        abiss_home=abiss_home,
        mode=_selected_mode(args, replay),
        param=param,
        execution_bbox=execution_bbox,
        score_bbox=score_bbox,
        replay_spec=spec,
    )


def _top_mip(chunk_grid: Sequence[int]) -> int:
    dims = [max(1, int(value)) for value in chunk_grid]
    mip = 0
    while dims != [1, 1, 1]:
        dims = [(value + 1) // 2 for value in dims]
        mip += 1
    return mip


def _output_chunk_size(chunk_size_xyz: Sequence[int] | None, fallback: Sequence[int]) -> list[int]:
    """Storage chunk size for the WS/SEG/size-map output layers.

    ABISS uploads whole ``CHUNK_SIZE`` blocks back into these layers in parallel, so the
    layer's storage chunk must divide ``CHUNK_SIZE`` on every axis or CloudVolume would
    reject the non-aligned writes (see ``_validate_abiss_upload_alignment``). The affinity
    layer's own chunk size (which must divide the 1008^3 source blocks, e.g. [144,144,72])
    does not generally divide ``CHUNK_SIZE`` [512,512,256], so it cannot be reused here.
    We take each ``CHUNK_SIZE`` axis halved down to a reasonable tile (<=256); halving an
    even value preserves divisibility.
    """
    if not chunk_size_xyz:
        return [int(v) for v in fallback]
    out: list[int] = []
    for value in chunk_size_xyz:
        tile = int(value)
        while tile > 256 and tile % 2 == 0:
            tile //= 2
        out.append(tile)
    return out


def prepare_execution(
    resolved: ResolvedReplay,
    affinity_metadata: AffinityMetadata | None = None,
) -> PreparedConfig:
    """Build the runtime executor input without creating files."""

    top_mip = _top_mip(resolved.chunk_grid)
    root_tag = f"{top_mip}_0_0_0" if top_mip else "0_0_0_0"
    runtime_cloudvolume_root = resolved.run_root / ".cloudvolume"
    return PreparedConfig(
        workdir=resolved.workdir,
        secrets_dir=resolved.secrets_dir,
        param_path=resolved.param_path,
        abiss_home=resolved.abiss_home,
        param_payload=dict(resolved.param),
        root_tag=root_tag,
        top_mip=top_mip,
        runtime_secrets_dir=runtime_cloudvolume_root / "secrets",
        output_layer_spec=(
            OutputLayerSpec(
                resolution_xyz=affinity_metadata.resolution_xyz,
                chunk_size_xyz=_output_chunk_size(
                    resolved.param.get("CHUNK_SIZE"), affinity_metadata.chunk_size_xyz
                ),
            )
            if affinity_metadata is not None
            else None
        ),
        extra_env={"CLOUD_VOLUME_DIR": str(runtime_cloudvolume_root)},
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _abiss_build_id(abiss_home: Path) -> str:
    git_marker = abiss_home / ".git"
    if git_marker.exists():
        result = subprocess.run(
            ["git", "-C", str(abiss_home), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"git:{result.stdout.strip()}"

    ws_binary = abiss_home / "build" / "ws"
    if ws_binary.is_file():
        return f"ws-sha256:{_sha256_bytes(ws_binary.read_bytes())}"

    raise RuntimeError(
        f"Cannot identify ABISS build at {abiss_home}: expected its .git metadata or build/ws."
    )


def _expected_manifest(resolved: ResolvedReplay, *, abiss_build_id: str) -> dict[str, Any]:
    return {
        "provenance_sha": _sha256_bytes(resolved.provenance_path.read_bytes()),
        "whitelisted_param_sha": _json_sha(resolved.param),
        "abiss_build_id": abiss_build_id,
        "execution_bbox": list(resolved.execution_bbox),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Resume requires existing manifest {path}.") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {path} must contain a JSON object.")
    return manifest


def _check_manifest_compatibility(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"existing": actual.get(key), "requested": expected[key]}
        for key in MANIFEST_COMPATIBILITY_KEYS
        if actual.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Resume manifest is incompatible with this replay: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _apply_output_mode(resolved: ResolvedReplay, expected_manifest: Mapping[str, Any]) -> None:
    run_root = resolved.run_root
    if run_root.is_symlink():
        raise ValueError(f"Refusing to use symlinked replay output root {run_root}.")

    if resolved.mode == "fresh":
        if run_root.exists() and (not run_root.is_dir() or any(run_root.iterdir())):
            raise ValueError(
                f"Fresh replay requires an empty output namespace: {run_root}. "
                "Choose a new --name, --mode resume, or explicit --mode overwrite."
            )
    elif resolved.mode == "resume":
        if not run_root.is_dir():
            raise ValueError(f"Resume requires existing replay directory {run_root}.")
        _check_manifest_compatibility(_read_manifest(resolved.manifest_path), expected_manifest)
    elif resolved.mode == "overwrite":
        if resolved.out_root == Path(resolved.out_root.anchor):
            raise ValueError(
                "Refusing destructive overwrite directly below filesystem root "
                f"{resolved.out_root}."
            )
        if run_root.exists():
            if not run_root.is_dir():
                raise ValueError(f"Replay output root exists and is not a directory: {run_root}.")
            shutil.rmtree(run_root)
    else:  # pragma: no cover - resolve_replay validates this invariant.
        raise AssertionError(f"Unexpected mode {resolved.mode!r}")


def _write_manifest(resolved: ResolvedReplay, expected: Mapping[str, Any]) -> None:
    resolved.run_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(expected)
    manifest.update(
        {
            "mode": resolved.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = resolved.manifest_path.with_name(".manifest.json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, resolved.manifest_path)


def _has_explicit_gs_credentials(secrets_dir: Path) -> bool:
    if secrets_dir.is_dir() and any(path.is_file() for path in secrets_dir.iterdir()):
        return True
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    return bool(credentials and Path(credentials).expanduser().is_file())


@contextmanager
def _cloudvolume_root_view(secrets_dir: Path) -> Iterator[Path]:
    """Expose any configured secrets directory at CloudVolume's canonical path."""

    if secrets_dir.name == "secrets":
        yield secrets_dir.parent
        return

    with tempfile.TemporaryDirectory(prefix="pytc-seuron-cloudvolume-") as temporary:
        root = Path(temporary)
        canonical_secrets = root / "secrets"
        if secrets_dir.is_dir():
            canonical_secrets.symlink_to(secrets_dir.resolve(), target_is_directory=True)
        else:
            canonical_secrets.mkdir()
        yield root


def _preflight_abiss(resolved: ResolvedReplay) -> None:
    missing = [
        path
        for path in (
            resolved.abiss_home / "scripts" / "run_batch.sh",
            resolved.abiss_home / "scripts" / "remap_batch.sh",
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "ABISS installation is incomplete; missing " + ", ".join(map(str, missing))
        )


def _preflight_affinity(
    resolved: ResolvedReplay,
    *,
    cloudvolume_root: Path | None = None,
) -> AffinityMetadata:
    if cloudvolume_root is None:
        with _cloudvolume_root_view(resolved.secrets_dir) as root:
            return _preflight_affinity(resolved, cloudvolume_root=root)

    affinity_path = str(resolved.param["AFF_PATH"])
    parsed = urlparse(affinity_path)
    if parsed.scheme == "file":
        local_path = Path(unquote(parsed.path))
        if not local_path.exists():
            raise RuntimeError(f"Local affinity precomputed layer does not exist: {local_path}")
    elif parsed.scheme == "gs" and not _has_explicit_gs_credentials(resolved.secrets_dir):
        raise RuntimeError(
            f"No GCS credentials found in {resolved.secrets_dir}. Provide --secrets-dir "
            "or use --mirror /path/to/local/precomputed/crop."
        )

    requested_resolution = tuple(int(value) for value in resolved.param["AFF_RESOLUTION"])
    if len(requested_resolution) != 3:
        raise ValueError(
            "AFF_RESOLUTION must contain three XYZ values, got "
            f"{resolved.param['AFF_RESOLUTION']!r}."
        )
    requested_mip = int(resolved.param["AFF_MIP"])

    old_cloudvolume_root = os.environ.get("CLOUD_VOLUME_DIR")
    os.environ["CLOUD_VOLUME_DIR"] = str(cloudvolume_root)
    try:
        from cloudvolume import CloudVolume

        try:
            volume = CloudVolume(
                affinity_path,
                mip=list(requested_resolution),
                bounded=True,
                fill_missing=False,
                progress=False,
            )
            actual_mip = int(volume.mip)
            actual_resolution = tuple(int(value) for value in volume.resolution[:3])
            volume_bbox = tuple(int(value) for value in volume.bounds.minpt[:3]) + tuple(
                int(value) for value in volume.bounds.maxpt[:3]
            )
            chunk_size = tuple(int(value) for value in volume.chunk_size[:3])
        except Exception as exc:
            raise RuntimeError(
                f"Affinity preflight failed for {affinity_path}: {exc}. Check credentials "
                "or use --mirror /path/to/local/precomputed/crop."
            ) from exc

        if actual_mip != requested_mip:
            raise ValueError(
                f"AFF_RESOLUTION {list(requested_resolution)} selects mip {actual_mip}, "
                f"but provenance declares AFF_MIP {requested_mip}."
            )
        if actual_resolution != requested_resolution:
            raise ValueError(
                f"Affinity resolution is {actual_resolution}, expected {requested_resolution}."
            )
        if not _contains(volume_bbox, resolved.execution_bbox):
            raise ValueError(
                f"execution bbox {list(resolved.execution_bbox)} is outside affinity bounds "
                f"{list(volume_bbox)}."
            )

        x, y, z, x1, y1, z1 = resolved.execution_bbox
        # Read one voxel, not just metadata, so inaccessible buckets and incomplete
        # mirrors fail before output directories or manifests are created. Reading
        # both corners also catches a locally mirrored crop missing an edge chunk.
        try:
            _ = volume[x : x + 1, y : y + 1, z : z + 1]
            _ = volume[x1 - 1 : x1, y1 - 1 : y1, z1 - 1 : z1]
        except Exception as exc:
            raise RuntimeError(
                f"Affinity preflight failed for {affinity_path}: {exc}. Check credentials "
                "or use --mirror /path/to/local/precomputed/crop."
            ) from exc
        if len(chunk_size) != 3:
            raise ValueError(f"Affinity chunk size must have three XYZ values, got {chunk_size}.")
        metadata = AffinityMetadata(
            resolution_xyz=(
                actual_resolution[0],
                actual_resolution[1],
                actual_resolution[2],
            ),
            chunk_size_xyz=(chunk_size[0], chunk_size[1], chunk_size[2]),
        )
    finally:
        if old_cloudvolume_root is None:
            os.environ.pop("CLOUD_VOLUME_DIR", None)
        else:
            os.environ["CLOUD_VOLUME_DIR"] = old_cloudvolume_root
    return metadata


def execute_replay(resolved: ResolvedReplay) -> RunResult:
    """Preflight, enforce namespace safety, write a manifest, and run ABISS."""

    _preflight_abiss(resolved)
    with _cloudvolume_root_view(resolved.secrets_dir) as cloudvolume_root:
        affinity_metadata = _preflight_affinity(
            resolved,
            cloudvolume_root=cloudvolume_root,
        )
        build_id = _abiss_build_id(resolved.abiss_home)
        expected_manifest = _expected_manifest(resolved, abiss_build_id=build_id)
        _apply_output_mode(resolved, expected_manifest)
        _write_manifest(resolved, expected_manifest)
        return run_abiss_large(
            prepare_execution(resolved, affinity_metadata),
            execute=True,
        )


def _resolution_report(resolved: ResolvedReplay, *, execute: bool) -> dict[str, Any]:
    output_paths = {
        key: resolved.param[key] for key in sorted(GENERATED_OUTPUT) if key in resolved.param
    }
    input_paths = {
        key: resolved.param[key] for key in ("AFF_PATH", "IMAGE_PATH") if key in resolved.param
    }
    return {
        "backend": resolved.backend,
        "chunk_grid_xyz": list(resolved.chunk_grid),
        "data": resolved.data,
        "deferred_igneous_tasks": [dict(task) for task in resolved.replay_spec.igneous_blocks],
        "execute": execute,
        "execution_bbox_xyz": list(resolved.execution_bbox),
        "input_paths": input_paths,
        "mode": resolved.mode,
        "name": resolved.name,
        "output_paths": output_paths,
        "param": resolved.param,
        "param_path": str(resolved.param_path),
        "provenance_path": str(resolved.provenance_path),
        "score_bbox_xyz": list(resolved.score_bbox) if resolved.score_bbox else None,
        "workdir": str(resolved.workdir),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file with a seuron_replay block")
    parser.add_argument("--out-root", help="Fresh replay namespace parent")
    parser.add_argument("--name", help="Replay namespace name")
    parser.add_argument("--secrets-dir", help="CloudVolume secrets directory")
    parser.add_argument("--abiss-home", help="ABISS checkout/build root")
    parser.add_argument(
        "--exec-bbox", nargs=6, type=int, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1")
    )
    parser.add_argument(
        "--score-bbox", nargs=6, type=int, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1")
    )
    parser.add_argument("--aff-path", help="Highest-precedence affinity CloudVolume path")
    parser.add_argument(
        "--mirror",
        help="Existing local precomputed affinity crop used instead of provenance AFF_PATH",
    )
    parser.add_argument(
        "--mode",
        choices=("fresh", "resume", "overwrite"),
        help="Execution output policy (default: fresh)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Alias for the explicit destructive --mode overwrite",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run ABISS after preflight; omission performs a pure resolve",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        resolved = resolve_replay(args)
        print(
            json.dumps(_resolution_report(resolved, execute=args.execute), indent=2, sort_keys=True)
        )
        if not args.execute:
            return 0
        result = execute_replay(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Manifest: {resolved.manifest_path}")
    print(f"Final segmentation layer: {result.seg_cloudpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
