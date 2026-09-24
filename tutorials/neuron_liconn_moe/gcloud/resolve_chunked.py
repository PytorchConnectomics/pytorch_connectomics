#!/usr/bin/env python3
"""Pure resolution and I/O-only preflight for the S3 chunked ABISS run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

import h5py
import numpy as np

CHUNKED_STAGES = ("watershed", "remap_watershed", "agglomerate_mean_edge", "remap_agglomeration")
CRITERIA = ("mean",)
VARIANT_CHUNKS = {
    "A": ([256, 256, 128], [128, 128, 128]),
    "B": ([192, 192, 96], [64, 64, 96]),
}


def artifact_bbox(path: Path) -> list[int]:
    """Return the XYZ BBOX implied by a canonical CZYX artifact."""
    with h5py.File(path, "r") as handle:
        if "main" not in handle or handle["main"].ndim != 4:
            raise ValueError(f"{path} must contain a 4-D main dataset")
        _, zdim, ydim, xdim = handle["main"].shape
    return [0, 0, 0, int(xdim), int(ydim), int(zdim)]


def variant_specs(bbox: Sequence[int]) -> dict[str, tuple[list[int], list[int]]]:
    dims = [int(bbox[3]), int(bbox[4]), int(bbox[5])]
    return {"one": (dims, [128, 128, 128]), **VARIANT_CHUNKS}


def stages_for_criterion(criterion: str) -> tuple[str, ...]:
    _criterion_guard(criterion)
    return CHUNKED_STAGES


def _ints(values: Sequence[Any], name: str, n: int = 3) -> list[int]:
    if len(values) != n:
        raise ValueError(f"{name} must have {n} values, got {list(values)}")
    return [int(v) for v in values]


def _criterion_guard(value: str) -> None:
    if value not in CRITERIA:
        stages = ", ".join(CHUNKED_STAGES)
        extra = " The chunked path ships no max binary; whole-volume ws accepts max." if value == "max" else ""
        raise ValueError(
            f"Requested merge criterion {value!r} is unavailable for chunked ABISS. "
            f"Accepted criteria: {{{', '.join(CRITERIA)}}}; backing stage: "
            f"agglomerate_mean_edge; available chunked stages: {stages}.{extra}"
        )


def _alignment_guard(chunk: Sequence[int], storage: Sequence[int]) -> None:
    chunk = _ints(chunk, "param.CHUNK_SIZE")
    storage = _ints(storage, "seg_chunk_size_xyz")
    for axis, name in enumerate("XYZ"):
        if chunk[axis] % storage[axis] != 0:
            raise ValueError(
                f"param.CHUNK_SIZE {chunk} is not aligned with seg_chunk_size_xyz {storage} "
                f"on axis {name}: {chunk[axis]} is not a multiple of {storage[axis]}."
            )


def resolve(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve S3 settings without touching the filesystem, network, or outputs."""
    criterion = str(config.get("merge_criterion", "mean"))
    stages_for_criterion(criterion)
    chunk = _ints(config["CHUNK_SIZE"], "param.CHUNK_SIZE")
    storage = _ints(config["seg_chunk_size_xyz"], "seg_chunk_size_xyz")
    bbox = _ints(config["BBOX"], "BBOX", 6)
    _alignment_guard(chunk, storage)
    counts = [(bbox[i + 3] - bbox[i]) + chunk[i] - 1 for i in range(3)]
    counts = [counts[i] // chunk[i] for i in range(3)]
    if any(v < 2 for v in counts):
        raise ValueError(
            f"param.CHUNK_SIZE {chunk} yields only {counts} chunks for BBOX {bbox}; "
            "at least two chunks per axis are required."
        )
    channels = [int(v) for v in config.get("AFF_CHANNELS", [0, 1, 2])]
    if channels != [0, 1, 2]:
        raise ValueError(f"AFF_CHANNELS must be [0, 1, 2] for aff_canon.h5, got {channels}.")
    return {**dict(config), "merge_criterion": criterion, "chunked_stages": stages_for_criterion(criterion),
            "CHUNK_SIZE": chunk,
            "seg_chunk_size_xyz": storage, "BBOX": bbox, "AFF_CHANNELS": channels,
            "chunk_counts": counts}


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read and validate the canonical affinity artifact header."""
    path = Path(str(config["affinity_h5"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical affinity artifact does not exist: {path}")
    bbox = _ints(config["BBOX"], "BBOX", 6)
    with h5py.File(path, "r") as handle:
        if "main" not in handle:
            raise ValueError(f"Canonical affinity artifact {path} is missing dataset main.")
        ds = handle["main"]
        expected = (3, bbox[5] - bbox[2], bbox[4] - bbox[1], bbox[3] - bbox[0])
        if tuple(ds.shape) != expected or ds.dtype != "float32":
            raise ValueError(
                f"Canonical affinity main must be float32 {expected}, got {ds.dtype} {ds.shape}."
            )
        values = ds[...]
        exposed = np.ones(values.shape, dtype=bool)
        exposed[0, :, :, 0] = False
        exposed[1, :, 0, :] = False
        exposed[2, 0, :, :] = False
        active = values[exposed]
        saturated = int(np.count_nonzero(active == 1.0))
        saturated_fraction = saturated / max(1, int(active.size))
        if active.size == 0 or float(active.min()) <= 0.0 or float(active.max()) > 1.0:
            raise ValueError(
                "Canonical affinity values off the three destination faces must satisfy "
                f"0 < min <= max <= 1; got min={float(active.min()):.8g}, "
                f"max={float(active.max()):.8g}; see "
                f"{config.get('diagnostic_path', 'the affinity diagnostic JSON')}."
            )
    return {**dict(config), "affinity_h5": str(path.resolve()),
            "saturated_count": saturated, "saturated_fraction": saturated_fraction}


def write_affinity_diagnostic(
    path: Path, output: Path, *, ws_high: float | None = None, ws_low: float | None = None
) -> dict[str, Any]:
    """Record saturation plus the resolved thresholds actually passed to ``ws``."""
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["main"][:], dtype=np.float32)
    exposed = np.ones(values.shape, dtype=bool)
    exposed[0, :, :, 0] = False
    exposed[1, :, 0, :] = False
    exposed[2, 0, :, :] = False
    active = values[exposed]
    p20, p94 = (float(v) for v in np.percentile(active, [20, 94]))
    diagnostic = {
        "artifact": str(path.resolve()),
        "dtype": "float32",
        "saturated_fraction": float(np.mean(active == 1.0)),
        "saturated_count": int(np.count_nonzero(active == 1.0)),
        "active_count": int(active.size),
        "percentile_20": p20,
        "percentile_94": p94,
        "percentile_20_strictly_inside_unsaturated_range": 0.0 < p20 < 1.0,
        "percentile_94_strictly_inside_unsaturated_range": 0.0 < p94 < 1.0,
        "unsaturated_range": "(0, 1)",
    }
    if ws_high is not None:
        diagnostic["ws_high"] = float(ws_high)
    if ws_low is not None:
        diagnostic["ws_low"] = float(ws_low)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return diagnostic


def resolve_variant(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Resolve one named candidate; its chunk size is part of the candidate."""
    bbox = _ints(config["BBOX"], "BBOX", 6)
    specs = variant_specs(bbox)
    if variant not in specs:
        raise ValueError(f"unknown chunked variant {variant!r}; expected one of {tuple(specs)}")
    chunk, storage = specs[variant]
    resolved = dict(config)
    resolved.update({"CHUNK_SIZE": chunk, "seg_chunk_size_xyz": storage})
    if variant != "one":
        return resolve(resolved)
    _criterion_guard(str(resolved.get("merge_criterion", "mean")))
    _ints(resolved["BBOX"], "BBOX", 6)
    return {**resolved, "chunked_stages": CHUNKED_STAGES,
            "CHUNK_SIZE": [int(v) for v in chunk],
            "seg_chunk_size_xyz": [int(v) for v in storage]}


def write_variant_config(
    template: Path, output: Path, *, run_prefix: Path, affinity_h5: Path,
    variant: str, ws_high: float, ws_low: float,
    bbox: Sequence[int], criterion: str | None = None,
) -> Path:
    """Materialize a strict ``abiss_chunk`` YAML for one candidate."""
    import yaml

    raw = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "abiss_chunk" not in raw:
        raise ValueError(f"{template} must contain an abiss_chunk section")
    ab = raw["abiss_chunk"]
    chunk, storage = variant_specs(bbox)[variant]
    root = run_prefix / f"chunked_{variant}"
    ab.update({
        "workdir": str(root / "work"),
        "secrets_dir": str(root / "secrets"),
        "param_path": str(root / "secrets" / "param"),
        "source_affinity_h5": str(affinity_h5),
        "seg_chunk_size_xyz": storage,
    })
    param = ab.setdefault("param", {})
    param.update({
        "BBOX": [int(v) for v in bbox],
        "CHUNK_SIZE": chunk,
        "WS_HIGH_THRESHOLD": float(ws_high),
        "WS_LOW_THRESHOLD": float(ws_low),
        "AGG_THRESHOLD": float(param.get("AGG_THRESHOLD", 0.1394546)),
        "WS_PATH": f"file://{root / 'ws'}",
        "SEG_PATH": f"file://{root / 'seg'}",
        "SCRATCH_PATH": f"file://{root / 'scratch'}",
        "CHUNKMAP_OUTPUT": f"file://{root / 'chunkmap'}",
        "AFF_PATH": f"file://{root / 'aff'}",
        "AFF_CHANNELS": [0, 1, 2],
        "MERGE_CRITERION": str(criterion or ab.get("merge_criterion", raw.get("merge_criterion", "mean"))),
    })
    ab["abiss_home"] = os.environ.get("ABISS_HOME", str(ab.get("abiss_home", "/opt/abiss")))
    ab["aff_chunk_size_xyz"] = storage
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def _scratch_path(path: str | Path) -> Path:
    value = str(path)
    if value.startswith("file://"):
        return Path(url2pathname(urlparse(value).path))
    if "://" in value:
        raise ValueError(f"task-flag inspection requires a local SCRATCH_PATH, got {value}")
    return Path(value)


def task_flag_keys(statsd_prefix: str, tasks: Sequence[Sequence[str]]) -> set[str]:
    """Build ABISS's exact ``prefix_stage_op_chunk`` flag keys."""
    return {"_".join((statsd_prefix, str(stage), str(op), str(chunk)))
            for stage, op, chunk in tasks}


def read_task_flag_keys(scratch_path: str | Path) -> set[str]:
    """Read ``TASK_KEY`` names from ABISS's local ``done/*.txt`` markers."""
    flag_dir = _scratch_path(scratch_path) / "done"
    if not flag_dir.is_dir():
        return set()
    return {entry.name[:-4] for entry in flag_dir.iterdir()
            if entry.is_file() and entry.name.endswith(".txt")}


def compare_task_keys(expected: Sequence[str] | set[str], scratch_path: str | Path) -> None:
    """Compare expected ABISS keys with both directions of the stored key set."""
    expected_set = set(expected)
    actual_set = read_task_flag_keys(scratch_path)
    missing, extra = sorted(expected_set - actual_set), sorted(actual_set - expected_set)
    if missing or extra:
        raise ValueError(f"chunk task key mismatch: missing={missing}, extra={extra}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    import json
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(preflight(resolve(config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
