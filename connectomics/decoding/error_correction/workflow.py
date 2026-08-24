"""YAML-driven runner for whole-volume morphology error correction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import load_frozen_merge_roots, reject_evaluation_path
from ...utils.yaml_config import load_yaml_with_bases_and_params

STAGES = (
    "sizes",
    "skeletonize",
    "skeletons",
    "contacts",
    "contact_graph",
    "candidates",
    "junction_scope",
    "junction_features",
    "boundary",
    "resolve",
    "prepare_output",
    "postprocess",
    "verify",
)
ARRAY_STAGES = frozenset({"skeletonize", "contacts", "postprocess"})
ROOT_KEYS = {
    "version",
    "segmentation",
    "affinity_chunks",
    "keep_mask",
    "nucleus_manifest",
    "size_glob",
    "workdir",
    "output_segmentation",
    "task_count",
    "expected_chunks",
    "expected_size_files",
    "volume_shape_zyx",
    "affinity_chunk_size",
    "restore_sigmoid_scale",
    "contact_z_slab",
    "core_xyz",
    "halo_xyz",
    "downsample_zyx",
    "min_contact_voxels",
    "min_skeleton_voxels",
    "skeleton_parallel",
    "skeleton_spacing_nm",
    "junction_workers",
    "erosion_radius_zyx",
    "parameters",
}
PARAMETER_KEYS = {
    "morphology",
    "candidates",
    "junction",
    "resolve_v2",
    "resolve_v3",
    "resolve_v4",
    "resolve_v5",
    "resolve_v7",
}
PARAMETER_FIELDS = {
    "morphology": {
        "max_port_distance_nm",
        "prune_nm",
        "tangent_nm",
        "twig_nm",
        "branch_profile_nm",
        "spine_radius_ratio",
        "spine_parent_collinearity",
        "spine_parent_radius_balance",
        "spine_perpendicularity",
    },
    "candidates": {"min_segment_length_nm"},
    "junction": {"min_affinity"},
    "resolve_v2": {
        "scope_min_affinity_ge08_fraction",
        "min_affinity_mean",
        "min_affinity_ge09_fraction",
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_turn_short_deg",
        "min_radius_ratio",
        "local_competition_nm",
        "max_component_segments",
        "min_internal_source_length_nm",
        "min_internal_host_length_nm",
        "min_internal_host_ratio",
        "min_internal_affinity_mean",
        "min_internal_affinity_ge08_fraction",
        "min_internal_affinity_ge09_fraction",
    },
    "resolve_v3": {
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_turn_short_deg",
        "min_radius_ratio",
        "min_connector_length_nm",
        "max_connector_length_nm",
        "min_connector_host_length_nm",
        "min_connector_host_ratio",
        "min_connector_affinity_mean",
        "min_connector_affinity_ge08_fraction",
        "min_connector_affinity_ge09_fraction",
        "min_connector_end_separation_nm",
        "min_connector_end_separation_ratio",
        "max_component_segments",
    },
    "resolve_v4": {
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_turn_short_deg",
        "min_radius_ratio",
        "min_internal_source_length_nm",
        "max_internal_source_length_nm",
        "min_internal_host_length_nm",
        "min_internal_host_ratio",
        "min_internal_affinity_mean",
        "min_internal_affinity_ge08_fraction",
        "min_internal_affinity_ge09_fraction",
        "max_component_segments",
    },
    "resolve_v5": {
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_turn_short_deg",
        "min_radius_ratio",
        "min_source_length_nm",
        "min_nucleus_host_length_nm",
        "min_affinity_mean",
        "min_affinity_ge08_fraction",
        "min_affinity_ge09_fraction",
        "max_component_segments",
    },
    "resolve_v7": {
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_multiscale_turn_deg",
        "min_radius_ratio",
        "min_segment_length_nm",
        "min_affinity_mean",
        "min_affinity_ge09_fraction",
        "local_competition_nm",
        "max_component_segments",
    },
}
REPO = Path(__file__).resolve().parents[3]


def _triple(value: Any, name: str, *, allow_zero: bool = False) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    result = tuple(int(item) for item in value)
    if any(item < 0 if allow_zero else item <= 0 for item in result):
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {adjective}")
    return result[0], result[1], result[2]


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    reject_evaluation_path(value)
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


@dataclass(frozen=True)
class ErrorCorrectionConfig:
    segmentation: Path
    affinity_chunks: Path
    keep_mask: Path
    nucleus_manifest: Path
    size_glob: str
    workdir: Path
    output_segmentation: Path
    task_count: int
    expected_chunks: int
    expected_size_files: int | None
    volume_shape_zyx: tuple[int, int, int]
    affinity_chunk_size: int
    restore_sigmoid_scale: float
    contact_z_slab: int
    core_xyz: tuple[int, int, int]
    halo_xyz: tuple[int, int, int]
    downsample_zyx: tuple[int, int, int]
    min_contact_voxels: int
    min_skeleton_voxels: int
    skeleton_parallel: int
    skeleton_spacing_nm: float
    junction_workers: int
    erosion_radius_zyx: tuple[int, int, int]
    parameters: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "ErrorCorrectionConfig":
        payload = load_yaml_with_bases_and_params(path)
        if not isinstance(payload, dict) or set(payload) != {"error_correction"}:
            raise ValueError("config must have exactly one top-level 'error_correction' key")
        data = payload["error_correction"]
        if not isinstance(data, dict):
            raise ValueError("error_correction must be a mapping")
        unknown = set(data) - ROOT_KEYS
        if unknown:
            raise ValueError(f"unknown error_correction keys: {sorted(unknown)}")
        if int(data.get("version", 0)) != 1:
            raise ValueError("error_correction.version must be 1")
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("error_correction.parameters must be a mapping")
        unknown_parameters = set(parameters) - PARAMETER_KEYS
        if unknown_parameters:
            raise ValueError(f"unknown parameter groups: {sorted(unknown_parameters)}")
        for group, values in parameters.items():
            if not isinstance(values, dict):
                raise ValueError(f"parameters.{group} must be a mapping")
            unknown_fields = set(values) - PARAMETER_FIELDS[group]
            if unknown_fields:
                raise ValueError(f"unknown parameters.{group} keys: {sorted(unknown_fields)}")
        size_glob = data.get("size_glob")
        if not isinstance(size_glob, str) or not size_glob:
            raise ValueError("size_glob must be a nonempty string")
        reject_evaluation_path(size_glob)
        if not Path(size_glob).is_absolute():
            size_glob = str(REPO / size_glob)
        result = cls(
            segmentation=_path(data.get("segmentation"), "segmentation"),
            affinity_chunks=_path(data.get("affinity_chunks"), "affinity_chunks"),
            keep_mask=_path(data.get("keep_mask"), "keep_mask"),
            nucleus_manifest=_path(data.get("nucleus_manifest"), "nucleus_manifest"),
            size_glob=size_glob,
            workdir=_path(data.get("workdir"), "workdir"),
            output_segmentation=_path(data.get("output_segmentation"), "output_segmentation"),
            task_count=int(data.get("task_count", 1)),
            expected_chunks=int(data.get("expected_chunks", 0)),
            expected_size_files=(
                None
                if data.get("expected_size_files") is None
                else int(data["expected_size_files"])
            ),
            volume_shape_zyx=_triple(data.get("volume_shape_zyx"), "volume_shape_zyx"),
            affinity_chunk_size=int(data.get("affinity_chunk_size", 1008)),
            restore_sigmoid_scale=float(data.get("restore_sigmoid_scale", 0.2)),
            contact_z_slab=int(data.get("contact_z_slab", 8)),
            core_xyz=_triple(data.get("core_xyz"), "core_xyz"),
            halo_xyz=_triple(data.get("halo_xyz"), "halo_xyz", allow_zero=True),
            downsample_zyx=_triple(data.get("downsample_zyx"), "downsample_zyx"),
            min_contact_voxels=int(data.get("min_contact_voxels", 200)),
            min_skeleton_voxels=int(data.get("min_skeleton_voxels", 100_000)),
            skeleton_parallel=int(data.get("skeleton_parallel", 8)),
            skeleton_spacing_nm=float(data.get("skeleton_spacing_nm", 250.0)),
            junction_workers=int(data.get("junction_workers", 8)),
            erosion_radius_zyx=_triple(
                data.get("erosion_radius_zyx", [0, 0, 0]),
                "erosion_radius_zyx",
                allow_zero=True,
            ),
            parameters=parameters,
        )
        if (
            result.task_count <= 0
            or result.expected_chunks <= 0
            or result.affinity_chunk_size <= 0
            or result.contact_z_slab <= 0
            or result.skeleton_spacing_nm <= 0
            or result.restore_sigmoid_scale <= 0
        ):
            raise ValueError("task/chunk/spacing/scale values must be positive")
        if result.min_contact_voxels < 200:
            raise ValueError("min_contact_voxels cannot be below the ABISS dust floor (200)")
        return result

    def artifact(self, name: str) -> Path:
        paths = {
            "sizes": self.workdir / "segment_sizes.data",
            "skeleton_chunks": self.workdir / "skeleton_chunks",
            "graph": self.workdir / "segment_skeleton_graph.h5",
            "stitch": self.workdir / "stitch_edges.npz",
            "features": self.workdir / "segment_morphology.npz",
            "endpoints": self.workdir / "segment_endpoints.npz",
            "interiors": self.workdir / "segment_interiors.npz",
            "contact_chunks": self.workdir / "contact_chunks",
            "contacts": self.workdir / "contact_graph.npz",
            "candidates": self.workdir / "contact_merge_candidates.npz",
            "candidate_report": self.workdir / "contact_merge_audit.json",
            "scope": self.workdir / "junction_scope.npz",
            "junctions": self.workdir / "junction_features_raw.npz",
            "skeleton_cache": self.workdir / "skeleton_cache",
            "boundary": self.workdir / "boundary_inventory.npz",
        }
        return paths[name]

    def resolve_artifacts(self, version: str) -> dict[str, Path]:
        root = self.workdir / "resolver" / version
        return {
            "audit": root / "junction_merge_candidates.npz",
            "report": root / "junction_merge_audit.json",
            "proposals": root / "frozen_junction_merges.npz",
            "frozen_report": root / "frozen_junction_merges.json",
        }


def _option(name: str) -> str:
    return "--" + name.replace("_", "-")


def _parameter_args(values: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        flag = _option(name)
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.extend([flag, *(str(item) for item in value)])
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def _module(name: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        "-m",
        f"connectomics.decoding.error_correction.{name}",
        *(str(value) for value in arguments),
    ]


def _resolve_commands(config: ErrorCorrectionConfig) -> list[list[str]]:
    common = [
        "--candidates",
        config.artifact("candidates"),
        "--junctions",
        config.artifact("junctions"),
        "--nucleus-manifest",
        config.nucleus_manifest,
    ]
    commands: list[list[str]] = []
    previous: dict[str, Path] | None = None
    for version, module_name in (
        ("v2", "resolve_v2"),
        ("v3", "resolve_v3"),
        ("v4", "resolve_v4"),
        ("v5", "resolve_v5"),
        ("v7", "resolve_v7"),
    ):
        current = config.resolve_artifacts(version)
        args: list[object] = [*common]
        if version in {"v2", "v3", "v4"}:
            args.extend(["--boundary-inventory", config.artifact("boundary")])
        if previous is not None:
            args.extend(
                [
                    "--base-proposals",
                    previous["proposals"],
                    "--base-report",
                    previous["frozen_report"],
                ]
            )
        args.extend(
            [
                "--audit",
                current["audit"],
                "--report",
                current["report"],
                "--proposals",
                current["proposals"],
                "--frozen-report",
                current["frozen_report"],
                *_parameter_args(config.parameters.get(f"resolve_{version}", {})),
                "--freeze",
            ]
        )
        commands.append(_module(module_name, *args))
        previous = current
    return commands


def stage_commands(
    config: ErrorCorrectionConfig,
    stage: str,
    *,
    task_id: int,
    num_tasks: int,
    overwrite: bool,
    max_owned_chunks: int | None,
) -> list[list[str]]:
    a = config.artifact
    overwrite_args = ["--overwrite"] if overwrite else []
    limit_args = (
        ["--max-owned-chunks", str(max_owned_chunks)] if max_owned_chunks is not None else []
    )
    if stage == "sizes":
        args: list[object] = ["--input-glob", config.size_glob, "--output", a("sizes")]
        if config.expected_size_files is not None:
            args.extend(["--expected-files", config.expected_size_files])
        return [_module("sizes", *args)]
    if stage == "skeletonize":
        return [
            _module(
                "skeletonize",
                "--seg",
                config.segmentation,
                "--sizes",
                a("sizes"),
                "--output",
                a("skeleton_chunks"),
                "--min-global-voxels",
                config.min_skeleton_voxels,
                "--core-xyz",
                *config.core_xyz,
                "--halo-xyz",
                *config.halo_xyz,
                "--downsample-zyx",
                *config.downsample_zyx,
                "--task-id",
                task_id,
                "--num-tasks",
                num_tasks,
                "--parallel",
                config.skeleton_parallel,
                "--skeleton-spacing-nm",
                config.skeleton_spacing_nm,
                *limit_args,
                *overwrite_args,
            )
        ]
    if stage == "skeletons":
        return [
            _module(
                "morphology",
                "--stage",
                "all",
                "--chunks",
                a("skeleton_chunks"),
                "--graph",
                a("graph"),
                "--stitch",
                a("stitch"),
                "--sizes",
                a("sizes"),
                "--features",
                a("features"),
                "--endpoints",
                a("endpoints"),
                "--interiors",
                a("interiors"),
                "--nucleus-targets",
                config.nucleus_manifest,
                "--expected-chunks",
                config.expected_chunks,
                *_parameter_args(config.parameters.get("morphology", {})),
            )
        ]
    if stage == "contacts":
        return [
            _module(
                "contacts",
                "--seg",
                config.segmentation,
                "--sizes",
                a("sizes"),
                "--affinity",
                config.affinity_chunks,
                "--keep-mask",
                config.keep_mask,
                "--output",
                a("contact_chunks"),
                "--min-global-voxels",
                config.min_contact_voxels,
                "--core-xyz",
                *config.core_xyz,
                "--volume-shape-zyx",
                *config.volume_shape_zyx,
                "--affinity-chunk-size",
                config.affinity_chunk_size,
                "--restore-sigmoid-scale",
                config.restore_sigmoid_scale,
                "--z-slab",
                config.contact_z_slab,
                "--task-id",
                task_id,
                "--num-tasks",
                num_tasks,
                *limit_args,
                *overwrite_args,
            )
        ]
    if stage == "contact_graph":
        return [
            _module(
                "aggregate_contacts",
                "--chunks",
                a("contact_chunks"),
                "--output",
                a("contacts"),
                "--expected-chunks",
                config.expected_chunks,
            )
        ]
    if stage == "candidates":
        return [
            _module(
                "candidates",
                "--features",
                a("features"),
                "--endpoints",
                a("endpoints"),
                "--interiors",
                a("interiors"),
                "--contacts",
                a("contacts"),
                "--candidates",
                a("candidates"),
                "--audit-report",
                a("candidate_report"),
                *_parameter_args(config.parameters.get("candidates", {})),
            )
        ]
    if stage == "junction_scope":
        return [
            _module(
                "junction_features",
                "decoder-scope",
                "--candidates",
                a("candidates"),
                "--out",
                a("scope"),
                *_parameter_args(config.parameters.get("junction", {})),
            )
        ]
    if stage == "junction_features":
        return [
            _module(
                "junction_features",
                "compute",
                "--scope-path",
                a("scope"),
                "--out",
                a("junctions"),
                "--graph",
                a("graph"),
                "--stitch",
                a("stitch"),
                "--cache-dir",
                a("skeleton_cache"),
                "--workers",
                config.junction_workers,
            )
        ]
    if stage == "boundary":
        return [
            _module(
                "boundary",
                "--contacts",
                a("contacts"),
                "--candidates",
                a("candidates"),
                "--junctions",
                a("junctions"),
                "--output",
                a("boundary"),
            )
        ]
    if stage == "resolve":
        return _resolve_commands(config)
    final = config.resolve_artifacts("v7")
    postprocess_common: list[object] = [
        "--source",
        config.segmentation,
        "--output",
        config.output_segmentation,
        "--workdir",
        config.workdir,
        "--core-xyz",
        *config.core_xyz,
        "--erosion-radius-zyx",
        *config.erosion_radius_zyx,
    ]
    if stage == "prepare_output":
        return [_module("postprocess", "prepare", *postprocess_common)]
    proposal_args: list[object] = [
        "--proposals",
        final["proposals"],
        "--proposal-report",
        final["frozen_report"],
    ]
    if stage == "postprocess":
        return [
            _module(
                "postprocess",
                "run",
                *postprocess_common,
                *proposal_args,
                "--task-id",
                task_id,
                "--num-tasks",
                num_tasks,
                *limit_args,
                *overwrite_args,
            )
        ]
    if stage == "verify":
        return [_module("postprocess", "verify", *postprocess_common, *proposal_args)]
    raise ValueError(f"unknown stage: {stage}")


def _remove_resolver_outputs(config: ErrorCorrectionConfig) -> None:
    for version in ("v2", "v3", "v4", "v5", "v7"):
        for path in config.resolve_artifacts(version).values():
            if path.is_file():
                path.unlink()


def run_stage(
    config: ErrorCorrectionConfig,
    stage: str,
    *,
    task_id: int,
    num_tasks: int,
    overwrite: bool,
    max_owned_chunks: int | None,
    dry_run: bool,
) -> None:
    if stage == "resolve":
        final = config.resolve_artifacts("v7")
        if (
            final["proposals"].is_file()
            and final["frozen_report"].is_file()
            and not overwrite
            and not dry_run
        ):
            load_frozen_merge_roots(final["proposals"], final["frozen_report"])
            print("frozen v7 proposal exists and passed provenance checks; skip")
            return
        if overwrite and not dry_run:
            _remove_resolver_outputs(config)
    commands = stage_commands(
        config,
        stage,
        task_id=task_id,
        num_tasks=num_tasks,
        overwrite=overwrite,
        max_owned_chunks=max_owned_chunks,
    )
    for command in commands:
        print(" ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=REPO, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--max-owned-chunks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = ErrorCorrectionConfig.load(args.config)
    num_tasks = args.num_tasks or config.task_count
    if not 0 <= args.task_id < num_tasks:
        raise ValueError("task-id must satisfy 0 <= task-id < num-tasks")
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.stage == "all" and num_tasks != 1:
        raise ValueError("--stage all is serial; use --num-tasks 1 or run array stages separately")
    print(
        json.dumps(
            {
                "config": str(args.config.resolve()),
                "stages": stages,
                "task_id": args.task_id,
                "num_tasks": num_tasks,
                "gt_free": True,
            },
            indent=2,
        )
    )
    for stage in stages:
        run_stage(
            config,
            stage,
            task_id=args.task_id,
            num_tasks=num_tasks,
            overwrite=args.overwrite,
            max_owned_chunks=args.max_owned_chunks,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
