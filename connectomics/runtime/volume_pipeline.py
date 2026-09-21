"""Artifact-aware execution shared by the volume segmentation playbooks."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf

from connectomics.utils.yaml_config import load_yaml_with_bases_and_params

REPO = Path(__file__).resolve().parents[2]
FITTED_KEYS = ("WS_HIGH_THRESHOLD", "WS_LOW_THRESHOLD", "AGG_THRESHOLD", "CHUNK_SIZE")


def load_params(path: Path) -> DictConfig:
    """Read the tutorial's parameters with its local interpolations resolved."""
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    return config.params


def load_workflow_yaml(path: Path) -> DictConfig:
    """Resolve recursive workflow inheritance through the canonical YAML loader."""
    return OmegaConf.create(load_yaml_with_bases_and_params(path))


def load_pytc_config(path: Path):
    """Load a schema-backed tutorial through the repository config loader."""
    from connectomics.config import load_config

    return load_config(str(path))


def _fitted_owners(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Path]:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"cyclic YAML inheritance: {path}")
    raw = OmegaConf.load(path)
    bases = raw.get("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    owners = {}
    for base in bases:
        owners.update(_fitted_owners(path.parent / str(base), (*stack, path)))
    block = raw.get("abiss_chunk") or {}
    params = block.get("param") or {}
    owners.update({key: path for key in FITTED_KEYS if key in params})
    return owners


def require_dataset_values(config: Mapping, *, path: Path | None = None) -> None:
    """Reject missing/null fitted ABISS values and values inherited across datasets.

    Invoke on the resolved ``abiss_chunk`` block before ABISS preparation supplies
    defaults. A smoke recipe may inherit its dataset's full-volume recipe, but a
    shared base or another tutorial cannot supply fitted values.
    """
    params = config.get("param", {}) or {}
    absent = [key for key in FITTED_KEYS if key not in params]
    null = [key for key in FITTED_KEYS if key in params and params[key] is None]
    errors = []
    if absent:
        errors.append("absent: " + ", ".join(absent))
    if null:
        errors.append("null: " + ", ".join(null))
    if path is not None:
        owners = _fitted_owners(path)
        inherited = [
            f"{key} ({owner})"
            for key, owner in owners.items()
            if owner.parent != path.resolve().parent
        ]
        if inherited:
            errors.append("inherited outside dataset: " + ", ".join(inherited))
    if errors:
        location = f" in {path}" if path is not None else ""
        raise ValueError(f"required dataset ABISS values{location}: " + "; ".join(errors))


@dataclass
class Status:
    done: bool
    detail: str


def check_paths(*paths: Path) -> Status:
    missing = [p for p in paths if not p.exists()]
    if missing:
        return Status(False, "MISSING " + ", ".join(str(p) for p in missing))
    return Status(True, "found " + ", ".join(str(p) for p in paths))


def check_shards(out: Path, num_shards: int) -> Status:
    """Complete when every shard left its `.done.<id>` sentinel next to `out`.

    The mask zarrs have fill_value "keep", so an unwritten region is
    indistinguishable from a written one -- trusting the store would turn a dead
    shard into a silently wrong mask. The sentinels are the only honest signal.
    """
    if not out.exists():
        return Status(False, f"MISSING {out}")
    done = sum(1 for i in range(num_shards) if Path(f"{out}.done.{i}").exists())
    return Status(done == num_shards, f"{done}/{num_shards} shards done, {out}")


def check_download(store: Path, dataset: str, slab: int, tile: int) -> Status:
    """Complete when the progress sidecars account for every (z-slab x XY-tile) job."""
    array = store / dataset
    if not array.exists():
        return Status(False, f"MISSING {array}")
    import zarr  # noqa: PLC0415

    shape = zarr.open(str(store), mode="r")[dataset].shape
    tile = tile or max(shape[1], shape[2])
    expected = 1
    for size, step in zip(shape, (slab, tile, tile)):
        expected *= -(-size // step)
    done = set()
    for sidecar in glob.glob(f"{store}.progress*"):
        done |= set(Path(sidecar).read_text().split())
    return Status(len(done) >= expected, f"{len(done)}/{expected} blocks in {array}")


def check_checkpoint(save_path: Path, explicit: Path | None) -> Status:
    if explicit is not None:
        return check_paths(explicit)
    found = sorted(save_path.glob("*/checkpoints/*.ckpt"))
    if found:
        return Status(True, f"{len(found)} checkpoint(s), newest {found[-1]}")
    return Status(False, f"no *.ckpt under {save_path}/*/checkpoints/")


def check_affinity(save_path: Path, suffix: str = "") -> Status:
    """Complete when every chunk listed in the store's index.json is on disk.

    `suffix` is the config's `decoding.save_suffix`, which ends the store name.
    Matching on it keeps two windows' stores in one run directory apart -- and
    the match has to be anchored, because one suffix is a prefix of the other.
    """
    indexes = sorted(save_path.glob("**/*.h5.index.json"))
    if suffix:
        tail = f"_{suffix}.h5.index.json"
        indexes = [i for i in indexes if i.name.endswith(tail)] or indexes
    if not indexes:
        return Status(False, f"no *.h5.index.json under {save_path}")
    index = indexes[-1]
    store = index.parent / (index.name[: -len(".index.json")] + ".chunks")
    chunks = json.loads(index.read_text()).get("chunks", [])
    root = index.parent
    written = sum(1 for c in chunks if (root / c["path"]).exists())
    return Status(
        written == len(chunks) and bool(chunks), f"{written}/{len(chunks)} chunks in {store}"
    )


def input_exists(path: Path) -> bool:
    """Existence test that also accepts a glob pattern (the EC size tables)."""
    text = str(path)
    if any(ch in text for ch in "*?["):
        return bool(glob.glob(text, recursive=True))
    return path.exists()


def check_layer(layer: Path, expected: int) -> Status:
    """Complete when every storage chunk is on disk.

    `info` appears at prepare time, long before a voxel is written, so it cannot be the
    completion test -- a half-decoded volume would look finished.
    """
    scale = next((d for d in sorted(layer.glob("*_*_*")) if d.is_dir()), None)
    if scale is None:
        return Status(False, f"MISSING {layer}")
    written = sum(1 for _ in scale.iterdir())
    return Status(written >= expected, f"{written}/{expected} chunks in {scale}")


def check_abiss_progress(workdir: Path) -> str:
    """Per-chunk resume markers, so an interrupted decode reports how far it got."""
    done = workdir.parent / "scratch" / "done"
    if not done.is_dir():
        return ""
    return f", {sum(1 for _ in done.iterdir())} chunk markers"


@dataclass
class Step:
    name: str
    title: str
    command: str
    status: Callable[[], Status]
    resources: str = ""
    inputs: list[tuple[str, Path]] = field(default_factory=list)
    chain: list = field(default_factory=list)
    array: int = 0  # >0 submits a Slurm array of this size (shard-id per task)
    prepare: str = ""  # cheap, dependency-free command run locally before launch
    prepare_fn: Callable[[], None] | None = None  # same, in-process
    skip: str = ""  # non-empty = why this step is disabled in params.yaml


def sbatch_resources(params, block: str, gpus: int = 0) -> str:
    """Preserve both existing tutorial resource schemas and their flag order."""
    if params.get("pipeline") != "cube_from_scratch" and block in params.cluster:
        cfg = params.cluster[block]
        partition = cfg.get("partition", params.cluster.get("partition"))
        flags = ["-p", str(partition)]
        if gpus:
            flags.append(f"--gres=gpu:{gpus}")
        flags += ["-c", str(cfg.cpus), "--mem", str(cfg.memory), "-t", str(cfg.time)]
    else:
        cfg = params[block]
        flags = []
        if cfg.get("slurm_partition"):
            flags += ["-p", str(cfg.slurm_partition)]
        if gpus:
            flags.append(f"--gres=gpu:{gpus}")
        flags += ["-c", str(cfg.cpus), "--mem", str(cfg.memory), "-t", str(cfg.time)]
        if params.cluster.get("account"):
            flags += ["-A", str(params.cluster.account)]
        if params.cluster.get("extra"):
            flags += shlex.split(str(params.cluster.extra))
    return shlex.join(flags)


def run_local(step: Step, dry_run: bool) -> None:
    if step.chain:
        for _, command, array, _ in step.chain:
            for task_id in range(max(array, 1)):
                cmd = command + (f" --task-id {task_id} --num-tasks {array}" if array else "")
                print(f"  $ {cmd}", flush=True)
                if not dry_run:
                    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
        return
    commands = [step.command]
    if step.array > 1:
        commands = [
            f"{step.command} --shard-id {i} --num-shards {step.array}" for i in range(step.array)
        ]
    for command in commands:
        print(f"  $ {command}", flush=True)
        if not dry_run:
            subprocess.run(command, shell=True, cwd=REPO, check=True)


def _sbatch(name, inner, dependency, array, resources, dry_run, array_flags=None, job_prefix="cube"):
    sbatch = ["sbatch", "--parsable", f"--job-name={job_prefix}-{name}"]
    if dependency:
        sbatch.append(f"--dependency=afterok:{dependency}")
    if array > 1:
        flags = array_flags or ("--shard-id", "--num-shards")
        sbatch.append(f"--array=0-{array - 1}")
        inner = f"{inner} {flags[0]} $SLURM_ARRAY_TASK_ID {flags[1]} {array}"
    sbatch += shlex.split(resources) + [
        f"--wrap=export PATH={shlex.quote(str(Path(sys.executable).parent))}:$PATH && {inner}"
    ]
    print(f"  $ {shlex.join(sbatch)}", flush=True)
    if dry_run:
        return None
    result = subprocess.run(sbatch, cwd=REPO, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    print(f"  submitted {job_id}", flush=True)
    return job_id


def run_slurm(step: Step, dependency: str | None, dry_run: bool, *, job_prefix: str = "cube") -> str | None:
    if step.chain:
        for suffix, command, array, resources in step.chain:
            dependency = _sbatch(
                f"{step.name}-{suffix}", command, dependency, array, resources, dry_run,
                array_flags=("--task-id", "--num-tasks") if array > 1 else None, job_prefix=job_prefix,
            )
        return dependency
    inner = step.command
    return _sbatch(step.name, inner, dependency, step.array, step.resources, dry_run, job_prefix=job_prefix)


def execute_steps(
    steps: list[Step],
    *,
    launcher: str = "local",
    selected: Iterable[str] | None = None,
    forced: Iterable[str] = (),
    check: bool = False,
    dry_run: bool = False,
    job_prefix: str = "cube",
) -> int:
    """Execute selected steps in canonical order, resuming completed artifacts."""
    names = {step.name for step in steps}
    selected = names if selected is None else set(selected)
    forced = set(forced)
    unknown = (selected | forced) - names
    if unknown:
        raise ValueError(
            f"unknown steps: {', '.join(sorted(unknown))}; available: "
            + ", ".join(step.name for step in steps)
        )
    if launcher not in {"local", "slurm"}:
        raise ValueError(f"unknown launcher: {launcher}; available: local, slurm")
    dependency = None
    for step in steps:
        if step.name not in selected:
            continue
        status = step.status()
        print(f"[{'done' if status.done else 'todo'}] {step.title}\n       {status.detail}")
        for label, path in step.inputs:
            print(f"       input {label}: {'ok' if input_exists(path) else 'MISSING'} {path}")
        if check:
            if step.skip:
                print(f"       disabled: {step.skip}")
            print()
            continue
        if step.skip:
            print(f"       skipping ({step.skip})\n")
            continue
        if status.done and step.name not in forced:
            print("       skipping (use --force to rerun)\n")
            continue
        if step.prepare_fn is not None and not dry_run:
            step.prepare_fn()
        missing = [f"{label} ({path})" for label, path in step.inputs if not input_exists(path)]
        if missing and launcher == "local":
            print(f"       BLOCKED, missing input: {'; '.join(missing)}\n")
            return 1
        if step.prepare:
            print(f"  $ {step.prepare}", flush=True)
            if not dry_run:
                subprocess.run(step.prepare, shell=True, cwd=REPO, check=True)
        if launcher == "slurm":
            dependency = run_slurm(step, dependency, dry_run, job_prefix=job_prefix)
        else:
            run_local(step, dry_run)
        print()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    default_tutorial: str | Path | None = None,
    job_prefix: str | None = None,
) -> int:
    """CLI used by the generic entrypoint and existing tutorial invocation paths."""
    from connectomics.playbooks.cube_decode import build_steps

    parser = argparse.ArgumentParser(description=__doc__)
    default_params = None
    if default_tutorial is not None:
        tutorial = Path(default_tutorial)
        if not tutorial.is_absolute():
            tutorial = REPO / "tutorials" / tutorial
        default_params = tutorial / "params.yaml"
    parser.add_argument(
        "--params", type=Path, default=default_params, required=default_params is None
    )
    parser.add_argument(
        "--steps", help="comma-separated subset; canonical sequence order is retained"
    )
    parser.add_argument(
        "--force", default="", help="comma-separated steps to rerun even if complete"
    )
    parser.add_argument("--check", action="store_true", help="report status and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    parser.add_argument(
        "--local", action="store_true", help="run in the foreground instead of sbatch"
    )
    args = parser.parse_args(argv)
    params_path = args.params.resolve()
    params = load_params(params_path)
    steps = build_steps(params, params_path.parent)
    selected = (
        None if args.steps is None else [s.strip() for s in args.steps.split(",") if s.strip()]
    )
    forced = {s.strip() for s in args.force.split(",") if s.strip()}
    launcher = "local" if args.local else str(params.cluster.get("launcher", "slurm"))
    prefix = job_prefix or params_path.parent.name.removeprefix("neuron_")
    print(f"params:   {params_path}")
    print(f"launcher: {launcher}\n")
    return execute_steps(
        steps,
        launcher=launcher,
        selected=selected,
        forced=forced,
        check=args.check,
        dry_run=args.dry_run,
        job_prefix=prefix,
    )
