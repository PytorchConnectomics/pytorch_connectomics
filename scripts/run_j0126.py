#!/usr/bin/env python3
"""Run the j0126 tutorial end to end: train -> infer -> abiss -> error correction.

Every step declares the artifact that proves it finished. The driver checks that
artifact before running the step, skips the step when it is already there, and
prints what it skipped and why -- so re-running the same command resumes a
partial pipeline instead of recomputing it.

  python scripts/run_j0126.py --check                 # report every input/output path
  python scripts/run_j0126.py --dry-run               # print the commands that would run
  python scripts/run_j0126.py --checkpoint ckpt/a.ckpt  # run whatever is missing
  python scripts/run_j0126.py --steps infer,abiss     # only these steps
  python scripts/run_j0126.py --force infer           # rerun a step that looks complete

Adapting it to a cluster: paths come from `tutorials/neuron_j0126/params.yaml`,
so that file is the only thing to edit. `--launcher slurm` wraps each step in
`sbatch --wrap` with a per-step resource string and chains the steps with
`afterok`, instead of running them in the foreground:

  python scripts/run_j0126.py --launcher slurm \\
      --slurm-infer "-p gpu --gres=gpu:1 -c 8 --mem 64G -t 8:00:00" \\
      --slurm-abiss "-p cpu -c 64 --mem 250G -t 24:00:00" \\
      --num-shards 80

Step 4's five input paths are pinned to the frozen reference run in
`4_error_correction.yaml`. The driver reports which of them are missing but does NOT
repoint them at this run's outputs; see that file's comment.
"""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TUTORIAL = REPO / "tutorials" / "neuron_j0126"
PARAMS = TUTORIAL / "params.yaml"


# --------------------------------------------------------------------------- config


def load_workflow_yaml(path: Path) -> OmegaConf:
    """Load a `_base_: params.yaml` workflow YAML with its ${params...} resolved."""
    cfg = OmegaConf.merge(OmegaConf.load(PARAMS), OmegaConf.load(path))
    OmegaConf.resolve(cfg)
    return cfg


def load_pytc_config(path: Path):
    """Load a schema-backed tutorial config through the repository loader."""
    sys.path.insert(0, str(REPO))
    from connectomics.config import load_config  # noqa: PLC0415

    return load_config(str(path))


# --------------------------------------------------------------------------- checks


@dataclass
class Status:
    done: bool
    detail: str


def check_checkpoint(save_path: Path, explicit: Path | None) -> Status:
    if explicit is not None:
        ok = explicit.exists()
        return Status(ok, f"{'found' if ok else 'MISSING'} {explicit}")
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
    payload = json.loads(index.read_text())
    chunks = payload.get("chunks", [])
    root = index.parent
    written = sum(1 for c in chunks if (root / c["path"]).exists())
    detail = f"{written}/{len(chunks)} chunks in {store}"
    return Status(written == len(chunks) and bool(chunks), detail)


def check_path(path: Path, label: str) -> Status:
    return Status(path.exists(), f"{'found' if path.exists() else 'MISSING'} {label}")


def input_exists(path: Path) -> bool:
    """Existence test that also accepts a glob pattern (the EC size tables)."""
    text = str(path)
    if any(ch in text for ch in "*?["):
        return bool(glob.glob(text, recursive=True))
    return path.exists()


# --------------------------------------------------------------------------- steps


@dataclass
class Step:
    name: str
    title: str
    command: list[str]
    status: Callable[[], Status]
    inputs: list[tuple[str, Path]] = field(default_factory=list)
    optional: bool = False
    array: int = 0  # >0 submits a Slurm array of this size (shard-id per task)


def build_steps(args) -> list[Step]:
    params = OmegaConf.load(PARAMS)
    OmegaConf.resolve(params)
    output_root = Path(params.params.paths.output_root)

    infer_yaml = TUTORIAL / args.infer_config
    train_yaml = TUTORIAL / "1_train.yaml"
    abiss_yaml = TUTORIAL / "3_abiss.yaml"
    ec_yaml = TUTORIAL / "4_error_correction.yaml"

    train_cfg = load_pytc_config(train_yaml)
    infer_cfg = load_pytc_config(infer_yaml)
    train_save = Path(train_cfg.save_path)
    infer_save = Path(infer_cfg.save_path)
    infer_suffix = str(infer_cfg.decoding.save_suffix or "")

    abiss = load_workflow_yaml(abiss_yaml).abiss_chunk
    seg_info = Path(str(abiss.param.SEG_PATH).replace("file://", "")) / "info"
    affinity_h5 = Path(str(abiss.source_affinity_h5))

    ec = load_workflow_yaml(ec_yaml).error_correction
    ec_manifest = Path(ec.workdir) / "error_correction_manifest.json"

    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    steps = [
        Step(
            name="train",
            title="1. train the affinity model (optional)",
            command=["python", "scripts/main.py", "--config", str(train_yaml), "--mode", "train"],
            status=lambda: check_checkpoint(train_save, checkpoint),
            inputs=[
                ("dense images", Path(params.params.data.dense_images)),
                ("dense labels", Path(params.params.data.dense_labels)),
            ],
            optional=True,
        ),
        Step(
            name="infer",
            title="2. predict affinity",
            command=[
                "python", "scripts/main.py", "--config", str(infer_yaml),
                "--mode", "test", "--checkpoint", str(checkpoint or "<checkpoint>"),
            ],
            status=lambda: check_affinity(infer_save, infer_suffix),
            inputs=[("EM volume", Path(str(params.params.data.raw_em)))],
            array=args.num_shards,
        ),
        Step(
            name="abiss",
            title="3. ABISS decode",
            command=["python", "scripts/run_abiss_chunk.py", "--config", str(abiss_yaml)],
            status=lambda: check_path(seg_info, f"{seg_info}"),
            inputs=[
                ("affinity store", affinity_h5.with_suffix(affinity_h5.suffix + ".chunks")),
                ("ABISS build", Path(str(abiss.abiss_home))),
            ],
        ),
        Step(
            name="ec",
            title="4. morphology error correction",
            command=[
                "python", "scripts/run_error_correction.py", "--config", str(ec_yaml),
                "--stage", "all", "--num-tasks", "1",
            ],
            status=lambda: check_path(ec_manifest, f"{ec_manifest}"),
            inputs=[
                ("segmentation", Path(str(ec.segmentation))),
                ("affinity chunks", Path(str(ec.affinity_chunks))),
                ("keep mask", Path(str(ec.keep_mask))),
                ("nucleus manifest", Path(str(ec.nucleus_manifest))),
                ("size table", Path(str(ec.size_glob))),
            ],
        ),
    ]
    _ = output_root  # resolved for the report below
    return steps


# --------------------------------------------------------------------------- running


def run_local(step: Step, dry_run: bool) -> None:
    commands = [step.command]
    if step.array > 1:
        commands = [
            step.command + ["--shard-id", str(i), "--num-shards", str(step.array)]
            for i in range(step.array)
        ]
    for command in commands:
        print(f"  $ {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=REPO, check=True)


def run_slurm(step: Step, resources: str, dependency: str | None, dry_run: bool) -> str | None:
    inner = shlex.join(step.command)
    sbatch = ["sbatch", "--parsable", f"--job-name=j0126-{step.name}"]
    if dependency:
        sbatch.append(f"--dependency=afterok:{dependency}")
    if step.array > 1:
        sbatch.append(f"--array=0-{step.array - 1}")
        inner = f"{inner} --shard-id $SLURM_ARRAY_TASK_ID --num-shards {step.array}"
    sbatch += shlex.split(resources) + [f"--wrap={inner}"]
    print(f"  $ {shlex.join(sbatch)}", flush=True)
    if dry_run:
        return None
    result = subprocess.run(sbatch, cwd=REPO, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    print(f"  submitted {job_id}", flush=True)
    return job_id


# --------------------------------------------------------------------------- cli


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", default="train,infer,abiss,ec", help="comma-separated subset, in order")
    ap.add_argument("--force", default="", help="comma-separated steps to rerun even if complete")
    ap.add_argument("--checkpoint", help="affinity checkpoint; skips step 1 when given")
    ap.add_argument(
        "--infer-config",
        default="2_infer.yaml",
        help="config for step 2; use 1_train.yaml for a j0126-trained checkpoint",
    )
    ap.add_argument("--num-shards", type=int, default=1, help="shard step 2 across this many jobs")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument("--launcher", choices=("local", "slurm"), default="local")
    for name in ("train", "infer", "abiss", "ec"):
        ap.add_argument(f"--slurm-{name}", default="", help=f"sbatch resource flags for the {name} step")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    steps = build_steps(args)
    selected = [s.strip() for s in args.steps.split(",") if s.strip()]
    forced = {s.strip() for s in args.force.split(",") if s.strip()}

    print(f"params: {PARAMS}\n")
    dependency = None
    for step in steps:
        if step.name not in selected:
            continue
        status = step.status()
        mark = "done" if status.done else "todo"
        print(f"[{mark}] {step.title}\n       {status.detail}")
        for label, path in step.inputs:
            print(f"       input {label}: {'ok' if input_exists(path) else 'MISSING'} {path}")
        if args.check:
            print()
            continue
        if status.done and step.name not in forced:
            print("       skipping (use --force to rerun)\n")
            continue
        if step.optional and args.checkpoint:
            print("       skipping (checkpoint supplied)\n")
            continue
        missing = [f"{label} ({path})" for label, path in step.inputs if not input_exists(path)]
        if missing:
            print(f"       BLOCKED, missing input: {'; '.join(missing)}\n")
            return 1
        if args.launcher == "slurm":
            dependency = run_slurm(
                step, getattr(args, f"slurm_{step.name}"), dependency, args.dry_run
            )
        else:
            run_local(step, args.dry_run)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
