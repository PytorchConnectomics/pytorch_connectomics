#!/usr/bin/env python3
"""Run the whole Moritz L4 decode from one file.

Edit `tutorials/neuron_moritz_l4/params.yaml` -- paths and the Slurm resources for each
step -- then:

    python scripts/run_moritz_l4.py

That builds the blood-vessel/border keep mask, runs the pre-flight box, decodes the whole
volume with ABISS, guards the result against percolation and scores it against the 96
manual skeletons. Each step is submitted with sbatch and chained by
`--dependency=afterok`, so the command queues the pipeline and returns.

This tutorial starts at the DECODE. The affinity already exists -- it is the output of a
separate training/inference run (`outputs/segem_banis_plus/20260824_165054/`, see
README.md) and is used in place, never copied: it is 645 GB.

Every step declares the artifact that proves it finished, so re-running the same command
resumes a partial pipeline instead of recomputing it. ABISS itself resumes per chunk.

    python scripts/run_moritz_l4.py --check             # what exists, what is missing
    python scripts/run_moritz_l4.py --dry-run           # print the commands only
    python scripts/run_moritz_l4.py --steps abiss,guard # only these steps
    python scripts/run_moritz_l4.py --force guard       # rerun a step that looks complete
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parent.parent
TUTORIAL = REPO / "tutorials" / "neuron_moritz_l4"
PARAMS = TUTORIAL / "params.yaml"

ORDER = ("mask", "smoke", "abiss", "guard", "score")

# One storage chunk of the seg layer is 128 x 128 x 32 regardless of the ABISS logical
# CHUNK_SIZE, so this count does not change when the chunk grid does.
SEG_STORAGE_CHUNKS = 44 * 67 * 104


def load_params() -> DictConfig:
    params = OmegaConf.load(PARAMS)
    OmegaConf.resolve(params)
    return params.params


def load_workflow_yaml(path: Path) -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.load(PARAMS), OmegaConf.load(path))
    OmegaConf.resolve(cfg)
    return cfg


@dataclass
class Status:
    done: bool
    detail: str


def check_paths(*paths: Path) -> Status:
    missing = [p for p in paths if not p.exists()]
    if missing:
        return Status(False, "MISSING " + ", ".join(str(p) for p in missing))
    return Status(True, "found " + ", ".join(str(p) for p in paths))


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


def sbatch_resources(params, block: str) -> str:
    cfg = params.cluster[block]
    partition = cfg.get("partition", params.cluster.get("partition"))
    flags = ["-p", str(partition), "-c", str(cfg.cpus), "--mem", str(cfg.memory),
             "-t", str(cfg.time)]
    return shlex.join(flags)


def build_steps(params) -> list[Step]:
    abiss = load_workflow_yaml(TUTORIAL / "2_abiss.yaml").abiss_chunk
    smoke = load_workflow_yaml(TUTORIAL / "2_abiss_smoke.yaml").abiss_chunk
    seg = Path(str(abiss.param.SEG_PATH).replace("file://", ""))
    smoke_seg = Path(str(smoke.param.SEG_PATH).replace("file://", ""))
    workdir = Path(str(abiss.workdir))
    reports = Path(str(params.paths.output_root)) / "reports"
    keep_mask = Path(str(params.data.keep_mask))

    return [
        Step(
            name="mask",
            title="0. keep mask: NOT(blood vessel) AND tissue, on the 4x8x8 grid",
            command=f"python scripts/build_moritz_l4_keep_mask.py --out {shlex.quote(str(keep_mask))}",
            status=lambda: check_paths(keep_mask),
            resources=sbatch_resources(params, "mask"),
        ),
        Step(
            name="smoke",
            title="1. pre-flight: the same recipe on 1.2% of the volume",
            command="python scripts/run_abiss_chunk.py --config tutorials/neuron_moritz_l4/2_abiss_smoke.yaml",
            status=lambda: check_layer(smoke_seg, 8 * 8 * 56),   # 1024x1024x1792 in 128x128x32 storage chunks
            resources=sbatch_resources(params, "smoke"),
        ),
        Step(
            name="abiss",
            title="2. whole-volume decode: masked affinity -> watershed -> nuclei -> agglomeration",
            command="python scripts/run_abiss_chunk.py --config tutorials/neuron_moritz_l4/2_abiss.yaml",
            status=lambda: Status(
                check_layer(seg, SEG_STORAGE_CHUNKS).done,
                check_layer(seg, SEG_STORAGE_CHUNKS).detail + check_abiss_progress(workdir),
            ),
            resources=sbatch_resources(params, "abiss"),
        ),
        Step(
            name="guard",
            title="3. percolation guard (GT-free): largest segment must be < 5% of the volume",
            command=(
                f"mkdir -p {shlex.quote(str(reports))} && "
                f"python scripts/seg_percolation_guard.py {shlex.quote(str(seg))} "
                f"--blocks 200 --out {shlex.quote(str(reports / 'guard_seg.json'))}"
            ),
            status=lambda: check_paths(reports / "guard_seg.json"),
            resources=sbatch_resources(params, "score"),
        ),
        Step(
            name="score",
            title="4. NERL + skeleton VOI against the 96 manual reconstructions",
            command=(
                f"mkdir -p {shlex.quote(str(reports))} && "
                f"python scripts/score_moritz_l4_nerl.py {shlex.quote(str(seg))} "
                f"--skeletons {shlex.quote(str(params.data.gt_skeletons))} "
                f"--out {shlex.quote(str(reports / 'nerl.json'))}"
            ),
            status=lambda: check_paths(reports / "nerl.json"),
            resources=sbatch_resources(params, "score"),
        ),
    ]


def run_local(step: Step, dry_run: bool) -> None:
    print(f"  $ {step.command}", flush=True)
    if not dry_run:
        subprocess.run(step.command, shell=True, cwd=REPO, check=True)


def run_slurm(step: Step, dependency: str | None, dry_run: bool) -> str | None:
    sbatch = ["sbatch", "--parsable", f"--job-name=moritz-{step.name}"]
    if dependency:
        sbatch.append(f"--dependency=afterok:{dependency}")
    sbatch += shlex.split(step.resources) + [f"--wrap={step.command}"]
    print(f"  $ {shlex.join(sbatch)}", flush=True)
    if dry_run:
        return None
    result = subprocess.run(sbatch, cwd=REPO, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    print(f"  submitted {job_id}", flush=True)
    return job_id


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", default=",".join(ORDER), help="comma-separated subset, in order")
    ap.add_argument("--force", default="", help="comma-separated steps to rerun even if complete")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument("--local", action="store_true", help="run in the foreground instead of sbatch")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    params = load_params()
    steps = build_steps(params)
    selected = [s.strip() for s in args.steps.split(",") if s.strip()]
    forced = {s.strip() for s in args.force.split(",") if s.strip()}

    print(f"params:   {PARAMS}")
    print(f"launcher: {'local' if args.local else 'slurm'}\n")
    dependency = None
    for step in steps:
        if step.name not in selected:
            continue
        status = step.status()
        print(f"[{'done' if status.done else 'todo'}] {step.title}\n       {status.detail}")
        if args.check:
            print()
            continue
        if status.done and step.name not in forced:
            print("       skipping (use --force to rerun)\n")
            continue
        if args.local:
            run_local(step, args.dry_run)
        else:
            dependency = run_slurm(step, dependency, args.dry_run)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
