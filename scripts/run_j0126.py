#!/usr/bin/env python3
"""Run the whole j0126 tutorial from one file.

Edit `tutorials/neuron_j0126/params.yaml` -- paths, whether to train, and the
Slurm resources for each step -- then:

    python scripts/run_j0126.py

That downloads the data, builds the exclusion mask, trains (or downloads) the
affinity model, predicts affinity, decodes with ABISS and runs morphology error
correction. With `cluster.launcher: slurm` it submits every step with sbatch,
chained by `--dependency=afterok`, and returns as soon as the last one is
queued; with `local` it runs them in the foreground, in order.

Every step declares the artifact that proves it finished. The driver checks that
artifact first and skips the step when it is already there, so re-running the
same command resumes a partial pipeline instead of recomputing it.

    python scripts/run_j0126.py --check              # what exists, what is missing
    python scripts/run_j0126.py --dry-run            # print the commands only
    python scripts/run_j0126.py --steps infer,abiss  # only these steps
    python scripts/run_j0126.py --force abiss        # rerun a step that looks complete
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

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parent.parent
TUTORIAL = REPO / "tutorials" / "neuron_j0126"
PARAMS = TUTORIAL / "params.yaml"

ORDER = ("fetch", "em", "tissue", "keep", "train", "infer", "abiss", "ec")


# --------------------------------------------------------------------------- config


def load_params() -> DictConfig:
    params = OmegaConf.load(PARAMS)
    OmegaConf.resolve(params)
    return params.params


def load_workflow_yaml(path: Path) -> DictConfig:
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
    return Status(written == len(chunks) and bool(chunks), f"{written}/{len(chunks)} chunks in {store}")


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
    command: str
    status: Callable[[], Status]
    resources: str = ""
    inputs: list[tuple[str, Path]] = field(default_factory=list)
    array: int = 0            # >0 submits a Slurm array of this size (shard-id per task)
    prepare: str = ""         # cheap, dependency-free command run locally before launch
    prepare_fn: Callable[[], None] | None = None   # same, in-process
    skip: str = ""            # non-empty = why this step is disabled in params.yaml


def sbatch_resources(params, block: str, gpus: int = 0) -> str:
    """Compose one step's sbatch flags from its params.yaml block."""
    cfg = params[block]
    flags: list[str] = []
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


def build_steps(params) -> list[Step]:
    dataset_root = Path(params.paths.dataset_root)
    train_yaml = TUTORIAL / "1_train.yaml"
    infer_yaml = TUTORIAL / "2_infer.yaml"
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
    nucleus_manifest = Path(str(ec.nucleus_manifest))

    training = bool(params.train.enabled)
    em_store = Path(str(params.data.raw_em)).parent
    em_dataset = Path(str(params.data.raw_em)).name
    tissue = Path(str(params.data.tissue_mask))
    keep = Path(str(params.data.keep_mask))
    checkpoint = dataset_root / "ckpt" / params.download.checkpoint_url.rsplit("/", 1)[-1]

    # The trained checkpoint lands in a timestamped run directory, so it cannot be
    # named when the job is submitted; resolve it in the job's own shell instead.
    infer_ckpt = (
        f'"$(ls -t {shlex.quote(str(train_save))}/*/checkpoints/*.ckpt | head -1)"'
        if training
        else shlex.quote(str(checkpoint))
    )

    dl = params.download
    em_bbox = " ".join(str(int(v)) for v in dl.em_bbox)
    em_common = (
        f"python scripts/download_precompute.py {shlex.quote(str(dl.em_source))}"
        f" --out {shlex.quote(str(em_store))} --dataset {em_dataset} --mip 0"
        f" --slab {dl.slab} --tile-xy {dl.tile_xy}"
        + (f" --bbox {em_bbox}" if em_bbox else "")
    )
    mask_tool = "python scripts/build_j0126_keep_mask.py"

    fetch_parts = []
    if training:
        zip_path = dataset_root / "j0126-train-33vol.zip"
        fetch_parts.append(
            f"mkdir -p {shlex.quote(str(dataset_root / 'train'))}"
            f" && curl -fL {shlex.quote(str(dl.train_zip))} -o {shlex.quote(str(zip_path))}"
            f" && unzip -q -o {shlex.quote(str(zip_path))} -d {shlex.quote(str(dataset_root / 'train'))}"
            f" && rm -f {shlex.quote(str(zip_path))}"
        )
    else:
        fetch_parts.append(
            f"mkdir -p {shlex.quote(str(checkpoint.parent))}"
            f" && curl -fL {shlex.quote(str(dl.checkpoint_url))} -o {shlex.quote(str(checkpoint))}"
        )

    download_off = "" if params.download.enabled else "download.enabled is false"
    mask_off = "" if params.mask.enabled else "mask.enabled is false"

    return [
        Step(
            name="fetch",
            title="0a. download the training cubes" if training else "0a. download the affinity model",
            command=" && ".join(fetch_parts),
            status=lambda: check_paths(
                *(
                    (Path(str(params.data.dense_images)), Path(str(params.data.dense_labels)))
                    if training
                    else (checkpoint,)
                )
            ),
            resources=sbatch_resources(params, "download"),
            skip=download_off,
        ),
        Step(
            name="em",
            title="0b. download the EM volume",
            command=em_common,
            status=lambda: check_download(em_store, em_dataset, int(dl.slab), int(dl.tile_xy)),
            resources=sbatch_resources(params, "download"),
            array=int(dl.num_shards),
            prepare=f"{em_common} --init-only",
            skip=download_off,
        ),
        Step(
            name="tissue",
            title="0c. FFN tissue mask",
            command=f"{mask_tool} --stage tissue --out {shlex.quote(str(tissue))}",
            status=lambda: check_shards(tissue, int(params.mask.num_shards)),
            resources=sbatch_resources(params, "mask"),
            array=int(params.mask.num_shards),
            prepare=f"{mask_tool} --stage tissue --init --out {shlex.quote(str(tissue))}",
            skip=mask_off,
        ),
        Step(
            name="keep",
            title="0d. tissue + border keep-mask",
            command=(
                f"{mask_tool} --stage keep --out {shlex.quote(str(keep))}"
                f" --tissue {shlex.quote(str(tissue))}"
                f" --em {shlex.quote(str(params.data.raw_em))}"
            ),
            status=lambda: check_shards(keep, int(params.mask.num_shards)),
            resources=sbatch_resources(params, "mask"),
            array=int(params.mask.num_shards),
            prepare=f"{mask_tool} --stage keep --init --out {shlex.quote(str(keep))}",
            inputs=[("tissue mask", tissue), ("EM volume", Path(str(params.data.raw_em)))],
            skip=mask_off,
        ),
        Step(
            name="train",
            title="1. train the affinity model",
            command=f"python scripts/main.py --config {train_yaml} --mode train",
            status=lambda: check_checkpoint(train_save, None if training else checkpoint),
            resources=sbatch_resources(params, "train", gpus=int(params.train.num_gpus)),
            inputs=[
                ("dense images", Path(str(params.data.dense_images))),
                ("dense labels", Path(str(params.data.dense_labels))),
            ],
            skip="" if training else "train.enabled is false, using the downloaded checkpoint",
        ),
        Step(
            name="infer",
            title="2. predict affinity",
            command=(
                f"python scripts/main.py --config {infer_yaml} --mode test"
                f" --checkpoint {infer_ckpt}"
            ),
            status=lambda: check_affinity(infer_save, infer_suffix),
            # One GPU per shard by design: the shards are independent processes, not DDP.
            resources=sbatch_resources(params, "inference", gpus=1),
            array=int(params.inference.num_shards),
            inputs=[("EM volume", Path(str(params.data.raw_em)))],
        ),
        Step(
            name="abiss",
            title="3. ABISS decode",
            # The virtual dataset is what makes the chunk store readable as the single
            # h5 ABISS wants; it copies nothing, so it belongs to this step's setup.
            command=(
                f"python scripts/stitch_chunked_prediction.py --vds"
                f" --discover {shlex.quote(str(infer_save))}"
                f" --out {shlex.quote(str(affinity_h5))} --force"
                f" && python scripts/run_abiss_chunk.py --config {abiss_yaml}"
            ),
            status=lambda: check_paths(seg_info),
            resources=sbatch_resources(params, "abiss"),
            inputs=[("ABISS build", Path(str(abiss.abiss_home))), ("keep mask", keep)],
        ),
        Step(
            name="ec",
            title="4. morphology error correction",
            command=(
                f"python scripts/run_error_correction.py --config {ec_yaml}"
                f" --stage all --num-tasks 1"
            ),
            status=lambda: check_paths(ec_manifest),
            resources=sbatch_resources(params, "error_correction"),
            prepare_fn=lambda: stub_nucleus_manifest(params, nucleus_manifest),
            inputs=[
                ("segmentation", Path(str(ec.segmentation))),
                ("affinity chunks", Path(str(ec.affinity_chunks))),
                ("keep mask", Path(str(ec.keep_mask))),
                ("nucleus manifest", nucleus_manifest),
                ("size table", Path(str(ec.size_glob))),
            ],
        ),
    ]


def stub_nucleus_manifest(params, path: Path) -> None:
    """Write the empty identity table ABISS would have written, and say so.

    Step 4 treats the manifest as an external firewall: it refuses to join two
    segments carrying different nucleus identities. With no nucleus volume there
    are no identities, the gate never fires, and the run is the README's row
    WITHOUT the nucleus instance certificate. That is a real difference in the
    result, so it is printed rather than assumed.
    """
    if path.exists() or params.data.get("nucleus_volume"):
        return
    print(
        "       NOTE data.nucleus_volume is empty, so ABISS ran no nucleus competition.\n"
        "            Writing an empty identity manifest: step 4 runs WITHOUT the nucleus\n"
        f"            firewall (README row 2, not row 3).  {path}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"qualified_segment_labels": {}, "qualified_segment_owners": {}}))


# --------------------------------------------------------------------------- running


def run_local(step: Step, dry_run: bool) -> None:
    commands = [step.command]
    if step.array > 1:
        commands = [
            f"{step.command} --shard-id {i} --num-shards {step.array}" for i in range(step.array)
        ]
    for command in commands:
        print(f"  $ {command}", flush=True)
        if not dry_run:
            subprocess.run(command, shell=True, cwd=REPO, check=True)


def run_slurm(step: Step, dependency: str | None, dry_run: bool) -> str | None:
    inner = step.command
    sbatch = ["sbatch", "--parsable", f"--job-name=j0126-{step.name}"]
    if dependency:
        sbatch.append(f"--dependency=afterok:{dependency}")
    if step.array > 1:
        sbatch.append(f"--array=0-{step.array - 1}")
        inner = f"{inner} --shard-id $SLURM_ARRAY_TASK_ID --num-shards {step.array}"
    sbatch += shlex.split(step.resources) + [f"--wrap={inner}"]
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
    ap.add_argument("--steps", default=",".join(ORDER), help="comma-separated subset, in order")
    ap.add_argument("--force", default="", help="comma-separated steps to rerun even if complete")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    params = load_params()
    steps = build_steps(params)
    selected = [s.strip() for s in args.steps.split(",") if s.strip()]
    forced = {s.strip() for s in args.force.split(",") if s.strip()}
    launcher = str(params.cluster.launcher)

    print(f"params:   {PARAMS}")
    print(f"launcher: {launcher}\n")
    dependency = None
    for step in steps:
        if step.name not in selected:
            continue
        status = step.status()
        print(f"[{'done' if status.done else 'todo'}] {step.title}\n       {status.detail}")
        for label, path in step.inputs:
            print(f"       input {label}: {'ok' if input_exists(path) else 'MISSING'} {path}")
        if args.check:
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
        if step.prepare_fn is not None and not args.dry_run:
            step.prepare_fn()
        missing = [f"{label} ({path})" for label, path in step.inputs if not input_exists(path)]
        if missing and launcher == "local":
            # Under slurm the inputs are produced by the jobs this one waits on, so
            # only a foreground run can conclude anything from them being absent.
            print(f"       BLOCKED, missing input: {'; '.join(missing)}\n")
            return 1
        if step.prepare:
            print(f"  $ {step.prepare}", flush=True)
            if not args.dry_run:
                subprocess.run(step.prepare, shell=True, cwd=REPO, check=True)
        if launcher == "slurm":
            dependency = run_slurm(step, dependency, args.dry_run)
        else:
            run_local(step, args.dry_run)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
