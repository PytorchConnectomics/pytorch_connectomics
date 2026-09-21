"""The canonical cube segmentation sequences and their step builders."""

from __future__ import annotations

import json
import math
import shlex
import sys
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from connectomics.runtime.volume_pipeline import (
    Status,
    Step,
    check_abiss_progress,
    check_affinity,
    check_checkpoint,
    check_download,
    check_layer,
    check_paths,
    check_shards,
    load_pytc_config,
    load_workflow_yaml,
    require_dataset_values,
    sbatch_resources,
)

REPO = Path(__file__).resolve().parents[2]
PYTHON = shlex.quote(sys.executable)

EC_STAGES = (
    "sizes", "skeletonize", "skeletons", "contacts", "contact_graph", "candidates",
    "junction_scope", "junction_features", "boundary", "resolve", "prepare_output",
    "postprocess", "verify",
)
EC_ARRAY_STAGES = frozenset({"skeletonize", "contacts", "postprocess"})


CUBE = ("mask", "smoke", "abiss", "guard", "score")
CUBE_FROM_SCRATCH = ("fetch", "em", "tissue", "keep", "train", "infer", "abiss", "ec", "eval")
SEQUENCES = {"cube": CUBE, "cube_from_scratch": CUBE_FROM_SCRATCH}


def _load_abiss(path: Path):
    abiss = load_workflow_yaml(path).abiss_chunk
    require_dataset_values(abiss, path=path)
    return abiss


def _cli_path(path: Path) -> str:
    """Retain repo-relative command spelling where the original driver used it."""
    try:
        path = path.relative_to(REPO)
    except ValueError:
        pass
    return shlex.quote(str(path))


def _storage_chunks(abiss) -> int:
    bbox = abiss.param.BBOX
    chunks = abiss.get("seg_chunk_size_xyz", abiss.get("copy_block_shape_xyz", [512, 512, 64]))
    return math.prod(-(-(int(bbox[i + 3]) - int(bbox[i])) // int(chunks[i])) for i in range(3))


def _smoke_config(abiss, output_root: Path):
    """Derive a pre-flight extent; retain the dataset's fitted chunk size/thresholds."""
    smoke = OmegaConf.create(OmegaConf.to_container(abiss, resolve=True))
    root = output_root / "abiss_smoke"
    smoke.workdir = str(root / "run")
    smoke.secrets_dir = str(root / "secrets")
    smoke.param_path = str(root / "secrets/param")
    if smoke.get("source_affinity_h5"):
        smoke.param.AFF_PATH = f"file://{root / 'precomputed/affinity'}"
    for key, suffix in (
        ("WS_PATH", "precomputed/ws"),
        ("SEG_PATH", "precomputed/seg"),
        ("SCRATCH_PATH", "scratch"),
        ("CHUNKMAP_OUTPUT", "chunkmap"),
    ):
        smoke.param[key] = f"file://{root / suffix}"
    if "CHUNKMAP_INPUT" in smoke.param:
        smoke.param.CHUNKMAP_INPUT = smoke.param.CHUNKMAP_OUTPUT
    if "NUC_COMPETITION_MANIFEST" in smoke.param:
        smoke.param.NUC_COMPETITION_MANIFEST = str(root / "run/nucleus_competition/manifest.json")
    bbox = [int(v) for v in abiss.param.BBOX]
    smoke.param.BBOX = bbox[:3] + [
        min(bbox[i + 3], bbox[i] + limit) for i, limit in enumerate((1024, 1024, 1792))
    ]
    # The executor computes these from the derived extent.
    smoke.pop("top_mip", None)
    smoke.pop("root_tag", None)
    return smoke


def _write_smoke_config(path: Path, smoke) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"abiss_chunk": smoke}), path)


def _cube_steps(params, tutorial: Path) -> list[Step]:
    abiss_yaml = tutorial / "2_abiss.yaml"
    abiss = _load_abiss(abiss_yaml)
    output_root = Path(str(params.paths.output_root))
    smoke_yaml = tutorial / "2_abiss_smoke.yaml"
    prepare_smoke = None
    if smoke_yaml.exists():
        smoke = _load_abiss(smoke_yaml)
    else:
        smoke = _smoke_config(abiss, output_root)
        smoke_yaml = output_root / "abiss_smoke/config.yaml"

        prepare_smoke = partial(_write_smoke_config, smoke_yaml, smoke)

    seg = Path(str(abiss.param.SEG_PATH).replace("file://", ""))
    smoke_seg = Path(str(smoke.param.SEG_PATH).replace("file://", ""))
    workdir = Path(str(abiss.workdir))
    reports = output_root / "reports"
    keep_mask = Path(str(params.data.keep_mask))

    # Preserve the existing tutorial's public command strings and wrapper flags.
    moritz = tutorial.name == "neuron_moritz_l4"
    mask_tool = (
        "python scripts/build_moritz_l4_keep_mask.py"
        if moritz
        else f"python scripts/build_keep_mask.py --params {_cli_path(tutorial / 'params.yaml')}"
    )
    frame_flags = ""
    if not moritz:
        frame = params.frame
        for flag, values in (
            ("--volume-shape-zyx", frame.volume_shape_zyx),
            ("--global-origin-zyx", frame.volume_origin_global_zyx),
            ("--resolution-xyz-nm", frame.resolution_xyz_nm),
        ):
            frame_flags += f" {flag} " + shlex.join([str(v) for v in values])

    return [
        Step(
            name="mask",
            title=(
                "0. keep mask: NOT(blood vessel) AND tissue, on the 4x8x8 grid"
                if moritz
                else "0. keep mask: combine source masks and the tissue border"
            ),
            command=f"{mask_tool} --out {shlex.quote(str(keep_mask))}",
            status=lambda: check_paths(keep_mask),
            resources=sbatch_resources(params, "mask"),
        ),
        Step(
            name="smoke",
            title=(
                "1. pre-flight: the same recipe on 1.2% of the volume"
                if moritz
                else "1. pre-flight: the same recipe on a bounded volume"
            ),
            command=f"python scripts/run_abiss_chunk.py --config {_cli_path(smoke_yaml)}",
            status=lambda: check_layer(smoke_seg, _storage_chunks(smoke)),
            resources=sbatch_resources(params, "smoke"),
            prepare_fn=prepare_smoke,
        ),
        Step(
            name="abiss",
            title="2. whole-volume decode: masked affinity -> watershed -> nuclei -> agglomeration",
            command=f"python scripts/run_abiss_chunk.py --config {_cli_path(abiss_yaml)}",
            status=lambda: Status(
                check_layer(seg, _storage_chunks(abiss)).done,
                check_layer(seg, _storage_chunks(abiss)).detail + check_abiss_progress(workdir),
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
            title=(
                "4. NERL + skeleton VOI against the 96 manual reconstructions"
                if moritz
                else "4. NERL + skeleton VOI against the supplied reconstructions"
            ),
            command=(
                f"mkdir -p {shlex.quote(str(reports))} && "
                f"python scripts/score_moritz_l4_nerl.py {shlex.quote(str(seg))} "
                f"--skeletons {shlex.quote(str(params.data.gt_skeletons))} "
                f"--out {shlex.quote(str(reports / 'nerl.json'))}{frame_flags}"
            ),
            status=lambda: check_paths(reports / "nerl.json"),
            resources=sbatch_resources(params, "score"),
        ),
    ]


def _from_scratch_steps(params, tutorial: Path) -> list[Step]:
    dataset_root = Path(params.paths.dataset_root)
    output_root = Path(params.paths.output_root)
    train_yaml = tutorial / "1_train.yaml"
    infer_yaml = tutorial / "2_infer.yaml"
    abiss_yaml = tutorial / "3_abiss.yaml"
    ec_yaml = tutorial / "4_error_correction.yaml"

    train_cfg = load_pytc_config(train_yaml)
    infer_cfg = load_pytc_config(infer_yaml)
    train_save = Path(train_cfg.save_path)
    infer_save = Path(infer_cfg.save_path)
    infer_suffix = str(infer_cfg.decoding.save_suffix or "")

    abiss = _load_abiss(abiss_yaml)

    if list(params.download.em_bbox):
        z0, z1, y0, y1, x0, x1 = (int(v) for v in params.download.em_bbox)
        shape_xyz = [x1 - x0, y1 - y0, z1 - z0]
        cropped = OmegaConf.load(abiss_yaml)
        cropped["_base_"] = str(tutorial / "params.yaml")
        cropped.abiss_chunk.param.BBOX = [0, 0, 0, *shape_xyz]
        chunk = list(cropped.abiss_chunk.param.CHUNK_SIZE)
        cropped.abiss_chunk.param.CHUNK_SIZE = [
            min(chunk[i], shape_xyz[i]) for i in range(3)
        ]

        keep_src = str(abiss.param.get("AFF_KEEP_MASK") or "")
        if keep_src:
            keep_dst = Path(str(output_root)) / "abiss" / "keep_mask_cropped.zarr"
            if not any((keep_dst / n).exists() for n in ("zarr.json", ".zarray")):
                import zarr
                src = zarr.open(keep_src, mode="r")
                if hasattr(src, "keys") and "main" in list(src.keys()):
                    src = src["main"]
                sub = src[z0:z1, y0:y1, x0:x1]
                keep_dst.parent.mkdir(parents=True, exist_ok=True)
                chunks = tuple(min(c, d) for c, d in zip((126, 504, 504), sub.shape))
                try:
                    out = zarr.create_array(
                        store=str(keep_dst), shape=sub.shape, dtype=sub.dtype, chunks=chunks
                    )
                except AttributeError:
                    out = zarr.open(
                        str(keep_dst), mode="w", shape=sub.shape, dtype=sub.dtype, chunks=chunks
                    )
                out[:] = sub
            cropped.abiss_chunk.param.AFF_KEEP_MASK = str(keep_dst)

        abiss_yaml = Path(str(output_root)) / "abiss" / "3_abiss.resolved.yaml"
        abiss_yaml.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cropped, abiss_yaml)
        print(f"em_bbox is set: wrote {abiss_yaml} with BBOX {[0, 0, 0, *shape_xyz]}")

        ec_cropped = OmegaConf.load(ec_yaml)
        ec_cropped["_base_"] = str(tutorial / "params.yaml")
        ec_cropped.error_correction.volume_shape_zyx = [z1 - z0, y1 - y0, x1 - x0]
        core = list(ec_cropped.error_correction.core_xyz)
        ec_cropped.error_correction.expected_chunks = (
            -(-shape_xyz[0] // core[0]) * -(-shape_xyz[1] // core[1])
            * -(-(z1 - z0) // core[2])
        )
        ec_yaml = Path(str(output_root)) / "error_correction_v7" / "4_ec.resolved.yaml"
        ec_yaml.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(ec_cropped, ec_yaml)
        print(f"em_bbox is set: wrote {ec_yaml} with volume_shape_zyx "
              f"{[z1 - z0, y1 - y0, x1 - x0]} and expected_chunks "
              f"{ec_cropped.error_correction.expected_chunks}\n")
    seg_info = Path(str(abiss.param.SEG_PATH).replace("file://", "")) / "info"
    _bbox = [int(v) for v in abiss.param.BBOX]
    _seg_chunk = [int(v) for v in abiss.seg_chunk_size_xyz]
    seg_chunk_count = 1
    for _i in range(3):
        seg_chunk_count *= -(-(_bbox[_i + 3] - _bbox[_i]) // _seg_chunk[_i])

    affinity_h5 = Path(
        str(abiss.get("source_affinity_h5") or abiss.param.AFF_PATH).replace("file://", "")
    )

    ec = load_workflow_yaml(ec_yaml).error_correction
    ec_tasks = int(ec.get("task_count", 80))
    ec_task_resources = shlex.join(
        shlex.split(sbatch_resources(params, "error_correction"))[:2]  # keep -p <partition>
        + ["-c", "8", "--mem", "64G", "-t", str(params.error_correction.time)]
    )
    _ec_seg = Path(str(ec.output_segmentation))
    eval_seg = _ec_seg if (_ec_seg / "info").exists() else Path(
        str(abiss.param.SEG_PATH).replace("file://", "")
    )
    eval_report = output_root / "eval" / "nerl.json"
    skeletons = Path(str(params.data.skeletons))
    ec_manifest = Path(ec.workdir) / "error_correction_manifest.json"
    nucleus_manifest = Path(str(ec.nucleus_manifest))

    training = bool(params.train.enabled)
    em_store = Path(str(params.data.raw_em)).parent
    em_dataset = Path(str(params.data.raw_em)).name
    tissue = Path(str(params.data.tissue_mask))
    keep = Path(str(params.data.keep_mask))
    checkpoint = (
        infer_save / "downloaded" / "checkpoints"
        / params.download.checkpoint_url.rsplit("/", 1)[-1]
    )

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
        f"{PYTHON} scripts/download_precompute.py {shlex.quote(str(dl.em_source))}"
        f" --out {shlex.quote(str(em_store))} --dataset {em_dataset} --mip 0"
        f" --slab {dl.slab} --tile-xy {dl.tile_xy}"
        + (f" --bbox {em_bbox}" if em_bbox else "")
    )
    mask_tool = f"{PYTHON} scripts/build_j0126_keep_mask.py"

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

    for url, dest in (
        (dl.get("skeletons_url"), Path(str(params.data.skeletons))),
        (dl.get("ffn_node_lut_url"), Path(str(params.data.ffn_node_lut))),
    ):
        if url:
            fetch_parts.append(
                f"[ -s {shlex.quote(str(dest))} ] ||"
                f" curl -fL {shlex.quote(str(url))} -o {shlex.quote(str(dest))}"
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
            command=f"{PYTHON} scripts/main.py --config {train_yaml} --mode train",
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
                f"{PYTHON} scripts/main.py --config {infer_yaml} --mode test"
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
                f"{PYTHON} scripts/stitch_chunked_prediction.py --vds"
                f" --discover {shlex.quote(str(output_root))}"
                f" --out {shlex.quote(str(affinity_h5))} --force"
                f" && {PYTHON} scripts/run_abiss_chunk.py --config {abiss_yaml}"
            ),
            status=lambda: check_layer(seg_info.parent),
            resources=sbatch_resources(params, "abiss"),
            inputs=[("ABISS build", Path(str(abiss.abiss_home))), ("keep mask", keep)],
        ),
        Step(
            name="ec",
            title="4. morphology error correction",
            command="",
            chain=[
                (
                    stage,
                    f"{PYTHON} scripts/run_error_correction.py --config {ec_yaml}"
                    f" --stage {stage}"
                    + ("" if stage in EC_ARRAY_STAGES else " --num-tasks 1 --task-id 0"),
                    ec_tasks if stage in EC_ARRAY_STAGES else 0,
                    ec_task_resources if stage in EC_ARRAY_STAGES
                    else sbatch_resources(params, "error_correction"),
                )
                for stage in EC_STAGES
            ],
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
        Step(
            name="eval",
            title="5. NERL / VOI against the 50 test skeletons",
            command=(
                f"{PYTHON} scripts/evaluate_j0126.py"
                f" --segmentation \"$([ -f {shlex.quote(str(_ec_seg / 'info'))} ]"
                f" && echo {shlex.quote(str(_ec_seg))}"
                f" || echo {shlex.quote(str(eval_seg))})\""
                f" --skeletons {shlex.quote(str(skeletons))}"
                f" --output {shlex.quote(str(eval_report))}"
            ),
            status=lambda: check_paths(eval_report),
            resources=sbatch_resources(params, "error_correction"),
            inputs=[("segmentation", eval_seg / "info"), ("test-50 skeletons", skeletons)],
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


# Builders declare step contents; only SEQUENCES determines execution order.
STEP_BUILDERS = {"cube": _cube_steps, "cube_from_scratch": _from_scratch_steps}


def build_steps(params, tutorial: Path) -> list[Step]:
    sequence = str(params.pipeline)
    if sequence not in SEQUENCES:
        raise ValueError(f"Unknown pipeline {sequence!r}; choose from {', '.join(SEQUENCES)}")
    steps = {step.name: step for step in STEP_BUILDERS[sequence](params, Path(tutorial))}
    return [steps[name] for name in SEQUENCES[sequence]]
