#!/usr/bin/env python3
"""Build-time smoke test for the LICONN moe cloud image.

Run as a Dockerfile `RUN` step so a broken image fails in the builder rather
than at hour one of a GPU run. Every check below corresponds to something that
has actually cost a run in this project's history.

It is a FILE and not an inline heredoc on purpose. `RUN <<EOF` heredocs are a
BuildKit-only Dockerfile feature, and the builder VM installs Debian's
`docker.io`, which has no BuildKit and runs `DOCKER_BUILDKIT=0`. An inline
heredoc would not parse there.
"""

from __future__ import annotations

import os
import subprocess
import sys

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, not the first
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
    else:
        print(f"ok   {name}{f' ({detail})' if detail else ''}")


def imports() -> str:
    import cv2
    import h5py
    import numpy as np
    import zarr

    # prepare_volume.py reads OME-Zarr and resamples with cv2.INTER_AREA;
    # neither zarr nor opencv-as-used is guaranteed by the base image.
    assert hasattr(zarr, "open_group")
    assert hasattr(cv2, "INTER_AREA")
    return f"zarr {zarr.__version__}, cv2 {cv2.__version__}, numpy {np.__version__}, h5py {h5py.__version__}"


def publishing_stack() -> str:
    # upload_seg_precomputed.py builds the precomputed layer with CloudVolume
    # and the meshes with igneous. Neither is a pyproject dependency.
    import cloudvolume
    import igneous.task_creation as tc

    assert hasattr(cloudvolume.CloudVolume, "create_new_info")
    assert hasattr(tc, "create_downsampling_tasks")
    assert hasattr(tc, "create_meshing_tasks")
    return f"cloud-volume {cloudvolume.__version__}"


def checkpoint_fetcher() -> str:
    import huggingface_hub

    assert hasattr(huggingface_hub, "hf_hub_download")
    return f"huggingface_hub {huggingface_hub.__version__}"


def abiss() -> str:
    ws = os.environ.get("ABISS_HOME", "/opt/abiss") + "/build/ws"
    assert os.access(ws, os.X_OK), f"{ws} is not executable"
    # `ws` with no arguments prints usage and exits non-zero. That it runs at
    # all is the point: a missing libboost or libtbb fails here, in the builder.
    proc = subprocess.run([ws], capture_output=True, text=True)
    assert proc.returncode != 0 or proc.stdout, "ABISS ws produced no output at all"
    return ws


def pipeline_entrypoints() -> str:
    # The five scripts the container actually invokes, and the one module whose
    # path logic decides where the affinity lands. Import-only: no GPU here.
    import importlib.util
    from pathlib import Path

    root = Path("/workspace")
    for rel in (
        "scripts/main.py",
        "tutorials/neuron_liconn_moe/volumes.py",
        "tutorials/neuron_liconn_moe/run_prepare.py",
        "tutorials/neuron_liconn_moe/prepare_volume.py",
        "tutorials/neuron_liconn_moe/make_volume_config.py",
        "tutorials/neuron_liconn_moe/sweep_merge_threshold.py",
        "tutorials/neuron_liconn_moe/upload_seg_precomputed.py",
        "scripts/run_abiss_volume.py",
    ):
        assert (root / rel).is_file(), f"missing {rel}"
    assert importlib.util.find_spec("connectomics.runtime.checkpoint_dispatch")
    return "8 scripts + checkpoint_dispatch"


def checkpoint_output_path() -> str:
    """The path rule the whole run depends on, asserted rather than trusted.

    `get_output_base_from_checkpoint` walks a checkpoint's parents for a
    `YYYYmmdd_HHMMSS` directory. With one, the affinity lands under it. WITHOUT
    one it falls back to `<ckpt>/../../<stem>` — for a checkpoint in a
    top-level directory that is a path at the filesystem root, and the run
    fails on permissions after the GPU work is done. `run_volume.sh` stages the
    checkpoint under such a directory; this pins the behaviour it relies on.
    """
    from connectomics.runtime.checkpoint_dispatch import get_checkpoint_test_output_dir

    got = get_checkpoint_test_output_dir(
        "/work/ckpt/20260921_000000/checkpoints/affinity_expid82_18nm_128x128x128.ckpt"
    )
    want = "/work/ckpt/20260921_000000/test_affinity_expid82_18nm_128x128x128"
    assert str(got) == want, f"expected {want}, got {got}"
    return want


for name, fn in (
    ("volume i/o and resample (zarr, cv2, h5py)", imports),
    ("precomputed publishing (cloud-volume, igneous)", publishing_stack),
    ("checkpoint fetcher (huggingface_hub)", checkpoint_fetcher),
    ("ABISS ws binary runs", abiss),
    ("pipeline scripts present", pipeline_entrypoints),
    ("checkpoint -> affinity output path", checkpoint_output_path),
):
    check(name, fn)

if FAILURES:
    print("\nIMAGE SMOKE FAILED:", file=sys.stderr)
    for line in FAILURES:
        print(f"  {line}", file=sys.stderr)
    raise SystemExit(1)
print("\nimage smoke passed")
