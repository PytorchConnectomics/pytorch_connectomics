#!/usr/bin/env python3
"""Publish a moe ABISS segmentation as a neuroglancer precomputed layer + meshes.

Target `gs://donglai_public/liconn/moe/<family>/<kind>/` (see
`volumes.py::gcs_folder`), which
also holds the OME-Zarr image groups, so image and segmentation are in ONE
bucket. (Until 2026-09-16 the images lived in the private `donglai` bucket and
this file said overlaying spanned two buckets; the data was moved and
`gs://donglai/liconn/` is now empty. Confirmed by Donglai 2026-09-16 and by
listing: 12 .zarr image groups and 13 seg layers under the public prefix.)
Alignment is unaffected either way: both carry true
physical resolution and share an origin. The image group is at its native
spacing, this layer is at the prepared volume's *effective* spacing (the
`spacing_zyx` in `volumes.py::plan`, which is `native_shape * native / shape` --
NOT the nominal [24,18,18] target). Area resampling maps the whole source extent
onto the whole output extent, so corners coincide and no voxel_offset is needed;
using the nominal spacing instead would drift the layer by up to a voxel per
hundred across the field.

    python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume <name> --create --downsample
    python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume <name> --mesh --parallel 16
    python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume <name> --upload --verify

WHY IT IS BUILT LOCALLY AND THEN rsync'd, RATHER THAN WRITTEN STRAIGHT TO GCS.
There is no `~/.config/gcloud/application_default_credentials.json` on this host,
so CloudVolume / igneous / tensorstore cannot authenticate to either bucket at
all; only the `gcloud` CLI can (as `donglai@mindspan.org`). The compute service
account `pytc-trainer@...` has objectViewer on `donglai` but not objectCreator,
so a key would not help either.

THE ENCODING IS LOAD-BEARING, NOT AN OPTIMISATION. `gcloud storage rsync` uploads
bytes verbatim and does NOT set `Content-Encoding: gzip`. Anything CloudVolume
gzips locally (its `compress=True`, and igneous meshing's `compress='gzip'`
DEFAULT) would arrive as gzip bytes that neuroglancer reads as if they were raw
and fails on. So everything here is written with compression OFF, and the size is
recovered instead via `compressed_segmentation`, which neuroglancer decodes
natively with no HTTP content negotiation. Do not "optimise" by turning gzip back
on unless you also switch to a writer that sets the object metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402

CHUNK = [128, 128, 64]
MESH_DIR = "mesh_mip_0_err_40"
# The interactive login lives in whichever gcloud the user actually runs, so
# prefer the one on PATH and keep the BC install as the fallback. The container
# has no gcloud at all -- it only ever runs the --create/--downsample/--mesh
# stages, and the host uploads.
GCLOUD = (shutil.which("gcloud")
          or os.environ.get("GCLOUD")
          or str(Path.home() / "google-cloud-sdk/bin/gcloud"))


class Target:
    """Everything the five stages need, resolved from the volume name."""

    def __init__(self, name: str, seg: Path | None = None, layer: str | None = None):
        self.name = name
        summary_path = V.work_dir(name) / "mt_sweep.json"
        if seg is None or layer is None:
            if not summary_path.exists():
                raise SystemExit(f"{summary_path} missing -- run sweep_merge_threshold.py first")
            s = json.loads(summary_path.read_text())
            seg = seg or Path(s["segmentation"])
            layer = layer or V.layer_name(name, s["chosen"]["mt"])
        self.seg_h5 = Path(seg)
        self.layer = layer
        self.staging = V.OUT_ROOT / "precomputed" / self.layer
        # <family>/<kind>/<layer>; kind follows the scoped OUT_ROOT (eb2|eb8).
        self.gcs = f"{V.gcs_folder(name)}/{self.layer}"
        # CloudVolume wants XYZ; `spacing_zyx` is the effective spacing.
        self.res_xyz = [float(v) for v in reversed(V.plan(name)["spacing_zyx"])]


def _read_seg(t: Target) -> np.ndarray:
    import h5py

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    # `run_abiss_volume.py` writes uint64 in sweep mode (the in-pipeline decoder
    # would have cast it); precomputed segmentation layers are uint32. Cast per
    # z slab so a mip0 volume (5-6 Gvox) never holds the uint64 copy too.
    with h5py.File(t.seg_h5, "r") as f:
        d = f["main"]
        seg = np.empty(d.shape, dtype=np.uint32)
        top = 0
        for z0 in range(0, d.shape[0], 64):
            slab = d[z0:z0 + 64]
            top = max(top, int(slab.max()))
            if top >= 2**32:
                raise RuntimeError(f"{t.seg_h5} max id {top} does not fit uint32")
            seg[z0:z0 + 64] = slab
    if top == 0:
        raise RuntimeError(
            f"{t.seg_h5} has no foreground. A decode that writes an all-zero volume "
            "still exits [OK]; see tutorials/neuron_liconn_moe/README.md."
        )
    return seg


def do_create(t: Target) -> None:
    from cloudvolume import CloudVolume

    seg = _read_seg(t)  # (Z, Y, X)
    Zc, Yc, Xc = seg.shape
    print(f"source {seg.shape} {seg.dtype} max_id {int(seg.max())} res_xyz {t.res_xyz}", flush=True)
    t.staging.parent.mkdir(parents=True, exist_ok=True)
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint32",
        encoding="compressed_segmentation",
        resolution=t.res_xyz,
        voxel_offset=[0, 0, 0],
        volume_size=[Xc, Yc, Zc],
        chunk_size=CHUNK,
    )
    cv = CloudVolume(f"file://{t.staging}", info=info, compress=False, progress=False)
    cv.commit_info()
    # Move it in z slabs so the XYZ transpose never doubles peak RSS.
    for z0 in range(0, Zc, 64):
        z1 = min(z0 + 64, Zc)
        cv[:, :, z0:z1] = np.transpose(seg[z0:z1], (2, 1, 0))[..., np.newaxis]
    print(f"mip0 written -> {t.staging}", flush=True)


def do_downsample(t: Target, num_mips: int, parallel: int) -> None:
    import igneous.task_creation as tc
    from cloudvolume import CloudVolume
    from taskqueue import LocalTaskQueue

    tq = LocalTaskQueue(parallel=parallel)
    # factor is explicit: at ~18x18x24 nm the voxel is near-isotropic, so the
    # default z-last schedule would needlessly keep z at full resolution.
    tq.insert(
        tc.create_downsampling_tasks(
            f"file://{t.staging}", mip=0, num_mips=num_mips, factor=(2, 2, 2),
            compress=False, preserve_chunk_size=True,
        )
    )
    tq.execute()
    print("scales:", [s["key"] for s in CloudVolume(f"file://{t.staging}").info["scales"]], flush=True)


def do_mesh(t: Target, min_voxels: int, parallel: int) -> None:
    import igneous.task_creation as tc
    from cloudvolume import CloudVolume
    from taskqueue import LocalTaskQueue

    seg = _read_seg(t)
    ids, cnt = np.unique(seg, return_counts=True)
    keep = ids[(ids != 0) & (cnt >= min_voxels)]
    fg = cnt[ids != 0].sum()
    print(
        f"meshing {len(keep)} of {int((ids != 0).sum())} objects "
        f"(>= {min_voxels} vox = {cnt[(ids != 0) & (cnt >= min_voxels)].sum() / fg:.3f} "
        f"of foreground; median object is {np.percentile(cnt[ids != 0], 50):.0f} vox "
        "of ABISS dust)",
        flush=True,
    )
    del seg

    tq = LocalTaskQueue(parallel=parallel)
    tq.insert(
        tc.create_meshing_tasks(
            f"file://{t.staging}", mip=0, mesh_dir=MESH_DIR,
            max_simplification_error=40, shape=(256, 256, 256),
            object_ids=[int(i) for i in keep],
            compress=False,          # see module docstring
            fill_missing=True,
        )
    )
    tq.execute()
    print("building manifests...", flush=True)
    tq.insert(tc.create_mesh_manifest_tasks(f"file://{t.staging}", mesh_dir=MESH_DIR))
    tq.execute()

    cv = CloudVolume(f"file://{t.staging}")
    cv.info["mesh"] = MESH_DIR
    cv.commit_info()
    print(f"mesh attached: mesh_dir={MESH_DIR}", flush=True)


def _tree_size(root: Path) -> tuple[int, int]:
    n = sum(len(f) for _, _, f in os.walk(root))
    b = sum((Path(r) / x).stat().st_size for r, _, f in os.walk(root) for x in f)
    return n, b


def _gzip_guard(t: Target) -> None:
    """Refuse to upload if anything got gzipped after all."""
    bad = []
    for root, _, files in os.walk(t.staging):
        for name in files:
            p = Path(root) / name
            if p.suffix == ".json" or name == "info":
                continue
            with open(p, "rb") as fh:
                if fh.read(2) == b"\x1f\x8b":
                    bad.append(str(p))
            if len(bad) > 5:
                break
        if len(bad) > 5:
            break
    if bad:
        raise RuntimeError(
            "gzip-compressed chunks found; `gcloud storage rsync` will not set "
            "Content-Encoding and neuroglancer will fail to read them:\n  "
            + "\n  ".join(bad)
        )


def do_upload(t: Target) -> None:
    _gzip_guard(t)
    n, b = _tree_size(t.staging)
    print(f"uploading {n} objects / {b / 1e9:.2f} GB -> {t.gcs}", flush=True)
    subprocess.run([GCLOUD, "storage", "rsync", "-r", str(t.staging), t.gcs], check=True)
    # layer_url now needs the volume: the GCS path is <family>/<kind>/<layer>,
    # and family is derived from the volume name, not the layer name.
    print(f"\n{V.layer_url(t.layer, t.name)}", flush=True)


def do_verify(t: Target) -> bool:
    loc_n, loc_b = _tree_size(t.staging)
    rem_b = int(subprocess.run([GCLOUD, "storage", "du", "-s", t.gcs],
                               capture_output=True, text=True, check=True).stdout.split()[0])
    rem_n = subprocess.run([GCLOUD, "storage", "ls", "-r", f"{t.gcs}/**"],
                           capture_output=True, text=True, check=True).stdout.strip().count("\n") + 1
    ok = (loc_b == rem_b and loc_n == rem_n)
    print(f"local  {loc_n:7d} objects  {loc_b:14d} bytes")
    print(f"remote {rem_n:7d} objects  {rem_b:14d} bytes")
    print("MATCH" if ok else "MISMATCH")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True, help="moe volume name (see volumes.py)")
    ap.add_argument("--seg", type=Path, help="override the segmentation h5")
    ap.add_argument("--layer", help="override the published layer name")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--downsample", action="store_true")
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--num-mips", type=int, default=3)
    ap.add_argument("--min-voxels", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()

    t = Target(a.volume, a.seg, a.layer)
    print(f"layer {t.layer}\n  seg {t.seg_h5}\n  staging {t.staging}\n  gcs {t.gcs}", flush=True)
    if a.create:
        do_create(t)
    if a.downsample:
        do_downsample(t, a.num_mips, a.parallel)
    if a.mesh:
        do_mesh(t, a.min_voxels, a.parallel)
    if a.upload:
        do_upload(t)
    if a.verify:
        return 0 if do_verify(t) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
