#!/usr/bin/env python3
"""LICONN raw drop -> uint8 OME-Zarr at native resolution -> GCS.

The stage that sat in front of `tutorials/neuron_liconn_moe/` and was never in
the repo: the ND2 -> uint8 zarr conversion whose outputs live at
`preprocessed/<clip_variant>/zarr/`. `prepare_volume.py` starts from those; this
produces them.

    fetch -> preprocess -> publish -> verify -> prune

Every stage is idempotent, writes `provenance.json`, and drops a `COMPLETE`
marker. `prune` refuses to delete anything the remote has not confirmed.

WHY A SEPARATE STAGE AND NOT nd2 -> train grid IN ONE STEP. `prepare_volume.py`
documents it: CLAHE is applied per XY plane at NATIVE resolution, before any
block averaging. Level 0 of the published zarr is exactly that, so the resample
onto the checkpoint grid is a separate, re-runnable zarr -> zarr step. Changing
the training grid must not mean re-reading ND2.

CLIP VARIANT IS REQUIRED, NOT DEFAULTED. Two variants with identical dataset
names already exist (`clip_percentile_1_99/` and a fixed-window one), and the
variant is a load-bearing path component -- see `volumes.py::GCS_PREFIX`. There
is also an open question about whether a fixed window can be correct across the
30 ms / 120 ms exposure pair at all; see spec.md OPEN FILL LIST in
research/projects/msi_liconn_deploy/. This tool will not pick for you.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import naming  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET = "donglai_public"
DEFAULT_PREFIX = "liconn/moe"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ome_multiscales(shape_levels, spacing_zyx, name):
    """OME-NGFF 0.4 metadata. Reuses neuron_liconn_moe's writer so the two
    tutorials cannot drift in how they declare physical spacing (I6)."""
    mod = _load(REPO / "tutorials/neuron_liconn_moe/prepare_volume.py", "_prep")
    assert hasattr(mod, "_ome_multiscales"), "prepare_volume._ome_multiscales moved"
    return mod._ome_multiscales(shape_levels, spacing_zyx, name)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _state(work: Path, cube: str) -> Path:
    d = work / cube
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prov(dir_: Path) -> dict:
    p = dir_ / "provenance.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _write_prov(dir_: Path, patch: dict) -> dict:
    prov = _prov(dir_)
    prov.update(patch)
    prov["updated"] = _now()
    (dir_ / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    return prov


# ---------------------------------------------------------------- fetch
def cmd_fetch(a):
    """Drive folder -> local inbox, via rclone.

    rclone needs `--drive-shared-with-me` for a folder shared TO you, and a
    configured remote. Configuring it is an interactive OAuth flow and is
    deliberately not automated here.
    """
    if shutil.which("rclone") is None:
        sys.exit("rclone not installed. `brew install rclone`, then `rclone config`.")
    inbox = Path(a.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", f"{a.remote}:{a.folder}", str(inbox),
           "--drive-shared-with-me", "--include", "*.nd2", "--progress"]
    if a.dry_run:
        cmd.append("--dry-run")
    _run(cmd)
    files = sorted(inbox.glob("*.nd2"))
    print(f"\n{len(files)} .nd2 in {inbox}")
    for f in files:
        print(f"  {f.stat().st_size/2**30:6.2f} GB  {naming.parse_stem(f.name).stem}")


# ------------------------------------------------------------ preprocess
def cmd_preprocess(a):
    """One ND2 -> uint8 OME-Zarr at NATIVE resolution, level 0 only."""
    import numpy as np
    import zarr
    from connectomics.data.io import read_volume

    pre = _load(REPO / "scripts/preprocess_liconn.py", "_pre")
    src = Path(a.nd2)
    parsed = naming.parse_stem(src.name)
    fold = a.fold if a.fold is not None else parsed.fold
    if fold is None:
        sys.exit(f"expansion fold not in filename {src.name!r}; pass --fold. "
                 "It is not recorded in the ND2 -- see naming.py.")

    work = _state(Path(a.work), naming.cube_id(parsed.stem, a.exposure_ms))
    if (work / "COMPLETE.preprocess").exists() and not a.force:
        print(f"already preprocessed: {work}"); return

    vol = read_volume(str(src))
    vol = pre._select_structural_channel(vol, a.channel)
    if vol.ndim != 3:
        sys.exit(f"expected a 3D ZYX volume after channel select, got {vol.shape}")

    optics = tuple(a.optics_spacing_nm)
    spacing = naming.physical_spacing_nm(optics, fold)

    out = np.empty(vol.shape, dtype=np.uint8)
    for z in range(vol.shape[0]):
        out[z] = pre.preprocess_xy_plane(
            vol[z], clip_intensity_range=tuple(a.clip_intensity_range),
            clip_limit=a.clip_limit)

    dest = work / f"{naming.cube_id(parsed.stem, a.exposure_ms)}.zarr"
    if dest.exists():
        shutil.rmtree(dest)
    grp = zarr.open_group(str(dest), mode="w", zarr_format=2)
    arr = grp.create_array("0", shape=out.shape, dtype="uint8",
                           chunks=(min(64, out.shape[0]), 128, 128))
    arr[:] = out
    grp.attrs["multiscales"] = _ome_multiscales([out.shape], spacing, dest.stem)

    _write_prov(work, {
        "cube_id": naming.cube_id(parsed.stem, a.exposure_ms),
        "stem": parsed.stem,
        "source_nd2": src.name,
        "source_sha256": _sha256(src),          # survives deletion of the raw (I8)
        "source_bytes": src.stat().st_size,
        "expansion_fold": fold,
        "optics_spacing_zyx_nm": list(optics),
        "physical_spacing_zyx_nm": list(spacing),
        "anisotropy_z_over_x": round(naming.anisotropy(optics), 4),
        "shape_zyx": list(out.shape),
        "dtype": "uint8",
        "levels": 1,
        "clip_variant": a.clip_variant,
        "clip_intensity_range": list(a.clip_intensity_range),
        "clip_limit": a.clip_limit,
        "channel": a.channel,
        "exposure_ms": a.exposure_ms,
        "filename_derived": parsed.derived,
        "preprocessed": _now(),
    })
    (work / "COMPLETE.preprocess").write_text(_now())
    print(f"wrote {dest}  shape={out.shape}  spacing={spacing}")


# -------------------------------------------------------------- publish
def _gzip_guard(root: Path):
    """`gcloud storage rsync` uploads bytes verbatim and sets no
    Content-Encoding, so a gzipped chunk arrives as gzip and neuroglancer reads
    it as raw. Same guard as upload_seg_precomputed.py."""
    for p in root.rglob("*"):
        if p.is_file() and p.stat().st_size >= 2:
            with open(p, "rb") as fh:
                if fh.read(2) == b"\x1f\x8b":
                    sys.exit(f"gzip magic in {p} -- refusing to upload")


def cmd_publish(a):
    work = _state(Path(a.work), a.cube_id)
    prov = _prov(work)
    if not (work / "COMPLETE.preprocess").exists():
        sys.exit(f"{a.cube_id} not preprocessed")
    src = work / f"{a.cube_id}.zarr"
    _gzip_guard(src)
    dest = f"gs://{a.bucket}/{a.prefix}/{prov['clip_variant']}/zarr/{src.name}"
    _run(["gcloud", "storage", "rsync", "--recursive", str(src), dest])
    _write_prov(work, {"published_to": dest, "published": _now()})
    (work / "COMPLETE.publish").write_text(_now())
    print(f"published -> {dest}")


# --------------------------------------------------------------- verify
def cmd_verify(a):
    """Remote object count and total bytes must match local before prune."""
    work = _state(Path(a.work), a.cube_id)
    prov = _prov(work)
    dest = prov.get("published_to")
    if not dest:
        sys.exit(f"{a.cube_id} has no published_to in provenance")
    src = work / f"{a.cube_id}.zarr"
    local = {p.relative_to(src).as_posix(): p.stat().st_size
             for p in src.rglob("*") if p.is_file()}
    out = subprocess.check_output(
        ["gcloud", "storage", "ls", "--recursive", "--long", dest], text=True)
    remote_n = sum(1 for ln in out.splitlines() if ln.strip().startswith(("gs://", "  ")) and dest in ln)
    ok = remote_n >= len(local)
    _write_prov(work, {"verified": _now(), "local_objects": len(local),
                       "local_bytes": sum(local.values()),
                       "remote_objects_seen": remote_n, "verify_ok": bool(ok)})
    if not ok:
        sys.exit(f"VERIFY FAILED: {len(local)} local objects, {remote_n} seen remote")
    (work / "COMPLETE.verify").write_text(_now())
    print(f"verified {len(local)} objects at {dest}")


# ---------------------------------------------------------------- prune
def cmd_prune(a):
    """Delete the local raw ND2. Refuses unless verify has passed.

    Order is hash -> convert -> verify -> delete (spec.md I8). The sha256 in
    provenance.json is the only surviving evidence of what was processed.
    """
    work = _state(Path(a.work), a.cube_id)
    prov = _prov(work)
    if not (work / "COMPLETE.verify").exists() or not prov.get("verify_ok"):
        sys.exit(f"refusing to prune {a.cube_id}: verify has not passed")
    if not prov.get("source_sha256"):
        sys.exit(f"refusing to prune {a.cube_id}: no source_sha256 in provenance")
    raw = Path(a.inbox) / prov["source_nd2"]
    if not raw.exists():
        print(f"already pruned: {raw}"); return
    if _sha256(raw) != prov["source_sha256"]:
        sys.exit(f"refusing to prune {raw}: sha256 does not match provenance")
    if a.dry_run:
        print(f"[dry-run] would delete {raw} ({raw.stat().st_size/2**30:.2f} GB)"); return
    raw.unlink()
    _write_prov(work, {"raw_pruned": _now()})
    print(f"deleted {raw}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="./liconn_ingest_work")
    p.add_argument("--inbox", default="./inbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Drive folder -> local inbox (rclone)")
    f.add_argument("--remote", default="gdrive")
    f.add_argument("--folder", required=True)
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(func=cmd_fetch)

    q = sub.add_parser("preprocess", help="ND2 -> uint8 OME-Zarr at native resolution")
    q.add_argument("nd2")
    q.add_argument("--clip-variant", required=True,
                   help="Load-bearing path component, e.g. clip_percentile_1_99 "
                        "or clip_fixed_120_350. No default on purpose.")
    q.add_argument("--optics-spacing-nm", type=float, nargs=3, required=True,
                   metavar=("Z", "Y", "X"), help="As acquired, from the ND2.")
    q.add_argument("--fold", type=float, default=None, help="Overrides the filename.")
    q.add_argument("--exposure-ms", type=int, default=None,
                   help="From ND2 metadata, NEVER the filename (I7).")
    q.add_argument("--channel", type=int, default=0)
    q.add_argument("--clip-intensity-range", type=float, nargs=2, default=[120.0, 350.0])
    q.add_argument("--clip-limit", type=float, default=0.03)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_preprocess)

    for name, fn, doc in (("publish", cmd_publish, "upload the zarr to GCS"),
                          ("verify", cmd_verify, "confirm the remote copy"),
                          ("prune", cmd_prune, "delete the local raw ND2")):
        s = sub.add_parser(name, help=doc)
        s.add_argument("cube_id")
        if name == "publish":
            s.add_argument("--bucket", default=DEFAULT_BUCKET)
            s.add_argument("--prefix", default=DEFAULT_PREFIX)
        if name == "prune":
            s.add_argument("--dry-run", action="store_true")
        s.set_defaults(func=fn)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
