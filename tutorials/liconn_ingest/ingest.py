#!/usr/bin/env python3
"""LICONN raw drop -> uint8 OME-Zarr pyramid -> gs://donglai_public.

    fetch -> preprocess -> publish -> verify -> prune        (or: `run`)

Every stage is idempotent, writes `provenance.json`, and drops a `COMPLETE`
marker. `prune` refuses to delete anything the remote has not confirmed.

WHERE THE WORK RUNS. `preprocess` needs `nd2` and `zarr`, which the driving
machine does not have and which is what `Dockerfile` supplies; pass `--docker`
and this script re-invokes itself inside `pytc:liconn-ingest`. `fetch`
(rclone) and `publish`/`verify` (gcloud) run on the host, where the Drive and
GCS credentials already live. That split is the whole containerisation story.

THE PUBLISHED LAYOUT IS NOT A CHOICE. Volumes already sit at

    gs://donglai_public/liconn/moe/<expid>/image/<name>.zarr

with a five-level pyramid and the `.zattrs` keys `preprocess_liconn.py` writes.
`zarr_writer.py` reproduces that byte-for-byte in form; an earlier revision of
this file invented `<clip_variant>/zarr/<cube_id>.zarr`, which matches nothing
in the bucket. Names carry the drop folder (`..._cerebellum`) and, when one
Drive folder holds two files of the same name, the first 8 characters of the
Drive file ID (`..._cerebellum_1Byqvupl`) -- both conventions read back off the
published set, not invented here.

DRIVE FILENAMES ARE NOT UNIQUE. The `Cerebellum` drop holds two distinct files
both called `ExPID99_18x_2.nd2` (10.66 GB and 12.73 GB). `rclone copy` of that
folder silently keeps one. `fetch` therefore works from file IDs, one
`rclone copyid` per file, and records the ID in provenance.

CLIP VARIANT IS REQUIRED, NOT DEFAULTED. The published set is all
`clip_mode=percentile, clip_percentiles=[1,99]`, but `lessons/` calls fixed
`120-350` the reference recipe; the two disagree and the choice is scientific.
See spec.md OPEN FILL LIST. This tool will not pick for you.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
DEFAULT_IMAGE = "pytc:liconn-ingest"

# BANDWIDTH THROTTLE. Uncapped, `gcloud storage rsync` pushed ~7 MB/s of
# upload through this host's VPN tunnel. CPU, memory and disk were all idle,
# but the uplink was saturated, so interactive SSH (the operator is on
# Tailscale) froze in bursts while keystroke ACKs queued behind the bulk
# transfer -- textbook bufferbloat. The job runs overnight; the operator's
# terminal does not have that luxury.
#
# rclone takes --bwlimit directly. `gcloud storage` has no bandwidth flag at
# all, so it is throttled by cutting its concurrency instead, via environment
# variables rather than `gcloud config set` so the operator's global gcloud
# configuration is left alone.
DEFAULT_BWLIMIT = "3M"
# 1x1 is deliberate. Uncapped, gcloud fans out one process per core with 4
# threads each; that tree was measured pushing 7 MB/s. There is no bandwidth
# flag and no ADC on this host (so rclone, which does have --bwlimit, cannot
# reach GCS here) and no `trickle`, so serialising the transfer IS the
# throttle. Slower per volume, but the job runs unattended and the operator's
# SSH does not.
GCS_ENV = {"CLOUDSDK_STORAGE_PROCESS_COUNT": "1",
           "CLOUDSDK_STORAGE_THREAD_COUNT": "1"}


def _gcs_env() -> dict:
    env = dict(os.environ)
    env.update(GCS_ENV)
    return env
_EXPID = re.compile(r"^ExPID(\d+)", re.IGNORECASE)


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
    d = Path(work) / cube
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prov(dir_: Path) -> dict:
    p = Path(dir_) / "provenance.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _write_prov(dir_: Path, patch: dict) -> dict:
    prov = _prov(dir_)
    prov.update(patch)
    prov["updated"] = _now()
    (Path(dir_) / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    return prov


def expid_group(stem: str) -> str:
    """`ExPID99_32x_1` -> `expid99`. The bucket's first path component."""
    m = _EXPID.match(stem)
    if not m:
        raise ValueError(f"cannot derive an ExPID group from {stem!r}; expected an 'ExPID<n>' prefix")
    return f"expid{int(m.group(1))}"


def published_name(stem: str, drop_slug: str | None = None, drive_id: str | None = None) -> str:
    """`<stem>[_<drop>][_<driveid8>]` -- the published group name.

    The Drive ID suffix is only added when the caller knows the filename is
    ambiguous within its drop; adding it unconditionally would rename the
    volumes already in the bucket.
    """
    parts = [stem]
    if drop_slug:
        parts.append(drop_slug)
    if drive_id:
        parts.append(drive_id[:8])
    return "_".join(parts)


def gcs_dest(bucket: str, prefix: str, stem: str, name: str) -> str:
    return f"gs://{bucket}/{prefix}/{expid_group(stem)}/image/{name}.zarr"


# ---------------------------------------------------------------- fetch
def _rclone_listing(remote: str, folder_id: str) -> list[dict]:
    out = subprocess.check_output(
        ["rclone", "lsjson", f"{remote}:", "--drive-root-folder-id", folder_id], text=True)
    return [f for f in json.loads(out) if not f.get("IsDir") and f["Name"].lower().endswith(".nd2")]


def fetch_one(inbox: Path, entry: dict, remote: str = "gdrive",
              bwlimit: str = DEFAULT_BWLIMIT) -> Path:
    """Download one Drive file by ID. Idempotent on byte count.

    `rclone backend copyid`, not `rclone copyid`: copy-by-ID is a Drive BACKEND
    command and the top-level spelling does not exist (checked against rclone
    v1.60.1). A destination without a trailing slash is the target filename,
    which is how the Drive-ID prefix gets onto same-named files.
    """
    target = Path(inbox) / entry["local"]
    if target.exists():
        got = target.stat().st_size
        if got == entry["bytes"]:
            print(f"already fetched: {target.name}")
            return target
        print(f"discarding partial {target.name}: {got} bytes, expected {entry['bytes']}")
        target.unlink()
    _run(["rclone", "backend", "copyid", f"{remote}:", entry["drive_id"], str(target),
          "--stats", "60s", "--stats-one-line", "--bwlimit", bwlimit])
    if not target.exists():
        sys.exit(f"copyid reported success but {target} is missing")
    got = target.stat().st_size
    if got != entry["bytes"]:
        sys.exit(f"{target.name}: got {got} bytes, Drive lists {entry['bytes']}")
    print(f"fetched {target.name} ({got/2**30:.2f} GB)")
    return target


def cmd_fetch(a):
    """Drive folder (BY ID) -> local inbox, one `copyid` per file.

    By ID and not by name because the drop folder is not reachable from
    rclone's shared-with-me root listing, and because two files in it share a
    name -- see the module docstring.
    """
    if shutil.which("rclone") is None:
        sys.exit("rclone not installed, or not on PATH.")
    inbox = Path(a.inbox)
    inbox.mkdir(parents=True, exist_ok=True)

    files = _rclone_listing(a.remote, a.folder_id)
    names = [f["Name"] for f in files]
    drop = []
    for f in files:
        ambiguous = names.count(f["Name"]) > 1
        stem = naming.parse_stem(f["Name"]).stem
        local = inbox / (f"{f['ID'][:8]}_{f['Name']}" if ambiguous else f["Name"])
        drop.append({"drive_id": f["ID"], "drive_name": f["Name"], "bytes": f["Size"],
                     "stem": stem, "ambiguous_name": ambiguous, "local": local.name,
                     "published_name": published_name(
                         stem, a.drop_slug, f["ID"] if ambiguous else None)})

    print(f"{len(drop)} .nd2 in Drive folder {a.folder_id}")
    for d in drop:
        mark = "  (duplicate name)" if d["ambiguous_name"] else ""
        print(f"  {d['bytes']/2**30:6.2f} GB  {d['drive_name']} -> {d['published_name']}{mark}")

    (inbox / "drop.json").write_text(json.dumps(
        {"remote": a.remote, "folder_id": a.folder_id, "drop_slug": a.drop_slug,
         "fetched": _now(), "files": drop}, indent=2))

    if a.dry_run:
        print("\n[dry-run] nothing downloaded")
        return
    for d in drop:
        fetch_one(inbox, d, a.remote, a.bwlimit)


# ------------------------------------------------------------ preprocess
def _docker_preprocess(a) -> None:
    """Run THIS volume's preprocess inside the ingest image.

    The argv is rebuilt from the resolved arguments rather than forwarded from
    `sys.argv`: under `run` the real argv says `run ... --docker`, and
    replaying that inside the container would re-enter the driver and reach
    for a gcloud that is deliberately not installed there.
    """
    nd2 = Path(a.nd2).resolve()
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    name = a.name or published_name(naming.parse_stem(nd2.name).stem, a.drop_slug)
    argv = ["--work", str(work), "preprocess", str(nd2), "--name", name,
            "--clip-variant", a.clip_variant, "--clip-mode", a.clip_mode,
            "--clip-values", *[str(v) for v in a.clip_values],
            "--clip-limit", str(a.clip_limit), "--channel", str(a.channel),
            "--levels", str(a.levels)]
    if a.fold is not None:
        argv += ["--fold", str(a.fold)]
    if a.fold_unknown:
        argv.append("--fold-unknown")
    if getattr(a, "fold_assumed", False):
        argv.append("--fold-assumed")
    if getattr(a, "exposure_ms", None) is not None:
        argv += ["--exposure-ms", str(int(a.exposure_ms))]
    if getattr(a, "channel_name", None):
        argv += ["--channel-name", a.channel_name]
    if a.optics_spacing_nm:
        argv += ["--optics-spacing-nm", *[str(v) for v in a.optics_spacing_nm]]
    if a.drop_slug:
        argv += ["--drop-slug", a.drop_slug]
    if a.force:
        argv.append("--force")
    _run(["docker", "run", "--rm",
          "-u", f"{os.getuid()}:{os.getgid()}",
          "-v", f"{nd2.parent}:{nd2.parent}:ro",
          "-v", f"{work}:{work}",
          a.image, "python", "/workspace/tutorials/liconn_ingest/ingest.py", *argv])


def cmd_preprocess(a):
    """One ND2 -> uint8 OME-Zarr pyramid at NATIVE resolution."""
    if getattr(a, "docker", False):
        return _docker_preprocess(a)

    import nd2_source
    import zarr_writer

    src = Path(a.nd2)
    if not src.exists():
        sys.exit(f"no such file: {src}")
    parsed = naming.parse_stem(src.name)
    fold = a.fold if a.fold is not None else parsed.fold
    if fold is None and not a.fold_unknown:
        sys.exit(f"expansion fold not in filename {src.name!r}; pass --fold, or "
                 "--fold-unknown to publish raw optics spacing instead. The fold "
                 "is NOT in the ND2 either -- see naming.py.")
    if a.fold_unknown and fold is not None:
        sys.exit(f"--fold-unknown contradicts the fold {fold} available for {src.name!r}")
    if getattr(a, "fold_assumed", False) and a.fold is None:
        sys.exit("--fold-assumed needs an explicit --fold: it labels THAT value an assumption")

    exposure_ms = getattr(a, "exposure_ms", None)
    name = a.name or published_name(
        naming.cube_id(parsed.stem, exposure_ms), a.drop_slug)
    work = _state(a.work, name)
    if (work / "COMPLETE.preprocess").exists() and not a.force:
        print(f"already preprocessed: {work}")
        return

    info = nd2_source.inspect(src)
    optics = tuple(a.optics_spacing_nm) if a.optics_spacing_nm else tuple(info["optics_spacing_zyx_nm"])
    spacing_source = "cli" if a.optics_spacing_nm else "nd2_metadata"
    # With no fold there is nothing to divide by, so the recorded spacing is the
    # optics spacing and says so. See zarr_writer.provenance_attrs.
    if fold is None:
        basis = "optics_uncorrected"
    elif getattr(a, "fold_assumed", False):
        basis = "biological_assumed_fold"
    else:
        basis = "biological"
    spacing = optics if fold is None else naming.physical_spacing_nm(optics, fold)
    shape = tuple(info["shape_zyx"])
    print(f"{src.name}: {info['axes']} {info['shape']} {info['dtype']}  "
          f"optics {optics} nm ({spacing_source})  fold {fold}  -> {spacing} nm [{basis}]")
    if basis == "optics_uncorrected":
        print("  WARNING: expansion fold unknown -- publishing RAW OPTICS spacing. "
              "This group is not spatially comparable with the folded groups.")
    elif basis == "biological_assumed_fold":
        print(f"  NOTE: fold {fold} is ASSUMED, not recorded anywhere in the data. "
              "Spacing is biological and comparable, but rests on that assumption.")

    planes = nd2_source.iter_planes(src, channel=a.channel, clip_mode=a.clip_mode,
                                    clip_values=a.clip_values, clip_limit=a.clip_limit)
    attrs = zarr_writer.provenance_attrs(
        source=str(src), source_channel=a.channel, source_shape_zyx=shape,
        spacing_nm_zyx=spacing, clip_mode=a.clip_mode, clip_values=a.clip_values,
        clip_limit=a.clip_limit, spacing_basis=basis)
    dest = work / f"{name}.zarr"
    written = zarr_writer.write_pyramid(dest, planes, shape, spacing, attrs,
                                        levels=a.levels)

    _write_prov(work, {
        "name": name,
        "stem": parsed.stem,
        "expid_group": expid_group(parsed.stem),
        "source_nd2": src.name,
        "source_sha256": _sha256(src),          # survives deletion of the raw (I8)
        "source_bytes": src.stat().st_size,
        "nd2_axes": info["axes"],
        "nd2_shape": info["shape"],
        "nd2_dtype": info["dtype"],
        "expansion_fold": fold,
        "fold_source": ("assumed" if getattr(a, "fold_assumed", False)
                        else "cli" if a.fold is not None
                        else parsed.derived.get("fold", "unknown_declared"
                                                if a.fold_unknown else "unknown")),
        "spacing_basis": basis,
        "optics_spacing_zyx_nm": list(optics),
        "optics_spacing_source": spacing_source,
        "physical_spacing_zyx_nm": list(spacing) if fold is not None else None,
        "recorded_spacing_zyx_nm": list(spacing),
        "anisotropy_z_over_x": round(naming.anisotropy(optics), 4),
        "shape_zyx": list(shape),
        "dtype": "uint8",
        "levels": written["levels"],
        "level_shapes": written["shapes"],
        "clip_variant": a.clip_variant,
        "clip_mode": a.clip_mode,
        "clip_values": list(a.clip_values),
        "clip_limit": a.clip_limit,
        "channel": a.channel,
        "channel_name": getattr(a, "channel_name", None),
        "exposure_ms": exposure_ms,
        "exposure_source": "nd2_metadata" if exposure_ms is not None else None,
        "z_step_nm": parsed.z_step_nm,
        "filename_exposures_ms": sorted(parsed.exposures_ms) or None,
        "filename_derived": parsed.derived,
        "preprocessed": _now(),
    })
    (work / "COMPLETE.preprocess").write_text(_now())
    print(f"wrote {dest}  shape={shape}  spacing={spacing}  levels={written['levels']}")


def cmd_inspect(a):
    """Print the ND2's shape, spacing and per-channel exposures as JSON.

    Exists because `run` has to know the exposure of each channel BEFORE it can
    name the outputs, and the host has no `nd2` -- that is the whole reason for
    the image. `_inspect_via_docker` is the host-side caller.
    """
    import nd2_source

    info = nd2_source.inspect(a.nd2)
    try:
        info["channels"] = nd2_source.channel_exposures_ms(a.nd2)
    except ValueError as exc:
        info["channels_error"] = str(exc)
    print(json.dumps(info, indent=2, sort_keys=True))


def _inspect_via_docker(image: str, nd2: Path) -> dict:
    nd2 = Path(nd2).resolve()
    out = subprocess.check_output(
        ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
         "-v", f"{nd2.parent}:{nd2.parent}:ro", image,
         "python", "/workspace/tutorials/liconn_ingest/ingest.py",
         "inspect", str(nd2)], text=True)
    return json.loads(out[out.index("{"):])


def _exposure_plan(a, raw: Path, entry: dict) -> list:
    """[(name, channel, exposure_ms, channel_name)] for one raw file.

    Without --split-exposures this is one entry and nothing is read from the
    ND2. With it, the exposures come from ND2 metadata and are cross-checked
    against the filename's unordered set: the filename is trusted for WHICH
    exposures exist, never for which channel they are (I7).
    """
    if not a.split_exposures:
        return [(entry["published_name"], a.channel, None, None)]

    info = _inspect_via_docker(a.image, raw)
    if "channels_error" in info:
        sys.exit(f"{raw.name}: {info['channels_error']}")
    channels = {int(k): v for k, v in info["channels"].items()}
    from_nd2 = sorted(int(round(v["exposure_ms"])) for v in channels.values())
    from_name = sorted(naming.parse_stem(entry["drive_name"]).exposures_ms)
    if from_name and from_name != from_nd2:
        sys.exit(f"{raw.name}: filename lists exposures {from_name} but the ND2 "
                 f"records {from_nd2}; refusing to publish either reading")
    if len(channels) < 2:
        sys.exit(f"{raw.name}: --split-exposures but only {len(channels)} channel(s)")

    plan = []
    for idx in sorted(channels):
        ms = int(round(channels[idx]["exposure_ms"]))
        plan.append((published_name(naming.cube_id(entry["stem"], ms), a.drop_slug),
                     idx, ms, channels[idx]["name"]))
    return plan


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


def _local_objects(src: Path) -> dict:
    return {p.relative_to(src).as_posix(): p.stat().st_size
            for p in src.rglob("*") if p.is_file()}


def _remote_objects(dest: str) -> dict:
    out = subprocess.check_output(
        ["gcloud", "storage", "ls", "--recursive", "--long", dest], text=True)
    objects = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[-1].startswith(dest):
            objects[parts[-1][len(dest):].lstrip("/")] = int(parts[0])
    return objects


def cmd_publish(a):
    work = _state(a.work, a.name)
    prov = _prov(work)
    if not (work / "COMPLETE.preprocess").exists():
        sys.exit(f"{a.name} not preprocessed")
    src = work / f"{a.name}.zarr"
    _gzip_guard(src)
    dest = gcs_dest(a.bucket, a.prefix, prov["stem"], a.name)
    if a.dry_run:
        print(f"[dry-run] would rsync {src} -> {dest}")
        return
    _run(["gcloud", "storage", "rsync", "--recursive", str(src), dest], env=_gcs_env())
    _write_prov(work, {"published_to": dest, "published": _now()})
    (work / "COMPLETE.publish").write_text(_now())
    print(f"published -> {dest}")


# --------------------------------------------------------------- verify
def cmd_verify(a):
    """Every local object must exist remotely at the same size.

    The previous revision compared only object COUNTS, which passes when the
    remote holds the right number of wrong objects -- and the bucket already
    contains seven `*.zarr/` prefixes with the right shape of empty levels and
    no metadata at all, which is exactly what a count check cannot see.
    """
    work = _state(a.work, a.name)
    prov = _prov(work)
    dest = prov.get("published_to")
    if not dest:
        sys.exit(f"{a.name} has no published_to in provenance")
    src = work / f"{a.name}.zarr"
    local = _local_objects(src)
    remote = _remote_objects(dest)

    missing = sorted(set(local) - set(remote))
    mismatched = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])
    ok = not missing and not mismatched
    _write_prov(work, {"verified": _now(), "local_objects": len(local),
                       "local_bytes": sum(local.values()), "remote_objects": len(remote),
                       "missing_objects": missing[:20], "size_mismatches": mismatched[:20],
                       "verify_ok": bool(ok)})
    if not ok:
        sys.exit(f"VERIFY FAILED for {dest}: {len(missing)} missing, "
                 f"{len(mismatched)} size mismatches (first: {(missing + mismatched)[:3]})")
    (work / "COMPLETE.verify").write_text(_now())
    print(f"verified {len(local)} objects at {dest}")


def expected_object_count(zattrs: dict, zarrays: dict) -> int:
    """How many objects a COMPLETE group must hold, from metadata alone.

    Group `.zgroup` + `.zattrs`, then per level `.zarray` + `.zattrs` + one
    object per chunk. The per-level `.zattrs` is easy to forget -- zarr v2
    writes one (usually `{}`) beside every `.zarray`, and omitting it
    undercounts by exactly one per level, which would let a group missing a
    few chunks pass as complete. Checked against the real 1185-object
    ExPID71_2Hippocampus_500nm_40XW in test_ingest.py.

    Needs no local copy, so it works on a group whose raw and scratch are long
    gone.
    """
    total = 2
    for level in (d["path"] for d in zattrs["multiscales"][0]["datasets"]):
        za = zarrays[level]
        chunks = 1
        for extent, chunk in zip(za["shape"], za["chunks"]):
            chunks *= -(-extent // chunk)      # ceil
        total += 2 + chunks
    return total


class BucketUnreachable(RuntimeError):
    """gcloud could not answer -- which is NOT the same as 'the data is absent'.

    Raised rather than folded into a False so callers cannot quietly treat an
    expired token or a throttled API as 'this group was never published' and
    redo the work. That conflation cost a 7.75 GB re-download of an already
    verified volume on 2026-09-21, minutes after the same mistake was written
    up as a lesson.
    """


_NOT_FOUND = ("matched no objects", "matched no urls", "not found",
              "one or more urls matched no objects")


def remote_group_complete(dest: str) -> tuple:
    """(complete, found, expected, why) for a published group.

    Raises BucketUnreachable if the answer cannot be determined.

    WHY THIS IS NOT `ls .zattrs`. An interrupted `rsync` can leave `.zattrs`
    in place with most chunks missing, and the group then looks published to
    any check that only tests for that one object. That is not hypothetical:
    it happened to ExPID71_Hippocampus_600nm_40XW02 on 2026-09-21 (312 of 1185
    objects, skipped as done).
    """
    def _cat(path):
        r = subprocess.run(["gcloud", "storage", "cat", path],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout) if r.stdout.strip() else None
        err = (r.stderr or "").lower()
        if any(m in err for m in _NOT_FOUND):
            return None                     # genuinely absent
        raise BucketUnreachable(f"{path}: {(r.stderr or '').strip()[:200]}")

    zattrs = _cat(f"{dest}/.zattrs")
    if zattrs is None:
        return False, 0, None, "no .zattrs"
    if "multiscales" not in zattrs:
        return False, 0, None, ".zattrs has no multiscales"

    zarrays = {}
    for level in (d["path"] for d in zattrs["multiscales"][0]["datasets"]):
        za = _cat(f"{dest}/{level}/.zarray")
        if za is None:
            return False, 0, None, f"level {level} has no .zarray"
        zarrays[level] = za

    expected = expected_object_count(zattrs, zarrays)
    out = subprocess.run(["gcloud", "storage", "ls", "--recursive", dest],
                         capture_output=True, text=True)
    if out.returncode != 0 and not any(m in (out.stderr or "").lower() for m in _NOT_FOUND):
        raise BucketUnreachable(f"ls {dest}: {(out.stderr or '').strip()[:200]}")
    found = sum(1 for ln in out.stdout.splitlines()
                if ln.startswith("gs://") and not ln.rstrip().endswith((":", "/")))
    return found >= expected, found, expected, ("ok" if found >= expected
                                                else f"{expected - found} objects missing")


def cmd_check(a):
    """Audit published groups for completeness. Read-only."""
    prefix = a.target if a.target.startswith("gs://") else \
        f"gs://{a.bucket}/{a.prefix}/{a.target}/image"
    out = subprocess.run(["gcloud", "storage", "ls", prefix],
                         capture_output=True, text=True)
    groups = [ln.strip().rstrip("/") for ln in out.stdout.splitlines()
              if ln.strip().endswith(".zarr/")]
    if not groups:
        sys.exit(f"no *.zarr groups under {prefix}")
    bad = 0
    for g in groups:
        ok, found, expected, why = remote_group_complete(g)
        bad += not ok
        print(f"{'OK  ' if ok else 'BAD '} {g.rsplit('/', 1)[-1]:48s} "
              f"{found}/{expected if expected is not None else '?'}  {why}")
    print(f"\n{len(groups) - bad}/{len(groups)} complete")
    sys.exit(1 if bad else 0)


# --------------------------------------------------------------- refold
def cmd_refold(a):
    """Change a published group's expansion fold WITHOUT re-converting it.

    The fold never touches a pixel -- it divides the optics spacing to give
    biological spacing, so it lives entirely in `.zattrs` (the multiscales
    scales plus the provenance keys). Re-running `preprocess` to change it
    would redo ~30 minutes of CLAHE per volume and re-upload several GB to
    produce byte-identical chunks.

    So this rewrites `.zattrs` from `provenance.json` and re-uploads that one
    object. It regenerates the whole file rather than patching fields, so a
    refolded group is indistinguishable from one that was right first time.

    Runs on the host: no numpy, no container, no ND2 -- see the note at the top
    of zarr_writer.py.
    """
    import zarr_writer

    work = _state(a.work, a.name)
    prov = _prov(work)
    if not prov:
        sys.exit(f"no provenance.json for {a.name} in {a.work}")
    src = work / f"{a.name}.zarr"
    if not (src / ".zattrs").exists():
        sys.exit(f"{src} has no .zattrs to rewrite")

    optics = prov.get("optics_spacing_zyx_nm")
    shapes = prov.get("level_shapes")
    if not optics or not shapes:
        sys.exit(f"{a.name}: provenance lacks optics_spacing_zyx_nm / level_shapes")

    if a.fold is None:
        basis, spacing, fold = "optics_uncorrected", tuple(optics), None
    else:
        fold = a.fold
        spacing = naming.physical_spacing_nm(optics, fold)
        basis = "biological_assumed_fold" if a.fold_assumed else "biological"

    before = json.loads((src / ".zattrs").read_text())
    attrs = zarr_writer.provenance_attrs(
        source=prov["source"] if "source" in prov else prov.get("source_nd2", "unknown"),
        source_channel=prov.get("channel", 0),
        source_shape_zyx=prov["shape_zyx"],
        spacing_nm_zyx=spacing,
        clip_mode=prov.get("clip_mode", "percentile"),
        clip_values=prov.get("clip_values", [1.0, 99.0]),
        clip_limit=prov.get("clip_limit", 0.03),
        spacing_basis=basis)
    attrs["multiscales"] = zarr_writer.ome_multiscales(shapes, spacing, f"{a.name}.zarr")

    old_scale = before["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    print(f"{a.name}: fold {prov.get('expansion_fold')} -> {fold}")
    print(f"  level-0 scale {[round(v, 6) for v in old_scale]}"
          f"  ->  {[round(v, 6) for v in spacing]}   [{basis}]")
    if a.dry_run:
        print("  [dry-run] nothing written")
        return

    (src / ".zattrs").write_text(json.dumps(attrs, indent=2, sort_keys=True))
    _write_prov(work, {
        "expansion_fold": fold,
        "fold_source": (a.fold_source or
                        ("assumed" if a.fold_assumed else
                         ("cli" if fold is not None else "unknown_declared"))),
        "spacing_basis": basis,
        "physical_spacing_zyx_nm": list(spacing) if fold is not None else None,
        "recorded_spacing_zyx_nm": list(spacing),
        "refolded": _now(),
        "refolded_from": {"fold": prov.get("expansion_fold"),
                          "spacing": list(old_scale),
                          "basis": prov.get("spacing_basis")},
    })

    # `published_to` is only a cache of where this went; it goes missing when a
    # run is interrupted between publish and its provenance write. Deriving the
    # destination and asking the bucket is authoritative, and avoids the silent
    # half-success where the local .zattrs is relabelled and the remote is not
    # -- which in a loop over nine volumes reads exactly like nine successes.
    dest = prov.get("published_to") or gcs_dest(a.bucket, a.prefix, prov["stem"], a.name)
    exists = subprocess.run(["gcloud", "storage", "ls", f"{dest}/.zgroup"],
                            capture_output=True, text=True)
    if exists.returncode != 0:
        err = (exists.stderr or "").lower()
        if not any(m in err for m in _NOT_FOUND):
            sys.exit(f"cannot reach {dest}: {(exists.stderr or '').strip()[:200]}")
        print(f"  NOT PUBLISHED at {dest}; local .zattrs updated only")
        return
    _run(["gcloud", "storage", "cp", str(src / ".zattrs"), f"{dest}/.zattrs"],
         env=_gcs_env())
    # The pixels never changed, so verify still holds; only re-assert the one
    # object we touched.
    remote = _remote_objects(dest)
    if ".zattrs" not in remote:
        sys.exit(f"{dest}/.zattrs missing after upload")
    local_bytes = (src / ".zattrs").stat().st_size
    if remote[".zattrs"] != local_bytes:
        sys.exit(f"{dest}/.zattrs is {remote['.zattrs']} bytes, local is {local_bytes}")
    print(f"  re-published .zattrs -> {dest}")


# ---------------------------------------------------------------- prune
def cmd_prune(a):
    """Delete the local raw ND2. Refuses unless verify has passed.

    Order is hash -> convert -> verify -> delete (spec.md I8). The sha256 in
    provenance.json is the only surviving evidence of what was processed.
    """
    work = _state(a.work, a.name)
    prov = _prov(work)
    if not (work / "COMPLETE.verify").exists() or not prov.get("verify_ok"):
        sys.exit(f"refusing to prune {a.name}: verify has not passed")
    if not prov.get("source_sha256"):
        sys.exit(f"refusing to prune {a.name}: no source_sha256 in provenance")
    raw = Path(a.inbox) / prov["source_nd2"]
    if not raw.exists():
        # `fetch` prefixes the Drive ID onto duplicate names; try that too.
        candidates = [p for p in Path(a.inbox).glob(f"*{prov['source_nd2']}")]
        if len(candidates) != 1:
            print(f"already pruned (or not in inbox): {raw}")
            return
        raw = candidates[0]
    if a.dry_run:
        print(f"[dry-run] would verify sha256 and delete {raw} ({raw.stat().st_size/2**30:.2f} GB)")
        return
    if _sha256(raw) != prov["source_sha256"]:
        sys.exit(f"refusing to prune {raw}: sha256 does not match provenance")
    raw.unlink()
    _write_prov(work, {"raw_pruned": _now()})
    print(f"deleted {raw}")


# ------------------------------------------------------------------- run
def cmd_run(a):
    """fetch -> (preprocess -> publish -> verify -> prune) for a whole drop.

    Skips any volume already present in the bucket, so re-running after an
    interruption costs one `gcloud storage ls` per volume and nothing else.
    """
    drop_path = Path(a.inbox) / "drop.json"
    if not drop_path.exists():
        sys.exit(f"no {drop_path}; run `fetch` first (it writes the drop manifest)")
    drop = json.loads(drop_path.read_text())

    def _published(stem, name):
        """Complete, not merely present -- see remote_group_complete.

        A BucketUnreachable here is fatal for the whole drop on purpose. If we
        cannot read the bucket we cannot know what is already done, and
        carrying on means re-fetching and re-converting volumes that are
        finished -- which is precisely what happened when the gcloud token
        expired mid-run on 2026-09-21.
        """
        dest = gcs_dest(a.bucket, a.prefix, stem, name)
        ok, found, expected, why = remote_group_complete(dest)
        if not ok and found:
            print(f"   {name}: INCOMPLETE in bucket ({why}) -- will re-publish")
        return dest, ok

    try:
        _published(drop["files"][0]["stem"], drop["files"][0]["published_name"])
    except BucketUnreachable as exc:
        sys.exit(f"ABORTING: cannot read the bucket, so 'already published' cannot "
                 f"be established and continuing would redo finished work.\n  {exc}\n"
                 f"  If this is an expired token: gcloud auth login")

    for entry in drop["files"]:
        # Without --split-exposures the output name is known from the Drive
        # listing alone, so skip before spending a download.
        if not a.split_exposures:
            dest, done = _published(entry["stem"], entry["published_name"])
            if done and not a.force:
                print(f"== {entry['published_name']}: already published, skipping ({dest})")
                continue

        # Fetch lazily. Downloading a whole drop first means 112 GB on disk
        # before the first volume is published; fetching here keeps peak disk
        # at roughly one raw plus one zarr when --prune is on, and gets the
        # first published group out while the rest are still downloading.
        # ALWAYS go through fetch_one, never `if not raw.exists()`. A file
        # that exists is not a file that is complete: an interrupted transfer
        # leaves a short .nd2 behind, and gating on existence alone hands that
        # truncated file straight to the reader. That is exactly how
        # ExPID107_14.5x_04.nd2 reached the ND2 parser at 7.16 of 9.15 GB on
        # 2026-09-21 and died on an invalid ChunkMap signature. fetch_one is
        # idempotent and validates the size itself.
        raw = fetch_one(Path(a.inbox), entry, a.remote, a.bwlimit)

        plan = _exposure_plan(a, raw, entry)
        done_all = True
        for name, channel, exposure_ms, channel_name in plan:
            dest, done = _published(entry["stem"], name)
            if done and not a.force:
                print(f"== {name}: already published, skipping ({dest})")
                continue
            print(f"\n== {name}"
                  + (f"  (channel {channel}, {exposure_ms} ms, {channel_name!r})"
                     if exposure_ms is not None else ""))
            stage = argparse.Namespace(**vars(a))
            stage.nd2, stage.name, stage.channel = str(raw), name, channel
            stage.exposure_ms, stage.channel_name = exposure_ms, channel_name
            try:
                cmd_preprocess(stage)
                cmd_publish(stage)
                cmd_verify(stage)
            except (SystemExit, subprocess.CalledProcessError, OSError) as exc:
                # One bad volume must not strand the rest of the drop. Catching
                # only SystemExit was not enough: a non-zero docker exit raises
                # CalledProcessError, which on 2026-09-21 killed the whole
                # expid107 run over one corrupt .nd2 and left two volumes
                # unprocessed.
                print(f"!! {name} FAILED: {type(exc).__name__}: {exc}")
                done_all = False
        # Prune only once every cube derived from this raw has landed. A
        # refusal here (nothing verified, hash drift) must not kill the batch:
        # keeping a raw is cheap, losing the remaining volumes is not.
        if a.prune and done_all:
            stage = argparse.Namespace(**vars(a))
            stage.name = plan[0][0]
            try:
                cmd_prune(stage)
            except SystemExit as exc:
                print(f"!! prune skipped for {entry['local']}: {exc}")


def _add_preprocess_args(p):
    p.add_argument("--clip-variant", required=True,
                   help="Load-bearing provenance label, e.g. clip_percentile_1_99 "
                        "or clip_fixed_120_350. No default on purpose.")
    p.add_argument("--clip-mode", choices=("percentile", "fixed_intensity"),
                   default="percentile")
    p.add_argument("--clip-values", type=float, nargs=2, default=[1.0, 99.0],
                   help="Percentiles for --clip-mode percentile, raw intensities for fixed.")
    p.add_argument("--clip-limit", type=float, default=0.03)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--exposure-ms", type=int, default=None,
                   help="Suffixes the name as <stem>_eNNN. From ND2 metadata, NEVER "
                        "the filename (I7). Set automatically by --split-exposures.")
    p.add_argument("--channel-name", default=None, help="ND2 channel name, for provenance.")
    p.add_argument("--split-exposures", action="store_true",
                   help="One ND2 holding two exposures of the same plane is two cubes, "
                        "not a two-channel volume (spec.md). Emits <stem>_e030/_e120.")
    p.add_argument("--fold", type=float, default=None, help="Overrides the filename.")
    p.add_argument("--fold-assumed", action="store_true",
                   help="The --fold value is an ASSUMPTION, not read from the data. "
                        "Records fold_source=assumed and a spacing_warning in .zattrs.")
    p.add_argument("--fold-unknown", action="store_true",
                   help="No expansion fold exists for this specimen: record RAW OPTICS "
                        "spacing and flag it in .zattrs. Not comparable with folded groups.")
    p.add_argument("--optics-spacing-nm", type=float, nargs=3, default=None,
                   metavar=("Z", "Y", "X"), help="Default: read from the ND2.")
    p.add_argument("--levels", type=int, default=4, help="Pyramid levels below level 0")
    p.add_argument("--drop-slug", default=None,
                   help="Drop folder tag appended to the published name, e.g. cerebellum.")
    p.add_argument("--docker", action="store_true", help="Run preprocess in the ingest image")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--force", action="store_true")


def _add_gcs_args(p):
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="./liconn_ingest_work")
    p.add_argument("--inbox", default="./inbox")
    p.add_argument("--bwlimit", default=DEFAULT_BWLIMIT,
                   help="rclone bandwidth cap (e.g. 3M, or 'off'). Protects "
                        "interactive SSH from the bulk transfer -- see the note "
                        "at the top of this file.")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Drive folder (by ID) -> local inbox")
    f.add_argument("--remote", default="gdrive")
    f.add_argument("--folder-id", required=True, help="Drive folder ID, not its name.")
    f.add_argument("--drop-slug", default=None)
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(func=cmd_fetch)

    q = sub.add_parser("preprocess", help="ND2 -> uint8 OME-Zarr pyramid")
    q.add_argument("nd2")
    q.add_argument("--name", default=None, help="Published group name; default derives it.")
    _add_preprocess_args(q)
    q.set_defaults(func=cmd_preprocess)

    for cmd, fn, doc in (("publish", cmd_publish, "upload the zarr to GCS"),
                         ("verify", cmd_verify, "confirm the remote copy object by object"),
                         ("prune", cmd_prune, "delete the local raw ND2")):
        s = sub.add_parser(cmd, help=doc)
        s.add_argument("name")
        if cmd in ("publish", "verify"):
            _add_gcs_args(s)
        if cmd in ("publish", "prune"):
            s.add_argument("--dry-run", action="store_true")
        s.set_defaults(func=fn)

    ck = sub.add_parser("check", help="audit published groups for completeness")
    ck.add_argument("target", help="an expid (e.g. expid96) or a full gs:// prefix")
    _add_gcs_args(ck)
    ck.set_defaults(func=cmd_check)

    rf = sub.add_parser("refold", help="change a group's expansion fold (metadata only)")
    rf.add_argument("name")
    rf.add_argument("--fold", type=float, default=None,
                    help="New fold. Omit to revert to raw optics spacing.")
    rf.add_argument("--fold-assumed", action="store_true",
                    help="Label the new fold an assumption in .zattrs.")
    rf.add_argument("--fold-source", default=None,
                    help="What the fold rests on, recorded in provenance -- e.g. "
                         "'submitter_confirmed_2026-09-22'. Without this a value "
                         "passed by hand is only ever recorded as 'cli', which says "
                         "how it arrived and not why it is believed.")
    rf.add_argument("--dry-run", action="store_true")
    _add_gcs_args(rf)
    rf.set_defaults(func=cmd_refold)

    i = sub.add_parser("inspect", help="print ND2 shape, spacing and per-channel exposures")
    i.add_argument("nd2")
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("run", help="whole drop: preprocess -> publish -> verify [-> prune]")
    _add_preprocess_args(r)
    _add_gcs_args(r)
    r.add_argument("--remote", default="gdrive", help="rclone remote for the lazy fetch")
    r.add_argument("--prune", action="store_true", help="delete each raw once verified")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
