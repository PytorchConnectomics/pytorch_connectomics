#!/usr/bin/env python3
"""Publish one volume's error-analysis and semantic sidecars beside its seg layer.

Generalizes the ExPID96 S1 `semantic/publish.py` to any run directory: the
destination is derived by `volume.py` from the run dir, never typed. Every
object is skipped if the live bytes already match, else uploaded only if its
generation is unchanged since it was read, then downloaded and compared by
SHA-256. Replaced objects are not kept (one copy) unless --keep-previous. Refuses to publish
into a bucket that answers anonymous listings. Bucket permissions are never
changed.

    LICONN_EA_RUN=<run dir> python publish.py --dry-run
    LICONN_EA_RUN=<run dir> python publish.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

GCLOUD = str(Path.home() / "google-cloud-sdk/bin/gcloud")
# (local path relative to the run dir, remote name inside the layer, content type,
# gzip). segment_properties/info is what Neuroglancer renders -- the five classes
# as `class:` tags -- and is fetched directly, so it is stored uncompressed.
FILES = [
    ("error_analysis/error_analysis.json", "error_analysis.json", "application/json", True),
    ("error_analysis/split_links.json", "split_links.json", "application/json", True),
    ("semantic/semantic_segmentation.json", "semantic_segmentation.json", "application/json", True),
    ("semantic/semantic_summary.json", "semantic_summary.json", "application/json", True),
    ("semantic/semantic_review_queue.json", "semantic_review_queue.json", "application/json", True),
    ("semantic/semantic_composition.png", "semantic_composition.png", "image/png", False),
    ("error_analysis/segment_properties/info", "segment_properties/info", "application/json", False),
    (f"reports/{V.NAME.lower()}/report.json", "report.json", "application/json", True),
    (f"reports/{V.NAME.lower()}/report.md", "report.md", "text/markdown", False),
]


def anonymous_listing_status() -> int:
    url = f"https://storage.googleapis.com/storage/v1/b/{V.GCS_BUCKET}/o?maxResults=1"
    try:
        with urlopen(Request(url), timeout=30) as response:
            return int(response.status)
    except HTTPError as error:
        return int(error.code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-previous", action="store_true",
                        help="save each replaced remote object under <run>/publish/previous")
    args = parser.parse_args()

    prefix = V.GCS_ANALYSIS_PREFIX
    plan = [(V.RUN / local, f"{prefix}/{remote}", kind, gz) for local, remote, kind, gz in FILES]
    missing = [str(path) for path, _, _, _ in plan if not path.exists()]
    properties = json.loads((V.RUN / "error_analysis/segment_properties/info").read_text())
    if not any(p["id"] == "tags" and any(t.startswith("class:") for t in p["tags"])
               for p in properties["inline"]["properties"]):
        raise ValueError("segment_properties/info has no class: tags; rerun make_segment_properties")
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    # The sidecars must describe the segmentation the layer was built from.
    analysis = json.loads(plan[0][0].read_text())["metadata"]
    if Path(analysis["segmentation"]).name != V.SEG.name:
        raise ValueError(f"analysis is of {analysis['segmentation']}, not {V.SEG}")
    status = anonymous_listing_status()
    print(f"anonymous listing -> HTTP {status}")
    if status == 200:
        raise SystemExit("Refusing: bucket is anonymously listable; check IAM first.")
    for path, name, _, _ in plan:
        print(f"  {path.stat().st_size / 1e6:8.2f} MB  gs://{V.GCS_BUCKET}/{name}")
    print(f"  layer info gets \"segment_properties\" if missing: gs://{V.GCS_BUCKET}/{prefix}/info")
    if args.dry_run:
        print("dry run; nothing uploaded")
        return 0

    token = subprocess.run(
        [GCLOUD, "auth", "print-access-token"], capture_output=True, text=True, check=True
    ).stdout.strip()

    def request(url, data=None, headers=None, method=None):
        combined = {"Authorization": "Bearer " + token, **(headers or {})}
        with urlopen(Request(url, data=data, headers=combined, method=method), timeout=600) as r:
            return r.read()

    out = V.RUN / "publish"
    manifest = {"verified_at_utc": None, "layer": f"gs://{V.GCS_BUCKET}/{prefix}", "objects": []}
    for path, name, kind, compressed in plan:
        api = f"https://storage.googleapis.com/storage/v1/b/{V.GCS_BUCKET}/o/{quote(name, safe='')}"
        raw = path.read_bytes()
        payload = gzip.compress(raw, mtime=0) if compressed else raw
        generation = "0"
        try:
            previous = json.loads(request(api))
        except HTTPError as error:
            if error.code != 404:
                raise
            previous = None
        if previous is not None:
            generation = previous["generation"]
            md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
            if previous.get("md5Hash") == md5:
                print(f"unchanged {name} generation {generation}", flush=True)
                manifest["objects"].append({
                    "local": str(path), "gcs_uri": f"gs://{V.GCS_BUCKET}/{name}",
                    "generation": generation, "sha256": hashlib.sha256(raw).hexdigest(),
                    "unchanged": True,
                })
                continue
            if args.keep_previous:
                backup = out / "previous" / generation / path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(
                    request(api + "?alt=media", headers={"Accept-Encoding": "gzip"})
                )
        metadata = {"name": name, "contentType": kind, "cacheControl": "no-cache"}
        if compressed:
            metadata["contentEncoding"] = "gzip"
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
            + json.dumps(metadata).encode()
            + f"\r\n--{boundary}\r\nContent-Type: {kind}\r\n\r\n".encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        uploaded = json.loads(request(
            f"https://storage.googleapis.com/upload/storage/v1/b/{V.GCS_BUCKET}/o"
            f"?uploadType=multipart&ifGenerationMatch={generation}",
            data=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            method="POST",
        ))
        downloaded = request(
            api + "?alt=media&generation=" + uploaded["generation"],
            headers={"Accept-Encoding": "gzip"},
        )
        if downloaded.startswith(b"\x1f\x8b"):
            downloaded = gzip.decompress(downloaded)
        sha = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(downloaded).hexdigest() != sha:
            raise ValueError(f"downloaded bytes differ: {name}")
        manifest["objects"].append({
            "local": str(path), "gcs_uri": f"gs://{V.GCS_BUCKET}/{name}",
            "replaced_generation": None if generation == "0" else generation,
            "generation": uploaded["generation"], "sha256": sha, "download_verified": True,
        })
        print(f"verified {name} generation {uploaded['generation']}", flush=True)
    # The layer's own info must name the sidecar or Neuroglancer never reads it.
    name = f"{prefix}/info"
    api = f"https://storage.googleapis.com/storage/v1/b/{V.GCS_BUCKET}/o/{quote(name, safe='')}"
    live = json.loads(request(api))
    info = json.loads(request(api + "?alt=media"))
    if info.get("segment_properties") == "segment_properties":
        print(f"layer info already names segment_properties (generation {live['generation']})")
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"layer_info.original.{live['generation']}.json").write_text(json.dumps(info))
        info["segment_properties"] = "segment_properties"
        raw = json.dumps(info).encode()
        metadata = {"name": name, "contentType": "application/json", "cacheControl": "no-cache"}
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
            + json.dumps(metadata).encode()
            + f"\r\n--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode()
            + raw + f"\r\n--{boundary}--\r\n".encode()
        )
        uploaded = json.loads(request(
            f"https://storage.googleapis.com/upload/storage/v1/b/{V.GCS_BUCKET}/o"
            f"?uploadType=multipart&ifGenerationMatch={live['generation']}",
            data=body, headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            method="POST",
        ))
        check = json.loads(request(api + "?alt=media&generation=" + uploaded["generation"]))
        if check != info:
            raise ValueError("patched layer info did not read back")
        manifest["layer_info"] = {"replaced_generation": live["generation"],
                                  "generation": uploaded["generation"]}
        print(f"patched layer info generation {uploaded['generation']}")
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "gcs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'gcs_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
