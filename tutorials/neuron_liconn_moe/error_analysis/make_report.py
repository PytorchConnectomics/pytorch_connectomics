#!/usr/bin/env python3
"""Render the GT-free analysis into the spec's `reports/<cube_id>/` contract.

Satisfies the acceptance clauses of card MSIDEPLOY-EVAL-002 in
`research/projects/msi_liconn_deploy/cards/2026-09-16_gt-free-segment-records.md`:
an atomically written `report.json` carrying one proxy-namespaced record per
analysed segment, the `min_voxels` cutoff, physical voxel size in ZYX µm, the
module provenance and `has_gt: false`; a `report.md` carrying the spec's
mandatory GT-free sentence verbatim; and a `status: "failed"` artifact with an
exit code when the run dies, rather than a missing file.

**Why the nesting is structural.** Mindspan's two-pie rule says proxy errors and
true errors are two different charts and an unlabelled one reads as measurement.
A caption can be dropped by a renderer; a key cannot. Every measured field
therefore sits under a per-segment `proxy` object, and the only `quality` object
in the file states that no quality number exists. Reaching a number in this file
requires having gone through a key named `proxy`.

    LICONN_EA_RUN=<run dir> python tutorials/neuron_liconn_moe/error_analysis/make_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

CUBE_ID = V.NAME.lower()

# Verbatim from spec.md task 7. Do not paraphrase: the spec requires this exact
# sentence whenever has_gt is false, and a reworded version does not satisfy it.
GT_FREE_SENTENCE = (
    "No ground-truth skeletons were supplied, so no segmentation-quality number "
    "is reported. Do not read this segmentation as validated."
)

MODULE_FILES = (
    "connectomics/metrics/unsupervised/morphology.py",
    "connectomics/metrics/unsupervised/arbor.py",
    "connectomics/metrics/unsupervised/classification.py",
    "connectomics/metrics/unsupervised/continuity.py",
    "connectomics/metrics/unsupervised/pieces.py",
    "connectomics/evaluation/semantic.py",
    "connectomics/decoding/error_correction/split_links.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def module_provenance() -> dict:
    """Commit plus per-file hashes, and an explicit flag when files are uncommitted.

    The card asks for "the module commit". Reporting one alone would be false
    here: the analysis used two files that are not in any commit, so the commit
    does not describe the code that produced these numbers. Both are recorded and
    `clean` says which situation the reader is in.
    """
    files = {}
    dirty = []
    for relative in MODULE_FILES:
        path = ROOT / relative
        if not path.exists():
            continue
        files[relative] = sha256(path)
        if git("status", "--short", "--", relative):
            dirty.append(relative)
    return {
        "repo_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "last_commit_touching_metrics_unsupervised": git(
            "log", "-1", "--format=%H %s", "--", "connectomics/metrics/unsupervised"
        ),
        "file_sha256": files,
        "uncommitted_files": sorted(dirty),
        "clean": not dirty,
        "note": (
            "Files listed in `uncommitted_files` are not in any commit, so "
            "`repo_head` does not describe the code that produced this report."
        ),
    }


def probe_layer(layer_uri: str) -> dict:
    """Check the published precomputed layer's `info` actually reads.

    Spec invariant I5: a 403 is not a link. The report states the layer's
    verification state rather than presenting an unchecked URL as working.
    """
    gcloud = str(Path.home() / "google-cloud-sdk/bin/gcloud")
    target = layer_uri.replace("precomputed://", "") + "/info"
    try:
        done = subprocess.run(
            [gcloud, "storage", "ls", target], capture_output=True, text=True, timeout=120
        )
        return {
            "checked": target,
            "authenticated_read_ok": done.returncode == 0,
            "anonymous_read_ok": False,
            "detail": (done.stdout or done.stderr).strip()[:300],
        }
    except Exception as error:  # noqa: BLE001 - the report records the failure
        return {"checked": target, "authenticated_read_ok": False,
                "anonymous_read_ok": False, "detail": f"probe failed: {error}"[:300]}


def build_segment_record(segment: dict) -> dict:
    """One segment, with every measurement behind a `proxy` key."""
    return {
        "id": segment["id"],
        "voxel_count": segment["voxel_count"],
        "proxy": {
            "is_proxy": True,
            "basis": "geometry of the predicted segmentation only; no reference labels",
            "classification": segment["classification"],
            "arbor": segment["arbor"],
            "continuity": {
                **segment["continuity"],
                "semantic_type": segment["semantic_type"],
                "completeness_class": segment["completeness_class"],
                "done": segment["done"],
                "free_end_count": segment["free_end_count"],
                "free_ends": segment["free_ends"],
            },
            "morphology_profile": segment["profile"],
        },
        "quality": {
            "has_gt": False,
            "status": "unverified",
            "is_correct": None,
            "probability_correct": None,
            "note": "No quality number exists for this segment. See report.md.",
        },
    }


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
    temporary.replace(path)


def failure_payload(cube_id: str, exit_code: int, reason: str) -> dict:
    return {
        "schema_name": "pytc.msideploy.report",
        "schema_version": "1.0.0",
        "cube_id": cube_id,
        "status": "failed",
        "exit_code": exit_code,
        "failure_reason": reason[:2000],
        "has_gt": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "segments": [],
    }


def semantic_markdown(summary: dict) -> str:
    rows = summary.get("semantic_classes")
    if not rows:
        return ""
    body = "\n".join(
        f"| `{r['class']}` | {r['segment_count']} | {r['percent_foreground']:.1f}% "
        f"| {r['assessment_status']} |"
        for r in rows
    )
    return (
        "Five-class semantic catalog (`semantic_segmentation.json`, all labels; "
        "% of foreground voxels):\n\n"
        "| class | segments | % foreground | status |\n|---|---:|---:|---|\n"
        f"{body}\n\n"
    )


def render_markdown(payload: dict, layer_uri: str, probe: dict) -> str:
    meta = payload["metadata"]
    summary = payload["summary"]
    verified = probe.get("authenticated_read_ok")
    # Spec task 7: first line is the Neuroglancer URL. I5: a 403 is not a link,
    # so the line states the layer's verification state instead of implying one.
    first = (
        f"{layer_uri}  "
        f"({'authenticated read verified' if verified else 'NOT VERIFIED'}; "
        f"requires Google authentication — anonymous access returns 401/403)"
    )
    return f"""{first}

# GT-free report — `{payload['cube_id']}`

{GT_FREE_SENTENCE}

| | |
|---|---|
| Cube | `{meta['source_volume']}` |
| Model / training set | {meta['model_provenance']['training_set']} |
| Checkpoint | `{meta['model_provenance']['checkpoint']}` |
| Decode threshold | {meta['decode']['merge_threshold']} ({meta['decode']['decoder']}) |
| Physical voxel size (ZYX µm) | {meta['voxel_size_um_zyx']} |
| Volume shape (ZYX) | {meta['shape_zyx']} |
| `min_voxels` cutoff | {meta['min_voxels']} |
| `has_gt` | {str(payload['has_gt']).lower()} |
| Segments analysed | {summary['segments_analysed']} of {summary['labels_total']} |

## What is in `report.json`

One record per analysed segment. Every measurement sits under a per-segment
`proxy` object, because none of it is a quality measurement: it describes the
geometry of the predicted segmentation with no reference labels anywhere in the
computation. The only `quality` object per segment states that no quality number
exists.

Proxy class counts (**not** error counts):

| semantic type (proxy) | segments |
|---|---:|
{chr(10).join(f"| `{k}` | {v} |" for k, v in summary['semantic_type_counts'].items())}

| completeness (proxy) | segments |
|---|---:|
{chr(10).join(f"| `{k}` | {v} |" for k, v in summary['completeness_class_counts'].items())}

{semantic_markdown(summary)}A `free end` is a skeleton terminal that is not explained by the crop boundary.
It is a **candidate** for a false split, not a proven one — a genuine axon
terminal also ends inside the volume, and nothing here separates the two.
Likewise `done` does not certify a segment: a process fused end to end by a
false merge is `done` by this definition.

## Limits that are load-bearing

- `min_voxels = {meta['min_voxels']}` chooses the population. A different cutoff
  gives a different distribution; the cutoff is part of the result.
- The caliber gates were calibrated on ExPID96 S1 and **not** recalibrated for
  this volume's voxel size and expansion fold; `error_analysis/caliber_survey.txt`
  holds this volume's own caliber distribution. Read the semantic classes as
  geometry, never as cell type.
- Nothing here ranks this cube against any other cube or any other checkpoint.

{chr(10).join('- ' + line for line in payload['limitations'])}

---
Generated {meta['completed_at_utc']} · card `MSIDEPLOY-EVAL-002`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=V.ANALYSIS)
    parser.add_argument("--links", type=Path, default=V.LINKS)
    parser.add_argument("--cube-id", default=CUBE_ID)
    parser.add_argument("--reports-root", type=Path, default=V.RUN / "reports")
    parser.add_argument("--fail-here", action="store_true",
                        help="raise on purpose, to exercise the failure artifact")
    args = parser.parse_args()
    destination = args.reports_root / args.cube_id / "report.json"

    try:
        started = time.monotonic()
        if args.fail_here:
            raise RuntimeError("deliberate failure to exercise the status contract")
        analysis = json.loads(args.analysis.read_text())
        links = json.loads(args.links.read_text()) if args.links.exists() else None
        meta = analysis["metadata"]
        layer_uri = (
            f"precomputed://gs://{V.GCS_BUCKET}/{V.GCS_PREFIX}/{V.SEG_LAYER}"
        )
        probe = probe_layer(layer_uri)
        sweep = json.loads(V.SWEEP.read_text())
        with np.load(V.SIZES) as sizes:
            labels_total = int(len(sizes["ids"]))
        semantic_path = V.RUN / "semantic" / "semantic_summary.json"
        semantic = None
        if semantic_path.exists():
            semantic = json.loads(semantic_path.read_text())
            semantic = semantic.get("summary", semantic)

        payload = {
            "schema_name": "pytc.msideploy.report",
            "schema_version": "1.0.0",
            "cube_id": args.cube_id,
            "status": "completed",
            "exit_code": 0,
            "has_gt": False,
            "gt_free_statement": GT_FREE_SENTENCE,
            "proxy_namespace": "proxy",
            "proxy_namespace_note": (
                "Every per-segment measurement is under the `proxy` key. Nothing in "
                "this file is a segmentation-quality measurement, and no field may be "
                "rendered beside a true-error metric without that label."
            ),
            "metadata": {
                "source_volume": V.NAME,
                "segmentation": str(V.SEG),
                "segmentation_sha256": meta["segmentation_sha256"],
                "shape_zyx": meta["shape_zyx"],
                "voxel_size_um_zyx": [round(v / 1000.0, 9) for v in V.SPACING_NM_ZYX],
                "voxel_size_nm_zyx": list(V.SPACING_NM_ZYX),
                "voxel_size_axis_order": "ZYX (array order). Neuroglancer needs XYZ nm.",
                "min_voxels": meta["morphology_config"]["min_voxels"],
                "model_provenance": {
                    "training_set": "IST-LICONN (ExPID82_1), banis+ 200k, applied "
                                    f"cross-sample to this {V.NAME.split('_')[0]} volume",
                    "checkpoint": "outputs/liconn_final_banis_plus_tube/20260728_032436/"
                                  "checkpoints/step=00200000.ckpt",
                    "effective_batch": "eb2 (1 GPU)",
                    "in_domain": False,
                },
                "decode": {"decoder": "ABISS", "merge_threshold": sweep["chosen"]["mt"],
                           "selected_by": f"percentile {sweep['percentile']}"},
                "module_provenance": module_provenance(),
                "neuroglancer_layer": layer_uri,
                "neuroglancer_layer_probe": probe,
                "analysis_artifacts": {
                    "error_analysis_json": str(args.analysis),
                    "split_links_json": str(args.links) if links else None,
                    "figure_png": str(args.analysis.with_name("error_analysis.png")),
                },
                "card": "MSIDEPLOY-EVAL-002",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": None,
            },
            "summary": {
                "labels_total": labels_total,
                "segments_analysed": len(analysis["segments"]),
                "semantic_type_counts": analysis["summary"]["semantic_type_counts"],
                "completeness_class_counts": analysis["summary"][
                    "completeness_class_counts"
                ],
                "free_end_total": analysis["summary"]["free_end_total"],
                "link_proposals": (links["summary"]["pairs"] if links else None),
                "link_groups": (links["summary"]["groups"] if links else None),
                "semantic_classes": (
                    [
                        {k: c[k] for k in ("class", "segment_count", "voxel_count",
                                           "percent_foreground", "assessment_status")}
                        for c in semantic["categories"]
                    ]
                    if semantic else None
                ),
            },
            "limitations": analysis["limitations"],
            "segments": [build_segment_record(s) for s in analysis["segments"]],
        }
        payload["metadata"]["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_atomic(destination, payload)
        markdown = destination.with_name("report.md")
        markdown.write_text(render_markdown(payload, layer_uri, probe))
        print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        print(f"wrote {markdown}")
        print(f"cube_id {args.cube_id}  segments {len(payload['segments'])}  "
              f"has_gt false  module clean="
              f"{payload['metadata']['module_provenance']['clean']}")
        return 0
    except Exception as error:  # noqa: BLE001
        # A crash must leave an artifact, not a hole: a reader cannot tell a
        # missing file from a job that never started.
        write_atomic(destination, failure_payload(args.cube_id, 1, repr(error)))
        print(f"FAILED -- wrote status:failed to {destination}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
