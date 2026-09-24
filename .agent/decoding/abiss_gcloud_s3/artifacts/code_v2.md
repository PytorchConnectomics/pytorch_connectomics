# Code v2

## Overview

Fixed all accepted N1–N7 and m1–m9 findings. S3 now validates canonical affinity correctly, derives volume geometry from the artifact, performs 4-D CloudVolume readback, guards criteria before GPU work, and supports acceptance gating.

## What Changed

- Corrected CZYX destination-face validation.
- Added float32 saturation and percentile diagnostics.
- Fixed CloudVolume `(X,Y,Z,1)` readback.
- Derived `BBOX` and one-chunk size from `aff_canon.h5`.
- Added `S3_ACCEPT` gating for whole-volume references and equivalence.
- Threaded `$MERGE_CRITERION` through resolution and stage generation.
- Corrected per-variant precomputed affinity paths.
- Added one-chunk plumbing validation.
- Strengthened tests with production converter fixtures and 4-D readback fakes.

## Implementation Details

Canonical affinity remains float32 probability-space data. Values may saturate at `1.0`; preflight accepts `max <= 1`, records saturation counts/fraction, and writes 20th/94th percentile diagnostics before decoding.

Whole-volume reference decodes are only required when `DECODE=whole` or `S3_ACCEPT=1`. Chunked runs derive geometry from the canonical artifact and use per-variant precomputed affinity directories.

## Files Changed

| File | Purpose |
|---|---|
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | Safe import behavior |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | Early criterion resolution, dynamic geometry, acceptance gating, diagnostics, stage threading, and S3 exit behavior |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | Canonical chunked configuration template and resolution metadata |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | 4-D readback, plumbing mode, masked VOI, boundary detection, and sparse-label handling |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | Canonical affinity conversion |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | Criterion/alignment guards, artifact-derived BBOX, preflight, diagnostics, and variant config generation |
| `tutorials/neuron_liconn_moe/gcloud/resolve_thresholds.py` | Threshold resolution directly from canonical affinity values |
| `tests/unit/test_abiss_s3_chunked.py` | Converter-backed fixtures, 4-D readback, BBOX derivation, saturation, and plumbing tests |

## Git Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f  
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Verification

- `bash -n tutorials/neuron_liconn_moe/gcloud/run_volume.sh` — PASS
- S3 unit tests — 11 passed
- Required focused suites — 18 passed
- `git diff --check` — PASS
- `py_compile` for changed Python files — PASS
- `cloudvolume` availability check — unavailable in the `pytc` environment
- VM/cloud acceptance — not run

## Review Focus

Verified N1/N2 with production converter and 4-D readback fixtures rather than hand-built assumptions. Verified N3 diagnostics and relaxed range validation, N5 dynamic sizing and acceptance gating, N6 early criterion resolution, and N7 per-variant precomputed affinity paths.

## Risks and Unknowns

ABISS VM execution, CloudVolume production I/O, whole-volume integrity, chunk-size invariance, and ExPID108 VOI acceptance remain unverified. Exact percentile resolution retains full-artifact memory requirements.

## Changes Since Previous Code Version

- **N1:** Corrected destination-face masks to `[0,:,:,0]`, `[1,:,0,:]`, and `[2,0,:,:]`; tests build artifacts through `make_prob_affinity.convert`.
- **N2:** Readback now requires `(X,Y,Z,1)`, removes the channel axis, then transposes; the fake readback is 4-D.
- **N3:** Relaxed canonical affinity validation to `max <= 1`, records saturated fraction, and writes 20th/94th percentile diagnostics. Integrity failures cite the diagnostic JSON.
- **N4:** S3 decode exits before the legacy precomputed/mesh section.
- **N5:** `BBOX` and one-chunk geometry derive from `aff_canon.h5`; whole-volume references and equivalence are gated by `S3_ACCEPT`.
- **N6:** Criterion resolution runs at the top of `run_volume.sh`, uses `$MERGE_CRITERION`, and stages come from `stages_for_criterion`.
- **N7:** `AFF_PATH` now points to each variant’s precomputed `aff` directory; `aff_chunk_size_xyz` matches storage chunks.
- **m1:** Candidate paths are parsed as strings and receive explicit `file://` handling.
- **m2:** `AGG_THRESHOLD`, `ABISS_HOME`, and `resolution_xyz` now use the intended configuration values.
- **m3:** Candidate chunk sizes are derived from resolver variant specifications.
- **m4:** Added `--plumbing` mode requiring exact zero VOI.
- **m5:** Added explicit task-key set validation through `compare_task_keys`.
- **m6:** Source affinity dataset names are discovered instead of assuming `main`.
- **m7:** Documented the full-artifact memory requirement for exact threshold percentiles.
- **m8:** Strengthened tests for converter output, saturation, readback dimensionality, BBOX derivation, plumbing, and configuration paths.
- **m9:** Relabeled sparse segmentation IDs before `find_objects` to avoid allocation by maximum label ID.