# Code v1

## Overview

Fixed all accepted review findings except resolved minor 8. Added complete opt-in S3 orchestration, threshold resolution, chunk variants, precomputed readback, and offline tests.

## What Changed

- Restored default `sweep_merge_threshold.py` behavior.
- Added canonical absolute watershed thresholds recorded in JSON.
- Added one-chunk, A, and B chunk configurations.
- Added CloudVolume XYZ→ZYX readback.
- Added preflight validation and efficient boundary detection.
- Preserved criterion threading and chunk-size invariance checks.

## Implementation Details

Chunked runs now use `prepare_config`-compatible generated YAMLs with explicit ABISS parameters. Thresholds are resolved once from `aff_canon.h5` using the existing whole-volume affinity conversion and percentile resolver.

## Files Changed

| File | Purpose |
|---|---|
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | Safe import behavior |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | Opt-in S3 orchestration and default sweep preservation |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | ABISS chunk configuration template |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | Resolution, preflight, variants, config generation |
| `tutorials/neuron_liconn_moe/gcloud/resolve_thresholds.py` | Absolute percentile threshold resolution |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | Canonical affinity conversion |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | VOI, boundary masks, and precomputed readback |
| `tests/unit/test_abiss_s3_chunked.py` | Offline S3 guardrail and artifact tests |

## Git Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f  
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Verification

- `bash -n tutorials/neuron_liconn_moe/gcloud/run_volume.sh` — PASS
- S3 unit tests — 10 passed
- Required focused suites — 28 passed
- `git diff --check` — PASS
- Cloud/VM acceptance checks — NOT RUN

## Review Focus

Validated config loading, criterion guards, threshold consistency, preflight checks, canonical affinity layout, one-pass bounding boxes, per-candidate chunk sizes, masked VOI, and precomputed readback.

## Risks and Unknowns

ABISS execution, CloudVolume production readback, ExPID108 equivalence, and VM-scale acceptance remain unverified.

## Changes Since Previous Code Version

1. Added complete `abiss_chunk` configuration generation with required schema fields and per-variant configs.

2. Threaded `MERGE_CRITERION` through resolution and generated ABISS stage selection; removed the weaker shell-only criterion rejection.

3. Added one-time absolute 94%/20% threshold resolution from `aff_canon.h5`, persisted under `RUN_PREFIX`, and reused by references and chunked runs.

4. Wired the required pipeline order, one-chunk plumbing run, A/B runs, integrity gate, equivalence test, and candidate-specific chunk sizes.

5. Each candidate now carries its own chunk size; invariance remains checked under the `W != 0` mask.

6. Replaced per-label voxel scans with `scipy.ndimage.find_objects`.

7. Made S3 opt-in via explicit `DECODE`; the default path still runs `sweep_merge_threshold.py`.

8. No change needed; `set -euo pipefail` already provides failure propagation.

9. Added source existence, shape, dtype, and canonical affinity range checks in `preflight`.

10. Added config-loading, threshold, preflight, and precomputed-readback tests.

11. Added `--input-dataset main` to canonical calls and documented the slab-face restoration behavior.